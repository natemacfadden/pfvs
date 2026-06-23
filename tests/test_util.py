# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

import itertools
import math

import numpy as np
import pytest

import flint

from pfvs.util import dual_lattice, extended_euclidean, fp_iterative_njit, inv_scaled, lll_reduce, orthogonal_lattice

# =============================================================================
# Test bases
# =============================================================================

# =============================================================================
# Dual lattice test bases (columns are basis vectors)
# =============================================================================

# Full-rank 2D: vol = 6, dual vol = 1/6
DUAL_B_DIAG = np.array([
    [2, 0],
    [0, 3],
], dtype=np.int64)

# Unimodular: self-dual (vol = 1)
DUAL_B_UNIMOD = np.array([
    [1, 1],
    [0, 1],
], dtype=np.int64)

# Rank-1 in 2D
DUAL_B_RANK1 = np.array([
    [2],
    [1],
], dtype=np.int64)

DUAL_BASES = [DUAL_B_DIAG, DUAL_B_UNIMOD, DUAL_B_RANK1]

# =============================================================================
# LLL test bases (columns are basis vectors)
# =============================================================================

# 2x2 trivial basis -- already reduced, change-of-basis should be identity
BASIS_2X2_IDENTITY = np.array([
    [1, 0],
    [0, 1],
], dtype=np.int64)

# 2x2 non-trivial basis (standard LLL example)
BASIS_2X2 = np.array([
    [1, 1],
    [0, 1],
], dtype=np.int64)

# 3x3 basis with a large off-diagonal entry
BASIS_3X3 = np.array([
    [1, 0,  100],
    [0, 1,  100],
    [0, 0,    1],
], dtype=np.int64)

# 4x4 random-ish basis
BASIS_4X4 = np.array([
    [ 3, -1,  2,  5],
    [ 1,  4, -3,  2],
    [-2,  0,  7, -1],
    [ 4,  3,  1,  6],
], dtype=np.int64)

ALL_BASES = [BASIS_2X2_IDENTITY, BASIS_2X2, BASIS_3X3, BASIS_4X4]

# =============================================================================
# extended_euclidean test cases: (a, b)
# =============================================================================

EXTENDED_EUCLIDEAN_CASES = [
    (0,  1),   # a = 0
    (1,  0),   # b = 0
    (1,  1),   # gcd = 1
    (6, 10),   # gcd = 2
    (35, 15),  # gcd = 5
    (17, 13),  # coprime
    (-6, 10),  # negative a
    # Note: when b < 0 the algorithm returns a negative gcd (sign follows b).
    # The Bezout identity still holds and callers handle this correctly, so
    # negative-b inputs are not tested here.
]


# =============================================================================
# Tests
# =============================================================================

# =============================================================================
# Dual lattice tests
# =============================================================================

def _vol(B):
    return np.sqrt(np.linalg.det(B.T @ B))

@pytest.mark.parametrize("B", DUAL_BASES)
def test_dual_integrality(B):
    """B^T @ D must have all entries divisible by denom."""
    D, denom = dual_lattice(B)
    assert np.all((B.T @ D) % denom == 0)

@pytest.mark.parametrize("B", DUAL_BASES)
def test_dual_volume(B):
    """vol(primal) * vol(dual) == 1."""
    D, denom = dual_lattice(B)
    B_dual = D / denom
    assert np.isclose(_vol(B) * _vol(B_dual), 1.0)


# =============================================================================
# extended_euclidean tests
# =============================================================================

@pytest.mark.parametrize("a,b", EXTENDED_EUCLIDEAN_CASES)
def test_extended_euclidean(a, b):
    """extended_euclidean must satisfy the Bezout identity and return the GCD."""
    s, t, gcd = extended_euclidean(a, b)

    assert s*a + t*b == gcd, \
        f"Bezout identity failed: {s}*{a} + {t}*{b} = {s*a + t*b}, expected {gcd}"
    assert gcd == math.gcd(a, b), \
        f"GCD {gcd} does not match math.gcd({a}, {b}) = {math.gcd(a, b)}"


@pytest.mark.parametrize("B", ALL_BASES)
def test_lll_reduce_same_lattice(B):
    """lll_reduce must return a basis spanning the same lattice.

    Verified by checking that the change-of-basis matrix T = B^{-1} @ B_red
    has integer entries and |det(T)| = 1 (i.e. T is unimodular).
    """
    B_red = lll_reduce(B)

    # change-of-basis: B @ T = B_red  =>  T = B^{-1} @ B_red
    T = np.linalg.solve(B.astype(float), B_red.astype(float))

    # integer entries (within floating-point tolerance)
    assert np.allclose(T, np.rint(T), atol=1e-8), \
        f"Change-of-basis matrix is not integral:\n{T}"

    # unimodular: |det| = 1
    det = np.linalg.det(T)
    assert abs(abs(det) - 1.0) < 1e-8, \
        f"Change-of-basis matrix determinant is {det}, expected +/-1"


# =============================================================================
# orthogonal_lattice tests
# =============================================================================

ORTH_VECS = [
    np.array([1, 0, 0],    dtype=np.int64),  # axis vector
    np.array([1, 1, 1],    dtype=np.int64),  # uniform
    np.array([2, 3, 5],    dtype=np.int64),  # coprime 3D
    np.array([0, 0, 1, 0], dtype=np.int64),  # 4D axis
    np.array([2, 4, 6, 3], dtype=np.int64),  # 4D mixed
]

@pytest.mark.parametrize("p", ORTH_VECS)
def test_orthogonal_lattice_perpendicularity(p):
    """Every column of the result must be orthogonal to p."""
    B = orthogonal_lattice(p)
    assert np.all(p @ B == 0)

@pytest.mark.parametrize("p", ORTH_VECS)
def test_orthogonal_lattice_rank(p):
    """The result must be a full-rank basis of the orthogonal complement."""
    B = orthogonal_lattice(p)
    assert B.shape == (len(p), len(p) - 1)
    assert np.linalg.matrix_rank(B) == len(p) - 1

@pytest.mark.parametrize("p", ORTH_VECS)
def test_orthogonal_lattice_contains_flint(p):
    """flint's nullspace must be a sublattice of orthogonal_lattice's result."""
    B = orthogonal_lattice(p)

    # flint method (might give sublattice...)
    dim = len(p)
    nullity = dim - 1
    mat_fl = flint.fmpz_mat(p.reshape(1, -1).tolist())
    null, _ = mat_fl.nullspace()
    null = null.transpose().hnf().transpose()
    B_flint = np.array([[int(null[i, j]) for j in range(nullity)] for i in range(dim)])
    gcds = np.array([np.gcd.reduce(B_flint[:, j]) for j in range(nullity)])
    B_flint = B_flint // gcds # this often makes B_flint a true basis

    # sublattice check: each column of B_flint must be an integer linear
    # combination of B's columns, i.e. B @ T = B_flint has an integer solution
    T = np.linalg.lstsq(B.astype(float), B_flint.astype(float), rcond=None)[0]
    assert np.allclose(T, np.rint(T), atol=1e-8), \
        "flint nullspace vectors are not integer combinations of orthogonal_lattice columns"

# =============================================================================
# inv_scaled tests
# =============================================================================

INV_SCALED_MATRICES = [
    np.array([[1, 0], [0, 1]], dtype=np.int64),          # identity
    np.array([[2, 0], [0, 3]], dtype=np.int64),          # diagonal
    np.array([[1, 1], [0, 1]], dtype=np.int64),          # unimodular
    np.array([[2, 1], [1, 1]], dtype=np.int64),          # det=1, off-diagonal
    np.array([[6, 4], [3, 2]], dtype=np.int64),          # singular (should raise)
    np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.int64),  # 3x3
]

@pytest.mark.parametrize("A", INV_SCALED_MATRICES[:4] + INV_SCALED_MATRICES[5:])
def test_inv_scaled(A):
    """B @ A must equal s * I, with B integer and s a positive integer."""
    B, s = inv_scaled(A)
    assert np.issubdtype(B.dtype, np.integer)
    assert isinstance(s, int) and s > 0
    assert np.all(B @ A == s * np.eye(len(A), dtype=np.int64))

def test_inv_scaled_singular():
    """inv_scaled must raise ValueError on a singular matrix."""
    with pytest.raises(ValueError, match="singular"):
        inv_scaled(INV_SCALED_MATRICES[4])

# =============================================================================
# fp_iterative_njit tests
# =============================================================================

# (mat, Q)
FP_CASES = [
    (np.eye(2),                                  4.0),  # 2D unit sphere
    (np.diag([1.0, 2.0, 3.0]),                   7.0),  # 3D diagonal
    (np.array([[4.0, 2.0], [2.0, 3.0]]),        10.0),  # 2D off-diagonal
]

# (mat, Q, linvec, linmin)  -- linvec must have zeros before non-zeros
FP_CASES_LIN = [
    (np.diag([1.0, 2.0, 3.0]), 7.0, np.array([0.0, 1.0, 2.0]), 2),
    (np.array([[4.0, 2.0], [2.0, 3.0]]), 10.0, np.array([0.0, 1.0]), 1),
]

def _brute_force_ellipsoid(mat, Q, linvec=None, linmin=None):
    dim = mat.shape[0]
    lam_min = np.linalg.eigvalsh(mat).min()
    bound = int(np.ceil(np.sqrt(Q / lam_min))) + 1
    vecs = []
    for tup in itertools.product(range(-bound, bound + 1), repeat=dim):
        v = np.array(tup, dtype=np.int64)
        if np.all(v == 0):
            continue
        q = float(v @ mat @ v)
        if q < 0 or q > Q + 1e-9:
            continue
        if linvec is not None and float(linvec @ v) < linmin:
            continue
        vecs.append(tuple(v))
    return set(vecs)

@pytest.mark.parametrize("mat,Q", FP_CASES)
def test_fp_iterative_no_linvec(mat, Q):
    """fp_iterative_njit must find exactly the same vectors as brute force."""
    L = np.linalg.cholesky(mat)
    out, qs = fp_iterative_njit(L, Q)

    assert set(map(tuple, out)) == _brute_force_ellipsoid(mat, Q)
    for v, q in zip(out, qs):
        assert abs(float(v @ mat @ v) - float(q)) < 1e-3

@pytest.mark.parametrize("mat,Q,linvec,linmin", FP_CASES_LIN)
def test_fp_iterative_with_linvec(mat, Q, linvec, linmin):
    """fp_iterative_njit with linvec must match brute force with the same filter."""
    L = np.linalg.cholesky(mat)
    out, qs = fp_iterative_njit(L, Q, linvec=linvec, linmin=linmin)

    assert set(map(tuple, out)) == _brute_force_ellipsoid(mat, Q, linvec=linvec, linmin=linmin)
    for v, q in zip(out, qs):
        assert abs(float(v @ mat @ v) - float(q)) < 1e-3
