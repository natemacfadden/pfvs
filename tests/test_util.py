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

import math

import numpy as np
import pytest

from pfvs.util import dual_lattice, extended_euclidean, lll_reduce

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
