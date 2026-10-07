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
#
# -----------------------------------------------------------------------------
# Description:  Exactness of the Fincke-Pohst kernel (fp_kernel.h) through
#               `conipfv_kernel` and `pfv_kernel`: full output lists (points,
#               order, values) against tests/oracle.py.
# -----------------------------------------------------------------------------

import gzip
import json
import math
from pathlib import Path

import flint
import numpy as np
import pytest
from oracle import kernel_reference

from pfvs.conipfv_kernel import conipfv_kernel
from pfvs.pfv_kernel import pfv_kernel

with gzip.open(Path(__file__).parent / "data" / "fixtures.json.gz") as _f:
    FIXTURES = json.load(_f)
BIG = 10**7


def chol_U(mat):
    return np.ascontiguousarray(np.linalg.cholesky(np.array(mat, dtype=float)).T)


def as_obj(a):
    return np.array([[int(x) for x in r] for r in a], dtype=object)


def run(kind, mat, Q, dil, H, linvec=None, linmin=None, pass_mat=True, **kw):
    U = chol_U(mat)
    m = np.array(mat, dtype=np.int64) if pass_mat else None
    if kind == "coni":
        pts, qs, st = conipfv_kernel(U, Q, dil, linvec, linmin, H, BIG, mat=m, **kw)
    else:
        pts, qs, st = pfv_kernel(U, Q, dil, H, BIG, mat=m, **kw)
    return [tuple(int(x) for x in r) for r in pts], [int(q) for q in qs], st


# =============================================================================
# Real instances (from the coni-PFV dataset), vs. the frozen exact oracle
# =============================================================================

KERNEL_CASES = [(k, d) for k in FIXTURES["kernel"] for d in k["expected"]]


@pytest.mark.parametrize(
    "inst,dil", KERNEL_CASES,
    ids=[f"h11={k['h11']}-poly{k['polyID']}-p{i}-D{d}" for i, (k, d) in enumerate(KERNEL_CASES)])
def test_real_instances_match_oracle(inst, dil):
    exp = inst["expected"][dil]
    H = as_obj(inst["H"])
    pts, qs, st = run("coni", inst["mat"], inst["Q"], float(dil), H,
                      np.array(inst["linvec"]), inst["M0min"])
    assert st == 0
    assert pts == [tuple(c) for c in exp["pts"]]
    assert qs == exp["qs"]


def test_real_instances_include_big_H():
    """The fixtures exercise the arbitrary-precision (GMP) H rows."""
    bits = max(abs(int(x)).bit_length() for k in FIXTURES["kernel"] for r in k["H"] for x in r)
    assert bits > 64


def test_mat_recovered_from_U():
    """Omitting mat= (the old API) gives identical output."""
    for inst in FIXTURES["kernel"]:
        if max(abs(int(x)) for r in inst["mat"] for x in r) >= 2**50:
            continue   # not recoverable from a float U; mat= is required
        H = as_obj(inst["H"])
        for dil in inst["expected"]:
            a = run("coni", inst["mat"], inst["Q"], float(dil), H,
                    np.array(inst["linvec"]), inst["M0min"], pass_mat=True)
            b = run("coni", inst["mat"], inst["Q"], float(dil), H,
                    np.array(inst["linvec"]), inst["M0min"], pass_mat=False)
            assert a == b


# =============================================================================
# Synthetic adversarial instances, vs. the live oracle
# =============================================================================

def random_pd(rng, dim, scale):
    A = rng.integers(-scale, scale + 1, (dim, dim))
    return (A.T @ A + np.eye(dim, dtype=np.int64)).astype(np.int64)


def random_H(rng, dim, style):
    G = rng.integers(-6, 7, (dim + 1, dim))
    if style == "hnf":
        return as_obj(flint.fmpz_mat(G.tolist()).hnf().tolist())
    if style == "dense":            # not echelon: still valid, prunes later
        return as_obj(G)
    if style == "rankdef":          # rank < dim, zero rows
        G[:, 0] = 0
        G[-2:] = 0
        return as_obj(flint.fmpz_mat(G.tolist()).hnf().tolist())
    if style == "big":              # entries far beyond int64
        H = as_obj(flint.fmpz_mat(G.tolist()).hnf().tolist())
        H[:, -1] = [int(x) * (2**100 + 7) + int(y) for x, y in zip(H[:, -1], G[:, -1])]
        return H
    if style == "bigpivot":         # a pivot with a huge common factor
        H = as_obj(flint.fmpz_mat(G.tolist()).hnf().tolist())
        H[-2] = [int(x) * 2**90 for x in H[-2]]
        return H
    if style == "empty":            # GCD cut disabled
        return np.zeros((0, dim), dtype=object)
    raise ValueError(style)


STYLES = ["hnf", "dense", "rankdef", "big", "bigpivot", "empty"]


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("kind", ["coni", "pfv"])
@pytest.mark.parametrize("seed", range(6))
def test_synthetic_vs_oracle(style, kind, seed):
    rng = np.random.default_rng(1000 * seed + STYLES.index(style) + (kind == "pfv") * 97)
    dim = int(rng.integers(2, 6))
    mat = random_pd(rng, dim, 3)
    Q = int(rng.integers(3, 12))
    dil = float(rng.choice([1, 2.5, 7, 20]))
    H = random_H(rng, dim, style)
    linvec = linmin = None
    if kind == "coni":
        linvec = rng.integers(-3, 4, dim)          # zeros need not be leading
        linmin = int(rng.integers(-5, 6))
    exp_pts, exp_qs = kernel_reference(mat, Q, dil, H, linvec, linmin, strict=(kind == "coni"))
    pts, qs, st = run(kind, mat, Q, dil, H, linvec, linmin)
    assert st == 0
    assert pts == exp_pts
    assert qs == exp_qs


def test_boundary_points_included():
    """Points with q == floor(dilation*Q) exactly are kept (and q+1 is not)."""
    mat = np.array([[3, 1, 0], [1, 5, 2], [0, 2, 7]], dtype=np.int64)
    H = np.zeros((0, 3), dtype=object)
    for Q, dil in [(13, 1), (7, 3), (10, 2.5)]:
        pts, qs, _ = run("pfv", mat, Q, dil, H)
        exp_pts, exp_qs = kernel_reference(mat, Q, dil, H)
        assert pts == exp_pts and qs == exp_qs
        assert max(qs) <= math.floor(dil * Q)
    # some point sits exactly on the boundary for Q=13, dil=1 (q(1,1,0)=10, ...)
    pts, qs, _ = run("pfv", mat, 10, 1, H)
    assert 10 in qs


def test_all_zero_linvec():
    """linvec == 0 means M0 == 0: every point passes iff linmin <= 0."""
    mat = random_pd(np.random.default_rng(0), 3, 2)
    H = np.zeros((0, 3), dtype=object)
    z = np.zeros(3, dtype=np.int64)
    assert run("coni", mat, 5, 4, H, z, 1)[0] == []
    assert run("coni", mat, 5, 4, H, z, 0)[0] == kernel_reference(mat, 5, 4, H)[0]


def test_max_N_out():
    """Exceeding max_N_out returns status -2 and the first max_N_out points."""
    mat = random_pd(np.random.default_rng(1), 4, 2)
    H = np.zeros((0, 4), dtype=object)
    U = chol_U(mat)
    full, _, st = pfv_kernel(U, 10, 30, H, BIG, mat=mat)
    assert st == 0 and len(full) > 10
    part, _, st = pfv_kernel(U, 10, 30, H, 10, mat=mat)
    assert st == -2
    assert np.array_equal(part, full[:10])
    none, _, st = pfv_kernel(U, 10, 30, H, 0, mat=mat)
    assert st == -2 and len(none) == 0


def test_non_integral_ellipsoid_rejected():
    """U must come from an integer matrix (else pass mat=)."""
    U = np.array([[1.0, 0.3], [0.0, 1.1]])
    with pytest.raises(ValueError):
        pfv_kernel(U, 5, 1, np.zeros((0, 2), dtype=object), 10)


def test_n_nodes_reported():
    mat = random_pd(np.random.default_rng(2), 3, 2)
    *_, n = pfv_kernel(chol_U(mat), 5, 3, np.zeros((0, 3), dtype=object), BIG,
                       mat=mat, return_n_nodes=True)
    assert n >= 1


def test_huge_mat_needs_mat_kwarg():
    """A mat beyond double precision cannot be recovered from U: refuse."""
    inst = max(FIXTURES["kernel"], key=lambda k: max(abs(int(x)) for r in k["mat"] for x in r))
    assert max(abs(int(x)) for r in inst["mat"] for x in r) >= 2**55
    with pytest.raises(OverflowError):
        run("coni", inst["mat"], inst["Q"], 1.0, as_obj(inst["H"]),
            np.array(inst["linvec"]), inst["M0min"], pass_mat=False)


def _sheared(rng, dim, lo, hi, n_shears=3):
    """
    mat = B^T diag(d) B with B unimodular and huge shears: an ill-conditioned
    ellipsoid (huge entries) whose points have small q. Returns (mat, B, d).
    """
    B = np.eye(dim, dtype=object)
    for _ in range(n_shears):
        i, j = rng.choice(dim, 2, replace=False)
        B[i] = B[i] + int(rng.integers(lo, hi)) * B[j]
    d = [int(x) for x in rng.integers(1, 4, dim)]
    return (B.T * d) @ B, B, d


def _sheared_reference(B, d, qmax):
    """Exact points of the sheared ellipsoid via y = B c, in kernel order."""
    import itertools
    dim = len(d)
    Binv = flint.fmpz_mat([[int(x) for x in r] for r in B]).inv()
    Binv = np.array([[int(Binv[i, j]) for j in range(dim)] for i in range(dim)], dtype=object)
    r = [math.isqrt(qmax // di) for di in d]
    pts = []
    for y in itertools.product(*[range(-k, k + 1) for k in r]):
        q = sum(di * yi * yi for di, yi in zip(d, y))
        if q <= qmax:
            c = tuple(int(x) for x in Binv @ np.array(y, dtype=object))
            pts.append((c, q))
    # the kernel's DFS order: last coordinate outermost, each ascending
    pts.sort(key=lambda cq: cq[0][::-1])
    return [c for c, _ in pts], [q for _, q in pts]


@pytest.mark.parametrize("seed", range(40))
def test_cancellation_heavy_ellipsoid(seed):
    """
    Ill-conditioned mat (entries up to ~2^50) whose valid vectors have small q.
    The float partial norms suffer massive cancellation; the kernel's exact
    factorization + error bound must still return exactly the oracle's set.
    """
    rng = np.random.default_rng(seed)
    dim = int(rng.integers(2, 5))
    mat, B, d = _sheared(rng, dim, 2**7, 2**9)
    if max(abs(int(x)) for x in mat.ravel()) >= 2**62:
        pytest.skip("mat does not fit int64")
    H = np.zeros((0, dim), dtype=object)
    exp_pts, exp_qs = _sheared_reference(B, d, 12)
    assert len(exp_pts) > 1
    pts, qs, st = pfv_kernel(np.eye(dim), 1, 12, H, BIG, mat=np.array(mat.tolist(), dtype=np.int64))
    assert st == 0
    assert [tuple(int(x) for x in r) for r in pts] == exp_pts
    assert [int(q) for q in qs] == exp_qs


@pytest.mark.parametrize("seed", range(20))
def test_coordinates_beyond_int32_flagged(seed):
    """Valid vectors with coordinates beyond int32: status -8, never silence."""
    rng = np.random.default_rng(100 + seed)
    dim = 3   # dim 4 can make the (finite) search tree astronomically large
    mat, B, d = _sheared(rng, dim, 2**18, 2**19, n_shears=2)
    if max(abs(int(x)) for x in mat.ravel()) >= 2**62:
        pytest.skip("mat does not fit int64")
    H = np.zeros((0, dim), dtype=object)
    exp_pts, _ = _sheared_reference(B, d, 12)
    pts, _, st = pfv_kernel(np.eye(dim), 1, 12, H, BIG, mat=np.array(mat.tolist(), dtype=np.int64))
    if max(abs(x) for c in exp_pts for x in c) >= 2**31:
        assert st == -8
    else:
        assert st == 0 and [tuple(int(x) for x in r) for r in pts] == exp_pts
