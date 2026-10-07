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

import collections
import warnings

import pytest
import numpy as np
from pathlib import Path

from pfvs import CYData, PFV, pvecs, coniZpM

try:
    import cytools  # noqa: F401
    CYTOOLS_AVAILABLE = True
except ImportError:
    CYTOOLS_AVAILABLE = False

# =============================================================================
# Tests PFV class via hard-coded Manwe data, extracted from CYTools.
# (from https://arxiv.org/abs/2406.13751)
# =============================================================================

# Manwe as a CY
# -------------
H11, H21 = 8, 150

VERTS = [
    [ 0, 0, 0, 0], [ 1,-1,-1,-1], [-1, 2, 1, 1], [-1,-1, 0, 0],
    [-1,-1, 2, 0], [-1,-1, 2, 1], [-1, 0, 0, 2], [-1,-1, 0, 2],
    [-1, 0, 0, 1], [-1, 0, 1, 0], [-1,-1, 0, 1], [-1,-1, 1, 0],
    [-1,-1, 1, 1], [-1, 0, 1, 1], [-1, 1, 1, 1], [ 0,-1, 0, 0],
]
HEIGHTS = [0, 35, 29, 35, 31, 35, 35, 35, 15, 17, 31, 9, 21]

C2    = [184, 112, 10, 10, 26, 2, 2, -6]
KAPPA = [
    [
        [130, 80,  5,  7, 16,  2,  0,  0],
        [ 80, 48,  2,  4, 10,  0,  0,  0],
        [  5,  2, -3,  0,  0,  0,  1,  0],
        [  7,  4,  0, -3,  3,  0,  1,  0],
        [ 16, 10,  0,  3,  0,  1,  0,  0],
        [  2,  0,  0,  0,  1, -4,  0,  0],
        [  0,  0,  1,  1,  0,  0, -2,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
    ],
    [
        [ 80, 48,  2,  4, 10,  0,  0,  0],
        [ 48, 28,  0,  2,  6,  0,  0,  0],
        [  2,  0, -2,  0,  0,  0,  0,  0],
        [  4,  2,  0, -2,  2,  0,  0,  0],
        [ 10,  6,  0,  2,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
    ],
    [
        [  5,  2, -3,  0,  0,  0,  1,  0],
        [  2,  0, -2,  0,  0,  0,  0,  0],
        [ -3, -2,  1,  0,  0,  0, -1,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  1,  0, -1,  0,  0,  0, -1,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
    ],
    [
        [  7,  4,  0, -3,  3,  0,  1,  0],
        [  4,  2,  0, -2,  2,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [ -3, -2,  0,  1, -1,  0, -1,  0],
        [  3,  2,  0, -1,  1,  0,  1,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  1,  0,  0, -1,  1,  0, -1,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
    ],
    [
        [ 16, 10,  0,  3,  0,  1,  0,  0],
        [ 10,  6,  0,  2,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  3,  2,  0, -1,  1,  0,  1,  0],
        [  0,  0,  0,  1, -1,  0, -1,  1],
        [  1,  0,  0,  0,  0, -2,  0,  1],
        [  0,  0,  0,  1, -1,  0, -1,  1],
        [  0,  0,  0,  0,  1,  1,  1, -3],
    ],
    [
        [  2,  0,  0,  0,  1, -4,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  1,  0,  0,  0,  0, -2,  0,  1],
        [ -4,  0,  0,  0, -2,  5, -1,  1],
        [  0,  0,  0,  0,  0, -1, -1,  1],
        [  0,  0,  0,  0,  1,  1,  1, -3],
    ],
    [
        [  0,  0,  1,  1,  0,  0, -2,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  1,  0, -1,  0,  0,  0, -1,  0],
        [  1,  0,  0, -1,  1,  0, -1,  0],
        [  0,  0,  0,  1, -1,  0, -1,  1],
        [  0,  0,  0,  0,  0, -1, -1,  1],
        [ -2,  0, -1, -1, -1, -1,  5,  1],
        [  0,  0,  0,  0,  1,  1,  1, -3],
    ],
    [
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [  0,  0,  0,  0,  1,  1,  1, -3],
        [  0,  0,  0,  0,  1,  1,  1, -3],
        [  0,  0,  0,  0,  1,  1,  1, -3],
        [  0,  0,  0,  0, -3, -3, -3,  9],
    ],
]
H = np.array([
    [ 1,  0,  0,  5, -3,  0, -3,  0],
    [ 3,  2,  0,  0,  0,  1,  1,  0],
    [ 5,  3,  0,  0,  1,  0,  0,  0],
    [ 3,  2, -1,  0,  0,  0,  1,  0],
    [ 4,  3,  1,  0,  1,  0,  0,  0],
    [ 0,  0,  0,  0,  0, -1,  0,  0],
    [ 2,  1,  0,  0,  0,  0,  0,  0],
    [ 3,  2,  0,  0,  0,  0,  1,  0],
    [ 8,  5,  0,  0,  1,  0,  1,  0],
    [ 1,  0, -1,  0,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  0, -1,  0,  0],
    [ 2,  0,  0,  0,  1,  0,  0,  0],
    [ 1,  0,  0, -1,  1,  0, -1,  0],
    [ 1,  0,  0,  0,  0,  0, -2,  1],
    [ 6,  4,  0,  1,  0,  1,  0,  0],
    [ 2,  0,  4,  0,  1,  0,  0,  0],
    [ 1,  0,  2,  0,  0,  0,  0,  1],
    [ 3,  2,  0,  0,  0,  0,  0,  1],
    [ 2,  2,  1,  0,  0,  1,  0,  0],
    [ 6,  4,  1, -1,  2,  0,  0,  0],
    [ 2,  0, -1,  1,  0,  0,  0,  0],
    [ 1,  0,  0,  3, -1,  0, -3,  0],
    [ 1,  0,  0,  1,  0,  0, -3,  0],
    [ 4,  3,  0,  1,  0,  0,  0,  0],
    [ 2,  0,  0,  1,  0, -3,  0,  0],
    [ 1,  0,  0,  1,  0,  0, -2,  0],
    [ 3,  0,  0,  0,  1, -5,  1,  0],
    [ 4,  2, -1,  1,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  0,  1,  0,  0],
    [ 2,  2,  1,  1,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  0, -1,  1,  0],
    [ 5,  3,  0,  1,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  0, -2,  0,  1],
    [ 1,  0,  0,  0,  0,  1, -1,  0],
    [ 4,  2, -1,  0,  0,  1,  0,  0],
    [ 2,  1,  0, -1,  1,  0,  0,  0],
    [ 1,  0,  0,  0,  0,  1,  1,  0],
    [ 7,  4,  0,  0,  1,  0,  0,  0],
    [ 1,  0, -1,  0,  0,  0, -1,  0],
    [ 3,  2,  0,  1,  0,  0,  0,  0],
    [ 1,  1,  0,  0,  0,  0,  0,  0],
    [ 2,  1, -1,  0,  0,  0,  1,  0],
    [ 2,  0, -1,  0,  0,  1, -2,  0],
    [ 0,  0,  1,  0,  0,  1,  0,  0],
    [ 1,  0,  0,  0,  0, -2,  1,  0],
    [ 1,  0,  0,  0,  0, -2, -2,  3],
    [ 2,  0,  0,  1,  0,  0, -3,  0],
    [ 1,  0,  0,  0,  2,  0,  2, -5],
    [ 0,  0,  0,  1, -1,  0, -1,  1],
    [ 2,  0,  0,  1,  0,  1, -4,  0],
    [ 1,  1,  1,  0,  0,  0,  0,  0],
    [ 2,  0, -1,  0,  0,  0, -1,  0],
    [ 2,  0,  0,  0,  1, -4,  0,  0],
    [ 2,  0,  0,  1,  0,  1,  0,  0],
    [ 2,  0,  3,  1,  0,  0,  0,  0],
    [ 3,  1, -1,  0,  0,  0,  0,  0],
    [ 6,  4,  0,  0,  1,  0,  0,  0],
    [ 4,  0,  1,  0,  2, -7,  0,  0],
    [ 3,  2,  0, -1,  1,  0,  1,  0],
    [ 1,  0,  1,  0,  1,  0, -3,  0],
    [ 0,  0,  1,  1,  0,  0, -2,  0],
    [ 1,  0,  0,  0,  0,  0,  0,  1],
    [ 0,  0,  0,  0,  1,  1,  1, -3],
    [ 1,  0,  0,  0,  2,  0,  0, -3],
    [ 1,  1,  0,  0,  0,  0,  1,  0],
    [ 2,  0,  0,  0,  1, -3,  0,  0],
    [11,  7,  1,  0,  2,  0,  0,  0],
    [ 2,  1, -1,  0,  0,  0,  0,  0],
    [ 1,  0,  0,  0,  2,  2,  0, -5],
    [ 1,  1,  0,  0,  0,  1,  0,  0],
    [ 1,  0,  1,  0,  0,  0,  1,  0],
    [ 1,  0, -1,  0,  0,  0,  1,  0],
    [ 0,  0,  0,  0,  0, -1, -1,  1],
    [ 3,  0,  0,  0,  1, -4,  0,  0],
], dtype=np.int32)

# Manwe's conifold information
# ----------------------------
CONI_CURVE = [0, 0, 0, 0, 0, -1, 0, 0]
COB = np.array([
    [ 0, 0, 0, 0, 0,-1, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [ 0,-1, 0, 0, 0, 0, 0, 0],
    [ 0, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 0, 0,-1, 0, 0, 0, 0],
    [ 0, 0, 0, 0,-1, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0,-1, 0],
    [ 0, 0, 0, 0, 0, 0, 0,-1],
], dtype=np.int32)

# Manwe as a PFV
# --------------
# Requires dilation 20 to find PFV from scratch...
K_MANWE = [-6, -1,   0, 1, -3,  2,  0, -1]
M_MANWE = [16, 10, -26, 8, 32, 30, 18, 28]
P_MANWE = [-8, 0, -2, 4, 5, 5, 4] # p-vector (pgrading[1:] in cob basis)

# Precomputed GVs (COO format, raw CYTools basis, degree <= 10)
GVS_PATH = Path(__file__).parent / "manwe_gvs_deg10.csv"

# Ground-truth physics values (computed from CYTools GVs, degree <= 10)
TAU0_MANWE    = 15.508799225354053j
GS_MANWE      = 0.0644795245247087
LOG10W0_MANWE = -1.907313582194736


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def coni_data():
    return CYData(h21=H21, kappa=KAPPA, c2=C2, H=H,
                  coni_curve=CONI_CURVE, coni_cob=COB)

@pytest.fixture(scope="module")
def manwe(coni_data):
    pfv = PFV(coni_data, K=K_MANWE, M=M_MANWE)
    pfv.gvs = np.loadtxt(GVS_PATH, dtype=int, delimiter=',')
    return pfv

@pytest.fixture(scope="module")
def coni_scan(coni_data):
    """coniZpM over 10k p-vectors (shared - the scan is the slow part)."""
    return coniZpM(
        data=coni_data,
        ps=pvecs(coni_data, min_N_pts=10_000),
        M0min=13,
        ellipsoid_dilation=50,
        max_N_pfvs=10_000_000,
        n_jobs=1,
        verbosity=0,
    )


# =============================================================================
# Tests
# =============================================================================

def test_coniZpM_finds_manwe(coni_data):
    """With Manwe's p-vector, coniZpM finds exactly Manwe."""
    Ks, Ms = coniZpM(
        data=coni_data,
        ps=np.array([P_MANWE]),
        Q=H11 + H21 + 4,
        M0min=13,
        ellipsoid_dilation=50,
        max_N_pfvs=10_000_000,
        n_jobs=1,
        verbosity=0,
    )

    assert any(np.all(K == K_MANWE) and np.all(M == M_MANWE) for K, M in zip(Ks, Ms))

def test_coniZpM_pfv_count(coni_scan):
    """Scan over 10k p-vectors."""
    Ks, Ms = coni_scan

    assert len(Ks) == 111

def test_coniZpM_kperp_gcds(coni_scan):
    """Kperp need not be primitive.

    Kperp_gcd is bounded by Kperp_gcd < Q*K_gcd/Qs, not fixed to 1. Exactly one
    PFV in this scan has GCD(K[1:]) = 2; a bare count assertion does not notice
    if it goes missing, so pin the multiset.
    """
    Ks, Ms = coni_scan
    gcds = np.gcd.reduce(np.asarray(Ks)[:,1:], axis=1)

    assert sorted(collections.Counter(gcds.tolist()).items()) == [(1, 110), (2, 1)]

def test_manwe_tau0(manwe):
    """tau0 matches the value computed from degree-10 GVs."""
    assert manwe.tau0 == pytest.approx(TAU0_MANWE, rel=1e-6)

def test_manwe_gs(manwe):
    """gs matches the value computed from degree-10 GVs."""
    assert manwe.gs == pytest.approx(GS_MANWE, rel=1e-6)

def test_manwe_W0(manwe):
    """log10(|W0|) matches the value computed from degree-10 GVs."""
    assert manwe.W0(as_logs=True) == pytest.approx(LOG10W0_MANWE, rel=1e-6)

@pytest.mark.skipif(not CYTOOLS_AVAILABLE, reason="requires CYTools")
def test_manwe_gvs_match_cytools():
    """Precomputed COO GVs are identical to a fresh CYTools computation."""
    from cytools import Polytope
    cy = Polytope(VERTS).triangulate(heights=HEIGHTS).cy()
    gvs_cytools = cy.compute_gvs(max_deg=10).coo
    gvs_saved   = np.loadtxt(GVS_PATH, dtype=int, delimiter=',')

    assert gvs_cytools.shape == gvs_saved.shape

    # Sort both by charge columns so row order doesn't matter
    def sort_coo(arr):
        return arr[np.lexsort(arr[:, :-1].T[::-1])]

    np.testing.assert_array_equal(sort_coo(gvs_cytools), sort_coo(gvs_saved))

@pytest.mark.skipif(not CYTOOLS_AVAILABLE, reason="requires CYTools")
def test_from_str_roundtrip():
    """PFV.from_str(str(pfv)) rebuilds the same PFV."""
    from cytools import Polytope
    cy = Polytope(VERTS).triangulate(heights=HEIGHTS).cy()
    data = CYData.from_cy(cy, coni_curve=CONI_CURVE, coni_cob=COB)
    pfv = PFV(data, K=K_MANWE, M=M_MANWE)
    back = PFV.from_str(str(pfv))
    assert back.coni and np.array_equal(back.cob, pfv.cob)
    assert np.array_equal(back.K, pfv.K) and np.array_equal(back.M, pfv.M)

@pytest.mark.parametrize("bad", [
    "verts = [[0, 0, 0, 0]]\nheights = [0]\nK = [1]\nM = [1]\nimport os",
    "verts = [[0, 0, 0, 0]]\nheights = [0]\nK = __import__('os')\nM = [1]",
    "verts = [[0, 0, 0, 0]]\nheights = [0]\nK = [1]",
])
def test_from_str_only_reads_literals(bad):
    """PFV.from_str never executes its input: anything other than
    `name = literal` lines (and the cy line) is rejected."""
    with pytest.raises(ValueError, match="not a PFV string"):
        PFV.from_str(bad)


# =============================================================================
# Non-coni ZpM: three independent routes must agree
# =============================================================================

@pytest.mark.parametrize("dilation", [1, 3])
def test_ZpM_routes_agree(dilation):
    """
    Non-coni Manwe: (Python lattice, numba enumerator), (Python lattice, C
    kernel) and (C lattice, C kernel) find the same PFV set.
    """
    from pfvs import ZpM
    data = CYData(h21=H21, kappa=KAPPA, c2=C2, H=H)          # non-coni
    ps = pvecs(data, min_N_pts=150)

    def as_set(KM):
        return {(tuple(map(int, k)), tuple(map(int, m))) for k, m in zip(*KM)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = as_set(ZpM(data, ps, ellipsoid_dilation=dilation, n_jobs=1,
                         use_c_kernel=False, use_c_lattice=False))
        ck = as_set(ZpM(data, ps, ellipsoid_dilation=dilation, n_jobs=1,
                        use_c_kernel=True, use_c_lattice=False))
        cl = as_set(ZpM(data, ps, ellipsoid_dilation=dilation, n_jobs=1,
                        use_c_kernel=True, use_c_lattice=True))
    assert len(ref) > 0
    assert ck == ref
    assert cl == ref


def test_coniZpM_c_lattice_matches_python_lattice(coni_data):
    """The C lattice setup finds the same coni-PFVs as the Python one."""
    ps = pvecs(coni_data, min_N_pts=2_000)
    kw = {"M0min": 13, "ellipsoid_dilation": 30, "max_N_pfvs": 10_000_000, "n_jobs": 1}
    a = coniZpM(data=coni_data, ps=ps, use_c_lattice=True, **kw)
    b = coniZpM(data=coni_data, ps=ps, use_c_lattice=False, **kw)
    sa = {(tuple(map(int, k)), tuple(map(int, m))) for k, m in zip(*a)}
    sb = {(tuple(map(int, k)), tuple(map(int, m))) for k, m in zip(*b)}
    assert len(sa) > 0 and sa == sb


def test_deprecated_parameters_warn(coni_data):
    """No-op parameters warn when set, and stay silent at their defaults."""
    from pfvs import coni_M_ellipsoid
    p = np.array([0] + P_MANWE)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # defaults: silent
        coni_M_ellipsoid(p, data=coni_data)
        coniZpM(coni_data, np.array([P_MANWE]), ellipsoid_dilation=5, n_jobs=1)
    with pytest.warns(FutureWarning, match="extra_checks"):
        coni_M_ellipsoid(p, data=coni_data, extra_checks=True)
    with pytest.warns(FutureWarning, match="extra_checks"):
        coniZpM(coni_data, np.array([P_MANWE]), ellipsoid_dilation=5, n_jobs=1,
                extra_checks=True)
    with pytest.warns(FutureWarning, match="low_level_parallelism"):
        coniZpM(coni_data, np.array([P_MANWE]), ellipsoid_dilation=5, n_jobs=1,
                low_level_parallelism=True)


def test_postprocessing_exact_path_matches(coni_data, monkeypatch):
    """
    The post-processing's exact (Python-int) fallback, used when an int64
    bound fails, gives exactly the int64 path's output.
    """
    import pfvs.coni as cz
    ps = pvecs(coni_data, min_N_pts=2_000)
    kw = {"M0min": 13, "ellipsoid_dilation": 30, "max_N_pfvs": 10_000_000, "n_jobs": 1}
    Ks, Ms = coniZpM(data=coni_data, ps=ps, **kw)
    real = cz._pfvs_from_points

    def forced_exact(Mv, Kv, qv, key, *a, **k):          # force the object path
        Mv, Kv, qv = cz.util.as_exact(Mv, Kv, qv)
        return real(Mv, Kv, qv, key, *a, **k)
    monkeypatch.setattr(cz, "_pfvs_from_points", forced_exact)
    Ks2, Ms2 = coniZpM(data=coni_data, ps=ps, **kw)
    assert len(Ks) > 0
    assert [list(map(int, r)) for r in Ks] == [list(map(int, r)) for r in Ks2]
    assert [list(map(int, r)) for r in Ms] == [list(map(int, r)) for r in Ms2]


def _inv_fraction(A):
    """Exact inverse by Gauss-Jordan in Fractions (independent reference)."""
    from fractions import Fraction
    n = len(A)
    M = [[Fraction(int(x)) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]


def test_K_ellipsoid_is_exact():
    """K_ellipsoid's matrix equals -B^T (kappa p)^{-1} B exactly."""
    from fractions import Fraction

    from pfvs import K_ellipsoid
    data = CYData(h21=H21, kappa=KAPPA, c2=C2, H=H)          # non-coni
    for p in pvecs(data, min_N_pts=20)[:10]:
        mat, B = K_ellipsoid(p, data=data)
        Ainv = _inv_fraction((np.array(KAPPA) @ p).tolist())
        Bl = [[int(x) for x in r] for r in B.tolist()]
        n, m = len(Bl), len(Bl[0])
        for i in range(m):
            for j in range(m):
                ref = -sum(Bl[a][i] * Ainv[a][b] * Bl[b][j] for a in range(n) for b in range(n))
                got = mat[i][j]
                if np.asarray(mat).dtype.kind in "iuO":
                    assert Fraction(int(got)) == ref            # integral: exact
                else:
                    assert abs(float(got) - float(ref)) <= 1e-12 * max(1.0, abs(float(ref)))
