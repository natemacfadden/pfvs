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

import pytest
import numpy as np
from pathlib import Path

from pfvs import CYData, PFV, pvecs, coniZpM

try:
    import cytools as _cytools
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

def test_coniZpM_pfv_count(coni_data):
    """Scan over 10k p-vectors."""
    Ks, Ms = coniZpM(
        data=coni_data,
        ps=pvecs(coni_data, min_N_pts=10_000),
        M0min=13,
        ellipsoid_dilation=50,
        max_N_pfvs=10_000_000,
        n_jobs=1,
        verbosity=0,
    )

    assert len(Ks) == 110

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
