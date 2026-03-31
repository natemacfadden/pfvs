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

from pfvs import CYData, PFV, pvecs, coniZpM

# =============================================================================
# Tests PFV class via hard-coded Manwe data, extracted from CYTools.
# (from https://arxiv.org/abs/2406.13751)
# =============================================================================

# Manwe as a CY
# -------------
H11, H21 = 8, 150

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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def coni_data():
    return CYData(h21=H21, kappa=KAPPA, c2=C2, H=H,
                  coni_curve=CONI_CURVE, coni_cob=COB)

@pytest.fixture(scope="module")
def manwe(coni_data):
    return PFV(coni_data, K=K_MANWE, M=M_MANWE)

@pytest.fixture(scope="module")
def coni_pvecs(coni_data):
    return pvecs(coni_data, min_N_pts=200)


# =============================================================================
# CYData tests
# =============================================================================

def test_cydata_coni_smoke(coni_data):
    data = coni_data
    assert data.h11 == H11
    assert data.coni
    assert data.H_cob.shape[1] == H11 - 1
    assert data.kappa_cob.shape == (H11, H11, H11)
    assert data.c2_cob.shape == (H11,)
    assert np.all(data.cob @ data.coni_curve == [1] + [0]*(H11-1))

def test_cydata_M_lattice(coni_data):
    M_lat = coni_data.M_lattice()
    assert M_lat.shape == (H11, H11)
    assert np.all((coni_data.a @ M_lat) % 2 == 0)
    assert np.all((coni_data.b @ M_lat) % 24 == 0)

def test_cydata_bad_cob():
    bad_cob = np.eye(H11, dtype=int)
    with pytest.raises(ValueError, match="coni_cob is not a valid change of basis"):
        CYData(h21=H21, kappa=KAPPA, c2=C2, H=H,
               coni_curve=CONI_CURVE, coni_cob=bad_cob)


# =============================================================================
# PFV constructor / smoke tests
# =============================================================================

def test_pfv_smoke(manwe):
    assert manwe.K.shape == (H11,)
    assert manwe.M.shape == (H11,)
    assert np.all(manwe.K == K_MANWE)
    assert np.all(manwe.M == M_MANWE)

def test_pfv_bad_K_shape(coni_data):
    with pytest.raises(ValueError, match="K must have shape"):
        PFV(coni_data, K=[1, 2, 3], M=M_MANWE)

def test_pfv_bad_M_shape(coni_data):
    with pytest.raises(ValueError, match="M must have shape"):
        PFV(coni_data, K=K_MANWE, M=[1, 2, 3])

def test_pfv_check_all(manwe):
    assert manwe.check_all()

def test_pfv_N_invertible(manwe):
    assert manwe.check_Ninvertible()


# =============================================================================
# pvecs tests
# =============================================================================

def test_pvecs_count(coni_pvecs):
    assert len(coni_pvecs) >= 200

def test_pvecs_kahler_cone(coni_data, coni_pvecs):
    H_cob = coni_data.H_cob
    assert np.all(H_cob @ coni_pvecs.T > 0)

def test_pvecs_primitive(coni_pvecs):
    gcds = np.gcd.reduce(coni_pvecs, axis=1)
    assert np.all(gcds == 1)

def test_pvecs_integer(coni_pvecs):
    assert np.issubdtype(coni_pvecs.dtype, np.integer)

def test_pvecs_min_N_pts_validation(coni_data):
    with pytest.raises(ValueError, match="min_N_pts must be > 0"):
        pvecs(coni_data, min_N_pts=0)


# =============================================================================
# coniZpM tests
# =============================================================================

@pytest.fixture(scope="module")
def coniZpM_results(coni_data, coni_pvecs):
    Q = H11 + H21 + 4  # 162
    return coniZpM(
        data=coni_data,
        ps=coni_pvecs,
        Q=Q,
        M0min=13,
        ellipsoid_dilation=50,
        n_jobs=1,
        verbosity=0,
    )

def test_coniZpM_finds_pfvs(coniZpM_results):
    Ks, Ms = coniZpM_results
    assert len(Ks) > 0
    assert Ks.shape[1] == H11
    assert Ms.shape[1] == H11

def test_coniZpM_tadpole(coniZpM_results):
    Ks, Ms = coniZpM_results
    Q = H11 + H21 + 4
    tadpoles = -np.sum(Ks * Ms, axis=1)
    assert np.all(tadpoles == Q)

def test_coniZpM_M0min(coniZpM_results):
    Ks, Ms = coniZpM_results
    assert np.all(Ms[:, 0] >= 13)

def test_coniZpM_finds_manwe(coni_data):
    """With Manwe's p-vector, coniZpM finds exactly Manwe."""
    Ks, Ms = coniZpM(
        data=coni_data,
        ps=np.array([P_MANWE]),
        Q=H11 + H21 + 4,
        M0min=13,
        ellipsoid_dilation=50,
        n_jobs=1,
        verbosity=0,
    )

    assert any(np.all(K == K_MANWE) and np.all(M == M_MANWE) for K, M in zip(Ks, Ms))

def test_coniZpM_pfv_count(coni_data):
    Ks, Ms = coniZpM(
        data=coni_data,
        ps=pvecs(coni_data, min_N_pts=10_000),
        M0min=13,
        ellipsoid_dilation=50,
        n_jobs=1,
        verbosity=0,
    )

    assert len(Ks) == 110

def test_coniZpM_empty_ps(coni_data):
    with pytest.raises(ValueError, match="ps must be non-empty"):
        coniZpM(data=coni_data, ps=np.empty((0, 7), dtype=int), Q=162)

def test_coniZpM_bad_dilation(coni_data, coni_pvecs):
    with pytest.raises(ValueError, match="ellipsoid_dilation must be > 0"):
        coniZpM(data=coni_data, ps=coni_pvecs, Q=162, ellipsoid_dilation=-1)
