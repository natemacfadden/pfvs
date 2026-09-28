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
# Description:  The optional CUDA backend (pfvs.gpu). Its results must be
#               identical to the CPU path's -- the same arrays, in the same
#               order. Skipped unless the backend is built and a device found.
#               The device tests share one GPU: run them serially
#               (pytest -n 0 tests/test_gpu.py) or with few workers.
# -----------------------------------------------------------------------------

import numpy as np
import pytest
from latticepts import box_enum

from pfvs import coniZpM, gpu
from test_dataset import GEOMS, cydata

needs_gpu = pytest.mark.skipif(not gpu.available(), reason=gpu.unavailable_reason() or "")

# geometries with p-vectors, a spread of h11
CASES = [g for g in GEOMS if 3 <= g["h11"] <= 11][::3]


def _ps(data, B):
    ps, st, _ = box_enum(B, np.ascontiguousarray(data.H_cob.astype(np.int32)), 1,
                         10**6, primitive=True)
    assert st == 0
    return ps


def test_device_argument_validated():
    data = cydata(GEOMS[0])
    ps = _ps(data, GEOMS[0]["B"])
    with pytest.raises(ValueError, match="device"):
        coniZpM(data, ps[:1], device="tpu", n_jobs=1)


@needs_gpu
@pytest.mark.parametrize("dil", [1, 150])
@pytest.mark.parametrize(
    "g", CASES, ids=[f"h11={g['h11']}-poly{g['polyID']}-coni{g['coniID']}" for g in CASES])
def test_gpu_matches_cpu(g, dil):
    data = cydata(g)
    ps = _ps(data, g["B"])
    if len(ps) == 0:
        pytest.skip("no p-vectors")
    Kc, Mc = coniZpM(data, ps, ellipsoid_dilation=dil, n_jobs=1, device="cpu")
    Kg, Mg = coniZpM(data, ps, ellipsoid_dilation=dil, n_jobs=1, device="gpu")
    np.testing.assert_array_equal(Kg, Kc)
    np.testing.assert_array_equal(Mg, Mc)


@needs_gpu
def test_multi_geometry_batch_matches_single():
    """One call over several geometries equals one call per geometry."""
    geoms, ps_all, pgeo, singles = [], [], [], []
    for g in CASES[:6]:
        data = cydata(g)
        ps = _ps(data, g["B"])[:200]
        if len(ps) == 0:
            continue
        p_full = np.hstack([np.zeros((len(ps), 1), dtype=np.int64), ps])
        spec = dict(kappa=data.kappa_cob, Mbasis=data.M_lattice(),
                    Q=g["h11"] + g["h21"] + 4, dilation=150, M0min=13)
        k = len(geoms)
        geoms.append(spec)
        ps_all += list(p_full)
        pgeo += [k] * len(p_full)
        singles.append(gpu.coni_batch_multi([spec], p_full, np.zeros(len(p_full), dtype=np.int32)))
    M, Kn, q, pidx, pstat, _ = gpu.coni_batch_multi(geoms, ps_all, np.array(pgeo))
    off = 0
    for M1, K1, q1, p1, s1, _ in singles:
        n = len(s1)
        sel = (pidx >= off) & (pidx < off + n)
        np.testing.assert_array_equal(M[sel], M1)
        np.testing.assert_array_equal(Kn[sel], K1)
        np.testing.assert_array_equal(q[sel], q1)
        np.testing.assert_array_equal(pidx[sel] - off, p1)
        np.testing.assert_array_equal(pstat[off:off + n], s1)
        off += n


@needs_gpu
def test_gpu_required_raises_when_unsupported():
    g = CASES[0]
    data = cydata(g)
    ps = _ps(data, g["B"])
    with pytest.raises(RuntimeError, match="device='gpu' unavailable"):
        coniZpM(data, ps, n_jobs=1, device="gpu", use_c_lattice=False)
