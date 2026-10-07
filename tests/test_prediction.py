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
# Description:  pfvs.prediction: exact counts by search, the Gaussian-heuristic
#               estimate, and ranking conifolds, on the coni examples of
#               arXiv:2406.13751 (tests/data/paper_examples.json).
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pytest

from pfvs import CYData, coniZpM, pvecs
from pfvs.prediction import (
    _first_N,
    count_coni_pfvs,
    estimate_coni_pfvs,
    expected_pfvs_per_p,
    rank_coni_geometries,
)

with open(Path(__file__).parent / "data" / "paper_examples.json") as _f:
    CONI = [e for e in json.load(_f)["examples"] if e["kind"] == "coni"]

D = 10


def _data(e):
    return CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"],
                  coni_curve=e["coni_curve"], coni_cob=e["cob"])


@pytest.fixture(scope="module")
def datas():
    seen, out = set(), []
    for e in CONI:                       # one per geometry
        key = (e["h21"], json.dumps(e["kappa"]))
        if key not in seen:
            seen.add(key)
            out.append(_data(e))
    return out


def test_first_N(datas):
    """Exactly N distinct p-vectors of pvecs' box: all of the inner box and a
    seeded subset of the outermost shell."""
    for data in datas:
        full = pvecs(data, 300)
        ps = _first_N(data, 300)
        assert len(ps) == min(300, len(full)) and len(np.unique(ps, axis=0)) == len(ps)
        rad, frad = np.abs(ps).max(1), np.abs(full).max(1)
        assert rad.max() == frad.max()
        assert np.sum(rad < rad.max()) == np.sum(frad < frad.max())
        assert {tuple(p) for p in ps} <= {tuple(p) for p in full}
        assert np.array_equal(ps, _first_N(data, 300))
        assert not np.array_equal(ps, _first_N(data, 300, seed=1)) or len(full) == 300


def test_count_is_coniZpM(datas):
    """The count is exactly what coniZpM returns on the same p-vectors, given
    as N or as ps, and across the chunk boundary."""
    import pfvs.prediction as pred
    for data in datas[:3]:
        ps = _first_N(data, 300)
        n_ref = len(coniZpM(data, ps, ellipsoid_dilation=D, n_jobs=1)[0])
        assert count_coni_pfvs(data, ps=ps, ellipsoid_dilation=D, n_jobs=1) == (n_ref, 300)
        assert count_coni_pfvs(data, N=300, ellipsoid_dilation=D, n_jobs=1) == (n_ref, 300)
        chunk, pred._COUNT_CHUNK = pred._COUNT_CHUNK, 97
        try:
            assert count_coni_pfvs(data, ps=ps, ellipsoid_dilation=D, n_jobs=1)[0] == n_ref
        finally:
            pred._COUNT_CHUNK = chunk


def test_rank(datas):
    """Ranking orders by the scores, most first, under both methods."""
    for method in ("search", "estimate"):
        kw = dict(n_jobs=1) if method == "search" else {}
        order, scores = rank_coni_geometries(datas, N=200, ellipsoid_dilation=D,
                                             method=method, **kw)
        assert sorted(order.tolist()) == list(range(len(datas)))
        assert np.all(np.diff(scores[order]) <= 0)
    ref = [count_coni_pfvs(d, N=200, ellipsoid_dilation=D, n_jobs=1)[0] for d in datas[:3]]
    for n_prefetch in (0, 2):
        _, s = rank_coni_geometries(datas[:3], N=200, ellipsoid_dilation=D, n_jobs=1,
                                    n_prefetch=n_prefetch)
        assert s.tolist() == ref
    with pytest.raises(ValueError):
        rank_coni_geometries(datas, N=10, method="nope")


def test_estimate(datas):
    """The estimate is positive, nondecreasing in the dilation, sums the per-p
    estimates exactly when every p is used, and rejects bad arguments."""
    data = datas[0]
    ps = pvecs(data, 100)
    kw = dict(kappa=data.kappa_cob, Mbasis=data.M_lattice(),
              Q=data.h11 + data.h21 + 4, ellipsoid_dilation=D)
    E = np.array([expected_pfvs_per_p(p, **kw) for p in ps])
    assert np.all(E >= 0) and E.sum() > 0
    cum = expected_pfvs_per_p(ps[0], return_cumulative=True, **kw)
    assert len(cum) == D and np.all(np.diff(cum) >= 0) and cum[-1] == pytest.approx(E[0])
    est = estimate_coni_pfvs(data, ps=ps, ellipsoid_dilation=D, n_samp=len(ps))
    assert est == pytest.approx(E.sum())
    # N within the enumeration limit: the search's own p-vectors
    ps_N = _first_N(data, 100)
    E_N = sum(expected_pfvs_per_p(p, **kw) for p in ps_N)
    est_N = estimate_coni_pfvs(data, N=100, ellipsoid_dilation=D, n_samp=100)
    assert est_N == pytest.approx(E_N)
    with pytest.raises(ValueError):
        estimate_coni_pfvs(data, ellipsoid_dilation=D)
    with pytest.raises(ValueError):
        count_coni_pfvs(data, N=10, ps=ps)
