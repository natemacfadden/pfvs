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
# Description:  pfvs.scoring on the paper examples
#               (tests/data/paper_examples.json), coni and non-coni.
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pytest

from pfvs import CYData, ZpM, coniZpM, pvecs
from pfvs.scoring import (
    _count_pfvs,
    _first_N,
    estimate_coni_pfvs,
    expected_pfvs_per_p,
    score_geometries,
)

with open(Path(__file__).parent / "data" / "paper_examples.json") as _f:
    EXAMPLES = json.load(_f)["examples"]
CONI = [e for e in EXAMPLES if e["kind"] == "coni"]

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


@pytest.fixture(scope="module")
def nonconi():
    return [CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"])
            for e in EXAMPLES if e["kind"] == "non-coni"]


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


@pytest.mark.filterwarnings("ignore:non-coni Zp methods")
def test_count_is_search(datas, nonconi):
    """The internal count is exactly coniZpM's / ZpM's, also across chunks."""
    import pfvs.scoring as sc
    for data in datas[:3] + nonconi[:2]:
        search = coniZpM if data.coni else ZpM
        ps = _first_N(data, 300)
        n_ref = len(search(data, ps, ellipsoid_dilation=D, n_jobs=1)[0])
        assert _count_pfvs(data, ps, D, n_jobs=1) == n_ref
        chunk, sc._COUNT_CHUNK = sc._COUNT_CHUNK, {True: 97, False: 97}
        try:
            assert _count_pfvs(data, ps, D, n_jobs=1) == n_ref
        finally:
            sc._COUNT_CHUNK = chunk


@pytest.mark.filterwarnings("ignore:non-coni Zp methods")
def test_score(datas, nonconi):
    """Search scores are the count on the first N p-vectors, with or without
    prefetching, for coni and non-coni; estimate scores are
    estimate_coni_pfvs, coni only."""
    mixed = datas[:2] + nonconi[:2]
    ref = [len((coniZpM if d.coni else ZpM)(d, _first_N(d, 200), ellipsoid_dilation=D,
                                            n_jobs=1)[0]) for d in mixed]
    for n_prefetch in (0, 2):
        s = score_geometries(mixed, N=200, ellipsoid_dilation=D, n_jobs=1,
                             n_prefetch=n_prefetch)
        assert s.shape == (4,) and s.tolist() == ref
    s = score_geometries(datas[:2], N=200, ellipsoid_dilation=D, method="estimate")
    assert s.tolist() == [estimate_coni_pfvs(d, N=200, ellipsoid_dilation=D) for d in datas[:2]]
    with pytest.raises(ValueError):
        score_geometries(nonconi, N=10, method="estimate")
    with pytest.raises(ValueError):
        score_geometries(datas, N=10, method="nope")


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
