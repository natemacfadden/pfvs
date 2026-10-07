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
# Description:  coniZpK (coni PFVs above a dilation D0) and coniZp (every
#               coni PFV of a direction), against ZpM at each direction's
#               dilation bound.
# -----------------------------------------------------------------------------

import math

import numpy as np
import pytest
from latticepts import box_enum

import pfvs.coni as CZ
from pfvs import IncompleteSearchError, coniZp, coniZpK, coniZpM, gpu
from pfvs.coni import _dilations, _pfvs_from_points
from pfvs.dilation import CostModel, coni_dilation_bounds
from pfvs.fp_kernel.fp_kernel import _coni_batch, _coni_zpk_batch
from test_dataset import GEOMS, as_set, cydata
from test_dilation_bound import _direction_and_delta

needs_gpu = pytest.mark.skipif(not gpu.available(), reason=gpu.unavailable_reason() or "")


def _cases(n_random, lo, hi, seed=0):
    """Per fixture geometry with lo <= h11 <= hi: (data, Q, ps), ps the
    directions of its known PFVs plus random ones, all with a bound."""
    rng = np.random.default_rng(seed)
    out = []
    for g in GEOMS:
        if not lo <= g["h11"] <= hi:
            continue
        data = cydata(g)
        kap, Q = data.kappa_cob, g["h11"] + g["h21"] + 4
        dirs = {tuple(_direction_and_delta(kap, K, M)[0][1:]) for K, M in zip(g["K"][:40], g["M"][:40])}
        ps, st, _ = box_enum(g["B"], np.ascontiguousarray(data.H_cob.astype(np.int32)), 1, 10**6,
                             primitive=True)
        if len(ps):
            dirs |= {tuple(p) for p in ps[rng.choice(len(ps), size=min(n_random, len(ps)), replace=False)]}
        ps = np.array(sorted(dirs), dtype=np.int64).reshape(-1, g["h11"] - 1)
        if not len(ps):
            continue
        b = coni_dilation_bounds(ps, kap, Q)
        ps = ps[[x is not None for x in b]]
        if len(ps):
            out.append((data, Q, ps))
    return out


CASES = _cases(20, 3, 9)            # (the fixtures' PFVs are at h11 <= 9)
CASES_HI = _cases(12, 10, 11)


def _zpm(data, Q, ps, D):
    """{p index: set of PFVs} of ZpM at dilation D (the batched C path)."""
    kap, h = data.kappa_cob, data.h11
    pf = np.hstack([np.zeros((len(ps), 1), dtype=np.int64), ps])
    M, Kn, q, pidx, st = _coni_batch(kap, data.M_lattice(), pf, Q, D, 13, 10**8)
    assert np.all(st == 0)
    Ks, Ms, keys = _pfvs_from_points(M.T, Kn.T, q, pidx, kap, h, Q, 13)
    return {i: as_set(Ks[keys == i], Ms[keys == i]) for i in range(len(ps))}


@pytest.mark.parametrize("D0", [10, 40, 400])
def test_zpk_completes_zpm(D0):
    """For every direction: ZpM at D0 together with ZpK above D0 is exactly
    ZpM at the direction's bound (every coni PFV of the direction)."""
    n_dirs = n_above = 0
    for data, Q, ps in CASES + (CASES_HI if D0 >= 400 else []):
        if D0 < 40 and data.h11 > 7:
            continue                                     # (many short K: slow)
        kap, h = data.kappa_cob, data.h11
        pf = np.hstack([np.zeros((len(ps), 1), dtype=np.int64), ps])
        M, Kn, q, pidx, st = _coni_zpk_batch(kap, data.M_lattice(), pf, Q, 13, D0, 10**8)
        Ks, Ms, keys = _pfvs_from_points(M.T, Kn.T, q, pidx, kap, h, Q, 13)
        low = _zpm(data, Q, ps, D0)
        b = coni_dilation_bounds(ps, kap, Q)
        for i in range(len(ps)):
            if st[i] != 0:
                continue
            full = _zpm(data, Q, ps[i:i + 1], max(math.ceil(b[i]), D0))[0]
            assert low[i] | as_set(Ks[keys == i], Ms[keys == i]) == full, ps[i].tolist()
            n_dirs += 1
            n_above += len(full - low[i])
    assert n_dirs > 200 and (n_above > 20 or D0 >= 400)


def test_coniZpK_is_the_pfvs_above_D0():
    """coniZpK returns exactly the PFVs with dilation > D0, each once."""
    D0 = 10
    n_above = 0
    for data, Q, ps in [c for c in CASES if c[0].h11 <= 7]:
        kap = data.kappa_cob
        Ks, Ms = coniZpK(data, ps, D0, Q=Q, n_jobs=2)
        pf = np.hstack([np.zeros((len(ps), 1), dtype=np.int64), ps])
        b = coni_dilation_bounds(ps, kap, Q)
        want = set()
        for i, p in enumerate(ps):
            K1, M1 = coniZpM(data, p[None], Q=Q, ellipsoid_dilation=max(math.ceil(b[i]), D0),
                             device="cpu", n_jobs=1)
            num, den = _dilations(kap, pf[i:i + 1], K1, M1, np.zeros(len(K1), dtype=np.int64))
            want |= {(tuple(map(int, k)), tuple(map(int, m)))
                     for k, m, n, d in zip(K1, M1, num, den) if n > D0 * d}
        assert as_set(Ks, Ms) == want and len(Ks) == len(want)
        n_above += len(want)
    assert n_above > 10


def test_dilations_match_the_pfv_directions():
    """_dilations agrees with the dilation recovered from (K, M) alone."""
    for g in [g for g in GEOMS if g["K"]][:6]:
        data = cydata(g)
        kap = data.kappa_cob
        for K, M in list(zip(g["K"], g["M"]))[:20]:
            p_hat, delta = _direction_and_delta(kap, K, M)
            num, den = _dilations(kap, p_hat[None], np.array([K]), np.array([M]), np.zeros(1, dtype=int))
            assert int(num[0]) * delta.denominator == int(den[0]) * delta.numerator


def _without_bound(monkeypatch, first):
    """Make the first p-vector of every call look as if its direction had no
    dilation bound (no direction in the fixtures' Kahler cones lacks one),
    and ZpK fail on it."""
    import pfvs.dilation as DL
    real_bounds, real_ceils, real_zpk = DL.coni_dilation_bounds, DL.coni_dilation_bound_ceils, \
        CZ._coni_zpk_batch

    def bounds(ps, *a, **k):
        b = real_bounds(ps, *a, **k)
        return [None if np.array_equal(p, first) else x for p, x in zip(np.atleast_2d(ps), b)]

    def ceils(ps, *a, **k):
        b = real_ceils(ps, *a, **k)
        b[[np.array_equal(p, first) for p in np.atleast_2d(ps)]] = 0
        return b

    def zpk(kappa, Mbasis, ps, *a):
        M, Kn, q, pidx, st = real_zpk(kappa, Mbasis, ps, *a)
        hit = np.array([np.array_equal(p[1:], first) for p in ps], dtype=bool)
        st[hit] = 1
        keep = ~hit[pidx]
        return M[keep], Kn[keep], q[keep], pidx[keep], st
    monkeypatch.setattr(DL, "coni_dilation_bounds", bounds)
    monkeypatch.setattr(DL, "coni_dilation_bound_ceils", ceils)
    monkeypatch.setattr(CZ, "_coni_zpk_batch", zpk)


def test_coniZpK_raises_without_a_bound(monkeypatch):
    data, Q, ps = CASES[20]
    _without_bound(monkeypatch, ps[0])
    coniZpK(data, ps[1:], 40, Q=Q, n_jobs=1)
    with pytest.raises(IncompleteSearchError, match="cannot be enumerated"):
        coniZpK(data, ps, 40, Q=Q, n_jobs=1)


def _check_exhaustive(device, cost_model=None, cases=CASES[::2]):
    """coniZp equals ZpM at each direction's bound, each
    PFV once."""
    n_pfvs = 0
    for data, Q, ps in cases:
        kap = data.kappa_cob
        Ks, Ms, complete = coniZp(data, ps, Q=Q, device=device, n_jobs=2,
                                  cost_model=cost_model)
        assert complete.all()
        b = coni_dilation_bounds(ps, kap, Q)
        want = set()
        for i in range(len(ps)):
            want |= _zpm(data, Q, ps[i:i + 1], math.ceil(b[i]))[0]
        assert as_set(Ks, Ms) == want and len(Ks) == len(want)
        n_pfvs += len(want)
    assert n_pfvs > 10


@pytest.mark.parametrize("plan", ["default", "to_bound", "split", "calibrated"])
def test_exhaustive_cpu(plan, monkeypatch):
    """Every plan gives the same PFVs: the default model, ZpM to the bound
    for every p (ZpK too dear), a split for every p (ZpM too dear), and the
    costs measured on a sample."""
    model = {"default": None, "calibrated": None,
             "to_bound": CostModel([100, 1000], [1, 1], [400, 800, 1600], [1e9, 1e9, 1e9]),
             "split": CostModel([100, 1000], [1, 1e9], [400, 800, 1600], [1, 1, 1])}[plan]
    if plan == "calibrated":
        monkeypatch.setattr(CZ, "_PLAN_MIN_PS", 1)
    _check_exhaustive("cpu", model)


def test_exhaustive_reports_directions_without_a_bound(monkeypatch):
    """A p-vector without a bound is searched at fallback_dilation, as
    coniZpM would, and marked incomplete; the others are exhaustive."""
    data, Q, ps = CASES[25]
    want = set()
    for i, b in enumerate(coni_dilation_bounds(ps, data.kappa_cob, Q)):
        want |= _zpm(data, Q, ps[i:i + 1], 30 if i == 0 else math.ceil(b))[0]
    _without_bound(monkeypatch, ps[0])
    Ks, Ms, complete = coniZp(data, ps, Q=Q, fallback_dilation=30,
                              device="cpu", n_jobs=2)
    assert not complete[0] and complete[1:].all()
    assert as_set(Ks, Ms) == want and len(Ks) == len(want)


@needs_gpu
@pytest.mark.parametrize("plan", ["default", "calibrated"])
def test_exhaustive_gpu(plan, monkeypatch):
    if plan == "calibrated":
        monkeypatch.setattr(CZ, "_PLAN_MIN_PS", 1)
    _check_exhaustive("gpu", None, CASES[1::2])


def test_cost_model():
    m = CostModel([100, 400, 1600], [1.0, 2.0, 8.0], [400, 800], [3.0, 2.0], c=0.5)
    assert m.G_at(400) == pytest.approx(2.0)
    assert m.G_at(200) == pytest.approx(2 ** 0.5)               # log-log
    assert m.G_at(6400) == pytest.approx(32.0)                  # the last slope
    assert m.G_at(25) == pytest.approx(0.5)                     # the first slope
    assert m.K_at(800) == 2.0
    with pytest.raises(ValueError):
        CostModel([400, 100], [1, 2], [400], [1])


def test_grid_up():
    D = np.array([1, 2, 3, 100, 799, 800, 801, 12345, 10**6])
    g = CZ._grid_up(D)
    assert np.all(g >= D) and np.all(g <= np.ceil(D * 2 ** (1 / 16)) + 1)
