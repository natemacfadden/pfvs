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
# Description:  End-to-end regression against the public coni-PFV dataset
#               (https://huggingface.co/datasets/natemacfadden/calabi-yau-coni-pfvs).
#               For each stored geometry, coniZpM over every primitive p-vector
#               with |p|_inf <= B at dilation D must reproduce exactly the
#               dataset's PFVs with p-infnorm <= B and required dilation <= D.
#               (The dataset predates automatic Kperp handling, so it holds the
#               PFVs with primitive K[1:]; additional PFVs with gcd(K[1:]) >= 2
#               must be valid.)
#               Fixtures: tests/data/build_fixtures.py.
# -----------------------------------------------------------------------------

import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from latticepts import box_enum

from pfvs import PFV, CYData, coniZpM

with gzip.open(Path(__file__).parent / "data" / "fixtures.json.gz") as _f:
    FIXTURES = json.load(_f)
GEOMS = FIXTURES["geometries"]


def cydata(g):
    """CYData for a dataset geometry (stored in the coni basis)."""
    h11 = g["h11"]
    kappa = np.zeros((h11,) * 3, dtype=np.int64)
    for i, j, k, v in g["kappa_coo"]:
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            kappa[a, b, c] = v
    Hc = np.array(g["H"], dtype=np.int64)
    H = np.hstack([np.zeros((len(Hc), 1), dtype=np.int64), Hc])
    e = np.eye(h11, dtype=np.int64)
    return CYData(g["h21"], kappa, g["c2"], H, coni_curve=e[0], coni_cob=e)


def as_set(Ks, Ms):
    return {(tuple(int(x) for x in k), tuple(int(x) for x in m)) for k, m in zip(Ks, Ms)}


@pytest.mark.parametrize(
    "g", GEOMS,
    ids=[f"h11={g['h11']}-poly{g['polyID']}-class{g['classID']}-coni{g['coniID']}" for g in GEOMS])
def test_reproduces_dataset(g):
    data = cydata(g)
    ps, st, _ = box_enum(g["B"], np.ascontiguousarray(data.H_cob.astype(np.int32)), 1,
                         10**6, primitive=True)
    assert st == 0
    expected = as_set(g["K"], g["M"])
    if len(ps) == 0:
        assert expected == set()
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no p-vector may be skipped
        Ks, Ms = coniZpM(data, ps, ellipsoid_dilation=g["D"], n_jobs=1)
    got = as_set(Ks, Ms)

    # The dataset predates automatic Kperp handling: it holds exactly the PFVs
    # with primitive K[1:]. Those must be reproduced exactly; every other PFV
    # found must have gcd(K[1:]) >= 2 and be a valid PFV.
    def primitive(KM):
        return np.gcd.reduce(np.array(KM[0][1:])) == 1

    assert expected <= got                                   # none missing
    assert {KM for KM in got if primitive(KM)} == expected   # exact, old convention
    for K, M in got - expected:
        assert not primitive((K, M))
        assert PFV(data, np.array(K), np.array(M)).check_all(stop_at_fail=False)


def test_dataset_fixture_nontrivial():
    """The fixture covers every h11 in the dataset and a few hundred PFVs."""
    assert {g["h11"] for g in GEOMS} == set(range(3, 12))
    assert sum(len(g["K"]) for g in GEOMS) > 200


@pytest.mark.parametrize("g", [g for g in GEOMS if g["K"]][:12],
                         ids=lambda g: f"h11={g['h11']}-poly{g['polyID']}")
def test_dataset_pfvs_are_valid(g):
    """Every reproduced PFV passes the independent PFV checks."""
    data = cydata(g)
    for K, M in zip(g["K"], g["M"]):
        assert PFV(data, K, M).check_all(stop_at_fail=False)


def _large_p_cases():
    """(geometry, p) pairs with large p, where int64 used to overflow."""
    from pfvs.util import _orthogonal_lattice_int64, exact_matmul
    rng = np.random.default_rng(11)
    cases = []
    for g in GEOMS:
        if g["h11"] < 8:
            continue
        data = cydata(g)
        base, _, _ = box_enum(4, np.ascontiguousarray(data.H_cob.astype(np.int32)), 1,
                               10**6, primitive=True)
        if len(base) < 3:
            continue
        for _ in range(20):
            w = rng.integers(40, 150, 3)
            p = (w[:, None] * base[rng.choice(len(base), 3)].astype(np.int64)).sum(0)
            p //= np.gcd.reduce(p)
            pf = np.concatenate([[0], p])
            Z = exact_matmul(data.kappa_cob.reshape(-1, g["h11"]), pf).reshape(g["h11"], -1)
            T = exact_matmul(Z, pf)
            v = exact_matmul(T, data.M_lattice())
            if v.dtype != object and not _orthogonal_lattice_int64(v)[1]:
                cases.append((g, p))
                break
        if len(cases) == 4:
            break
    return cases


@pytest.mark.parametrize("g,p", _large_p_cases(),
                         ids=lambda x: f"h11={x['h11']}" if isinstance(x, dict) else "p")
def test_large_p_not_skipped(g, p):
    """
    Large p-vectors whose M-lattice computation overflows int64 are handled
    exactly (previously: corrupted basis, non-PD ellipsoid, p silently skipped).
    """
    from pfvs.coniZp import coni_M_ellipsoid
    data = cydata(g)
    pf = np.concatenate([[0], p])
    mat, _, Binter = coni_M_ellipsoid(pf, data=data)
    T = np.array(data.kappa_cob, dtype=object) @ np.array(pf, dtype=object) @ np.array(pf, dtype=object)
    assert all(int(x) == 0 for x in T @ np.array(Binter.tolist(), dtype=object))
    assert np.all(np.linalg.eigvalsh(mat.astype(float)) > 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # must not be skipped
        Ks, Ms = coniZpM(data, np.array([p]), ellipsoid_dilation=20, n_jobs=1)
    for K, M in zip(Ks, Ms):
        assert PFV(data, K, M).check_all(stop_at_fail=False)


def test_incomplete_search_raises():
    """A p-vector that cannot be searched completely raises, never skips."""
    from pfvs import IncompleteSearchError
    g = max(GEOMS, key=lambda g: len(g["K"]))
    data = cydata(g)
    ps, _, _ = box_enum(g["B"], np.ascontiguousarray(data.H_cob.astype(np.int32)), 1,
                        10**6, primitive=True)
    # more outputs than allowed: status -2 must raise (it used to warn and
    # silently keep a truncated output)
    with pytest.raises(IncompleteSearchError, match="max_N_pfvs"):
        coniZpM(data, ps, ellipsoid_dilation=g["D"], n_jobs=1, max_N_pfvs=1)
