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
# Description:  The named examples of arXiv:2406.13751 (coniZpM) and
#               arXiv:2107.09064 (ZpM) are found by the search, from the
#               paper's p-vector and, for coni, from a box scan.
#               Geometries: tests/data/paper_examples.json.
#               PFVS_SLOW_TESTS=1 also runs the slow scans (~1 min / ~11 min).
# -----------------------------------------------------------------------------

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from latticepts import box_enum

from pfvs import PFV, CYData, ZpM, coniZpM, gpu

with open(Path(__file__).parent / "data" / "paper_examples.json") as _f:
    EXAMPLES = json.load(_f)["examples"]

SLOW = os.environ.get("PFVS_SLOW_TESTS") == "1"
DEVICES = ["cpu"] + (["gpu"] if gpu.available() else [])


def _id(e):
    tag = e["model"] or e["name"].replace(" ", "_")
    return f"{e['paper'].split(':')[1]}-{e['section']}-{tag}"


def _data(e):
    kw = dict(coni_curve=e["coni_curve"], coni_cob=e["cob"]) if e["kind"] == "coni" else {}
    return CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"], **kw)


def _search(e, data, ps, device="cpu"):
    """The paper's search: coni PFVs with tadpole exactly -M.K and M0 >= 13,
    or non-coni PFVs with -M.K up to the paper's."""
    Q = -int(np.dot(e["M"], e["K"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if e["kind"] == "coni":
            return coniZpM(data, ps, Q=Q, M0min=13, ellipsoid_dilation=e["min_dilation"],
                           n_jobs=1, device=device)
        return ZpM(data, ps, Qmin=0, Qmax=Q, ellipsoid_dilation=e["min_dilation"], n_jobs=1)


def _found(e, Ks, Ms):
    return any(np.array_equal(k, e["K"]) and np.array_equal(m, e["M"]) for k, m in zip(Ks, Ms))


def _p_search(e):
    """The p-vector as the search takes it: p[1:] in the coni basis for coni."""
    p = np.asarray(e["p"])
    return p[1:] if e["kind"] == "coni" else p


@pytest.mark.parametrize("e", EXAMPLES, ids=[_id(e) for e in EXAMPLES])
def test_paper_fluxes_form_a_pfv(e):
    """The fixture geometry is the paper's: its fluxes form a PFV there, with
    the paper's p (N p = K, up to the coni direction)."""
    data = _data(e)
    pfv = PFV(data, K=e["K"], M=e["M"])
    assert pfv.check_all(stop_at_fail=False)
    kappa = np.asarray(e["kappa"])
    if e["kind"] == "coni":
        B = np.asarray(e["cob"])
        kappa = np.einsum("ai,bj,ck,ijk->abc", B, B, B, kappa)
    N = np.einsum("abc,c->ab", kappa, np.asarray(e["M"]))
    lhs, rhs = N @ np.asarray(e["p"]), e["p_den"] * np.asarray(e["K"])
    if e["kind"] == "coni":
        lhs, rhs = lhs[1:], rhs[1:]
    assert np.array_equal(lhs, rhs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("e", EXAMPLES, ids=[_id(e) for e in EXAMPLES])
def test_finds_example_from_its_p(e, device):
    """From the paper's p-vector, the search returns the paper's (K, M)."""
    if device == "gpu" and e["kind"] != "coni":
        pytest.skip("the GPU backend runs the coni pipeline")
    data = _data(e)
    Ks, Ms = _search(e, data, np.array([_p_search(e)]), device)
    assert _found(e, Ks, Ms)


SCANS = [e for e in EXAMPLES if e["scan"] is not None]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("e", SCANS, ids=[_id(e) for e in SCANS])
def test_finds_example_in_full_scan(e, device):
    """A scan of every primitive p in the box |p|_inf <= max|p_i| of the
    Kahler cone (no knowledge of the paper's p) finds the example."""
    if e["scan"]["slow"] and not SLOW:
        pytest.skip("slow scan (set PFVS_SLOW_TESTS=1)")
    if device == "gpu" and e["kind"] != "coni":
        pytest.skip("the GPU backend runs the coni pipeline")
    data = _data(e)
    H = data.H_cob if e["kind"] == "coni" else data.H
    ps, st, _ = box_enum(e["scan"]["B"], np.ascontiguousarray(H.astype(np.int32)), 1,
                         10**8, primitive=True)
    assert st == 0 and len(ps) == e["scan"]["n_p"]
    assert any(np.array_equal(p, _p_search(e)) for p in ps)
    Ks, Ms = _search(e, data, ps, device)
    assert _found(e, Ks, Ms)


def test_coni_attributes_do_not_leak_to_non_coni():
    """A non-coni PFV made after a coni one has no coni-only attributes, and
    check_all does not apply coni-only checks to it (they used to be attached
    to the class by the first coni PFV)."""
    coni = next(e for e in EXAMPLES if e["kind"] == "coni")
    nonconi = next(e for e in EXAMPLES if e["kind"] == "non-coni" and e["section"] == "6.2")
    assert PFV(_data(coni), K=coni["K"], M=coni["M"]).check_Kprime()
    pfv = PFV(_data(nonconi), K=nonconi["K"], M=nonconi["M"])
    assert not hasattr(pfv, "Kprime") and not hasattr(pfv, "zcf")
    with pytest.raises(AttributeError):
        pfv.check_Kprime()
    assert pfv.check_all(stop_at_fail=False)
