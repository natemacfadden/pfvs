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
# Description:  pfvs.distributed: sharding is an exact partition of the p-box,
#               and a coordinator with CPU (and, if present, GPU) workers
#               returns exactly coniZpM's PFVs; a finished run resumes as done.
# -----------------------------------------------------------------------------

import socket
import threading

import numpy as np
import pytest
from latticepts import box_enum

from pfvs import coniZpM, distributed, gpu
from test_dataset import GEOMS, cydata

KEY = "pfvs-test-key"
CASES = [g for g in GEOMS if g["h11"] >= 4][:6]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pset(K, M, P=None):
    P = [()] * len(K) if P is None else P
    return {(tuple(map(int, k)), tuple(map(int, m))) for k, m in zip(K, M)}


def _expected(job):
    data = job["data"]
    ps, st, _ = box_enum(job["B"], np.ascontiguousarray(data.H_cob.astype(np.int32)), 1,
                         10**6, primitive=True)
    if len(ps) == 0:
        return set()
    K, M = coniZpM(data, ps, ellipsoid_dilation=job["D"], n_jobs=1, device="cpu")
    return _pset(K, M)


def _jobs(n_p=None):
    return distributed.make_jobs([cydata(g) for g in CASES], [g["B"] for g in CASES], 150,
                                 ids=[f"g{i}" for i in range(len(CASES))], n_p=n_p)


@pytest.mark.parametrize("n_units", [1, 3, 40])
def test_units_partition_the_box(n_units):
    g = max(CASES, key=lambda g: g["B"])
    job = distributed.make_jobs([cydata(g)], g["B"], 1, n_p=[n_units * 100])[0]
    shards = distributed._units_of(job, 100)
    assert len(shards) >= min(n_units, 2 * g["B"] + 1)
    parts = [distributed._enumerate_unit(job, s) for s in shards]
    got = [tuple(p) for part in parts for p in part]
    assert len(got) == len(set(got))                   # disjoint
    data = job["data"]
    ps, st, _ = box_enum(job["B"], np.ascontiguousarray(data.H_cob.astype(np.int32)), 1,
                         10**6, primitive=True)
    assert set(got) == {tuple(p) for p in ps}          # complete


@pytest.mark.parametrize("want", [1, 7, 50])
def test_split_shards_partition(want):
    g = max(CASES, key=lambda g: g["B"])
    job = distributed.make_jobs([cydata(g)], g["B"], 1)[0]
    dim = job["data"].H_cob.shape[1]
    base = distributed._enumerate_unit(job, [])
    children = distributed._split_shard([], job["B"], dim, len(base), want) or [[]]
    got = []
    for ch in children:                                 # and once more, recursively
        part = distributed._enumerate_unit(job, ch)
        grand = distributed._split_shard(ch, job["B"], dim, len(part), max(want // 2, 1)) or [ch]
        for gch in grand:
            got += [tuple(p) for p in distributed._enumerate_unit(job, gch)]
    assert len(got) == len(set(got)) and set(got) == {tuple(p) for p in base}


def _run(tmp_path, device, n_p=None, procs=2, max_unit_p=None):
    jobs = _jobs(n_p)
    port = _free_port()
    addr = f"127.0.0.1:{port}"
    out = str(tmp_path / "out")
    th = threading.Thread(target=distributed.serve, args=(jobs, out, addr, KEY),
                          kwargs=dict(target_p=64, verbose=False), daemon=True)
    th.start()
    distributed.work(addr, KEY, device=device, procs=procs, max_unit_p=max_unit_p)
    th.join(timeout=120)
    assert not th.is_alive()
    return jobs, out


def test_cpu_workers_match_coniZpM(tmp_path):
    jobs, out = _run(tmp_path, "cpu", n_p=[500] * len(CASES))
    res = distributed.load_results(out)
    assert set(res) == {j["id"] for j in jobs}
    for j in jobs:
        r = res[j["id"]]
        assert _pset(r["K"], r["M"]) == _expected(j)
        assert r["P"].shape == (len(r["K"]), j["data"].h11 - 1)
    # resuming a finished run does nothing and changes nothing
    pr = distributed.serve(jobs, out, f"127.0.0.1:{_free_port()}", KEY, verbose=False)
    assert pr["done"] == pr["units"] and pr["failed"] == 0


def test_splitting_workers_match_coniZpM(tmp_path):
    """Units far larger than the workers accept are split (recursively) on
    the fly; the results are unchanged."""
    jobs, out = _run(tmp_path, "cpu", procs=2, max_unit_p=40)
    res = distributed.load_results(out)
    for j in jobs:
        assert _pset(res[j["id"]]["K"], res[j["id"]]["M"]) == _expected(j)
    import os
    assert os.path.exists(os.path.join(out, "splits.pkl"))
    pr = distributed.serve(jobs, out, f"127.0.0.1:{_free_port()}", KEY, verbose=False)
    assert pr["done"] == pr["units"] and pr["failed"] == 0      # resumes with the splits


@pytest.mark.skipif(not gpu.available(), reason=gpu.unavailable_reason() or "")
def test_gpu_worker_matches_coniZpM(tmp_path):
    jobs, out = _run(tmp_path, "gpu:0", n_p=[500] * len(CASES))
    res = distributed.load_results(out)
    for j in jobs:
        assert _pset(res[j["id"]]["K"], res[j["id"]]["M"]) == _expected(j)


def test_authkey_required(tmp_path):
    jobs = _jobs()
    addr = f"127.0.0.1:{_free_port()}"
    th = threading.Thread(target=distributed.serve, args=(jobs, str(tmp_path / "o"), addr, KEY),
                          kwargs=dict(verbose=False), daemon=True)
    th.start()
    import multiprocessing
    with pytest.raises((multiprocessing.AuthenticationError, OSError, EOFError)):
        distributed._connect(addr, "wrong-key", retries=3)
    distributed.work(addr, KEY, device="cpu", procs=1)      # let it finish
    th.join(timeout=120)
