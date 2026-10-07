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
# Description:  Coni-PFV searches spread over many machines and devices.
#
#   A coordinator holds jobs -- one per geometry: every primitive p-vector in
#   its box |p|_inf <= B (the coniZpM setting), at dilation D -- split into
#   units: disjoint shards of the p-box (p restricted to value ranges of one
#   or two coordinates). Workers on any machine connect over TCP, lease
#   units, search them and send back the PFVs:
#     - GPU workers (NVIDIA or AMD, one per device) batch many units into one
#       device call and overlap the host work (p enumeration, post-processing)
#       with the device;
#     - CPU workers run the CPU pipeline in N processes.
#   Every unit's result is checkpointed on the coordinator's disk as it
#   arrives (a restarted coordinator resumes); units whose worker vanished are
#   reissued after a lease timeout; a failing unit is retried, then recorded.
#   When all units of a job are in, its PFVs are written to <out>/jobs/.
#
#   Results equal coniZpM's over the job's whole p-box (as a set; the order
#   is by unit, then p).
#
#   Exhaustive jobs (make_jobs(..., exhaustive=True)) find every coni PFV of
#   each direction, as coniZpM(..., exhaustive=True): a worker computes each
#   p-vector's dilation bound and routes it, at the coordinator's current
#   price lam (pfvs.dilation.CostModel.route_at_price), either to ZpM up to
#   its bound or to ZpM up to a split dilation D0 plus ZpK above D0. The ZpK
#   halves come back to the coordinator as ZpK units, which only CPU workers
#   take while search units remain. The coordinator raises lam (fewer
#   splits) while ZpK work piles up and lowers it while the CPU workers run
#   short of it, so the GPU and CPU workers finish together. Without GPU
#   workers, CPU workers do whole exhaustive search units themselves (ZpM and
#   ZpK on the CPU); GPU workers without search units do ZpK units on host
#   threads.
#
#   Usage (see README):
#     coordinator:  python -m pfvs.distributed serve jobs.pkl OUT --address 0.0.0.0:5055
#     workers:      python -m pfvs.distributed work HOST:5055 --device gpu:0
#                   python -m pfvs.distributed work HOST:5055 --device cpu --procs 16
#   with the same PFVS_AUTHKEY in the environment of both (the connection
#   is authenticated; run it on a trusted network only -- it exchanges
#   pickles).
# -----------------------------------------------------------------------------

import argparse
import math
import os
import pickle
import queue
import socket
import sys
import threading
import time
import traceback
import warnings
from multiprocessing.managers import BaseManager

import numpy as np

__all__ = ["make_jobs", "serve", "work", "load_results"]


# jobs and units
# --------------
def make_jobs(datas, B, D, Q=None, M0min=13, ids=None, n_p=None, exhaustive=False,
              cost_model=None):
    """
    **Description:**
    Jobs for `serve`: one per geometry.

    **Arguments:**
    - `datas` *(list of CYData)*: The geometries.
    - `B` *(int or list)*: p-box half-width (all primitive p in the Kahler
        cone with |p|_inf <= B, as coniZpM is usually driven).
    - `D` *(float or list)*: Ellipsoid dilation.
    - `Q` *(int, list or None)*: Tadpole (default h11 + h21 + 4).
    - `M0min` *(int, optional)*: As in coniZpM.
    - `ids` *(list, optional)*: Job names (default 0, 1, ...); used for the
        output file names, so they must be unique.
    - `n_p` *(list, optional)*: Estimated p-vector counts; used to shard large
        boxes into units of about `target_p` p-vectors (see `serve`).
    - `exhaustive` *(bool or list, optional)*: Every coni PFV of each
        direction, at any dilation (see the module description); `D` is then
        only used for p-vectors without a dilation bound, which are reported
        as incomplete.
    - `cost_model` *(CostModel or list, optional)*: The costs GPU workers
        route exhaustive jobs with (`pfvs.dilation.CostModel`: ZpM on the
        GPU, ZpK on the CPU workers). Only their shape matters -- the price
        lam absorbs the scale. Default: costs measured for the job's h11
        on an RTX 5090 and 24 CPU cores (so lam ~ 1 when the cluster has
        about 24 CPU cores per GPU).

    **Returns:**
    *(list of dict)* The jobs.
    """
    n = len(datas)
    def per(x):
        return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x] * n
    B, D, Q, M0min = per(B), per(D), per(Q), per(M0min)
    ids = list(ids) if ids is not None else list(range(n))
    n_p, exhaustive, cost_model = per(n_p), per(exhaustive), per(cost_model)
    if len(set(map(str, ids))) != n:
        raise ValueError("job ids must be unique")
    return [dict(id=str(ids[i]), data=datas[i], B=int(B[i]), D=float(D[i]),
                 Q=None if Q[i] is None else int(Q[i]), M0min=M0min[i], n_p=n_p[i],
                 exhaustive=bool(exhaustive[i]), cost_model=cost_model[i])
            for i in range(n)]


def _units_of(job, target_p):
    """Shard a job's p-box into units: lists of (coordinate, lo, hi)."""
    B = job["B"]
    n_est = job.get("n_p")
    if not n_est or n_est <= target_p:
        return [[]]
    dim = job["data"].H_cob.shape[1]
    want = math.ceil(n_est / target_p)
    vals = list(range(-B, B + 1))
    if want <= len(vals) or dim < 2:
        # contiguous value ranges of coordinate 0
        k = min(want, len(vals))
        bounds = np.linspace(0, len(vals), k + 1).round().astype(int)
        return [[(0, vals[a], vals[b - 1])] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    # both coordinates 0 and 1: every value of 0, ranges of 1
    k1 = min(math.ceil(want / len(vals)), len(vals))
    bounds = np.linspace(0, len(vals), k1 + 1).round().astype(int)
    return [[(0, v, v), (1, vals[a], vals[b - 1])] for v in vals
            for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _enumerate_unit(job, shard, parallel=False):
    """The p-vectors of a unit (rows, p[1:]), in box_enum's order."""
    from latticepts import box_enum
    H = np.ascontiguousarray(job["data"].H_cob, dtype=np.int32)
    rhs = np.ones(len(H), dtype=np.int64)
    for j, lo, hi in shard:                    # lo <= p_j <= hi
        e = np.zeros((1, H.shape[1]), dtype=np.int32); e[0, j] = 1
        H = np.vstack([H, e, -e]); rhs = np.concatenate([rhs, [lo, -hi]])
    H = np.ascontiguousarray(H)
    cap = max(4096, int(1.25 * (job.get("n_p") or 0) / max(1, job.get("_n_units", 1))))
    while True:
        # CPU workers enumerate single-threaded (they are the parallelism);
        # GPU workers with a few OpenMP threads (see _gpu_worker)
        ps, st, _ = box_enum(job["B"], H, rhs, cap, primitive=True, parallel=parallel)
        if st == 0:
            return np.ascontiguousarray(ps, dtype=np.int64)
        if len(ps) < cap:                       # not a full buffer: a real failure
            raise RuntimeError(f"box_enum status {st}")
        cap *= 4


def _job_Q(job):
    d = job["data"]
    return job["Q"] if job["Q"] is not None else d.h11 + d.h21 + 4


def _split_shard(shard, B, dim, n_found, want):
    """Split a shard into about n_found / want pieces along its first
    coordinate that still spans several values; None if it cannot be split."""
    ranges = {j: (lo, hi) for j, lo, hi in shard}
    for j in range(dim):
        lo, hi = ranges.get(j, (-B, B))
        if hi > lo:
            m = int(min(math.ceil(n_found / max(want, 1)), hi - lo + 1))
            if m < 2:
                return None
            cuts = np.linspace(lo, hi + 1, m + 1).round().astype(int)
            rest = [c for c in shard if c[0] != j]
            return [rest + [(j, int(a), int(b) - 1)] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    return None


# the coordinator
# ---------------
class _Coordinator:
    """Lives in the manager's server process; its methods are the RPCs.

    Units are keyed (job id, path): a path is a tuple of ints -- (k,) for the
    k-th initial shard of a job, (k, i) for the i-th piece it was split into,
    and so on; path + (-1,) is the ZpK unit of an exhaustive search unit
    (its split p-vectors). A job is complete when all its leaf units are
    done; its PFVs are the leaves' in path order. Splits are logged
    (splits.pkl) so a resumed run rebuilds the same leaves; ZpK units are
    rebuilt from their search units' results."""

    # the price controller: lam is raised (fewer splits) while the pending
    # ZpK work exceeds LAM_HI seconds of the CPU workers' recent throughput,
    # lowered below LAM_LO, at most every LAM_DT seconds
    LAM_HI, LAM_LO, LAM_DT, LAM_STEP, RATE_WINDOW = 600.0, 60.0, 30.0, 1.5, 300.0

    def __init__(self, jobs, out, target_p, lease_s, max_tries, backup_s=30.0, lam0=1.0):
        self.out = out
        self.lease_s = lease_s
        self.backup_s = backup_s
        self.max_tries = max_tries
        self.lock = threading.Lock()
        os.makedirs(os.path.join(out, "units"), exist_ok=True)
        os.makedirs(os.path.join(out, "jobs"), exist_ok=True)
        self.jobs = {j["id"]: j for j in jobs}
        self.shard, self.est, self.leaves = {}, {}, {}
        self.zpk = {}                                   # ZpK unit key -> (ps, D0)
        for j in jobs:
            shards = _units_of(j, target_p)
            j["_n_units"] = len(shards)
            self.leaves[j["id"]] = set()
            for k, sh in enumerate(shards):
                key = (j["id"], (k,))
                self.shard[key] = sh
                self.est[key] = (j.get("n_p") or 1000) / len(shards)
                self.leaves[j["id"]].add(key)
        self.split_log = []
        path = os.path.join(out, "splits.pkl")
        if os.path.exists(path):                        # resume: replay the splits
            with open(path, "rb") as f:
                self.split_log = pickle.load(f)
            for key, children, n_found in self.split_log:
                if key in self.shard:
                    self._apply_split(key, children, n_found)
        self.state, self.tries, self.failed = {}, {}, {}
        self.backups, self.workers, self.rate = {}, {}, {}
        self.t0 = time.time()
        self.n_p_done = 0
        self.lam, self.lam_t = float(lam0), 0.0
        self.zpk_log = []                               # (time, p-vectors) of finished ZpK units
        self.pending_zpk = []
        self.zpk_hungry = False                         # a CPU worker found no ZpK work
        self.zpk_seen = False                           # some search unit made ZpK work
        self.gpu_t = -math.inf                          # when a GPU worker last asked for work
        for jid in self.jobs:                           # resume: finished units
            for key in list(self.leaves[jid]):
                if os.path.exists(self._unit_path(key)):
                    self.state[key] = "done"
                    if self.jobs[jid].get("exhaustive") and not _is_zpk(key):
                        with open(self._unit_path(key), "rb") as f:
                            r = pickle.load(f)
                        self._add_zpk(key, r)
        self.pending = [key for jid in self.jobs for key in sorted(self.leaves[jid], key=lambda k: k[1])
                        if key not in self.state and not _is_zpk(key)]
        for jid in self.jobs:
            self._maybe_finish(jid)

    def _unit_path(self, key):
        jid, path = key
        return os.path.join(self.out, "units", f"{jid}__{'.'.join(map(str, path))}.pkl")

    def _add_zpk(self, key, result):
        """The ZpK unit of a finished exhaustive search unit, if it has one."""
        ps = result.get("zpk")
        if ps is None or len(ps) == 0:
            return
        jid, path = key
        zk = (jid, tuple(path) + (-1,))
        self.zpk_seen = True
        self.zpk[zk] = (ps, int(result["D0"]))
        self.est[zk] = len(ps)
        self.leaves[jid].add(zk)
        if os.path.exists(self._unit_path(zk)):
            self.state[zk] = "done"
        elif zk not in self.pending_zpk:
            self.pending_zpk.append(zk)

    def _apply_split(self, key, children, n_found):
        jid, path = key
        self.leaves[jid].discard(key)
        out = []
        for i, sh in enumerate(children):
            ck = (jid, path + (i,))
            self.shard[ck] = sh
            self.est[ck] = n_found / len(children)
            self.leaves[jid].add(ck)
            out.append(ck)
        return out

    def _n_done(self):
        return sum(1 for jid in self.jobs for key in self.leaves[jid] if self.state.get(key) == "done")

    def _n_leaves(self):
        return sum(len(v) for v in self.leaves.values())

    def _maybe_finish(self, jid):
        leaves = sorted(self.leaves[jid], key=lambda k: k[1])
        if any(self.state.get(key) != "done" for key in leaves):
            return
        path = os.path.join(self.out, "jobs", f"{jid}.pkl")
        if os.path.exists(path):
            return
        job = self.jobs[jid]
        h = job["data"].h11
        Ks, Ms, Ps, inc, n_p = [], [], [], [], 0
        for key in leaves:
            with open(self._unit_path(key), "rb") as f:
                r = pickle.load(f)
            Ks.append(r["K"]); Ms.append(r["M"]); Ps.append(r["P"]); n_p += r["n_p"]
            if r.get("incomplete") is not None:
                inc.append(r["incomplete"])
        K = np.vstack(Ks) if Ks else np.zeros((0, h), np.int64)
        M = np.vstack(Ms) if Ms else np.zeros((0, h), np.int64)
        P = np.vstack(Ps) if Ps else np.zeros((0, h - 1), np.int64)
        res = dict(id=jid, K=K, M=M, P=P, n_p=n_p, B=job["B"], D=job["D"], Q=_job_Q(job))
        if job.get("exhaustive"):
            # (ZpK's PFVs can repeat ZpM's): each once, by p, then (M, K)
            if len(K):
                A = np.hstack([P, M, K])
                if A.dtype == object:
                    A = np.array(sorted({tuple(int(x) for x in r) for r in A}), dtype=object)
                else:
                    A = np.unique(A, axis=0)
                res.update(P=A[:, :h - 1], M=A[:, h - 1:2 * h - 1], K=A[:, 2 * h - 1:])
            res.update(exhaustive=True,
                       incomplete=np.vstack(inc) if inc else np.zeros((0, h - 1), np.int64))
        _atomic_pickle(path, res)

    def _unit(self, key):
        if _is_zpk(key):
            ps, D0 = self.zpk[key]
            return dict(job=self.jobs[key[0]], path=key[1], kind="zpk", ps=ps, D0=D0)
        return dict(job=self.jobs[key[0]], path=key[1], kind="search", shard=self.shard[key],
                    lam=self.lam)

    def _update_lam(self, now):
        """The price controller (see LAM_HI): only while search units remain."""
        if now - self.lam_t < self.LAM_DT or not self.pending:
            return
        self.lam_t = now
        self.zpk_log = [(t, n) for t, n in self.zpk_log if now - t < self.RATE_WINDOW]
        backlog = sum(len(self.zpk[k][0]) for k in self.pending_zpk)
        hungry, self.zpk_hungry = self.zpk_hungry, False
        if not backlog:
            if hungry and self.zpk_seen:                # CPU workers idle: more splits
                self.lam = max(self.lam / self.LAM_STEP, 1e-6)
            return
        rate = sum(n for _, n in self.zpk_log) / min(self.RATE_WINDOW, max(now - self.t0, 1.0))
        secs = backlog / rate if rate > 0 else math.inf
        if secs > self.LAM_HI:
            self.lam = min(self.lam * self.LAM_STEP, 1e6)
        elif secs < self.LAM_LO:
            self.lam = max(self.lam / self.LAM_STEP, 1e-6)

    # --- RPCs ---
    def get_work(self, worker, max_p, kinds=("search",), gpu=False):
        """Units totalling about max_p p-vectors (estimated) of the given
        kinds ("zpk": ZpK units; "search"), in that order of preference, or
        [] if none is available now; None when everything is done. While a
        GPU worker (gpu=True) is active, CPU workers get no exhaustive search
        units (their ZpM would be far slower than the GPU's)."""
        with self.lock:
            now = time.time()
            self.workers[worker] = now
            if gpu:
                self.gpu_t = now
            skip_ex = not gpu and now - self.gpu_t < 60.0
            self._update_lam(now)
            if not (self.pending or self.pending_zpk):  # reissue expired leases
                for key, st in list(self.state.items()):
                    if isinstance(st, tuple) and now - st[1] > self.lease_s:
                        (self.pending_zpk if _is_zpk(key) else self.pending).append(key)
                        del self.state[key]
            if "zpk" in kinds and not self.pending_zpk:
                self.zpk_hungry = True
            queues = [self.pending_zpk if k == "zpk" else self.pending for k in kinds]
            if not any(queues):
                if not (self.pending or self.pending_zpk) and \
                        self._n_done() + len(self.failed) >= self._n_leaves():
                    return None
                # backup copies of long-leased units, for a worker much faster
                # than the holder (a slow worker must not hold up the end;
                # the first result wins)
                mine = self.rate.get(worker, 0.0)
                slow = sorted((st[1], key) for key, st in self.state.items()
                              if isinstance(st, tuple) and st[0] != worker
                              and ("zpk" if _is_zpk(key) else "search") in kinds
                              and not (skip_ex and not _is_zpk(key) and self.jobs[key[0]].get("exhaustive"))
                              and now - st[1] > self.backup_s and self.backups.get(key, 0) < 2
                              and mine >= 4 * self.rate.get(st[0], 0.0) and mine > 0)
                out = []
                for _, key in slow[:8]:
                    self.backups[key] = self.backups.get(key, 0) + 1
                    out.append(self._unit(key))
                return out
            out, tot = [], 0.0
            for q in queues:
                i = 0
                while i < len(q) and (not out or tot < max_p):
                    key = q[i]
                    if skip_ex and q is self.pending and self.jobs[key[0]].get("exhaustive"):
                        i += 1
                        continue
                    q.pop(i)
                    if self.state.get(key) == "done" or (key not in self.shard and key not in self.zpk):
                        continue
                    self.state[key] = (worker, now)
                    out.append(self._unit(key))
                    tot += self.est[key]
                if out:
                    break
            return out

    def split(self, worker, jid, path, n_found, want):
        """A worker found a unit much larger than it should take: split it
        (the worker drops it). Returns False if it cannot be split."""
        with self.lock:
            key = (jid, tuple(path))
            if key not in self.leaves[jid] or self.state.get(key) == "done" or _is_zpk(key):
                return False
            job = self.jobs[jid]
            children = _split_shard(self.shard[key], job["B"], job["data"].H_cob.shape[1],
                                    n_found, want)
            if not children:
                return False
            self.state.pop(key, None)
            self.split_log.append((key, children, n_found))
            _atomic_pickle(os.path.join(self.out, "splits.pkl"), self.split_log)
            new = self._apply_split(key, children, n_found)
            self.pending[:0] = new                      # next in line
            return True

    def put_result(self, worker, jid, path, result):
        result["worker"] = worker
        with self.lock:
            now = time.time()
            key = (jid, tuple(path))
            self.workers[worker] = now
            st = self.state.get(key)
            if st == "done" or (key not in self.shard and key not in self.zpk):
                return                                  # a backup copy, done twice
            if isinstance(st, tuple) and st[0] == worker and now > st[1] and not _is_zpk(key):
                r = result["n_p"] / (now - st[1])
                self.rate[worker] = r if worker not in self.rate else 0.7 * self.rate[worker] + 0.3 * r
            _atomic_pickle(self._unit_path(key), result)
            self.state[key] = "done"
            self.n_p_done += result["n_p"]
            if _is_zpk(key):
                self.zpk_log.append((now, len(self.zpk[key][0])))
            elif self.jobs[jid].get("exhaustive"):
                self._add_zpk(key, result)              # (before the job can finish)
            self._update_lam(now)
            self._maybe_finish(jid)

    def put_error(self, worker, jid, path, msg):
        with self.lock:
            key = (jid, tuple(path))
            if self.state.get(key) == "done":
                return                                  # a backup copy failed; no matter
            self.tries[key] = self.tries.get(key, 0) + 1
            self.state.pop(key, None)
            if self.tries[key] < self.max_tries:
                (self.pending_zpk if _is_zpk(key) else self.pending).append(key)
            else:
                self.failed[key] = msg
                with open(os.path.join(self.out, "failed.txt"), "a") as f:
                    f.write(f"{jid}\t{'.'.join(map(str, key[1]))}\t{msg}\n")

    def progress(self):
        with self.lock:
            now = time.time()
            active = sum(1 for t in self.workers.values() if now - t < 120)
            return dict(units=self._n_leaves(), done=self._n_done(), failed=len(self.failed),
                        leased=sum(1 for v in self.state.values() if isinstance(v, tuple)),
                        p_done=self.n_p_done, seconds=now - self.t0, workers=active,
                        lam=self.lam, zpk_pending=sum(len(self.zpk[k][0]) for k in self.pending_zpk))


def _is_zpk(key):
    return key[1][-1] == -1


def _atomic_pickle(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)


class _ServerManager(BaseManager):
    pass


class _ClientManager(BaseManager):
    pass


_ClientManager.register("coordinator")


def _authkey(authkey):
    key = authkey if authkey is not None else os.environ.get("PFVS_AUTHKEY")
    if not key:
        raise ValueError("an authkey is required (argument or PFVS_AUTHKEY)")
    return key.encode() if isinstance(key, str) else key


def _parse_address(address):
    host, _, port = address.rpartition(":")
    return (host or "0.0.0.0", int(port))


def serve(jobs, out, address="0.0.0.0:5055", authkey=None, target_p=1 << 20,
          lease_s=3600.0, max_tries=3, poll_s=10.0, verbose=True, backup_s=30.0, lam0=1.0):
    """
    **Description:**
    Run the coordinator until every unit is done (or has failed `max_tries`
    times). Blocks; results go to `out` (see `load_results`).

    **Arguments:**
    - `jobs` *(list of dict)*: From `make_jobs`.
    - `out` *(str)*: Output directory (created; an existing one is resumed).
    - `address` *(str)*: host:port to listen on.
    - `authkey` *(str, optional)*: Shared secret (default: PFVS_AUTHKEY).
    - `target_p` *(int, optional)*: p-vectors per unit for sharding jobs with
        an n_p estimate.
    - `lease_s` *(float, optional)*: Reissue a unit not returned in this long.
    - `max_tries` *(int, optional)*: Attempts per unit before it is recorded
        in <out>/failed.txt.
    - `backup_s` *(float, optional)*: Once no unit is pending, a unit leased
        for longer than this is also given to an idle worker (the first
        result wins), so slow workers do not hold up the end.
    - `lam0` *(float, optional)*: The initial price of exhaustive jobs'
        routing (GPU time per CPU time; see the module description).

    **Returns:**
    *(dict)* The final progress counters.
    """
    coord = _Coordinator(jobs, out, target_p, lease_s, max_tries, backup_s, lam0)
    _ServerManager.register("coordinator", callable=lambda: coord)
    mgr = _ServerManager(address=_parse_address(address), authkey=_authkey(authkey))
    server = mgr.get_server()
    def run_server():
        try:
            server.serve_forever()
        except SystemExit:          # (serve_forever exits via sys.exit on stop)
            pass

    th = threading.Thread(target=run_server, daemon=True)
    th.start()
    if verbose:
        pr = coord.progress()
        print(f"pfvs.distributed: serving {pr['units']} units ({pr['done']} already done) "
              f"on {address}", flush=True)
    last = None
    while True:
        pr = coord.progress()
        if pr["done"] + pr["failed"] >= pr["units"]:
            break
        if verbose and (last is None or time.time() - last >= poll_s):
            rate = pr["p_done"] / max(pr["seconds"], 1e-9)
            ex = f", lam {pr['lam']:.3g}, ZpK pending {pr['zpk_pending']:,} p" \
                if any(j.get("exhaustive") for j in jobs) else ""
            print(f"  {pr['done']}/{pr['units']} units, {pr['failed']} failed, "
                  f"{pr['leased']} in progress, {pr['workers']} workers, "
                  f"{pr['p_done']:,} p ({rate:,.0f} p/s){ex}", flush=True)
            last = time.time()
        time.sleep(0.5)
    time.sleep(1.0)                 # let workers see "done"
    server.stop_event.set()
    pr = coord.progress()
    if verbose:
        print(f"pfvs.distributed: finished {pr['done']}/{pr['units']} units "
              f"({pr['failed']} failed) in {pr['seconds']:.1f} s", flush=True)
    return pr


def load_results(out):
    """
    **Description:**
    The finished jobs of an output directory.

    **Returns:**
    *(dict)* job id -> dict(K, M, P, n_p, B, D, Q): the PFVs (K, M) and each
    one's p-vector P = p[1:]. Exhaustive jobs also have `incomplete`: the
    p-vectors without a dilation bound (searched at dilation D only); their
    PFVs are each listed once, by p-vector, then (M, K).
    """
    res = {}
    d = os.path.join(out, "jobs")
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.endswith(".pkl"):
            with open(os.path.join(d, name), "rb") as f:
                r = pickle.load(f)
            res[r["id"]] = r
    return res


# workers
# -------
def _post_process(job, ps, M, Kn, q, pidx, pstat, dil=None):
    """PFVs of one unit from its lattice points (the coniZpM post-processing);
    p-vectors the batched path could not finish go through coniZpM (at
    dil[i], each p-vector's dilation, if given; else the job's)."""
    from .coniZp import _pfvs_from_points, coniZpM
    from .fp_kernel.fp_kernel import _coni_batch  # noqa: F401  (import check)
    data = job["data"]
    h, Q = data.h11, _job_Q(job)
    Ks, Ms, keys = _pfvs_from_points(np.asarray(M).T, np.asarray(Kn).T, q, pidx,
                                     data.kappa_cob, h, Q, job["M0min"])
    Ks, Ms, keys = [Ks], [Ms], [keys]
    for ip in np.flatnonzero(pstat != 0):
        if pstat[ip] < 0:
            from .coniZp import _raise_kernel_status
            _raise_kernel_status(ps[ip], int(pstat[ip]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K1, M1 = coniZpM(data, ps[ip:ip + 1], Q=Q, M0min=job["M0min"],
                             ellipsoid_dilation=job["D"] if dil is None else float(dil[ip]),
                             n_jobs=1, device="cpu")
        Ks.append(np.asarray(K1).reshape(-1, h)); Ms.append(np.asarray(M1).reshape(-1, h))
        keys.append(np.full(len(K1), ip, dtype=np.int64))
    K, Mo, key = np.vstack(Ks), np.vstack(Ms), np.concatenate(keys)
    order = np.argsort(key, kind="stable")
    return dict(K=K[order], M=Mo[order], P=ps[key[order]] if len(key) else
                np.zeros((0, ps.shape[1] if ps.ndim == 2 else h - 1), np.int64), n_p=len(ps))


def _cpu_unit(job, ps):
    """One unit on the CPU (the batched C pipeline + post-processing)."""
    from .fp_kernel.fp_kernel import _coni_batch
    data = job["data"]
    h = data.h11
    if len(ps) == 0:
        return dict(K=np.zeros((0, h), np.int64), M=np.zeros((0, h), np.int64),
                    P=np.zeros((0, h - 1), np.int64), n_p=0)
    p_full = np.hstack([np.zeros((len(ps), 1), np.int64), ps])
    M, Kn, q, pidx, pstat = _coni_batch(data.kappa_cob, data.M_lattice(), p_full, _job_Q(job),
                                        job["D"], job["M0min"], 10**9, True)
    return _post_process(job, ps, M, Kn, q, pidx, pstat)


def _route(job, ps, lam, n_threads):
    """
    An exhaustive search unit's plan at price lam: each p-vector's ZpM
    dilation (its bound rounded up, the split dilation D0, or the job's D
    without a bound), the p-vectors split at D0 (their ZpK half becomes a
    ZpK unit), D0, and the p-vectors without a bound.
    """
    from .coniZp import _default_cost_model, _grid_up
    from .dilation import coni_dilation_bound_ceils
    data = job["data"]
    bceil = coni_dilation_bound_ceils(ps, data.kappa_cob, _job_Q(job), n_jobs=n_threads)
    has = bceil > 0
    model = job.get("cost_model") or _default_cost_model(True, data.h11, n_cpu=24)
    to_bound, D0 = model.route_at_price(np.where(has, bceil, np.inf), lam)
    to_bound |= has & (bceil <= D0)
    split = has & ~to_bound
    dil = np.where(to_bound, _grid_up(bceil), np.where(split, D0, 0)).astype(float)
    dil[~has] = job["D"]
    return dil, ps[split], D0, ps[~has]


def _zpk_unit(job, ps, D0, n_threads=1):
    """A ZpK unit: the PFVs above D0 of its p-vectors (all with a bound);
    those ZpK cannot do exactly are searched by ZpM up to their bound."""
    from .coniZp import _batch_points, _grid_up, _pfvs_from_points, coniZpM
    from .dilation import coni_dilation_bounds
    from .fp_kernel.fp_kernel import _coni_zpk_batch
    data = job["data"]
    h, Q, M0min, kap = data.h11, _job_Q(job), job["M0min"], data.kappa_cob
    p_full = np.hstack([np.zeros((len(ps), 1), np.int64), ps])
    M, Kn, q, pidx, st = _batch_points(_coni_zpk_batch, kap, data.M_lattice(), p_full,
                                       np.arange(len(ps)), n_threads, Q, M0min, D0, 10**9)
    Ks, Ms, keys = _pfvs_from_points(M.T, Kn.T, q, pidx, kap, h, Q, M0min)
    Ks, Ms, keys = [Ks], [Ms], [keys]
    bad = np.flatnonzero(st != 0)
    for i, b in zip(bad, coni_dilation_bounds(ps[bad], kap, Q)):
        if b is None:
            raise RuntimeError(f"p={ps[i].tolist()}: split, but without a dilation bound")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K1, M1 = coniZpM(data, ps[i:i + 1], Q=Q, M0min=M0min,
                             ellipsoid_dilation=float(_grid_up(math.ceil(b))), n_jobs=1, device="cpu")
        Ks.append(np.asarray(K1).reshape(-1, h)); Ms.append(np.asarray(M1).reshape(-1, h))
        keys.append(np.full(len(K1), i, dtype=np.int64))
    key = np.concatenate(keys)
    return dict(K=np.vstack(Ks), M=np.vstack(Ms), P=ps[key] if len(key) else np.zeros((0, h - 1), np.int64),
                n_p=0)


def _cpu_exhaustive_unit(job, ps):
    """An exhaustive search unit wholly on the CPU (ZpM and ZpK)."""
    from .coniZp import _coni_exhaustive, _default_cost_model
    data = job["data"]
    h = data.h11
    if len(ps) == 0:
        return dict(K=np.zeros((0, h), np.int64), M=np.zeros((0, h), np.int64),
                    P=np.zeros((0, h - 1), np.int64), n_p=0, incomplete=np.zeros((0, h - 1), np.int64))
    p_full = np.hstack([np.zeros((len(ps), 1), np.int64), ps])
    Ks, Ms, keys, has = _coni_exhaustive(data, ps, p_full, _job_Q(job), job["M0min"], job["D"],
                                         False, False, 1, 10**9, _default_cost_model(False, h), 0)
    return dict(K=Ks, M=Ms, P=ps[keys] if len(keys) else np.zeros((0, h - 1), np.int64),
                n_p=len(ps), incomplete=ps[~has])


def _connect(address, authkey, retries=30):
    for i in range(retries):
        try:
            mgr = _ClientManager(address=_parse_address(address), authkey=_authkey(authkey))
            mgr.connect()
            return mgr.coordinator()
        except (ConnectionRefusedError, OSError):
            if i == retries - 1:
                raise
            time.sleep(2.0)


def _cpu_worker(address, authkey, name, batch_p, max_unit_p=1 << 17):
    coord = _connect(address, authkey)
    while True:
        try:
            units = coord.get_work(name, batch_p, ("zpk", "search"))
        except (EOFError, ConnectionError, OSError):
            return                                     # the coordinator is gone
        if units is None:
            return
        if not units:
            time.sleep(2.0)
            continue
        for u in units:
            job = u["job"]
            try:
                if u.get("kind") == "zpk":
                    coord.put_result(name, job["id"], u["path"], _zpk_unit(job, u["ps"], u["D0"]))
                    continue
                ps = _enumerate_unit(job, u["shard"])
                if len(ps) > max_unit_p and coord.split(name, job["id"], u["path"], len(ps), max_unit_p):
                    continue                           # too big for a CPU worker: split
                res = _cpu_exhaustive_unit(job, ps) if job.get("exhaustive") else _cpu_unit(job, ps)
                coord.put_result(name, job["id"], u["path"], res)
            except Exception:
                coord.put_error(name, job["id"], u["path"], traceback.format_exc(limit=3)[-900:])


def _gpu_worker(address, authkey, name, device, batch_p, verbose, max_unit_p=1 << 23,
                bound_threads=None):
    """GPU worker: fetch+enumerate(+route) -> device -> post-process, as
    three overlapping stages (the ctypes device call releases the GIL).
    Exhaustive units' bounds are computed in bound_threads threads (default:
    half the cores)."""
    bound_threads = bound_threads or max(1, (os.cpu_count() or 2) // 2)
    # the enumeration of the next units uses a few OpenMP threads (set before
    # the OpenMP runtime starts), leaving the rest of the machine to others
    os.environ.setdefault("OMP_NUM_THREADS", str(max(2, (os.cpu_count() or 8) // 6)))
    from . import gpu
    if not gpu.available():
        raise RuntimeError("GPU worker: " + gpu.unavailable_reason())
    coord = _connect(address, authkey)
    q_in, q_out = queue.Queue(maxsize=2), queue.Queue(maxsize=2)
    STOP = object()
    tm = dict(rpc_get=0.0, enum=0.0, call=0.0, wait_in=0.0, post=0.0, rpc_put=0.0)

    def fetch():
        while True:
            t = time.perf_counter()
            try:
                # (ZpK units only when no search unit is left: on host threads)
                units = coord.get_work(name, batch_p, ("search", "zpk"), True)
            except (EOFError, ConnectionError, OSError):
                units = None                           # the coordinator is gone
            if units is None:
                q_in.put(STOP)
                return
            if not units:
                time.sleep(2.0)
                continue
            tm["rpc_get"] += time.perf_counter() - t
            batch = []
            t = time.perf_counter()
            for u in units:
                try:
                    if u.get("kind") == "zpk":
                        coord.put_result(name, u["job"]["id"], u["path"],
                                         _zpk_unit(u["job"], u["ps"], u["D0"], bound_threads))
                        continue
                    ps = _enumerate_unit(u["job"], u["shard"], parallel=True)
                    if len(ps) > max_unit_p and coord.split(name, u["job"]["id"], u["path"],
                                                           len(ps), max_unit_p):
                        continue
                    dil = extra = None
                    if u["job"].get("exhaustive") and len(ps):
                        dil, zps, D0, inc = _route(u["job"], ps, u["lam"], bound_threads)
                        extra = dict(zpk=zps, D0=D0, incomplete=inc)
                    batch.append((u, ps, dil, extra))
                except Exception:
                    coord.put_error(name, u["job"]["id"], u["path"], traceback.format_exc(limit=3)[-900:])
            tm["enum"] += time.perf_counter() - t
            if batch:
                q_in.put(batch)

    def post():
        while True:
            item = q_out.get()
            if item is STOP:
                return
            batch, res, err = item
            if err and verbose:
                print(f"[{name}] device call failed; doing its {len(batch)} units on the CPU:\n{err}",
                      flush=True)
            for bi, (u, ps, dil, extra) in enumerate(batch):
                job = u["job"]
                try:
                    if err:                             # the device failed: CPU instead
                        coord.put_result(name, job["id"], u["path"],
                                         _cpu_exhaustive_unit(job, ps) if job.get("exhaustive")
                                         else _cpu_unit(job, ps))
                        continue
                    h = job["data"].h11
                    M, Kn, q, pidx, pstat, lo, hi = res[bi]
                    t = time.perf_counter()
                    r = _post_process(job, ps, M[:, :h], Kn[:, :h], q, pidx - lo, pstat[lo:hi], dil) \
                        if len(ps) else dict(K=np.zeros((0, h), np.int64), M=np.zeros((0, h), np.int64),
                                             P=np.zeros((0, h - 1), np.int64), n_p=0)
                    if extra is not None:
                        r.update(extra)
                    t2 = time.perf_counter(); tm["post"] += t2 - t
                    coord.put_result(name, job["id"], u["path"], r)
                    tm["rpc_put"] += time.perf_counter() - t2
                except Exception:
                    coord.put_error(name, job["id"], u["path"], traceback.format_exc(limit=3)[-900:])

    tf = threading.Thread(target=fetch, daemon=True)
    tp = threading.Thread(target=post, daemon=True)
    tf.start(); tp.start()
    while True:
        t = time.perf_counter()
        batch = q_in.get()
        tm["wait_in"] += time.perf_counter() - t
        if batch is STOP:
            q_out.put(STOP)
            break
        # one device call for the whole batch (one "geometry" per job and
        # dilation: exhaustive units have a dilation per p-vector)
        geo_idx, geoms, ranges = {}, [], []
        n_all = sum(len(ps) for _, ps, _, _ in batch)
        MH = gpu.max_h11()
        ps_all = np.zeros((n_all, MH), np.int64)          # p with its leading 0, padded
        pgeo = np.zeros(n_all, np.int32)
        lo = 0
        for u, ps, dil, _ in batch:
            job = u["job"]
            Ds, inv = np.unique(np.full(len(ps), job["D"]) if dil is None else dil, return_inverse=True)
            gi = np.zeros(len(Ds), np.int32)
            for k, D in enumerate(Ds):
                gk = (job["id"], float(D))
                if gk not in geo_idx:
                    d = job["data"]
                    geo_idx[gk] = len(geoms)
                    geoms.append(dict(kappa=d.kappa_cob, Mbasis=d.M_lattice(), Q=_job_Q(job),
                                      dilation=float(D), M0min=job["M0min"]))
                gi[k] = geo_idx[gk]
            hi = lo + len(ps)
            ps_all[lo:hi, 1:1 + ps.shape[1]] = ps
            pgeo[lo:hi] = gi[inv] if len(ps) else 0
            ranges.append((lo, hi))
            lo = hi
        res, err = [], None
        try:
            t = time.perf_counter()
            M, Kn, q, pidx, pstat, secs = gpu.coni_batch_multi(geoms, ps_all, pgeo, device=device)
            starts = np.searchsorted(pidx, [r[0] for r in ranges] + [len(ps_all)])
            for bi, (lo, hi) in enumerate(ranges):
                a, b = starts[bi], starts[bi + 1]
                res.append((M[a:b], Kn[a:b], q[a:b], pidx[a:b], pstat, lo, hi))
            tm["call"] += time.perf_counter() - t
            if verbose:
                print(f"[{name}] {len(batch)} units, {len(ps_all):,} p: device {secs:.2f} s, "
                      f"call {time.perf_counter() - t:.2f} s | totals " +
                      " ".join(f"{k} {v:.1f}" for k, v in tm.items()), flush=True)
        except Exception:
            err = traceback.format_exc(limit=3)[-900:]
        q_out.put((batch, res, err))
    tp.join()


def work(address, authkey=None, device="cpu", procs=None, batch_p=None, verbose=False,
         max_unit_p=None, bound_threads=None):
    """
    **Description:**
    Run workers against a coordinator until it has no more work. Blocks.

    **Arguments:**
    - `address` *(str)*: The coordinator's host:port.
    - `authkey` *(str, optional)*: Shared secret (default: PFVS_AUTHKEY).
    - `device` *(str, optional)*: "cpu", "gpu:N" (device N) or "gpu" (every
        device of this machine, one worker each).
    - `procs` *(int, optional)*: CPU worker processes (default: all cores).
    - `batch_p` *(int, optional)*: p-vectors per request (GPU default 2^20,
        CPU default 2^14).
    - `max_unit_p` *(int, optional)*: A unit with more p-vectors than this is
        sent back to be split (GPU default 2^23, CPU default 2^17), so that no
        worker holds a unit for long.
    - `bound_threads` *(int, optional)*: Host threads a GPU worker computes
        exhaustive jobs' dilation bounds with (default: half the cores). The
        bounds are CPU work that feeds the GPU: with CPU workers on the same
        machine, split its cores between the two.
    """
    import multiprocessing as mp
    host = socket.gethostname()
    if device == "cpu":
        n = procs or os.cpu_count()
        bp = batch_p or (1 << 14)
        ctx = mp.get_context("spawn")
        mu = max_unit_p or (1 << 17)
        ps = [ctx.Process(target=_cpu_worker, args=(address, authkey, f"{host}/cpu{i}", bp, mu))
              for i in range(n)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        return
    if device.startswith("gpu"):
        from . import gpu
        _, _, dev = device.partition(":")
        devs = [int(dev)] if dev else list(range(gpu._load().pfg_device_count() if gpu.available() else 0))
        if not devs:
            raise RuntimeError("no GPU: " + gpu.unavailable_reason())
        bp = batch_p or (1 << 20)
        mu = max_unit_p or (1 << 23)
        if len(devs) == 1:
            _gpu_worker(address, authkey, f"{host}/gpu{devs[0]}", devs[0], bp, verbose, mu,
                        bound_threads)
            return
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=_gpu_worker, args=(address, authkey, f"{host}/gpu{d}", d, bp, verbose, mu,
                                                    bound_threads))
              for d in devs]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        return
    raise ValueError(f'device must be "cpu", "gpu" or "gpu:N", got {device!r}')


# command line
# ------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pfvs.distributed",
                                 description="Coni-PFV searches over many machines.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the coordinator")
    s.add_argument("jobs", help="pickle of a list of jobs (make_jobs)")
    s.add_argument("out", help="output directory (resumed if it exists)")
    s.add_argument("--address", default="0.0.0.0:5055")
    s.add_argument("--target-p", type=int, default=1 << 20)
    s.add_argument("--lease", type=float, default=3600.0, help="seconds before a unit is reissued")
    s.add_argument("--lam", type=float, default=1.0, help="initial price of exhaustive routing")
    w = sub.add_parser("work", help="run workers")
    w.add_argument("address", help="coordinator host:port")
    w.add_argument("--device", default="cpu", help='"cpu", "gpu" (all devices) or "gpu:N"')
    w.add_argument("--procs", type=int, default=None, help="CPU worker processes")
    w.add_argument("--batch-p", type=int, default=None)
    w.add_argument("--max-unit-p", type=int, default=None,
                   help="split units larger than this (default: 2^23 GPU, 2^17 CPU)")
    w.add_argument("--bound-threads", type=int, default=None,
                   help="GPU workers: host threads for exhaustive jobs' bounds (default: half the cores)")
    w.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        with open(a.jobs, "rb") as f:
            jobs = pickle.load(f)
        pr = serve(jobs, a.out, a.address, target_p=a.target_p, lease_s=a.lease, lam0=a.lam)
        return 0 if pr["failed"] == 0 else 1
    work(a.address, device=a.device, procs=a.procs, batch_p=a.batch_p, verbose=a.verbose,
         max_unit_p=a.max_unit_p, bound_threads=a.bound_threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
