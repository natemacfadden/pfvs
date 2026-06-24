"""
Benchmark: two implementations of the same ellipsoid (Zp-style) lattice-point
enumeration --

    * the current `conipfv_kernel` (C): a Fincke-Pohst search that prunes on
      every cut during enumeration, and
    * the old "dSv1" `points_in_ellipsoid` (from arXiv:2406.13751): enumerates
      the ellipsoid by materializing a temporary bounding box, filtering it to
      the ellipsoid, then rejection-sampling the cuts.

Both are the ellipsoid/Zp approach and differ only in how they enumerate the
points inside the ellipsoid. This is unrelated to the README's "box-style
algorithm" (which directly enumerates K and M); the bounding box used by
`points_in_ellipsoid` is just an internal enumeration scaffold.

This script is self-contained: the dSv1 method (a verbatim copy of
`points_in_ellipsoid`, driven by a small metric-LLL reduction) needs only this
repo and its dependencies (numpy).

Both methods enumerate integer vectors in the (dilated) ellipsoid

    0 <= x^T Z x <= dilation * Q                      (ellipsoid)

subject to the same two cuts the kernel applies internally:

    linvec . x   >= M0min                             (M0 cut)
    floor(x^T Z x / Q) <= gcd(H @ x)                  (tadpole / gcd cut)

The dSv1 method gets the ellipsoid for free and applies the M0 and gcd cuts by
rejection sampling. The two methods are checked to return identical vector sets.

Usage:
    python benchmarks/benchmark_dSv1_vs_new.py            # full sweep + tables
    python benchmarks/benchmark_dSv1_vs_new.py _run <c|old> <dilation> <reps>
        # internal: one isolated measurement, prints a json line
"""
import os
import sys
import json
import time
import resource
import platform
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# Manwe example (h11 = 7), identical to benchmark_conipfv.py
# ---------------------------------------------------------------------------
U = np.array([
    [19.131126469708992, -2.50900019274872, -12.022292590254283, 9.408750722807701, 10.97687584327565, 8.363333975829066, 24.77637690339361],
    [0.0, 20.8255832579255, 30.243381795036953, 31.864968637769284, 12.462603346669052, 6.193517125542706, 6.826408301055712],
    [0.0, 0.0, 31.73014873072524, -6.636894774358689, 15.791162726449791, 12.771246313546088, -22.268563128240686],
    [0.0, 0.0, 0.0, 10.772688211590271, -2.932867573104269, -3.2752582214981767, -13.592907165071438],
    [0.0, 0.0, 0.0, 0.0, 27.133543485699047, 2.9384531625551986, 1.5013990104854835],
    [0.0, 0.0, 0.0, 0.0, 0.0, 15.786970406543043, -15.837576931579958],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.49372858726784],
], dtype=np.float64)
Q = 162
LINVEC = np.array([0, 0, 0, 4, -4, -2, -2], dtype=np.int32)
LINMIN = 13.0
H = np.array([
    [2, 0, 0, 0, 62, 58974, -5086],
    [0, 2, 0, 0, 84, 224666, -19686],
    [0, 0, 2, 0, 12, 234014, -20736],
    [0, 0, 0, 4, 52, 78376, -6916],
    [0, 0, 0, 0, 120, 161172, -14052],
    [0, 0, 0, 0, 0, 262692, -23292],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=np.int64)

# the ellipsoid metric mat = U.T @ U (the C kernel takes U, the dSv1 path takes Z)
Z = U.T @ U

# realistic output cap (actual counts are tiny). The repo default of 1e8 would
# reserve a multi-gigabyte virtual output buffer that distorts the rss reading.
MAX_N_OUT = 1_000_000

# dSv1 memory guard: predicted candidate-box bytes above this are skipped
BUDGET_GB = 6.0


# ---------------------------------------------------------------------------
# self-contained metric-LLL: reduce the identity basis of Z^n with respect to
# the inner product <u,v> = u^T Z v. This is the same reduction the dSv1
# `LLL_reduction_wrt_metric` performs, reimplemented here (numpy only) so the
# benchmark needs no external research tree. Returns a unimodular integer basis
# B (rows are reduced vectors) such that B @ Z @ B.T is nearly diagonal.
# ---------------------------------------------------------------------------
def lll_reduce_wrt_metric(Z, delta=0.75):
    n = len(Z)
    B = np.eye(n, dtype=np.int64)

    def gram_schmidt(B):
        Bstar = np.zeros((n, n))
        mu = np.zeros((n, n))
        for i in range(n):
            Bstar[i] = B[i].astype(float)
            for j in range(i):
                denom = Bstar[j] @ Z @ Bstar[j]
                mu[i, j] = (B[i] @ Z @ Bstar[j]) / denom
                Bstar[i] = Bstar[i] - mu[i, j] * Bstar[j]
        return Bstar, mu

    Bstar, mu = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                B[k] = B[k] - round(mu[k, j]) * B[j]
                Bstar, mu = gram_schmidt(B)
        lhs = Bstar[k] @ Z @ Bstar[k]
        rhs = (delta - mu[k, k - 1] ** 2) * (Bstar[k - 1] @ Z @ Bstar[k - 1])
        if lhs >= rhs:
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            Bstar, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


# ---------------------------------------------------------------------------
# dSv1 kernel: `points_in_ellipsoid` (verbatim from PFV_search.py of
# arXiv:2406.13751), driven by the metric-LLL above. maximum_box_size is left at
# inf so the box is not truncated (its original default of 1e3 makes it silently
# inexact by shrinking the search box).
# ---------------------------------------------------------------------------
def points_in_ellipsoid(Zin, Qbound, fluxbound=2, maximum_box_size=np.inf):
    dimension = len(Zin)
    if min(np.linalg.eig(Zin)[0]) < 1e-3:
        return np.array([np.arange(len(Zin))])[[]]
    new_basis = lll_reduce_wrt_metric(Zin)
    Zp = new_basis @ Zin @ new_basis.T
    bounds = np.rint(fluxbound * np.sqrt(Qbound / np.diagonal(Zp))).astype(int)
    box_vol = np.prod(2 * bounds)
    if box_vol > maximum_box_size:
        bounds = np.rint(bounds * (maximum_box_size / box_vol) ** (1 / dimension)).astype(int)
    candidates = np.indices(bounds * 2 + 1).reshape(len(bounds), -1).T - bounds
    solsp = candidates[np.sum(candidates * (candidates @ Zp.T), axis=1) <= Qbound]
    return solsp @ new_basis


def apply_cuts(vecs, dilation):
    """Reject candidates failing the kernel's M0 and gcd cuts.

    The ellipsoid itself is already enforced by `points_in_ellipsoid`.
    """
    if len(vecs) == 0:
        return vecs
    vecs = np.atleast_2d(vecs).astype(np.int64)
    qform = np.einsum("ij,jk,ik->i", vecs, Z, vecs)
    keep = (vecs @ LINVEC.astype(np.int64)) >= LINMIN          # M0 cut
    Hx = vecs @ H.T
    gcds = np.array([np.gcd.reduce(np.abs(Hx[i])) for i in range(len(vecs))])
    keep &= np.floor(qform / Q).astype(np.int64) <= gcds       # tadpole gcd cut
    return vecs[keep]


def predict_box_gb(dilation):
    new_basis = lll_reduce_wrt_metric(Z)
    Zp = new_basis @ Z @ new_basis.T
    bounds = np.rint(2 * np.sqrt(Q * dilation / np.diagonal(Zp))).astype(int)
    elems = float(np.prod(2.0 * bounds + 1))
    return elems * len(Z) * 8 / 1e9


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def run_c(dilation):
    from pfvs.conipfv_kernel import conipfv_kernel
    out, _, _ = conipfv_kernel(U, Q, dilation, LINVEC, LINMIN, H, MAX_N_OUT)
    return out

def run_old(dilation):
    return apply_cuts(points_in_ellipsoid(Z, Q * dilation), dilation)


# ---------------------------------------------------------------------------
# one isolated measurement (subprocess): time (min of warm runs) + peak rss
# ---------------------------------------------------------------------------
_RSS_SCALE = 1.0 if platform.system() == "Darwin" else 1024.0  # bytes vs kB
def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_SCALE / 1e6

def _measure(method, dilation, repeats):
    result = {"method": method, "dilation": dilation}
    runner = {"c": run_c, "old": run_old}[method]

    if method == "old":
        gb = predict_box_gb(dilation)
        result["pred_box_gb"] = round(gb, 3)
        if gb > BUDGET_GB:
            result["status"] = "skipped_oom"
            return result
    else:
        # warm up the new kernels so the one-time jit-compile / extension-init
        # cost lands in the baseline, not in the per-call memory delta
        runner(dilation)

    base = _peak_rss_mb()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = runner(dilation)
        times.append(time.perf_counter() - t0)
    peak = _peak_rss_mb()

    result.update({
        "status": "ok",
        "n_found": int(len(out)),
        "t_min_s": min(times),
        "rss_delta_mb": round(peak - base, 1),
    })
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
DILATIONS = [1, 2, 5, 10, 15, 20]

def _spawn(method, dilation, repeats):
    p = subprocess.run([sys.executable, os.path.abspath(__file__),
                        "_run", method, str(dilation), str(repeats)],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.strip().startswith("{"):
            return json.loads(line.strip())
    return {"method": method, "dilation": dilation, "status": "error",
            "stderr": "\n".join(p.stderr.splitlines()[-3:])}

def _check_identical_output():
    """Confirm dSv1 and the C kernel return the same vector set (feasible dil)."""
    from pfvs.conipfv_kernel import conipfv_kernel
    dil = 20
    out_c, _, _ = conipfv_kernel(U, Q, dil, LINVEC, LINMIN, H, MAX_N_OUT)
    out_old = run_old(dil)
    sc = set(map(tuple, np.atleast_2d(out_c).astype(np.int64).tolist())) if len(out_c) else set()
    so = set(map(tuple, np.atleast_2d(out_old).astype(np.int64).tolist())) if len(out_old) else set()
    return sc == so, len(sc)

def main():
    ok, n = _check_identical_output()
    print(f"output-equality check (dSv1 vs C kernel, dilation=20): "
          f"{'identical' if ok else 'MISMATCH'} ({n} vectors)\n")

    results = {}
    print("Running sweep (each measurement in an isolated process)...\n")
    for dil in DILATIONS:
        for m in ("c", "old"):
            r = _spawn(m, dil, 3 if dil <= 10 else 2)
            results[(m, dil)] = r
            st = r.get("status")
            if st == "ok":
                print(f"  {m:>4} dil={dil:<5} n={r['n_found']} "
                      f"t={r['t_min_s']*1e3:.3f}ms  mem~{r['rss_delta_mb']}MB")
            elif st == "skipped_oom":
                print(f"  {m:>4} dil={dil:<5} skipped (predicted box {r['pred_box_gb']} GB)")
            else:
                print(f"  {m:>4} dil={dil:<5} error {r.get('stderr','')}")

    def t(m, d):
        r = results.get((m, d), {})
        return f"{r['t_min_s']*1e3:.3f}" if r.get("status") == "ok" else (
            "oom" if r.get("status") == "skipped_oom" else "n/a")
    def mem(m, d):
        r = results.get((m, d), {})
        return f"{r['rss_delta_mb']:.1f}" if r.get("status") == "ok" else (
            "oom" if r.get("status") == "skipped_oom" else "n/a")
    def nfound(d):
        for m in ("c", "old"):
            r = results.get((m, d), {})
            if r.get("status") == "ok":
                return r["n_found"]
        return "?"
    def speedup(m, d):
        ro, rn = results.get(("old", d), {}), results.get((m, d), {})
        if ro.get("status") == "ok" and rn.get("status") == "ok" and rn["t_min_s"] > 0:
            return f"{ro['t_min_s'] / rn['t_min_s']:,.0f}x"
        return "oom" if ro.get("status") == "skipped_oom" else "n/a"

    print("\n=== wall time (ms, min of warm runs) ===")
    print(f"{'dilation':>9} {'C':>9} {'dSv1':>9} {'n_found':>9}")
    for d in DILATIONS:
        print(f"{d:>9} {t('c',d):>9} {t('old',d):>9} {str(nfound(d)):>9}")

    print("\n=== peak working memory (MB, rss over baseline) ===")
    print(f"{'dilation':>9} {'C':>9} {'dSv1':>9}")
    for d in DILATIONS:
        print(f"{d:>9} {mem('c',d):>9} {mem('old',d):>9}")

    def mem_reduction(d):
        # both methods store the same output; the new kernel adds only an O(h11)
        # recursion stack, while dSv1 additionally materializes the whole box.
        # here the output is tiny (<=2 vectors), so the new kernel's footprint is
        # below measurement resolution -- report a conservative lower bound
        # rather than dividing by ~0.
        ro, rc = results.get(("old", d), {}), results.get(("c", d), {})
        if ro.get("status") != "ok":
            return "oom" if ro.get("status") == "skipped_oom" else "n/a"
        old_mb, new_mb = ro["rss_delta_mb"], rc.get("rss_delta_mb", 0.0)
        if new_mb >= 1.0:
            return f"{old_mb / new_mb:,.0f}x"
        return f">{old_mb:,.0f}x" if old_mb >= 1 else "~1x"

    print("\n=== memory reduction of new kernel vs dSv1 (dSv1_mem / new_mem) ===")
    print(f"{'dilation':>9} {'dSv1 mem':>12} {'reduction':>12}")
    for d in DILATIONS:
        print(f"{d:>9} {mem('old',d) + ' MB':>12} {mem_reduction(d):>12}")
    print("(both methods store the output; the new kernel adds only an O(h11) "
          "stack, while dSv1 also builds the whole box ~(Q*d)^(dim/2). With only "
          "a handful of outputs here the new kernel stays sub-MB. Reductions are "
          "conservative lower bounds -- the box overhead is what is eliminated.)")

    print("\n=== speedup of new kernel over dSv1 (dSv1_time / new_time) ===")
    print(f"{'dilation':>9} {'C':>12}")
    for d in DILATIONS:
        print(f"{d:>9} {speedup('c',d):>12}")
    print("(speedup grows ~d^3.5: the dSv1 box scales as (Q*d)^(dim/2), while the "
          "kernel is roughly flat over this range.)")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "_run":
        print(json.dumps(_measure(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))))
    else:
        main()
