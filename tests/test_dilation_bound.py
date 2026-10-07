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
# Description:  The Schur-complement bound delta < Q/mu0 on a coni PFV's
#               dilation (pfvs.dilation), checked against PFVs from real data
#               and against searches run past the bound.
# -----------------------------------------------------------------------------

import gzip
import itertools
import math
import json
import os
import warnings
from fractions import Fraction
from pathlib import Path

import flint
import numpy as np
import pytest

from pfvs import CYData, coniZpM
from pfvs.dilation import (bound_routing, coni_dilation_bound, coni_dilation_bound_ceils,
                           coni_dilation_bounds, coni_mu0)

DATA = Path(__file__).parent / "data"
with gzip.open(DATA / "fixtures.json.gz") as _f:
    GEOMS = json.load(_f)["geometries"]
with open(DATA / "paper_examples.json") as _f:
    CONI_EXAMPLES = [e for e in json.load(_f)["examples"] if e["kind"] == "coni"]


def _dense(kappa_coo, h11):
    kappa = np.zeros((h11,) * 3, dtype=np.int64)
    for i, j, k, v in kappa_coo:
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            kappa[a, b, c] = v
    return kappa


def _geom_data(g):
    """CYData for a dataset geometry (stored in the coni basis)."""
    h11 = g["h11"]
    Hc = np.array(g["H"], dtype=np.int64)
    H = np.hstack([np.zeros((len(Hc), 1), dtype=np.int64), Hc])
    e = np.eye(h11, dtype=np.int64)
    return CYData(g["h21"], _dense(g["kappa_coo"], h11), g["c2"], H, coni_curve=e[0], coni_cob=e)


def _direction_and_delta(kappa, K, M):
    """(p_hat, delta) of a coni PFV from its fluxes alone: K_r = N_rr p_r with
    N = kappa . M, p = p_hat / delta, p_hat primitive with p_hat_0 = 0."""
    N = np.einsum("abc,c->ab", np.asarray(kappa, dtype=object), np.asarray(M, dtype=object))
    Nrr = flint.fmpq_mat([[int(x) for x in row[1:]] for row in N[1:]])
    Kr = flint.fmpq_mat([[int(x)] for x in K[1:]])
    pr = [Fraction(int(x.p), int(x.q)) for x in (Nrr.inv() * Kr).entries()]
    den = 1
    for x in pr:
        den = den * x.denominator // np.gcd(den, x.denominator)
    ints = [int(x * den) for x in pr]
    g = int(np.gcd.reduce(np.abs(ints)))
    p_hat = np.array([0] + [x // g for x in ints], dtype=np.int64)
    return p_hat, Fraction(den, g)        # p_r = p_hat_r / delta


def _pfvs_with_direction():
    """Every coni PFV in the fixtures and paper examples, as
    (kappa (coni basis), Q, p_hat, delta)."""
    out = []
    for g in GEOMS:
        kappa = _dense(g["kappa_coo"], g["h11"])
        Q = g["h11"] + g["h21"] + 4
        for K, M in zip(g["K"], g["M"]):
            out.append((kappa, Q, *_direction_and_delta(kappa, K, M)))
    for e in CONI_EXAMPLES:
        B = np.asarray(e["cob"], dtype=np.int64)
        kappa = np.einsum("ai,bj,ck,ijk->abc", B, B, B, np.asarray(e["kappa"], dtype=np.int64))
        Q = -int(np.dot(e["K"], e["M"]))
        out.append((kappa, Q, *_direction_and_delta(kappa, e["K"], e["M"])))
    return out


PFVS = _pfvs_with_direction()


def test_fixture_coverage():
    """Hundreds of PFVs across h11 = 3..9 (the fixtures' h11 = 10, 11
    geometries have none), including the papers' examples. The full dataset
    (h11 up to 11) is checked by test_bound_holds_on_full_dataset."""
    assert len(PFVS) > 300
    assert {k.shape[0] for k, *_ in PFVS} >= set(range(3, 10))


def test_bound_holds_for_every_known_pfv():
    """delta < Q/mu0(p_hat) for every PFV in the fixtures and the papers, and
    the bound's hypotheses (s <= 0, S positive definite on Lambda) hold."""
    for kappa, Q, p_hat, delta in PFVS:
        b = coni_dilation_bound(p_hat[1:], kappa, Q)
        assert b is not None, f"hypotheses fail for p_hat={p_hat.tolist()}"
        assert isinstance(b, Fraction)
        assert delta < b, f"delta={delta} >= bound={b} for p_hat={p_hat.tolist()}"


def test_mu0_is_attained_on_lambda():
    """mu0 is the form's value at a nonzero lattice vector orthogonal to p_hat."""
    for kappa, _Q, p_hat, _ in PFVS[::7]:
        mu0, K = coni_mu0(p_hat[1:], kappa)
        K = np.asarray(K, dtype=object)
        assert any(K) and int(np.dot(K, p_hat[1:])) == 0
        A = np.einsum("abc,c->ab", np.asarray(kappa, dtype=object), p_hat.astype(object))
        Arr = flint.fmpq_mat([[int(x) for x in row[1:]] for row in A[1:]])
        v = flint.fmpq_mat([[int(x)] for x in K])
        val = -(v.transpose() * Arr.inv() * v)[0, 0]
        assert Fraction(int(val.p), int(val.q)) == mu0


def _brute_mu0(p_hat_r, kappa, box):
    """min of K^T S K over nonzero K in Z^(h11-1), |K_i| <= box, p_hat.K = 0."""
    p_hat = np.concatenate([[0], p_hat_r]).astype(object)
    A = np.einsum("abc,c->ab", np.asarray(kappa, dtype=object), p_hat)
    Sinv = flint.fmpq_mat([[int(x) for x in row[1:]] for row in A[1:]]).inv()
    best = None
    for K in itertools.product(range(-box, box + 1), repeat=len(p_hat_r)):
        if not any(K) or np.dot(K, p_hat_r) != 0:
            continue
        v = flint.fmpq_mat([[k] for k in K])
        val = -(v.transpose() * Sinv * v)[0, 0]
        val = Fraction(int(val.p), int(val.q))
        best = val if best is None or val < best else best
    return best


@pytest.mark.parametrize("h11", [3, 4, 5])
def test_mu0_matches_brute_force_at_low_h11(h11):
    """At low h11 a box search finds the true minimum over Lambda: the
    lattice minimum must equal it (and can never exceed it)."""
    cases = [c for c in PFVS if c[0].shape[0] == h11][:6]
    assert cases
    for kappa, _Q, p_hat, _ in cases:
        mu0, _ = coni_mu0(p_hat[1:], kappa)
        brute = _brute_mu0(p_hat[1:], kappa, box=6)
        assert mu0 <= brute
        assert mu0 == brute


def test_unimodular_invariance():
    """mu0 does not depend on the basis of the non-coni directions: a signed
    permutation of directions 1..h11-1 applied to kappa and p_hat leaves it
    unchanged."""
    rng = np.random.default_rng(0)
    for kappa, _Q, p_hat, _ in PFVS[::25]:
        h = kappa.shape[0]
        perm = rng.permutation(h - 1)
        signs = rng.choice([-1, 1], size=h - 1)
        U = np.eye(h, dtype=np.int64)
        U[1:, 1:] = 0
        U[1 + np.arange(h - 1), 1 + perm] = signs          # new coords x' = U x
        Uinv = np.round(np.linalg.inv(U)).astype(np.int64)
        kappa2 = np.einsum("ia,jb,kc,ijk->abc", Uinv, Uinv, Uinv, kappa)   # kappa(x) invariant
        p2 = U @ p_hat
        assert coni_mu0(p2[1:], kappa2)[0] == coni_mu0(p_hat[1:], kappa)[0]


def test_hypothesis_failures_return_none():
    """Where the argument does not apply the function says so (None) rather
    than returning a number: a singular A_rr, and a form that is not positive
    definite on Lambda."""
    h = 4
    assert coni_dilation_bound([1, 2, 3], np.zeros((h, h, h), dtype=np.int64), 100) is None
    # kappa with A_rr = diag(1, 1, 1) for p_hat = (0, 1, 0, 0): S = -I is
    # negative definite on Lambda
    kappa = np.zeros((h, h, h), dtype=np.int64)
    for i in (1, 2, 3):
        for a, b, c in {(i, i, 1), (i, 1, i), (1, i, i)}:
            kappa[a, b, c] = 1
    assert coni_dilation_bound([1, 0, 0], kappa, 100) is None


# which geometries and p-vectors the search tests use: small h11 and boxes so
# a search past the bound stays cheap
def _search_cases():
    cases = []
    for g in GEOMS:
        if g["h11"] > 6 or not g["K"]:
            continue
        kappa = _dense(g["kappa_coo"], g["h11"])
        dirs = {tuple(_direction_and_delta(kappa, K, M)[0][1:]) for K, M in zip(g["K"], g["M"])}
        for p in sorted(dirs)[:2]:
            cases.append((g, np.array(p)))
    return cases[:10]


SEARCH_CASES = _search_cases()


@pytest.mark.parametrize("g,p", SEARCH_CASES,
                         ids=[f"h11={g['h11']}-poly{g['polyID']}-p{p.tolist()}" for g, p in SEARCH_CASES])
def test_search_past_the_bound_finds_nothing_more(g, p):
    """For a direction p_hat, searching at dilation Q/mu0 already finds every
    PFV: a search at a much larger dilation finds the same set, and every PFV
    in it has delta < Q/mu0."""
    data = _geom_data(g)
    kappa, Q = data.kappa_cob, g["h11"] + g["h21"] + 4
    b = coni_dilation_bound(p, kappa, Q)
    assert b is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = coniZpM(data, np.array([p]), Q=Q, ellipsoid_dilation=float(b), n_jobs=1, device="cpu")
        big = coniZpM(data, np.array([p]), Q=Q, ellipsoid_dilation=float(2 * b + 20), n_jobs=1,
                      device="cpu")
    def as_set(KM):
        return {(tuple(map(int, k)), tuple(map(int, m))) for k, m in zip(*KM)}

    assert as_set(at) == as_set(big)
    for K, M in zip(*big):
        p_hat, delta = _direction_and_delta(kappa, K, M)
        assert np.array_equal(p_hat[1:], p) and delta < b


FULL = os.environ.get("PFVS_CONI_DATASET")     # path to coni_pfvs.parquet


@pytest.mark.skipif(not FULL, reason="set PFVS_CONI_DATASET to the dataset's coni_pfvs.parquet")
def test_bound_holds_on_full_dataset():
    """Every PFV of the published coni dataset (h11 = 3..11) obeys the bound,
    and the hypotheses hold for every direction."""
    cols = ["h11", "h21", "kappa_coo", "K", "M"]
    try:
        import pyarrow.parquet as pq
        rows = pq.read_table(FULL, columns=cols).to_pylist()
    except ImportError:
        duckdb = pytest.importorskip("duckdb", reason="reading the parquet needs pyarrow or duckdb")
        cur = duckdb.connect().execute(f"select {', '.join(cols)} from '{FULL}'")
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    n = 0
    for r in rows:
        if not r["K"]:
            continue
        kappa = _dense([list(x) for x in r["kappa_coo"]], int(r["h11"]))
        Q = int(r["h11"]) + int(r["h21"]) + 4
        for K, M in zip(r["K"], r["M"]):
            p_hat, delta = _direction_and_delta(kappa, list(K), list(M))
            b = coni_dilation_bound(p_hat[1:], kappa, Q)
            assert b is not None and delta < b, (int(r["h11"]), p_hat.tolist(), delta, b)
            n += 1
    assert n > 30000


def test_pfv_dilation_bound_diagnostic():
    """PFV.dilation_bound (coni PFVs only) is the bound for the PFV's own
    direction and tadpole, and lies above its dilation."""
    from pfvs import PFV
    with open(DATA / "paper_examples.json") as f:
        examples = json.load(f)["examples"]
    for e in examples:
        if e["kind"] == "coni":
            data = CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"],
                          coni_curve=e["coni_curve"], coni_cob=e["cob"])
            pfv = PFV(data, K=e["K"], M=e["M"])
            b = pfv.dilation_bound
            p_hat, delta = _direction_and_delta(data.kappa_cob, e["K"], e["M"])
            assert b == coni_dilation_bound(p_hat[1:], data.kappa_cob, -int(np.dot(e["K"], e["M"])))
            assert delta < b
        else:
            data = CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"])
            assert not hasattr(PFV(data, K=e["K"], M=e["M"]), "dilation_bound")


def test_c_matches_python():
    """The C mu0 equals the exact python-flint one (or both give no bound) on
    every PFV direction and on arbitrary directions (which often fail the
    hypotheses); its minimizer lies in Lambda and attains mu0."""
    from pfvs.dilation import _coni_mu0_c
    rng = np.random.default_rng(3)
    decided = 0
    for kappa, _Q, p_hat, _ in PFVS[::3]:
        h = kappa.shape[0]
        ps = np.vstack([p_hat[1:], rng.integers(-5, 6, size=(8, h - 1))])
        ps = ps[np.any(ps != 0, axis=1)]
        st, num, den, K = _coni_mu0_c(ps, kappa)
        for i, p in enumerate(ps):
            ref = coni_mu0(p, kappa, use_c=False)
            if st[i] < 0:
                continue                                 # undecided in C: Python path
            decided += 1
            if st[i] == 1:
                assert ref is None
                continue
            assert ref is not None and ref[0] == Fraction(int(num[i]), int(den[i]))
            assert any(K[i]) and int(np.dot(K[i].astype(object), p.astype(object))) == 0
    assert decided > 100


@pytest.mark.parametrize("shared", [False, True])
def test_bound_routing_is_optimal(shared):
    """The threshold rule's run time equals the best over every assignment of
    the p-vectors (ZpM-to-bound or split) and every candidate D0; p-vectors
    without a bound are left out."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        a, kc = rng.uniform(0.5, 1.8), rng.uniform(0.1, 20)
        def G(D, a=a):
            return (np.asarray(D, dtype=float) / 100) ** a
        def K(D0, kc=kc):
            return kc * 800 / D0
        b = rng.uniform(200, 3000, size=9)
        b[rng.random(9) < 0.2] = np.inf
        D0s, c = [400, 800, 1600], rng.uniform(0, 0.5)
        b_star, D0, t = bound_routing(b, G, K, D0s, c, shared=shared)
        bf = b[np.isfinite(b)]

        def run_time(x, D, bf=bf, G=G, K=K, c=c):
            t_gpu = np.where(x, G(np.where(x, bf, D)), G(D)).sum()
            t_cpu = (~x).sum() * K(D) + len(bf) * c
            return ((t_gpu + t_cpu) if shared else max(t_gpu, t_cpu)) / max(len(bf), 1)
        best = min(run_time(np.array(x, dtype=bool), D)
                   for D in D0s for x in itertools.product([0, 1], repeat=len(bf)))
        assert t == pytest.approx(best)
        assert run_time(bf <= b_star, D0) == pytest.approx(t)    # the rule's own assignment


def test_bound_ceils():
    """coni_dilation_bound_ceils is ceil of coni_dilation_bounds (0 for none),
    threaded or not, including directions without a bound."""
    rng = np.random.default_rng(4)
    for kappa, Q, p_hat, _ in PFVS[::5]:
        h = kappa.shape[0]
        ps = np.vstack([p_hat[1:], rng.integers(-5, 6, size=(12, h - 1))])
        ps = ps[np.any(ps != 0, axis=1)]
        want = [math.ceil(b) if b is not None else 0 for b in coni_dilation_bounds(ps, kappa, Q)]
        for n_jobs in (1, 3):
            assert coni_dilation_bound_ceils(ps, kappa, Q, n_jobs=n_jobs).tolist() == want

