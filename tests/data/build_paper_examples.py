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
# Description:  Builds tests/data/paper_examples.json: the named PFV examples of
#   arXiv:2406.13751 (coni; data from github.com/AndreasSchachner/
#   kklt_de_sitter_vacua) and arXiv:2107.09064 (non-coni; data printed in its
#   Sec. 6), as pfvs inputs. Two stages, as they need different environments:
#
#     python build_paper_examples.py geometry REPO_DIR geometry.json   # CYTools + pandas
#     python build_paper_examples.py finalize geometry.json paper_examples.json   # pfvs
#
#   Bases. Neither paper's basis can be taken from CYTools' defaults: the
#   default divisor basis has changed between CYTools versions (for Examples 4
#   and C.3 i/j the current default is not the paper's), and 2107.09064 states
#   its basis only for Sec. 6.1. So every basis is *recovered*: among the
#   h11-subsets of prime toric divisors, the one in which the paper's fluxes
#   satisfy the PFV conditions exactly (N(M) p = K, and K.p = 0 or the coni
#   condition), with -- for coni -- the conifold curve of GV invariant 2 in
#   the repository's own coni basis transformation. Each coni example has a
#   unique such basis; the non-coni triangulation is also searched (all FRSTs),
#   and Sec. 6.1's recovered kappa and c2 equal the printed ones. The basis
#   found is stored, and the fixture's kappa/c2/H are in it: the tests never
#   depend on CYTools.
#
#   Cones. H is CYTools' mori_cone_cap when the paper's p lies inside its
#   dual; for Secs. 6.1 and 6.2 it does not -- one generator of the cap there
#   has GV invariant 0 (not an effective curve) and q.p < 0 -- so H is the
#   cone of the curves with nonzero GV invariants (degree <= 6), inside whose
#   dual every example's p lies (q.p > 0; = 0 only for the conifold curve).
#
#   Paper typos resolved by the data: Example 3 has M = (..., -2, -10, 13, 5)
#   and p_8 = +8/44 (the paper prints -8; only +8 satisfies N p = K).
# -----------------------------------------------------------------------------

import itertools
import json
import math
import sys
from fractions import Fraction as F

import numpy as np

# 2107.09064, Sec. 6: vertices of Delta (columns), fluxes, p (as printed)
NONCONI = {
    "6.1": dict(V=[[1, -3, -3, 0, 0, 0, -5, -2], [0, -2, -1, 0, 0, 1, -3, -1], [0, 0, -1, 0, 1, 0, 0, 1],
                   [0, 0, 0, 1, 0, 0, -1, -1]],
                M=[0, 2, 4, 11, -8], K=[8, -15, 11, -2, 13],
                p=[F(7, 58), F(15, 58), F(101, 116), F(151, 58), F(-13, 116)], h21=113),
    "6.2": dict(V=[[1, 1, -2, -2, 0, -2, 0, 0], [0, 0, -1, -1, 1, 1, 0, 0], [0, 1, -1, 1, 0, -1, 0, 1],
                   [0, 1, 1, -1, 0, -1, 1, 0]],
                M=[4, 4, 0, -3, 2, 0, -2], K=[-4, -4, -3, 2, -3, 3, 3],
                p=[F(13, 6), F(1, 3), F(-2, 3), F(1), F(7, 10), F(8, 5), F(11, 10)], h21=51),
    "6.3": dict(V=[[1, -2, -2, -2, -2, 0, 0, 0, 0], [0, -1, -1, 0, 0, 0, 0, 1, 1], [0, -1, 0, -1, 0, 0, 1, 0, 1],
                   [0, 1, 0, 0, -1, 1, 0, 0, -1]],
                M=[3, -5, 2, -2, -5], K=[-5, 5, -4, -1, 5],
                p=[F(13, 8), F(59, 24), F(5, 4), F(5, 4), F(5, 12)], h21=81),
    "6.4": dict(V=[[-1, 1, -1, -1, -1, -1, -1], [2, -1, -1, -1, -1, -1, -1], [-1, 0, 1, 1, 2, 2, 3],
                   [-1, 0, 1, 2, 1, 3, 2]],
                M=[10, 11, -11, -5], K=[-3, -5, 8, 6],
                p=[F(293, 110), F(163, 110), F(163, 110), F(13, 22)], h21=214),
}
# Sec. 6.1 prints kappa in the basis of the prime toric divisors of its first
# five vertices (entries a <= b <= c) and c2: a check of the recovery
KAPPA_61 = {(1, 1, 1): 89, (1, 1, 3): 16, (1, 1, 4): 12, (1, 1, 5): 7, (1, 3, 4): 3, (1, 4, 5): 3,
            (1, 5, 5): -3, (2, 2, 2): 8, (2, 2, 3): -2, (2, 2, 4): -2, (2, 2, 5): -2, (2, 3, 4): 1,
            (2, 4, 5): 1, (5, 5, 5): -1}
C2_61 = [146, -4, 24, 24, 14]
GV_DEG = 6


def _lcm(xs):
    r = 1
    for x in xs:
        r = r * x // math.gcd(r, x)
    return r


def _load_repo(repo):
    """The repository's data frames, unpickled allowing only numpy/pandas
    containers (no code runs)."""
    import gzip
    import importlib
    import pickle
    allowed = {("builtins", "slice"), ("builtins", "complex"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.numeric", "_frombuffer"),
               ("numpy", "ndarray"), ("pandas._libs.internals", "_unpickle_block"),
               ("pandas.core.frame", "DataFrame"), ("pandas.core.indexes.base", "_new_Index"),
               ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.range", "RangeIndex"),
               ("pandas.core.internals.managers", "BlockManager")}
    remap = {"numpy.core.multiarray": "numpy._core.multiarray", "numpy.core.numeric": "numpy._core.numeric"}

    class Safe(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f"refusing {module}.{name}")
            try:
                return getattr(importlib.import_module(module), name)
            except (ImportError, AttributeError):
                return getattr(importlib.import_module(remap.get(module, module)), name)

    rows = []
    for f, kind in [("data/dS_examples/data.p", "dS"), ("data/dS_examples/data_extra_dS.p", "extra dS"),
                    ("data/nonSUSY_AdS_examples/data.p", "non-SUSY AdS")]:
        with gzip.open(f"{repo}/{f}") as fh:
            for _, r in Safe(fh).load().iterrows():
                P = [F(float(x)).limit_denominator(1000) for x in np.asarray(r["P vector"], dtype=float)]
                den = _lcm(q.denominator for q in P)
                rows.append(dict(kind=kind, name=r["paper name"], section=r["paper section"],
                                 model=str(r["name"]) if "name" in r and r["name"] == r["name"] else None,
                                 dual_points=np.asarray(r["dual points"]).astype(int).tolist(),
                                 mirror_heights=np.asarray(r["mirror heights"], dtype=float).tolist(),
                                 coni_curve=np.asarray(r["conifold curve"]).astype(int).tolist(),
                                 cob=np.asarray(r["basis transformation"]).astype(int).tolist(),
                                 M=np.asarray(r["M vector"]).astype(int).tolist(),
                                 K=np.asarray(r["K vector"]).astype(int).tolist(),
                                 p=[int(q * den) for q in P], p_den=den))
    return rows


def _gvs(cy):
    gv = cy.compute_gvs(max_deg=GV_DEG)
    return [(np.array(q), int(n)) for q, n in gv.dok.items()]


def _cones(cy, gvs):
    from cytools.cone import Cone
    cap = np.asarray(cy.mori_cone_cap(in_basis=True).rays()).astype(int)
    nz = np.array([q for q, n in gvs if n != 0])
    eff = np.asarray(Cone(rays=nz).extremal_rays()).astype(int)
    return cap, eff, nz


def geometry(repo, out):
    from cytools import Polytope
    res = []
    # coni examples: basis recovered per example; examples with an identical
    # flux problem (the extra dS vacua) are recorded as aliases
    seen = {}
    for r in _load_repo(repo):
        tag = f"{r['kind']}/{r['name']}({r['section']})"
        tri = Polytope(r["dual_points"]).triangulate(heights=np.array(r["mirror_heights"]))
        cy = tri.get_cy()
        h = cy.h11()
        B, M, K, p = (np.array(r[x]) for x in ("cob", "M", "K", "p"))
        kall = np.asarray(cy.intersection_numbers(in_basis=False, format="dense")).astype(np.int64)
        hits = []
        for S in itertools.combinations(range(1, kall.shape[0]), h):
            kc = np.einsum("ai,bj,ck,ijk->abc", B, B, B, kall[np.ix_(S, S, S)])
            if np.array_equal((np.einsum("abc,c->ab", kc, M) @ p)[1:], r["p_den"] * K[1:]):
                hits.append(list(S))
        good = []
        for S in hits:
            c = tri.get_cy()
            try:
                c.set_divisor_basis(S)
            except Exception:
                continue
            gvs = _gvs(c)
            cc = np.linalg.solve(B, np.eye(h)[0]).round().astype(int)
            if dict((tuple(q), n) for q, n in gvs).get(tuple(cc)) == 2 and cc.tolist() == r["coni_curve"]:
                good.append((S, c, gvs))
        assert len(good) == 1, f"{tag}: {len(good)} bases fit the fluxes and conifold"
        S, c, gvs = good[0]
        key = json.dumps([S, r["dual_points"], r["mirror_heights"], r["M"], r["K"], r["p"]])
        if key in seen:
            seen[key]["aliases"].append(tag)
            continue
        cap, eff, nz = _cones(c, gvs)
        qp = nz @ B.T @ p
        assert (qp >= 0).all() and (qp == 0).sum() == 1, tag            # only the conifold curve
        Hc = cap @ B.T
        in_cap = bool(np.all((Hc[:, 1:] @ p[1:] > 0) | np.all(Hc[:, 1:] == 0, axis=1)))
        rec = dict(paper="arXiv:2406.13751", kind="coni", section=r["section"], name=r["name"], model=r["model"],
                   aliases=[], h11=h, h21=int(c.h21()),
                   kappa=np.asarray(c.intersection_numbers(in_basis=True, format="dense")).astype(int).tolist(),
                   c2=np.asarray(c.second_chern_class(in_basis=True)).astype(int).tolist(),
                   H=(cap if in_cap else eff).tolist(),
                   H_kind="mori_cone_cap" if in_cap else f"effective curves (GV invariants of degree <= {GV_DEG})",
                   coni_curve=r["coni_curve"], cob=r["cob"], M=r["M"], K=r["K"], p=r["p"], p_den=r["p_den"],
                   source=dict(dual_points=r["dual_points"], mirror_heights=r["mirror_heights"], divisor_basis=S))
        seen[key] = rec
        res.append(rec)
        print(f"{tag}: basis {S}, H = {rec['H_kind']}", flush=True)

    # non-coni examples: triangulation and basis recovered
    for sec, e in NONCONI.items():
        P = Polytope(np.array(e["V"]).T.tolist())
        h = len(e["M"])
        den = _lcm(q.denominator for q in e["p"])
        pd, M, K = np.array([int(q * den) for q in e["p"]]), np.array(e["M"]), np.array(e["K"])
        tris = P.all_triangulations(only_fine=True, only_regular=True, only_star=True, as_list=True)
        hits = []
        for t in tris:
            cy = t.get_cy()
            kall = np.asarray(cy.intersection_numbers(in_basis=False, format="dense")).astype(np.int64)
            for S in itertools.combinations(range(1, kall.shape[0]), h):
                N = np.einsum("abc,c->ab", kall[np.ix_(S, S, S)], M)
                if np.array_equal(N @ pd, den * K) and int(K @ pd) == 0:
                    hits.append((t, list(S)))
        assert hits, sec
        t, S = hits[0]
        cy = t.get_cy()
        cy.set_divisor_basis(S)
        kappa = np.asarray(cy.intersection_numbers(in_basis=True, format="dense")).astype(int)
        c2 = np.asarray(cy.second_chern_class(in_basis=True)).astype(int)
        if sec == "6.1":
            k61 = np.zeros((5, 5, 5), dtype=int)
            for (a, b, c), v in KAPPA_61.items():
                for i, j, k in set(itertools.permutations((a - 1, b - 1, c - 1))):
                    k61[i, j, k] = v
            assert np.array_equal(kappa, k61) and c2.tolist() == C2_61, "6.1 does not reproduce the paper"
        gvs = _gvs(cy)
        cap, eff, nz = _cones(cy, gvs)
        assert (nz @ pd > 0).all(), sec
        in_cap = bool(np.all(cap @ pd > 0))
        res.append(dict(paper="arXiv:2107.09064", kind="non-coni", section=sec, name=f"vacuum {sec}", model=None,
                        aliases=[], h11=h, h21=e["h21"], kappa=kappa.tolist(), c2=c2.tolist(),
                        H=(cap if in_cap else eff).tolist(),
                        H_kind="mori_cone_cap" if in_cap else f"effective curves (GV invariants of degree <= {GV_DEG})",
                        M=e["M"], K=e["K"], p=pd.tolist(), p_den=den,
                        source=dict(vertices=e["V"], heights=np.asarray(t.heights()).tolist(), divisor_basis=S,
                                    n_matching=len(hits))))
        print(f"{sec}: basis {S}, {len(hits)} (FRST, basis) matches, H = {res[-1]['H_kind']}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f)


def finalize(geom, out, max_dilation=200):
    import warnings
    from latticepts import box_enum
    from pfvs import CYData, ZpM, coniZpM
    warnings.simplefilter("ignore")
    with open(geom) as f:
        exs = json.load(f)
    for e in exs:
        coni = e["kind"] == "coni"
        kw = dict(coni_curve=e["coni_curve"], coni_cob=e["cob"]) if coni else {}
        data = CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"], **kw)
        p = np.array(e["p"])
        p_s = p[1:] if coni else p
        Q = -int(np.dot(e["M"], e["K"]))

        def run(ps, D, coni=coni, data=data, Q=Q):
            if coni:
                return coniZpM(data, ps, Q=Q, M0min=13, ellipsoid_dilation=D, n_jobs=-1, device="cpu")
            return ZpM(data, ps, Qmin=0, Qmax=Q, ellipsoid_dilation=D, n_jobs=-1)

        def hit(res, e=e):
            return any(np.array_equal(k, e["K"]) and np.array_equal(m, e["M"]) for k, m in zip(*res))

        e["min_dilation"] = next(D for D in range(1, max_dilation + 1) if hit(run(np.array([p_s]), D)))
        e["scan"] = None
        if e["H_kind"] == "mori_cone_cap":          # scans only inside the inner cone
            Hs = data.H_cob if coni else data.H
            B = int(np.abs(p_s).max())
            ps, st, _ = box_enum(B, np.ascontiguousarray(Hs.astype(np.int32)), 1, 10**7, primitive=True)
            if st == 0:
                assert hit(run(ps, e["min_dilation"])), e["section"]
                # slow for the test suite (single core): over a million p-vectors
                e["scan"] = dict(B=B, n_p=len(ps), slow=len(ps) > 10**6)
        print(e["section"], e["name"], "min dilation", e["min_dilation"], "scan", e["scan"], flush=True)
    doc = {"description": "Named PFV examples of arXiv:2406.13751 (coni: Sec. 5 and App. C) and "
           "arXiv:2107.09064 (non-coni: Sec. 6), as pfvs inputs. Built by "
           "tests/data/build_paper_examples.py; see there for provenance.", "examples": exs}
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("geometry", "finalize"):
        sys.exit(__doc__ or "usage: build_paper_examples.py geometry REPO_DIR geometry.json | "
                            "finalize geometry.json paper_examples.json")
    (geometry if sys.argv[1] == "geometry" else finalize)(sys.argv[2], sys.argv[3])
