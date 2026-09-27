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
# Description:  Builds tests/data/fixtures.json.gz from the public coni-PFV
#               dataset (https://huggingface.co/datasets/natemacfadden/
#               calabi-yau-coni-pfvs, pinned revision below). Not run by the
#               test suite; rerun by hand to regenerate:
#
#                   pip install duckdb huggingface_hub
#                   python tests/data/build_fixtures.py
#
#               Two kinds of fixture:
#                 - "geometries": CY data + the dataset's PFVs, restricted to
#                   the (B, D) at which the test reruns coniZpM. The dataset
#                   holds every PFV with p-vector infnorm <= B and required
#                   dilation <= D, so coniZpM must reproduce the set exactly.
#                 - "kernel": real (mat, H, linvec, Q) kernel inputs built
#                   from those geometries, with outputs from the exact
#                   rational oracle (tests/oracle.py), at several dilations.
#               Kernel inputs are built here with exact (flint) arithmetic,
#               independently of the pfvs package.
# -----------------------------------------------------------------------------

import gzip
import json
import sys
import time
from pathlib import Path

import flint
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from oracle import kernel_reference

REPO     = "natemacfadden/calabi-yau-coni-pfvs"
REVISION = "63d2db7942df733e1c92845451f5557cbbd16e87"

# geometries: per h11, how many, at which (B, D)
GEOMS_PER_H11 = 8
GEOM_B, GEOM_D = 5, 150
# kernel instances: per geometry, how many p-vectors; dilations; oracle budget
PS_PER_GEOM = 3
BIG_PS_PER_GEOM = 1
KERNEL_DILATIONS = (1, 4, 13, 40)
ORACLE_BUDGET_S = 1.0


def load_rows():
    import duckdb
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, "coni_pfvs.parquet", repo_type="dataset",
                           revision=REVISION)
    con = duckdb.connect()
    cur = con.execute(f"""
        select * from (
            select *, row_number() over (
                partition by h11 order by hash(polyID, classID, coniID)) rn
            from '{path}')
        where rn <= {GEOMS_PER_H11} order by h11, rn""")
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def kappa_dense(h11, coo):
    kappa = np.zeros((h11,) * 3, dtype=object)
    for i, j, k, v in coo:
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            kappa[a, b, c] = int(v)
    return kappa


def _np(m):
    return np.array([[int(x) for x in r] for r in m.tolist()], dtype=object)


def _lll_cols(B):
    return _np(flint.fmpz_mat([[int(x) for x in r] for r in B.T]).lll()).T


def M_lattice(row):
    """M-lattice basis (columns) from the package's CYData (coni basis)."""
    from pfvs import CYData
    h11 = row["h11"]
    Hc = np.array(row["H"], dtype=np.int64)
    Hfull = np.hstack([np.zeros((len(Hc), 1), dtype=np.int64), Hc])
    e0 = np.eye(h11, dtype=np.int64)
    data = CYData(row["h21"], kappa_dense(h11, row["kappa_coo"]).astype(np.int64),
                  row["c2"], Hfull, coni_curve=e0[0], coni_cob=e0)
    return np.array(data.M_lattice(), dtype=object)


def coni_instance(kappa, Mbasis, p_perp):
    """Exact coni ellipsoid (mat), Binter row 0, and H = HNF((Z Binter)[1:])."""
    p = np.array([0] + [int(x) for x in p_perp], dtype=object)
    Z = kappa @ p
    T = Z @ p
    v = [int(x) for x in T @ Mbasis]
    U = flint.fmpz_mat([[x] for x in v]).hnf(transform=True)[1]
    orth = _np(flint.fmpz_mat([[U[i, j] for j in range(len(v))]
                               for i in range(1, len(v))]).lll()).T
    B = _lll_cols(Mbasis @ orth)
    B = B[:, np.argsort(B[0] != 0, kind="stable")]
    assert all(x == 0 for x in T @ B)
    mat = -(B.T @ (Z @ B))
    ZB = Z @ B
    H = _np(flint.fmpz_mat([[int(x) for x in r] for r in ZB[1:]]).hnf())
    return mat, H, B[0]


def pvec_candidates(Hc, B, rng, n, big):
    """Primitive p with Hc p >= 1 and |p|_inf <= B; 'big' ones are sums."""
    from latticepts import box_enum
    ps, _, _ = box_enum(B, np.ascontiguousarray(np.array(Hc, dtype=np.int32)),
                         1, 10**7, primitive=True)
    if len(ps) == 0:
        return []
    out = [ps[i] for i in rng.choice(len(ps), size=min(n, len(ps)), replace=False)]
    for _ in range(big):
        k = rng.choice(len(ps), size=3)
        w = rng.integers(20, 120, 3)
        p = (w[:, None] * ps[k].astype(np.int64)).sum(0)
        out.append(p // np.gcd.reduce(p))
    return out


def main():
    rng = np.random.default_rng(20260927)
    rows = load_rows()
    geoms, kernel = [], []
    for row in rows:
        h11 = row["h11"]
        B = min(GEOM_B, min(row["frontier_infnorm"]))
        D = min(GEOM_D, min(row["frontier_dil"]))
        inf = np.array(row["pfv_infnorm"], dtype=np.int64)
        rd = np.array(row["pfv_reqdil"], dtype=np.int64)
        sel = (inf <= B) & (rd <= D)
        geoms.append({
            "h11": h11, "h21": row["h21"], "polyID": row["polyID"],
            "classID": row["classID"], "coniID": row["coniID"],
            "kappa_coo": row["kappa_coo"], "c2": row["c2"], "H": row["H"],
            "B": B, "D": D,
            "K": [k for k, s in zip(row["K"], sel) if s],
            "M": [m for m, s in zip(row["M"], sel) if s]})

        kappa = kappa_dense(h11, row["kappa_coo"])
        Mbasis = M_lattice(row)
        Q = h11 + row["h21"] + 4
        for p in pvec_candidates(row["H"], 6, rng, PS_PER_GEOM, BIG_PS_PER_GEOM):
            mat, H, linvec = coni_instance(kappa, Mbasis, p)
            inst = {"h11": h11, "polyID": row["polyID"], "coniID": row["coniID"],
                    "p": [int(x) for x in p], "Q": Q, "M0min": 13,
                    "mat": mat.tolist(), "H": H.tolist(), "linvec": linvec.tolist(),
                    "expected": {}}
            for dil in KERNEL_DILATIONS:
                t = time.perf_counter()
                pts, qs = kernel_reference(mat, Q, dil, H, linvec, 13, strict=True)
                if time.perf_counter() - t > ORACLE_BUDGET_S:
                    break
                inst["expected"][str(dil)] = {"pts": [list(c) for c in pts], "qs": qs}
            if inst["expected"]:
                kernel.append(inst)
        print(f"h11={h11} poly={row['polyID']} coni={row['coniID']}: "
              f"{int(sel.sum())} PFVs; {len(kernel)} kernel instances so far",
              flush=True)

    out = {"source": f"https://huggingface.co/datasets/{REPO}", "revision": REVISION,
           "geometries": geoms, "kernel": kernel}
    with gzip.open(HERE / "fixtures.json.gz", "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {len(geoms)} geometries, {len(kernel)} kernel instances")


if __name__ == "__main__":
    main()
