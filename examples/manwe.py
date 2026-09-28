"""
Manwe (arXiv:2406.13751, Example 1): check a known conifold PFV, then find
it again from scratch.

    python examples/manwe.py            # checks, the scan and a W0 summary
    python examples/manwe.py --plot     # ... and save manwe.png (W0 vs align)

Needs no CYTools: the geometry (intersection numbers, c2, Kahler cone, all in
the paper's basis) and the Gopakumar-Vafa invariants are read from the test
data. With CYTools the same geometry is

    from cytools import Polytope
    cy = Polytope(verts).triangulate(heights=heights).cy()
    data = CYData.from_cy(cy, coni_curve=q, coni_cob=cob)

with verts, heights, q and cob as in tests/test_manwe.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pfvs import PFV, CYData, coniZpM, pvecs

TESTS = Path(__file__).resolve().parent.parent / "tests"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-p", type=int, default=100_000, help="p-vectors to scan (at least)")
    ap.add_argument("--dilation", type=float, default=50, help="ellipsoid dilation")
    ap.add_argument("--plot", action="store_true", help="save manwe.png: W0 vs align")
    args = ap.parse_args()

    # the geometry, with Manwe's conifold curve q and a change of basis taking
    # q to (1, 0, ..., 0)
    # -----------------------------------------------------------------------
    with open(TESTS / "data" / "paper_examples.json") as f:
        e = next(x for x in json.load(f)["examples"] if x["model"] == "manwe")
    data = CYData(h21=e["h21"], kappa=e["kappa"], c2=e["c2"], H=e["H"],
                  coni_curve=e["coni_curve"], coni_cob=e["cob"])
    gvs = np.loadtxt(TESTS / "manwe_gvs_deg10.csv", dtype=int, delimiter=",")

    # Manwe itself
    # ------------
    manwe = PFV(data, K=e["K"], M=e["M"])
    manwe.gvs = gvs                          # GVs up to degree 10 (for W0, gs, ...)
    assert manwe.check_all()
    print(manwe)
    print(f"\nW0 = {manwe.W0():.4g}, gs = {manwe.gs:.4g}, gs*M = {manwe.gsM:.4g}, "
          f"align = {manwe.align:.4g}\n")

    # find it from scratch: every primitive p-vector in a box of the Kahler
    # cone facet, then every coni PFV with tadpole h11 + h21 + 4 and M0 >= 13
    # -----------------------------------------------------------------------
    t = time.perf_counter()
    ps = pvecs(data, args.n_p)
    found = coniZpM(data, ps, Q=data.h11 + data.h21 + 4, M0min=13,
                    ellipsoid_dilation=args.dilation, return_formal_pfvs=True)
    print(f"{len(found)} coni PFVs from {len(ps)} p-vectors "
          f"(dilation {args.dilation:g}) in {time.perf_counter() - t:.1f} s")
    i = next((i for i, pfv in enumerate(found)
              if np.array_equal(pfv.K, manwe.K) and np.array_equal(pfv.M, manwe.M)), None)
    print("Manwe:", "not found (try a larger --n-p or --dilation)" if i is None else f"found, #{i}")

    # rank them by W0 (the GVs are those of the geometry, shared by all)
    # -----------------------------------------------------------------
    for pfv in found:
        pfv.gvs = gvs
    W0 = np.array([pfv.W0() for pfv in found])
    ok = np.flatnonzero(np.isfinite(W0))
    print(f"\n{len(ok)} with a valid racetrack; the smallest W0:")
    print(f"{'W0':>10} {'gs':>8} {'gs*M':>8} {'align':>8}  M")
    for j in ok[np.argsort(W0[ok])][:10]:
        pfv = found[j]
        print(f"{W0[j]:10.3g} {pfv.gs:8.4f} {pfv.gsM:8.4f} {pfv.align:8.3g}  {pfv.M.tolist()}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sc = plt.scatter(W0[ok], [found[j].align for j in ok],
                         c=[found[j].gsM for j in ok], s=6)
        plt.scatter([manwe.W0()], [manwe.align], c="r", marker="*", s=120, label="Manwe")
        plt.colorbar(sc, label="gs M")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("W0")
        plt.ylabel("align")
        plt.title("PFVs from Manwe's conifold")
        plt.legend()
        plt.savefig("manwe.png", dpi=150)
        print("\nsaved manwe.png")


if __name__ == "__main__":
    main()
