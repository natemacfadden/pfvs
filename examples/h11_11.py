"""
Coni PFVs of an (h11, h21) = (11, 347) geometry, built with CYTools.

    python examples/h11_11.py                 # 100k p-vectors, dilation 80
    python examples/h11_11.py --n-p 1000000   # a larger scan

Needs CYTools (https://cy.tools). Stops if q is no longer a conifold curve in
the installed CYTools' divisor basis (it has changed between versions).
"""

import argparse
import time

import numpy as np

from pfvs import CYData, coniZpM, pvecs

VERTS = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [-6, -12, 8, -1], [-6, -11, 7, -1],
         [-3, -1, -1, 0], [0, 0, 1, 0]]
HEIGHTS = [-7.166666666666668, 0.0, 21.000000000000004, 2.0, 0.0, 6.336593692049902e-17, 0.0, 0.0,
           10.000000000000002, 5.000000000000001, 2.0, 0.0, -1.0000000000000002, 4.333333333333335,
           4.500000000000001, 1.666666666666667]
Q_CURVE = [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-p", type=int, default=100_000, help="p-vectors to scan (at least)")
    ap.add_argument("--dilation", type=float, default=80, help="ellipsoid dilation")
    ap.add_argument("--gv-degree", type=int, default=7, help="max degree of the GV invariants")
    args = ap.parse_args()

    try:
        from cytools import Polytope
    except ImportError:
        raise SystemExit("this example needs CYTools (https://cy.tools); "
                         "examples/manwe.py runs without it") from None

    t = time.perf_counter()
    cy = Polytope(VERTS).triangulate(heights=HEIGHTS).cy()
    gvs = cy.compute_gvs(max_deg=args.gv_degree)
    print(f"(h11, h21) = ({cy.h11()}, {cy.h21()}); GVs to degree {args.gv_degree} "
          f"in {time.perf_counter() - t:.1f} s")
    if gvs.dok.get(tuple(Q_CURVE)) != 2:
        raise SystemExit(f"q = {Q_CURVE} is not a conifold curve in this CYTools version's "
                         f"divisor basis {cy.divisor_basis().tolist()}")

    # the change of basis to q = (1, 0, ..., 0) is chosen by CYData
    data = CYData.from_cy(cy, coni_curve=Q_CURVE)

    t = time.perf_counter()
    ps = pvecs(data, args.n_p)
    found = coniZpM(data, ps, ellipsoid_dilation=args.dilation, return_formal_pfvs=True)
    print(f"{len(found)} coni PFVs (tadpole h11 + h21 + 4, M0 >= 13) from {len(ps)} "
          f"p-vectors (dilation {args.dilation:g}) in {time.perf_counter() - t:.1f} s")

    # W0 needs a valid racetrack: the two leading GV terms of the flux
    # superpotential must have the right sign ratio
    coo = gvs.coo
    for pfv in found:
        pfv.gvs = coo
    W0 = np.array([pfv.W0() for pfv in found])
    ok = np.flatnonzero(np.isfinite(W0))
    print(f"{len(ok)} with a valid racetrack; the smallest W0:")
    print(f"{'W0':>10} {'gs':>8} {'gs*M':>8} {'align':>8}  M")
    for j in ok[np.argsort(W0[ok])][:10]:
        pfv = found[j]
        print(f"{W0[j]:10.3g} {pfv.gs:8.4f} {pfv.gsM:8.4f} {pfv.align:8.3g}  {pfv.M.tolist()}")
    if len(ok):
        print("\nthe smallest:")
        found[ok[np.argmin(W0[ok])]].diagnostics()


if __name__ == "__main__":
    main()
