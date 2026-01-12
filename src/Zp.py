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
# Description:  This module contains utilities constructing PFVs using the "Zp"
#               style algorithms. These operate by fixing some p-vectors and
#               then searching for lattice points in a derived ellipsoid, one
#               for each p-vector.
# -----------------------------------------------------------------------------

# external imports
import functools
import math
import numpy as np
from ortools.sat.python import cp_model
from tqdm.auto import tqdm

# local imports
from . import lattice

# compute p-vectors
# -----------------
def pvecs(data: "CYData", max_deg: int) -> "ArrayLike":
    """
    **Description:**
    Computes possible p-vectors.

    For non-Coni PFVs, these are integral vectors p such that
        H @ p > 0
    for H the hyperplanes of the Kahler cone. I.e., p is in the interior of the
    Kahler cone.

    For Coni PFVs, these are integral vectors p such that
        (H\\coni_normal) @ p > 0
        dot(coni_normal, p)  = 0
    for H the hyperplanes of the Kahler cone. I.e., p is in the interior of the
    facet of the Kahler cone defined by coni_normal.

    **Arguments:**
    - `data`:    The CYData describing the CY.
    - `max_deg`: The maximum degree to compute points to.

    **Returns:**
    Possible p-vectors.
    """
    if data.coni:
        H = data.H_cob
    else:
        H = data.H

    # compute a grading vector
    # (just needs to be in strict interior of dual cone)
    grading = data.H.sum(axis=0)
    grading = grading//functools.reduce(math.gcd,grading)

    # define a constraint-programming model to solve
    model  = cp_model.CpModel()
    max_xi = cp_model.INT32_MAX - 1

    x_vars = [model.NewIntVar(-max_xi, max_xi, f'x{i}') for i in range(H.shape[1])]

    for row in H:
        model.Add(sum(int(row[j])*x_vars[j] for j in range(H.shape[1])) >= 1)
    model.Add(sum(int(grading[j])*x_vars[j] for j in range(H.shape[1])) <= max_deg)

    # enumerate all solutions up to max_deg
    solver = cp_model.CpSolver()

    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, x_vars):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.x_vars = x_vars
            self.points = []

        def on_solution_callback(self):
            self.points.append([self.Value(v) for v in self.x_vars])

    printer = SolutionPrinter(x_vars)

    solver.SearchForAllSolutions(model, printer)

    # return
    return printer.points

# Zp
# --
# ZpM
def ZpM(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: float,
    Qmin: float = 0,
    tadpole_dilation: float = 1, # typically want >=1
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0
    ):
    """
    A Python "Zp" implementation that takes in p-vectors and outputs PFVs.

    Operates by constructing a lattice of valid M-vectors and writing the
    K-vector in terms of M and p. Then the tadpole constraint 0 <= -K.M <= Q
    defines an ellipsoid of M-vectors

    filter_Ninvertible = whether the PFV has a invertible N matrix

    returns a list of Ks and Ms
    """
    kappa  = data.kappa
    Mbasis = data.M_lattice()

    # the search
    # ----------
    all_Ks = np.zeros((0,data.h11), dtype=int)
    all_Ms = np.zeros((0,data.h11), dtype=int)

    # iterate over p-vectors
    if verbosity >= 1:
        iterator = tqdm(ps)
    else:
        iterator = ps
    for _i, p in enumerate(iterator):
        if verbosity > 0:
            print(f"progress={_i}/{len(ps)-1}", end='\r')

        # helper variable
        Z = kappa@p
        
        # define the lattices for M
        # -------------------------
        # (need M = Mbasis@c and T.T@M = 0)
        T = kappa@p@p

        # need T.T @ Mbasis@c = 0
        # thus just need c in the orthogonal lattice to Mbasis.T @ T
        # (the output will be lattice generators of such cs... we'll want
        #  lattice generators of valid Ms so we multiply on left by Mbasis)
        Binter = Mbasis@lattice.orthogonal_lattice(p=T.T@Mbasis)
        
        # find lattice points in tadpole
        mat = -(Binter).T@(Z@Binter)

        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)

        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        lattice_points = lattice.fp_ellipsoid(
            mat,
            tadpole_dilation*Qmax,
            Q_lower=Qmin,
            max_N_out=max_N_pfvs,
            verbosity=verbosity-1)

        # only keep primitive lattice points (can reclaim other PFVs easily)
        primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
        lattice_points = lattice_points[primitiveQ]
        
        # compute Ms, Ks, and reduced by GCD
        Ms = Binter@lattice_points.T # as columns
        Ks = Z@Ms

        #if reduce_gcds:
        #K_gcds = gcd_row(Ks.T)
        K_gcds = np.gcd.reduce(Ks, axis=0)
        Ks = Ks//K_gcds

        if True:#filter_tadpole:
            Qs = -np.sum(Ks*Ms,axis=0)
            under_tadpole = Qs<Qmax
            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{len(under_tadpole)-sum(under_tadpole)}/{len(under_tadpole)} 'PFVs' violated tadpole :(")
                if sum(under_tadpole):
                    print(f"but {sum(under_tadpole)} under tadpole!!!")
            Ks = Ks[:,under_tadpole]
            Ms = Ms[:,under_tadpole]

        # filter by N invertibility
        if True:#filter_Ninvertible:
            batch_size = 5000
            singular = []
            for i in range(0, len(Ms), batch_size):
                chunk = Ms[i:i+batch_size]

                Ns = np.tensordot(kappa, Ms, axes=([2], [0]))
                Ns = Ns.transpose(2,0,1)

                sign, logdet = np.linalg.slogdet(Ns)
                is_zero = (sign == 0) | (logdet <= -1e-4)

                singular.append(is_zero)

            singular = np.concatenate(singular)

            if verbosity >= 1:
                if not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

            Ks = Ks[:,~singular]
            Ms = Ms[:,~singular]

        # transpose to row-wise
        Ks, Ms = Ks.T, Ms.T

        # save to data structures
        all_Ks = np.vstack([all_Ks, Ks])
        all_Ms = np.vstack([all_Ms, Ms])

    # return
    return all_Ks, all_Ms
