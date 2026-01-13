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
# Description:  This module contains methods for constructing PFVs using the
#               "Zp" style algorithms. These operate by fixing some p-vectors
#               and then searching for lattice points in an ellipsoid, one
#               for each p-vector.
# -----------------------------------------------------------------------------

# external imports
import math
import numpy as np
from ortools.sat.python import cp_model
from tqdm.auto import tqdm

# local imports
from . import lattice

# compute p-vectors
# -----------------
def pvecs(
    data: "CYData",
    max_deg: int = None,
    min_pts: int = None,
    backend: str = "cpsat") -> "ArrayLike":
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
    - `backend`: THe backend to use. Either

    **Returns:**
    Possible p-vectors.
    """
    # check inputs
    if (max_deg is None) ^ (min_pts is None):
        pass
    else:
        raise ValueError("Either `max_deg` or `min_pts` must be set...")

    # read hyperplanes
    if data.coni:
        H = data.H_cob
    else:
        H = data.H

    # compute a grading vector
    # (just needs to be in strict interior of dual cone)
    grading = H.sum(axis=0)
    grading = grading//np.gcd.reduce(grading)

    if backend == "cpsat":
        if max_deg is None:
            raise ValueError("CP-SAT backend assumes max_deg is set...")

        # define a constraint-programming model to solve
        model  = cp_model.CpModel()
        max_pi = cp_model.INT32_MAX - 1

        p_vars = [model.NewIntVar(-max_pi, max_pi, f'x{i}') for i in\
                                                            range(H.shape[1])]

        for row in H:
            model.Add(sum(int(row[j])*p_vars[j] for j in range(H.shape[1]))>=1)
        model.Add(sum(int(grading[j])*p_vars[j] for j in range(H.shape[1])) <=\
                                                                        max_deg)

        # enumerate all solutions up to max_deg
        solver = cp_model.CpSolver()

        class SolutionPrinter(cp_model.CpSolverSolutionCallback):
            def __init__(self, p_vars):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.p_vars = p_vars
                self.points = []

            def on_solution_callback(self):
                self.points.append([self.Value(v) for v in self.p_vars])

        printer = SolutionPrinter(p_vars)

        solver.SearchForAllSolutions(model, printer)

        # return
        return printer.points
    else:
        # import gurobi
        import gurobipy as gp

        # define the model
        model = gp.Model("pFinder")
        model.setParam('OutputFlag', False)

        p = model.addMVar(
            (H.shape[1],),
            lb=-float('inf'), ub=float('inf'),
            vtype=gp.GRB.INTEGER)
        model.setMObjective(
            Q=None,
            c=grading,
            constant=0,
            xc=p,
            sense=gp.GRB.MINIMIZE)
        model.addMConstr(H, p, '>', np.full(len(H),0.5))

        # optimize for N solutions
        model.setParam('PoolSearchMode', 2)
        model.setParam('PoolSolutions', min_pts)

        model.optimize()

        # read the solutions
        sols = []
        for i in range(model.SolCount):
            model.setParam('SolutionNumber', i)

            sols.append(np.rint(p.xn).astype(int))

        return np.array(sols).tolist()

# Zp helpers
# ----------
def allow_gcds(Ks, Ms, Qmax, h11):
    """
    Introduce nontrivial GCDs into (K,M) pairs
    """
    num_input = len(Ks)
    if num_input == 0:
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)

    KMs_out = set()
    
    for K,M in zip(Ks,Ms):
        Qtmp = -np.dot(K,M)

        for a in range(1,Qmax//Qtmp+1):
            for b in range(1,Qmax//(a*Qtmp)+1):
                Ktmp = (a*K).tolist()
                Mtmp = (b*M).tolist()
                KMs_out.add(tuple(Ktmp+Mtmp))

    # split back into K and M arrays
    Ks, Ms = [], []
    for KM in KMs_out:
        Ks.append(KM[:h11])
        Ms.append(KM[h11:])

    if len(Ks):
        return np.vstack(Ks), np.vstack(Ms)
    else:
        raise ValueError(f"#input = {num_input}")
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)

def all_coni_K0(Ks, Ms, Qmax, h11, verbosity: int = 0):
    """
    For Coni PFV, set K0 to all permissible values
    """
    num_input = len(Ks)
    if num_input == 0:
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)
    Ms_input = Ms.copy()

    KMs_out = set()

    for K,M in zip(Ks,Ms):
        if M[0] == 0:
            if verbosity >= 1:
                print(f"infinite # of PFVs K=[?,{K[1:].tolist()}], M={M.tolist()}")
            continue

        Qtmp  = -np.dot(K[1:],M[1:])

        # compute the ranges such that (K,M) are under tadpole
        # simple to derive:
        #     0         <= -M[0]*K[0] + Qtmp <= Qmax
        #     -Qtmp     <= -M[0]*K[0]        <= Qmax-Qtmp
        #     Qtmp/M[0] ?= K[0]              ?= (Qtmp-Qmax)/M[0]
        # where ?= is either >= or <= depending on whether M[0]>0 or M[0]<0
        K0min, K0max = Qtmp/M[0], (Qtmp-Qmax)/M[0]
        if K0min > K0max:
            K0min, K0max = K0max, K0min

        Ktmp = K[1:].tolist()
        Mtmp = M.tolist()
        for K0 in range(math.floor(K0min), math.ceil(K0max)+1):
            Q = Qtmp - K0*M[0]
            if (Q < 0) or (Q > Qmax):
                continue

            KMs_out.add(tuple([K0]+Ktmp+Mtmp))

    # split back into K and M arrays
    Ks, Ms = [], []
    for KM in KMs_out:
        Ks.append(KM[:h11])
        Ms.append(KM[h11:])

    if len(Ks):
        return np.vstack(Ks), np.vstack(Ms)
    else:
        print(f"#input = {num_input}, Ms={Ms_input}")
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)

# Zp
# --
# ZpM
def ZpM(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: float = None,
    Qmin: float = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0
    ) -> tuple["ArrayLike", "ArrayLike"]:
    """
    A Python "Zp" implementation that takes in p-vectors and outputs PFVs.

    Operates by constructing a lattice of valid M-vectors and writing the
    K-vector in terms of M and p. Then the tadpole constraint 0 <= -K.M <= Q
    defines an ellipsoid of M-vectors

    filter_Ninvertible = whether the PFV has a invertible N matrix

    returns a list of Ks and Ms
    """
    assert not data.coni

    # misc
    only_positive_news = False

    # read data
    kappa  = data.kappa
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = data.h11+data.h21+2

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

        # helper variable (K = Z@M)
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
            ellipsoid_dilation*Qmax,
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
            in_tadpole = (Qs>Qmin) & (Qs<Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{len(in_tadpole)-sum(in_tadpole)}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

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
    return allow_gcds(all_Ks, all_Ms, Qmax, data.h11)

def ZpK(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: float = None,
    Qmin: float = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0
    ) -> tuple["ArrayLike", "ArrayLike"]:
    """
    A Python Zp implementation that takes in p-vectors and outputs PFVs

    filter_Ninvertible = whether the PFV has a invertible N matrix

    returns a list of Ks and Ms
    """
    assert not data.coni

    # misc
    only_positive_news = False

    # read data
    kappa  = data.kappa
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = data.h11+data.h21+2

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

        # helper variables
        A = kappa@p@Mbasis
        try:
            Ainv, scale = lattice.inv_scaled(A)#np.linalg.inv(A)
        except:
            print(f"PANIC!!!! {p.tolist()} CAUSED WEIRD AINV!!!")
        
        # define the lattices for K
        # -------------------------
        # (need K^T@p = 0)
        B = lattice.orthogonal_lattice(p=p)
        
        # find lattice points in tadpole
        mat = -B.T@np.linalg.inv(kappa@p)@B

        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)

        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        lattice_points = lattice.fp_ellipsoid(
            mat,
            ellipsoid_dilation*Qmax,
            Q_lower=Qmin,
            max_N_out=max_N_pfvs,
            verbosity=verbosity-1)

        # only keep primitive lattice points (can reclaim other PFVs easily)
        primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
        lattice_points = lattice_points[primitiveQ]
        
        # compute Ms, Ks, and reduced by GCD

        # read the data
        cs = ((Ainv@B)@lattice_points.T).T
        gcds = np.gcd.reduce(cs,axis=1)
        cs_scaled = cs//gcds.reshape(-1,1)

        Ks = B@lattice_points.T # as columns
        Ms = Mbasis@cs_scaled.T

        # filter
        #if reduce_gcds:
        #    Ks = Ks//np.gcd.reduce(Ks, axis=0)

        if True:#filter_tadpole:
            Qs = -np.sum(Ks*Ms,axis=0)
            in_tadpole = (Qs>Qmin) & (Qs<Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{len(in_tadpole)-sum(in_tadpole)}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

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

            if verbosity >= 2:
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
    return allow_gcds(all_Ks, all_Ms, Qmax, data.h11)

# Coni-Zp
# -------
def coniZpM(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: float = None,
    Qmin: float = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0
    ) -> tuple["ArrayLike", "ArrayLike"]:
    """
    A Python "Zp" implementation that takes in p-vectors and outputs PFVs.

    Operates by constructing a lattice of valid M-vectors and writing the
    K-vector in terms of M and p. Then the tadpole constraint 0 <= -K.M <= Q
    defines an ellipsoid of M-vectors

    filter_Ninvertible = whether the PFV has a invertible N matrix

    returns a list of Ks and Ms
    """
    assert data.coni

    # misc
    only_positive_news = False

    # read data
    kappa  = data.kappa_cob
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = (data.h11+data.h21+2) + 2

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

        _0p = np.concatenate([[0],p])

        # projection matrix
        proj = np.hstack([
            np.zeros((data.h11-1,1), dtype=int),
            np.identity(data.h11-1,  dtype=int)
        ])

        # helper variable (K[1:] = (Z@M)[1:])
        Z = kappa@_0p

        # define the lattices for M
        # -------------------------
        # (need M = Mbasis@c and T.T@M = 0)
        T = kappa@_0p@_0p

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
            ellipsoid_dilation*Qmax,
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
            in_tadpole = (Qs>Qmin) & (Qs<Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{len(in_tadpole)-sum(in_tadpole)}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

        # filter by N invertibility
        if True:#filter_Ninvertible:
            batch_size = 5000
            singular = []
            for i in range(0, len(Ms), batch_size):
                chunk = Ms[i:i+batch_size]

                Ns = np.tensordot(kappa, Ms, axes=([2], [0]))
                Ns = Ns.transpose(2,0,1)
                Ns = Ns[:,1:,1:]

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
    all_Ks, all_Ms = allow_gcds(all_Ks, all_Ms, Qmax, data.h11)
    all_Ks, all_Ms = all_coni_K0(all_Ks, all_Ms, Qmax, data.h11)
    return all_Ks, all_Ms
