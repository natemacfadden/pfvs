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
import flint
import math
import numba
import numpy as np
from ortools.sat.python import cp_model
from tqdm.auto import tqdm

# local imports
from . import lattice

# p-vectors
# =========
def pvecs(
    data: "CYData",
    max_deg: int = None,
    requested_N_pts: int = None,
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
    - `data`:            The CYData describing the CY.
    - `max_deg`:         The maximum degree to compute points to.
    - `requested_N_pts`: The number of points to request. Might get fewer due
                         to non-trivial GCDs
    - `backend`:         The backend to use. Either

    **Returns:**
    Possible p-vectors.
    """
    ps = []

    # check inputs
    if (max_deg is None) ^ (requested_N_pts is None):
        pass
    else:
        raise ValueError("Either `max_deg` or `requested_N_pts` must be set...")

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
        ps = np.array(printer.points)
    elif backend == "gurobi":
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
        model.setParam('PoolSolutions', requested_N_pts)

        model.optimize()

        # read the solutions
        sols = []
        for i in range(model.SolCount):
            model.setParam('SolutionNumber', i)

            sols.append(np.rint(p.xn).astype(int))

        ps = np.array(sols)
    else:
        raise ValueError()

    # reduce by GCD
    gcds = np.gcd.reduce(ps,axis=1)
    ps = ps[gcds==1]
    return ps.tolist()

# Zp helpers
# ==========
# generic
# -------
@numba.njit(parallel=True)
def check_singular(Ns, tol=1e-4):
    n = Ns.shape[0]
    singular = np.zeros(n, dtype=np.bool_)
    for i in numba.prange(n):
        s, ld = np.linalg.slogdet(Ns[i])
        if s == 0.0 or ld <= -tol:
            singular[i] = True
    return singular

# non-coni
# --------
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

# coni
# ----
#@numba.njit
#def gcd_vec(vec):
#    g = abs(vec[0])
#    for i in range(1, len(vec)):
#        x = abs(vec[i])
#        while x != 0:
#            g, x = x, g % x
#    return g

@numba.njit
def try_coni_K0(Qperps, Kperps, Ms, h11, lo, up, Qmin, Qmax, max_N_out: int = 100_000_000):
    """
    columnwise
    """
    # check if empty
    num_pfvs = Ms.shape[1]
    if num_pfvs == 0:
        return np.empty((h11,0), dtype=np.int64), np.empty((h11,0), dtype=np.int64)

    # make the output objects
    Ks_out = np.empty((h11, max_N_out), dtype=np.int64)
    Ms_out = np.empty((h11, max_N_out), dtype=np.int64)
    op = 0

    # fill them
    for i in range(num_pfvs):
        for K0 in range(lo[i], up[i]+1):
            Qtest = Qperps[i] - K0*Ms[0,i]

            # save if this is in the permissible range
            if (Qmin<=Qtest) and (Qtest<=Qmax):
                if op >= max_N_out:
                    print('saturated maximum allowed outputs')
                    return Ks_out, Ms_out

                Ks_out[0,op]  = K0
                Ks_out[1:,op] = Kperps[:,i]
                Ms_out[:,op]  = Ms[:,i]
                op += 1

    return Ks_out[:,:op], Ms_out[:,:op]

@numba.njit
def set_coni_K0(Ks, Ms, Kperp_gcd, h11, Qmin, Qmax,
    #Qperps=None,
    max_N_out: int = 1_000_000, verbosity: int = 0):
    """
    columnwise
    """
    if np.any(Ms[0]==0):
        raise ValueError("needs M0!=0")

    # check if empty
    num_pfvs = len(Ms)
    if num_pfvs == 0:
        return np.empty((h11,0), dtype=np.int64), np.empty((h11,0), dtype=np.int64)

    # output objects
    # --------------
    Ks_out = np.empty((h11, max_N_out), dtype=np.int64)
    Ms_out = np.empty((h11, max_N_out), dtype=np.int64)
    op = 0
    N_skipped = 0
    
    # need to accommodate two things:
    #    1) K0 is effectively unfixed
    #    2) K->K/gcd(K) is allowed (introduces denominator to p-vec)
    # observe:
    #    Qmin <= (-K0*M0 - dot(Kperp, Mperp)) / gcd(K) <= Qmax
    #    Qmin*gcd(K) <= -K0*M0 - dot(Kperp, Mperp) <= Qmax*gcd(K)
    # also observe that 1 <= gcd(K) <= gcd(Kperp) so we can rewrite
    #    Qmin <= -K0*M0 - dot(Kperp, Mperp) <= Qmax*gcd(Kperp)
    # not all such K0 will be valid, but any valid K0 will satisfy this inequality
    # such a K0 can be found simply - rewrte:
    #    Qmin + dot(Kperp, Mperp) <= -K0*M0 <= Qmax*gcd(Kperp) + dot(Kperp, Mperp)
    # so the bounds on K0 are (which one is upper vs. lower depends on sign of M0)
    #    -(Qmin + dot(Kperp, Mperp))/M0
    # and
    #    -(Qmax*gcd(Kperp) + dot(Kperp, Mperp))/M0
    #if Qperps is None:
    #    Qperps = -np.sum(Ks[:,1:]*Ms[:,1:],axis=1)
    Ktest = np.empty(h11, dtype=np.int64)
    for i in range(num_pfvs):
        # read info from K,M
        M0    = Ms[0,i]
        Mperp = Ms[1:,i]
        Kperp = Ks[1:,i]
        #Qperp = Qperps[i]
        Qperp = 0
        for i in range(h11-1):
            Qperp -= Kperp[i]*Mperp[i]

        # bounds on K0
        #Kperp_gcd = gcd_vec(Kperp)
        lo = -(Qmax*Kperp_gcd - Qperp)/M0
        up = -(Qmin - Qperp)/M0
        if up<lo:
            lo,up = up,lo

        # try all values of K0
        for K0 in range(math.floor(lo),math.ceil(up)+1):
            Ktest[0]  = K0
            Ktest[1:] = Kperp

            # compute the dot product
            Qtest = -K0*M0 + Qperp

            # save if this is in the permissible range
            if (Qmin<=Qtest) and (Qtest<=Qmax):
                if op >= max_N_out:
                    print('saturated maximum allowed outputs')
                    return Ks_out, Ms_out

                Ks_out[:,op]  = Ktest
                Ms_out[0,op]  = M0
                Ms_out[1:,op] = Mperp
                op += 1

    return Ks_out[:,:op], Ms_out[:,:op]

# non-coni Zp
# ===========
def ZpM(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: int = None,
    Qmin: int = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    fp_recursive: bool = False,
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
        # helper variable (K = Z@M)
        Z = kappa@p
        
        # define the lattices for M
        # -------------------------
        # need K.p = 0
        #
        # note that K = kappa@M@p
        # thus need dot(p, kappa@M@p) = 0
        # equivalently, dot(kappa@p@p, M) = 0
        T = kappa@p@p

        # need T.T @ Mbasis@c = 0
        # thus just need c in the orthogonal lattice to Mbasis.T @ T
        # (the output will be lattice generators of such cs... we'll want
        #  lattice generators of valid Ms so we multiply on left by Mbasis)
        Binter = Mbasis@lattice.orthogonal_lattice(p=T.T@Mbasis)

        # lll-reduce Binter
        # (doesn't seem to have a huge effect...)
        Binter = lattice.lll_reduce(Binter)
        
        # find lattice points in tadpole
        mat = -(Binter).T@(Z@Binter)

        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)

        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        try:
            lattice_points = lattice.fp_ellipsoid(
                mat,
                ellipsoid_dilation*Qmax,
                Q_lower=Qmin,
                max_N_out=max_N_pfvs,
                recursive=fp_recursive,
                verbosity=verbosity-1)
        except Exception as e:
            print("ERROR!!!")
            print(f"LIKELY mat={mat.tolist()} ISN'T POSITIVE DEFINITE...")
            #print(f"vertices = {self.polytope().vertices().tolist()}")
            #print(f"heights  = {self.triangulation().heights().tolist()}")
            print(f"p        = {np.array(p).tolist()}")
            raise e

        # only keep primitive lattice points (can reclaim other PFVs easily)
        primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
        lattice_points = lattice_points[primitiveQ]
        
        # compute Ms
        # ----------
        Ms = Binter@lattice_points.T # as columns

        # filter by N invertibility
        # -------------------------
        if True:#filter_Ninvertible:
            batch_size = 5000
            singular = []
            for i in range(0, len(Ms), batch_size):
                chunk = Ms[i:i+batch_size]

                #Ns = np.tensordot(kappa, Ms, axes=([2], [0]))
                Ns = (kappa.reshape(data.h11*data.h11,data.h11)@Ms).reshape(data.h11,data.h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)

                #sign, logdet = np.linalg.slogdet(Ns)
                #is_zero = (sign == 0) | (logdet <= -1e-4)
                #singular.append(is_zero)
                singular.append(check_singular(Ns.astype(float)))

            singular = np.concatenate(singular)

            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

            Ks = Ks[:,~singular]
            Ms = Ms[:,~singular]

        # compute Ks, reduce by GCDs
        # --------------------------
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
    Qmax: int = None,
    Qmin: int = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    fp_recursive: bool = False,
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

        # lll-reduce B
        # (doesn't seem to have a huge effect...)
        B = lattice.lll_reduce(B)
        
        # find lattice points in tadpole
        mat = -B.T@np.linalg.inv(kappa@p)@B

        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)

        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        try:
            lattice_points = lattice.fp_ellipsoid(
                mat,
                ellipsoid_dilation*Qmax,
                Q_lower=Qmin,
                max_N_out=max_N_pfvs,
                recursive=fp_recursive,
                verbosity=verbosity-1)
        except Exception as e:
            print("ERROR!!!")
            print(f"LIKELY mat={mat.tolist()} ISN'T POSITIVE DEFINITE...")
            #print(f"vertices = {self.polytope().vertices().tolist()}")
            #print(f"heights  = {self.triangulation().heights().tolist()}")
            print(f"p        = {np.array(p).tolist()}")
            raise e

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

                #Ns = np.tensordot(kappa, Ms, axes=([2], [0]))
                Ns = (kappa.reshape(data.h11*data.h11,data.h11)@Ms).reshape(data.h11,data.h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)

                #sign, logdet = np.linalg.slogdet(Ns)
                #is_zero = (sign == 0) | (logdet <= -1e-4)
                #singular.append(is_zero)
                singular.append(check_singular(Ns.astype(float)))

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

# coni Zp
# =======
def coniMellipsoid(p, data):
    kappa  = data.kappa_cob
    Mbasis = data.M_lattice()

    p = np.array(p).ravel()
    if len(p) == data.h11-1:
        p = np.concatenate([[0], p])

    # helper variable (K[1:] = (Z@M)[1:])
    Z = kappa@p

    # define the lattices for M
    # -------------------------
    # need K.p = 0
    #
    # note that K[1:] = (kappa@M@p)[1:]
    # (it can get K[0] wrong but that's OK since p[0]=0)
    #
    # thus need dot(p, kappa@M@p) = 0
    # equivalently, dot(kappa@p@p, M) = 0
    T = kappa@p@p

    # need T.T @ Mbasis@c = 0
    # thus just need c in the orthogonal lattice to Mbasis.T @ T
    # (the output will be lattice generators of such cs... we'll want
    #  lattice generators of valid Ms so we multiply on left by Mbasis)
    Binter = Mbasis@lattice.orthogonal_lattice(p=T.T@Mbasis)

    # lll-reduce Binter
    # (doesn't seem to have a huge effect...)
    Binter = lattice.lll_reduce(Binter)

    # enforce a maximal number of 0s in the 0th row
    """
    Bflint = flint.fmpz_mat(Binter.tolist()).transpose()
    Binter = Bflint.hnf(transform=False).transpose().tolist()
    Binter = np.array(Binter).astype(int)
    """

    # sort Binter so columns which don't affect M0 come first
    Binter = Binter[:,np.argsort(Binter[0]!=0)]

    # find lattice points in tadpole
    # ------------------------------
    mat = -(Binter).T@(Z@Binter)

    if np.allclose(mat, np.round(mat)):
        mat = np.rint(mat).astype(int)

    return mat, Z, Binter


def coniZpM(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: int = None,
    Qmin: int = 0,
    M0min: int = 13,
    M0max: int = float('inf'),
    max_Kperp_gcd: int = 4,
    ellipsoid_dilation: float = 1, # typically want >=1
    cut_Kprime: bool = True,
    use_box: bool = False,
    fp_recursive: bool = False,
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
        _0p = np.concatenate([[0],p])

        # construct the quadratic form defining the ellipsoid
        mat, Z, Binter = coniMellipsoid(_0p, data)

        # solve for lattice points maybe in tadpole
        # =========================================
        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        try:
            if not use_box:
                lattice_points = lattice.fp_ellipsoid(
                    mat=mat,
                    Q=ellipsoid_dilation*Qmax,
                    Q_lower=0,
                    linvec = Binter[0],
                    lindot_min = M0min,
                    lindot_max = M0max,
                    max_N_out=max_N_pfvs,
                    recursive=fp_recursive,
                    verbosity=verbosity-1
                )
            else:
                lattice_points = lattice.boundingbox_enumerate(
                    mat,
                    ellipsoid_dilation*Qmax,
                    max_N_out=max_N_pfvs)
        except Exception as e:
            print("ERROR!!!")
            print(f"LIKELY mat={mat.tolist()} ISN'T POSITIVE DEFINITE...")
            #print(f"vertices = {self.polytope().vertices().tolist()}")
            #print(f"heights  = {self.triangulation().heights().tolist()}")
            print(f"p        = {np.array(p).tolist()}")
            raise e

        if verbosity >= 2:
            print(f'# lattice_points = {lattice_points.shape[0]}')

        # compute/cut Ms
        # ==============
        # (process in chunks)
        chunk_size = 10_000_000
        chunk_num  = 0
        for chunk_start in range(0, lattice_points.shape[0], chunk_size):
            if lattice_points.shape[0] > chunk_size:
                print(f"Chunk #{chunk_num}..."); chunk_num += 1

            # compute Ms
            # ==========
            Ms = Binter @ lattice_points[chunk_start:chunk_start+chunk_size].T

            if verbosity >= 2:
                print(f'# M0s in [M0min, M0max] = {Ms.shape[1]}')

            # compute Ks
            # ==========
            Ks = Z@Ms

            # OPTIONAL: override the K[0] entry for clarity
            # only constraints on K[0] are
            #    - tadpole Qmin<=-dot(K,M)<=Qmax
            #    - K' > 0
            # we later set K[0] to all values obeying tadpole and then rejection
            # sample on K'>0
            natural_K0s = Ks[0].copy()
            Ks[0] = 0

            # remove GCDs (we later scan over GCDs...)
            K_gcds = np.gcd.reduce(Ks[1:,:], axis=0)
            Ks = Ks//K_gcds   
        
            # set K0s
            # (set to obey tadpole ranges)
            # ----------------------------
            bare_Ks = Ks
            bare_Ms = Ms
            bare_Qs = -np.sum(bare_Ks*bare_Ms, axis=0)
            
            Ks = np.zeros((data.h11,0), dtype=int)
            Ms = np.zeros((data.h11,0), dtype=int)
            if bare_Ks.shape[1]:
                for Kperp_gcd in range(1,max_Kperp_gcd+1):
                    # ranges for K0 to exactly hit tadpole
                    # ------------------------------------
                    Qperps = Kperp_gcd*bare_Qs
                    # Q             = Qperp - M[0]*K[0]
                    # Qmin         <= Qperp - M[0]*K[0] <= Qmax
                    # Qmin - Qperp <=       - M[0]*K[0] <= Qmax - Qperp
                    # Qperp - Qmin >=         M[0]*K[0] >= Qperp - Qmax
                    # if M[0] > 0:
                    #    (Qperp - Qmax)/M[0] <= K[0] <= (Qperp - Qmin)/M[0]
                    if M0min > 0:
                        lo = -(-(Qperps-Qmax)//bare_Ms[0]) # round lower bound upwards
                        up = (Qperps-Qmin)//bare_Ms[0]     # round upper bound downwards
                    else:
                        raise ValueError

                    # ranges for K0 to give K'>0
                    # --------------------------
                    # Kperp  = (natural Kperp) * Kperp_gcd/K_gcds
                    # K'     = -K[0] + (natural K)[0] * Kperp_gcd/K_gcds
                    # K' > 0 => K[0] < (natural K)[0] * Kperp_gcd/K_gcds
                    # (subtract 1e-4 to enforce K'>0, not K'>=0)
                    if cut_Kprime:
                        tmp = (natural_K0s*Kperp_gcd-1e-4)//K_gcds
                        up  = np.minimum(up, tmp.astype(int))

                    # compute the PFVs
                    # (any lo<=K0<=up should work...)
                    # ===============================
                    # get a mask for the (Kperp, M) pairs that have PFVs
                    num_K0s_perM  = 1 + up - lo
                    mask          = (num_K0s_perM > 0)

                    # compute the number of PFVs
                    num_K0s_perM  = num_K0s_perM[mask]  # trim the 0s...
                    total_pfvs    = np.sum(num_K0s_perM)

                    # fill K0 ranges
                    # --------------
                    # (think: K0s = lo + range(up))

                    # set K0s = lo
                    K0s  = np.repeat(lo[mask], num_K0s_perM)

                    # add range(up)
                    K0s += np.arange(total_pfvs)
                    K0s -= np.repeat(np.cumsum(num_K0s_perM) - num_K0s_perM, num_K0s_perM)

                    assert np.all(np.repeat(lo[mask], num_K0s_perM) <= K0s)
                    assert np.all(np.repeat(up[mask], num_K0s_perM) >= K0s)

                    # prepend the K0s to the Ks
                    # -------------------------
                    new_Ks = np.repeat(Kperp_gcd*bare_Ks[1:,mask], num_K0s_perM, axis=1)
                    new_Ks = np.vstack([K0s, new_Ks])

                    # get the Ms
                    new_Ms = np.repeat(bare_Ms[:,mask], num_K0s_perM, axis=1)

                    # save
                    # ====
                    Ks = np.hstack([Ks, new_Ks])
                    Ms = np.hstack([Ms, new_Ms])

            if verbosity >= 2:
                print(f'# PFVs after setting K0s = {Ms.shape[1]}')

            # filter by N invertibility
            # -------------------------
            batch_size = 5000
            singular = []
            for i in range(0, len(Ms), batch_size):
                chunk = Ms[i:i+batch_size]

                #Ns = np.tensordot(kappa, Ms, axes=([2], [0]))
                Ns = (kappa.reshape(data.h11*data.h11,data.h11)@Ms).reshape(data.h11,data.h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)
                Ns = Ns[:,1:,1:]

                #sign, logdet = np.linalg.slogdet(Ns)
                #is_zero = (sign == 0) | (logdet <= -1e-4)
                #singular.append(is_zero)
                singular.append(check_singular(Ns.astype(float)))

            singular = np.concatenate(singular)

            if verbosity >= 2:
                if not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

            Ms = Ms[:,~singular]
            Ks = Ks[:,~singular]

            if verbosity >= 2:
                print(f'# invertible = {Ms.shape[1]}')

            # transpose to row-wise
            Ks, Ms = Ks.T, Ms.T

            # save to data structures
            all_Ks        = np.vstack([all_Ks, Ks])
            all_Ms        = np.vstack([all_Ms, Ms])

    # return
    return all_Ks, all_Ms
