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
from . import lattice, diagnostics

# p-vectors
# =========
def pvecs(
    data: "CYData",
    min_N_pts: int = None,
    max_deg: int = None,
    min_deg: int = 0,
    deg_window: int = 1,
    max_window_i: int = 10_000,
    max_time: float = 120, # 2 min...
    verbosity: int = 0) -> "ArrayLike":
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
    - `data`:         The CYData describing the CY.
    - `min_N_pts`:    Grab at least this many p-vectors.
    - `max_deg`:      The maximum degree to compute p-vectors to.
    - `min_deg`:      The minimum permissible degree to allow p-vectors to have.
    - `deg_window`:   When setting `min_N_pts`, it operates by sliding a degree
                      window, grabbing all points in the window, and quitting
                      only once enough points have been generated. This
                      argument sets the width of the window.
    - `max_window_i`: The maximum number of windows to try if setting min_N_pts.
    - `max_time`:     The maximum amount of time (in seconds) per degree window
                      to search for p-vectors.
    - `verbosity`:    The verbosity level.

    **Returns:**
    Possible p-vectors.
    """
    # check inputs
    if (max_deg is None) ^ (min_N_pts is None):
        pass
    else:
        raise ValueError("Either `max_deg` or `min_N_pts` must be set...")

    # read hyperplanes
    if data.coni:
        H = data.H_cob
    else:
        H = data.H

    # requesting N points
    # -------------------
    # use shifting degree windows until we get enough points
    if min_N_pts is not None:
        ps = np.empty((0,H.shape[1]), dtype=int)
        N_ps = 0

        for window_i in tqdm(range(max_window_i+1)):
            _min = min_deg + (window_i+0)*deg_window + window_i
            _max = min_deg + (window_i+1)*deg_window + window_i

            if verbosity >= 1:
                print(f"Have {N_ps} p-vectors for degrees <{_min}", end=' '*20 + '\r')

            # compute new p-vectors, save them
            new_ps = pvecs(
                data,
                min_deg = _min,
                max_deg = _max,
                max_time = max_time)
            ps     = np.vstack([ps, new_ps])
            N_ps  += len(new_ps)

            # break if done
            if N_ps >= min_N_pts:
                break

        return ps

    # solving via min/max degree
    # --------------------------
    # compute a grading vector
    # (just needs to be in strict interior of dual cone)
    grading = H.sum(axis=0)
    grading = grading//np.gcd.reduce(grading)

    # define a constraint-programming model to solve
    model  = cp_model.CpModel()
    max_pi = cp_model.INT32_MAX - 1

    p_vars = [model.NewIntVar(-max_pi, max_pi, f'x{i}') for i in\
                                                        range(H.shape[1])]

    for row in H:
        model.Add(sum(int(row[j])*p_vars[j] for j in range(H.shape[1])) >= 1)
    model.Add(sum(int(grading[j])*p_vars[j] for j in range(H.shape[1])) >=\
                                                                    min_deg)
    model.Add(sum(int(grading[j])*p_vars[j] for j in range(H.shape[1])) <=\
                                                                    max_deg)

    # enumerate all solutions up to max_deg
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time

    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self, p_vars):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.p_vars = p_vars
            self.points = []

        def on_solution_callback(self):
            self.points.append([self.Value(v) for v in self.p_vars])

    printer = SolutionPrinter(p_vars)

    status = solver.SearchForAllSolutions(model, printer)
    if status == cp_model.OPTIMAL:
        pass
    elif status == cp_model.UNKNOWN:
        print("LIKELY HIT TIME LIMIT!!!\nLIKELY HIT TIME LIMIT!!!\nLIKELY HIT TIME LIMIT!!!")
        print(min_deg, max_deg)
    elif status != cp_model.INFEASIBLE:
        print("UNKNOWN ERROR\nUNKNOWN ERROR\nUNKNOWN ERROR")
        print(min_deg, max_deg, status)

    # return
    ps = np.array(printer.points)
    if len(ps) == 0:
        return np.empty((0,H.shape[1]), dtype=int)

    # reduce by GCD
    gcds = np.gcd.reduce(ps,axis=1)
    ps = ps[gcds==1]
    return ps.tolist()

def mindeg_pvec_gurobi(data: "CYData", verbosity: int = 0):
    """
    **Description:**
    Use gurobi to find the minimum degree integral p-vector.

    **Arguments:**
    - `data`:      The CYData describing the CY.
    - `verbosity`: The verbosity level.

    **Returns:**
    The minimum-degree p-vector
    """
    # import gurobi
    import gurobipy as gp

    # setup
    # -----
    # read hyperplanes
    if data.coni:
        H = data.H_cob
    else:
        H = data.H

    # compute a grading vector
    # (just needs to be in strict interior of dual cone)
    grading = H.sum(axis=0)
    grading = grading//np.gcd.reduce(grading)

    # define the model
    # ----------------
    model = gp.Model("pFinder")
    model.setParam('OutputFlag', (verbosity > 0))

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

    # optimize
    # --------
    model.setParam("PoolSolutions", 1)
    model.optimize()

    # return the p-vector
    model.setParam('SolutionNumber', 0)
    return np.rint(p.xn).astype(int)

# Zp helpers
# ==========
# generic
# -------
if False:
    # NOT RELIABLE/STABLE
    @numba.njit(parallel=True)
    def check_singular(Ns, tol=1e-4):
        n = Ns.shape[0]
        singular = np.zeros(n, dtype=np.bool_)
        for i in numba.prange(n):
            s, ld = np.linalg.slogdet(Ns[i])
            if s == 0.0 or ld <= -tol:
                singular[i] = True
        return singular
else:
    def check_singular(Ns, rtol=1e-12):
        svals = np.linalg.svdvals(Ns)
        singular = svals[:,-1] <= rtol * svals[:,0]

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
            lattice_points, _ = lattice.fp_ellipsoid(
                mat=mat,
                Q=ellipsoid_dilation*Qmax,
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
                singular.append(check_singular(Ns))

            singular = np.concatenate(singular)

            if verbosity >= 2:
                if len(singular) and not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

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
            lattice_points, _ = lattice.fp_ellipsoid(
                mat,
                ellipsoid_dilation*Qmax,
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
                singular.append(check_singular(Ns))

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
    """
    **Description:**
    Compute the matrices defining the M-ellipsoid in coni-ZpM.

    **Arguments:**
    - `p`:    The relevant p-vector
    - `data`: The CYData describing the CY.

    **Returns:**
    mat, Z, Binter
    """
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
    if True:
        #       M-term     K-term
        mat = -(Binter.T)@(Z@Binter)
    else:
        raise ValueError()
        # try to just enforce Qperp <= Qmax
        # ---------------
        # DOESN'T WORK!!!
        # ---------------
        # matrix to project out 0th component
        proj = np.eye(data.h11, dtype=np.int32)[1:,:]
        #       Mperp-term          Kperp-term
        mat = -(Binter.T @ proj.T) @ (proj @ (kappa@([0]+p)) )@Binter

    if np.allclose(mat, np.round(mat)):
        mat = np.rint(mat).astype(int)

    return mat, Z, Binter

def Kperp_gcd_lattice(data, Z, Binter, gcd):
    # compute the matrix A such that Kperp = A@c
    # ------------------------------------------
    proj = np.hstack([
        np.zeros((data.h11-1,1), dtype=int),
        np.eye(data.h11-1, dtype=int)
    ])
    A = proj@Z@Binter

    # compute the basis B such that (A @ (B@d)) % gcd == 0
    # ----------------------------------------------------
    # equiv: compute null lattice of [A, -gcd*identity]...
    #        first #A.shape[1] rows of null-lattice correspond to B...
    A_extended    = np.hstack([A, -gcd*np.eye(A.shape[0], dtype=int) ])
    A_extended_fl = flint.fmpz_mat(A_extended.tolist())

    # get the null lattice via HNF
    Ht, Tt = A_extended_fl.transpose().hnf(transform=True)
    H = Ht.transpose()
    T = Tt.transpose() # last ? columns of T correspond to null lattice

    # extract the data corresponding to null lattice
    # ----------------------------------------------
    # think: T[:T.nrows()//2, first_null_ind:] is the desired null lattice
    first_null_ind = 0
    for j in range(H.ncols()):
        for i in range(H.nrows()):
            if H[i,j] != 0:
                break
        else:
            first_null_ind = j
            break

    # extract the data
    null_fl = flint.fmpz_mat(T.nrows()//2, H.ncols()-first_null_ind)
    for i in range(null_fl.nrows()):
        for j in range(null_fl.ncols()):
            null_fl[i,j] = T[i,j+first_null_ind]

    # LLL transform and map to NumPy array
    null = np.array(null_fl.transpose().lll().transpose().tolist()).astype(int)
    assert np.all((A@null % gcd) == 0)

    # sort null to maximize leading 0s in (Binter@null)[0]
    sort_inds = np.argsort((Binter@null)[0]!=0)
    null = null[:,sort_inds]

    # return
    # ------
    return null

@numba.njit(parallel=True, fastmath=False)
def gcd_of_matmul(A, C):
    """
    Computes gcd(A@C, axis=0)

    Gives better performance than NumPy, but uses parallelism
    (if you want to parallelize at a higher level, maybe not
    the best idea...)
    """
    k, N = C.shape
    out  = np.empty(N, dtype=np.int32)
    for j in numba.prange(N):
        g = 0
        for i in range(k):
            s = 0
            for t in range(k):
                s += A[i, t] * C[t, j]
            g = math.gcd(g, s)
        out[j] = g
    return out

print("IDK if K'>0 cut works for max_Kperp_gcd>1")
def coniZpM(
    data: "cydata",
    ps: "ArrayLike",
    Qmax: int = None,
    M0min: int = 13,
    max_Kperp_gcd: int = 4,
    ellipsoid_dilation: float = 1, # typically want >=1
    use_box: bool = False,
    use_gcd_lattice: bool = False,
    max_N_pfvs: int = 1_000_000_000,
    verbosity: int = 0,
    low_level_parallelism: bool = True,
    return_formal_pfvs: bool = False,
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
    all_Ks = np.zeros((0,data.h11), dtype=np.int32)
    all_Ms = np.zeros((0,data.h11), dtype=np.int32)

    # iterate over p-vectors
    if verbosity >= 0:
        iterator = tqdm(ps, mininterval=2.0, maxinterval=5.0)
    else:
        iterator = ps
    for _i, p in enumerate(iterator):
        _0p = np.concatenate([[0],p])

        # construct the quadratic form defining the ellipsoid
        mat, Z, Binter = coniMellipsoid(_0p, data)

        ZBinter = np.ascontiguousarray(Z@Binter)
        Binter  = np.ascontiguousarray(  Binter)

        # solve for lattice points maybe in tadpole
        # =========================================
        #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
        try:
            # BOX METHOD
            # ==========
            if use_box:
                print("NOT RECOMMENDED!!!")
                lattice_points = lattice.boundingbox_enumerate(
                    mat,
                    ellipsoid_dilation*Qmax,
                    max_N_out=max_N_pfvs)

            # ELLIPSOIDAL METHODS
            # ===================
            else:

                # do the computation in the standard way...
                 # -----------------------------------------
                if not use_gcd_lattice:
                    # find relevant lattice points in ellipsoid c.T@mat@c <= Q
                    proj = np.hstack([
                        np.zeros((data.h11-1, 1), dtype=int),
                        np.eye(data.h11-1, dtype=int)
                    ])
                    G    = proj@ZBinter
                    G_fl = flint.fmpz_mat(G.tolist())

                    try:
                        H    = np.array(G_fl.hnf().tolist()).astype(int)
                    except Exception as e:
                        print("C long error :(")
                        raise e

                    lattice_points, rawQs = lattice.coni_kernel(
                        # ellipsoid definition
                        L=np.linalg.cholesky(mat),
                        Q=Qmax,
                        dilation=ellipsoid_dilation,
                        # M0 cuts:
                        Binter0=Binter[0,:],
                        M0min=M0min,
                        # K' cuts:
                        H = H,
                        # misc:
                        max_N_out=max_N_pfvs)
                
                # use GCD lattices
                # ----------------
                else:
                    # i.e., encode gcd(Kperp) == val as a lattice
                    # scan in each lattice
                    lattice_points = np.empty((0,Binter.shape[1]), dtype=int)
                    rawQs          = np.empty((0,), dtype=int)

                    for gcd in range(1,ellipsoid_dilation+1):
                        Bgcd = Kperp_gcd_lattice(data, Z, Binter, gcd)
                        
                        vs, vQs = lattice.fp_iterative_lincut(
                            # ellipsoid definition
                            L=np.linalg.cholesky(Bgcd.T@mat@Bgcd),
                            Q=gcd*Qmax,
                            # M0 cuts:
                            linvec = (Binter@Bgcd)[0],
                            linmin = 13,
                            # misc:
                            max_N_out=max_N_pfvs)

                        # concatenate
                        lattice_points = np.vstack([
                            lattice_points,
                            vs@Bgcd.T
                        ])
                        rawQs = np.concatenate([rawQs, vQs])

                # clean Qs
                # --------
                rawQs = np.rint(rawQs).astype(int)

                if verbosity >= 1:
                    print(f"found {len(lattice_points)} lattice points...")

        except Exception as e:
            print("ERROR!!!")
            print(f"LIKELY mat={mat.tolist()} ISN'T POSITIVE DEFINITE...")
            #print(f"vertices = {self.polytope().vertices().tolist()}")
            #print(f"heights  = {self.triangulation().heights().tolist()}")
            print(f"p        = {np.array(p).tolist()}")
            raise e

        lattice_points = lattice_points.T
        #lattice_points = np.ascontiguousarray(lattice_points)

        # compute/cut Ms
        # ==============
        # (process in chunks)
        chunk_size = 10_000_000
        chunk_num  = 0
        for chunk_start in range(0, lattice_points.shape[1], chunk_size):
            if (verbosity >= 1) and (lattice_points.shape[1] > chunk_size):
                print(f"Chunk #{chunk_num}..."); chunk_num += 1

            # read data from fp_ellipsoid
            # ---------------------------
            cs  = lattice_points[:,chunk_start:chunk_start+chunk_size]
            Qs  = rawQs[chunk_start:chunk_start+chunk_size]
            
            # cut on feasibility of finding a K0 giving K'>0
            # ----------------------------------------------
            # for 0 <= K_scaling <= 1
            # for Qperp = -dot(Knat[1:],M[1:])
            #
            # (A)
            # K' = -K[0] * Knat[0]*K_scaling
            # K' > 0  <=> K[0] < Knat[0]*K_scaling
            #
            # (B)
            # Q = -K[0]*M[0] + Qperp*K_scaling
            # K[0] = (Qperp*K_scaling) - Q)/M[0]
            #
            # (B into LHS of A)
            # (Qperp*K_scaling) - Q)/M[0] < Knat[0]*K_scaling
            # ...
            # Qperp < M[0]*Knat[0] + Q/Kscaling
            #
            # (more obvious format)
            # -M[1:]^T@Knat[1:] - M[0]*Knat[0] < Q/Kscaling
            # -M.T @ Knat < Q/Kscaling
            # -c.T @ Binter.T @ Z @ Binter @ c < Q/Kscaling
            # DOH!... same matrix!
            if low_level_parallelism:
                K_gcds = gcd_of_matmul(ZBinter[1:], cs)
            else:
                Kperps = ZBinter[1:]@cs
                K_gcds = np.gcd.reduce(Kperps, axis=0)

            # cure cases where K_gcd = 0
            # think this should just occur if K = (x!=0,0,...,0)
            K_gcds[K_gcds<1] = 1

            mask   = Qs/K_gcds < Qmax
            cs     = cs[:,mask]
            Qs     = Qs[mask]
            K_gcds = K_gcds[mask]

            if low_level_parallelism:
                Kperps = ZBinter[1:]@cs
            else:
                Kperps = Kperps[:,mask]

            # compute Ks
            # ----------
            natural_K0s = ZBinter[0]@cs
            Ks          = np.vstack(
                [np.zeros((1,Kperps.shape[1]),dtype=np.int32),
                Kperps
            ])
            Kperps      = Ks//K_gcds

            # compute M0s
            # -----------
            #Ms  = Binter @ cs
            M0s = Binter[0] @ cs

            # Q considerations
            # ----------------
            # subtract the K[0]*M[0] contribution
            rawQperps = Qs + M0s*natural_K0s
            rawQperps = rawQperps//K_gcds

            # set K0s
            # (set to obey tadpole ranges, K'>0)
            # ----------------------------------
            Ks = np.zeros((data.h11,0), dtype=np.int32)
            Ms = np.zeros((data.h11,0), dtype=np.int32)

            if Kperps.shape[1]:
                for Kperp_gcd in range(1,max_Kperp_gcd+1):
                    # ranges for K0 to exactly hit tadpole
                    # ------------------------------------
                    Qperps = Kperp_gcd*rawQperps
                    # Q             = Qperp - M[0]*K[0]
                    # Qmin         <= Qperp - M[0]*K[0] <= Qmax
                    # Qmin - Qperp <=       - M[0]*K[0] <= Qmax - Qperp
                    # Qperp - Qmin >=         M[0]*K[0] >= Qperp - Qmax
                    # if M[0] > 0:
                    #    (Qperp - Qmax)/M[0] <= K[0] <= (Qperp - Qmin)/M[0]
                    if M0min > 0:
                        #lo = -(-(Qperps-Qmax)//M0s) # round lower bound upwards
                        #up = (Qperps-Qmin)//M0s     # round upper bound downwards
                        lo = np.ceil(( Qperps - Qmax)/M0s).astype(int)
                        up = np.floor((Qperps - Qmax)/M0s).astype(int)
                    else:
                        raise ValueError

                    # ranges for K0 to give K'>0
                    # --------------------------
                    # Kperp  = (natural Kperp) * Kperp_gcd/K_gcds
                    # K'     = -K[0] + (natural K)[0] * Kperp_gcd/K_gcds
                    # K' > 0 => K[0] < (natural K)[0] * Kperp_gcd/K_gcds
                    # (subtract 1e-4 to enforce K'>0, not K'>=0)
                    tmp = np.floor((natural_K0s*Kperp_gcd-1e-4)/K_gcds).astype(int)
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

                    # prepend the K0s to the Ks
                    # -------------------------
                    new_Ks = np.repeat(Kperp_gcd*Kperps[1:,mask], num_K0s_perM, axis=1)
                    new_Ks = np.vstack([K0s, new_Ks])

                    # get the Ms
                    Mperps = Binter[1:]@cs[:,mask]
                    new_M0s    = np.repeat(M0s[mask].reshape(1,-1), num_K0s_perM, axis=1)
                    new_Mperps = np.repeat(Mperps, num_K0s_perM, axis=1)
                    new_Ms = np.vstack([new_M0s, new_Mperps])

                    if not all(-np.sum(new_Ks*new_Ms, axis=0) <= Qmax):
                        inds = np.where(-np.sum(new_Ks*new_Ms, axis=0) > Qmax)
                        i = inds[0]

                        print(new_Ks[:,i].T.tolist(), new_Ms[:,i].T.tolist())
                        print(-np.sum(new_Ks*new_Ms, axis=0)[i], Qmax)
                        print(np.repeat(lo[mask], num_K0s_perM)[i])
                        print(np.repeat(up[mask], num_K0s_perM)[i])
                        raise ValueError()

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
                singular.append(check_singular(Ns))

            singular = np.concatenate(singular)

            if verbosity >= 2:
                if len(singular) and not only_positive_news:
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
    if return_formal_pfvs:
        return [diagnostics.PFV(data, K, M) for K,M in zip(all_Ks, all_Ms)]
    else:
        return all_Ks, all_Ms
