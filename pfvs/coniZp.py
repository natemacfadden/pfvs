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
# Description:  This module contains methods for constructing coni PFVs
#               using the "Zp" style algorithms. These operate by fixing some
#               p-vectors and then searching for lattice points in an ellipsoid,
#               one for each p-vector.
# -----------------------------------------------------------------------------

# external imports
import flint
import joblib
import math
import numba
import numpy as np
import os

from numpy.typing import ArrayLike

# local imports
from . import lattice, diagnostics
from .c_kernels import coni_kernel
from .cydata import CYData

# coniZp helpers
# ==============
def check_singular(Ns: ArrayLike, rtol: float=1e-12):
    # for a length-n stack of mxm matrices Ns (shape nxmxm), return a length-n
    # vector whose ith value is 1 iff Ns[i] is singular
    svals = np.linalg.svdvals(Ns)
    singular = svals[:,-1] <= rtol * svals[:,0]

    return singular

# we often compute projection matrices that project out 0th component
# these are used in matrix product, so mutability is not a concern
# compute these once and for all using global variabls
projs = [None]*100
def get_proj(dim):
    # Get a dim->(dim-1) projection matrix, projecting out the 0th component.
    if projs[dim] is None:
        projs[dim] = np.eye(dim, dtype=int)[1:,:]

    return projs[dim]

# very coni-specific helpers
def coniMellipsoid(p,
                   data=None,
                   kappa=None,
                   Mbasis=None,
                   extra_lll_reduction=True,
                   extra_checks=False):
    """
    **Description:**
    Compute the matrices defining the M-ellipsoid in coni-ZpM.

    **Arguments:**
    - `p`:    The relevant p-vector
    - `data`: The CYData describing the CY.

    **Returns:**
    mat, Z, Binter
    """
    if data is None:
        assert kappa is not None
        assert Mbasis is not None
        h11 = kappa.shape[0]
    else:
        assert kappa is None
        assert Mbasis is None
        h11    = data.h11
        kappa  = data.kappa_cob
        Mbasis = data.M_lattice()

    p = np.array(p).ravel()
    if len(p) == h11-1:
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
    orthog = lattice.orthogonal_lattice(p=T.T@Mbasis)
    if extra_lll_reduction:
        orthog = lattice.lll_reduce(orthog)
    Binter = Mbasis@orthog

    # lll-reduce Binter
    # (doesn't seem to have a huge effect...)
    Binter = lattice.lll_reduce(Binter)

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
        proj = get_proj(data.h11)
        #       Mperp-term          Kperp-term
        mat = -(Binter.T @ proj.T) @ (proj @ (kappa@([0]+p)) )@Binter

    if extra_checks:
        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)
    else:
        mat = np.rint(mat).astype(int)

    return mat, Z, Binter

def coniHmatrix(ZBinter, proj = None):
    if proj is None:
        proj = get_proj(ZBinter.shape[0])

    H    = proj@ZBinter
    H_fl = flint.fmpz_mat(H.tolist())

    H_list = H_fl.hnf().tolist()
    H = np.array([[int(x) for x in row] for row in H_list], dtype=object)

    return H

def Kperp_gcd_lattice(data, Z, Binter, gcd):
    # compute the matrix A such that Kperp = A@c
    # ------------------------------------------
    proj = get_proj(data.h11)
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

# coni Zp
# =======
print("IDK if K'>0 cut works for max_Kperp_gcd>1")
def coniZpM(
    # problem definition
    data: CYData,
    ps: ArrayLike,
    Qmax: int = None,
    M0min: int = 13,
    max_Kperp_gcd: int = 1,
    ellipsoid_dilation: float = 1, # typically want >=1
    # algorithm selection
    use_njit: bool = False,
    use_gcd_lattice: bool = False,
    low_level_parallelism: bool = False,
    n_jobs: int = -1,
    # misc
    extra_checks: bool = False,
    extra_lll_reduction: bool = True,
    # output/verbosity
    max_N_pfvs: int = 1_000_000_000,
    return_formal_pfvs: bool = False,
    verbosity: int = 0,
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

    if low_level_parallelism:
        assert n_jobs == 1
    if n_jobs == -1:
        n_jobs = 2*os.cpu_count()

    # misc
    only_positive_news = False

    # read data
    kappa  = data.kappa_cob
    h11    = data.h11
    h21    = data.h21
    proj   = get_proj(h11)

    kappa  = data.kappa_cob
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = (h11+h21+2) + 2

    # the search
    # ----------
    if use_njit:
        print("the c-code is 2x faster, or so ;) ",end="")
        pritn("Turn `use_njit=False` to try it")

    # iterate over p-vectors
    chunk_size = max(100, len(ps)//n_jobs+1)
    p_chunks   = [ps[i:i+chunk_size] for i in range(0,len(ps),chunk_size)]

    def make_pfvs(p_chunk, job_i=0):
        all_Ks = np.zeros((0,h11), dtype=np.int32)
        all_Ms = np.zeros((0,h11), dtype=np.int32)

        for p in p_chunk:
            _0p = np.concatenate([[0],p])

            # construct the quadratic form defining the ellipsoid
            mat, Z, Binter = coniMellipsoid(
                _0p,
                kappa=kappa,
                Mbasis=Mbasis,
                extra_lll_reduction=extra_lll_reduction,
                extra_checks=extra_checks)

            ZBinter = np.ascontiguousarray(Z@Binter)
            Binter  = np.ascontiguousarray(  Binter)

            # solve for lattice points maybe in tadpole
            # =========================================
            #lattice_points = rejection_ellipsoid(mat,tadpole_mult*Q)
            try:
                if not use_gcd_lattice:
                    # find relevant lattice points in ellipsoid c.T@mat@c <= Q
                    try:
                        H = coniHmatrix(ZBinter, proj)
                    except Exception as e:
                        print(f"C long error for p={p.tolist()} :(",flush=True)
                        print(e)
                        continue

                    try:
                        L = np.linalg.cholesky(mat)
                    except Exception as e:
                        print(f"couldn't compute cholesky decomposition of mat for p={p.tolist()} :(",flush=True)
                        print(e)
                        continue

                    if use_njit:
                        lattice_points, rawQs = lattice.coni_kernel_njit(
                            # ellipsoid definition
                            L=L,
                            Q=Qmax,
                            dilation=ellipsoid_dilation,
                            # M0 cuts:
                            Binter0=Binter[0,:],
                            M0min=M0min,
                            # K' cuts:
                            H = H.astype(int),
                            # misc:
                            max_N_out=max_N_pfvs)

                        if len(lattice_points) >= max_N_pfvs:
                            print(f"SATURATED (>={max_N_iter}) PFV COUNT",
                                  flush=True)
                            break
                    else:
                        try:
                            lattice_points, rawQs, status = coni_kernel(
                                U=np.ascontiguousarray(L.T),
                                Q=Qmax,
                                dilation=ellipsoid_dilation,
                                linvec=np.ascontiguousarray(Binter[0,:].astype(np.int32)),
                                linmin=M0min,
                                H=H,
                                max_N_out=max_N_pfvs,
                                eps=1e-4
                            )

                            if status != 0:
                                print(f"KERNEL RETURNED STATUS {status}!!!",
                                      flush=True)
                        except Exception as e:
                            raise e

                    if extra_checks and (not np.allclose(rawQs, np.sum(lattice_points*(lattice_points@mat.T),axis=1))):
                        print(lattice_points.tolist())
                        print(rawQs.tolist())
                        print(mat.tolist())
                        print(Qmax)
                        print(ellipsoid_dilation),
                        print(Binter[0,:].tolist()),
                        print(H.tolist())
                        print(max_N_pfvs)
                        print(np.linalg.cholesky(mat).T.dtype, Binter[0,:].astype(np.int32).dtype, H.dtype)
                        raise Exception

                # use GCD lattices
                # ----------------
                else:
                    print("LIKELY OLD CODE THAT COULD BE REFRESHED")
                    # i.e., encode gcd(Kperp) == val as a lattice
                    # scan in each lattice
                    lattice_points = np.empty((0,Binter.shape[1]), dtype=int)
                    rawQs          = np.empty((0,), dtype=int)

                    for gcd in range(1,ellipsoid_dilation+1):
                        Bgcd = Kperp_gcd_lattice(data, Z, Binter, gcd)

                        vs, vQs = lattice.fp_iterative_njit(
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
                if False:
                    cs     = np.array(lattice_points)
                    testQs = np.sum(cs * (cs@mat.T), axis=1)
                    if not all(rawQs == testQs):
                        print("PANIC!!!")

                if verbosity >= 1:
                    print(f"found {len(lattice_points)} lattice points...")
                    if verbosity >= 10:
                        print("they were:")
                        print(lattice_points)

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
                Ks = np.zeros((h11,0), dtype=np.int32)
                Ms = np.zeros((h11,0), dtype=np.int32)

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
                        tmp = np.floor((natural_K0s*Kperp_gcd-1e-4)/K_gcds)
                        tmp = tmp.astype(int)
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
                        K0s -= np.repeat(
                            np.cumsum(num_K0s_perM) - num_K0s_perM,
                            num_K0s_perM
                        )

                        # prepend the K0s to the Ks
                        # -------------------------
                        new_Ks = np.repeat(
                            Kperp_gcd*Kperps[1:,mask],
                            num_K0s_perM,
                            axis=1
                        )
                        new_Ks = np.vstack([K0s, new_Ks])

                        # get the Ms
                        Mperps = Binter[1:]@cs[:,mask]
                        new_M0s    = np.repeat(
                            M0s[mask].reshape(1,-1),
                            num_K0s_perM,
                            axis=1
                        )
                        new_Mperps = np.repeat(Mperps, num_K0s_perM, axis=1)
                        new_Ms = np.vstack([new_M0s, new_Mperps])

                        if not all(-np.sum(new_Ks*new_Ms, axis=0) <= Qmax):
                            inds = np.where(-np.sum(new_Ks*new_Ms,axis=0) >Qmax)
                            i = inds[0]

                            print("VIOLATED QMAX!!!")
                            print(new_Ks[:,i].T.tolist(), new_Ms[:,i].T.tolist())
                            print(-np.sum(new_Ks*new_Ms, axis=0)[i], Qmax)
                            print(np.repeat(lo[mask], num_K0s_perM)[i])
                            print(np.repeat(up[mask], num_K0s_perM)[i])
                            print(Qperps, Qmax, M0s)
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
                    Ns = (kappa.reshape(h11*h11,h11)@Ms).reshape(h11,h11,-1)
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

        print(f"Finished job #{job_i}...",flush=True)

        return all_Ks, all_Ms

    # actually run the jobs
    if n_jobs > 1:
        output = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(make_pfvs)(p_chunk, job_i) for job_i, p_chunk in enumerate(p_chunks))
        all_Ks, all_Ms = zip(*output)
        all_Ks = np.vstack(all_Ks)
        all_Ms = np.vstack(all_Ms)
    else:
        all_Ks, all_Ms = make_pfvs(ps)

    # return
    if return_formal_pfvs:
        return [diagnostics.PFV(data, K, M) for K,M in zip(all_Ks, all_Ms)]
    else:
        return all_Ks, all_Ms
