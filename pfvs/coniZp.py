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
# Description:  This module contains methods for constructing coniPFVs using the
#               "Zp" style algorithms. These operate by fixing some p-vectors
#               and then searching for lattice points in an ellipsoid, one for
#               each p-vector.
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
from . import util
from .c_kernels import coni_kernel
from .cydata import CYData

# coniZp helpers
# ==============
def _check_singular(Ns: ArrayLike, rtol: float = 1e-12) -> np.ndarray:
    """
    Check which matrices in a stack are singular.

    Parameters
    ----------
    Ns : ArrayLike, shape (n, m, m)
        Stack of n square matrices.
    rtol : float, optional
        Relative tolerance: matrix is singular if
        sval_min <= rtol * sval_max.

    Returns
    -------
    np.ndarray, shape (n,), dtype bool
        True at index i iff Ns[i] is singular.
    """
    svals    = np.linalg.svdvals(Ns)
    singular = (svals[:,-1] <= rtol * svals[:,0])

    return singular

# we often compute projection matrices that project out 0th component
# these are only used in matrix product, so mutability is not a concern
# compute these once and for all using global variables
projs = [None]*100
def _get_proj(dim: int) -> np.ndarray:
    """
    Return a (dim-1) x dim projection matrix that drops the 0th component.

    Results are cached in the module-level `projs` list.

    Parameters
    ----------
    dim : int
        Dimension of the input space.

    Returns
    -------
    np.ndarray, shape (dim-1, dim), dtype int64
        The projection matrix eye(dim)[1:, :].
    """
    if projs[dim] is None:
        projs[dim] = np.eye(dim, dtype=np.int64)[1:,:]

    return projs[dim]

@numba.njit(parallel=True, fastmath=False)
def _gcd_of_matmul(A, C):
    """
    Compute the column-wise GCD of the matrix product A @ C.

    Equivalent to np.gcd.reduce(A @ C, axis=0), but faster due to Numba
    parallelism. Note: if parallelizing at a higher level, the internal
    parallelism here may be counterproductive.

    Parameters
    ----------
    A : ArrayLike, shape (k, k)
        Left matrix factor.
    C : ArrayLike, shape (k, N)
        Right matrix factor (columns are vectors).

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
        GCD of each column of A @ C.
    """
    k, N = C.shape
    out  = np.empty(N, dtype=np.int64)
    for j in numba.prange(N):
        g = 0
        for i in range(k):
            s = 0
            for t in range(k):
                s += A[i, t] * C[t, j]
            g = math.gcd(g, s)
        out[j] = g
    return out

# very coni-specific helpers
# --------------------------
def coni_M_ellipsoid(p: ArrayLike,
                   data: CYData = None,
                   kappa: ArrayLike = None,
                   Mbasis: ArrayLike = None,
                   extra_lll_reduction: bool = True,
                   extra_checks: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the matrices defining the M-ellipsoid in coni-ZpM.

    In brief detail,
        - M lives in a lattice M = Binter@c
        - the component of K perpendicular to the conifold curve can be computed
          as Kperp = (Z@M)[1:] for Z = kappa@p
        - the parallel component of K (i.e., K[0]) is unconstrained, other than
          K[0] > 0 (from the physics)
        - one can show (see `coniZpM`) a K[0]>0 exists s.t. -dot(K,M) <= Qmax
          iff -c^T @ Binter^T @ Z @ Binter @ c <= Qmax. Define
          mat = -Binter^T @ Z @ Binter.
    That last constraint c^T @ mat @ c <= Qmax is the ellipsoid constraint. One
    can actually dilate the ellipsoid as long as
    GCD(Kperp) > (c^T @ mat @ c)/Qmax - see the 'cut on feasibility of finding a
    K0 giving K'>0' section of `coniZpM`.

    Parameters
    ----------
    p : ndarray of shape (h11,) or (h11-1,)
        The p-vector.
    data : CYData, optional
        The relevant data from the associated CY. Mutually exclusive with kappa
        and Mbasis.
    kappa : ndarray of shape (h11, h11, h11), optional
        The triple intersection numbers of the CY. Mutually exclusive with data.
        If provided, it is assumed that Mbasis is also provided.
    Mbasis : ndarray of shape (h11, h11), optional
        The lattice basis for M-vectors. Mutually exclusive with data.
        If provided, it is assumed that kappa is also provided.
    extra_lll_reduction : bool, optional
        Whether to perform an extra (technically unnecessary) LLL reduction on
        the updated M vector lattice basis, Binter. Useful since otherwise
        there are sometimes overflows. Defaults to True.
    extra_checks : bool, optional
        Whether to check that the defining matrix of the output ellipsoid, mat,
        is actually integer before casting it to int. This check has never
        failed and can actually add non-negligible timing, so it defaults to
        False.

    Returns
    -------
    mat : ndarray of shape (h11-1, h11-1)
        The matrix defining the ellipsoid. I.e., c^T @ mat @ c <= Qmax. We
        typically dilate this ellipsoid via
        c^T @ mat @ c <= ellipsoid_dilation * Qmax
    Z : ndarray of shape (h11, h11)
        The matrix relating M and K. Specifically, K[1:] = (Z@M)[1:]
    Binter : ndarray of shape (h11, h11-1)
        Updated M-vector lattice basis, integrating the dot(K,p)=0 constraint.
    """
    if data is None:
        if kappa is None or Mbasis is None:
            raise ValueError("If data is None, both kappa and Mbasis must be provided.")
        h11 = kappa.shape[0]
    else:
        if kappa is not None or Mbasis is not None:
            raise ValueError("kappa and Mbasis must be None when data is provided.")
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
    # need dot(K,p) = 0
    #
    # note that K[1:] = (kappa @ M @ p)[1:]
    # (K[0] is unconstrained so (kappa @ M @ p)[0] is semi-meaningless)
    #
    # thus need dot(p, kappa @ M @ p) = 0
    # equivalently, dot((kappa @ p) @ p, M) = 0
    T = kappa@p@p

    # need T^T @ Mbasis @ c = 0
    # thus just need c in the orthogonal lattice to Mbasis^T @ T
    # (the output will be lattice generators of such cs... we'll want
    #  lattice generators of valid Ms so we multiply on left by Mbasis)
    orthog = util.orthogonal_lattice(p=T.T@Mbasis)
    if extra_lll_reduction:
        orthog = util.lll_reduce(orthog)
    Binter = Mbasis@orthog

    # lll-reduce Binter
    # (doesn't seem to have a huge effect...)
    Binter = util.lll_reduce(Binter)

    # sort Binter so columns which don't affect M0 come first
    Binter = Binter[:,np.argsort(Binter[0]!=0)]

    # define ellipsoid
    #       M-term     K-term
    mat = -(Binter.T)@(Z@Binter)

    if extra_checks:
        if np.allclose(mat, np.round(mat)):
            mat = np.rint(mat).astype(int)
        else:
            raise ValueError
    else:
        mat = np.rint(mat).astype(int)

    return mat, Z, Binter

def coni_H_matrix(ZBinter: ArrayLike, proj: ArrayLike = None):
    """
    Compute the H-matrix for use in coni-ZpM. This is the HNF of (Z@Binter)[1:].

    This is the preferred approach for enforcing the GCD(K[1:]) cut in
    `coniZpM`. The alternative lattice-based approach is `_Kperp_gcd_lattice`
    (not recommended in practice).

    In coni-ZpM, one wants to ensure GCD(K[1:]) is sufficiently large. A point
    c in the M-ellipsoid has an associated valuation c^T @ mat @ c. For dilated
    ellipsoids, this can have c^T @ mat @ c > Qmax. This would give rise to a K
    and M which violates tadpole (i.e., -dot(K,M) > Qmax) unless
        GCD(K[1:]) > (c^T @ mat @ c)/Qmax,
    in which case one can divide both p and K by GCD(K[1:]) to bring the
    solution back under tadpole. The strict inequality is correct but
    unintuitive. See the 'cut on feasibility of finding a K0 giving K'>0'
    section of `coniZpM`.

    Recall that
        1 The M-vector is built incrementally via the relationship M = Binter c,
          using a modified Fincke-Pohst algorithm.
        2 K[1:] = (Z @ Binter @ c)[1:]
    Naively, one would have to fully set c before checking GCD(K[1:]).

    A trick, though:
        FP sets c from right to left, beginning with c[-1], then c[-2], etc.

        This uses the fact that FP provides a monotonically increasing lower
        bound on  c^T @ mat @ c as further components of c are set.

        Similarly, since H is upper triangular, H[-m:,-m:] @ c[-m:] is a
        monotonically decreasing upper bound on
            GCD(H@c) = GCD((Z@Binter)[1:,:] @ c) = GCD(K[1:]).
        This is because (H@c)[-m:] = H[-m:,-m:]@c[-m:] and
        GCD((H@c)[-m:]) >= GCD((H@c)[-n:]) for m<n.

        Thus, during FP, one can check if the current upper bound on the GCD
        is sufficiently large compared to the current lower bound on the
        valuation. If not, then one can immediately prune the current branch.

    Parameters
    ----------
    ZBinter : ndarray of shape (h11,h11-1)
        The product of matrices Z and Binter from coni_M_ellipsoid. Has
        interpretation that K[1:] = (ZBinter c)[1:].
    proj : ndarray of shape (h11-1,h11)
        An optional projection matrix, since we want the HNF of (Z Binter)[1:].
        This is trivial: identity(h11)[1:,:]. If not provided, then it's
        computed using `_get_proj`.

    Returns
    -------
    H : ndarray of shape (h11-1, h11-1)
        The HNF (Z@Binter)[1:]. Has interpretation that GCD(H@c) = GCD(K[1:])
        and that GCD(H[-m:,-m:]@c[-m:]) >= GCD(H[-n:,-n:]@c[-n:]) for m<n.
    """
    if proj is None:
        proj = _get_proj(ZBinter.shape[0])

    H    = proj@ZBinter
    H_fl = flint.fmpz_mat(H.tolist())

    H_list = H_fl.hnf().tolist()
    H      = np.array([[int(x) for x in row] for row in H_list], dtype=object)

    return H

def _Kperp_gcd_lattice(data: CYData, Z: ArrayLike, Binter: ArrayLike, gcd: int):
    """
    (Not recommended in practice - just prune FP using `coni_H_matrix`)

    When finding c in the `coni_M_ellipsoid`, one wants to guarantee that Kperp
    has sufficiently large GCD (see `coni_H_matrix`). The collection of c giving
    rise to GCD(Kperp) = g (or integer multiples of it) forms a lattice. This
    function computes a basis of that lattice.

    This enables scans over different lattice bases without having to explicitly
    check the GCD using, e.g., the early-pruning in FP.

    In practice, the majority of the cost in coniPFV enumeration is actually in
    lattice generation, not the FP, so this is not recommended (since it just
    adds more lattice generation).

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY.
    Z : ndarray of shape (h11, h11)
        The matrix relating M and K. Specifically, K[1:] = (Z@M)[1:]
    Binter : ndarray of shape (h11, h11-1)
        Updated M-vector lattice basis, integrating the dot(K,p)=0 constraint.
    gcd : integer
        The imposed gcd for which we return a lattice basis.

    Returns
    -------
    Bgcd : ndarray of shape (h11-1, h11-1)
        Basis vectors (as columns) satisfy A @ Bgcd % gcd = 0 where
        A = proj @ Z @ Binter.
    """
    # compute the matrix A such that Kperp = A@c
    # ------------------------------------------
    proj = _get_proj(data.h11)
    A    = proj@Z@Binter

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
    # here we find the first column that's all 0
    first_null_ind = None
    for j in range(H.ncols()):
        for i in range(H.nrows()):
            if H[i,j] != 0:
                break
        else:
            first_null_ind = j
            break
    if first_null_ind is None:
        raise ValueError

    # extract the data
    null_fl = flint.fmpz_mat(T.nrows()//2, H.ncols()-first_null_ind)
    for i in range(null_fl.nrows()):
        for j in range(null_fl.ncols()):
            null_fl[i,j] = T[i,j+first_null_ind]

    # LLL transform and map to NumPy array
    null = np.array(null_fl.transpose().lll().transpose().tolist()).astype(int)
    if not np.all((A@null % gcd) == 0):
        raise RuntimeError("_Kperp_gcd_lattice: computed null lattice does not satisfy A@null % gcd == 0.")

    # sort null to maximize leading 0s in (Binter@null)[0]
    sort_inds = np.argsort((Binter@null)[0]!=0)
    null = null[:,sort_inds]

    # return
    # ------
    return null

# coni Zp
# =======
def coniZpM(
    # problem definition
    data: CYData,
    ps: ArrayLike,
    Q: int | None = None,
    M0min: int = 13,
    max_Kperp_gcd: int = 1,
    ellipsoid_dilation: float = 1, # typically want >=1
    # algorithm selection
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
    ) -> tuple[ArrayLike, ArrayLike]:
    """
    A 'Zp' implementation that computes coniPFVs from input integer p-vectors.

    The logic is
        1 an integer p-vector defines a certain ellipsoid (see `coni_M_ellipsoid`)
        2 a lattice point c in this ellipsoid defines an M-vector via Binter@c.
          this also defines (most of) a K-vector via K[1:] = (Z@Binter@c)[1:]
    so one wants to enumerate such c-vectors. This is done via Fincke-Pohst.

    As discussed in `coni_M_ellipsoid` and `coni_H_matrix`, this ellipsoid can be
    dilated, but then only c vectors that give rise to K[1:] with sufficiently
    large GCD are allowed. This is integrated into the Fincke-Pohst solver via
    the H-matrix from `coni_H_matrix`. An alternative lattice-based approach is
    available via `_Kperp_gcd_lattice` (controlled by `use_gcd_lattice`), but
    is not recommended.

    Likewise, one can impose constraints on M[0] >= 13 early in FP by ordering
    the columns of the M-vector lattice basis such that the first row of this
    basis (that corresponding to M[0]) has a maximal number of leading 0s.

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY.
    ps : iterable of shape (N, h11-1)
        Each row of the iterable corresponds to the perpendicular component of a
        p-vector. I.e., p[1:]
    Q : integer, optional
        Only return PFVs with -dot(K,M) = Q (exact equality, unlike the
        ``Qmin``/``Qmax`` range in non-coni ``ZpM``/``ZpK``). If not
        provided, set to h11+h21+4.
    M0min : integer, optional
        Only return PFVs with M[0] >= M0min. Defaults to 13 to match physics.
    max_Kperp_gcd : integer, optional
        When solving for PFVs, one hardcodes the GCD of Kperp (since we
        previously cleared the GCD, making Kperp primitive). Allow GCDs up to
        this value. **Not well tested - defaults to 1.** Leave at default.
    ellipsoid_dilation : float, optional
        The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
        Empirically, runtime scales linearly with this value. Defaults to 1.
    use_gcd_lattice : bool, optional
        Whether to construct explicit lattice bases for guaranteeing sufficient
        GCD of Kperp. Not recommended - it's generally quicker to just prune FP.
        Defaults to False.
    low_level_parallelism : bool, optional
        Allow certain low-level methods to be parallelized. Not generally
        recommended since, typically, one introduces parallelism at the p-by-p
        level (since this is embarrassingly parallel). Defaults to False.
    n_jobs : int, optional
        How many jobs to spawn if not doing low-level parallelism. Defaults to
        twice the CPU count.
    extra_checks : bool, optional
        Whether to do extra sanity checks in the ellipsoid generation. Never
        seen these fail so defaults to False.
    extra_lll_reduction : bool, optional
        Whether to perform an extra (technically unnecessary) LLL reduction on
        the updated M vector lattice basis, Binter. Useful since otherwise
        there are sometimes overflows. Defaults to True.
    max_N_pfvs : int, optional
        The maximum number of PFVs that can be output. The C-kernel requires a
        limit. Defaults excessively high to 1,000,000,000.
    return_formal_pfvs : bool, optional
        Whether to return "PFV" objects as in pfv.py. Otherwise, an
        array of K-vectors (as rows) and an array of M-vectors (as rows) are
        returned. Defaults to False.
    verbosity : int, optional
        The verbosity level. Higher is more verbose. Defaults to 0.

    Returns
    -------
    Ks : ndarray of shape (N, h11)
      K-vectors of the PFVs, one per row. Only returned if
      return_formal_pfvs=False.
    Ms : ndarray of shape (N, h11)
        M-vectors of the PFVs, one per row. Only returned if
        return_formal_pfvs=False.
    pfvs : list of length N
         PFV objects (see ``pfv.PFV``). Only returned if
         return_formal_pfvs=True.
    """
    if not data.coni:
        raise ValueError(
            "coniZpM only applies to coni contexts. "
            "Use Zp.py for non-coni PFVs."
        )
    if len(ps) == 0:
        raise ValueError("ps must be non-empty.")
    if ellipsoid_dilation <= 0:
        raise ValueError(f"ellipsoid_dilation must be > 0, got {ellipsoid_dilation}.")

    if low_level_parallelism:
        if n_jobs != 1:
            print("Setting n_jobs = 1 since low_level_parallelism = True...")
            n_jobs = 1
    if n_jobs == -1:
        n_jobs = 2*os.cpu_count()

    if max_Kperp_gcd > 1:
        print("WARNING This code hasn't been well-tested for max_Kperp_gcd > 1...")

    # misc (left for future debugging)
    only_positive_news = False

    # read data
    kappa  = data.kappa_cob
    h11    = data.h11
    h21    = data.h21
    proj   = _get_proj(h11)
    Mbasis = data.M_lattice()

    if Q is None:
        Q = (h11+h21+2) + 2

    # the search
    # ----------
    # iterate over p-vectors
    chunk_size = max(100, len(ps)//n_jobs+1)
    p_chunks   = [ps[i:i+chunk_size] for i in range(0,len(ps),chunk_size)]

    def _make_pfvs(p_chunk, job_i=0):
        # define a factory function here for later parallelization
        all_Ks = np.zeros((0,h11), dtype=np.int32)
        all_Ms = np.zeros((0,h11), dtype=np.int32)

        for p in p_chunk:
            p_full = np.concatenate([[0],p])

            # construct the quadratic form defining the ellipsoid
            mat, Z, Binter = coni_M_ellipsoid(
                p_full,
                kappa=kappa,
                Mbasis=Mbasis,
                extra_lll_reduction=extra_lll_reduction,
                extra_checks=extra_checks)

            ZBinter = np.ascontiguousarray(Z@Binter)
            Binter  = np.ascontiguousarray(  Binter)

            # solve for lattice points under tadpole
            # ======================================
            try:
                if not use_gcd_lattice:
                    # find relevant lattice points in ellipsoid c.T@mat@c <= Q
                    # just uses FP with pruning on GCDs and M0 - no GCD lattice
                    try:
                        H = coni_H_matrix(ZBinter, proj)
                    except Exception as e:
                        print(f"C long error for p={p.tolist()} :(",flush=True)
                        print(e)
                        continue

                    try:
                        L = np.linalg.cholesky(mat)
                    except Exception as e:
                        print(f"couldn't compute cholesky decomposition of "
                              f"mat for p={p.tolist()} :(", flush=True)
                        print(e)
                        continue

                    lattice_points, rawQs, status = coni_kernel(
                        # ellipsoid definition
                        U=np.ascontiguousarray(L.T),
                        Q=Q,
                        dilation=ellipsoid_dilation,
                        # M0 cuts
                        linvec=np.ascontiguousarray(Binter[0,:].astype(np.int32)),
                        linmin=M0min,
                        # gcd cuts
                        H=H,
                        # misc
                        max_N_out=max_N_pfvs,
                        eps=1e-4
                    )

                    if status != 0:
                        print(f"KERNEL RETURNED STATUS {status}!!!",
                              flush=True)

                # use GCD lattices
                # ----------------
                else:
                    print("WARNING LIKELY OLD CODE THAT COULD BE REFRESHED")
                    # i.e., encode gcd(Kperp) == val as a lattice
                    # scan in each lattice
                    lattice_points = np.empty((0,Binter.shape[1]), dtype=int)
                    rawQs          = np.empty((0,), dtype=int)

                    for gcd in range(1,np.ceil(ellipsoid_dilation)+1):
                        Bgcd = _Kperp_gcd_lattice(data, Z, Binter, gcd)

                        vs, vQs = util.fp_iterative_njit(
                            # ellipsoid definition
                            L=np.linalg.cholesky(Bgcd.T@mat@Bgcd),
                            Q=Q,
                            dilation=gcd,
                            # M0 cuts
                            linvec = (Binter@Bgcd)[0],
                            linmin = 13,
                            # misc
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
                    if verbosity >= 10:
                        print("they were:")
                        print(lattice_points)

            except Exception as e:
                raise RuntimeError(
                    f"Kernel failed for p={np.array(p).tolist()}. "
                    f"mat may not be positive definite: mat={mat.tolist()}"
                ) from e

            lattice_points = lattice_points.T

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
                # K' = -K[0] + Knat[0]*K_scaling
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
                    K_gcds = _gcd_of_matmul(ZBinter[1:], cs)
                else:
                    Kperps = ZBinter[1:]@cs
                    K_gcds = np.gcd.reduce(Kperps, axis=0)

                # cure cases where K_gcd = 0
                # think this should just occur if K = (x!=0,0,...,0)
                K_gcds[K_gcds<1] = 1

                mask   = Qs < Q * K_gcds
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
                            lo = np.ceil(( Qperps - Q)/M0s).astype(int)
                            up = np.floor((Qperps - Q)/M0s).astype(int)
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

                        if not all(-np.sum(new_Ks*new_Ms, axis=0) <= Q):
                            inds = np.where(-np.sum(new_Ks*new_Ms,axis=0) > Q)
                            i = inds[0]

                            tadpole = -np.sum(new_Ks*new_Ms, axis=0)[i]
                            raise RuntimeError(
                                f"Tadpole violation: -dot(K,M)={tadpole} "
                                f"> Q={Q}. K={new_Ks[:,i].T.tolist()}, "
                                f"M={new_Ms[:,i].T.tolist()}"
                            )

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
                for i in range(0, Ms.shape[1], batch_size):
                    chunk = Ms[:,i:i+batch_size]

                    Ns = (kappa.reshape(h11*h11,h11)@chunk).reshape(h11,h11,-1)
                    Ns = Ns.transpose(2,0,1) # (N,h11,h11)
                    Ns = Ns[:,1:,1:]

                    singular.append(_check_singular(Ns))

                if not singular:
                    continue
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
        output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_make_pfvs)(p_chunk, job_i) for job_i, p_chunk in\
                                                            enumerate(p_chunks))
        all_Ks, all_Ms = zip(*output)
        all_Ks = np.vstack(all_Ks)
        all_Ms = np.vstack(all_Ms)
    else:
        all_Ks, all_Ms = _make_pfvs(ps)

    # return
    if return_formal_pfvs:
        from .pfv import PFV
        return [PFV(data, K, M) for K,M in zip(all_Ks, all_Ms)]
    else:
        return all_Ks, all_Ms
