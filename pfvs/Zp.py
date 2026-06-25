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
# Description:  This module contains methods for constructing non-coni PFVs
#               using the "Zp" style algorithms. These operate by fixing some
#               p-vectors and then searching for lattice points in an ellipsoid,
#               one for each p-vector.
# -----------------------------------------------------------------------------

# external imports
import flint
import joblib
import numpy as np
import os
import warnings

from numpy.typing import ArrayLike

# local imports
from . import util
from .cydata import CYData
from .pfv_kernel import pfv_kernel

# Zp helpers
# ==========
# generic
# -------
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
    svals = np.linalg.svdvals(Ns)
    singular = (svals[:,-1] <= rtol * svals[:,0])

    return singular

# non-coni
# --------
def _allow_gcds(Ks: ArrayLike, Ms: ArrayLike, Qmax: int, h11: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-introduce nontrivial GCDs into primitive (K, M) pairs.

    Non-coni PFVs naturally operate on primitive K, M (i.e., gcd(K)=gcd(M)=1).
    This function expands each such pair to all (aK, bM) with a, b >= 1 that
    still satisfy the tadpole constraint -dot(aK, bM) <= Qmax.

    Parameters
    ----------
    Ks : ArrayLike of shape (N, h11)
        Primitive K-vectors, one per row.
    Ms : ArrayLike of shape (N, h11)
        Primitive M-vectors, one per row.
    Qmax : int
        Maximum allowed tadpole -dot(K, M).
    h11 : int
        Number of Kahler moduli.

    Returns
    -------
    Ks : ndarray of shape (M, h11)
        Expanded K-vectors including nontrivial GCDs.
    Ms : ndarray of shape (M, h11)
        Expanded M-vectors including nontrivial GCDs.
    """
    num_input = len(Ks)
    if num_input == 0:
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)

    # add the GCDs
    # (don't think there should be duplicates but cheap to do this in a set...)
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

    # return
    if len(Ks):
        return np.vstack(Ks), np.vstack(Ms)
    else:
        msg = f"Found 0 PFVs after introducing GCDs... #input = {num_input}"
        raise ValueError(msg)

# very PFV-specific helpers
# -------------------------
def M_ellipsoid(p: ArrayLike,
               data: CYData = None,
               kappa: ArrayLike = None,
               Mbasis: ArrayLike = None,
               extra_lll_reduction: bool = True,
               extra_checks: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the matrices defining the M-ellipsoid in nonconi-ZpM.

    In brief detail,
        - M lives in a lattice M = Binter@c
        - K can be computed as Z @ M
        - tadpole is obeyed (-dot(K,M)<=Qmax) iff
          -c^T @ Binter^T @ Z @ Binter @ c <= Qmax. Define
          mat = -Binter^T @ Z @ Binter.
    That last constraint c^T @ mat @ c <= Qmax is the ellipsoid constraint. One
    can actually dilate the ellipsoid as long as
    GCD(Kperp) >= (c^T @ mat @ c)/Qmax. This is subtly different from coni
    contexts since, here, we don't typically enforce K[0] > 0

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
        The matrix relating M and K. Specifically, K = Z @ M
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
        kappa  = data.kappa
        Mbasis = data.M_lattice()

    p = np.array(p).ravel()

    # helper variable (K = Z @ M)
    Z = kappa@p

    # define the lattices for M
    # -------------------------
    # need dot(K,p) = 0
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

def K_ellipsoid(p: ArrayLike,
               data: CYData = None,
               kappa: ArrayLike = None,
               Mbasis: ArrayLike = None,
               extra_lll_reduction: bool = True,
               extra_checks: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the matrices defining the K-ellipsoid in nonconi-ZpK.

    In brief detail,
        - K lives in the orthog lattice to p. Call a basis for this lattice B
        - M can be computed as (kappa p)^{-1} K
        - tadpole is obeyed (-dot(K,M)<=Qmax) iff
          -d^T @ B^T @ (kappa p)^{-1} @ B @ d <= Qmax. Define
          mat = -B^T @ Ainv @ B.
    That last constraint d^T @ mat @ d <= Qmax is the ellipsoid constraint.

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
        The matrix defining the ellipsoid. I.e., d^T @ mat @ d <= Qmax
    B : ndarray of shape (h11, h11-1)
        The K-vector lattice basis, integrating just the dot(K,p)=0 constraint.
    """
    if data is None:
        if kappa is None or Mbasis is None:
            raise ValueError("If data is None, both kappa and Mbasis must be provided.")
        h11 = kappa.shape[0]
    else:
        if kappa is not None or Mbasis is not None:
            raise ValueError("kappa and Mbasis must be None when data is provided.")
        h11    = data.h11
        kappa  = data.kappa
        Mbasis = data.M_lattice()

    p = np.array(p).ravel()

    # define the lattices for K
    # -------------------------
    # (need K^T@p = 0)
    B = util.orthogonal_lattice(p=p)

    # lll-reduce B
    # (doesn't seem to have a huge effect...)
    B = util.lll_reduce(B)

    # find lattice points in tadpole
    try:
        Ainv = np.linalg.inv(kappa@p)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"K_ellipsoid: kappa@p is singular for p={np.array(p).tolist()}."
        ) from e
    mat = -B.T@Ainv@B

    if np.allclose(mat, np.round(mat)):
        mat = np.rint(mat).astype(int)

    return mat, B

def H_matrix(ZBinter: ArrayLike):
    """
    Compute the H-matrix for use in non-coni ZpM. This is the HNF of Z@Binter.

    Analogous to `coni_H_matrix` in coniZp.py, but without the projection that
    drops K[0]: in the non-coni context, K = Z @ Binter @ c directly (no free
    K[0] component), so the full matrix is used.

    In ZpM, one wants to ensure GCD(K) is sufficiently large. A point c in the
    M-ellipsoid has an associated valuation c^T @ mat @ c. For dilated ellipsoids,
    this can have c^T @ mat @ c > Qmax. This would give rise to a K and M which
    violates tadpole (i.e., -dot(K,M) > Qmax) unless
        GCD(K) >= (c^T @ mat @ c)/Qmax,
    in which case one can divide K by GCD(K) to bring the solution under tadpole.

    Recall that K = Z @ Binter @ c. Since H is the row-HNF of Z @ Binter,
    GCD(H @ c) = GCD(K), and GCD(H[-m:,-m:] @ c[-m:]) >= GCD(H[-n:,-n:] @ c[-n:])
    for m < n. This gives a monotonically decreasing upper bound on GCD(K) as FP
    sets components of c from right to left, enabling early pruning.

    Parameters
    ----------
    ZBinter : ndarray of shape (h11, h11-1)
        The product of matrices Z and Binter from M_ellipsoid. Has interpretation
        that K = ZBinter @ c.

    Returns
    -------
    H : ndarray of shape (h11, h11-1)
        The row-HNF of Z@Binter. Has interpretation that GCD(H@c) = GCD(K) and
        that GCD(H[-m:,-m:]@c[-m:]) >= GCD(H[-n:,-n:]@c[-n:]) for m<n.
    """
    H    = ZBinter
    H_fl = flint.fmpz_mat(H.tolist())

    H_list = H_fl.hnf().tolist()
    H      = np.array([[int(x) for x in row] for row in H_list], dtype=object)

    return H

# non-coni Zp
# ===========
def ZpM(
    # problem definition
    data: CYData,
    ps: ArrayLike,
    Qmax: int | None = None,
    Qmin: int = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    # algorithm selection
    use_c_kernel: bool = False,
    n_jobs: int = -1,
    # misc
    extra_checks: bool = False,
    extra_lll_reduction: bool = True,
    # output/verbosity
    max_N_pfvs: int = 1_000_000_000,
    return_formal_pfvs: bool = False,
    verbosity: int = 0
    ) -> tuple[ArrayLike, ArrayLike]:
    """
    A 'Zp' implementation that computes non-coni PFVs from input integer
    p-vectors.

    The logic is
        1 an integer p-vector defines a certain ellipsoid (see `M_ellipsoid`)
        2 a lattice point c in this ellipsoid defines an M-vector via Binter@c.
          this also defines a K-vector via K = Z @ Binter @ c
    so one wants to enumerate such c-vectors. This is done via Fincke-Pohst.
    GCD re-introduction is applied post-hoc via `_allow_gcds`.

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY.
    ps : iterable of shape (N, h11-1)
        Each row of the iterable corresponds to the perpendicular component of a
        p-vector. I.e., p[1:]
    Qmax : integer, optional
        Only return PFVs with -dot(K,M) <= Qmax. If not provided, set to
        h11+h21+2.
    Qmin : integer, optional
        Only return PFVs with -dot(K,M) >= Qmin. If not provided, set to 0.
    ellipsoid_dilation : float, optional
        The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
        Empirically, runtime scales linearly with this value. Defaults to 1.
    use_c_kernel : bool, optional
        Enumeration backend. False (default) uses `util.fp_iterative_njit`
        (Numba, no GCD pruning). True uses `pfv_kernel` (C, with GCD pruning;
        faster for dilated ellipsoids). Both produce the same output.
    n_jobs : int, optional
        How many jobs to spawn for per-p-vector parallelism. Defaults to twice
        the CPU count.
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

    Raises
    ------
    ValueError
        If ``data.coni`` is True, ``ps`` is empty, ``Qmax < Qmin``,
        ``ellipsoid_dilation <= 0``, or if ``_allow_gcds`` finds no valid
        GCD expansions.
    """
    warnings.warn("non-coni Zp methods are slightly outdated relative to the coni path", stacklevel=2)
    if data.coni:
        raise ValueError(
            "Methods in Zp.py only apply to non-coni contexts. "
            "Use coniZp.py for coni PFVs."
        )
    if len(ps) == 0:
        raise ValueError("ps must be non-empty.")
    ps = np.array(ps)
    if not np.all(data.H @ ps.T >= 1):
        raise ValueError("some p-vectors are not in the kahler cone (H@ps.T >= 1 failed)")

    # misc (left for future debugging)
    only_positive_news = False

    # read data
    kappa  = data.kappa
    h11    = data.h11
    h21    = data.h21
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = h11+h21+2
    if Qmax < Qmin:
        raise ValueError(f"Qmax ({Qmax}) must be >= Qmin ({Qmin}).")
    if ellipsoid_dilation <= 0:
        raise ValueError(f"ellipsoid_dilation must be > 0, got {ellipsoid_dilation}.")

    if n_jobs == -1:
        n_jobs = 2*os.cpu_count()

    # the search
    # ----------
    # iterate over p-vectors
    chunk_size = max(100, len(ps)//n_jobs+1)
    p_chunks   = [ps[i:i+chunk_size] for i in range(0,len(ps),chunk_size)]

    def _make_pfvs(p_chunk, job_i=0):
        # define a factory function here for later parallelization
        all_Ks = np.zeros((0,h11), dtype=int)
        all_Ms = np.zeros((0,h11), dtype=int)

        for p in p_chunk:
            mat, Z, Binter = M_ellipsoid(
                p,
                kappa=kappa,
                Mbasis=Mbasis,
                extra_lll_reduction=extra_lll_reduction,
                extra_checks=extra_checks
            )

            # the core enumeration
            # --------------------
            if use_c_kernel:
                ZBinter = np.ascontiguousarray(Z @ Binter)
                try:
                    H = H_matrix(ZBinter)
                except Exception as e:
                    warnings.warn(f"skipping p={p.tolist()}: H_matrix failed ({e})", stacklevel=2)
                    continue
                try:
                    L = np.linalg.cholesky(mat)
                except Exception as e:
                    warnings.warn(f"skipping p={p.tolist()}: cholesky of mat failed ({e})", stacklevel=2)
                    continue
                lattice_points, _, status = pfv_kernel(
                    U=np.ascontiguousarray(L.T),
                    Q=Qmax,
                    dilation=ellipsoid_dilation,
                    H=H,
                    max_N_out=max_N_pfvs,
                    eps=1e-4
                )
                if status != 0:
                    warnings.warn(f"pfv_kernel returned status {status} for p={p.tolist()}", stacklevel=2)
            else:
                try:
                    L = np.linalg.cholesky(mat)
                    lattice_points, _ = util.fp_iterative_njit(
                        L=L,
                        Q=ellipsoid_dilation*Qmax,
                        max_N_out=max_N_pfvs)
                except Exception as e:
                    raise RuntimeError(
                        f"Kernel failed for p={np.array(p).tolist()}: {type(e).__name__}: {e}"
                    ) from e

            # only keep primitive lattice points (can reclaim other PFVs easily)
            primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
            lattice_points = lattice_points[primitiveQ]

            # compute Ms
            # ----------
            Ms = Binter@lattice_points.T # as columns

            # filter by N invertibility
            # -------------------------
            batch_size = 5000
            singular = []
            for i in range(0, Ms.shape[1], batch_size):
                chunk = Ms[:,i:i+batch_size]

                Ns = (kappa.reshape(h11*h11,h11)@chunk).reshape(h11,h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)

                singular.append(_check_singular(Ns))

            if not singular:
                continue
            singular = np.concatenate(singular)

            if verbosity >= 2:
                if len(singular) and not only_positive_news:
                    print(f"{sum(singular)}/{len(singular)} 'PFVs' had det(N)=0 :(")

            Ms = Ms[:,~singular]

            # compute Ks, reduce by GCDs
            # --------------------------
            Ks = Z@Ms

            K_gcds = np.gcd.reduce(Ks, axis=0)
            Ks = Ks//K_gcds

            # filter by tadpole
            Qs = -np.sum(Ks*Ms,axis=0)
            in_tadpole = (Qs>=Qmin) & (Qs<=Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    n_bad = len(in_tadpole) - sum(in_tadpole)
                    print(f"{n_bad}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

            # transpose to row-wise
            Ks, Ms = Ks.T, Ms.T

            # save to data structures
            all_Ks = np.vstack([all_Ks, Ks])
            all_Ms = np.vstack([all_Ms, Ms])

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
    all_Ks, all_Ms = _allow_gcds(all_Ks, all_Ms, Qmax, data.h11)
    if return_formal_pfvs:
        from .pfv import PFV
        return [PFV(data, K, M) for K, M in zip(all_Ks, all_Ms)]
    return all_Ks, all_Ms

def ZpK(
    # problem definition
    data: CYData,
    ps: ArrayLike,
    Qmax: int | None = None,
    Qmin: int = 0,
    ellipsoid_dilation: float = 1, # typically want >=1
    # algorithm selection
    n_jobs: int = -1,
    # misc
    extra_checks: bool = False,
    extra_lll_reduction: bool = True,
    # output/verbosity
    max_N_pfvs: int = 1_000_000_000,
    return_formal_pfvs: bool = False,
    verbosity: int = 0
    ) -> tuple[ArrayLike, ArrayLike]:
    """
    A 'Zp' implementation that computes non-coni PFVs from input integer
    p-vectors.

    The logic is
        1 an integer p-vector defines a certain ellipsoid (see `K_ellipsoid`)
        2 a lattice point d in this ellipsoid defines a K-vector via B @ d.
          this also defines an M-vector via M = (kappa @ p)^{-1} @ B @ d
    so one wants to enumerate such d-vectors. This is done via Fincke-Pohst.
    GCD re-introduction is applied post-hoc via `_allow_gcds`. [WIP: GCD
    pruning is not yet integrated into the Fincke-Pohst solver here, unlike
    `coniZpM`.]

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY.
    ps : iterable of shape (N, h11-1)
        Each row of the iterable corresponds to the perpendicular component of a
        p-vector. I.e., p[1:]
    Qmax : integer, optional
        Only return PFVs with -dot(K,M) <= Qmax. If not provided, set to
        h11+h21+2.
    Qmin : integer, optional
        Only return PFVs with -dot(K,M) >= Qmin. If not provided, set to 0.
    ellipsoid_dilation : float, optional
        The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
        Empirically, runtime scales linearly with this value. Defaults to 1.
    n_jobs : int, optional
        How many jobs to spawn for per-p-vector parallelism. Defaults to twice
        the CPU count.
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

    Raises
    ------
    ValueError
        If ``data.coni`` is True, ``ps`` is empty, ``Qmax < Qmin``,
        ``ellipsoid_dilation <= 0``, or if ``_allow_gcds`` finds no valid
        GCD expansions.
    """
    warnings.warn("non-coni Zp methods are slightly outdated relative to the coni path", stacklevel=2)
    if data.coni:
        raise ValueError(
            "Methods in Zp.py only apply to non-coni contexts. "
            "Use coniZp.py for coni PFVs."
        )
    if len(ps) == 0:
        raise ValueError("ps must be non-empty.")
    ps = np.array(ps)
    if not np.all(data.H @ ps.T >= 1):
        raise ValueError("some p-vectors are not in the kahler cone (H@ps.T >= 1 failed)")

    # misc (left for future debugging)
    only_positive_news = False

    # read data
    kappa  = data.kappa
    h11    = data.h11
    h21    = data.h21
    Mbasis = data.M_lattice()

    if Qmax is None:
        Qmax = h11+h21+2
    if Qmax < Qmin:
        raise ValueError(f"Qmax ({Qmax}) must be >= Qmin ({Qmin}).")
    if ellipsoid_dilation <= 0:
        raise ValueError(f"ellipsoid_dilation must be > 0, got {ellipsoid_dilation}.")

    if n_jobs == -1:
        n_jobs = 2*os.cpu_count()

    # the search
    # ----------
    # iterate over p-vectors
    chunk_size = max(100, len(ps)//n_jobs+1)
    p_chunks   = [ps[i:i+chunk_size] for i in range(0,len(ps),chunk_size)]

    def _make_pfvs(p_chunk, job_i=0):
        # define a factory function here for later parallelization
        all_Ks = np.zeros((0,h11), dtype=int)
        all_Ms = np.zeros((0,h11), dtype=int)

        for p in p_chunk:
            # helper variables
            A = kappa@p@Mbasis
            try:
                Ainv, _ = util.inv_scaled(A)
            except Exception as e:
                raise ValueError(
                    f"inv_scaled failed for p={p.tolist()}; kappa@p@Mbasis "
                    f"may be singular."
                ) from e

            mat, B = K_ellipsoid(
                p,
                kappa=kappa,
                Mbasis=Mbasis,
                extra_lll_reduction=extra_lll_reduction,
                extra_checks=extra_checks
            )

            # the core enumeration
            # --------------------
            try:
                L = np.linalg.cholesky(mat)
                lattice_points, _ = util.fp_iterative_njit(
                    L=L,
                    Q=ellipsoid_dilation*Qmax,
                    max_N_out=max_N_pfvs)
            except Exception as e:
                raise RuntimeError(
                    f"Kernel failed for p={np.array(p).tolist()}: {type(e).__name__}: {e}"
                ) from e

            # only keep primitive lattice points (can reclaim other PFVs easily)
            primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
            lattice_points = lattice_points[primitiveQ]

            # compute Ms, Ks, and reduced by GCD
            # ----------------------------------
            # read the data
            cs = ((Ainv@B)@lattice_points.T).T
            gcds = np.gcd.reduce(cs,axis=1)
            cs_scaled = cs//gcds.reshape(-1,1)

            Ks = B@lattice_points.T # as columns
            Ms = Mbasis@cs_scaled.T

            # filter on tadpole
            Qs = -np.sum(Ks*Ms,axis=0)
            in_tadpole = (Qs>=Qmin) & (Qs<=Qmax)
            if verbosity >= 2:
                if not only_positive_news:
                    n_bad = len(in_tadpole) - sum(in_tadpole)
                    print(f"{n_bad}/{len(in_tadpole)} 'PFVs' violated tadpole :(")
                if sum(in_tadpole):
                    print(f"but {sum(in_tadpole)} in tadpole!!!")
            Ks = Ks[:,in_tadpole]
            Ms = Ms[:,in_tadpole]

            # filter by N invertibility
            batch_size = 5000
            singular = []
            for i in range(0, Ms.shape[1], batch_size):
                chunk = Ms[:,i:i+batch_size]

                Ns = (kappa.reshape(h11*h11,h11)@chunk).reshape(h11,h11,-1)
                Ns = Ns.transpose(2,0,1) # (N,h11,h11)

                singular.append(_check_singular(Ns))

            if not singular:
                continue
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
    all_Ks, all_Ms = _allow_gcds(all_Ks, all_Ms, Qmax, data.h11)
    if return_formal_pfvs:
        from .pfv import PFV
        return [PFV(data, K, M) for K, M in zip(all_Ks, all_Ms)]
    return all_Ks, all_Ms
