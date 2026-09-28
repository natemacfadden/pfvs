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
import math
import numpy as np
import os
import warnings

from fractions import Fraction
from numpy.typing import ArrayLike

# local imports
from . import util
from .cydata import CYData
from .pfv_kernel import pfv_kernel
from .fp_kernel.fp_kernel import _enumerate, _lattice_build

# Zp helpers
# ==========
# generic
# -------
def _check_singular(Ns: ArrayLike, rtol: float | None = None) -> np.ndarray:
    """
    Which integer matrices in the stack Ns (n, m, m) are singular, decided
    exactly (see util.singular_mask; a float SVD is only a prefilter). rtol
    is ignored (kept for backwards compatibility).
    """
    return util.singular_mask(Ns)

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
               extra_checks: bool = False,
                   _maxes: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Deprecated, no effect (warns if set): mat is always computed in
        exact integer arithmetic, so there is nothing to check.

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
    if extra_checks:
        util.warn_unused("extra_checks", "the ellipsoid matrix is always computed in exact integer arithmetic")
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
    p = util._as_integral(p)

    # All arithmetic below is exact. It runs in int64 when a bound on every
    # intermediate (from max|kappa|, max|p|, max|Mbasis|) shows it cannot
    # overflow -- the common case -- else with Python ints (large p).
    # (_maxes: precomputed (max|kappa|, max|Mbasis|), fixed across p-vectors)
    kmax, Mmax = _maxes if _maxes is not None else (util.absmax(kappa), util.absmax(Mbasis))
    pmax = util.absmax(p)
    small = h11**3 * kmax * pmax**2 * Mmax < 2**62 and p.dtype != object

    if small:
        Z = kappa @ p
    else:
        Z = util.exact_matmul(kappa.reshape(-1, h11), p).reshape(h11, h11)

    # define the lattices for M
    # -------------------------
    # need dot(K,p) = 0
    # thus need dot(p, kappa @ M @ p) = 0
    # equivalently, dot((kappa @ p) @ p, M) = 0
    T = Z @ p if small else util.exact_matmul(Z, p)

    # need T^T @ Mbasis @ c = 0
    # thus just need c in the orthogonal lattice to Mbasis^T @ T
    # (the output will be lattice generators of such cs... we'll want
    #  lattice generators of valid Ms so we multiply on left by Mbasis)
    orthog = util.orthogonal_lattice(p=T @ Mbasis if small else util.exact_matmul(T, Mbasis))
    if extra_lll_reduction:
        orthog = util.lll_reduce(orthog)
    if small and orthog.dtype != object and h11 * Mmax * util.absmax(orthog) < 2**62:
        Binter = Mbasis @ orthog
    else:
        Binter = util.exact_matmul(Mbasis, orthog)

    # lll-reduce Binter
    # (doesn't seem to have a huge effect...)
    Binter = util.lll_reduce(Binter)

    # define ellipsoid
    #       M-term     K-term
    # (exact integer arithmetic: all inputs are integral)
    bmax = util.absmax(Binter)
    if small and Binter.dtype != object and h11**2 * (h11 * kmax * pmax) * bmax**2 < 2**62:
        mat = -(Binter.T @ (Z @ Binter))
    else:
        mat = -util.exact_matmul(Binter.T, util.exact_matmul(Z, Binter))
    if mat.dtype == object:
        raise OverflowError(
            f"ellipsoid matrix entries exceed int64 for p={np.array(p).tolist()}")

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
        Deprecated, no effect (warns if set): it is not used here.

    Returns
    -------
    mat : ndarray of shape (h11-1, h11-1)
        The matrix defining the ellipsoid. I.e., d^T @ mat @ d <= Qmax
    B : ndarray of shape (h11, h11-1)
        The K-vector lattice basis, integrating just the dot(K,p)=0 constraint.
    """
    if extra_checks:
        util.warn_unused("extra_checks", "it is not used by the K-ellipsoid path")
    if data is None:
        if kappa is None or Mbasis is None:
            raise ValueError("If data is None, both kappa and Mbasis must be provided.")
    else:
        if kappa is not None or Mbasis is not None:
            raise ValueError("kappa and Mbasis must be None when data is provided.")
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

    # the ellipsoid -B^T (kappa p)^{-1} B, exactly: s (kappa p)^{-1} is an
    # integer matrix (inv_scaled), so mat = mat_int / s
    mat_int, s = _K_ellipsoid_scaled(p, kappa, B)
    if all(int(x) % s == 0 for x in np.asarray(mat_int).ravel()):
        mat = util._to_int64_or_object(np.asarray(mat_int, dtype=object) // s)
    else:
        mat = np.array([[int(x) / s for x in row] for row in np.asarray(mat_int).tolist()])
    return mat, B


def _K_ellipsoid_scaled(p: ArrayLike, kappa: ArrayLike, B: ArrayLike) -> tuple[np.ndarray, int]:
    """
    (mat_int, s) with -B^T (kappa p)^{-1} B = mat_int / s exactly (mat_int
    integer; int64 when it fits, else Python ints).
    """
    h11 = kappa.shape[0]
    p = util._as_integral(np.asarray(p).ravel())
    A = util.exact_matmul(np.asarray(kappa).reshape(-1, h11), p).reshape(h11, h11)
    if A.dtype == object:
        raise OverflowError(f"kappa@p exceeds int64 for p={np.array(p).tolist()}")
    try:
        Ainv_s, s = util.inv_scaled(A)                # Ainv_s A = s I
    except ValueError as e:
        raise ValueError(
            f"K_ellipsoid: kappa@p is singular for p={np.array(p).tolist()}.") from e
    mat_int = -util.exact_matmul(np.asarray(B).T, util.exact_matmul(Ainv_s, B))
    return mat_int, int(s)

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
    use_c_kernel: bool = True,
    use_c_lattice: bool = True,
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
    use_c_lattice : bool, optional
        Whether to build each p-vector's lattice data (Binter, the ellipsoid
        and, with use_c_kernel, the H-matrix) in C (fast, exact, with
        automatic fallback to the Python path on overflow). The C path picks
        a different, equally valid LLL-reduced basis, so it finds the same
        PFVs but may list them in a different order. Set False to reproduce
        the previous order exactly. Defaults to True.
    use_c_kernel : bool, optional
        Enumeration backend. True (default) uses `pfv_kernel` (C: exact
        decisions, GCD pruning; much faster for dilated ellipsoids). False
        uses `util.fp_iterative_njit` (Numba, floating-point decisions, no
        GCD pruning), kept for reference. They find the same PFVs.
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
    if extra_checks:
        util.warn_unused("extra_checks", "the ellipsoid matrix is always computed in exact integer arithmetic", stacklevel=2)
        extra_checks = False
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
        eye_U  = np.eye(h11-1)  # kernel signature needs U; mat= is what is used
        maxes  = (util.absmax(kappa), util.absmax(Mbasis))

        for p in p_chunk:
            # C path (status -15: all but H built; other nonzero: fall back)
            H_pre, st = None, -1
            if use_c_lattice:
                st, Z, Binter, _, mat, H_pre = _lattice_build(
                    kappa, Mbasis, np.asarray(p, dtype=np.int64), False,
                    extra_lll_reduction)
            if st not in (0, -15):
                try:
                    mat, Z, Binter = M_ellipsoid(
                        p,
                        kappa=kappa,
                        Mbasis=Mbasis,
                        extra_lll_reduction=extra_lll_reduction,
                        extra_checks=extra_checks,
                        _maxes=maxes
                    )
                except OverflowError as e:
                    raise util.IncompleteSearchError(f"p={p.tolist()}: {e}") from e

            # the core enumeration
            # --------------------
            if use_c_kernel:
                ZBinter = np.ascontiguousarray(util.exact_matmul(Z, Binter))
                try:
                    H = H_pre if H_pre is not None else H_matrix(ZBinter)
                except Exception as e:
                    raise util.IncompleteSearchError(f"p={p.tolist()}: H_matrix failed ({e})") from e
                # U is unused when mat= is given (the kernel factors the
                # exact mat itself), but is part of the signature
                try:
                    lattice_points, _, status = pfv_kernel(
                        U=eye_U,
                        Q=Qmax,
                        dilation=ellipsoid_dilation,
                        H=H,
                        max_N_out=max_N_pfvs,
                        eps=1e-4,
                        mat=mat,
                    )
                except ValueError as e:
                    # the kernel decides positive-definiteness exactly
                    if "positive definite" not in str(e):
                        raise
                    raise util.IncompleteSearchError(
                        f"p={p.tolist()}: the ellipsoid matrix is not positive "
                        f"definite, so its lattice points cannot be enumerated") from e
                if status != 0:
                    raise util.IncompleteSearchError(
                        f"p={p.tolist()}: pfv_kernel returned status {status} "
                        + ("(more than max_N_pfvs outputs; increase max_N_pfvs)"
                           if status == -2 else
                           "(lattice-point coordinates exceed int32)"
                           if status == -8 else ""))
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
            Ms = util.exact_matmul(Binter, lattice_points.T) # as columns

            # filter by N invertibility
            # -------------------------
            batch_size = 5000
            singular = []
            for i in range(0, Ms.shape[1], batch_size):
                chunk = Ms[:,i:i+batch_size]

                Ns = util.exact_matmul(kappa.reshape(h11*h11,h11), chunk).reshape(h11,h11,-1)
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
            Ks = util.exact_matmul(Z, Ms)

            K_gcds = np.gcd.reduce(Ks, axis=0)
            Ks = Ks//K_gcds

            # filter by tadpole
            Qs = -util.colsum_prod(Ks, Ms)
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

        if verbosity > 0:
            print(f"Finished job #{job_i}...", flush=True)

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
        Deprecated, no effect (warns if set): it is not used by the K-ellipsoid path.
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
    if extra_checks:
        util.warn_unused("extra_checks", "it is not used by the K-ellipsoid path", stacklevel=2)
        extra_checks = False
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
            A = util.exact_matmul(
                util.exact_matmul(kappa.reshape(-1, h11), p).reshape(h11, h11), Mbasis)
            try:
                Ainv, _ = util.inv_scaled(A)
            except OverflowError as e:
                raise util.IncompleteSearchError(
                    f"p={p.tolist()}: scaled inverse of kappa@p@Mbasis exceeds int64") from e
            except Exception as e:
                raise ValueError(
                    f"inv_scaled failed for p={p.tolist()}; kappa@p@Mbasis "
                    f"may be singular."
                ) from e

            # the K-lattice and the (rational) ellipsoid, exactly:
            # c^T mat c <= D Qmax  <=>  c^T mat_int c <= floor(s D Qmax)
            B = util.lll_reduce(util.orthogonal_lattice(p=p))
            try:
                mat_int, s_den = _K_ellipsoid_scaled(p, kappa, B)
            except OverflowError as e:
                raise util.IncompleteSearchError(f"p={p.tolist()}: {e}") from e
            qmax = math.floor(Fraction(ellipsoid_dilation) * Qmax * s_den)
            if mat_int.dtype == object or qmax >= 2**62:
                raise util.IncompleteSearchError(
                    f"p={p.tolist()}: the scaled K-ellipsoid exceeds int64")

            # the core enumeration: the exact C kernel (no GCD cut here)
            # --------------------
            d = mat_int.shape[0]
            try:
                lattice_points, _, status = _enumerate(
                    np.eye(d), mat_int, 1, qmax, np.zeros((0, d), dtype=object),
                    None, 0, False, max_N_pfvs, 1e-4, 0)
            except ValueError as e:
                raise util.IncompleteSearchError(
                    f"p={p.tolist()}: K-ellipsoid not positive definite ({e})") from e
            if status != 0:
                raise util.IncompleteSearchError(
                    f"p={p.tolist()}: kernel status {status} "
                    + ("(more than max_N_pfvs outputs; increase max_N_pfvs)"
                       if status == -2 else ""))

            # only keep primitive lattice points (can reclaim other PFVs easily)
            primitiveQ = np.gcd.reduce(lattice_points, axis=1) == 1
            lattice_points = lattice_points[primitiveQ]

            # compute Ms, Ks, and reduced by GCD
            # ----------------------------------
            # read the data
            cs = util.exact_matmul(util.exact_matmul(Ainv, B), lattice_points.T).T
            gcds = np.gcd.reduce(cs,axis=1)
            cs_scaled = cs//gcds.reshape(-1,1)

            Ks = util.exact_matmul(B, lattice_points.T) # as columns
            Ms = util.exact_matmul(Mbasis, cs_scaled.T)

            # filter on tadpole
            Qs = -util.colsum_prod(Ks, Ms)
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

                Ns = util.exact_matmul(kappa.reshape(h11*h11,h11), chunk).reshape(h11,h11,-1)
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

        if verbosity > 0:
            print(f"Finished job #{job_i}...", flush=True)

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
