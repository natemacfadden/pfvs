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
# Description:  Non-coni PFVs by "Zp" search: for each p-vector, enumerate
#               the lattice points of an ellipsoid.
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
    """Exact singularity mask of the integer stack Ns (n, m, m). rtol is ignored."""
    return util.singular_mask(Ns)

# non-coni
# --------
def _allow_gcds(Ks: ArrayLike, Ms: ArrayLike, Qmax: int, h11: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand primitive (K, M) pairs to all (aK, bM), a, b >= 1, with
    -dot(aK, bM) <= Qmax. Ks, Ms: shape (N, h11). Raises ValueError if none
    remain.
    """
    num_input = len(Ks)
    if num_input == 0:
        return np.zeros((0,h11),dtype=int), np.zeros((0,h11),dtype=int)

    # add the GCDs
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
    The M-ellipsoid of non-coni ZpM: M = Binter @ c, K = Z @ M, and the
    tadpole -dot(K, M) <= Qmax becomes c^T mat c <= Qmax with
    mat = -Binter^T Z Binter. Dilating is allowed where gcd(K) >= c^T mat c / Qmax.

    Parameters
    ----------
    p : ndarray of shape (h11,)
        The p-vector.
    data : CYData, optional
        The CY. Or pass kappa and Mbasis (data.M_lattice()) instead.
    extra_lll_reduction : bool, optional
        Extra LLL reduction of Binter; avoids some overflows. Defaults to True.
    extra_checks : bool, optional
        Deprecated, no effect.

    Returns
    -------
    mat : ndarray of shape (h11-1, h11-1)
    Z : ndarray of shape (h11, h11)
        K = Z @ M.
    Binter : ndarray of shape (h11, h11-1)
        Basis of the M-lattice with dot(K, p) = 0.
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

    # exact arithmetic: int64 when a bound shows no overflow, else Python ints.
    # _maxes: precomputed (max|kappa|, max|Mbasis|)
    kmax, Mmax = _maxes if _maxes is not None else (util.absmax(kappa), util.absmax(Mbasis))
    pmax = util.absmax(p)
    small = h11**3 * kmax * pmax**2 * Mmax < 2**62 and p.dtype != object

    if small:
        Z = kappa @ p
    else:
        Z = util.exact_matmul(kappa.reshape(-1, h11), p).reshape(h11, h11)

    # dot(K, p) = 0  <=>  dot((kappa @ p) @ p, M) = 0
    T = Z @ p if small else util.exact_matmul(Z, p)

    # c in the lattice orthogonal to T @ Mbasis; Binter = Mbasis @ (that basis)
    orthog = util.orthogonal_lattice(p=T @ Mbasis if small else util.exact_matmul(T, Mbasis))
    if extra_lll_reduction:
        orthog = util.lll_reduce(orthog)
    if small and orthog.dtype != object and h11 * Mmax * util.absmax(orthog) < 2**62:
        Binter = Mbasis @ orthog
    else:
        Binter = util.exact_matmul(Mbasis, orthog)

    Binter = util.lll_reduce(Binter)

    # the ellipsoid, exactly
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
    The K-ellipsoid of non-coni ZpK: K = B @ d with B a basis of the lattice
    orthogonal to p, M = (kappa p)^{-1} K, and the tadpole becomes
    d^T mat d <= Qmax with mat = -B^T (kappa p)^{-1} B.

    Parameters
    ----------
    p : ndarray of shape (h11,)
        The p-vector.
    data : CYData, optional
        The CY. Or pass kappa and Mbasis (data.M_lattice()) instead.
    extra_lll_reduction : bool, optional
        Extra LLL reduction of Binter; avoids some overflows. Defaults to True.
    extra_checks : bool, optional
        Deprecated, no effect.

    Returns
    -------
    mat : ndarray of shape (h11-1, h11-1)
    B : ndarray of shape (h11, h11-1)
        Basis of the K-lattice orthogonal to p.
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

    # K-lattice: K^T p = 0
    B = util.orthogonal_lattice(p=p)
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
    Row-HNF of Z@Binter (K = ZBinter @ c), for gcd(K) pruning in ZpM: since
    gcd(H @ c) = gcd(K) and the gcd of trailing blocks only decreases as FP
    fixes c from the right, a dilated point with gcd(K) < c^T mat c / Qmax is
    pruned early. As `coni_H_matrix`, without dropping K[0].
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
    Non-coni PFVs from p-vectors: Fincke-Pohst over the M-ellipsoid of each p
    (`M_ellipsoid`), with gcd(K) pruning; GCDs are re-introduced afterwards
    (`_allow_gcds`).

    Parameters
    ----------
    data : CYData
        The CY (non-coni).
    ps : ArrayLike of shape (N, h11)
        The p-vectors, inside the Kahler cone (H @ p >= 1).
    Qmax, Qmin : integer, optional
        Keep PFVs with Qmin <= -dot(K, M) <= Qmax. Default h11+h21+2 and 0.
    ellipsoid_dilation : float, optional
        Ellipsoid dilation; runtime grows about linearly. Defaults to 1.
    use_c_kernel : bool, optional
        True: the exact C kernel. False: the Numba reference (float, no GCD
        pruning; same PFVs). Defaults to True.
    use_c_lattice : bool, optional
        Build each p's lattice data in C (falls back to Python on overflow;
        same PFVs, possibly in another order). Defaults to True.
    n_jobs : int, optional
        Parallel jobs. Defaults to 2x the CPU count.
    extra_checks : bool, optional
        Deprecated, no effect.
    extra_lll_reduction : bool, optional
        As in `M_ellipsoid`.
    max_N_pfvs : int, optional
        Output limit (the C kernel needs one). Defaults to 1e9.
    return_formal_pfvs : bool, optional
        Return PFV objects instead of (Ks, Ms). Defaults to False.
    verbosity : int, optional
        Defaults to 0.

    Returns
    -------
    Ks, Ms : ndarrays of shape (N, h11)
        The PFVs, one per row (or a list of PFV objects).

    Raises
    ------
    ValueError
        If data.coni, ps is empty or outside the cone, Qmax < Qmin,
        ellipsoid_dilation <= 0, or no PFVs survive `_allow_gcds`.
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
                # U is ignored when mat= is given
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
    Non-coni PFVs from p-vectors: enumerate the K-ellipsoid of each p
    (`K_ellipsoid`), without GCD pruning; GCDs are re-introduced afterwards.

    Parameters
    ----------
    data : CYData
        The CY (non-coni).
    ps : ArrayLike of shape (N, h11)
        The p-vectors, inside the Kahler cone (H @ p >= 1).
    Qmax, Qmin : integer, optional
        Keep PFVs with Qmin <= -dot(K, M) <= Qmax. Default h11+h21+2 and 0.
    ellipsoid_dilation : float, optional
        Ellipsoid dilation; runtime grows about linearly. Defaults to 1.
    n_jobs : int, optional
        Parallel jobs. Defaults to 2x the CPU count.
    extra_checks : bool, optional
        Deprecated, no effect.
    extra_lll_reduction : bool, optional
        As in `K_ellipsoid`.
    max_N_pfvs : int, optional
        Output limit (the C kernel needs one). Defaults to 1e9.
    return_formal_pfvs : bool, optional
        Return PFV objects instead of (Ks, Ms). Defaults to False.
    verbosity : int, optional
        Defaults to 0.

    Returns
    -------
    Ks, Ms : ndarrays of shape (N, h11)
        The PFVs, one per row (or a list of PFV objects).

    Raises
    ------
    ValueError
        If data.coni, ps is empty or outside the cone, Qmax < Qmin,
        ellipsoid_dilation <= 0, or no PFVs survive `_allow_gcds`.
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
