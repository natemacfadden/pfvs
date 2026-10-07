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
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

from numpy.typing import ArrayLike

# local imports
from . import util
from .conipfv_kernel import conipfv_kernel
from .fp_kernel.fp_kernel import _coni_batch, _coni_zpk_batch, _lattice_build
from .cydata import CYData

# coniZp helpers
# ==============
def _check_singular(Ns: ArrayLike, rtol: float | None = None) -> np.ndarray:
    """
    Which integer matrices in the stack Ns (n, m, m) are singular, decided
    exactly (see util.singular_mask; a float SVD is only a prefilter). rtol
    is ignored (kept for backwards compatibility).
    """
    return util.singular_mask(Ns)

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
                   extra_checks: bool = False,
                   _maxes: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Deprecated, no effect (warns if set): mat is always computed in
        exact integer arithmetic, so there is nothing to check.

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
        kappa  = data.kappa_cob
        Mbasis = data.M_lattice()

    p = np.array(p).ravel()
    if len(p) == h11-1:
        p = np.concatenate([[0], p])
    p = util._as_integral(p)

    # All arithmetic below is exact. It runs in int64 when a bound on every
    # intermediate (from max|kappa|, max|p|, max|Mbasis|) shows it cannot
    # overflow -- the common case -- else with Python ints (large p).
    # (_maxes: precomputed (max|kappa|, max|Mbasis|), fixed across p-vectors)
    kmax, Mmax = _maxes if _maxes is not None else (util.absmax(kappa), util.absmax(Mbasis))
    pmax = util.absmax(p)
    small = h11**3 * kmax * pmax**2 * Mmax < 2**62 and p.dtype != object

    # helper variable (K[1:] = (Z@M)[1:])
    if small:
        Z = kappa @ p
    else:
        Z = util.exact_matmul(kappa.reshape(-1, h11), p).reshape(h11, h11)

    # define the lattices for M
    # -------------------------
    # need dot(K,p) = 0
    #
    # note that K[1:] = (kappa @ M @ p)[1:]
    # (K[0] is unconstrained so (kappa @ M @ p)[0] is semi-meaningless)
    #
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

    # sort Binter so columns which don't affect M0 come first
    Binter = Binter[:,np.argsort(Binter[0]!=0)]

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
        raise ValueError(
            f"no all-zero column in H ({H.nrows()}x{H.ncols()}), so the null "
            "lattice is empty"
        )

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


def _pfvs_from_points(Ms: np.ndarray, Kns: np.ndarray, Qs: np.ndarray,
                      key: np.ndarray, kappa: np.ndarray, h11: int, Q: int,
                      M0min: int, verbosity: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn lattice points into coni-PFVs (the post-processing of coniZpM),
    vectorized over points from any number of p-vectors.

    Parameters
    ----------
    verbosity : int
        >= 2 prints the counts after each filter (as coniZpM used to).
    Ms, Kns : ndarray of shape (h11, N)
        For each lattice point c: M = Binter c and K_nat = (Z Binter) c.
    Qs : ndarray of shape (N,)
        c^T mat c (exact).
    key : ndarray of shape (N,)
        Output order key (the p-vector's index); results are returned
        stably sorted by it, matching p-by-p processing.

    Returns
    -------
    Ks, Ms, keys : ndarrays of shape (n, h11), (n, h11), (n,)
    """
    Ms, Kns = np.asarray(Ms), np.asarray(Kns)
    Qs, key = np.asarray(Qs), np.asarray(key)
    empty = (np.zeros((0, h11), dtype=np.int64), np.zeros((0, h11), dtype=np.int64),
             np.zeros(0, dtype=np.int64))
    if Qs.shape[0] == 0:
        return empty
    if M0min <= 0:
        raise ValueError("coniZpM requires M0min > 0")

    # All arithmetic below is exact: int64 when a bound on every intermediate
    # shows it cannot overflow (the normal case), else Python ints.
    Mb, Kb, Qb = util.absmax(Ms), util.absmax(Kns), util.absmax(Qs)
    exact = Ms.dtype == object or Kns.dtype == object or Qs.dtype == object \
        or Q * Kb >= 2**62
    if exact:
        Ms, Kns, Qs = util.as_exact(Ms, Kns, Qs)

    # cut on feasibility of finding a K0 giving K'>0 (see coniZpM)
    Kperps = Kns[1:]
    K_gcds = np.gcd.reduce(Kperps, axis=0)
    K_gcds[K_gcds < 1] = 1          # K = (x != 0, 0, ..., 0)
    mask   = Qs < Q * K_gcds
    Ms, Kns, Qs, K_gcds, key = Ms[:, mask], Kns[:, mask], Qs[mask], K_gcds[mask], key[mask]
    if Qs.shape[0] == 0:
        return empty

    # canonical order within a p-vector, by (M, K_nat): the search order
    # depends on the lattice basis, which can differ between the CPU and GPU
    # paths, and this makes their outputs identical. (Sorted after the cut
    # above, which keeps few of the raw points; nothing before it depends on
    # the order.)
    order = np.lexsort(tuple(Kns[::-1]) + tuple(Ms[::-1]) + (key,))
    Ms, Kns, Qs, K_gcds, key = Ms[:, order], Kns[:, order], Qs[order], K_gcds[order], key[order]

    # Kperp need not be primitive. Its gcd g is bounded per point:
    #   tadpole:  K0 = (g*rawQperp - Q)/M0
    #   K'>0:     K0 <  g*natural_K0/K_gcd
    # eliminating K0 with rawQperp*K_gcd - M0*natural_K0 = q gives
    #   g*q < Q*K_gcd,  i.e.  g <= (Q*K_gcd - 1) // q.
    # Each point is expanded into its own g = 1..gmax_i.
    gcap = (Q * K_gcds - 1) // np.maximum(Qs, 1)            # >= 1 by the mask
    gmax = int(np.max(gcap))
    Qperp_b = gmax * (Qb + Mb * Kb)                          # |Qperp|
    K0_b    = Qperp_b + Q + gmax * Kb                        # |K0| (lo/up ranges)
    K_b     = max(K0_b, gmax * Kb)                           # |entries of K|
    if not exact and max(Qperp_b, K0_b + len(Qs), h11 * K_b * Mb) >= 2**62:
        exact = True
        Ms, Kns, Qs, K_gcds = util.as_exact(Ms, Kns, Qs, K_gcds)
        gcap = (Q * K_gcds - 1) // np.maximum(Qs, 1)

    natural_K0s = Kns[0]
    Kperps      = Kns[1:] // K_gcds
    M0s         = Ms[0]             # > 0 (the kernel enforces M0 >= M0min)
    rawQperps   = (Qs + M0s * natural_K0s) // K_gcds

    # expand points by their admissible gcds, in (p, g, point) order
    counts = np.array([int(x) for x in gcap], dtype=np.int64) if exact \
        else gcap.astype(np.int64)
    idx = np.repeat(np.arange(len(Qs)), counts)
    g   = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts) + 1
    order = np.lexsort((idx, g, np.asarray(key[idx], dtype=np.int64)))
    idx, g = idx[order], g[order]
    if exact:
        g = g.astype(object)

    # K0 ranges to hit tadpole exactly: (Qperp - Q)/M0 (exact integers)
    Qperps = g * rawQperps[idx]
    M0i    = M0s[idx]
    lo = -((Q - Qperps) // M0i)                              # ceil
    up = (Qperps - Q) // M0i                                 # floor
    # K' > 0: K0 < natural_K0 * g / K_gcd, i.e. K0 <= ceil(.) - 1
    up = np.minimum(up, -((-natural_K0s[idx] * g) // K_gcds[idx]) - 1)

    Ks_out, Ms_out, key_out = [], [], []
    num = 1 + up - lo
    sel = num > 0
    num = num[sel]
    if num.size:
        if num.dtype == object:          # counts of K0 values: small
            num = np.array([int(x) for x in num], dtype=np.int64)
        total = int(np.sum(num))
        K0s = np.repeat(lo[sel], num) + np.arange(total) \
            - np.repeat(np.cumsum(num) - num, num)
        pts = idx[sel]
        new_Ks = np.vstack([K0s, np.repeat(g[sel] * Kperps[:, pts], num, axis=1)])
        new_Ms = np.repeat(Ms[:, pts], num, axis=1)
        new_key = np.repeat(key[pts], num)

        tad = -util.colsum_prod(new_Ks, new_Ms)
        if np.any(tad > Q):
            i = int(np.argmax(tad > Q))
            raise RuntimeError(
                f"Tadpole violation: -dot(K,M)={tad[i]} > Q={Q}. "
                f"K={new_Ks[:, i].tolist()}, M={new_Ms[:, i].tolist()}")
        Ks_out.append(new_Ks); Ms_out.append(new_Ms); key_out.append(new_key)

    if not Ks_out:
        if verbosity >= 2:
            print("# PFVs after setting K0s = 0")
        return empty
    Ks, Ms, key = np.hstack(Ks_out), np.hstack(Ms_out), np.concatenate(key_out)
    if verbosity >= 2:
        print(f"# PFVs after setting K0s = {Ms.shape[1]}")

    # filter by N invertibility
    singular = []
    for i in range(0, Ms.shape[1], 5000):
        chunk = Ms[:, i:i+5000]
        Ns = util.exact_matmul(kappa.reshape(h11*h11, h11), chunk).reshape(h11, h11, -1)
        Ns = Ns.transpose(2, 0, 1)[:, 1:, 1:]
        singular.append(_check_singular(Ns))
    singular = np.concatenate(singular)
    if verbosity >= 2:
        print(f"{int(np.sum(singular))}/{len(singular)} 'PFVs' had det(N)=0 :(")
    Ks, Ms, key = Ks[:, ~singular], Ms[:, ~singular], key[~singular]
    if verbosity >= 2:
        print(f"# invertible = {Ms.shape[1]}")

    order = np.argsort(key, kind="stable")
    return Ks.T[order], Ms.T[order], key[order]

# below this many p-vectors, "auto" stays on the CPU (device start-up and
# transfer costs dominate)
_GPU_MIN_PS = 256


def _use_gpu(device, h11, n_ps, use_c_lattice, use_gcd_lattice, extra_lll):
    """Resolve coniZpM's `device` argument to use-the-GPU or not."""
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f'device must be "auto", "cpu" or "gpu", got {device!r}')
    if device == "auto":
        env = os.environ.get("PFVS_DEVICE", "auto").lower()
        if env not in ("auto", "cpu", "gpu"):
            raise ValueError(f'PFVS_DEVICE must be "auto", "cpu" or "gpu", got {env!r}')
        device = env
    if device == "cpu":
        return False
    from . import gpu
    reasons = []
    if not gpu.available():
        reasons.append(gpu.unavailable_reason())
    elif not 3 <= h11 <= gpu.max_h11():
        reasons.append(f"h11 = {h11} is outside the GPU build's 3..{gpu.max_h11()}")
    if not use_c_lattice or use_gcd_lattice or not extra_lll:
        reasons.append("the GPU path needs use_c_lattice=True, use_gcd_lattice=False, "
                       "extra_lll_reduction=True")
    if device == "gpu":
        if reasons:
            raise RuntimeError("device='gpu' unavailable: " + "; ".join(reasons))
        return True
    return not reasons and n_ps >= _GPU_MIN_PS


def _raise_kernel_status(p, status: int):
    """Raise IncompleteSearchError for a nonzero kernel status."""
    reason = {-2: "(more than max_N_pfvs outputs; increase max_N_pfvs)",
              -8: "(lattice-point coordinates exceed int32)",
              -9: "(the ellipsoid matrix is not positive definite, so its "
                  "lattice points cannot be enumerated)"}.get(status, "")
    raise util.IncompleteSearchError(
        f"p={np.asarray(p).tolist()}: conipfv_kernel returned status {status} {reason}")


# helpers for coniZpK and the exhaustive search
# ==============================================
def _thread_map(fn, items, n_threads):
    """[fn(x) for x in items], in threads (for C calls that release the GIL)."""
    if n_threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(n_threads, len(items))) as ex:
        return list(ex.map(fn, items))


def _batch_points(f, kappa, Mbasis, p_full, idx, n_threads, *args):
    """
    A batched C pipeline (`_coni_batch` or `_coni_zpk_batch`: f(kappa,
    Mbasis, ps, *args)) over the rows p_full[idx], in n_threads threads:
    (M, Kn, q, pidx, status) with pidx indexing p_full and status[i] the
    status of p_full[idx[i]].
    """
    idx = np.asarray(idx, dtype=np.int64)
    chunks = [c for c in np.array_split(idx, max(1, min(4 * n_threads, len(idx) // 16))) if len(c)]
    res = _thread_map(lambda c: f(kappa, Mbasis, p_full[c], *args), chunks, n_threads)
    h = p_full.shape[1]
    if not res:
        z = np.zeros((0, h), dtype=np.int64)
        return z, z.copy(), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), \
            np.zeros(0, dtype=np.int32)
    M = np.vstack([r[0] for r in res])
    Kn = np.vstack([r[1] for r in res])
    q = np.concatenate([r[2] for r in res])
    pidx = np.concatenate([c[r[3]] for c, r in zip(chunks, res)])
    status = np.concatenate([r[4] for r in res])
    return M, Kn, q, pidx, status


def _canonical(Ks, Ms, keys):
    """Deduplicate PFVs and sort them by key (p-vector), then (M, K)."""
    keys = np.asarray(keys)
    if len(keys) == 0:
        return Ks, Ms, keys.astype(np.int64)
    if Ks.dtype == object or Ms.dtype == object:
        rows = sorted({(int(k), tuple(int(x) for x in m), tuple(int(x) for x in kk))
                       for k, kk, m in zip(keys, Ks, Ms)})
        return (np.array([r[2] for r in rows], dtype=object),
                np.array([r[1] for r in rows], dtype=object),
                np.array([r[0] for r in rows], dtype=np.int64))
    h = Ms.shape[1]
    A = np.unique(np.hstack([keys[:, None].astype(np.int64), Ms, Ks]), axis=0)
    return A[:, 1 + h:], A[:, 1:1 + h], A[:, 0]


def _dilations(kappa, p_full, Ks, Ms, keys):
    """
    The dilation of each PFV (K, M) of the p-vector p_full[key]:
    delta = gcd((Z M)_r) / gcd(K_r) with Z = kappa . p (since
    (Z M)_r = delta K_r), as (numerator, denominator) integer arrays
    (object dtype if int64 could overflow).
    """
    keys = np.asarray(keys)
    n, h = len(keys), np.asarray(kappa).shape[0]
    big = Ks.dtype == object or Ms.dtype == object or \
        h * h * util.absmax(kappa) * util.absmax(p_full) * max(util.absmax(Ms), 1) >= 2**62
    dt = object if big else np.int64
    num = np.zeros(n, dtype=dt)
    for k in np.unique(keys):
        sel = np.flatnonzero(keys == k)
        Z = util.exact_matmul(np.asarray(kappa).reshape(h * h, h), p_full[k]).reshape(h, h)
        ZM = util.exact_matmul(np.asarray(Ms[sel]), np.asarray(Z).T)
        num[sel] = np.gcd.reduce(np.abs(np.asarray(ZM, dtype=dt)[:, 1:]), axis=1)
    den = np.gcd.reduce(np.abs(np.asarray(Ks, dtype=dt)[:, 1:]), axis=1) if n else np.zeros(0, dtype=dt)
    return num, den


def _in_kahler_cone(H, ps):
    """Whether H p >= 1 for every row p of ps (in float64 -- BLAS -- when
    every partial sum is an integer below 2^53, so exact; else exactly)."""
    H, ps = np.asarray(H), np.asarray(ps)
    if H.dtype != object and ps.dtype != object and np.issubdtype(ps.dtype, np.integer) \
            and H.shape[1] * util.absmax(H) * util.absmax(ps) < 2**53:
        for i in range(0, len(ps), 1 << 18):
            if not np.all(H.astype(np.float64) @ ps[i:i + (1 << 18)].T.astype(np.float64) >= 1):
                return False
        return True
    return bool(np.all(H @ ps.T >= 1))


def _validate_ps(data, ps, checked=False):
    if not data.coni:
        raise ValueError("coniZpM/coniZpK only apply to coni contexts. Use Zp.py for non-coni PFVs.")
    if len(ps) == 0:
        raise ValueError("ps must be non-empty.")
    ps = np.array(ps)
    if not checked and not _in_kahler_cone(data.H_cob, ps):
        raise ValueError("some p-vectors are not in the kahler cone (H_cob@ps.T >= 1 failed)")
    p_int = util._as_integral(ps)
    if p_int.dtype == object:
        raise util.IncompleteSearchError("p-vector entries exceed int64")
    return ps, np.hstack([np.zeros((len(ps), 1), dtype=np.int64), p_int.astype(np.int64)])


def coniZpK(
    data: CYData,
    ps: ArrayLike,
    D0: int,
    Q: int | None = None,
    M0min: int = 13,
    n_jobs: int = -1,
    max_N_pfvs: int = 1_000_000_000,
    return_formal_pfvs: bool = False,
    verbosity: int = 0,
    ) -> tuple[ArrayLike, ArrayLike]:
    """
    The coni PFVs of each p-vector with dilation above D0: those with
    p = p_hat / delta, delta > D0, for the direction p_hat given. Together
    with ``coniZpM(..., ellipsoid_dilation=D0)``, which finds those with
    delta <= D0, this is every coni PFV of the direction.

    Each PFV above D0 has a short K_r: with A = kappa . p_hat,
    S = -A_rr^-1 and s the Schur complement of A_rr (see `pfvs.dilation`),
    the tadpole forces K_r^T S K_r < Q / D0 when s < 0 and S is positive
    definite on the lattice {K : p_r . K = 0}. ZpK enumerates those K_r and,
    for each, the PFVs it belongs to (exact C kernel, coni_zpk.h). A
    p-vector it cannot handle exactly is searched by ``coniZpM`` up to its
    dilation bound instead.

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY.
    ps : iterable of shape (N, h11-1)
        The p-vectors (directions) without their coni entry, p[1:], as for
        ``coniZpM``.
    D0 : int
        Find the PFVs with dilation > D0 (a positive integer).
    Q, M0min, n_jobs, max_N_pfvs, return_formal_pfvs, verbosity :
        As for ``coniZpM`` (n_jobs: threads; the C kernel releases the GIL).

    Returns
    -------
    Ks, Ms : ndarrays of shape (N, h11), or a list of PFV objects
        The PFVs, grouped by p-vector (in the order of ps) and sorted by
        (M, K) within one.

    Raises
    ------
    IncompleteSearchError
        For a p-vector whose PFVs above D0 cannot be enumerated: s >= 0, or
        S is not positive definite on the lattice (then its PFVs' dilation
        is not bounded this way), or its search is not exact.
    """
    ps, p_full = _validate_ps(data, ps)
    if int(D0) != D0 or D0 < 1:
        raise ValueError(f"D0 must be a positive integer, got {D0}")
    D0 = int(D0)
    kappa, Mbasis, h11 = data.kappa_cob, data.M_lattice(), data.h11
    if Q is None:
        Q = h11 + data.h21 + 4
    if n_jobs == -1:
        n_jobs = os.cpu_count()

    Mv, Kv, qv, pv, st = _batch_points(_coni_zpk_batch, kappa, Mbasis, p_full,
                                       np.arange(len(ps)), n_jobs, Q, M0min, D0, max_N_pfvs)
    Ks, Ms, keys = _pfvs_from_points(Mv.T, Kv.T, qv, pv, kappa, h11, Q, M0min, verbosity)
    pieces = [(Ks, Ms, keys)]
    bad = np.flatnonzero(st != 0)
    if verbosity >= 1:
        print(f"ZpK: {len(qv)} lattice points for {len(ps)} p-vectors "
              f"({len(bad)} searched by coniZpM to their bound instead)")
    if len(bad):
        from .dilation import coni_dilation_bounds
        for i, b in zip(bad, coni_dilation_bounds(ps[bad], kappa, Q)):
            if b is None:
                raise util.IncompleteSearchError(
                    f"p={ps[i].tolist()}: its PFVs above D0 cannot be enumerated (s >= 0, or "
                    f"S not positive definite on the lattice orthogonal to p)")
            K1, M1 = coniZpM(data, ps[i][None], Q=Q, M0min=M0min,
                             ellipsoid_dilation=max(math.ceil(b), D0), n_jobs=1, device="cpu",
                             max_N_pfvs=max_N_pfvs)
            pieces.append((K1, M1, np.full(len(K1), i, dtype=np.int64)))
    Ks = np.vstack([pc[0] for pc in pieces])
    Ms = np.vstack([pc[1] for pc in pieces])
    keys = np.concatenate([pc[2] for pc in pieces])
    num, den = _dilations(kappa, p_full, Ks, Ms, keys)
    above = np.asarray(num > D0 * den, dtype=bool)
    Ks, Ms, keys = _canonical(Ks[above], Ms[above], keys[above])
    if return_formal_pfvs:
        from .pfv import PFV
        return [PFV(data, K, M) for K, M in zip(Ks, Ms)]
    return Ks, Ms


# exhaustive search: each p-vector up to its dilation bound
# ========================================================
# candidate split dilations; p-vector counts above which the costs are
# measured on a sample (below: the default model -- only speed depends on it)
_SPLIT_D0S = (200, 400, 800, 1600)
_PLAN_MIN_PS = 4096
_PLAN_SAMPLE = 1 << 15
_CAL_PS = (32, 256)


def _grid_up(D):
    """The smallest value >= D (integers >= 1) on a grid of ratio 2^(1/16):
    ZpM at that dilation, so p-vectors with nearby bounds share one search
    (rounding up keeps it exhaustive)."""
    D = np.maximum(np.asarray(D, dtype=np.int64), 1)
    k = np.ceil(16 * np.log2(D)).astype(np.int64)
    g = np.ceil(2.0 ** (k / 16)).astype(np.int64)
    return np.where(g >= D, g, D)


# Measured costs per p-vector (us; 2026-10-05, Intel Core Ultra 7 270K,
# RTX 5090, dataset geometries' frontier p-vectors): ZpM on one core and on
# the GPU (wall, batches of 4000) at the dilations _COST_DS, and ZpK on one
# core above the split dilations _SPLIT_D0S.
_COST_DS = (150, 200, 400, 800, 1600, 3200, 6400)
_DEFAULT_COSTS = {
    4: ([3.35, 3.43, 2.88, 4.15, 7.06, 14.5, 28.4],
        [0.374, 0.406, 0.351, 0.359, 0.38, 0.417, 0.451],
        [2.62, 2.42, 2.13, 2.08]),
    5: ([7.4, 7.8, 13.2, 31.0, 75.8, 263.0, 582.0],
        [0.305, 0.312, 0.333, 0.349, 0.38, 0.439, 0.536],
        [6.62, 5.06, 4.23, 3.94]),
    6: ([9.69, 11.2, 18.8, 43.4, 115.0, 307.0, 805.0],
        [0.341, 0.35, 0.385, 0.442, 0.536, 0.69, 1.06],
        [14.4, 9.55, 7.49, 6.93]),
    7: ([13.2, 14.7, 23.2, 51.5, 136.0, 394.0, 1090.0],
        [0.467, 0.484, 0.489, 0.524, 0.592, 0.746, 1.12],
        [106.0, 35.8, 17.8, 13.0]),
    8: ([31.0, 42.3, 104.0, 296.0, 848.0, 2490.0, 7030.0],
        [0.491, 0.514, 0.565, 0.698, 1.02, 1.83, 3.96],
        [207.0, 49.6, 21.7, 16.7]),
    9: ([50.9, 71.9, 190.0, 544.0, 1580.0, 4350.0, 12400.0],
        [0.828, 0.85, 0.95, 1.16, 1.73, 3.17, 6.99],
        [499.0, 82.4, 31.0, 24.0]),
    10: ([114.0, 170.0, 483.0, 1400.0, 4040.0, 12200.0, 34700.0],
         [0.852, 0.902, 1.09, 1.58, 2.8, 6.01, 14.8],
         [1030.0, 122.0, 40.1, 31.5]),
    11: ([98.5, 139.0, 368.0, 1050.0, 3050.0, 8660.0, 24800.0],
         [1.04, 1.08, 1.22, 1.58, 2.5, 4.92, 11.5],
         [2570.0, 199.0, 53.3, 42.9]),
}


def _default_cost_model(use_gpu, h11=None, n_cpu=None):
    """
    The measured costs above for h11 (the nearest measured h11 outside
    4..11): ZpM on the GPU with ZpK on n_cpu cores (default: this machine's,
    at 75% efficiency), or ZpM and ZpK sharing the CPU.
    """
    from .dilation import CostModel
    h = min(max(int(h11 or 10), min(_DEFAULT_COSTS)), max(_DEFAULT_COSTS))
    cpu, gpu, zpk = _DEFAULT_COSTS[h]
    if use_gpu:
        cores = 0.75 * (n_cpu or os.cpu_count() or 1)
        return CostModel(_COST_DS, gpu, _SPLIT_D0S, [k / cores for k in zpk])
    return CostModel(_COST_DS, cpu, _SPLIT_D0S, zpk, shared=True)


def _zpm(data, p_full, idx, dil, Q, M0min, max_N_pfvs, use_gpu, gpu_required, n_jobs):
    """
    ZpM for the p-vectors p_full[idx], each at its own dilation dil[idx],
    in one GPU call or in CPU threads: [(Ks, Ms, keys)] with keys indexing
    p_full.
    """
    kappa, Mbasis, h11 = data.kappa_cob, data.M_lattice(), data.h11
    idx = np.asarray(idx, dtype=np.int64)
    if not len(idx):
        return []
    Ds, gi = np.unique(np.asarray(dil)[idx], return_inverse=True)
    pieces, redo = [], []
    if use_gpu:
        from . import gpu
        try:
            geoms = [dict(kappa=kappa, Mbasis=Mbasis, Q=Q, dilation=float(D), M0min=M0min) for D in Ds]
            M, Kn, q, pidx, pstat, _ = gpu.coni_batch_multi(geoms, p_full[idx], gi.astype(np.int32))
            counts = np.bincount(pidx, minlength=len(idx))
            pstat[counts > max_N_pfvs] = 1               # (the CPU path reports it)
            keep = pstat[pidx] == 0
            pieces.append(_pfvs_from_points(M[keep, :h11].T, Kn[keep, :h11].T, q[keep],
                                            idx[pidx[keep]], kappa, h11, Q, M0min))
            redo = list(idx[pstat != 0])
        except RuntimeError as e:
            if gpu_required:
                raise
            warnings.warn(f"GPU path failed ({e}); using the CPU", RuntimeWarning, stacklevel=4)
            use_gpu = False
    if not use_gpu:
        for k, D in enumerate(Ds):
            sel = idx[gi == k]
            M, Kn, q, pidx, st = _batch_points(_coni_batch, kappa, Mbasis, p_full, sel, n_jobs,
                                               Q, float(D), M0min, max_N_pfvs)
            for j in np.flatnonzero(st < 0):
                _raise_kernel_status(p_full[sel[j], 1:], int(st[j]))
            pieces.append(_pfvs_from_points(M.T, Kn.T, q, pidx, kappa, h11, Q, M0min))
            redo += list(sel[st == 1])
    for i in redo:                                       # the exact per-p path
        K1, M1 = coniZpM(data, p_full[i, 1:][None], Q=Q, M0min=M0min, ellipsoid_dilation=float(dil[i]),
                         n_jobs=1, device="cpu", max_N_pfvs=max_N_pfvs)
        pieces.append((K1, M1, np.full(len(K1), i, dtype=np.int64)))
    return pieces


def _calibrate(data, p_full, bceil, has, Q, M0min, max_N_pfvs, use_gpu, n_jobs, c):
    """
    Measure the costs of `bound_routing` on a sample of the p-vectors: ZpM
    at the split dilations and at quantiles of the bounds, ZpK above each
    split dilation (time per p-vector: on the GPU a larger sample, less the
    time of a one-p-vector call; on the CPU one thread).
    """
    from .dilation import CostModel
    idx = np.flatnonzero(has)
    rng = np.random.default_rng(0)
    ns = int(min(len(idx), _CAL_PS[1], max(_CAL_PS[0], len(idx) // 32)))
    s = np.sort(rng.choice(idx, ns, replace=False))
    sg = np.sort(rng.choice(idx, min(len(idx), 8 * _CAL_PS[1]), replace=False)) if use_gpu else s
    qs = np.quantile(bceil[idx], [0.5, 0.9, 0.99])
    Ds = sorted(set(_SPLIT_D0S) | {int(x) for x in qs if x >= 1})
    kappa, Mbasis = data.kappa_cob, data.M_lattice()

    def clock(fn):
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t

    def per_p(fn, sel, budget=0.05):
        """fn over pieces of sel until `budget` seconds: time per p-vector"""
        t, done = 0.0, 0
        for piece in np.array_split(sel, max(1, len(sel) // 16)):
            t += clock(lambda piece=piece: fn(piece))
            done += len(piece)
            if t > budget:
                break
        return t / done

    def zpm(sel, D):
        return _zpm(data, p_full, sel, np.full(len(p_full), D), Q, M0min, max_N_pfvs, use_gpu, False, 1)
    zpm(sg[:1], Ds[0])                                   # (warm-up: device start, caches)
    if use_gpu:                                          # (less the per-call overhead)
        t0 = clock(lambda: zpm(sg[:1], Ds[0]))
        G = [max(clock(lambda D=D: zpm(sg, D)) - t0, 1e-9) / len(sg) for D in Ds]
    else:
        G = [per_p(lambda sel, D=D: zpm(sel, D), s) for D in Ds]
    # (one thread: a small sample would not keep a pool busy; with a GPU,
    # ZpK's time on the pool is its time on one core over 0.75 n_jobs)
    K = [per_p(lambda sel, D0=D0: _coni_zpk_batch(kappa, Mbasis, p_full[sel], Q, M0min, D0, max_N_pfvs), s)
         for D0 in _SPLIT_D0S]
    if use_gpu:
        K = [k / (0.75 * n_jobs) for k in K]
    return CostModel(Ds, G, _SPLIT_D0S, K, c, shared=not use_gpu)


def _coni_exhaustive(data, ps, p_full, Q, M0min, dilation, use_gpu, gpu_required, n_jobs,
                     max_N_pfvs, cost_model, verbosity):
    """coniZpM(..., exhaustive=True): (Ks, Ms, keys, complete), keys the
    index of each PFV's p-vector.

    The plan (b*, D0) comes from a random sample of the bounds; then the
    bounds are computed in chunks (CPU threads), each chunk routed and handed
    to ZpM (one GPU thread) as soon as it is, and ZpK runs on the CPU
    threads once all bounds are in. So the GPU works while the CPU computes
    bounds and runs ZpK."""
    from .dilation import coni_dilation_bound_ceils
    kappa, Mbasis, h11 = data.kappa_cob, data.M_lattice(), data.h11
    n = len(ps)
    t0 = time.perf_counter()
    samp = np.sort(np.random.default_rng(0).choice(n, min(n, _PLAN_SAMPLE), replace=False))
    bs = coni_dilation_bound_ceils(ps[samp], kappa, Q, n_jobs=n_jobs)
    c = (time.perf_counter() - t0) / len(samp)
    if cost_model is None:
        cost_model = _calibrate(data, p_full[samp], bs, bs > 0, Q, M0min, max_N_pfvs, use_gpu,
                                n_jobs, c) \
            if np.sum(bs > 0) * n / len(samp) >= _PLAN_MIN_PS else _default_cost_model(use_gpu, h11)
    b_star, D0, t = cost_model.route(np.where(bs > 0, bs, np.inf))
    D0 = int(D0)
    t2 = time.perf_counter()

    # ZpM at ceil(b) >= b finds every PFV of the direction (delta < b)
    bceil = np.zeros(n, dtype=np.int64)
    dil = np.zeros(n)
    gq, pieces = [], []

    def route(idx):
        bceil[idx] = coni_dilation_bound_ceils(ps[idx], kappa, Q, n_jobs=n_jobs)
        b = bceil[idx]
        to_bound = (b > 0) & (b <= max(b_star, D0))
        dil[idx] = np.where(to_bound, _grid_up(b), np.where(b > 0, D0, 0))
        dil[idx[b == 0]] = dilation

    chunks = np.array_split(np.arange(n), max(1, min(64, n // 65536)))
    with ThreadPoolExecutor(1) as gpu_ex:
        for idx in chunks:
            route(idx)
            if use_gpu:           # (one device call per chunk, in order, on one thread)
                gq.append(gpu_ex.submit(_zpm, data, p_full, idx, dil, Q, M0min, max_N_pfvs,
                                        True, gpu_required, n_jobs))
        t3 = time.perf_counter()
        has = bceil > 0
        split = has & (bceil > max(b_star, D0))
        idx_k = np.flatnonzero(split)
        if not use_gpu:           # ZpM in its own thread, beside ZpK's threads
            gq.append(gpu_ex.submit(_zpm, data, p_full, np.arange(n), dil, Q, M0min, max_N_pfvs,
                                    False, False, n_jobs))
        Mv, Kv, qv, pv, st = _batch_points(_coni_zpk_batch, kappa, Mbasis, p_full, idx_k, n_jobs,
                                           Q, M0min, D0, max_N_pfvs)
        t4 = time.perf_counter()
        for f in gq:
            pieces += f.result()
    t5 = time.perf_counter()
    if verbosity >= 1:
        print(f"exhaustive: {n} p-vectors, {int(np.sum(~has))} without a bound (searched at "
              f"dilation {dilation}), {int(np.sum(has & ~split))} by ZpM to their bound "
              f"(b* = {b_star:g}), {len(idx_k)} split at D0 = {D0}; {cost_model}")
    pieces.append(_pfvs_from_points(Mv.T, Kv.T, qv, pv, kappa, h11, Q, M0min))
    # split p-vectors ZpK could not do exactly: ZpM to their bound instead
    redo = idx_k[st != 0]
    if len(redo):
        dil2 = np.zeros(n)
        dil2[redo] = _grid_up(bceil[redo])
        pieces += _zpm(data, p_full, redo, dil2, Q, M0min, max_N_pfvs, use_gpu, False, n_jobs)
    Ks = np.vstack([pc[0] for pc in pieces])
    Ms = np.vstack([pc[1] for pc in pieces])
    keys = np.concatenate([pc[2] for pc in pieces])
    Ks, Ms, keys = _canonical(Ks, Ms, keys)
    if verbosity >= 1:
        print(f"exhaustive: plan {t2 - t0:.2f} s, bounds {t3 - t2:.2f} s (with ZpM on the GPU), "
              f"ZpK {t4 - t3:.2f} s, ZpM done {t5 - t2:.2f} s after the plan, the rest "
              f"{time.perf_counter() - t5:.2f} s")
    return Ks, Ms, keys, has


# coni Zp
# =======
def coniZpM(
    # problem definition
    data: CYData,
    ps: ArrayLike,
    Q: int | None = None,
    M0min: int = 13,
    ellipsoid_dilation: float = 1, # typically want >=1
    # algorithm selection
    use_gcd_lattice: bool = False,
    use_c_lattice: bool = True,
    low_level_parallelism: bool = False,
    n_jobs: int = -1,
    # misc
    extra_checks: bool = False,
    extra_lll_reduction: bool = True,
    device: str = "auto",
    # output/verbosity
    max_N_pfvs: int = 1_000_000_000,
    return_formal_pfvs: bool = False,
    verbosity: int = 0,
    # exhaustive search
    exhaustive: bool = False,
    cost_model=None,
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
    ellipsoid_dilation : float, optional
        The dilation of the ellipsoid. Typically want >>1 to capture more PFVs.
        Empirically, runtime scales linearly with this value. Defaults to 1.
    use_c_lattice : bool, optional
        Whether to build each p-vector's lattice data (the M-lattice basis
        Binter, the ellipsoid and the H-matrix) in C (fast, exact, with
        automatic fallback to the Python path on overflow). Either way the
        PFVs of each p-vector are listed in a canonical order (by M, then K),
        independent of the lattice basis. Defaults to True.
    use_gcd_lattice : bool, optional
        Whether to construct explicit lattice bases for guaranteeing sufficient
        GCD of Kperp. Not recommended - it's generally quicker to just prune FP.
        Defaults to False.
    low_level_parallelism : bool, optional
        Deprecated, no effect beyond forcing n_jobs = 1 (warns if set): the
        gcd step it parallelized is now vectorized. Parallelize over
        p-vectors with n_jobs instead.
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
    device : str, optional
        Where the lattice setup and search run: "cpu", "gpu" (the GPU
        backend, NVIDIA or AMD; raises if it is not built or no device is
        present) or
        "auto" (the GPU when available and worthwhile, else the CPU; the
        environment variable PFVS_DEVICE overrides "auto"). Results are
        identical either way. Defaults to "auto".
    max_N_pfvs : int, optional
        The maximum number of PFVs that can be output. The C-kernel requires a
        limit. Defaults excessively high to 1,000,000,000.
    return_formal_pfvs : bool, optional
        Whether to return "PFV" objects as in pfv.py. Otherwise, an
        array of K-vectors (as rows) and an array of M-vectors (as rows) are
        returned. Defaults to False.
    verbosity : int, optional
        The verbosity level. Higher is more verbose. Defaults to 0.
    exhaustive : bool, optional
        Find every coni PFV of each direction, at any dilation, instead of
        those within ``ellipsoid_dilation``. Each p-vector's dilation bound
        b (``pfvs.dilation``: every coni PFV of the direction has
        delta < b) decides how: ZpM up to the bound, or ZpM up to a split
        dilation D0 plus ``coniZpK`` above it, whichever balances the GPU
        and the CPU best (``pfvs.dilation.bound_routing``; on the CPU alone,
        whichever is cheaper). Results are the same either way. A p-vector
        without a bound (the bound's hypotheses fail) is searched up to
        ``ellipsoid_dilation`` as usual and reported as incomplete. Defaults
        to False.
    cost_model : pfvs.dilation.CostModel, optional
        The costs the exhaustive search plans with. Defaults to measuring
        them on a sample of the p-vectors (from 4096 p-vectors with a
        bound; fewer use a fixed rough model).

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
    complete : ndarray of bool, shape (len(ps),)
        Only if exhaustive=True: whether each p-vector's search found every
        coni PFV of its direction. The PFVs are then grouped by p-vector and
        sorted by (M, K) within one.
    """
    n_threads = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    if not data.coni:
        raise ValueError(
            "coniZpM only applies to coni contexts. "
            "Use Zp.py for non-coni PFVs."
        )
    if len(ps) == 0:
        raise ValueError("ps must be non-empty.")
    ps = np.array(ps)
    if not _in_kahler_cone(data.H_cob, ps):
        raise ValueError("some p-vectors are not in the kahler cone (H_cob@ps.T >= 1 failed)")
    if ellipsoid_dilation <= 0:
        raise ValueError(f"ellipsoid_dilation must be > 0, got {ellipsoid_dilation}.")

    if low_level_parallelism:
        util.warn_unused("low_level_parallelism",
                         "the gcd step it parallelized is now vectorized; "
                         "parallelize over p-vectors with n_jobs instead", stacklevel=2)
        if n_jobs != 1:
            print("Setting n_jobs = 1 since low_level_parallelism = True...")
            n_jobs = 1
    if extra_checks:
        util.warn_unused("extra_checks", "the ellipsoid matrix is always computed in exact integer arithmetic", stacklevel=2)
        extra_checks = False
    if n_jobs == -1:
        n_jobs = 2*os.cpu_count()


    # read data
    kappa  = data.kappa_cob
    h11    = data.h11
    h21    = data.h21
    proj   = _get_proj(h11)
    Mbasis = data.M_lattice()

    if Q is None:
        Q = (h11+h21+2) + 2

    use_gpu = _use_gpu(device, h11, len(ps), use_c_lattice, use_gcd_lattice,
                       extra_lll_reduction)
    gpu_required = use_gpu and device == "gpu"
    if exhaustive:
        if not use_c_lattice or use_gcd_lattice or not extra_lll_reduction:
            raise ValueError("exhaustive=True needs use_c_lattice=True, use_gcd_lattice=False, "
                             "extra_lll_reduction=True")
        ps, p_full = _validate_ps(data, ps, checked=True)
        Ks, Ms, _, complete = _coni_exhaustive(
            data, ps, p_full, Q, M0min, ellipsoid_dilation, use_gpu, gpu_required,
            n_threads, max_N_pfvs, cost_model, verbosity)
        if return_formal_pfvs:
            from .pfv import PFV
            return [PFV(data, K, M) for K, M in zip(Ks, Ms)], complete
        return Ks, Ms, complete
    if use_gpu:
        n_jobs = 1      # one device call for all p-vectors

    # the search
    # ----------
    # iterate over p-vectors
    chunk_size = max(100, len(ps)//n_jobs+1)
    p_chunks   = [ps[i:i+chunk_size] for i in range(0,len(ps),chunk_size)]

    def _make_pfvs(p_chunk, job_i=0):
        # define a factory function here for later parallelization
        eye_U  = np.eye(h11-1)  # kernel signature needs U; mat= is what is used
        maxes  = (util.absmax(kappa), util.absmax(Mbasis))

        pieces = []                     # (Ks, Ms, order key = index in p_chunk)
        p_chunk = np.asarray(p_chunk)
        todo = range(len(p_chunk))

        # batched C path: lattice setup + kernel for all p at once; p-vectors
        # it cannot handle exactly (status 1) go through the per-p path below
        if use_c_lattice and not use_gcd_lattice and len(p_chunk):
            p_int = util._as_integral(p_chunk)       # raises if non-integral
            if p_int.dtype == object:
                raise util.IncompleteSearchError("p-vector entries exceed int64")
            p_full_all = np.hstack([np.zeros((len(p_chunk), 1), dtype=np.int64),
                                    p_int.astype(np.int64)])
            batch_done = False
            if use_gpu:
                from . import gpu
                try:
                    Mv, Kv, qv, pv, pstat = gpu.coni_batch(
                        kappa, Mbasis, p_full_all, Q, ellipsoid_dilation, M0min,
                        max_N_pfvs)
                    batch_done = True
                except RuntimeError as e:
                    if gpu_required:
                        raise
                    # "auto": e.g. device memory taken by other processes
                    warnings.warn(f"GPU path failed ({e}); using the CPU", RuntimeWarning,
                                  stacklevel=3)
            if not batch_done:
                Mv, Kv, qv, pv, pstat = _coni_batch(
                    kappa, Mbasis, p_full_all, Q, ellipsoid_dilation, M0min,
                    max_N_pfvs, extra_lll_reduction)
            for ip in np.flatnonzero(pstat < 0):
                _raise_kernel_status(p_chunk[ip], int(pstat[ip]))
            if verbosity >= 1:
                print(f"found {len(qv)} lattice points for {len(p_chunk)} p-vectors "
                      f"({int(np.sum(pstat == 1))} handled by the per-p path)...")
                if verbosity >= 10:
                    print("they were (M = Binter c, one per row):")
                    print(Mv)
            pieces.append(_pfvs_from_points(Mv.T, Kv.T, qv, pv, kappa, h11, Q,
                                            M0min, verbosity))
            todo = np.flatnonzero(pstat == 1)

        for ip in todo:
            p = p_chunk[ip]
            p_full = np.concatenate([[0],p])

            # construct the quadratic form defining the ellipsoid
            # (C path; status -15: all but H built; other nonzero: fall back)
            H_pre, st = None, -1
            if use_c_lattice and not use_gcd_lattice:
                st, Z, Binter, ZBinter, mat, H_pre = _lattice_build(
                    kappa, Mbasis, p_full.astype(np.int64), True,
                    extra_lll_reduction)
            if st not in (0, -15):
                try:
                    mat, Z, Binter = coni_M_ellipsoid(
                        p_full,
                        kappa=kappa,
                        Mbasis=Mbasis,
                        extra_lll_reduction=extra_lll_reduction,
                        extra_checks=extra_checks,
                        _maxes=maxes)
                except OverflowError as e:
                    raise util.IncompleteSearchError(f"p={p.tolist()}: {e}") from e
                ZBinter = util.exact_matmul(Z, Binter)

            ZBinter = np.ascontiguousarray(ZBinter)
            Binter  = np.ascontiguousarray(  Binter)

            # solve for lattice points under tadpole
            # ======================================
            try:
                if not use_gcd_lattice:
                    # find relevant lattice points in ellipsoid c.T@mat@c <= Q
                    # just uses FP with pruning on GCDs and M0 - no GCD lattice
                    try:
                        H = H_pre if H_pre is not None else coni_H_matrix(ZBinter, proj)
                    except Exception as e:
                        raise util.IncompleteSearchError(f"p={p.tolist()}: coni_H_matrix failed ({e})") from e

                    # U is unused when mat= is given (the kernel factors
                    # the exact mat itself), but is part of the signature
                    try:
                        lattice_points, rawQs, status = conipfv_kernel(
                            # ellipsoid definition
                            U=eye_U,
                            Q=Q,
                            dilation=ellipsoid_dilation,
                            # M0 cuts
                            linvec=np.ascontiguousarray(Binter[0,:], dtype=np.int64),
                            linmin=M0min,
                            # gcd cuts
                            H=H,
                            # misc
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
                        _raise_kernel_status(p, status)

                # use GCD lattices
                # ----------------
                else:
                    warnings.warn("the GCD-lattice branch is old code and may be stale", stacklevel=2)
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

            except util.IncompleteSearchError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Kernel failed for p={np.array(p).tolist()}: {type(e).__name__}: {e}"
                ) from e

            # post-processing (shared with the batched path)
            Mv = util.exact_matmul(Binter, np.asarray(lattice_points).T)
            Kv = util.exact_matmul(ZBinter, np.asarray(lattice_points).T)
            pieces.append(_pfvs_from_points(
                Mv, Kv, rawQs, np.full(len(rawQs), ip), kappa, h11, Q, M0min,
                verbosity))

        if verbosity > 0:
            print(f"Finished job #{job_i}...", flush=True)

        Ks   = np.vstack([pc[0] for pc in pieces] or [np.zeros((0, h11), dtype=np.int64)])
        Ms   = np.vstack([pc[1] for pc in pieces] or [np.zeros((0, h11), dtype=np.int64)])
        keys = np.concatenate([pc[2] for pc in pieces] or [np.zeros(0, dtype=np.int64)])
        order = np.argsort(keys, kind="stable")      # p-by-p order
        return Ks[order], Ms[order]

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
