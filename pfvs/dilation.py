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
# Description:  An upper bound on the dilation of a coni PFV with a given
#               direction, from the Schur complement of kappa . p_hat.
#
#               In the coni basis (conifold direction 0, the rest r), write
#               p = p_hat / delta with p_hat integral, p_hat_0 = 0, and
#               A = kappa . p_hat. Constraint 7 gives K_r = (A M)_r / delta,
#               so M_r = A_rr^-1 (delta K_r - a_r0 M0), and K' > 0 fixes
#               K0 = (A M)_0 / delta - eps with eps > 0. The tadpole becomes
#
#                 Q = M0 eps + (M0^2 / delta) (-s) + delta K_r^T S K_r,
#                 s = a00 - a0r^T A_rr^-1 a_r0,     S = -A_rr^-1.
#
#               If s <= 0 and S is positive definite on
#               Lambda = {K in Z^(h11-1) : p_hat_r . K = 0} (where every
#               solution's K_r lies, by K.p = 0, and K_r != 0 since N_rr is
#               invertible), all three terms are >= 0 and the first is > 0, so
#
#                 delta < Q / mu0,   mu0 = min_{0 != K in Lambda} K^T S K.
#
#               Everything here is exact: C (fp_kernel/coni_bound.h, integer
#               arithmetic, float-pruned exact enumeration), with this
#               module's python-flint version as the fallback and reference.
#               Both hypotheses are checked; if either fails, no bound is
#               returned.
#               Searching a direction up to its bound finds all of its coni
#               PFVs: coniZpM(..., exhaustive=True), planned by
#               bound_routing below (the bound is typically ~10^3 at
#               h11 >= 8, so most directions are split: ZpM up to D0 and
#               coniZpK above it). Also a diagnostic (PFV.dilation_bound).
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from fractions import Fraction

import flint
import numpy as np
from numpy.typing import ArrayLike

from . import util


def _frac(x) -> Fraction:
    """flint fmpq (or fmpz) -> Fraction."""
    return Fraction(int(x.p), int(x.q)) if hasattr(x, "q") else Fraction(int(x))


def _schur_form(p_r: ArrayLike, kappa: ArrayLike):
    """(s, S, B): the Schur complement s of A_rr in A = kappa . p_hat, the
    form S = -A_rr^-1 on K_r, and a basis B (columns) of Lambda; None if A_rr
    is singular."""
    kappa = np.asarray(kappa)
    h = kappa.shape[0]
    p_r = [int(x) for x in np.asarray(p_r).ravel()]
    if len(p_r) != h - 1:
        raise ValueError(f"p must have h11-1 = {h - 1} entries (p_hat without the coni entry)")
    p_full = np.array([0] + p_r, dtype=object)
    A = np.einsum("abc,c->ab", kappa.astype(object), p_full)
    Arr = flint.fmpq_mat([[int(x) for x in row[1:]] for row in A[1:]])
    if Arr.det() == 0:
        return None
    Arr_inv = Arr.inv()
    a = flint.fmpq_mat([[int(x)] for x in A[1:, 0]])
    s = _frac(int(A[0, 0]) - (a.transpose() * Arr_inv * a)[0, 0])
    B = util.orthogonal_lattice(np.array(p_r, dtype=object))
    B = flint.fmpq_mat([[int(x) for x in row] for row in np.asarray(B)])
    return s, -Arr_inv, B


def _int_gram(G: flint.fmpq_mat) -> tuple[flint.fmpz_mat, int]:
    """(L*G as an integer matrix, L) with L the lcm of G's denominators."""
    n = G.nrows()
    L = 1
    for i in range(n):
        for j in range(n):
            L = math.lcm(L, int(G[i, j].q))
    Gi = flint.fmpz_mat([[int(G[i, j] * L) for j in range(n)] for i in range(n)])
    return Gi, L


def _positive_definite(G: flint.fmpz_mat) -> bool:
    """Sylvester's criterion, exactly."""
    n = G.nrows()
    return all(
        flint.fmpz_mat([[G[i, j] for j in range(k)] for i in range(k)]).det() > 0
        for k in range(1, n + 1))


def _shortest_vector(G: flint.fmpz_mat) -> tuple[int, list[int]]:
    """Exact min of y^T G y over nonzero integer y, and a minimizer, for a
    positive-definite integer Gram matrix G (Fincke-Pohst on the LLL-reduced
    form; exact rational LDL^T, floats only to bound coordinate ranges, which
    are widened by one so no candidate is lost)."""
    n = G.nrows()
    R, T = G.lll(transform=True, rep="gram", gram="exact")   # R = T G T^T
    Gr = [[Fraction(int(R[i, j])) for j in range(n)] for i in range(n)]
    # LDL^T: y^T Gr y = sum_i d_i (y_i + sum_{j>i} m[j][i] y_j)^2
    d = [Fraction(0)] * n
    m = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        d[i] = Gr[i][i] - sum(m[i][k] ** 2 * d[k] for k in range(i))
        for j in range(i + 1, n):
            m[j][i] = (Gr[j][i] - sum(m[j][k] * m[i][k] * d[k] for k in range(i))) / d[i]
    # start from the best basis vector (a valid candidate), then look for
    # anything strictly shorter; the bound tightens as better vectors appear
    k0 = min(range(n), key=lambda k: Gr[k][k])
    best_val, best_y = Gr[k0][k0], [int(i == k0) for i in range(n)]
    y = [0] * n

    def rec(i: int, partial: Fraction):
        nonlocal best_val, best_y
        if i < 0:
            if any(y) and partial < best_val:
                best_val, best_y = partial, list(y)
            return
        c = -sum(m[j][i] * y[j] for j in range(i + 1, n))
        room = best_val - partial                        # >= 0 here
        r = math.sqrt(float(room / d[i]))
        lo, hi = math.floor(float(c) - r) - 1, math.ceil(float(c) + r) + 1
        for yi in range(lo, hi + 1):
            t = d[i] * (yi - c) ** 2
            if partial + t <= best_val:                   # exact prune
                y[i] = yi
                rec(i - 1, partial + t)
        y[i] = 0

    rec(n - 1, Fraction(0))
    yv = flint.fmpz_mat([best_y])
    val = int((yv * R * yv.transpose())[0, 0])
    # back to the original basis: y_orig = y T
    y_orig = [int(x) for x in (yv * T).entries()]
    return val, y_orig


def _coni_mu0_c(ps, kappa):
    """The C version on the rows of ps: (status, mu_num, mu_den, K), or None
    if the extension is unavailable. status 0: decided; 1: no bound; < 0:
    use the Python path."""
    try:
        from .fp_kernel.fp_kernel import _coni_mu0_batch
    except ImportError:
        return None
    return _coni_mu0_batch(np.asarray(kappa, dtype=np.int64), np.asarray(ps, dtype=np.int64))


def coni_mu0(p: ArrayLike, kappa: ArrayLike, use_c: bool = True) -> tuple[Fraction, list[int]] | None:
    """
    The minimum of K^T S K over nonzero K in Lambda, S = -A_rr^-1.

    Parameters
    ----------
    p : array of shape (h11-1,)
        The direction p_hat without its (zero) coni entry, in the coni basis.
    kappa : array of shape (h11, h11, h11)
        Triple intersection numbers in the coni basis (``CYData.kappa_cob``).

    Returns
    -------
    (mu0, K) : the exact minimum and a minimizing K (length h11-1), or None
        if the bound's hypotheses fail (A_rr singular, s > 0, or S not
        positive definite on Lambda).
    use_c : bool, optional
        Use the C version (falling back to Python where it cannot decide).
        False: the exact python-flint version only.
    """
    if use_c:
        r = _coni_mu0_c(np.atleast_2d(np.asarray(p, dtype=np.int64)), kappa)
        if r is not None and r[0][0] >= 0:
            st, num, den, K = r
            if st[0] == 1:
                return None
            return Fraction(int(num[0]), int(den[0])), [int(k) for k in K[0]]
    f = _schur_form(p, kappa)
    if f is None:
        return None
    s, S, B = f
    if s > 0:
        return None
    Gi, L = _int_gram(B.transpose() * S * B)
    if not _positive_definite(Gi):
        return None
    val, y = _shortest_vector(Gi)
    K = [_frac(x) for x in (B * flint.fmpq_mat([[v] for v in y])).entries()]
    return Fraction(val, L), [int(k) for k in K]


def coni_dilation_bound(p: ArrayLike, kappa: ArrayLike, Q: int) -> Fraction | None:
    """
    Upper bound on the dilation of every coni PFV with direction p_hat.

    Every coni PFV with -K.M = Q and p = p_hat / delta has delta < Q / mu0
    (see the module description). Searching p_hat at ellipsoid dilation
    Q / mu0 therefore finds all of them.

    Parameters
    ----------
    p : array of shape (h11-1,)
        The direction p_hat without its (zero) coni entry, in the coni basis
        (as ``coniZpM`` takes p-vectors).
    kappa : array of shape (h11, h11, h11)
        Triple intersection numbers in the coni basis (``CYData.kappa_cob``).
    Q : int
        The tadpole, -K.M.

    Returns
    -------
    Fraction or None
        Q / mu0 (exact; the bound is strict), or None if the hypotheses fail.
    """
    r = coni_mu0(p, kappa)
    if r is None:
        return None
    return Fraction(int(Q)) / r[0]


def coni_dilation_bound_ceils(ps: ArrayLike, kappa: ArrayLike, Q: int, n_jobs: int = 1) -> np.ndarray:
    """
    ceil(`coni_dilation_bound`) for each row of `ps`, as an int64 array (0:
    no bound): ZpM at that dilation finds every coni PFV of the direction.
    Like `coni_dilation_bounds`, without building a Fraction per row.
    """
    ps = np.atleast_2d(np.asarray(ps, dtype=np.int64))
    if n_jobs > 1 and len(ps) >= 2 * n_jobs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(n_jobs) as ex:      # (the C call releases the GIL)
            parts = list(ex.map(lambda c: _coni_mu0_c(c, kappa), np.array_split(ps, 4 * n_jobs)))
        r = None if parts[0] is None else tuple(np.concatenate(x) for x in zip(*parts))
    else:
        r = _coni_mu0_c(ps, kappa)
    out = np.zeros(len(ps), dtype=np.int64)
    if r is None:
        todo = np.arange(len(ps))
    else:
        st, num, den = r[0], r[1], r[2]
        ok = st == 0
        small = ok & (den < 2**62 // max(int(Q), 1))
        out[small] = -((-int(Q) * den[small]) // num[small])          # ceil(Q den / num)
        for i in np.flatnonzero(ok & ~small):
            out[i] = -((-int(Q) * int(den[i])) // int(num[i]))
        todo = np.flatnonzero(st < 0)
    for i in todo:                                                   # the Python path
        m = coni_mu0(ps[i], kappa, use_c=False)
        if m is not None:
            out[i] = math.ceil(Fraction(int(Q)) / m[0])
    return out


def coni_dilation_bounds(ps: ArrayLike, kappa: ArrayLike, Q: int,
                         n_jobs: int = 1) -> list[Fraction | None]:
    """
    `coni_dilation_bound` for each row of `ps` (shape (n, h11-1)), in C (the
    Python path only for rows the C version cannot decide), in `n_jobs`
    threads.
    """
    ps = np.atleast_2d(np.asarray(ps, dtype=np.int64))
    if n_jobs > 1 and len(ps) >= 2 * n_jobs:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(n_jobs) as ex:      # (the C call releases the GIL)
            parts = list(ex.map(lambda c: _coni_mu0_c(c, kappa), np.array_split(ps, 4 * n_jobs)))
        r = None if parts[0] is None else tuple(np.concatenate(x) for x in zip(*parts))
    else:
        r = _coni_mu0_c(ps, kappa)
    out = []
    for i, p in enumerate(ps):
        if r is not None and r[0][i] == 0:
            out.append(Fraction(int(Q)) / Fraction(int(r[1][i]), int(r[2][i])))
        elif r is not None and r[0][i] == 1:
            out.append(None)
        else:
            m = coni_mu0(p, kappa, use_c=False)
            out.append(None if m is None else Fraction(int(Q)) / m[0])
    return out


def bound_routing(bounds: ArrayLike, G, K, D0s: ArrayLike,
                  c: float = 0.0, shared: bool = False) -> tuple[float, float, float]:
    """
    How to search the p-vectors of one geometry exhaustively, given their
    dilation bounds: p goes GPU-to-bound (ZpM at dilation b(p)) if
    b(p) <= b*, else it is split at D0 (ZpM up to D0 on the GPU, ZpK above
    D0 on the CPU). Chooses D0 and b* to minimize the run time, with the GPU
    and the CPU running at the same time.

    Parameters
    ----------
    bounds : array of shape (n,)
        b(p) = Q / mu0(p) for each p-vector (`coni_dilation_bounds`).
        Non-finite entries (no bound: such p cannot be searched
        exhaustively) are ignored.
    G : callable
        G(D): mean GPU wall time per p-vector of ZpM at dilation D,
        increasing in D (vectorized).
    K : callable
        K(D0): mean CPU wall time per p-vector of ZpK above D0.
    D0s : array
        Candidate split dilations (where K was measured).
    c : float, optional
        CPU wall time per p-vector to compute b(p).
    shared : bool, optional
        ZpM and ZpK run on the same processors (no GPU), so the run takes
        T_GPU + T_CPU instead of the max: then p goes ZpM-to-bound exactly
        when that is cheaper than its split.

    Returns
    -------
    (b_star, D0, t) : the threshold (-inf: split every p), the split
        dilation, and the run time per p-vector, max(T_GPU, T_CPU) / n
        (T_GPU + T_CPU if shared).
    """
    # Each p is searched exhaustively in one of two ways (x_p = 1 or 0):
    #   GPU-to-bound (x_p = 1): G(b(p)) of GPU time
    #   split at D0  (x_p = 0): G(D0) of GPU time and K(D0) of CPU time
    # The GPU and CPU run at the same time, so the run takes
    # max(T_GPU, T_CPU) with
    #   T_GPU = sum_p [x_p G(b(p)) + (1 - x_p) G(D0)]
    #   T_CPU = sum_p (1 - x_p) K(D0) + n c.
    # Fix D0. Every p sent GPU-to-bound saves the same K(D0) of CPU time and
    # costs dG(p) = G(b(p)) - G(D0) more GPU time, which increases with
    # b(p). So for any number of p sent, T_GPU is smallest if they are those
    # with the smallest b(p): x_p = [b(p) <= b*]. With F(b) the fraction of
    # p with b(p) <= b, per p-vector:
    #   T_GPU / n = G(D0) + mean_p dG(p) [b(p) <= b*]
    #   T_CPU / n = (1 - F(b*)) K(D0) + c.
    # As b* rises, T_GPU increases and T_CPU decreases, so the max is
    # smallest where they cross. Then take the D0 with the smallest max.
    # (Several geometries sharing one GPU and one CPU pool: balance the
    # combined totals instead. At the optimum every geometry has the same
    # marginal price lam = (G(b*) - G(D0)) / K(D0), the GPU time paid per
    # CPU time saved; otherwise moving work between geometries would shorten
    # the run.)
    # (shared: the run takes T_GPU + T_CPU, smallest where sending one more
    # p stops paying, dG(p) >= K(D0): the same threshold rule, at lam = 1.)
    b = np.sort(np.asarray(bounds, dtype=float))
    b = b[np.isfinite(b)]
    n = len(b)
    best = (-math.inf, math.nan, math.inf)
    for D0 in np.atleast_1d(D0s):
        gD0, kD0 = float(G(D0)), float(K(D0))
        # T_GPU / n and T_CPU / n after sending the k smallest bounds,
        # k = 0..len(b); bounds below D0 have dG < 0 and are always sent
        t_gpu = gD0 + np.r_[0.0, np.cumsum(np.asarray(G(b), dtype=float) - gD0)] / max(n, 1)
        t_cpu = (n - np.arange(len(b) + 1)) / max(n, 1) * kD0 + c
        t = t_gpu + t_cpu if shared else np.maximum(t_gpu, t_cpu)
        k = int(np.argmin(t))
        if t[k] < best[2]:
            best = (b[k - 1] if k else -math.inf, float(D0), float(t[k]))
    return best


class CostModel:
    """
    Measured search costs for one geometry, as `bound_routing` takes them:
    the mean wall time per p-vector of ZpM at dilations D (on the device it
    runs on) and of ZpK above split dilations D0 (on the CPU), and of
    computing the bound. G between and beyond the measured dilations is
    interpolated linearly in log-log (extrapolated with the last slope).

    Parameters
    ----------
    D, G : sequences
        Dilations (increasing) and ZpM's time per p-vector at each (made
        nondecreasing).
    D0s, K : sequences
        Split dilations (integers) and ZpK's time per p-vector above each.
    c : float, optional
        Time per p-vector to compute the bound.
    shared : bool, optional
        ZpM and ZpK share the same processors (no GPU): see `bound_routing`.
    """

    def __init__(self, D, G, D0s, K, c: float = 0.0, shared: bool = False):
        self.D = np.asarray(D, dtype=float)
        # (bound_routing assumes G increasing: noisy measurements are made so)
        self.G = np.maximum.accumulate(np.maximum(np.asarray(G, dtype=float), 1e-12))
        self.D0s = [int(d) for d in D0s]
        self.K = dict(zip(self.D0s, (float(k) for k in K)))
        self.c, self.shared = float(c), bool(shared)
        if len(self.D) < 1 or np.any(np.diff(self.D) <= 0) or len(self.D0s) != len(self.K):
            raise ValueError("CostModel: D must be increasing, one K per D0")

    def __repr__(self):
        return (f"CostModel(D={self.D.tolist()}, G={self.G.tolist()}, D0s={self.D0s}, "
                f"K={list(self.K.values())}, c={self.c}, shared={self.shared})")

    def G_at(self, D):
        """ZpM's time per p-vector at dilation(s) D."""
        lD, lG = np.log(self.D), np.log(self.G)
        x = np.log(np.maximum(np.asarray(D, dtype=float), 1.0))
        if len(lD) == 1:
            return np.exp(lG[0] + (x - lD[0]))                   # (linear in D)
        y = np.interp(x, lD, lG)
        lo, hi = x < lD[0], x > lD[-1]
        y = np.where(lo, lG[0] + (lG[1] - lG[0]) / (lD[1] - lD[0]) * (x - lD[0]), y)
        y = np.where(hi, lG[-1] + (lG[-1] - lG[-2]) / (lD[-1] - lD[-2]) * (x - lD[-1]), y)
        return np.exp(y)

    def K_at(self, D0):
        """ZpK's time per p-vector above D0 (one of D0s)."""
        return self.K[int(D0)]

    def route(self, bounds: ArrayLike) -> tuple[float, float, float]:
        """`bound_routing` with these costs: (b_star, D0, t)."""
        return bound_routing(bounds, self.G_at, self.K_at, self.D0s, self.c, self.shared)

    def route_at_price(self, bounds: ArrayLike, lam: float) -> tuple[np.ndarray, int]:
        """
        The routing rule at a given price lam (GPU time per CPU time; see
        `bound_routing`: when several geometries share the hardware, lam is
        common to all and set by balancing the totals). The split dilation
        D0 minimizes G(D0) + lam K(D0), the cost of a split, and p goes to
        the bound iff G(b) - G(D0) <= lam K(D0).

        Returns (to_bound, D0): a mask over bounds (False where a bound is
        not finite) and the split dilation.
        """
        b = np.asarray(bounds, dtype=float)
        D0 = min(self.D0s, key=lambda d: float(self.G_at(d)) + lam * self.K[d])
        fin = np.isfinite(b)
        cost = self.G_at(np.where(fin, b, 1.0))
        return fin & (cost <= float(self.G_at(D0)) + lam * self.K[D0]), int(D0)

