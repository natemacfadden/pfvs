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
# Description:  An exact upper bound on the dilation of a coni PFV with a
#               given direction p_hat (p = p_hat / delta, coni basis).
#
#               With A = kappa . p_hat, s = a00 - a0r^T A_rr^-1 a_r0 and
#               S = -A_rr^-1: if s <= 0 and S is positive definite on
#               Lambda = {K in Z^(h11-1) : p_hat_r . K = 0}, the tadpole forces
#
#                 delta < Q / mu0,   mu0 = min_{0 != K in Lambda} K^T S K.
#
#               Otherwise there is no bound. Computed in C (coni_bound.h), with
#               python-flint as fallback. Used by coniZpM(exhaustive=True), via
#               bound_routing, and as a diagnostic (PFV.dilation_bound).
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
    positive-definite integer Gram matrix G (exact Fincke-Pohst)."""
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
    # start from the best basis vector; the bound tightens as better ones appear
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
    """ceil(`coni_dilation_bound`) for each row of `ps` as int64 (0: no bound)."""
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
    """`coni_dilation_bound` for each row of `ps` (shape (n, h11-1))."""
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
    Plan an exhaustive search of one geometry's p-vectors: p with
    b(p) <= b* run ZpM to their bound (GPU); the rest are split at D0 (ZpM up
    to D0 on the GPU, ZpK above it on the CPU). Chooses b* and D0 to minimize
    the run time, GPU and CPU running concurrently.

    Parameters
    ----------
    bounds : array of shape (n,)
        b(p) for each p-vector (`coni_dilation_bounds`); non-finite ignored.
    G : callable
        G(D): GPU time per p-vector of ZpM at dilation D (vectorized,
        increasing).
    K : callable
        K(D0): CPU time per p-vector of ZpK above D0.
    D0s : array
        Candidate split dilations.
    c : float, optional
        CPU wall time per p-vector to compute b(p).
    shared : bool, optional
        ZpM and ZpK share the processors (no GPU): minimize T_GPU + T_CPU
        instead of the max.

    Returns
    -------
    (b_star, D0, t) : the threshold (-inf: split every p), the split
        dilation, and the run time per p-vector.
    """
    # For fixed D0, sending p to its bound saves K(D0) of CPU time and costs
    # G(b(p)) - G(D0) of GPU time, which grows with b(p); so the best set is
    # {b(p) <= b*}, and b* is where T_GPU and T_CPU cross. (shared: where
    # sending one more p stops paying.)
    b = np.sort(np.asarray(bounds, dtype=float))
    b = b[np.isfinite(b)]
    n = len(b)
    best = (-math.inf, math.nan, math.inf)
    for D0 in np.atleast_1d(D0s):
        gD0, kD0 = float(G(D0)), float(K(D0))
        # T_GPU / n and T_CPU / n after sending the k smallest bounds
        t_gpu = gD0 + np.r_[0.0, np.cumsum(np.asarray(G(b), dtype=float) - gD0)] / max(n, 1)
        t_cpu = (n - np.arange(len(b) + 1)) / max(n, 1) * kD0 + c
        t = t_gpu + t_cpu if shared else np.maximum(t_gpu, t_cpu)
        k = int(np.argmin(t))
        if t[k] < best[2]:
            best = (b[k - 1] if k else -math.inf, float(D0), float(t[k]))
    return best


class CostModel:
    """
    Measured per-p-vector search costs of one geometry, for `bound_routing`.
    G is interpolated (and extrapolated) linearly in log-log.

    Parameters
    ----------
    D, G : sequences
        Increasing dilations and ZpM's time per p-vector at each.
    D0s, K : sequences
        Split dilations (integers) and ZpK's time per p-vector above each.
    c : float, optional
        Time per p-vector to compute the bound.
    shared : bool, optional
        As in `bound_routing`.
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
        Routing at a price lam (GPU time per CPU time saved, shared by
        geometries on the same hardware): D0 minimizes G(D0) + lam K(D0), and
        p goes to its bound iff G(b) - G(D0) <= lam K(D0). Returns
        (to_bound mask, D0).
        """
        b = np.asarray(bounds, dtype=float)
        D0 = min(self.D0s, key=lambda d: float(self.G_at(d)) + lam * self.K[d])
        fin = np.isfinite(b)
        cost = self.G_at(np.where(fin, b, 1.0))
        return fin & (cost <= float(self.G_at(D0)) + lam * self.K[D0]), int(D0)

