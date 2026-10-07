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
# Description:  Score geometries by how many PFVs a search finds.
#
#               score_geometries: the number of PFVs coniZpM (coni) or ZpM
#               (non-coni) finds on each geometry's first N p-vectors.
#               estimate_coni_pfvs / expected_pfvs_per_p (coni only): a
#               Gaussian-heuristic prediction of that count, without searching. Overcounts 3-5x,
#               mostly because it ignores det N != 0. Ranks worse than the
#               search (Spearman 0.89 vs 0.98 on 63 dataset conifolds).
#
#               Heuristic for one p, with (mat, Z, Binter) = coni_M_ellipsoid(p),
#               l = Binter[0], s_i the Smith invariants of (Z@Binter)[1:]:
#                   E_p = sum_{g <= dilation} prod_i gcd(s_i, g)/g
#                         * Vol_n(Q g)/sqrt(det mat) * F_n(M0min/rho)/rho,
#                   rho = sqrt(Q g * l^T mat^-1 l),
#               i.e. the volume of {c^T mat c <= Q g, l.c >= M0min}, weighted by
#               the ~1/M0 chance of hitting the tadpole exactly.
# -----------------------------------------------------------------------------

# external imports
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import flint
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

# local imports
from .coni import coni_M_ellipsoid, coniZpM
from .cydata import CYData
from .pvectors import pvecs
from .nonconi import ZpM

# helpers
# =======
# grid for the M0 >= M0min cap integral F_n (see module description)
_T = np.linspace(0.0, 1.0, 20001)[1:]


@lru_cache(maxsize=None)
def _tail_table(n: int) -> np.ndarray:
    """
    F_n on the grid _T: int_{t0}^1 (1-t^2)^((n-1)/2) / t dt, normalized by the
    marginal's total mass int_{-1}^1 (1-t^2)^((n-1)/2) dt.
    """
    w    = (1 - _T**2) ** ((n - 1) / 2)
    norm = 2 * np.trapezoid(np.r_[1.0, w], np.r_[0.0, _T])
    f    = w / _T
    seg  = 0.5 * (f[1:] + f[:-1]) * np.diff(_T)
    return np.r_[np.cumsum(seg[::-1])[::-1], 0.0] / norm


def _tail(n: int, t0: np.ndarray) -> np.ndarray:
    """Evaluate F_n(t0) by interpolation (t0 clipped to the grid)."""
    return np.interp(np.clip(t0, _T[0], None), _T, _tail_table(n), right=0.0)


def _sample_frontier(data: CYData, N: int, n_samp: int, rng: np.random.Generator,
                     N0: int) -> np.ndarray:
    """
    Roughly uniform sample of `pvecs(data, N)` without enumerating it: scale
    pvecs(data, N0) up by (N/N0)^(1/n), with jitter. Undersamples near the
    cone walls, which lowers the estimate but not the ranking.
    """
    H  = data.H_cob
    P0 = pvecs(data, min(N, N0))
    n  = P0.shape[1]
    s  = (N / len(P0)) ** (1 / n)
    if s <= 1:
        return P0[rng.choice(len(P0), size=min(n_samp, len(P0)), replace=False)]

    out, n_out = [], 0
    for _ in range(1000):
        base = P0[rng.integers(len(P0), size=4 * n_samp)]
        q    = np.rint(s * (base + rng.random(base.shape) - 0.5)).astype(np.int64)
        ok   = (H @ q.T >= 1).all(axis=0) & (np.gcd.reduce(np.abs(q), axis=1) == 1)
        out.append(q[ok])
        n_out += int(ok.sum())
        if n_out >= n_samp:
            break
    return np.concatenate(out)[:n_samp]


# above this many p-vectors, estimate_coni_pfvs(N=...) samples by scaling
# instead of enumerating the search's p-vectors
_ENUM_MAX = 10_000_000

# p-vectors per search call in _count_pfvs, bounding memory (non-coni finds
# ~10 PFVs per p, coni far fewer)
_COUNT_CHUNK = {True: 1_000_000, False: 100_000}


def _first_N(data: CYData, N: int, seed: int = 0) -> np.ndarray:
    """
    Exactly N of `pvecs(data, N)`, which returns whole L-inf boxes: all of
    the inner box plus a random subset of the outermost shell.
    """
    ps = pvecs(data, N)
    if len(ps) <= N:
        return ps
    rad   = np.abs(ps).max(axis=1)
    inner = np.flatnonzero(rad < rad.max())
    shell = np.flatnonzero(rad == rad.max())
    keep  = np.random.default_rng(seed).choice(shell, size=N - len(inner), replace=False)
    return ps[np.sort(np.concatenate([inner, keep]))]


# per-p estimate
# ==============
def expected_pfvs_per_p(
    p: ArrayLike,
    data: CYData = None,
    kappa: ArrayLike = None,
    Mbasis: ArrayLike = None,
    Q: int | None = None,
    ellipsoid_dilation: float = 1,
    M0min: int = 13,
    return_cumulative: bool = False) -> float | np.ndarray:
    """
    Gaussian-heuristic expected number of coniPFVs coniZpM finds from one p
    (formula in the module description).

    Parameters
    ----------
    p : ArrayLike of shape (h11,) or (h11-1,)
        The p-vector, as passed to coniZpM, or full.
    data : CYData, optional
        The CY. Or pass kappa (data.kappa_cob), Mbasis (data.M_lattice()) and
        Q instead, to avoid recomputing them per p.
    Q, ellipsoid_dilation, M0min : optional
        As in coniZpM.
    return_cumulative : bool, optional
        Return E_p for each integer dilation 1..ellipsoid_dilation. Defaults
        to False.

    Returns
    -------
    float or ndarray
        The expected count (0 if the ellipsoid is not positive definite).
    """
    if data is not None:
        if kappa is not None or Mbasis is not None:
            raise ValueError("kappa and Mbasis must be None when data is provided.")
        kappa, Mbasis = data.kappa_cob, data.M_lattice()
        if Q is None:
            Q = (data.h11 + data.h21 + 2) + 2
    elif kappa is None or Mbasis is None or Q is None:
        raise ValueError("Without data, kappa, Mbasis and Q must all be provided.")

    h11  = kappa.shape[0]
    n    = h11 - 1
    Dmax = max(1, int(math.floor(ellipsoid_dilation)))
    p    = np.asarray(p).ravel()
    if len(p) == n:
        p = np.concatenate([[0], p])

    # the ellipsoid and the maps M0 = l.c, Kperp = A c
    mat, Z, Binter = coni_M_ellipsoid(p, kappa=kappa, Mbasis=Mbasis)
    mat = np.asarray(mat, dtype=float)
    try:
        np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        zero = np.zeros(Dmax)
        return zero if return_cumulative else 0.0

    _, logdet = np.linalg.slogdet(mat)
    A     = np.asarray(Z @ Binter)[1:].astype(np.int64)
    S     = flint.fmpz_mat(A.tolist()).snf()
    s     = np.array([abs(int(S[i, i])) for i in range(n)], dtype=np.int64)
    lvec  = np.asarray(Binter[0], dtype=float)
    sigma = math.sqrt(lvec @ np.linalg.solve(mat, lvec))

    # sum over the Kperp gcd g (each g admits the dilation-g ellipsoid)
    g        = np.arange(1, Dmax + 1)
    log_dens = (np.log(np.gcd.outer(g, s)) - np.log(g)[:, None]).sum(axis=1)
    log_Vn   = (n / 2) * math.log(math.pi) - gammaln(n / 2 + 1)
    R2       = Q * g
    log_vol  = log_Vn + (n / 2) * np.log(R2) - 0.5 * logdet
    rho      = np.sqrt(R2) * sigma
    E        = np.cumsum(np.exp(log_dens + log_vol) * _tail(n, M0min / rho) / rho)

    return E if return_cumulative else float(E[-1])


# search-level estimate
# =====================
def estimate_coni_pfvs(
    data: CYData,
    ps: ArrayLike | None = None,
    N: int | None = None,
    ellipsoid_dilation: float = 1,
    Q: int | None = None,
    M0min: int = 13,
    n_samp: int = 512,
    N0: int = 100_000,
    seed: int = 0) -> float:
    """
    Predicted number of coniPFVs coniZpM finds on a set of p-vectors: the
    mean of `expected_pfvs_per_p` over n_samp of them, times their number.
    Overcounts 3-5x; compare estimates with each other, not with counts.

    Parameters
    ----------
    data : CYData
        The CY (coni).
    ps : ArrayLike of shape (n, h11-1), optional
        The p-vectors. Or pass N instead.
    N : integer, optional
        Use the same N p-vectors as `score_geometries`. Above 10^7 they
        are sampled without enumerating them (absolute level then depends on
        N0; keep it fixed).
    Q, ellipsoid_dilation, M0min : optional
        As in coniZpM.
    n_samp : integer, optional
        Number of p-vectors evaluated. Defaults to 512.
    N0 : integer, optional
        Base size for sampling above 10^7. Defaults to 100,000.
    seed : integer, optional
        Seed for the subsample. Defaults to 0.

    Returns
    -------
    float
        The predicted count.
    """
    if not data.coni:
        raise ValueError("estimate_coni_pfvs only applies to coni contexts.")
    if (ps is None) == (N is None):
        raise ValueError("Pass exactly one of ps and N.")

    rng = np.random.default_rng(seed)
    if ps is not None:
        ps    = np.asarray(ps)
        total = len(ps)
        if total > n_samp:
            ps = ps[rng.choice(total, size=n_samp, replace=False)]
    elif N <= _ENUM_MAX:
        ps    = _first_N(data, N, seed)         # the search's own p-vectors
        total = len(ps)
        if total > n_samp:
            ps = ps[rng.choice(total, size=n_samp, replace=False)]
    else:
        total = N
        ps    = _sample_frontier(data, N, n_samp, rng, N0)

    kappa, Mbasis = data.kappa_cob, data.M_lattice()
    if Q is None:
        Q = (data.h11 + data.h21 + 2) + 2

    E = [expected_pfvs_per_p(p, kappa=kappa, Mbasis=Mbasis, Q=Q,
                             ellipsoid_dilation=ellipsoid_dilation, M0min=M0min)
         for p in ps]
    return float(total * np.mean(E))


# scores by search
# =================
def _count_pfvs(
    data: CYData,
    ps: np.ndarray,
    ellipsoid_dilation: float = 1,
    **kwargs) -> int:
    """Internal: len(coniZpM or ZpM(data, ps, ...)), in chunks."""
    search, chunk = (coniZpM if data.coni else ZpM), _COUNT_CHUNK[data.coni]
    n_pfvs = 0
    for i in range(0, len(ps), chunk):
        n_pfvs += len(search(data, ps[i:i + chunk],
                             ellipsoid_dilation=ellipsoid_dilation, **kwargs)[0])
    return n_pfvs


def score_geometries(
    datas: list[CYData],
    N: int,
    ellipsoid_dilation: float = 1,
    method: str = "search",
    n_prefetch: int = 4,
    seed: int = 0,
    verbosity: int = 0,
    **kwargs) -> np.ndarray:
    """
    Score each geometry by the number of PFVs coniZpM (coni) or ZpM
    (non-coni) finds on its first N p-vectors (`pvecs(data, N)`, trimmed to
    exactly N). Coni: seconds per geometry on a GPU for N = 2M at dilation
    150. Non-coni runs on the CPU but finds ~10 PFVs per p, so N ~ 1e4-1e5
    suffices.

    Parameters
    ----------
    datas : list of CYData
        The geometries, coni or not (can be mixed, but scores are only
        comparable within a kind).
    N : integer
        p-vectors per geometry.
    ellipsoid_dilation : float, optional
        As in coniZpM / ZpM. Defaults to 1.
    method : str, optional
        "search" (default) counts; "estimate" uses `estimate_coni_pfvs`
        instead (coni only; no GPU needed, ranks worse).
    n_prefetch : integer, optional
        Worker processes generating the next geometries' p-vectors during the
        search; each holds N x h11 int64. Defaults to 4.
    seed : integer, optional
        Seed for the trimming. Defaults to 0.
    verbosity : integer, optional
        1 prints a line per geometry. Defaults to 0.
    **kwargs :
        Passed to the search (e.g. n_jobs; Q, M0min, device for coniZpM;
        Qmax, Qmin for ZpM) or to `estimate_coni_pfvs`.

    Returns
    -------
    ndarray of shape (len(datas),)
        scores[i] for datas[i].
    """
    if method not in ("search", "estimate"):
        raise ValueError(f"method must be 'search' or 'estimate', got {method!r}.")
    if method == "estimate" and not all(d.coni for d in datas):
        raise ValueError("method='estimate' only applies to coni contexts.")
    scores = np.zeros(len(datas))

    def report(i, t0):
        if verbosity >= 1:
            print(f"{i + 1}/{len(datas)}: {scores[i]:g} ({time.time() - t0:.1f} s)", flush=True)

    if method == "estimate":
        for i, data in enumerate(datas):
            t0 = time.time()
            scores[i] = estimate_coni_pfvs(data, N=N, ellipsoid_dilation=ellipsoid_dilation,
                                           seed=seed, **kwargs)
            report(i, t0)
    elif n_prefetch <= 0:
        for i, data in enumerate(datas):
            t0 = time.time()
            scores[i] = _count_pfvs(data, _first_N(data, N, seed), ellipsoid_dilation, **kwargs)
            report(i, t0)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(n_prefetch, mp_context=ctx) as pool:
            futs = [pool.submit(_first_N, d, N, seed) for d in datas[:n_prefetch]]
            for i, data in enumerate(datas):
                t0 = time.time()
                ps = futs[i].result()
                futs[i] = None                      # free it once searched
                if i + n_prefetch < len(datas):
                    futs.append(pool.submit(_first_N, datas[i + n_prefetch], N, seed))
                scores[i] = _count_pfvs(data, ps, ellipsoid_dilation, **kwargs)
                report(i, t0)
    return scores
