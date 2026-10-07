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
# Description:  Scoring conifolds by how many coniPFVs a search finds.
#
#               `score_coni_geometries` scores each conifold by the number of
#               coniPFVs coniZpM finds on its first N p-vectors (those `pvecs`
#               generates, trimmed to exactly N). The score is exact for those
#               p-vectors and, on a GPU, cheap (2M p-vectors at dilation 150 in
#               1-10 s for h11 = 5-10).
#
#               `expected_pfvs_per_p` / `estimate_coni_pfvs` predict the same
#               count without searching: a parameter-free estimate from the
#               Gaussian heuristic applied to coniZpM's ellipsoids, for when no
#               GPU is available (about 0.3 ms per p-vector on one CPU core).
#
#               For a p-vector p, coniZpM enumerates c in Z^n (n = h11-1) with
#
#                   val = c^T mat c <= Q * dilation,   M0 = l.c >= M0min,
#                   val < Q * gcd(A c),
#
#               where (mat, Z, Binter) = coni_M_ellipsoid(p), A = (Z@Binter)[1:]
#               and l = Binter[0], and keeps c only if some K0 hits the tadpole
#               exactly, which needs M0 | (Qperp - Q): roughly a 1/M0 chance.
#               Replacing lattice-point counts by volumes (exact on average over
#               random lattices: Siegel's mean value theorem) gives
#
#                   E_p = sum_{g=1}^{dilation} dens(g) * I(g)
#
#                   dens(g) = prod_i gcd(s_i, g) / g    (s_i: Smith invariants
#                                                        of A; P(g | A c))
#                   I(g)    = int_{c^T mat c <= Q g, l.c >= M0min} dc / (l.c)
#                           = Vol_n(Q g) / sqrt(det mat) * F_n(t0) / rho,
#                   rho     = sqrt(Q g) * sqrt(l^T mat^-1 l),  t0 = M0min / rho,
#                   F_n(t0) = E[1[t >= t0] / t], t the unit ball's 1-D marginal.
#
#               All three ingredients (det mat, s_i, l^T mat^-1 l) are invariant
#               under unimodular changes of the c-basis, so the estimate does not
#               depend on how coni_M_ellipsoid reduces Binter.
#
#               Caveats. The estimate overcounts, by a factor that varies
#               between conifolds and grows with h11 and with |p|. Measured
#               stage by stage against coniZpM (208 conifolds of the coni_pfvs
#               dataset, all 500k p-vectors of each at dilation 150; median
#               over conifolds with PFVs):
#                 - det N != 0, which the estimate does not model, removes
#                   1.9x (h11 = 5) to 6.2x (h11 = 9-10) of the candidates that
#                   hit the tadpole. Singular candidates mostly recur from many
#                   p (p and p + t v give the same (K, M) when N(M) v = 0);
#                 - the 1/M0 chance of an exact tadpole hit is optimistic by
#                   about 2.5x;
#                 - volumes undercount the lattice points of these thin
#                   ellipsoids (0.2-0.9x), partly cancelling the above.
#               Net: 3-5x overall (interquartile 2.8-9x per conifold); the
#               K' > 0 cut removes nothing (the kernel's gcd cut implies it).
#               Rank-based use is still good: Spearman 0.87 against the true
#               counts on those 208 conifolds; on 2,626 conifolds of held-out
#               polytopes, AUC 0.945 for >0 PFVs, 0.979 for >50 (pfvscorer).
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
from .coniZp import coni_M_ellipsoid, coniZpM
from .cydata import CYData
from .pvectors import pvecs

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
    Approximately uniform sample of the primitive p-vectors that `pvecs(data, N)`
    would return, without enumerating all N of them.

    Enumerates a base set of ~min(N, N0) and scales it by s = (N/N0)^(1/n) with
    uniform jitter, rejecting points outside the cone or non-primitive. Scaling
    under-samples the H p ~ 1 wall shell, which lowers the estimate's absolute
    level (0.4-0.75x of direct sampling for N0 = 2e4-2.5e5) but not its ranking
    (Spearman 0.975 vs direct sampling). Keep N0 fixed when comparing estimates.
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

# p-vectors per coniZpM call in _count_coni_pfvs (bounds memory)
_COUNT_CHUNK = 1_000_000


def _first_N(data: CYData, N: int, seed: int = 0) -> np.ndarray:
    """
    Exactly N of the p-vectors `pvecs(data, N)` returns (it returns whole
    L-inf boxes, so up to several times N): every p inside the outermost
    shell |p|_inf = B, plus a uniform random subset of that shell. Makes
    searches of different conifolds the same size.
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
    Gaussian-heuristic expected number of coniPFVs coniZpM finds from one p.

    See the module description for the formula and its caveats.

    Parameters
    ----------
    p : ArrayLike of shape (h11,) or (h11-1,)
        The p-vector (perpendicular components, as passed to coniZpM, or full).
    data : CYData, optional
        The relevant data from the associated CY. Mutually exclusive with kappa
        and Mbasis (pass those to avoid recomputing them per p).
    kappa : ArrayLike of shape (h11, h11, h11), optional
        data.kappa_cob.
    Mbasis : ArrayLike of shape (h11, h11), optional
        data.M_lattice().
    Q : integer, optional
        The tadpole. Required with kappa/Mbasis; defaults to coniZpM's
        h11+h21+4 when data is given.
    ellipsoid_dilation : float, optional
        As in coniZpM. Defaults to 1.
    M0min : integer, optional
        As in coniZpM. Defaults to 13.
    return_cumulative : bool, optional
        If True, return E_p for every integer dilation 1..floor(ellipsoid_dilation)
        instead of only the last. Defaults to False.

    Returns
    -------
    float or ndarray
        The expected count (0 when the ellipsoid form is not positive definite,
        as coniZpM then finds nothing for this p).
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
    Estimated number of coniPFVs a coniZpM search finds. For ranking.

    Either pass the p-vectors you would give coniZpM (`ps`), or a search size
    `N` (the p-vectors `pvecs(data, N)` would return). The estimate averages
    `expected_pfvs_per_p` over at most `n_samp` of them and scales by the total.

    Parameters
    ----------
    data : CYData
        The relevant data from the associated CY (coni).
    ps : ArrayLike of shape (N, h11-1), optional
        The p-vectors of the search. Mutually exclusive with N.
    N : integer, optional
        Search size: estimate for the N p-vectors `score_coni_geometries`
        searches. Up to 10^7 they are enumerated and subsampled uniformly;
        beyond, they are sampled without enumerating them all (see
        `_sample_frontier`).
    ellipsoid_dilation : float, optional
        As in coniZpM. Defaults to 1.
    Q : integer, optional
        As in coniZpM. Defaults to h11+h21+4.
    M0min : integer, optional
        As in coniZpM. Defaults to 13.
    n_samp : integer, optional
        Number of p-vectors to average over. Per-p estimates are heavy-tailed;
        512 gave stable rankings on the coni_pfvs dataset. Defaults to 512.
    N0 : integer, optional
        Base enumeration size when sampling from N > 10^7. Changes the
        absolute level, not the ranking; keep it fixed across compared
        estimates. Defaults to 100,000.
    seed : integer, optional
        Seed for the p-vector subsample. Defaults to 0.

    Returns
    -------
    float
        Estimated count. Overestimates by 3-5x (median; see the module
        description) with a wide per-conifold spread; compare estimates with
        each other rather than with absolute counts.
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
def _count_coni_pfvs(
    data: CYData,
    ps: np.ndarray,
    ellipsoid_dilation: float = 1,
    **kwargs) -> int:
    """
    Internal to `score_coni_geometries`: the number of coniPFVs coniZpM
    finds on the p-vectors ps, searched in chunks of _COUNT_CHUNK to bound
    memory. kwargs (Q, M0min, device, n_jobs) are passed to coniZpM.
    """
    n_pfvs = 0
    for i in range(0, len(ps), _COUNT_CHUNK):
        n_pfvs += len(coniZpM(data, ps[i:i + _COUNT_CHUNK],
                              ellipsoid_dilation=ellipsoid_dilation, **kwargs)[0])
    return n_pfvs


def score_coni_geometries(
    datas: list[CYData],
    N: int,
    ellipsoid_dilation: float = 1,
    method: str = "search",
    n_prefetch: int = 4,
    seed: int = 0,
    verbosity: int = 0,
    **kwargs) -> np.ndarray:
    """
    Score conifolds by how many coniPFVs a size-N search finds in each.

    The search is coniZpM on exactly N p-vectors per conifold: those of
    `pvecs(data, N)` (every primitive p in the cone in the smallest L-inf box
    |p_i| <= B holding at least N), trimmed to N by keeping a uniform random
    subset of the outermost shell |p|_inf = B. On a GPU this takes 1-10 s per
    conifold for N = 2M at dilation 150 and h11 = 5-10, most of it generating
    the p-vectors.

    Parameters
    ----------
    datas : list of CYData
        The conifolds (coni contexts).
    N : integer
        Number of p-vectors searched per conifold.
    ellipsoid_dilation : float, optional
        As in coniZpM. Defaults to 1.
    method : str, optional
        "search" (the default): the exact count; use a GPU. "estimate":
        `estimate_coni_pfvs`, the Gaussian-heuristic prediction of it (CPU
        only; ranks worse, see the module description).
    n_prefetch : integer, optional
        For "search": generate the p-vectors of the next conifolds in this
        many worker processes while the current one is searched (generating
        them is single-threaded and, at h11 >= 10, often slower than the GPU
        search). Each holds N x h11 int64. 0 generates them in turn.
        Defaults to 4.
    seed : integer, optional
        Seed for the trimming of the outermost shell. Defaults to 0.
    verbosity : integer, optional
        >= 1 prints one line per conifold. Defaults to 0.
    **kwargs :
        Passed to coniZpM (Q, M0min, device, n_jobs) or `estimate_coni_pfvs`.

    Returns
    -------
    ndarray of shape (len(datas),)
        scores[i]: the number of coniPFVs found in datas[i] (or, for
        "estimate", the predicted number).
    """
    if method not in ("search", "estimate"):
        raise ValueError(f"method must be 'search' or 'estimate', got {method!r}.")
    for data in datas:
        if not data.coni:
            raise ValueError("score_coni_geometries only applies to coni contexts.")
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
            scores[i] = _count_coni_pfvs(data, _first_N(data, N, seed),
                                         ellipsoid_dilation, **kwargs)
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
                scores[i] = _count_coni_pfvs(data, ps, ellipsoid_dilation, **kwargs)
                report(i, t0)
    return scores
