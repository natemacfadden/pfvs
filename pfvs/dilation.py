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
#               Everything here is exact (python-flint rationals). Both
#               hypotheses are checked; if either fails, no bound is returned.
#               A diagnostic (PFV.dilation_bound), not a search parameter:
#               the bound is attained at low h11 but is typically ~10^3 at
#               h11 >= 8, far above the dilations one searches.
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


def coni_mu0(p: ArrayLike, kappa: ArrayLike) -> tuple[Fraction, list[int]] | None:
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
    """
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
