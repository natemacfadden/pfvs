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
# Description:  Exact reference ("oracle") for the Fincke-Pohst kernels. Pure
#               Python, rational arithmetic only: no floating point enters any
#               accept/reject decision. Slow, so only used on small instances.
# -----------------------------------------------------------------------------

import math
from fractions import Fraction


def ldl_upper(mat):
    """
    Exact decomposition mat = R^T diag(d) R with R unit upper-triangular.

    Then c^T mat c = sum_i d[i] * (c[i] + sum_{j>i} R[i][j] c[j])^2, which lets
    c be set from the last component to the first (as the kernels do).
    """
    n = len(mat)
    A = [[Fraction(int(mat[i][j])) for j in range(n)] for i in range(n)]
    d = [Fraction(0)] * n
    R = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for i in range(n):
        d[i] = A[i][i] - sum(d[k] * R[k][i] ** 2 for k in range(i))
        if d[i] <= 0:
            raise ValueError("mat is not positive definite")
        for j in range(i + 1, n):
            R[i][j] = (A[i][j] - sum(d[k] * R[k][i] * R[k][j] for k in range(i))) / d[i]
    return d, R


def _isqrt_frac_floor(x):
    """floor(sqrt(x)) for a nonnegative Fraction x."""
    return math.isqrt(x.numerator * x.denominator) // x.denominator


def ellipsoid_points(mat, qmax):
    """
    All integer c with c^T mat c <= qmax, exactly, in the kernels' DFS order
    (last component outermost, each component ascending).
    """
    n = len(mat)
    d, R = ldl_upper(mat)
    qmax = Fraction(qmax)
    out = []
    c = [0] * n

    def rec(i, rem):
        if i < 0:
            out.append(tuple(c))
            return
        t = sum((R[i][j] * c[j] for j in range(i + 1, n)), Fraction(0))
        # (c_i + t)^2 <= rem/d_i  <=>  |c_i + t| <= s,  s = sqrt(rem/d_i)
        lim = rem / d[i]
        s = _isqrt_frac_floor(lim) + 1  # s_floor+1 > sqrt(lim): superset
        lo = math.floor(-t) - s
        hi = math.ceil(-t) + s
        for v in range(lo, hi + 1):
            e = d[i] * (v + t) ** 2
            if e <= rem:
                c[i] = v
                rec(i - 1, rem - e)
        c[i] = 0

    rec(n - 1, qmax)
    return out


def qform(mat, c):
    n = len(c)
    return sum(int(mat[i][j]) * c[i] * c[j] for i in range(n) for j in range(n))


def row_gcd(H, c):
    """gcd of the entries of H @ c (0 iff H @ c == 0)."""
    g = 0
    for row in H:
        g = math.gcd(g, sum(int(h) * x for h, x in zip(row, c)))
    return g


def kernel_reference(mat, Q, dilation, H, linvec=None, linmin=None, strict=False):
    """
    The exact output set of the kernels, in their output order.

    c is returned iff
        - c^T mat c <= floor(dilation * Q)                     (ellipsoid)
        - linvec . c >= linmin            (if linvec given)    (M0 cut)
        - g == 0  or  Q*g >= c^T mat c    (Q*g > ... if strict) (GCD cut)
    where g = gcd(H @ c).

    Returns (points, qs) with qs[i] = c^T mat c exactly.
    """
    qmax = math.floor(Fraction(dilation) * Q + Fraction(1, 10**7))
    pts, qs = [], []
    for c in ellipsoid_points(mat, qmax):
        q = qform(mat, c)
        if linvec is not None and sum(int(a) * b for a, b in zip(linvec, c)) < linmin:
            continue
        g = row_gcd(H, c)
        if g != 0 and (Q * g <= q if strict else Q * g < q):
            continue
        pts.append(c)
        qs.append(q)
    return pts, qs
