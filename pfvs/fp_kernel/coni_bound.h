// =============================================================================
//    Copyright (C) 2026  Liam McAllister Group
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
// =============================================================================
#ifndef CONI_BOUND_H
#define CONI_BOUND_H

/*
**Description:**
mu0 of pfvs.dilation (the coni dilation bound is delta < Q / mu0) in C:

    A      = kappa . p_hat              p_hat = (0, p_r), integral
    S      = -A_rr^-1                   s = a00 - a0r^T A_rr^-1 a_r0
    Lambda = {K in Z^(h11-1) : p_r . K = 0}
    mu0    = min_{0 != K in Lambda} K^T S K

Exact: integer arithmetic (int64 values, checked int128 intermediates). The
form S on Lambda is G / L with G an integer Gram matrix (fraction-free
solve). G is LLL-reduced (floating-point decisions, exact integer
transform) and its minimum found by Fincke-Pohst (cb_min below): floats only
prune, widened by a rigorous rounding-error bound, and every candidate is
evaluated exactly. Requires pfv_lattice.h (with its implementation) in the
same translation unit.

**Returns:**
     0: mu0 = *mu_num / *mu_den (reduced), K a minimizer (length h11-1)
     1: no bound: A_rr singular, s > 0, or S not positive definite on Lambda
    <0: not decided here (int64/int128 overflow, a form too ill-conditioned
        for the floating-point pruning, ...): use the exact Python path
*/
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef CB_MAX_H11
#define CB_MAX_H11 20       /* larger h11: the Python path (stack use ~ CB_MAX_H11^2) */
#endif

// bits of |x| (0 for x = 0)
static inline int cb_bits(fpk_u128 x)
{
    uint64_t hi = (uint64_t)(x >> 64), lo = (uint64_t)x;
    return hi ? 128 - __builtin_clzll(hi) : lo ? 64 - __builtin_clzll(lo) : 0;
}

// r = a * b, 1 on overflow: fpk_mul_ovf without its division when the
// bit lengths already show the product fits
static inline int cb_mul(fpk_i128 a, fpk_i128 b, fpk_i128 *r)
{
    if (cb_bits(fpk_abs128(a)) + cb_bits(fpk_abs128(b)) <= 126) { *r = a * b; return 0; }
    return fpk_mul_ovf(a, b, r);
}

// Exact division by v (the dividend must be a multiple of v): shift out
// v's power of two, then multiply by the inverse of its odd part mod 2^128
typedef struct { int sh, neg; fpk_u128 inv; } cb_divisor;
static inline cb_divisor cb_divisor_make(fpk_i128 v)
{
    cb_divisor D;
    fpk_u128 a = fpk_abs128(v);
    uint64_t lo = (uint64_t)a;
    D.sh = lo ? __builtin_ctzll(lo) : 64 + __builtin_ctzll((uint64_t)(a >> 64));
    a >>= D.sh;
    fpk_u128 x = a;                       // a x = 1 mod 2^3; each step doubles the bits
    for (int k = 0; k < 6; ++k) x *= 2 - a * x;
    D.inv = x; D.neg = v < 0;
    return D;
}
static inline fpk_i128 cb_divexact(fpk_i128 x, cb_divisor D)
{
    fpk_i128 q = (fpk_i128)((fpk_u128)(x >> D.sh) * D.inv);
    return D.neg ? -q : q;
}

// LLL (delta = 0.99) of the rows of V (n x n, int64; the identity on entry)
// with respect to the form G: the Gram matrix of V is kept in double and
// only steers the reduction; V itself is updated exactly, so it stays
// unimodular whatever the rounding (the caller recomputes V G V^T exactly).
// 0, -1 on int64 overflow, -2 if a Gram-Schmidt norm is not positive, -3
// if it does not converge.
static int cb_lll(int64_t *V, int n, const int64_t *G)
{
    enum { MX = CB_MAX_H11 };
    double F[MX][MX] = {{0}}, mu[MX][MX], bb[MX];    // (F zeroed: silences -Wmaybe-uninitialized)
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) F[i][j] = (double)G[i * n + j];
    #define CB_GS(kk) do {                                                     \
        for (int j_ = 0; j_ < (kk); ++j_) {                                    \
            double d_ = F[kk][j_];                                             \
            for (int l_ = 0; l_ < j_; ++l_) d_ -= mu[j_][l_] * mu[kk][l_] * bb[l_]; \
            mu[kk][j_] = d_ / bb[j_];                                          \
        }                                                                      \
        double d_ = F[kk][kk];                                                 \
        for (int l_ = 0; l_ < (kk); ++l_) d_ -= mu[kk][l_] * mu[kk][l_] * bb[l_]; \
        bb[kk] = d_;                                                           \
    } while (0)
    CB_GS(0);
    if (!(bb[0] > 0)) return -2;
    if (n >= 2) CB_GS(1);
    int k = 1, iters = 0;
    while (k < n) {
        if (++iters > 100000) return -3;
        for (int j = k - 1; j >= 0; --j) {                       // size reduction
            double q = nearbyint(mu[k][j]);
            if (q == 0.0) continue;
            if (!(fabs(q) < 4.0e18)) return -1;
            int64_t qi = (int64_t)q;
            for (int c = 0; c < n; ++c) {
                int64_t t;
                if (__builtin_mul_overflow(qi, V[j * n + c], &t)
                        || __builtin_sub_overflow(V[k * n + c], t, &V[k * n + c])) return -1;
            }
            const double fkj = F[k][j];
            for (int l = 0; l < n; ++l) if (l != k) F[k][l] = F[l][k] = F[k][l] - q * F[j][l];
            F[k][k] += q * (q * F[j][j] - 2.0 * fkj);
            for (int l = 0; l < j; ++l) mu[k][l] -= q * mu[j][l];
            mu[k][j] -= q;
        }
        CB_GS(k);
        if (!(bb[k] > 0)) return -2;
        if (bb[k] < (0.99 - mu[k][k - 1] * mu[k][k - 1]) * bb[k - 1]) {
            for (int c = 0; c < n; ++c) {
                int64_t t = V[k * n + c]; V[k * n + c] = V[(k - 1) * n + c]; V[(k - 1) * n + c] = t;
            }
            for (int l = 0; l < n; ++l) { double t = F[k][l]; F[k][l] = F[k - 1][l]; F[k - 1][l] = t; }
            for (int l = 0; l < n; ++l) { double t = F[l][k]; F[l][k] = F[l][k - 1]; F[l][k - 1] = t; }
            k = k > 1 ? k - 1 : 1;
            if (k == 1) CB_GS(0);
            CB_GS(k - 1);
            CB_GS(k);
        } else if (++k < n) {
            CB_GS(k);
        }
    }
    #undef CB_GS
    return 0;
}

// Exact y^T G y (int128, checked): 0, or 1 on overflow.
static int cb_quad(const int64_t *G, const int64_t *y, int n, fpk_i128 *q)
{
    fpk_i128 s = 0, t, u;
    for (int i = 0; i < n; ++i) {
        if (!y[i]) continue;
        fpk_i128 r = 0;
        for (int j = 0; j < n; ++j)
            if (y[j] && (cb_mul((fpk_i128)G[i * n + j], (fpk_i128)y[j], &t) || fpk_add_ovf(r, t, &r))) return 1;
        if (cb_mul(r, (fpk_i128)y[i], &u) || fpk_add_ovf(s, u, &s)) return 1;
    }
    *q = s;
    return 0;
}

// min y^T G y over nonzero integer y, for G (n x n, int64, symmetric,
// LLL-reduced): *best enters as the value of the basis vector y (the
// candidate to beat) and is lowered to the minimum, y to a minimizer.
//
// Floating point only prunes. The double Cholesky factor R is backward
// stable: R^T R = G + E with |E| <= g1 |R|^T |R| entrywise, g1 = gamma_{n+1}.
// So for any y, y^T G y >= |R y|^2 - g1 |(|R| |y|)|^2 >= (1 - g1 C) |R y|^2,
// with C >= (|| |R| ||_2 ||R^-1||_2)^2 (Frobenius norms, doubled). Every y
// with y^T G y < best thus has |R y|^2 < best / (1 - g1 C) = rad, and the
// search finds all y with |R y|^2 <= rad: each float t_i = (R y)_i is within
// g2 a_i of the exact one (g2 = gamma_n, a_i = sum_j |R_ij y_j|), so a level
// is pruned only if sum (|t_i| - g2 a_i)_+^2 > rad, and the coordinate
// ranges are widened by one (their rounding error is checked to be < 1/2).
// Returns 0, 1 if G is not positive definite (exactly: a nonzero y with
// y^T G y <= 0 was found), or < 0: undecided here (ill-conditioned for the
// float bounds; an exact value overflowed int128).
static int cb_min(const int64_t *G, int n, int64_t *best, int64_t *y)
{
    enum { MX = CB_MAX_H11 };
    double R[MX][MX] = {{0}}, Ri[MX][MX] = {{0}};
    for (int j = 0; j < n; ++j) {                                  // G = R^T R
        double d = (double)G[j * n + j];
        for (int k = 0; k < j; ++k) d -= R[k][j] * R[k][j];
        if (!(d > 0)) return -31;                     // (indefinite, or too close to it): undecided
        R[j][j] = sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            double t = (double)G[j * n + i];
            for (int k = 0; k < j; ++k) t -= R[k][j] * R[k][i];
            R[j][i] = t / R[j][j];
        }
    }
    for (int j = n - 1; j >= 0; --j) {                             // Ri = R^-1
        Ri[j][j] = 1.0 / R[j][j];
        for (int i = j - 1; i >= 0; --i) {
            double t = 0;
            for (int k = i + 1; k <= j; ++k) t += R[i][k] * Ri[k][j];
            Ri[i][j] = -t / R[i][i];
        }
    }
    double nR = 0, nRi = 0;
    for (int i = 0; i < n; ++i) for (int j = i; j < n; ++j) { nR += R[i][j] * R[i][j]; nRi += Ri[i][j] * Ri[i][j]; }
    const double u = DBL_EPSILON / 2;
    const double g1 = (n + 1) * u / (1 - (n + 1) * u), g2 = n * u / (1 - n * u);
    const double C = 2.0 * nR * nRi;
    if (!(g1 * C < 1e-6)) return -32;                               // too ill-conditioned
    const double shrink = 1.0 / (1.0 - g1 * C) * (1 + 1e-12);

    double rad = (double)*best * shrink;
    double rem[MX + 1], aoff[MX];
    int64_t c[MX], lo[MX], hi[MX];
    rem[n] = rad;
    int i = n - 1;
    #define CB_ENTER(ii) do {                                                   \
        double o_ = 0, a_ = 0;                                                  \
        for (int j_ = (ii) + 1; j_ < n; ++j_) { double t_ = R[ii][j_] * c[j_]; o_ += t_; a_ += fabs(t_); } \
        double ctr_ = -o_ / R[ii][ii], w_ = sqrt(fmax(rem[(ii) + 1], 0.0)) / R[ii][ii]; \
        double e_ = g2 * (fabs(ctr_) + w_ + 1 + a_ / R[ii][ii]) * 4 + 4 * u * (fabs(ctr_) + w_); \
        if (!(e_ < 0.5) || !(fabs(ctr_) + w_ < 1e15)) return -33;               \
        lo[ii] = (int64_t)floor(ctr_ - w_) - 1; hi[ii] = (int64_t)ceil(ctr_ + w_) + 1; \
        aoff[ii] = a_; c[ii] = lo[ii];                                          \
    } while (0)
    CB_ENTER(i);
    int found_le0 = 0;
    for (;;) {
        if (c[i] > hi[i]) {                                         // level done
            if (++i == n) break;
            c[i]++;
            continue;
        }
        double o = 0;
        for (int j = i + 1; j < n; ++j) o += R[i][j] * c[j];
        const double t = R[i][i] * c[i] + o, a = fabs(R[i][i] * c[i]) + aoff[i];
        const double lb = fmax(fabs(t) - g2 * a, 0.0) * (1 - 4 * u);
        const double r = rem[i + 1] - lb * lb * (1 - 4 * u);
        if (r < 0) { c[i]++; continue; }                            // pruned (rigorously)
        if (i > 0) {
            rem[i] = r;
            --i;
            CB_ENTER(i);
            continue;
        }
        int nz = 0;                                                 // a leaf: exact
        for (int j = 0; j < n; ++j) nz |= c[j] != 0;
        if (nz) {
            fpk_i128 q;
            if (cb_quad(G, c, n, &q)) return -34;
            if (q <= 0) found_le0 = 1;
            else if (q < *best) {
                *best = (int64_t)q;
                for (int j = 0; j < n; ++j) y[j] = c[j];
                rad = (double)*best * shrink;                       // tighten
                rem[n] = rad;
            }
        }
        c[i]++;
    }
    #undef CB_ENTER
    return found_le0 ? 1 : 0;
}

static int cb_coni_mu0(int h, const int64_t *kappa, const int64_t *p_r,
                       int64_t *mu_num, int64_t *mu_den, int64_t *K)
{
    typedef fpk_i128 i128;
    enum { MX = CB_MAX_H11 };
    const int m = h - 1, n = h - 2;            // K has m entries; Lambda has rank n
    if (h < 3 || h > MX) return -20;

    // A = kappa . p_hat (p_hat_0 = 0)
    int64_t A[MX * MX];
    for (int a = 0; a < h; ++a)
        for (int b = 0; b < h; ++b) {
            i128 s = 0;
            for (int c = 1; c < h; ++c)
                if (fpk_add_ovf(s, (i128)kappa[(a * h + b) * h + c] * p_r[c - 1], &s)) return -1;
            if (s > INT64_MAX || s < -INT64_MAX) return -1;
            A[a * h + b] = (int64_t)s;
        }

    // Lambda: rows of O (n x m) span it
    int64_t O[MX * MX];
    if (pfl_orthogonal(p_r, m, O)) return -2;

    // fraction-free solve A_rr X = d [O^T | a_r0] (Bareiss), d = +-det A_rr
    i128 M[MX][2 * MX];
    const int w = m + n + 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) M[i][j] = A[(i + 1) * h + j + 1];
        for (int j = 0; j < n; ++j) M[i][m + j] = O[j * m + i];
        M[i][m + n] = A[(i + 1) * h];
    }
    i128 t1, t2;
    cb_divisor prev = cb_divisor_make(1);
    for (int k = 0; k < m; ++k) {
        int piv = k;
        while (piv < m && M[piv][k] == 0) ++piv;
        if (piv == m) return 1;                          // A_rr singular
        if (piv != k)
            for (int j = 0; j < w; ++j) { i128 t = M[k][j]; M[k][j] = M[piv][j]; M[piv][j] = t; }
        for (int i = k + 1; i < m; ++i) {
            for (int j = k + 1; j < w; ++j) {
                if (cb_mul(M[k][k], M[i][j], &t1) || cb_mul(M[i][k], M[k][j], &t2)
                        || fpk_add_ovf(t1, -t2, &t1)) return -3;
                M[i][j] = cb_divexact(t1, prev);         // exact (Bareiss)
            }
            M[i][k] = 0;
        }
        prev = cb_divisor_make(M[k][k]);
    }
    const i128 d = M[m - 1][m - 1];
    i128 X[MX][MX + 1];                                  // A_rr X = d [O^T | a_r0]
    cb_divisor piv[MX];
    for (int i = 0; i < m; ++i) piv[i] = cb_divisor_make(M[i][i]);
    for (int c = 0; c <= n; ++c)
        for (int i = m - 1; i >= 0; --i) {
            i128 s;
            if (cb_mul(d, M[i][m + c], &s)) return -3;
            for (int j = i + 1; j < m; ++j)
                if (cb_mul(M[i][j], X[j][c], &t1) || fpk_add_ovf(s, -t1, &s)) return -3;
            X[i][c] = cb_divexact(s, piv[i]);
            if (cb_mul(X[i][c], M[i][i], &t1) || t1 != s) return -4;   // cannot happen (exact)
        }

    // -s = (a_r0^T x - d a00) / d with x = X[:, n] = d A_rr^-1 a_r0; need s <= 0
    i128 sn;
    if (cb_mul(-d, (i128)A[0], &sn)) return -3;
    for (int i = 0; i < m; ++i)
        if (cb_mul((i128)A[(i + 1) * h], X[i][n], &t1) || fpk_add_ovf(sn, t1, &sn)) return -3;
    if ((sn < 0) != (d < 0) && sn != 0) return 1;        // -s < 0

    // S on Lambda = G_n / d with G_n = -O X[:, :n]; integer Gram G = G_n / g * sign(d), L = |d| / g
    i128 Gn[MX * MX], g = d < 0 ? -d : d;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {                   // (symmetric)
            i128 s = 0;
            for (int i = 0; i < m; ++i)
                if (O[a * m + i] && (cb_mul((i128)O[a * m + i], X[i][b], &t1) || fpk_add_ovf(s, -t1, &s))) return -3;
            Gn[a * n + b] = Gn[b * n + a] = s;
            if (g == 1) continue;
            if ((fpk_u128)g >> 64) g = (i128)fpk_gcd128((fpk_u128)g, fpk_abs128(s));
            else g = (i128)fpk_gcd64((uint64_t)g, (uint64_t)(fpk_abs128(s) % (uint64_t)g));
        }
    int64_t G[MX * MX];
    const i128 L = (d < 0 ? -d : d) / g;
    for (int k = 0; k < n * n; ++k) {
        i128 v = Gn[k] / g * (d < 0 ? -1 : 1);
        if (v > INT64_MAX || v < -INT64_MAX) return -5;
        G[k] = (int64_t)v;
    }
    if (L > INT64_MAX) return -5;

    // LLL-reduce: rows of V (Lambda coordinates), Gr = V G V^T (exact)
    int64_t V[MX * MX], Gr[MX * MX];
    for (int k = 0; k < n * n; ++k) V[k] = k / n == k % n;
    for (int k = 0; k < n; ++k) if (G[k * n + k] <= 0) return 1;      // not positive definite
    if (n > 1) {
        int rc = cb_lll(V, n, G);
        if (rc == -2 || rc == -3) return -6;                     // e.g. indefinite: undecided here
        if (rc) return -5;
    }
    i128 GV[MX * MX];                                            // G V^T (int64 * int64 never overflows int128)
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < n; ++b) {
            i128 r = 0;
            for (int j = 0; j < n; ++j)
                if (fpk_add_ovf(r, (i128)G[i * n + j] * V[b * n + j], &r)) return -3;
            GV[i * n + b] = r;
        }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            i128 s = 0;
            for (int i = 0; i < n; ++i)
                if (V[a * n + i] && (cb_mul((i128)V[a * n + i], GV[i * n + b], &t1) || fpk_add_ovf(s, t1, &s))) return -3;
            if (s > INT64_MAX || s < -INT64_MAX) return -5;
            Gr[a * n + b] = Gr[b * n + a] = (int64_t)s;
        }

    // the minimum: the best basis vector, unless cb_min finds a shorter one
    int i0 = 0;
    for (int k = 1; k < n; ++k) if (Gr[k * n + k] < Gr[i0 * n + i0]) i0 = k;
    int64_t best = Gr[i0 * n + i0], y[MX];
    if (best <= 0) return 1;
    for (int k = 0; k < n; ++k) y[k] = k == i0;
    int st = cb_min(Gr, n, &best, y);
    if (st) return st == 1 ? 1 : -10 + st;

    // K = O^T V^T y
    for (int i = 0; i < m; ++i) {
        i128 s = 0;
        for (int a = 0; a < n; ++a) {
            i128 z = 0;
            for (int b = 0; b < n; ++b)
                if (cb_mul((i128)V[b * n + a], (i128)y[b], &t1) || fpk_add_ovf(z, t1, &z)) return -3;
            if (cb_mul((i128)O[a * m + i], z, &t1) || fpk_add_ovf(s, t1, &s)) return -3;
        }
        if (s > INT64_MAX || s < -INT64_MAX) return -5;
        K[i] = (int64_t)s;
    }
    const i128 gg = (i128)fpk_gcd128((fpk_u128)best, (fpk_u128)L);
    *mu_num = (int64_t)(best / gg);
    *mu_den = (int64_t)(L / gg);
    return 0;
}

#endif  // CONI_BOUND_H
