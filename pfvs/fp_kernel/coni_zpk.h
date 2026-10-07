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
#ifndef CONI_ZPK_H
#define CONI_ZPK_H

/*
**Description:**
coni ZpK above D0: for a direction p_hat, the lattice points c of every coni
PFV with dilation delta > D0. (ZpM at ellipsoid dilation D0 finds those with
delta <= D0, so the two together are exhaustive.) Notation as in
coni_bound.h:

    A = kappa . p_hat,  S = -A_rr^-1,  -s = sn / sd,
    Lambda = {K in Z^m : p_r . K = 0}   (m = h11 - 1, rank n = m - 1)

A PFV with p = p_hat / delta has K_r in Lambda and tadpole

    Q = M0 eps + (M0^2 / delta)(-s) + delta K_r^T S K_r,    eps > 0,

so with -s > 0 and S positive definite on Lambda, delta > D0 forces
K_r^T S K_r < Q / D0: K_r is one of finitely many short vectors of Lambda,
enumerated by Fincke-Pohst. The lattice points c (M = Binter c) have
K_r = W c / delta, W = (Z Binter)_r. For one short K_r = g K' (K' primitive),
let o be the order of K' in Lambda / Gamma, Gamma = W Z^m. The c with W c
parallel to K_r are c = j c1 + t k (W c1 = o K', k spanning ker W), with
delta = j o / g. So delta > D0 is j o > g D0, the tadpole is

    g L sn M0^2 < sd j o (Q L - g j o kn),    kn = L K'^T S K' (an integer),

and M0 = l . c = j (l . c1) + t (l . k) >= M0min (l = Binter[0, :]): a range
of j, and for each j an arithmetic progression of M0 (hence of t).

The points are post-processed like ZpM's (coniZp._pfvs_from_points), which
also expands the gcd of K_r: some resulting PFVs can have delta <= D0, and a
point can be emitted more than once, so the union with ZpM's PFVs must be
deduplicated.

Exact: integer arithmetic (int64 values, checked int128 intermediates);
floating point only prunes the enumeration, widened by a rigorous
rounding-error bound (as cb_min in coni_bound.h), and screens the scan over
j with a margin far above its rounding error. Requires fpk_common.h,
pfv_lattice.h (with its implementation) and coni_bound.h.

**Returns** (czk_setup, czk_search, per direction):
     0: done
     1: no ZpK for this direction: s >= 0, A_rr singular, or S not positive
        definite on Lambda
    <0: not decided here (an int64 / int128 overflow, a form too
        ill-conditioned for the floating-point bounds, more than max_out
        points, ...): search the direction another way (e.g. ZpM up to its
        dilation bound)
*/
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CZK_MAX_H11 CB_MAX_H11

// everything the search needs for one direction (row-major)
typedef struct {
    int h, m, n;
    int64_t Z[CZK_MAX_H11 * CZK_MAX_H11];   // (h, h)   kappa . p_hat
    int64_t Binter[CZK_MAX_H11 * CZK_MAX_H11]; // (h, m) an M-lattice basis: M = Binter c
    int64_t ZB[CZK_MAX_H11 * CZK_MAX_H11];  // (h, m)   Z Binter
    int64_t mat[CZK_MAX_H11 * CZK_MAX_H11]; // (m, m)   -Binter^T Z Binter
    int64_t Gr[CZK_MAX_H11 * CZK_MAX_H11];  // (n, n)   LLL-reduced L S on Lambda (reduced coords x)
    int64_t V[CZK_MAX_H11 * CZK_MAX_H11];   // (n, n)   Lambda coords y = x V
    int64_t B[CZK_MAX_H11 * CZK_MAX_H11];   // (m, n)   K = B y
    int64_t HT[CZK_MAX_H11 * CZK_MAX_H11];  // (n, n)   Hadj V^T: order of y in Lambda / Gamma
    int64_t w[CZK_MAX_H11];                 // (n)      l Cs B V^T
    int64_t Cs[CZK_MAX_H11 * CZK_MAX_H11];  // (m, m)   c1 = Cs v / Cden solves W c1 = v (v in Gamma)
    int64_t kv[CZK_MAX_H11];                // (m)      spans ker W
    int64_t Hdet, Cden, L, sn, sd, lk;      // lk = l . kv
} czk_dir;

// the points found (and M = Binter c, Kn = Z Binter c, q = c^T mat c, the
// direction's index), appended to growing arrays
typedef struct {
    int64_t *M, *Kn, *q, *pidx;
    long n, cap;
} czk_points;

static void czk_points_free(czk_points *P)
{
    free(P->M); free(P->Kn); free(P->q); free(P->pidx);
    memset(P, 0, sizeof(*P));
}

static int czk_points_reserve(czk_points *P, int h, long need)
{
    if (need <= P->cap) return 0;
    long cap = P->cap ? 2 * P->cap : 64;
    while (cap < need) cap *= 2;
    void *a = realloc(P->M, (size_t)cap * h * sizeof(int64_t));
    if (!a) return -1;
    P->M = (int64_t *)a;
    if (!(a = realloc(P->Kn, (size_t)cap * h * sizeof(int64_t)))) return -1;
    P->Kn = (int64_t *)a;
    if (!(a = realloc(P->q, (size_t)cap * sizeof(int64_t)))) return -1;
    P->q = (int64_t *)a;
    if (!(a = realloc(P->pidx, (size_t)cap * sizeof(int64_t)))) return -1;
    P->pidx = (int64_t *)a;
    P->cap = cap;
    return 0;
}

static inline int czk_fits64(fpk_i128 x) { return x <= (fpk_i128)INT64_MAX && x >= -(fpk_i128)INT64_MAX; }
static inline fpk_i128 czk_fdiv(fpk_i128 a, fpk_i128 b)          // floor(a / b), b > 0
{
    fpk_i128 q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// r = a * b + c, checked (int64 factors: the product cannot overflow)
static inline int czk_muladd(fpk_i128 a, fpk_i128 b, fpk_i128 c, fpk_i128 *r)
{
    if (a == (int64_t)a && b == (int64_t)b) return fpk_add_ovf(c, a * b, r);
    fpk_i128 t;
    return cb_mul(a, b, &t) || fpk_add_ovf(c, t, r);
}

// r = c - a * b, checked
static inline int czk_mulsub(fpk_i128 a, fpk_i128 b, fpk_i128 c, fpk_i128 *r)
{
    if (a == (int64_t)a && b == (int64_t)b) return fpk_sub_ovf(c, a * b, r);
    fpk_i128 t;
    return cb_mul(a, b, &t) || fpk_sub_ovf(c, t, r);
}

// gcd(g, |s|) for g > 0, cheap when s is a multiple of g (the usual case)
static inline fpk_i128 czk_gcd_acc(fpk_i128 g, fpk_i128 s)
{
    if (g == 1 || s == 0) return g;
    fpk_u128 x = fpk_abs128(s);
    if ((fpk_u128)g >> 64) return (fpk_i128)fpk_gcd128((fpk_u128)g, x);
    uint64_t gg = (uint64_t)g, r = (x >> 64) ? (uint64_t)(x % gg) : (uint64_t)x % gg;
    return r ? (fpk_i128)fpk_gcd64(gg, r) : g;
}

// gcd(|a|, |b|), 64-bit when both fit
static inline fpk_i128 czk_gcd(fpk_i128 a, fpk_i128 b)
{
    fpk_u128 x = fpk_abs128(a), y = fpk_abs128(b);
    if (!(x >> 64) && !(y >> 64)) return (fpk_i128)fpk_gcd64((uint64_t)x, (uint64_t)y);
    if (!(x >> 64) && x) return (fpk_i128)fpk_gcd64((uint64_t)x, (uint64_t)(y % x));
    if (!(y >> 64) && y) return (fpk_i128)fpk_gcd64((uint64_t)y, (uint64_t)(x % y));
    return (fpk_i128)fpk_gcd128(x, y);
}

// Euclidean LLL of the r rows of Bm (length c): cb_lll on their Gram matrix
// (floats steer, the unimodular transform is exact), then Bm <- V Bm
// exactly; pfl_lll (int128 Gram) if the Gram or the result leaves int64.
static int czk_lll_rows(int64_t *Bm, int r, int c)
{
    typedef fpk_i128 i128;
    if (r < 2) return 0;
    int64_t G[CZK_MAX_H11 * CZK_MAX_H11], V[CZK_MAX_H11 * CZK_MAX_H11], out[CZK_MAX_H11 * CZK_MAX_H11];
    if (r > CZK_MAX_H11 || c > CZK_MAX_H11) return pfl_lll(Bm, r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j <= i; ++j) {
            i128 s = 0;
            for (int k = 0; k < c; ++k)
                if (fpk_add_ovf(s, (i128)Bm[i * c + k] * Bm[j * c + k], &s)) return pfl_lll(Bm, r, c);
            if (!czk_fits64(s)) return pfl_lll(Bm, r, c);
            G[i * r + j] = G[j * r + i] = (int64_t)s;
        }
    for (int k = 0; k < r * r; ++k) V[k] = k / r == k % r;
    if (cb_lll(V, r, G)) return pfl_lll(Bm, r, c);
    for (int i = 0; i < r; ++i)
        for (int k = 0; k < c; ++k) {
            i128 s = 0;
            for (int j = 0; j < r; ++j)
                if (V[i * r + j] && czk_muladd(V[i * r + j], Bm[j * c + k], s, &s)) return pfl_lll(Bm, r, c);
            if (!czk_fits64(s)) return pfl_lll(Bm, r, c);
            out[i * c + k] = (int64_t)s;
        }
    memcpy(Bm, out, sizeof(int64_t) * r * c);
    return 0;
}

// Z, Binter, ZB, mat: pfl_build's lattice without its final basis change
// and HNF, which only ZpM's search uses
static int czk_lattice(const int64_t *kappa, const int64_t *Mbasis, const int64_t *p, czk_dir *o)
{
    typedef fpk_i128 i128;
    const int h = o->h, m = o->m;
    int64_t T[CZK_MAX_H11], v[CZK_MAX_H11], O[CZK_MAX_H11 * CZK_MAX_H11], BT[CZK_MAX_H11 * CZK_MAX_H11];
    for (int i = 0; i < h * h * h; ++i) if (kappa[i] == INT64_MIN) return -1;
    for (int i = 0; i < h * h; ++i) if (Mbasis[i] == INT64_MIN) return -1;
    for (int i = 0; i < h; ++i) if (p[i] == INT64_MIN) return -1;
    for (int i = 0; i < h * h; ++i) {                    // Z = kappa . p
        i128 s = 0;
        for (int k = 0; k < h; ++k)
            if (fpk_add_ovf(s, (i128)kappa[i * h + k] * p[k], &s)) return -1;
        if (!czk_fits64(s)) return -1;
        o->Z[i] = (int64_t)s;
    }
    if (pfl_matmul(o->Z, p, T, h, h, 1) || pfl_matmul(T, Mbasis, v, 1, h, h)) return -1;
    // the M lattice orthogonal to p.Z: Binter = Mbasis O^T, O LLL-reduced
    // (ZpK's search is in K-space: Binter only has to keep the numbers small)
    if (pfl_orthogonal(v, h, O) || czk_lll_rows(O, m, h)) return -2;
    for (int a = 0; a < m; ++a)
        for (int i = 0; i < h; ++i) {
            i128 s = 0;
            for (int l = 0; l < h; ++l)
                if (fpk_add_ovf(s, (i128)Mbasis[i * h + l] * O[a * h + l], &s)) return -1;
            if (!czk_fits64(s)) return -1;
            BT[a * h + i] = (int64_t)s;
        }
    for (int i = 0; i < h; ++i)
        for (int a = 0; a < m; ++a) o->Binter[i * m + a] = BT[a * h + i];
    if (pfl_matmul(o->Z, o->Binter, o->ZB, h, h, m)) return -1;
    for (int a = 0; a < m; ++a)
        for (int b = 0; b < m; ++b) {
            i128 s = 0;
            for (int i = 0; i < h; ++i)
                if (fpk_add_ovf(s, (i128)o->Binter[i * m + a] * o->ZB[i * m + b], &s)) return -1;
            if (!czk_fits64(s)) return -1;
            o->mat[a * m + b] = (int64_t)(-s);
        }
    return 0;
}

// Unimodular U (as pfl_unimodular: U v = (gcd(v), 0, ..., 0)) and U^-1
static int czk_unimodular_inv(const int64_t *v, int n, int64_t *U, fpk_i128 *Uinv)
{
    typedef fpk_i128 i128;
    int64_t w[CZK_MAX_H11];
    for (int i = 0; i < n; ++i) {
        w[i] = v[i];
        for (int j = 0; j < n; ++j) { U[i * n + j] = i == j; Uinv[i * n + j] = i == j; }
    }
    for (int k = 1; k < n; ++k) {
        if (w[k] == 0) continue;
        int64_t s_, t_, g = pfl_xgcd(w[0], w[k], &s_, &t_);
        if (g == 0) return -2;
        // rows 0, k of U <- E (rows 0, k), E = [[s, t], [m10, m11]], det 1;
        // columns 0, k of U^-1 <- (columns 0, k) E^-1, E^-1 = [[m11, -t], [-m10, s]]
        const int64_t m10 = -pfl_fdiv(w[k], g), m11 = pfl_fdiv(w[0], g);
        w[0] = g; w[k] = 0;
        for (int r = 0; r < n; ++r) {
            i128 t1 = (i128)s_ * U[r] + (i128)t_ * U[k * n + r];
            i128 t2 = (i128)m10 * U[r] + (i128)m11 * U[k * n + r];
            if (!czk_fits64(t1) || !czk_fits64(t2)) return -1;
            U[r] = (int64_t)t1; U[k * n + r] = (int64_t)t2;
            i128 c0 = Uinv[r * n], ck = Uinv[r * n + k], a, b;
            if (cb_mul(c0, m11, &a) || czk_muladd(ck, -m10, a, &a)
                    || cb_mul(c0, -t_, &b) || czk_muladd(ck, s_, b, &b)) return -1;
            Uinv[r * n] = a; Uinv[r * n + k] = b;
        }
    }
    return 0;
}

static int czk_setup(int h, const int64_t *kappa, const int64_t *Mbasis, const int64_t *p, czk_dir *o)
{
    typedef fpk_i128 i128;
    enum { MX = CZK_MAX_H11 };
    const int m = h - 1, n = h - 2;
    if (h < 3 || h > MX) return -20;
    o->h = h; o->m = m; o->n = n;
    int rc = czk_lattice(kappa, Mbasis, p, o);
    if (rc) return -30 + rc;
    i128 t1;

    // Lambda: rows 1.. of U (U p_r = (g, 0, ...)), as the columns of B;
    // Lambda coordinates of v: (v^T U^-1)[1:]
    int64_t U[MX * MX];
    i128 Uinv[MX * MX];
    if (czk_unimodular_inv(&p[1], m, U, Uinv)) return -40;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) o->B[i * n + j] = U[(j + 1) * m + i];

    // Y (n x m): Lambda coordinates of the columns of W = ZB[1:], straight
    // into the rows of [Y^T | I_m]
    const int wd = n + m;
    int64_t Aug[MX * 2 * MX];
    for (int c = 0; c < m; ++c) {
        for (int j = 0; j < m; ++j) {
            i128 s = 0;
            for (int i = 0; i < m; ++i)
                if (czk_muladd(o->ZB[(i + 1) * m + c], Uinv[i * m + j], s, &s)) return -41;
            if (j == 0) { if (s != 0) return -42; }      // (W c is in Lambda)
            else { if (!czk_fits64(s)) return -41; Aug[c * wd + j - 1] = (int64_t)s; }
        }
        for (int j = 0; j < m; ++j) Aug[c * wd + n + j] = c == j;
    }
    // HNF [Y^T | I] -> rows 0..n-1 = [H' | E], row n = [0 | k^T]: Gamma (in
    // Lambda coordinates) is the row span of H', and Y k = 0
    i128 Hh[MX * 2 * MX];
    if (pfl_hnf(Aug, m, wd, Hh)) return -43;
    for (int r = 0; r < n; ++r) if (Hh[r * wd + r] <= 0) return -44;
    for (int j = 0; j < n; ++j) if (Hh[n * wd + j] != 0) return -44;
    for (int i = 0; i < m; ++i) {
        if (!czk_fits64(Hh[n * wd + n + i])) return -45;
        o->kv[i] = (int64_t)Hh[n * wd + n + i];
    }
    // Hdet = det H', Hadj = Hdet (H'^T)^-1 (lower triangular), by exact
    // forward substitution
    i128 Hadj[MX * MX], dH = 1;
    for (int i = 0; i < n; ++i) if (cb_mul(dH, Hh[i * wd + i], &dH)) return -45;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (i < j) { Hadj[i * n + j] = 0; continue; }
            i128 s = i == j ? dH : 0;
            for (int k = j; k < i; ++k)                  // (H'^T)[i][k] = Hh[k][i]
                if (czk_mulsub(Hh[k * wd + i], Hadj[k * n + j], s, &s)) return -45;
            if (s % Hh[i * wd + i]) return -46;
            Hadj[i * n + j] = s / Hh[i * wd + i];
        }
    if (!czk_fits64(dH)) return -45;
    o->Hdet = (int64_t)dH;
    // c1 = E^T Hadj y_v / Hdet for v in Gamma with Lambda coordinates y_v:
    // Cs = E^T Hadj Py / gcd, Py[b][i] = Uinv[i][b+1]
    i128 Cn[MX * MX], HP[MX * MX], gC = dH;
    for (int a = 0; a < n; ++a)
        for (int j = 0; j < m; ++j) {
            i128 hp = 0;
            for (int b = 0; b <= a; ++b)
                if (czk_muladd(Hadj[a * n + b], Uinv[j * m + b + 1], hp, &hp)) return -45;
            HP[a * m + j] = hp;
        }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            i128 s = 0;
            for (int a = 0; a < n; ++a)
                if (Hh[a * wd + n + i] && czk_muladd(Hh[a * wd + n + i], HP[a * m + j], s, &s)) return -45;
            Cn[i * m + j] = s;
            gC = czk_gcd_acc(gC, s);
        }
    for (int i = 0; i < m * m; ++i) {
        i128 v = Cn[i] / gC;
        if (!czk_fits64(v)) return -45;
        o->Cs[i] = (int64_t)v;
    }
    o->Cden = (int64_t)(dH / gC);

    // S on Lambda and -s: A_rr X = d [B | a_r0] (Bareiss, exact), then
    // G = -B^T X[:, :n] / d = Gi / L, -s = (a_r0^T X[:, n] - d a00) / d
    i128 M[MX][2 * MX];
    const int wa = m + n + 1;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) M[i][j] = o->Z[(i + 1) * h + j + 1];
        for (int j = 0; j < n; ++j) M[i][m + j] = o->B[i * n + j];
        M[i][m + n] = o->Z[(i + 1) * h];
    }
    cb_divisor prev = cb_divisor_make(1);
    for (int k = 0; k < m; ++k) {
        int piv = k;
        while (piv < m && M[piv][k] == 0) ++piv;
        if (piv == m) return 1;                          // A_rr singular
        if (piv != k)
            for (int j = 0; j < wa; ++j) { i128 t = M[k][j]; M[k][j] = M[piv][j]; M[piv][j] = t; }
        for (int i = k + 1; i < m; ++i) {
            for (int j = k + 1; j < wa; ++j) {
                if (cb_mul(M[k][k], M[i][j], &t1) || czk_mulsub(M[i][k], M[k][j], t1, &t1)) return -50;
                M[i][j] = cb_divexact(t1, prev);
            }
            M[i][k] = 0;
        }
        prev = cb_divisor_make(M[k][k]);
    }
    const i128 d = M[m - 1][m - 1];
    i128 X[MX][MX + 1];
    cb_divisor pv[MX];
    for (int i = 0; i < m; ++i) pv[i] = cb_divisor_make(M[i][i]);
    for (int c = 0; c <= n; ++c)
        for (int i = m - 1; i >= 0; --i) {
            i128 s;
            if (cb_mul(d, M[i][m + c], &s)) return -50;
            for (int j = i + 1; j < m; ++j)
                if (czk_mulsub(M[i][j], X[j][c], s, &s)) return -50;
            X[i][c] = cb_divexact(s, pv[i]);
            if (cb_mul(X[i][c], M[i][i], &t1) || t1 != s) return -51;   // cannot happen (exact)
        }
    i128 sn;
    if (cb_mul(-d, (i128)o->Z[0], &sn)) return -50;
    for (int i = 0; i < m; ++i)
        if (czk_muladd(o->Z[(i + 1) * h], X[i][n], sn, &sn)) return -50;
    i128 sd = d;
    if (sd < 0) { sd = -sd; sn = -sn; }
    if (sn <= 0) return 1;                               // -s <= 0
    i128 gs = czk_gcd(sn, sd);
    sn /= gs; sd /= gs;
    if (!czk_fits64(sn) || !czk_fits64(sd)) return -52;
    o->sn = (int64_t)sn; o->sd = (int64_t)sd;
    i128 Gn[MX * MX], g = sd * gs;                       // = |d|
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            i128 s = 0;
            for (int i = 0; i < m; ++i)
                if (o->B[i * n + a] && czk_mulsub(o->B[i * n + a], X[i][b], s, &s)) return -50;
            Gn[a * n + b] = Gn[b * n + a] = s;
            g = czk_gcd_acc(g, s);
        }
    int64_t Gi[MX * MX];
    const i128 Ls = (d < 0 ? -d : d) / g;
    for (int k = 0; k < n * n; ++k) {
        i128 v = Gn[k] / g * (d < 0 ? -1 : 1);
        if (!czk_fits64(v)) return -52;
        Gi[k] = (int64_t)v;
    }
    if (!czk_fits64(Ls)) return -52;
    o->L = (int64_t)Ls;

    // LLL-reduce Gi: rows of V, Gr = V Gi V^T (exact)
    for (int k = 0; k < n; ++k) if (Gi[k * n + k] <= 0) return 1;   // not positive definite
    for (int k = 0; k < n * n; ++k) o->V[k] = k / n == k % n;
    if (n > 1) {
        rc = cb_lll(o->V, n, Gi);
        if (rc) return -60 + rc;
    }
    i128 GV[MX * MX];
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < n; ++b) {
            i128 r = 0;
            for (int j = 0; j < n; ++j)
                if (fpk_add_ovf(r, (i128)Gi[i * n + j] * o->V[b * n + j], &r)) return -61;
            GV[i * n + b] = r;
        }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            i128 s = 0;
            for (int i = 0; i < n; ++i)
                if (o->V[a * n + i] && czk_muladd(o->V[a * n + i], GV[i * n + b], s, &s)) return -61;
            if (!czk_fits64(s)) return -61;
            o->Gr[a * n + b] = o->Gr[b * n + a] = (int64_t)s;
        }

    // HT = Hadj V^T; w = l Cs B V^T; lk = l . kv
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            i128 s = 0;
            for (int j = 0; j < n; ++j)
                if (czk_muladd(Hadj[a * n + j], o->V[b * n + j], s, &s)) return -70;
            if (!czk_fits64(s)) return -70;
            o->HT[a * n + b] = (int64_t)s;
        }
    i128 lC[MX], lCB[MX], lk = 0;
    for (int j = 0; j < m; ++j) {
        i128 s = 0;
        for (int i = 0; i < m; ++i)
            if (czk_muladd(o->Binter[i], o->Cs[i * m + j], s, &s)) return -70;
        lC[j] = s;
    }
    for (int b = 0; b < n; ++b) {
        i128 s = 0;
        for (int j = 0; j < m; ++j)
            if (czk_muladd(lC[j], o->B[j * n + b], s, &s)) return -70;
        lCB[b] = s;
    }
    for (int b = 0; b < n; ++b) {
        i128 s = 0;
        for (int j = 0; j < n; ++j)
            if (czk_muladd(lCB[j], o->V[b * n + j], s, &s)) return -70;
        if (!czk_fits64(s)) return -70;
        o->w[b] = (int64_t)s;
    }
    for (int i = 0; i < m; ++i)
        if (czk_muladd(o->Binter[i], o->kv[i], lk, &lk)) return -70;
    if (!czk_fits64(lk)) return -70;
    if (lk == 0) return -71;                             // (M0 independent of t)
    o->lk = (int64_t)lk;
    return 0;
}

// search state for one direction
typedef struct {
    const czk_dir *d;
    fpk_i128 QL, M0min, D0;
    long max_out, pidx;
    czk_points *out;
} czk_ctx;

// one point c = j c1 + t kv: M = Binter c, Kn = ZB c, q = c^T mat c
static int czk_emit(czk_ctx *C, const fpk_i128 *c1, fpk_i128 j, fpk_i128 t)
{
    typedef fpk_i128 i128;
    const czk_dir *d = C->d;
    const int h = d->h, m = d->m;
    czk_points *P = C->out;
    if (P->n >= C->max_out) return -80;
    if (czk_points_reserve(P, h, P->n + 1)) return -81;
    int64_t c[CZK_MAX_H11];
    i128 s;
    for (int i = 0; i < m; ++i) {
        if (cb_mul(j, c1[i], &s) || czk_muladd(t, d->kv[i], s, &s) || !czk_fits64(s)) return -82;
        c[i] = (int64_t)s;
    }
    for (int i = 0; i < h; ++i) {
        i128 a = 0, b = 0;
        for (int k = 0; k < m; ++k)
            if (czk_muladd(d->Binter[i * m + k], c[k], a, &a) || czk_muladd(d->ZB[i * m + k], c[k], b, &b)) return -82;
        if (!czk_fits64(a) || !czk_fits64(b)) return -82;
        P->M[P->n * h + i] = (int64_t)a;
        P->Kn[P->n * h + i] = (int64_t)b;
    }
    i128 q = 0;
    for (int i = 0; i < m; ++i) {
        i128 r = 0;
        for (int k = 0; k < m; ++k)
            if (czk_muladd(d->mat[i * m + k], c[k], r, &r)) return -82;
        if (czk_muladd(c[i], r, q, &q)) return -82;
    }
    if (!czk_fits64(q)) return -82;
    P->q[P->n] = (int64_t)q;
    P->pidx[P->n] = C->pidx;
    P->n++;
    return 0;
}

// the points of one short K (reduced coordinates x; val = x^T Gr x, hx =
// HT x, wx = w . x, all exact)
static int czk_per_K(czk_ctx *C, const int64_t *x, fpk_i128 val, const fpk_i128 *hx, fpk_i128 wx)
{
    typedef fpk_i128 i128;
    const czk_dir *d = C->d;
    const int n = d->n, m = d->m;
    i128 t1;
    uint64_t gu = 0;                                     // g = gcd(x) = gcd(K) (V unimodular, B saturated)
    for (int i = 0; i < n; ++i) gu = fpk_gcd64(gu, (uint64_t)(x[i] < 0 ? -x[i] : x[i]));
    const i128 g = (i128)gu, kn = val / (g * g);        // kn = L K'^T S K'
    // o = order of K' = K / g in Lambda / Gamma: Hdet / gcd(Hdet, Hadj y')
    uint64_t og = (uint64_t)d->Hdet;
    for (int i = 0; i < n && og != 1; ++i) {
        const i128 v = gu == 1 ? hx[i] : hx[i] / g;
        const uint64_t r = v == (int64_t)v ? (uint64_t)((int64_t)v < 0 ? -(int64_t)v : (int64_t)v) % og
                                           : (uint64_t)(fpk_abs128(v) % og);
        if (r) og = fpk_gcd64(og, r);
    }
    const i128 o = (i128)((uint64_t)d->Hdet / og);
    // l . c1 = o (w . y') / Cden, with W c1 = o K'
    i128 num, den;
    if (cb_mul(o, wx, &num) || cb_mul(g, d->Cden, &den)) return -90;
    if (num % den) return -91;                           // cannot happen
    const i128 lc1 = num / den, lk = d->lk, alk = lk < 0 ? -lk : lk;
    // j: j o > g D0 and g j o kn < Q L
    i128 gLsn, gokn, gD0;
    if (cb_mul(g, d->L, &gLsn) || cb_mul(gLsn, d->sn, &gLsn)
            || cb_mul(g, o, &gokn) || cb_mul(gokn, kn, &gokn) || cb_mul(g, C->D0, &gD0)) return -90;
    const i128 jlo = czk_fdiv(gD0, o) + 1, jhi = (C->QL - 1) / gokn;
    if (jlo > jhi) return 0;
    // for each j: M0 = j lc1 + t lk >= M0min with g L sn M0^2 < R(j) =
    // sd j o (Q L - g j o kn). M0 runs over a residue class mod |lk|; a
    // float screen (margin 1e-9, rounding errors ~1e-15) skips j whose
    // smallest admissible M0 fails, and everything else is decided exactly.
    const i128 step = ((lc1 % alk) + alk) % alk, m0m = ((C->M0min % alk) + alk) % alk;
    i128 res;
    if (cb_mul(jlo % alk, step, &res)) return -90;
    res %= alk;                                          // j lc1 mod |lk|, for j = jlo
    const double gLsn_d = (double)gLsn, sdo_d = (double)d->sd * (double)o;
    int have_c1 = 0;
    i128 c1[CZK_MAX_H11];
    for (i128 j = jlo; j <= jhi; ++j, res = res + step >= alk ? res + step - alk : res + step) {
        i128 off = res - m0m;
        if (off < 0) off += alk;
        const i128 M0c = C->M0min + off;                 // the smallest M0 >= M0min in the class
        i128 diff;                                       // Q L - g j o kn > 0, exact
        if (cb_mul(gokn, j, &diff) || fpk_sub_ovf(C->QL, diff, &diff)) return -90;
        const double Rd = sdo_d * (double)j * (double)diff;
        if ((double)M0c * (double)M0c * gLsn_d > Rd * (1 + 1e-9) + 1.0) continue;
        i128 R, jo;
        if (cb_mul(j, o, &jo) || cb_mul(jo, d->sd, &R) || cb_mul(R, diff, &R)) return -90;
        for (i128 M0 = M0c;; M0 += alk) {
            i128 lhs;
            if (cb_mul(M0, M0, &lhs) || cb_mul(lhs, gLsn, &lhs)) break;   // (beyond R)
            if (lhs >= R) break;
            if (cb_mul(j, lc1, &t1) || fpk_sub_ovf(M0, t1, &t1)) return -90;
            if (t1 % lk) return -92;                     // cannot happen
            if (!have_c1) {                              // c1 = Cs (o K') / Cden
                i128 y[CZK_MAX_H11], v[CZK_MAX_H11];
                for (int b = 0; b < n; ++b) {
                    i128 s = 0;
                    for (int a = 0; a < n; ++a)
                        if (czk_muladd(x[a], d->V[a * n + b], s, &s)) return -90;
                    y[b] = s / g;
                }
                for (int i = 0; i < m; ++i) {
                    i128 s = 0;
                    for (int b = 0; b < n; ++b)
                        if (czk_muladd(d->B[i * n + b], y[b], s, &s)) return -90;
                    if (cb_mul(o, s, &v[i])) return -90;
                }
                for (int i = 0; i < m; ++i) {
                    i128 s = 0;
                    for (int k = 0; k < m; ++k)
                        if (czk_muladd(d->Cs[i * m + k], v[k], s, &s)) return -90;
                    if (s % d->Cden) return -91;         // cannot happen
                    c1[i] = s / d->Cden;
                }
                have_c1 = 1;
            }
            int rc = czk_emit(C, c1, j, t1 / lk);
            if (rc) return rc;
        }
    }
    return 0;
}

// all short K (x^T Gr x D0 < Q L, x != 0), each through czk_per_K. As
// cb_min: floats only prune, by a rigorous lower bound on x^T Gr x; leaves
// are decided exactly (with partial sums carried down the tree exactly).
static int czk_search(const czk_dir *d, int64_t Q, int64_t M0min, int64_t D0, long max_out,
                      long pidx, czk_points *out)
{
    typedef fpk_i128 i128;
    enum { MX = CZK_MAX_H11 };
    const int n = d->n;
    czk_ctx C = { d, 0, M0min, D0, max_out, pidx, out };
    if (D0 < 1 || M0min < 1) return -100;
    if (cb_mul(Q, d->L, &C.QL)) return -101;
    double R[MX][MX] = {{0}}, Ri[MX][MX] = {{0}};
    for (int j = 0; j < n; ++j) {                        // Gr = R^T R
        double s = (double)d->Gr[j * n + j];
        for (int k = 0; k < j; ++k) s -= R[k][j] * R[k][j];
        if (!(s > 0)) return -102;
        R[j][j] = sqrt(s);
        for (int i = j + 1; i < n; ++i) {
            double t = (double)d->Gr[j * n + i];
            for (int k = 0; k < j; ++k) t -= R[k][j] * R[k][i];
            R[j][i] = t / R[j][j];
        }
    }
    for (int j = n - 1; j >= 0; --j) {
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
    if (!(g1 * 2.0 * nR * nRi < 1e-6)) return -103;     // too ill-conditioned
    // every x with x^T Gr x < Q L / D0 has |R x|^2 < rad (see cb_min)
    const double rad = (double)C.QL / (double)D0 / (1.0 - g1 * 2.0 * nR * nRi) * (1 + 1e-12);

    // exact partial sums over the levels >= i: Gr x, HT x, w . x, x^T Gr x
    // (|x_i| < 2^20 keeps them far inside int128)
    i128 gx[MX + 1][MX], hx[MX + 1][MX], wx[MX + 1], vals[MX + 1];
    double rem[MX + 1], aoff[MX], ofs[MX];
    int64_t x[MX], lo[MX], hi[MX];
    for (int k = 0; k < n; ++k) { gx[n][k] = 0; hx[n][k] = 0; }
    wx[n] = 0; vals[n] = 0; rem[n] = rad;
    int i = n - 1;
    #define CZK_ENTER(ii) do {                                                  \
        double o_ = 0, a_ = 0;                                                  \
        for (int j_ = (ii) + 1; j_ < n; ++j_) { double t_ = R[ii][j_] * x[j_]; o_ += t_; a_ += fabs(t_); } \
        double ctr_ = -o_ / R[ii][ii], w_ = sqrt(fmax(rem[(ii) + 1], 0.0)) / R[ii][ii]; \
        double e_ = g2 * (fabs(ctr_) + w_ + 1 + a_ / R[ii][ii]) * 4 + 4 * u * (fabs(ctr_) + w_); \
        if (!(e_ < 0.5) || !(fabs(ctr_) + w_ < 1e6)) return -104;               \
        lo[ii] = (int64_t)floor(ctr_ - w_) - 1; hi[ii] = (int64_t)ceil(ctr_ + w_) + 1; \
        aoff[ii] = a_; ofs[ii] = o_; x[ii] = lo[ii];                            \
    } while (0)
    CZK_ENTER(i);
    for (;;) {
        if (x[i] > hi[i]) {                                          // level done
            if (++i == n) break;
            x[i]++;
            continue;
        }
        const double t = R[i][i] * x[i] + ofs[i], a = fabs(R[i][i] * x[i]) + aoff[i];
        const double lb = fmax(fabs(t) - g2 * a, 0.0) * (1 - 4 * u);
        const double r = rem[i + 1] - lb * lb * (1 - 4 * u);
        if (r < 0) { x[i]++; continue; }                             // pruned (rigorously)
        const i128 xi = x[i];
        if (xi >= (1 << 20) || xi <= -(1 << 20)) return -105;
        for (int k = 0; k < n; ++k) {
            gx[i][k] = gx[i + 1][k] + (i128)d->Gr[k * n + i] * xi;
            hx[i][k] = hx[i + 1][k] + (i128)d->HT[k * n + i] * xi;
        }
        wx[i] = wx[i + 1] + (i128)d->w[i] * xi;
        vals[i] = vals[i + 1] + xi * (2 * gx[i + 1][i] + (i128)d->Gr[i * n + i] * xi);
        if (i > 0) {
            rem[i] = r;
            --i;
            CZK_ENTER(i);
            continue;
        }
        if (vals[0] > 0) {                                           // a leaf: exact (x = 0 has val 0)
            i128 lhs;
            if (cb_mul(vals[0], D0, &lhs)) return -101;
            if (lhs < C.QL) {
                int rc = czk_per_K(&C, x, vals[0], hx[0], wx[0]);
                if (rc) return rc;
            }
        } else {
            int nz = 0;
            for (int j = 0; j < n; ++j) nz |= x[j] != 0;
            if (nz) return 1;                                        // not positive definite (exactly)
        }
        x[i]++;
    }
    #undef CZK_ENTER
    return 0;
}

// czk_setup + czk_search for np directions (rows of ps, length h, p[0] = 0),
// appending to out; status[i] as for czk_setup / czk_search (its points
// are kept only if 0)
static void czk_batch(int h, const int64_t *kappa, const int64_t *Mbasis, const int64_t *ps, long np,
                      int64_t Q, int64_t M0min, int64_t D0, long max_out, czk_points *out, int *status)
{
    czk_dir *d = (czk_dir *)malloc(sizeof(czk_dir));
    if (!d) { for (long i = 0; i < np; ++i) status[i] = -81; return; }
    for (long i = 0; i < np; ++i) {
        const long n0 = out->n;
        int rc = czk_setup(h, kappa, Mbasis, &ps[i * h], d);
        if (!rc) rc = czk_search(d, Q, M0min, D0, max_out, i, out);
        if (rc) out->n = n0;
        status[i] = rc;
    }
    free(d);
}

#endif  // CONI_ZPK_H
