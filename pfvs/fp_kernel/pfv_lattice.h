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
#ifndef PFV_LATTICE_H
#define PFV_LATTICE_H

// HEADER
// ======
#include <stdint.h>

__extension__ typedef __int128 pfl_i128;
__extension__ typedef unsigned __int128 pfl_u128;

/*
**Description:**
The per-p-vector lattice setup of the Zp pipelines, in C: given the triple
intersection numbers kappa, the M-lattice basis Mbasis and an integer
p-vector p, builds

    Z      = kappa . p                                     (h11, h11)
    Binter = LLL-reduced basis of {M in Mbasis Z^h11 : p.Z.M = 0}
                                                           (h11, h11-1)
    mat    = -Binter^T Z Binter                            (h11-1, h11-1)
    H      = row-HNF of (Z Binter)[r0:]  (r0 = 1 coni, 0 non-coni)

exactly as coniZp.coni_M_ellipsoid / coni_H_matrix (resp. Zp.M_ellipsoid /
H_matrix) do, except that LLL-reduced bases are not unique: Binter may be a
different (equally valid) basis of the same lattice. All results depend
only on the lattice, so the PFVs found are the same.

All arithmetic is exact integer arithmetic (int64 values, int128
intermediates) with overflow detection. Floating point is only used inside
LLL to choose unimodular basis operations; the basis updates themselves are
exact, so the lattice is always preserved exactly -- floating-point error
can only make the basis less reduced (slower), never wrong. On overflow (or
any other failure) the function returns a nonzero status and the caller
falls back to the arbitrary-precision Python path.

**Returns:**
     0: success
    -1: an intermediate overflowed int64 / int128 (use the exact fallback)
    -2: degenerate input (e.g. p.Z.Mbasis == 0, or rank loss in LLL)
    -3: LLL did not converge within its iteration budget
Stage codes: -1x orthogonal lattice, -2x first LLL, -3x second LLL, -15 HNF.
On -15 (HNF intermediates exceed int128) everything except H is valid, so
the caller only needs to compute H itself.

Cut-aware basis (coni, m0_basis = 1): the kernel fixes M0 = Binter[0,:].c
once every coordinate where Binter[0,:] is nonzero is set, and prunes on
the ellipsoid geometry given by mat. So Binter is finally changed to
Binter T with T = [K | w] unimodular: K an LLL basis of ker(Binter[0,:])
reduced with respect to mat (not the Euclidean norm), w with
Binter[0,:].w = g, size-reduced against K. Then Binter[0,:] = (0,...,0,g):
M0 is fixed at the very first level searched, where M0 >= M0min becomes an
interval bound. Same lattice, same PFVs; ~4.7x smaller searches on heavy
h11 = 10 problems. Non-coni (no M0 cut): T is the LLL reduction of the whole
basis with respect to mat, which alone shrinks searches ~1.4-1.8x.
*/
typedef struct {
    int h11;
    const int64_t *kappa;    // (h11, h11, h11), symmetric
    const int64_t *Mbasis;   // (h11, h11), basis vectors as columns
    int coni;                // 1: coni (sort Binter, H from rows 1:), 0: non-coni
    int extra_lll;           // LLL-reduce the orthogonal lattice first (default 1)
    int m0_basis;            // final basis change w.r.t. mat (see below), default 1
} pfl_setup;

typedef struct {
    int64_t Z[64 * 64];      // (h11, h11)
    int64_t Binter[64 * 63]; // (h11, h11-1) row-major
    int64_t ZB[64 * 63];     // (h11, h11-1) = Z Binter
    int64_t mat[63 * 63];    // (h11-1, h11-1)
    pfl_i128 H[64 * 63];     // (nrows, h11-1), row-HNF (entries < 2^125)
    int     nrows;
} pfl_result;

#define PFL_MAX_H11 64

int pfl_build(const pfl_setup *S, const int64_t *p, pfl_result *R);


// IMPLEMENTATION
// ==============
#ifdef PFV_LATTICE_IMPLEMENTATION

#include <math.h>
#include <string.h>

#define PFL_I64_MAX ((pfl_i128)INT64_MAX)
#define PFL_I64_MIN ((pfl_i128)INT64_MIN)
// (INT64_MIN is excluded so negation and x / -1 are always defined)
#define PFL_FITS64(x) ((x) <= PFL_I64_MAX && (x) > PFL_I64_MIN)

// Extended Euclid on int64: returns g = gcd(a, b) >= 0 and s, t with
// s a + t b = g, matching util.extended_euclidean (including signs).
static inline int64_t pfl_xgcd(int64_t a, int64_t b, int64_t *s, int64_t *t)
{
    int64_t or_ = a, r = b, os = 1, s_ = 0, ot = 0, t_ = 1;
    while (r != 0) {
        // Python floor division semantics (util.extended_euclidean uses //)
        int64_t q = or_ / r;
        if ((or_ % r != 0) && ((or_ < 0) != (r < 0))) q--;
        int64_t tmp;
        tmp = or_ - q * r;  or_ = r;  r = tmp;
        tmp = os - q * s_;  os = s_;  s_ = tmp;
        tmp = ot - q * t_;  ot = t_;  t_ = tmp;
    }
    *s = os; *t = ot;
    return or_;
}

// |x| as unsigned (defined for INT64_MIN)
static inline uint64_t pfl_uabs(int64_t x)
{
    return x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x;
}

// floor division, as Python's //
static inline int64_t pfl_fdiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

// Basis of the lattice orthogonal to v (length n), as the n-1 rows of O
// (row-major, stride n). A port of util._orthogonal_lattice_int64 (same
// Bezout elimination, same result), with exact overflow checks.
static int pfl_unimodular(const int64_t *v, int n, int64_t *U);

static int pfl_orthogonal(const int64_t *v, int n, int64_t *O)
{
    int64_t U[PFL_MAX_H11 * PFL_MAX_H11];
    int rc = pfl_unimodular(v, n, U);
    if (rc) return rc;
    memcpy(O, &U[n], (size_t)(n - 1) * n * sizeof(int64_t));
    return 0;
}

// Unimodular U (n x n, row-major) with U v = (gcd(v), 0, ..., 0): rows 1..
// span the lattice orthogonal to v, row 0 has v . U[0] = gcd(v).
static int pfl_unimodular(const int64_t *v, int n, int64_t *Uout)
{
    int64_t U[PFL_MAX_H11 * PFL_MAX_H11];
    int64_t w[PFL_MAX_H11];
    for (int i = 0; i < n; ++i) {
        w[i] = v[i];
        for (int j = 0; j < n; ++j) U[i * n + j] = (i == j);
    }
    for (int k = 1; k < n; ++k) {
        if (w[k] == 0) continue;
        int64_t a = w[0], b = w[k], s, t;
        int64_t g = pfl_xgcd(a, b, &s, &t);
        if (g == 0) return -2;
        int64_t m10 = -pfl_fdiv(b, g), m11 = pfl_fdiv(a, g);
        w[0] = g;
        w[k] = 0;
        for (int r = 0; r < n; ++r) {
            pfl_i128 t1 = (pfl_i128)s * U[r] + (pfl_i128)t * U[k * n + r];
            pfl_i128 t2 = (pfl_i128)m10 * U[r] + (pfl_i128)m11 * U[k * n + r];
            if (!PFL_FITS64(t1) || !PFL_FITS64(t2)) return -1;
            U[r] = (int64_t)t1;
            U[k * n + r] = (int64_t)t2;
        }
    }
    memcpy(Uout, U, (size_t)n * n * sizeof(int64_t));
    return 0;
}

// LLL-reduce the m basis vectors (rows of B, length n, stride n), delta =
// 0.99. Floating-point Gram-Schmidt chooses the operations; the updates
// b_k -= q b_j and swaps are exact (checked) integer operations.
static int pfl_lll(int64_t *B, int m, int n)
{
    double mu[PFL_MAX_H11][PFL_MAX_H11], bb[PFL_MAX_H11];
    const double delta = 0.99;
    int k = 1, iters = 0;

    // Gram-Schmidt row k from scratch (rows < k are already done)
    #define PFL_GS_ROW(kk)                                                     \
    do {                                                                       \
        for (int j_ = 0; j_ < (kk); ++j_) {                                    \
            double d_ = 0.0;                                                   \
            for (int c_ = 0; c_ < n; ++c_)                                     \
                d_ += (double)B[(kk) * n + c_] * (double)B[j_ * n + c_];       \
            for (int l_ = 0; l_ < j_; ++l_) d_ -= mu[j_][l_] * mu[(kk)][l_] * bb[l_]; \
            mu[(kk)][j_] = d_ / bb[j_];                                        \
        }                                                                      \
        double d_ = 0.0;                                                       \
        for (int c_ = 0; c_ < n; ++c_)                                         \
            d_ += (double)B[(kk) * n + c_] * (double)B[(kk) * n + c_];         \
        for (int l_ = 0; l_ < (kk); ++l_) d_ -= mu[(kk)][l_] * mu[(kk)][l_] * bb[l_]; \
        bb[(kk)] = d_;                                                         \
    } while (0)

    if (m < 2) return 0;
    PFL_GS_ROW(0);
    if (!(bb[0] > 0)) return -2;
    PFL_GS_ROW(1);

    while (k < m) {
        if (++iters > 100000) return -3;
        // size reduction (repeat while floating point leaves |mu| > 1/2)
        for (int pass = 0; pass < 4; ++pass) {
            int changed = 0;
            for (int j = k - 1; j >= 0; --j) {
                double q = nearbyint(mu[k][j]);
                if (q == 0.0) continue;
                if (fabs(q) > 9.0e18) return -1;
                int64_t qi = (int64_t)q;
                for (int c = 0; c < n; ++c) {
                    pfl_i128 x = (pfl_i128)B[k * n + c] - (pfl_i128)qi * B[j * n + c];
                    if (!PFL_FITS64(x)) return -1;
                    B[k * n + c] = (int64_t)x;
                }
                changed = 1;
                for (int l = 0; l < j; ++l) mu[k][l] -= q * mu[j][l];
                mu[k][j] -= q;
            }
            if (!changed) break;
            PFL_GS_ROW(k);   // refresh after the integer updates
        }
        if (!(bb[k] > 0)) return -2;
        if (bb[k] < (delta - mu[k][k - 1] * mu[k][k - 1]) * bb[k - 1]) {
            for (int c = 0; c < n; ++c) {
                int64_t t = B[k * n + c];
                B[k * n + c] = B[(k - 1) * n + c];
                B[(k - 1) * n + c] = t;
            }
            k = k > 1 ? k - 1 : 1;
            if (k == 1) PFL_GS_ROW(0);
            PFL_GS_ROW(k - 1);
            PFL_GS_ROW(k);
        } else {
            k++;
            if (k < m) PFL_GS_ROW(k);
        }
    }
    #undef PFL_GS_ROW
    return 0;
}

// |x| as unsigned 128-bit
static inline pfl_u128 pfl_uabs128(pfl_i128 x)
{
    return x < 0 ? (pfl_u128)0 - (pfl_u128)x : (pfl_u128)x;
}

// floor division on int128 (b != 0; no overflow for |a|, |b| < 2^126)
static inline pfl_i128 pfl_fdiv128(pfl_i128 a, pfl_i128 b)
{
    pfl_i128 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

#define PFL_I128_LIM ((pfl_i128)1 << 125)
#define PFL_FITS125(x) ((x) < PFL_I128_LIM && (x) > -PFL_I128_LIM)

// Row-style Hermite normal form of the r x n integer matrix A (row-major),
// matching flint's fmpz_mat_hnf: upper echelon, positive pivots, entries
// above each pivot reduced into [0, pivot), zero rows last. Worked in int128
// (intermediates grow well beyond the final entries); exact, with overflow
// detection. Input int64 (A64), output int128 (Aout).
static int pfl_hnf(const int64_t *A64, int r, int n, pfl_i128 *Aout)
{
    pfl_i128 *A = Aout;
    for (int i = 0; i < r * n; ++i) A[i] = A64[i];
    int row = 0;
    for (int col = 0; col < n && row < r; ++col) {
        // Euclid on column `col` over rows row..r-1 until one nonzero remains
        for (;;) {
            int piv = -1;
            for (int i = row; i < r; ++i)
                if (A[i * n + col] != 0 &&
                    (piv < 0 || pfl_uabs128(A[i * n + col]) < pfl_uabs128(A[piv * n + col])))
                    piv = i;
            if (piv < 0) break;                      // column is zero here
            if (piv != row)
                for (int c = 0; c < n; ++c) {
                    pfl_i128 t = A[row * n + c];
                    A[row * n + c] = A[piv * n + c];
                    A[piv * n + c] = t;
                }
            int done = 1;
            for (int i = row + 1; i < r; ++i) {
                pfl_i128 a = A[i * n + col];
                if (a == 0) continue;
                pfl_i128 q = pfl_fdiv128(a, A[row * n + col]);
                for (int c = col; c < n; ++c) {
                    pfl_i128 x;
                    if (__builtin_mul_overflow(q, A[row * n + c], &x)) return -1;
                    if (__builtin_sub_overflow(A[i * n + c], x, &x)) return -1;
                    if (!PFL_FITS125(x)) return -1;
                    A[i * n + c] = x;
                }
                if (A[i * n + col] != 0) done = 0;
            }
            if (done) break;
        }
        if (A[row * n + col] == 0) continue;
        if (A[row * n + col] < 0)
            for (int c = col; c < n; ++c) A[row * n + c] = -A[row * n + c];
        // reduce the entries above the pivot into [0, pivot)
        pfl_i128 pv = A[row * n + col];
        for (int i = 0; i < row; ++i) {
            pfl_i128 q = pfl_fdiv128(A[i * n + col], pv);
            if (!q) continue;
            for (int c = col; c < n; ++c) {
                pfl_i128 x;
                if (__builtin_mul_overflow(q, A[row * n + c], &x)) return -1;
                if (__builtin_sub_overflow(A[i * n + c], x, &x)) return -1;
                if (!PFL_FITS125(x)) return -1;
                A[i * n + c] = x;
            }
        }
        row++;
    }
    return 0;
}

// C = A B for int64 matrices with exact overflow checks
static int pfl_matmul(const int64_t *A, const int64_t *B, int64_t *C,
                      int m, int k, int n)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            pfl_i128 s = 0;
            for (int l = 0; l < k; ++l) {
                pfl_i128 t = (pfl_i128)A[i * k + l] * B[l * n + j];
                if (__builtin_add_overflow(s, t, &s)) return -1;
            }
            if (!PFL_FITS64(s)) return -1;
            C[i * n + j] = (int64_t)s;
        }
    return 0;
}


// y -> G y (exact, checked int128); returns 0 on overflow
static inline int pfl_gvec(const int64_t *y, const int64_t *G, int n, pfl_i128 *Gy)
{
    for (int i = 0; i < n; ++i) {
        pfl_i128 r = 0;
        for (int j = 0; j < n; ++j) {
            pfl_i128 t;
            if (__builtin_mul_overflow((pfl_i128)G[i * n + j], (pfl_i128)y[j], &t) ||
                __builtin_add_overflow(r, t, &r)) return 0;
        }
        Gy[i] = r;
    }
    return 1;
}

// x . (G y) given Gy (exact, checked), as double; *ok cleared on overflow
static inline double pfl_gdot(const int64_t *x, const pfl_i128 *Gy, int n, int *ok)
{
    pfl_i128 s = 0;
    for (int i = 0; i < n; ++i) {
        if (!x[i]) continue;
        pfl_i128 t;
        if (__builtin_mul_overflow(Gy[i], (pfl_i128)x[i], &t) ||
            __builtin_add_overflow(s, t, &s)) { *ok = 0; return 0.0; }
    }
    return (double)s;
}

// LLL (delta 0.99) of the m coefficient vectors B (rows, length n) with
// respect to the quadratic form G; if `last` is given, afterwards size-reduce
// that vector (not swapped) against the reduced basis. The products G b_j are
// cached and updated exactly with the basis. Updates are exact (checked)
// integer operations; floating point only chooses them, so the lattice is
// preserved exactly.
static int pfl_lll_gram(int64_t *B, int m, int n, const int64_t *G, int64_t *last)
{
    double mu[PFL_MAX_H11][PFL_MAX_H11], bb[PFL_MAX_H11], mw[PFL_MAX_H11];
    pfl_i128 GB[PFL_MAX_H11][PFL_MAX_H11];      // GB[j] = G b_j
    const double delta = 0.99;
    int ok = 1, iters = 0;
    for (int j = 0; j < m; ++j) if (!pfl_gvec(&B[j * n], G, n, GB[j])) return -1;

    #define PFL_GGS_ROW(kk)                                                    \
    do {                                                                       \
        for (int j_ = 0; j_ < (kk); ++j_) {                                    \
            double d_ = pfl_gdot(&B[(kk) * n], GB[j_], n, &ok);                \
            for (int l_ = 0; l_ < j_; ++l_) d_ -= mu[j_][l_] * mu[(kk)][l_] * bb[l_]; \
            mu[(kk)][j_] = d_ / bb[j_];                                        \
        }                                                                      \
        double d_ = pfl_gdot(&B[(kk) * n], GB[(kk)], n, &ok);                  \
        for (int l_ = 0; l_ < (kk); ++l_) d_ -= mu[(kk)][l_] * mu[(kk)][l_] * bb[l_]; \
        bb[(kk)] = d_;                                                         \
    } while (0)

    if (m >= 1) { PFL_GGS_ROW(0); if (!(bb[0] > 0)) return -2; }
    if (m >= 2) PFL_GGS_ROW(1);
    int k = 1;
    while (k < m) {
        if (!ok) return -1;
        if (++iters > 100000) return -3;
        for (int pass = 0; pass < 4; ++pass) {
            int changed = 0;
            for (int j = k - 1; j >= 0; --j) {
                double q = nearbyint(mu[k][j]);
                if (q == 0.0) continue;
                if (fabs(q) > 9.0e18) return -1;
                int64_t qi = (int64_t)q;
                for (int c = 0; c < n; ++c) {
                    pfl_i128 x = (pfl_i128)B[k * n + c] - (pfl_i128)qi * B[j * n + c];
                    if (!PFL_FITS64(x)) return -1;
                    B[k * n + c] = (int64_t)x;
                    pfl_i128 t;
                    if (__builtin_mul_overflow(GB[j][c], (pfl_i128)qi, &t) ||
                        __builtin_sub_overflow(GB[k][c], t, &GB[k][c])) return -1;
                }
                changed = 1;
                for (int l = 0; l < j; ++l) mu[k][l] -= q * mu[j][l];
                mu[k][j] -= q;
            }
            if (!changed) break;
            PFL_GGS_ROW(k);
        }
        if (!ok) return -1;
        if (!(bb[k] > 0)) return -2;
        if (bb[k] < (delta - mu[k][k - 1] * mu[k][k - 1]) * bb[k - 1]) {
            for (int c = 0; c < n; ++c) {
                int64_t t = B[k * n + c];
                B[k * n + c] = B[(k - 1) * n + c];
                B[(k - 1) * n + c] = t;
                pfl_i128 u = GB[k][c]; GB[k][c] = GB[k - 1][c]; GB[k - 1][c] = u;
            }
            k = k > 1 ? k - 1 : 1;
            if (k == 1) PFL_GGS_ROW(0);
            PFL_GGS_ROW(k - 1);
            PFL_GGS_ROW(k);
        } else {
            k++;
            if (k < m) PFL_GGS_ROW(k);
        }
    }
    if (last && m > 0) {
        // Gram-Schmidt coefficients of `last`, then size-reduce it
        for (int pass = 0; pass < 4; ++pass) {
            for (int j = 0; j < m; ++j) {
                double d_ = pfl_gdot(last, GB[j], n, &ok);
                for (int l = 0; l < j; ++l) d_ -= mu[j][l] * mw[l] * bb[l];
                mw[j] = d_ / bb[j];
            }
            if (!ok) return -1;
            int changed = 0;
            for (int j = m - 1; j >= 0; --j) {
                double q = nearbyint(mw[j]);
                if (q == 0.0) continue;
                if (fabs(q) > 9.0e18) return -1;
                int64_t qi = (int64_t)q;
                for (int c = 0; c < n; ++c) {
                    pfl_i128 x = (pfl_i128)last[c] - (pfl_i128)qi * B[j * n + c];
                    if (!PFL_FITS64(x)) return -1;
                    last[c] = (int64_t)x;
                }
                for (int l = 0; l < j; ++l) mw[l] -= q * mu[j][l];
                mw[j] -= q;
                changed = 1;
            }
            if (!changed) break;
        }
    }
    #undef PFL_GGS_ROW
    return ok ? 0 : -1;
}

// Cut-aware basis change (see the header comment): Binter <- Binter T,
// ZB <- ZB T, mat <- T^T mat T with T = [K | w].
static int pfl_m0_basis(pfl_result *R, int h, int d, int coni)
{
    int64_t l[PFL_MAX_H11], V[PFL_MAX_H11 * PFL_MAX_H11];
    int64_t T[PFL_MAX_H11 * PFL_MAX_H11];      // columns: new basis vectors
    int64_t tmp[PFL_MAX_H11 * PFL_MAX_H11];
    int rc;
    if (!coni) {
        // no M0 cut: LLL-reduce the whole basis w.r.t. mat
        for (int a = 0; a < d; ++a)
            for (int i = 0; i < d; ++i) V[a * d + i] = (a == i);
        if ((rc = pfl_lll_gram(V, d, d, R->mat, NULL))) return rc;
        for (int a = 0; a < d; ++a)
            for (int i = 0; i < d; ++i) T[i * d + a] = V[a * d + i];
    } else {
        for (int a = 0; a < d; ++a) l[a] = R->Binter[a];
        int nz = 0;
        for (int a = 0; a < d; ++a) nz += (l[a] != 0);
        if (nz == 0) return 0;                 // M0 == 0: nothing to gain
        if ((rc = pfl_unimodular(l, d, V))) return rc;   // V l = (g, 0, ..., 0)
        // K = rows 1.. of V (d-1 vectors), w = row 0
        int64_t *K = &V[d];
        int64_t w[PFL_MAX_H11];
        for (int a = 0; a < d; ++a) w[a] = V[a];
        if ((rc = pfl_lll_gram(K, d - 1, d, R->mat, w))) return rc;
        for (int a = 0; a < d - 1; ++a)
            for (int i = 0; i < d; ++i) T[i * d + a] = K[a * d + i];
        for (int i = 0; i < d; ++i) T[i * d + (d - 1)] = w[i];
    }

    if ((rc = pfl_matmul(R->Binter, T, tmp, h, d, d))) return rc;
    memcpy(R->Binter, tmp, (size_t)h * d * sizeof(int64_t));
    if ((rc = pfl_matmul(R->ZB, T, tmp, h, d, d))) return rc;
    memcpy(R->ZB, tmp, (size_t)h * d * sizeof(int64_t));
    // mat <- T^T (mat T)
    int64_t MT[PFL_MAX_H11 * PFL_MAX_H11];
    if ((rc = pfl_matmul(R->mat, T, MT, d, d, d))) return rc;
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
            pfl_i128 s = 0;
            for (int i = 0; i < d; ++i) {
                pfl_i128 t = (pfl_i128)T[i * d + a] * MT[i * d + b];
                if (__builtin_add_overflow(s, t, &s)) return -1;
            }
            if (!PFL_FITS64(s)) return -1;
            tmp[a * d + b] = (int64_t)s;
        }
    memcpy(R->mat, tmp, (size_t)d * d * sizeof(int64_t));
    return 0;
}

int pfl_build(const pfl_setup *S, const int64_t *p, pfl_result *R)
{
    const int h = S->h11, d = S->h11 - 1;
    if (h < 2 || h > PFL_MAX_H11) return -2;
    int64_t T[PFL_MAX_H11], v[PFL_MAX_H11];
    int64_t O[PFL_MAX_H11 * PFL_MAX_H11];      // (d, h) rows = basis vectors
    int64_t BT[PFL_MAX_H11 * PFL_MAX_H11];     // Binter^T: (d, h)
    int rc;
    for (int i = 0; i < h * h * h; ++i) if (S->kappa[i] == INT64_MIN) return -1;
    for (int i = 0; i < h * h; ++i) if (S->Mbasis[i] == INT64_MIN) return -1;
    for (int i = 0; i < h; ++i) if (p[i] == INT64_MIN) return -1;

    // Z = kappa . p ; T = Z p ; v = T Mbasis
    for (int i = 0; i < h * h; ++i) {
        pfl_i128 s = 0;
        for (int k = 0; k < h; ++k) s += (pfl_i128)S->kappa[i * h + k] * p[k];
        if (!PFL_FITS64(s)) return -1;
        R->Z[i] = (int64_t)s;
    }
    if ((rc = pfl_matmul(R->Z, p, T, h, h, 1))) return rc;
    if ((rc = pfl_matmul(T, S->Mbasis, v, 1, h, h))) return rc;

    // orthogonal lattice (+ LLL), then Binter = Mbasis O^T (+ LLL)
    if ((rc = pfl_orthogonal(v, h, O))) return rc * 10 - 1;
    if (S->extra_lll && (rc = pfl_lll(O, d, h))) return rc * 10 - 2;
    // BT[a][i] = sum_l O[a][l] Mbasis[i][l]  (i.e. (Mbasis O^T)^T)
    for (int a = 0; a < d; ++a)
        for (int i = 0; i < h; ++i) {
            pfl_i128 s = 0;
            for (int l = 0; l < h; ++l) s += (pfl_i128)S->Mbasis[i * h + l] * O[a * h + l];
            if (!PFL_FITS64(s)) return -1;
            BT[a * h + i] = (int64_t)s;
        }
    // (with the final basis change, Binter is re-reduced w.r.t. mat anyway)
    if (!S->m0_basis && (rc = pfl_lll(BT, d, h))) return rc * 10 - 3;

    // coni: columns with Binter[0] == 0 first (stable)
    int order[PFL_MAX_H11], no = 0;
    if (S->coni) {
        for (int a = 0; a < d; ++a) if (BT[a * h] == 0) order[no++] = a;
        for (int a = 0; a < d; ++a) if (BT[a * h] != 0) order[no++] = a;
    } else {
        for (int a = 0; a < d; ++a) order[no++] = a;
    }
    for (int i = 0; i < h; ++i)
        for (int a = 0; a < d; ++a)
            R->Binter[i * d + a] = BT[order[a] * h + i];

    // ZB = Z Binter ; mat = -Binter^T ZB
    if ((rc = pfl_matmul(R->Z, R->Binter, R->ZB, h, h, d))) return rc;
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b) {
            pfl_i128 s = 0;
            for (int i = 0; i < h; ++i) {
                pfl_i128 t = (pfl_i128)R->Binter[i * d + a] * R->ZB[i * d + b];
                if (__builtin_add_overflow(s, t, &s)) return -1;
            }
            if (!PFL_FITS64(s) || s == PFL_I64_MIN) return -1;
            R->mat[a * d + b] = (int64_t)(-s);
        }

    if (S->m0_basis && (rc = pfl_m0_basis(R, h, d, S->coni)))
        return rc * 10 - 4;

    // H = HNF of ZB[r0:]
    int r0 = S->coni ? 1 : 0;
    R->nrows = h - r0;
    rc = pfl_hnf(&R->ZB[r0 * d], R->nrows, d, R->H);
    return rc ? rc * 10 - 5 : 0;
}

#endif // PFV_LATTICE_IMPLEMENTATION

#endif // PFV_LATTICE_H
