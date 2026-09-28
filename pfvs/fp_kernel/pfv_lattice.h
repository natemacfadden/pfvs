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

#include <gmp.h>
#include <math.h>
#include <stdlib.h>
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

// |x| as unsigned 128-bit
static inline pfl_u128 pfl_uabs128(pfl_i128 x)
{
    return x < 0 ? (pfl_u128)0 - (pfl_u128)x : (pfl_u128)x;
}

// r = a * b with overflow detection (returns 1 on overflow). Written out
// rather than __builtin_mul_overflow on __int128, which older clang lowers
// to a __muloti4 call that libgcc does not provide (link failure on Linux).
static inline int pfl_mul_ovf(pfl_i128 a, pfl_i128 b, pfl_i128 *r)
{
    pfl_u128 ua = pfl_uabs128(a), ub = pfl_uabs128(b);
    const pfl_u128 lim = (pfl_u128)1 << 63;
    if (ua < lim && ub < lim) { *r = a * b; return 0; }   /* |a b| < 2^126 */
    if (ua && ub > (((pfl_u128)1 << 127) - 1) / ua) return 1;  /* |a b| >= 2^127 */
    pfl_u128 m = ua * ub;
    *r = ((a < 0) != (b < 0)) ? -(pfl_i128)m : (pfl_i128)m;
    return 0;
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
                    if (pfl_mul_ovf(q, A[row * n + c], &x)) return -1;
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
                if (pfl_mul_ovf(q, A[row * n + c], &x)) return -1;
                if (__builtin_sub_overflow(A[i * n + c], x, &x)) return -1;
                if (!PFL_FITS125(x)) return -1;
                A[i * n + c] = x;
            }
        }
        row++;
    }
    return 0;
}

// The same HNF in GMP, for when the int128 version's intermediates exceed
// 2^125 (common at h11 >= 10: the final H fits, the elimination does not).
// Exact; the result (canonical, as flint's) is returned in int128, or -1 if
// an entry of the final H does not fit.
static int pfl_hnf_mpz(const int64_t *A64, int r, int n, pfl_i128 *Aout)
{
    mpz_t *A = malloc((size_t)r * n * sizeof(mpz_t));
    if (!A) return -1;
    mpz_t q;
    mpz_init(q);
    for (int i = 0; i < r * n; ++i) mpz_init_set_si(A[i], A64[i]);
    int row = 0, rc = 0;
    for (int col = 0; col < n && row < r; ++col) {
        for (;;) {
            int piv = -1;
            for (int i = row; i < r; ++i)
                if (mpz_sgn(A[i * n + col]) != 0 &&
                    (piv < 0 || mpz_cmpabs(A[i * n + col], A[piv * n + col]) < 0))
                    piv = i;
            if (piv < 0) break;
            if (piv != row)
                for (int c = 0; c < n; ++c) mpz_swap(A[row * n + c], A[piv * n + c]);
            int done = 1;
            for (int i = row + 1; i < r; ++i) {
                if (mpz_sgn(A[i * n + col]) == 0) continue;
                mpz_fdiv_q(q, A[i * n + col], A[row * n + col]);
                for (int c = col; c < n; ++c) mpz_submul(A[i * n + c], q, A[row * n + c]);
                if (mpz_sgn(A[i * n + col]) != 0) done = 0;
            }
            if (done) break;
        }
        if (mpz_sgn(A[row * n + col]) == 0) continue;
        if (mpz_sgn(A[row * n + col]) < 0)
            for (int c = col; c < n; ++c) mpz_neg(A[row * n + c], A[row * n + c]);
        for (int i = 0; i < row; ++i) {
            mpz_fdiv_q(q, A[i * n + col], A[row * n + col]);
            if (mpz_sgn(q) == 0) continue;
            for (int c = col; c < n; ++c) mpz_submul(A[i * n + c], q, A[row * n + c]);
        }
        row++;
    }
    for (int i = 0; i < r * n && !rc; ++i) {
        if (mpz_sizeinbase(A[i], 2) > 125) { rc = -1; break; }
        uint64_t w[2] = {0, 0};
        mpz_export(w, NULL, -1, sizeof(uint64_t), 0, 0, A[i]);
        pfl_i128 v = (pfl_i128)(((pfl_u128)w[1] << 64) | w[0]);
        Aout[i] = mpz_sgn(A[i]) < 0 ? -v : v;
    }
    for (int i = 0; i < r * n; ++i) mpz_clear(A[i]);
    mpz_clear(q);
    free(A);
    return rc;
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


// LLL core (delta 0.99) on the m basis vectors B (rows, length n), given
// their exact Gram matrix GG ((m + 1) x (m + 1), row-major, stride m + 1;
// under whatever inner product the caller chose). If `last` is given it is
// row m of GG: after the reduction it is size-reduced (not swapped) against
// the basis. The Gram matrix is kept exact (checked int128) under every
// basis operation, so Gram-Schmidt needs no inner products: floating point
// only chooses the unimodular operations, which are exact.
static int pfl_lll_core(int64_t *B, int m, int n, pfl_i128 *GG, int64_t *last)
{
    const int L = m + 1;                        // GG stride
    const int M = m + (last != NULL);           // rows tracked in GG
    double mu[PFL_MAX_H11][PFL_MAX_H11], bb[PFL_MAX_H11], mw[PFL_MAX_H11];
    const double delta = 0.99;
    int iters = 0;

    #define PFL_ROW(kk) ((kk) == m ? last : &B[(kk) * n])
    // b_k -= q b_j, on the vectors and on GG (exact)
    #define PFL_SUB(kk, jj, qi)                                                \
    do {                                                                       \
        int64_t *bk_ = PFL_ROW(kk); const int64_t *bj_ = PFL_ROW(jj);          \
        for (int c_ = 0; c_ < n; ++c_) {                                       \
            pfl_i128 x_ = (pfl_i128)bk_[c_] - (pfl_i128)(qi) * bj_[c_];        \
            if (!PFL_FITS64(x_)) return -1;                                    \
            bk_[c_] = (int64_t)x_;                                             \
        }                                                                      \
        pfl_i128 kj_ = GG[(kk) * L + (jj)], jj_ = GG[(jj) * L + (jj)], t_, u_; \
        /* new G_kk = G_kk - 2 q G_kj + q^2 G_jj */                            \
        if (pfl_mul_ovf(jj_, (pfl_i128)(qi), &t_) ||                           \
            __builtin_sub_overflow(t_, 2 * kj_, &t_) ||                        \
            pfl_mul_ovf(t_, (pfl_i128)(qi), &u_) ||                            \
            __builtin_add_overflow(GG[(kk) * L + (kk)], u_, &u_)) return -1;   \
        for (int l_ = 0; l_ < M; ++l_) {                                       \
            if (l_ == (kk)) continue;                                          \
            if (pfl_mul_ovf(GG[(jj) * L + l_], (pfl_i128)(qi), &t_) ||         \
                __builtin_sub_overflow(GG[(kk) * L + l_], t_, &t_)) return -1; \
            GG[(kk) * L + l_] = t_;                                            \
            GG[l_ * L + (kk)] = t_;                                            \
        }                                                                      \
        GG[(kk) * L + (kk)] = u_;                                              \
    } while (0)

    #define PFL_GS(kk)                                                         \
    do {                                                                       \
        for (int j_ = 0; j_ < (kk); ++j_) {                                    \
            double d_ = (double)GG[(kk) * L + j_];                             \
            for (int l_ = 0; l_ < j_; ++l_) d_ -= mu[j_][l_] * mu[(kk)][l_] * bb[l_]; \
            mu[(kk)][j_] = d_ / bb[j_];                                        \
        }                                                                      \
        double d_ = (double)GG[(kk) * L + (kk)];                               \
        for (int l_ = 0; l_ < (kk); ++l_) d_ -= mu[(kk)][l_] * mu[(kk)][l_] * bb[l_]; \
        bb[(kk)] = d_;                                                         \
    } while (0)

    if (m >= 1) { PFL_GS(0); if (!(bb[0] > 0)) return -2; }
    if (m >= 2) PFL_GS(1);
    int k = 1;
    while (k < m) {
        if (++iters > 100000) return -3;
        for (int pass = 0; pass < 4; ++pass) {  // size reduction
            int changed = 0;
            for (int j = k - 1; j >= 0; --j) {
                double q = nearbyint(mu[k][j]);
                if (q == 0.0) continue;
                if (!(fabs(q) <= 9.0e18)) return -1;   /* also NaN */
                int64_t qi = (int64_t)q;
                PFL_SUB(k, j, qi);
                for (int l = 0; l < j; ++l) mu[k][l] -= q * mu[j][l];
                mu[k][j] -= q;
                changed = 1;
            }
            if (!changed) break;
            PFL_GS(k);                          // refresh from the exact Gram
        }
        if (!(bb[k] > 0)) return -2;
        if (bb[k] < (delta - mu[k][k - 1] * mu[k][k - 1]) * bb[k - 1]) {
            for (int c = 0; c < n; ++c) {       // swap b_k, b_{k-1}
                int64_t t = B[k * n + c]; B[k * n + c] = B[(k - 1) * n + c]; B[(k - 1) * n + c] = t;
            }
            for (int l = 0; l < M; ++l) {       // ... and rows/columns of GG
                pfl_i128 t = GG[k * L + l]; GG[k * L + l] = GG[(k - 1) * L + l]; GG[(k - 1) * L + l] = t;
            }
            for (int l = 0; l < M; ++l) {
                pfl_i128 t = GG[l * L + k]; GG[l * L + k] = GG[l * L + (k - 1)]; GG[l * L + (k - 1)] = t;
            }
            k = k > 1 ? k - 1 : 1;
            if (k == 1) PFL_GS(0);
            PFL_GS(k - 1);
            PFL_GS(k);
        } else {
            k++;
            if (k < m) PFL_GS(k);
        }
    }
    if (last && m > 0) {                        // size-reduce `last` (row m)
        for (int pass = 0; pass < 4; ++pass) {
            for (int j = 0; j < m; ++j) {
                double d_ = (double)GG[m * L + j];
                for (int l = 0; l < j; ++l) d_ -= mu[j][l] * mw[l] * bb[l];
                mw[j] = d_ / bb[j];
            }
            int changed = 0;
            for (int j = m - 1; j >= 0; --j) {
                double q = nearbyint(mw[j]);
                if (q == 0.0) continue;
                if (!(fabs(q) <= 9.0e18)) return -1;
                int64_t qi = (int64_t)q;
                PFL_SUB(m, j, qi);
                for (int l = 0; l < j; ++l) mw[l] -= q * mu[j][l];
                mw[j] -= q;
                changed = 1;
            }
            if (!changed) break;
        }
    }
    #undef PFL_ROW
    #undef PFL_SUB
    #undef PFL_GS
    return 0;
}

// Euclidean LLL of the m rows of B (length n).
static int pfl_lll(int64_t *B, int m, int n)
{
    if (m < 2) return 0;
    pfl_i128 GG[(PFL_MAX_H11 + 1) * (PFL_MAX_H11 + 1)];
    const int L = m + 1;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) {
            pfl_i128 sum = 0;
            for (int c = 0; c < n; ++c)
                if (__builtin_add_overflow(sum, (pfl_i128)B[i * n + c] * B[j * n + c], &sum)) return -1;
            GG[i * L + j] = GG[j * L + i] = sum;
        }
    return pfl_lll_core(B, m, n, GG, NULL);
}

// LLL of the m coefficient vectors B (rows, length n) with respect to the
// quadratic form G (n x n); then size-reduce `last` (if given) against them.
static int pfl_lll_gram(int64_t *B, int m, int n, const int64_t *G, int64_t *last)
{
    pfl_i128 GG[(PFL_MAX_H11 + 1) * (PFL_MAX_H11 + 1)];
    pfl_i128 Gy[PFL_MAX_H11];
    const int L = m + 1, M = m + (last != NULL);
    for (int j = 0; j < M; ++j) {
        const int64_t *y = j == m ? last : &B[j * n];
        for (int i = 0; i < n; ++i) {           // Gy = G y (int64 * int64 terms)
            pfl_i128 r = 0;
            for (int c = 0; c < n; ++c)
                if (__builtin_add_overflow(r, (pfl_i128)G[i * n + c] * y[c], &r)) return -1;
            Gy[i] = r;
        }
        for (int i = 0; i <= j; ++i) {          // GG[i][j] = x_i . G y_j
            const int64_t *x = i == m ? last : &B[i * n];
            pfl_i128 sum = 0, t;
            for (int c = 0; c < n; ++c) {
                if (!x[c]) continue;
                if (pfl_mul_ovf(Gy[c], (pfl_i128)x[c], &t) || __builtin_add_overflow(sum, t, &sum))
                    return -1;
            }
            GG[i * L + j] = GG[j * L + i] = sum;
        }
    }
    return pfl_lll_core(B, m, n, GG, last);
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
        for (int k = 0; k < h; ++k)          /* each term < 2^126; the sum is checked */
            if (__builtin_add_overflow(s, (pfl_i128)S->kappa[i * h + k] * p[k], &s)) return -1;
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
            for (int l = 0; l < h; ++l)
                if (__builtin_add_overflow(s, (pfl_i128)S->Mbasis[i * h + l] * O[a * h + l], &s))
                    return -1;
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
    if (rc == -1)                 /* int128 intermediates overflowed: redo in GMP */
        rc = pfl_hnf_mpz(&R->ZB[r0 * d], R->nrows, d, R->H);
    return rc ? rc * 10 - 5 : 0;
}

#endif // PFV_LATTICE_IMPLEMENTATION

#endif // PFV_LATTICE_H
