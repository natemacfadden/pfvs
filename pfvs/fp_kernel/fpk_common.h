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
#ifndef FPK_COMMON_H
#define FPK_COMMON_H

// Exact-arithmetic building blocks shared by fp_kernel.h, pfv_lattice.h and
// the GPU pipeline: header-only C/C++, usable in device code (FPK_HD), no GMP
// or heap.

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define FPK_HD __host__ __device__
#else
#define FPK_HD
#endif

__extension__ typedef __int128          fpk_i128;
__extension__ typedef unsigned __int128 fpk_u128;


// Checked int128 add/sub (returns 1 on overflow). Written out rather than
// __builtin_{add,sub}_overflow, which CUDA device code does not provide.
FPK_HD static inline int fpk_add_ovf(fpk_i128 a, fpk_i128 b, fpk_i128 *r)
{
#if !defined(__CUDA_ARCH__) && defined(__GNUC__)
    return __builtin_add_overflow(a, b, r);      /* host: same result, faster */
#endif
    fpk_i128 s = (fpk_i128)((fpk_u128)a + (fpk_u128)b);
    *r = s;
    return ((a >= 0) == (b >= 0)) && ((s >= 0) != (a >= 0));
}

FPK_HD static inline int fpk_sub_ovf(fpk_i128 a, fpk_i128 b, fpk_i128 *r)
{
#if !defined(__CUDA_ARCH__) && defined(__GNUC__)
    return __builtin_sub_overflow(a, b, r);
#endif
    fpk_i128 s = (fpk_i128)((fpk_u128)a - (fpk_u128)b);
    *r = s;
    return ((a >= 0) != (b >= 0)) && ((s >= 0) != (a >= 0));
}

// integer helpers
// ---------------
FPK_HD static inline fpk_u128 fpk_abs128(fpk_i128 x)
{
    return x < 0 ? (fpk_u128)0 - (fpk_u128)x : (fpk_u128)x;
}

// r = a * b, returns 1 on overflow. Not __builtin_mul_overflow: older clang
// lowers it on __int128 to __muloti4, which libgcc lacks.
FPK_HD static inline int fpk_mul_ovf(fpk_i128 a, fpk_i128 b, fpk_i128 *r)
{
    fpk_u128 ua = fpk_abs128(a), ub = fpk_abs128(b);
    const fpk_u128 lim = (fpk_u128)1 << 63;
    if (ua < lim && ub < lim) { *r = a * b; return 0; }   /* |a b| < 2^126 */
    if (ua && ub > (((fpk_u128)1 << 127) - 1) / ua) return 1;  /* |a b| >= 2^127 */
    fpk_u128 m = ua * ub;
    *r = ((a < 0) != (b < 0)) ? -(fpk_i128)m : (fpk_i128)m;
    return 0;
}

// count trailing zeros of x != 0
FPK_HD static inline int fpk_ctz64(uint64_t x)
{
#ifdef __CUDA_ARCH__
    return __clzll(__brevll(x));
#else
    return __builtin_ctzll(x);
#endif
}

FPK_HD static inline int fpk_ctz128(fpk_u128 x)
{
    uint64_t lo = (uint64_t)x;
    return lo ? fpk_ctz64(lo) : 64 + fpk_ctz64((uint64_t)(x >> 64));
}

FPK_HD static inline uint64_t fpk_gcd64(uint64_t u, uint64_t v)
{
    // Stein's binary GCD
    if (u == 0) return v;
    if (v == 0) return u;
    int shift = fpk_ctz64(u | v);
    u >>= fpk_ctz64(u);
    do {
        v >>= fpk_ctz64(v);
        if (u > v) { uint64_t t = v; v = u; u = t; }
        v -= u;
    } while (v);
    return u << shift;
}

// gcd(u, v) if it is >= need, else 0 (u, v not both 0). Stein's algorithm,
// exiting early once gcd <= min(u, v) << shift drops below `need`.
FPK_HD static inline uint64_t fpk_gcd64_ge(uint64_t u, uint64_t v, uint64_t need)
{
    if (u == 0) return v >= need ? v : 0;
    if (v == 0) return u >= need ? u : 0;
    int shift = fpk_ctz64(u | v);
    // smallest odd-part bound that can still reach need: ceil(need / 2^shift)
    uint64_t need_s = (need >> shift) + ((need & ((1ULL << shift) - 1)) != 0);
    u >>= fpk_ctz64(u);
    do {
        v >>= fpk_ctz64(v);
        if (u > v) { uint64_t t = v; v = u; u = t; }
        if (u < need_s) return 0;
        v -= u;
    } while (v);
    return u << shift;
}


FPK_HD static inline fpk_u128 fpk_gcd128(fpk_u128 u, fpk_u128 v)
{
    if ((u >> 64) == 0 && (v >> 64) == 0)
        return fpk_gcd64((uint64_t)u, (uint64_t)v);
    if (u == 0) return v;
    if (v == 0) return u;
    int shift = fpk_ctz128(u | v);
    u >>= fpk_ctz128(u);
    do {
        v >>= fpk_ctz128(v);
        if (u > v) { fpk_u128 t = v; v = u; u = t; }
        v -= u;
        if ((u >> 64) == 0 && (v >> 64) == 0)
            return (fpk_u128)fpk_gcd64((uint64_t)u, (uint64_t)v) << shift;
    } while (v);
    return u << shift;
}

// A gcd value: machine-sized (u128) unless it does not fit, then GMP.
typedef struct {
    int      big;
    fpk_u128 s;
} fpk_gval;


// exact q(c) = c^T mat c in int128; returns 1 on overflow (then the caller
// computes it in GMP on the CPU)
FPK_HD static inline int fpk_exact_q128(const int64_t *mat, const int32_t *c, int dim,
                                        fpk_i128 *q_out)
{
    fpk_i128 q = 0;
    int ovf = 0;
    for (int i = 0; i < dim && !ovf; ++i) {
        if (c[i] == 0) continue;
        fpk_i128 row = 0;
        for (int j = i + 1; j < dim; ++j)
            row += (fpk_i128)mat[i * dim + j] * c[j];   // |.| < 2^102
        fpk_i128 t = 0;
        // (short-circuit: stop at the first overflow)
        ovf = fpk_mul_ovf(row, (fpk_i128)2, &t)
           || fpk_add_ovf(t, (fpk_i128)mat[i * dim + i] * c[i], &t)
           || fpk_mul_ovf(t, (fpk_i128)c[i], &t)
           || fpk_add_ovf(q, t, &q);
    }
    if (ovf) return 1;
    *q_out = q;
    return 0;
}



// Sparse candidates for a level whose GCD cut dominates
// ----------------------------------------------------
// At a level with a single H row the gcd is d(v) = gcd(G, pre + h v), a
// divisor of the parent gcd G, and v survives only if Q d(v) >= q_lb(v). For
// each divisor d >= need_min, the v with d | pre + h v are one residue class
// mod d / gcd(d, h); generate those within the radius where Q d >= q_lb and
// keep v if d(v) == d. Skips only candidates the GCD test would reject.
// Writes them ascending to `list`; returns their number, or -1 if not
// worthwhile.
#define FPK_SPARSE_MAX_G   (1u << 20)
#define FPK_SPARSE_CAP     256
#define FPK_SPARSE_MIN_W   16

FPK_HD static inline int64_t fpk_modinv(int64_t a, int64_t m)   // a invertible mod m > 1
{
    int64_t t = 0, nt = 1, r = m, nr = a % m;
    while (nr) {
        int64_t q = r / nr, tmp;
        tmp = t - q * nt; t = nt; nt = tmp;
        tmp = r - q * nr; r = nr; nr = tmp;
    }
    return t < 0 ? t + m : t;
}

// ascending insertion sort (small lists; qsort is not available on GPUs)
FPK_HD static inline void fpk_isort_i32(int32_t *x, int n)
{
    for (int i = 1; i < n; ++i) {
        int32_t v = x[i];
        int j = i - 1;
        while (j >= 0 && x[j] > v) { x[j + 1] = x[j]; --j; }
        x[j + 1] = v;
    }
}



// Fast exact factorization (int128)
// ---------------------------------
// fpk_factor_exact's elimination in int128. The Bareiss numerator
// p_k A_ij - A_ik A_kj may overflow but is divisible by p_{k-1}: form it mod
// 2^128 and recover the quotient as (N >> s) * odd(p_{k-1})^{-1} mod 2^(128-s),
// 2^s || p_{k-1}, valid if |quotient| < 2^(127-s) (checked first by a
// rigorous double estimate; else fall back to GMP). Also returns |V|-sums
// (sum_k |R_kj z_k|), keeping the M0 bound rigorous since V is accumulated
// in floating point.
#ifndef FPK_FAST_FACTOR_MAX_DIM
#define FPK_FAST_FACTOR_MAX_DIM 32
#endif

FPK_HD static inline fpk_u128 fpk_inv2k(fpk_u128 o)      // o odd: o^{-1} mod 2^128
{
    fpk_u128 x = o;                              // correct to 3 bits
    for (int i = 0; i < 6; ++i) x *= (fpk_u128)2 - o * x;
    return x;
}

// q = (a b - c d) / e exactly (e > 0 divides a b - c d); 0 on success
FPK_HD static inline int fpk_bareiss_div(fpk_i128 a, fpk_i128 b, fpk_i128 c, fpk_i128 d,
                                  fpk_i128 e, fpk_i128 *q)
{
    double ab = (double)a * (double)b, cd = (double)c * (double)d, ed = (double)e;
    double qf = (ab - cd) / ed;
    double qerr = 1e-15 * (fabs(ab) + fabs(cd)) / ed + 2.0;
    int s = fpk_ctz128((fpk_u128)e);
    if (s > 100 || fabs(qf) + qerr >= ldexp(1.0, 125 - s)) return -1;
    fpk_u128 N = (fpk_u128)a * (fpk_u128)b - (fpk_u128)c * (fpk_u128)d;
    fpk_u128 Q = (N >> s) * fpk_inv2k((fpk_u128)e >> s);
    if (s > 0) {                                 // keep 128-s bits, sign-extend
        fpk_u128 mask = ((fpk_u128)1 << (128 - s)) - 1;
        Q &= mask;
        if ((Q >> (127 - s)) & 1) Q |= ~mask;
    }
    *q = (fpk_i128)Q;
    return 0;
}

FPK_HD static inline int fpk_factor_fast(const int64_t *mat, int dim, double *U,
                           const int64_t *linvec, double *m0N, double *m0V,
                           double *m0Va)
{
    enum { MD = FPK_FAST_FACTOR_MAX_DIM };
    if (dim > MD) return -1;
    fpk_i128 A[MD][MD + 1];                      // last column: linvec
    int na = dim + (linvec != NULL);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) A[i][j] = mat[i * dim + j];
        if (linvec) A[i][dim] = linvec[i];
    }
    fpk_i128 prev = 1;
    double N = 0.0, V[MD], Va[MD];
    for (int j = 0; j < dim; ++j) V[j] = Va[j] = 0.0;

    for (int k = 0; k < dim; ++k) {
        fpk_i128 pk = A[k][k];
        if (pk <= 0) return -9;                  // exact: not positive definite
        double pkd = (double)pk, prevd = (double)prev;
        double ukk = sqrt(pkd / prevd);
        U[k * dim + k] = ukk;
        for (int j = 0; j < k; ++j) U[k * dim + j] = 0.0;
        for (int j = k + 1; j < dim; ++j) U[k * dim + j] = ukk * ((double)A[k][j] / pkd);
        if (linvec) {
            // z_k = zhat_k / p_{k-1}; 1/d_k = p_{k-1} / p_k
            double z = (double)A[k][dim] / prevd;
            N += z * z * (prevd / pkd);
            m0N[k] = N;
            for (int j = k + 1; j < dim; ++j) {
                double t = ((double)A[k][j] / pkd) * z;
                V[j] += t;
                Va[j] += fabs(t);
                m0V[k * dim + j] = V[j];
                m0Va[k * dim + j] = Va[j];
            }
        }
        // eliminate rows below k (symmetric part j >= i, plus the linvec column)
        for (int i = k + 1; i < dim; ++i) {
            for (int j = i; j < na; ++j) {
                if (j == dim && !linvec) break;
                fpk_i128 q;
                if (fpk_bareiss_div(pk, A[i][j], A[i][k], A[k][j], prev, &q)) return -1;
                A[i][j] = q;
                if (j < dim && j != i) A[j][i] = q;
            }
        }
        prev = pk;
    }
    return 0;
}


// Shared search (CPU and GPU)
// ---------------------------
// fp_kernel.h's search for the common case (every H row fits int64, dim <=
// FPK_SEARCH_MAXD), for host and device; same order, tests and output.
// For work splitting, c[dim-n_prefix:] can be fixed to `prefix`, and with
// stop_depth > 0 the search emits the top stop_depth coordinates (kind 1)
// instead of descending. Points are kind 0; a leaf whose exact q needs more
// than int128 is kind 2, for the caller to decide.
#ifndef FPK_SEARCH_MAXD
#define FPK_SEARCH_MAXD 256
#endif
// residue mode (fpk_search_impl.h) costs an int128 modulo per row per node:
// use it from this many candidates (GPU build: off)
// sparse candidate lists in fpk_search: 0 compiles them out (GPU build)
#ifndef FPK_SEARCH_SPARSE
#define FPK_SEARCH_SPARSE 1
#endif
#ifndef FPK_RES_MIN_W
#define FPK_RES_MIN_W 1
#endif

// emit(ctx, kind, c, n, q): kind 0 = point c[0:n] with exact q; kind 1 =
// prefix (the top n coordinates, c[dim-n:]); kind 2 = leaf c[0:n] still to be
// decided (all cuts but the exact q <= qmax and GCD tests passed). Returns
// nonzero to stop.
typedef int (*fpk_emit_fn)(void *ctx, int kind, const int32_t *c, int n, int64_t q);

typedef struct {
    int64_t n_nodes, n_cand, n_leaf;
} fpk_counts;

// The float-dependent parts, instantiated in double (names as is) and, for
// the GPU, float (suffix _f; see fpk_search_impl.h).
#define FPK_R            double
#define FPK_N(name)      name
#define FPK_L(x)         (x)
#define FPK_U            0x1p-53
#define FPK_G            1e-9
#define FPK_TINY         1e-300
#define FPK_I32MAX_R     2147483647.0
#define FPK_SQRT         sqrt
#define FPK_FABS         fabs
#define FPK_CEIL         ceil
#define FPK_FLOOR        floor
#define FPK_FMAX         fmax
#include "fpk_search_impl.h"
#if 1   /* float instantiation: GPUs, and fp_kernel.h's CPU fast path */
#define FPK_R            float
#define FPK_N(name)      name##_f
#define FPK_L(x)         ((float)(x))
#define FPK_U            0x1p-24f
#define FPK_G            (64 * 0x1p-24f)
#define FPK_TINY         1e-30f
#define FPK_I32MAX_R     2147483520.0f     /* largest float below 2^31 */
#define FPK_SQRT         sqrtf
#define FPK_FABS         fabsf
#define FPK_CEIL         ceilf
#define FPK_FLOOR        floorf
#define FPK_FMAX         fmaxf
#include "fpk_search_impl.h"
#endif

#endif // FPK_COMMON_H
