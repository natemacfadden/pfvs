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
#ifndef FP_KERNEL_H
#define FP_KERNEL_H

// HEADER
// ======
#include <stdint.h>
#include <gmp.h>

/*
**Description:**
Fincke-Pohst enumeration of integer vectors c in an ellipsoid, with the cuts
used to construct (coni-)PFVs pruned during the search. Shared by the coni
and non-coni pipelines. Returns exactly the vectors c in Z^dim with

    (ellipsoid)  q(c) := c^T @ mat @ c <= qmax
    (M0 cut)     dot(linvec, c) >= linmin               [only if linvec]
    (GCD cut)    g == 0  or  Q*g >= q(c)                [Q*g > q(c) if strict]

where g = gcd(H @ c). Every accept/reject decision at the leaves is made in
exact integer arithmetic. Floating point (a Cholesky factor U of mat,
computed here) only prunes the search, and every floating-point test is
widened by a rigorous bound on its rounding error (see "Error bound" below),
so pruning never discards a valid vector.

Why the GCD cut: a vector with q(c) > Q can still give a PFV under tadpole
Q once K is divided by g = gcd(K) (see `coni_H_matrix` in coniZp.py). The
search sets c from its last component to its first. A row of H whose first
nonzero entry is in column i is fully determined once c[i:] is set, so the
gcd of the determined rows is a nonincreasing upper bound on g, and q of
c[i:] alone is a nondecreasing lower bound on q(c). A branch is pruned as
soon as these bounds prove the cut must fail. The bound is sharpest when H is
in row-echelon form (e.g. an HNF), but any H is valid.

Error bound: the factor U (mat = U^T U) is computed exactly (fraction-free
LDL^T in GMP, i.e. Bareiss elimination) and only then rounded to double, so
each entry of U is within a few ulps of its true value however
ill-conditioned mat is. Writing t_i = sum_{j>=i} U_ij c_j and
a_i = sum_{j>=i} |U_ij c_j|, each float t_i is then within K*a_i of the true
one, K ~ (dim+8) * 2^-53, and so every float partial norm is within
K * sum_i (a_i^2 + rem_i) of the exact partial norm. The kernel carries this
bound down the search (with a 2x safety factor) and widens every float test
by it. For well-scaled problems the bound is negligible; it matters when mat
has huge entries but small q(c) (heavy cancellation).

Most of the work is in writing to `out`.

**Arguments:**
- `P`:   The problem (see `fpk_problem`). Not modified.
- `out`: The output (see `fpk_output`). Must be zero-initialized; the kernel
         allocates `out->pts` and `out->qs` (release with `fpk_output_free`).

**Returns:**
A status code according to following list:
     0: success
    -2: exceeded max_N_out outputs (the first max_N_out are returned)
    -6: problem dimension too high (currently >256) or < 1
    -7: out of memory
    -8: a coordinate bound overflows int32 (ellipsoid too large)
    -9: mat is not positive definite (decided exactly)
*/
typedef struct {
    int dim;
    // ellipsoid: q(c) = c^T mat c <= qmax
    const int64_t *mat;      // (dim, dim) row-major, symmetric positive definite
    int64_t        qmax;
    // GCD cut (disabled if nrows == 0)
    int64_t        Q;        // > 0
    int            nrows;
    mpz_t         *H;        // (nrows, dim) row-major (ignored if H64)
    const int64_t *H64;      // optional: H as int64 (skips GMP), or NULL
    int            strict;
    // M0 cut (disabled if linvec == NULL)
    const int64_t *linvec;   // (dim,)
    int64_t        linmin;
    // misc
    long           max_N_out;
    double         eps;      // extra absolute pruning slack on q
} fpk_problem;

typedef struct {
    int32_t *pts;            // (n, dim) row-major
    int64_t *qs;             // (n,) exact q(c)
    long     n;
    long     cap;
    long     n_nodes;        // search-tree nodes visited (diagnostic)
    long     n_cand;         // candidate values tried (diagnostic)
    long     n_leaf;         // exact leaf checks (diagnostic)
} fpk_output;

int  fpk_enumerate(const fpk_problem *P, fpk_output *out);
void fpk_output_free(fpk_output *out);


// IMPLEMENTATION
// ==============
#ifdef FP_KERNEL_IMPLEMENTATION

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "fpk_common.h"

#define FPK_MAX_DIM 256

// Optional search statistics (compile with -DFPK_STATS): rejected candidates
// by reason and level, fpk_stats[reason][level]. Zero cost when disabled.
#ifdef FPK_STATS
enum { FPK_REJ_ELLIPSOID, FPK_REJ_M0, FPK_REJ_GCD, FPK_REJ_LEAF, FPK_N_REJ };
long fpk_stats[FPK_N_REJ][FPK_MAX_DIM];
#define FPK_STAT(reason, level) (fpk_stats[(reason)][(level)]++)
#else
#define FPK_STAT(reason, level) ((void)0)
#endif

static inline void fpk_mpz_set_u128(mpz_t z, fpk_u128 x)
{
    uint64_t w[2] = { (uint64_t)x, (uint64_t)(x >> 64) };
    mpz_import(z, 2, -1, sizeof(uint64_t), 0, 0, w);
}

static inline int fpk_mpz_fits_u128(const mpz_t z)
{
    return mpz_sgn(z) >= 0 && mpz_sizeinbase(z, 2) <= 128;
}

static inline fpk_u128 fpk_mpz_get_u128(const mpz_t z)
{
    uint64_t w[2] = { 0, 0 };
    mpz_export(w, NULL, -1, sizeof(uint64_t), 0, 0, z);
    return ((fpk_u128)w[1] << 64) | w[0];
}

// z += h * v for a signed machine integer v
static inline void fpk_mpz_addmul_si(mpz_t z, const mpz_t h, int32_t v)
{
    if (v >= 0) mpz_addmul_ui(z, h, (unsigned long)v);
    else        mpz_submul_ui(z, h, (unsigned long)(-(int64_t)v));
}

// exact q(c) = c^T mat c, with a GMP fallback on (unlikely) int128 overflow
static int fpk_exact_q(const int64_t *mat, const int32_t *c, int dim,
                       fpk_i128 *q_out, mpz_t tmp, mpz_t acc)
{
    if (!fpk_exact_q128(mat, c, dim, q_out)) return 0;

    // slow path: exact in GMP (the result only matters vs qmax)
    mpz_set_ui(acc, 0);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            mpz_set_si(tmp, mat[i * dim + j]);
            mpz_mul_si(tmp, tmp, c[i]);
            mpz_mul_si(tmp, tmp, c[j]);
            mpz_add(acc, acc, tmp);
        }
    if (mpz_fits_slong_p(acc)) { *q_out = mpz_get_si(acc); return 0; }
    return 1;   // |q| >= 2^63: certainly > qmax
}

static int fpk_push(fpk_output *out, const int32_t *c, int dim, int64_t q,
                    long max_N_out)
{
    if (out->n >= max_N_out) return -2;
    if (out->n == out->cap) {
        long cap = out->cap ? 2 * out->cap : 64;
        if (cap > max_N_out) cap = max_N_out;
        int32_t *pts = realloc(out->pts, (size_t)cap * dim * sizeof(int32_t));
        if (!pts) return -7;
        out->pts = pts;
        int64_t *qs = realloc(out->qs, (size_t)cap * sizeof(int64_t));
        if (!qs) return -7;
        out->qs = qs;
        out->cap = cap;
    }
    memcpy(&out->pts[(size_t)out->n * dim], c, (size_t)dim * sizeof(int32_t));
    out->qs[out->n] = q;
    out->n++;
    return 0;
}

void fpk_output_free(fpk_output *out)
{
    free(out->pts);
    free(out->qs);
    out->pts = NULL;
    out->qs = NULL;
    out->n = out->cap = 0;
}

// Upper-triangular U with U^T U = mat, computed exactly and rounded to
// double entrywise. Fraction-free (Bareiss) elimination on mat: with
// p_k the k-th pivot (the leading principal minor of size k+1, p_{-1} = 1)
// and A^(k) the eliminated matrix,
//     d_k = p_k / p_{k-1},   U_kk = sqrt(d_k),   U_kj = U_kk A^(k)_kj / p_k.
// Returns -9 if mat is not positive definite, -7 on allocation failure.
//
// If linvec is given, also returns (exactly computed, then rounded) the data
// for bounding M0 = linvec . c from above at each level i. With R = the unit
// upper-triangular factor (mat = R^T diag(d) R) and z = R^{-T} linvec,
//     N_i      = sum_{k<=i} z_k^2 / d_k,
//     V_i[j]   = sum_{k<=i} R_kj z_k           (j > i),
// the maximum of linvec[:i+1] . c[:i+1] over the ellipsoid slice
// {sum_{l<=i} (U c)_l^2 <= rem} with c[i+1:] fixed is
//     sqrt(N_i * rem) - V_i . c[i+1:].
static int fpk_factor_exact(const int64_t *mat, int dim, double *U,
                            const int64_t *linvec, double *m0N, double *m0V)
{
    size_t n2 = (size_t)dim * dim;
    mpz_t *A = malloc(n2 * sizeof(mpz_t));
    if (!A) return -7;
    for (size_t k = 0; k < n2; ++k) mpz_init_set_si(A[k], mat[k]);
    mpz_t prev, t;
    mpq_t r;
    mpz_init_set_ui(prev, 1);
    mpz_init(t);
    mpq_init(r);
    int status = 0;

    for (int k = 0; k < dim && !status; ++k) {
        mpz_t *pk = &A[k * dim + k];
        if (mpz_sgn(*pk) <= 0) { status = -9; break; }
        // d_k = p_k / p_{k-1}
        mpq_set_num(r, *pk);
        mpq_set_den(r, prev);
        mpq_canonicalize(r);
        double ukk = sqrt(mpq_get_d(r));
        U[k * dim + k] = ukk;
        for (int j = 0; j < k; ++j) U[k * dim + j] = 0.0;
        for (int j = k + 1; j < dim; ++j) {
            // U_kj = U_kk * A_kj / p_k
            mpq_set_num(r, A[k * dim + j]);
            mpq_set_den(r, *pk);
            mpq_canonicalize(r);
            U[k * dim + j] = ukk * mpq_get_d(r);
        }
        // eliminate: A_ij <- (p_k A_ij - A_ik A_kj) / p_{k-1} (exact)
        for (int i = k + 1; i < dim; ++i)
            for (int j = i; j < dim; ++j) {
                mpz_mul(t, *pk, A[i * dim + j]);
                mpz_submul(t, A[i * dim + k], A[k * dim + j]);
                mpz_divexact(A[i * dim + j], t, prev);
                if (j != i) mpz_set(A[j * dim + i], A[i * dim + j]);
            }
        mpz_set(prev, *pk);
    }

    // M0 bound data. Row k of A now holds A^(k)_kj (j >= k), so
    // R_kj = A_kj / A_kk and d_k = A_kk / A_{k-1,k-1}.
    if (!status && linvec) {
        mpq_t *z = malloc(dim * sizeof(mpq_t));
        mpq_t *V = malloc((size_t)dim * sizeof(mpq_t));
        if (!z || !V) {
            free(z); free(V);
            status = -7;
        } else {
            mpq_t R, acc, N;
            mpq_inits(R, acc, N, NULL);
            for (int k = 0; k < dim; ++k) { mpq_init(z[k]); mpq_init(V[k]); }
            for (int k = 0; k < dim; ++k) {
                // z_k = l_k - sum_{m<k} R_mk z_m
                mpq_set_si(z[k], linvec[k], 1);
                for (int m = 0; m < k; ++m) {
                    mpq_set_num(R, A[m * dim + k]);
                    mpq_set_den(R, A[m * dim + m]);
                    mpq_canonicalize(R);
                    mpq_mul(acc, R, z[m]);
                    mpq_sub(z[k], z[k], acc);
                }
            }
            // prefix sums over k: N += z_k^2 / d_k, V[j] += R_kj z_k
            mpz_set_ui(prev, 1);
            for (int i = 0; i < dim; ++i) {
                mpq_mul(acc, z[i], z[i]);
                mpq_set_num(R, prev);                   // 1/d_i = p_{i-1}/p_i
                mpq_set_den(R, A[i * dim + i]);
                mpq_canonicalize(R);
                mpq_mul(acc, acc, R);
                mpq_add(N, N, acc);
                m0N[i] = mpq_get_d(N);
                for (int j = i + 1; j < dim; ++j) {
                    mpq_set_num(R, A[i * dim + j]);
                    mpq_set_den(R, A[i * dim + i]);
                    mpq_canonicalize(R);
                    mpq_mul(acc, R, z[i]);
                    mpq_add(V[j], V[j], acc);
                    m0V[i * dim + j] = mpq_get_d(V[j]);
                }
                mpz_set(prev, A[i * dim + i]);
            }
            for (int k = 0; k < dim; ++k) { mpq_clear(z[k]); mpq_clear(V[k]); }
            mpq_clears(R, acc, N, NULL);
            free(z);
            free(V);
        }
    }

    for (size_t k = 0; k < n2; ++k) mpz_clear(A[k]);
    free(A);
    mpz_clear(prev);
    mpz_clear(t);
    mpq_clear(r);
    return status;
}

#ifndef FPK_STATS   /* (statistics builds use the general loop only) */
// Fast path (the common case: every H row fits int64): the shared search of
// fpk_common.h, in float when qmax < 2^22 (exact; see fpk_search_impl.h) --
// same visiting order and output as the general loop in fpk_enumerate, ~25%
// faster with float plus residue-mode gcds. Returns a kernel status, or 1 if
// a leaf's exact q needs GMP (the caller then reruns the general loop).
typedef struct { fpk_output *out; long max_N_out; int rc, need_gmp; } fpk_fast_ctx;

static int fpk_fast_emit(void *vc, int kind, const int32_t *c, int n, int64_t q)
{
    fpk_fast_ctx *X = (fpk_fast_ctx *)vc;
    if (kind == 2) { X->need_gmp = 1; return 1; }
    int rc = fpk_push(X->out, c, n, q, X->max_N_out);
    if (rc) { X->rc = rc; return 1; }
    return 0;
}

static int fpk_enumerate_fast(const fpk_problem *P, fpk_output *out, const double *U,
                              const double *Uinv, const double *m0N, const double *m0V,
                              const double *m0Va, int m0_level, const int64_t *Hs,
                              const int *level_start, const int *order, int32_t *clist)
{
    const int dim = P->dim;
    fpk_fast_ctx X = {out, P->max_N_out, 0, 0};
    fpk_counts cnt = {0, 0, 0};
    int st = 0;
    if (P->qmax < (1LL << 22)) {
        size_t nf = 3 * (size_t)dim * dim + 2 * (size_t)dim;
        float *f = malloc(nf * sizeof(float));
        if (!f) return -7;
        float *Uf = f, *Vf = f + dim * dim, *Vaf = f + 2 * dim * dim;
        float *Uinvf = f + 3 * dim * dim, *Nf = Uinvf + dim;
        for (int k = 0; k < dim * dim; ++k) {
            Uf[k] = (float)U[k];
            Vf[k] = m0V ? (float)m0V[k] : 0.0f;
            Vaf[k] = m0Va ? (float)m0Va[k] : 0.0f;
        }
        for (int i = 0; i < dim; ++i) {
            Uinvf[i] = 1.0f / Uf[i * dim + i];
            Nf[i] = m0N ? (float)m0N[i] : 0.0f;
        }
        fpk_prep_f S = {dim, P->strict, P->nrows > 0, m0_level, P->Q, P->qmax, P->linmin,
                        (float)P->qmax, 0.0f, 0.0f, 0.0f, Uf, Uinvf, Nf, Vf, Vaf,
                        P->linvec, P->mat, Hs, level_start, order, NULL};
        fpk_search_consts_f(dim, P->qmax, P->eps, &S.slack, &S.Kerr, &S.max_err);
        st = fpk_search_f(&S, NULL, 0, 0, clist, fpk_fast_emit, &X, &cnt);
        free(f);
        if (st == -11) {                     // float too narrow (heavy cancellation)
            out->n = 0;
            X.rc = X.need_gmp = 0;
            cnt.n_nodes = cnt.n_cand = cnt.n_leaf = 0;
        }
    }
    if (P->qmax >= (1LL << 22) || st == -11) {
        fpk_prep S = {dim, P->strict, P->nrows > 0, m0_level, P->Q, P->qmax, P->linmin,
                      (double)P->qmax, 0.0, 0.0, 0.0, U, Uinv, m0N, m0V, m0Va,
                      P->linvec, P->mat, Hs, level_start, order, NULL};
        fpk_search_consts(dim, P->qmax, P->eps, &S.slack, &S.Kerr, &S.max_err);
        st = fpk_search(&S, NULL, 0, 0, clist, fpk_fast_emit, &X, &cnt);
    }
    out->n_nodes = cnt.n_nodes;
    out->n_cand  = cnt.n_cand;
    out->n_leaf  = cnt.n_leaf;
    if (X.need_gmp) return 1;
    if (st == -2) return X.rc ? X.rc : -2;
    return st;
}
#endif

int fpk_enumerate(const fpk_problem *P, fpk_output *out)
{
    const int dim = P->dim;
    if (dim < 1 || dim > FPK_MAX_DIM) return -6;

    const int64_t *linvec = P->linvec;
    const int      use_gcd = P->nrows > 0;
    const double   qmax_d = (double)P->qmax;
    // pruning slack on q (on top of the rigorous error bound `err`)
    const double   slack  = fmax(P->eps, 1e-9 * (qmax_d + 1.0));
    // error-bound constant (see "Error bound"): 2x safety
    const double   Kerr   = 2.0 * (dim + 8) * 0x1p-53;

    int status = 0;
    out->n_nodes = out->n_cand = out->n_leaf = 0;
    if (P->qmax < 0) return 0;

    double *U = malloc((size_t)dim * dim * sizeof(double));
    double *m0N = NULL, *m0V = NULL, *m0Va = NULL;
    if (linvec) {
        m0N  = malloc((size_t)dim * sizeof(double));
        m0V  = calloc((size_t)dim * dim, sizeof(double));
        m0Va = calloc((size_t)dim * dim, sizeof(double));
    }
    if (!U || (linvec && (!m0N || !m0V || !m0Va))) {
        free(U); free(m0N); free(m0V); free(m0Va);
        return -7;
    }
    // exact factorization: int128 fast path, GMP if it cannot be used
    status = fpk_factor_fast(P->mat, dim, U, linvec, m0N, m0V, m0Va);
    if (status == -1) {
        status = fpk_factor_exact(P->mat, dim, U, linvec, m0N, m0V);
        if (!status && linvec)          // exact V, rounded: |V| bounds its error
            for (int k = 0; k < dim * dim; ++k) m0Va[k] = fabs(m0V[k]);
    }
    if (status) { free(U); free(m0N); free(m0V); free(m0Va); return status; }

    // per-level state (indexed by the component being set)
    int32_t  c[FPK_MAX_DIM];
    int64_t  cur[FPK_MAX_DIM], hi[FPK_MAX_DIM];
    double   rem[FPK_MAX_DIM], off[FPK_MAX_DIM], Uinv[FPK_MAX_DIM];
    double   err[FPK_MAX_DIM];               // error bound on rem[i]
    double   aoff[FPK_MAX_DIM];              // sum_{j>i} |U_ij c_j|
    fpk_i128 m0p[FPK_MAX_DIM];               // sum_{j>i} linvec[j] c[j]
    fpk_gval g[FPK_MAX_DIM + 1];             // gcd of rows determined above i
    int      lmode[FPK_MAX_DIM];             // 1: iterate the sparse list
    int      lcnt[FPK_MAX_DIM], lidx[FPK_MAX_DIM];
    int32_t *clist = NULL;                   // (dim, FPK_SPARSE_CAP) lists

    for (int i = 0; i < dim; ++i) {
        Uinv[i] = 1.0 / U[i * dim + i];
        c[i] = 0;
    }

    // M0 is determined once c[m0_level:] is set (linvec[:m0_level] == 0)
    int m0_level = -1;
    if (linvec) {
        for (int j = 0; j < dim; ++j)
            if (linvec[j] != 0) { m0_level = j; break; }
        if (m0_level < 0 && 0 < P->linmin) return 0;   // M0 == 0 < linmin
    }

    // GCD rows, grouped by the level at which they become determined (their
    // first nonzero column). Rows whose entries all fit int64 are "small"
    // (value in int128: |H| < 2^63, |c| < 2^31, dim <= 256 => |.| < 2^102);
    // others are evaluated with GMP.
    int  nrows = use_gcd ? P->nrows : 0;
    int *row_level = NULL, *level_start = NULL, *order = NULL, *is_small = NULL;
    int64_t *Hs = NULL;
    fpk_i128 *pre_s = NULL;          // per row: sum_{j>level} H[r][j] c[j]
    mpz_t *pre_b = NULL, *Hb = NULL;
    mpz_t  gbig[FPK_MAX_DIM + 1];
    mpz_t  tmp, tmp2;
    int    have_big = 0, n_mpz_levels = 0;
    mpz_init(tmp);
    mpz_init(tmp2);

    if (use_gcd) {
        clist       = malloc((size_t)dim * FPK_SPARSE_CAP * sizeof(int32_t));
        row_level   = malloc(nrows * sizeof(int));
        level_start = malloc((dim + 1) * sizeof(int));
        order       = malloc(nrows * sizeof(int));
        is_small    = malloc(nrows * sizeof(int));
        Hs          = malloc((size_t)nrows * dim * sizeof(int64_t));
        pre_s       = malloc(nrows * sizeof(fpk_i128));
        if (!clist || !row_level || !level_start || !order || !is_small || !Hs || !pre_s) {
            status = -7;
            goto end;
        }
        int counts[FPK_MAX_DIM + 1] = {0};
        for (int r = 0; r < nrows; ++r) {
            int lvl = -1;
            is_small[r] = 1;
            for (int j = 0; j < dim; ++j) {
                if (P->H64) {
                    int64_t x = P->H64[r * dim + j];
                    if (lvl < 0 && x != 0) lvl = j;
                    Hs[r * dim + j] = x;
                    continue;
                }
                mpz_t *h = &P->H[r * dim + j];
                if (lvl < 0 && mpz_sgn(*h) != 0) lvl = j;
                if (mpz_fits_slong_p(*h) && sizeof(long) >= 8) {
                    Hs[r * dim + j] = mpz_get_si(*h);
                } else {
                    is_small[r] = 0;
                    Hs[r * dim + j] = 0;
                }
            }
            row_level[r] = lvl;   // -1: zero row, never contributes
            if (lvl >= 0) counts[lvl]++;
            if (!is_small[r]) have_big = 1;
        }
        // counting sort rows by level
        level_start[0] = 0;
        for (int l = 0; l < dim; ++l) level_start[l + 1] = level_start[l] + counts[l];
        int fill[FPK_MAX_DIM + 1];
        memcpy(fill, level_start, (dim + 1) * sizeof(int));
        for (int r = 0; r < nrows; ++r)
            if (row_level[r] >= 0) order[fill[row_level[r]]++] = r;

        if (have_big) {
            Hb    = malloc((size_t)nrows * dim * sizeof(mpz_t));
            pre_b = malloc(nrows * sizeof(mpz_t));
            if (!Hb || !pre_b) { free(Hb); free(pre_b); Hb = pre_b = NULL; status = -7; goto end; }
            for (int r = 0; r < nrows; ++r) {
                mpz_init(pre_b[r]);
                for (int j = 0; j < dim; ++j) mpz_init_set(Hb[r * dim + j], P->H[r * dim + j]);
            }
        }
        for (int l = 0; l <= dim; ++l) mpz_init(gbig[l]);
        n_mpz_levels = dim + 1;
    }

#ifndef FPK_STATS
    if (!have_big && dim <= FPK_SEARCH_MAXD) {
        status = fpk_enumerate_fast(P, out, U, Uinv, m0N, m0V, m0Va, m0_level,
                                    Hs, level_start, order, clist);
        if (status != 1) goto end;
        // an exact q needed GMP: start over with the general loop
        out->n = 0;
        out->n_nodes = out->n_cand = out->n_leaf = 0;
        status = 0;
    }
#endif

    // Prepare level i (c[i+1:] is set): center offset, per-row prefixes and
    // the candidate range. g[i+1], rem[i], m0p[i] are set by the caller.
    #define FPK_ENTER_LEVEL(i)                                                  \
    do {                                                                        \
        double o_ = 0.0, a_ = 0.0;                                              \
        for (int j_ = (i) + 1; j_ < dim; ++j_) {                                \
            double t_ = U[(i) * dim + j_] * c[j_];                              \
            o_ += t_;                                                           \
            a_ += fabs(t_);                                                     \
        }                                                                       \
        off[(i)] = o_;                                                          \
        aoff[(i)] = a_;                                                         \
        if (use_gcd) {                                                          \
            for (int k_ = level_start[(i)]; k_ < level_start[(i) + 1]; ++k_) {  \
                int r_ = order[k_];                                             \
                if (is_small[r_]) {                                             \
                    fpk_i128 s_ = 0;                                            \
                    for (int j_ = (i) + 1; j_ < dim; ++j_)                      \
                        s_ += (fpk_i128)Hs[r_ * dim + j_] * c[j_];              \
                    pre_s[r_] = s_;                                             \
                } else {                                                        \
                    mpz_set_ui(pre_b[r_], 0);                                   \
                    for (int j_ = (i) + 1; j_ < dim; ++j_)                      \
                        if (c[j_]) fpk_mpz_addmul_si(pre_b[r_], Hb[r_ * dim + j_], c[j_]); \
                }                                                               \
            }                                                                   \
        }                                                                       \
        /* true |t_i| <= sqrt(rem_true); float t_i is within Kerr*a_i, and   \
           on candidates a_i <= |t_i| + 2*a_ */                                 \
        double R_ = sqrt(fmax(rem[(i)] + slack + err[(i)], 0.0));               \
        R_ += Kerr * (R_ + 2.0 * a_) + 1e-300;                                   \
        double R0_ = R_;                                                        \
        /* GCD narrowing: the gcd below this node is at most G = g[i+1], so  \
           a candidate survives only if Q*G >= q_lb, i.e.                     \
           t_i^2 <= Q*G - (qmax - rem) + (error margins). */                   \
        int empty_ = 0;                                                         \
        if (use_gcd && !g[(i) + 1].big && g[(i) + 1].s != 0                     \
                && (g[(i) + 1].s >> 64) == 0) {                                 \
            double am_ = R_ + 2.0 * a_;                                         \
            double Rg2_ = (double)P->Q * (double)(uint64_t)g[(i) + 1].s         \
                        - (qmax_d - rem[(i)]) + slack + err[(i)]                \
                        + Kerr * (am_ * am_ + rem[(i)]);                        \
            if (Rg2_ < 0.0) {                                                   \
                empty_ = 1;                                                     \
            } else {                                                            \
                double Rg_ = sqrt(Rg2_);                                        \
                Rg_ += Kerr * (Rg_ + 2.0 * a_) + 1e-300;                        \
                if (Rg_ < R_) R_ = Rg_;                                         \
            }                                                                   \
        }                                                                       \
        /* M0 bound: even the best completion of c[:i+1] must reach linmin */ \
        if (!empty_ && linvec && (i) >= m0_level && m0_level >= 0) {            \
            double vc_ = 0.0, va_ = 0.0;                                        \
            for (int j_ = (i) + 1; j_ < dim; ++j_) {                            \
                vc_ += m0V[(i) * dim + j_] * c[j_];                             \
                va_ += m0Va[(i) * dim + j_] * fabs((double)c[j_]);              \
            }                                                                   \
            double sq_ = sqrt(fmax(m0N[(i)] * (rem[(i)] + slack + err[(i)]), 0.0)); \
            double ub_ = (double)m0p[(i)] - vc_ + sq_                           \
                       + Kerr * (va_ + sq_ + fabs((double)m0p[(i)])) + 1e-9;    \
            if (ub_ < (double)P->linmin) empty_ = 1;                            \
        }                                                                       \
        int64_t lo_ = 1, hi_ = 0;                                               \
        if (!empty_ && fpk_bounds(R_, off[(i)], Uinv[(i)], &lo_, &hi_)) {       \
            status = -8;                                                        \
            goto end;                                                           \
        }                                                                       \
        /* M0 is determined at this level: m0p + l_i v >= linmin is a        \
           linear bound on v (exact integer arithmetic) */                     \
        if (!empty_ && (i) == m0_level) {                                       \
            fpk_i128 need_ = (fpk_i128)P->linmin - m0p[(i)];                    \
            fpk_i128 l_ = linvec[(i)];                                          \
            if (l_ > 0) {                                                       \
                fpk_i128 b_ = need_ >= 0 ? (need_ + l_ - 1) / l_ : -((-need_) / l_); \
                if (b_ > lo_) lo_ = b_ > hi_ ? hi_ + 1 : (int64_t)b_;           \
            } else {                                                            \
                fpk_i128 nl_ = -l_;             /* v <= floor(-need/nl) */      \
                fpk_i128 b_ = need_ <= 0 ? (-need_) / nl_ : -((need_ + nl_ - 1) / nl_); \
                if (b_ < hi_) hi_ = b_ < lo_ ? lo_ - 1 : (int64_t)b_;           \
            }                                                                   \
        }                                                                       \
        lmode[(i)] = 0;                                                         \
        if (!empty_ && use_gcd && hi_ - lo_ + 1 >= FPK_SPARSE_MIN_W             \
                && level_start[(i) + 1] - level_start[(i)] == 1                 \
                && is_small[order[level_start[(i)]]] && !g[(i) + 1].big         \
                && g[(i) + 1].s != 0 && g[(i) + 1].s <= FPK_SPARSE_MAX_G) {     \
            int r_ = order[level_start[(i)]];                                   \
            double am_ = R0_ + 2.0 * a_;                                        \
            double mg_ = slack + err[(i)] + Kerr * (am_ * am_ + rem[(i)]);      \
            double base_ = qmax_d - rem[(i)];                                   \
            int n_ = fpk_sparse_candidates(                                     \
                (uint64_t)g[(i) + 1].s, Hs[r_ * dim + (i)], pre_s[r_], lo_, hi_, \
                off[(i)], Uinv[(i)], base_, mg_, P->Q,                          \
                fpk_need(base_ - mg_, P->Q, P->strict), Kerr, a_,               \
                &clist[(size_t)(i) * FPK_SPARSE_CAP]);                          \
            if (n_ >= 0) { lmode[(i)] = 1; lcnt[(i)] = n_; lidx[(i)] = 0; }     \
        }                                                                       \
        cur[(i)] = lo_;                                                         \
        hi[(i)]  = hi_;                                                         \
        out->n_nodes++;                                                         \
    } while (0)

    // top level
    int i = dim - 1;
    rem[i] = qmax_d;
    err[i] = Kerr * qmax_d;
    m0p[i] = 0;
    g[dim].big = 0;
    g[dim].s = 0;
    FPK_ENTER_LEVEL(i);

    while (i < dim) {
        int32_t v;
        if (lmode[i]) {
            if (lidx[i] == lcnt[i]) { c[i] = 0; i++; continue; }   // backtrack
            v = clist[(size_t)i * FPK_SPARSE_CAP + lidx[i]++];
        } else {
            if (cur[i] > hi[i]) { c[i] = 0; i++; continue; }       // backtrack
            v = (int32_t)cur[i]++;
        }
        c[i] = v;
        out->n_cand++;

        // ellipsoid
        double x  = U[i * dim + i] * v + off[i];
        double nr = rem[i] - x * x;
        double ai = fabs(U[i * dim + i] * v) + aoff[i];
        double en = err[i] + Kerr * (ai * ai + rem[i]);
        if (nr < -(slack + en)) { FPK_STAT(FPK_REJ_ELLIPSOID, i); continue; }

        // M0 cut (exact)
        fpk_i128 m0 = m0p[i] + (linvec ? (fpk_i128)linvec[i] * v : 0);
        if (i == m0_level && m0 < P->linmin) { FPK_STAT(FPK_REJ_M0, i); continue; }

        // GCD cut: fold in the rows determined at this level
        if (use_gcd) {
            fpk_gval gn = g[i + 1];
            if (gn.big) mpz_set(gbig[i], gbig[i + 1]);
            double q_lb = (qmax_d - nr) - (slack + en);   // lower bound on q(c)
            uint64_t need = fpk_need(q_lb, P->Q, P->strict);
            int pruned = 0;
            for (int k = level_start[i]; k < level_start[i + 1]; ++k) {
                // the gcd can only shrink: test the current bound first
                if (fpk_gcd_fails(&gn, P->Q, q_lb, P->strict)) { pruned = 1; break; }
                int r = order[k];
                if (is_small[r]) {
                    fpk_u128 a = fpk_abs128(pre_s[r] + (fpk_i128)Hs[r * dim + i] * v);
                    if (!gn.big && (gn.s >> 64) == 0 && (a >> 64) == 0 && (gn.s | a) != 0) {
                        // fast path: most candidates fail here, cheaply
                        uint64_t gg = fpk_gcd64_ge((uint64_t)gn.s, (uint64_t)a, need);
                        if (!gg) { pruned = 1; break; }
                        gn.s = gg;
                    } else if (!gn.big) {
                        gn.s = fpk_gcd128(gn.s, a);
                    } else {
                        fpk_mpz_set_u128(tmp, a);
                        mpz_gcd(gbig[i], gbig[i], tmp);
                        if (fpk_mpz_fits_u128(gbig[i])) { gn.big = 0; gn.s = fpk_mpz_get_u128(gbig[i]); }
                    }
                } else {
                    mpz_set(tmp, pre_b[r]);
                    fpk_mpz_addmul_si(tmp, Hb[r * dim + i], v);
                    if (!gn.big) {
                        fpk_mpz_set_u128(tmp2, gn.s);
                        mpz_gcd(gbig[i], tmp2, tmp);    // gcd(0, x) = |x|
                    } else {
                        mpz_gcd(gbig[i], gbig[i], tmp);
                    }
                    gn.big = !fpk_mpz_fits_u128(gbig[i]);
                    if (!gn.big) gn.s = fpk_mpz_get_u128(gbig[i]);
                }
            }
            if (pruned || fpk_gcd_fails(&gn, P->Q, q_lb, P->strict)) {
                FPK_STAT(FPK_REJ_GCD, i);
                continue;
            }
            g[i] = gn;
        }

        if (i > 0) {
            // descend
            m0p[i - 1] = m0;
            rem[i - 1] = nr;
            err[i - 1] = en;
            i--;
            FPK_ENTER_LEVEL(i);
            continue;
        }

        // leaf: exact checks
        // --------------------
        out->n_leaf++;
        fpk_i128 q;
        if (fpk_exact_q(P->mat, c, dim, &q, tmp, tmp2)) continue;
        if (q > P->qmax) continue;
        if (use_gcd && !g[0].big && g[0].s != 0) {
            // g < 2^64 here would be needed for Q*g to be <= qmax < 2^63
            if ((g[0].s >> 64) == 0) {
                fpk_i128 Qg = (fpk_i128)P->Q * (fpk_i128)(uint64_t)g[0].s;
                if (P->strict ? (Qg <= q) : (Qg < q)) continue;
            }
        }
        int rc = fpk_push(out, c, dim, (int64_t)q, P->max_N_out);
        if (rc) { status = rc; goto end; }
    }

end:
    #undef FPK_ENTER_LEVEL
    if (use_gcd) {
        if (Hb) {
            for (int r = 0; r < nrows; ++r) {
                mpz_clear(pre_b[r]);
                for (int j = 0; j < dim; ++j) mpz_clear(Hb[r * dim + j]);
            }
        }
        for (int l = 0; l < n_mpz_levels; ++l) mpz_clear(gbig[l]);
    }
    free(Hb); free(pre_b);
    free(clist);
    free(row_level); free(level_start); free(order); free(is_small);
    free(Hs); free(pre_s);
    free(U);
    free(m0N);
    free(m0V);
    free(m0Va);
    mpz_clear(tmp);
    mpz_clear(tmp2);
    return status;
}

#endif // FP_KERNEL_IMPLEMENTATION

#endif // FP_KERNEL_H
