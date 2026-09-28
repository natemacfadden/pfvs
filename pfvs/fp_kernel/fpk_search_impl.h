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
//
// Float-dependent part of fpk_common.h, generic in the floating-point type
// FPK_R; included by fpk_common.h once per type (no include guard). Names go
// through FPK_N (double: as is; float: suffix _f).
//
// Every floating-point test is widened by rigorous bounds written in terms of
// the type's unit roundoff (see "Error bound" in fp_kernel.h):
//     FPK_U     unit roundoff (2^-53, 2^-24): Kerr = 2 (dim + 8) FPK_U
//     FPK_G     relative guard on the few rounded divisions/products whose
//               error is not in Kerr (1e-9 for double; 64 FPK_U for float)
//     FPK_TINY  keeps widened radii strictly positive
// and slack >= FPK_G (qmax + 1) absorbs absolute rounding of O(qmax) terms.
// The float instantiation additionally needs qmax < 2^22. Then a product
// Q g < 2^24 is exact in float, and a larger one only ever gives a GCD
// radius beyond the ellipsoid's (sqrt(Q g - qmax) > sqrt(qmax)), so its
// rounding cannot tighten anything; likewise the GCD tests compare exact
// integers (as double does below 2^53). Leaf decisions are exact integers
// either way, so the type only affects how much is pruned, never the output.
// Callers get Kerr and slack from FPK_N(fpk_search_consts).

FPK_HD static inline void FPK_N(fpk_search_consts)(int dim, int64_t qmax, double eps,
                                                   FPK_R *slack, FPK_R *Kerr, FPK_R *max_err)
{
    FPK_R qd = (FPK_R)qmax;
    *slack = FPK_FMAX((FPK_R)eps, FPK_G * (qd + FPK_L(1.0)));
    *Kerr  = FPK_L(2.0) * (FPK_R)(dim + 8) * FPK_U;
    // error budget: a heavily cancelling (ill-conditioned) problem can have
    // a rounding bound so large that nothing is pruned; past this the search
    // stops with -11 and the caller uses a wider type (double does not stop)
    *max_err = sizeof(FPK_R) < sizeof(double) ? FPK_L(1.0) + FPK_L(0.05) * qd : (FPK_R)INFINITY;
}

// Smallest gcd that can pass the GCD cut when q(c) >= q_lb, rounded down so
// it never demands more than the exact cut: the cut fails iff g < need.
FPK_HD static inline uint64_t FPK_N(fpk_need)(FPK_R q_lb, int64_t Q, int strict)
{
    if (!(q_lb > 0)) return 0;
    FPK_R t = q_lb / (FPK_R)Q;
    t -= FPK_G * (FPK_L(1.0) + t);
    if (t < 0) t = 0;
    if (t >= FPK_L(1.8e19)) return UINT64_MAX;
    return strict ? (uint64_t)FPK_FLOOR(t) + 1 : (uint64_t)FPK_CEIL(t);
}

// The GCD-cut test. Returns 1 if a vector whose q is (at least) q_lb must
// fail Q*g >= q (or Q*g > q if strict), i.e. if the branch can be pruned.
// g == 0 never prunes. A big g exceeds any q we can accept.
FPK_HD static inline int FPK_N(fpk_gcd_fails)(const fpk_gval *g, int64_t Q, FPK_R q_lb, int strict)
{
    if (g->big || g->s == 0) return 0;
    if ((g->s >> 64) != 0)   return 0;
    FPK_R Qg = (FPK_R)Q * (FPK_R)(uint64_t)g->s;
    return strict ? (Qg <= q_lb) : (Qg < q_lb);
}

// Set the candidate range of c[i]: all integers v with |U_ii v + off| <= R,
// widened outward against rounding.
// Returns -8 if the range does not fit int32.
FPK_HD static inline int FPK_N(fpk_bounds)(FPK_R R, FPK_R off, FPK_R Uii_inv,
                             int64_t *lo, int64_t *hi)
{
    FPK_R a = (-R - off) * Uii_inv;
    FPK_R b = ( R - off) * Uii_inv;
    a = FPK_CEIL(a - FPK_G * (FPK_L(1.0) + FPK_FABS(a)));
    b = FPK_FLOOR(b + FPK_G * (FPK_L(1.0) + FPK_FABS(b)));
    if (!(a >= -FPK_I32MAX_R && b <= FPK_I32MAX_R)) return -8;   /* also NaN */
    *lo = (int64_t)a;
    *hi = (int64_t)b;
    return 0;
}

FPK_HD static inline int FPK_N(fpk_sparse_candidates)(
    uint64_t G, int64_t h, fpk_i128 pre, int64_t lo, int64_t hi,
    FPK_R off, FPK_R Uii_inv, FPK_R base, FPK_R margin, int64_t Q,
    uint64_t need_min, FPK_R Kerr, FPK_R aoff, int32_t *list)
{
    if (G == 0 || G > FPK_SPARSE_MAX_G || h == 0) return -1;

    // factor G (< 2^20) by trial division, then list divisors >= need_min
    uint32_t pr[20], ex[20];
    int np = 0;
    uint32_t n = (uint32_t)G;
    for (uint32_t f = 2; f * f <= n; f += (f == 2 ? 1 : 2)) {
        if (n % f) continue;
        pr[np] = f; ex[np] = 0;
        while (n % f == 0) { n /= f; ex[np]++; }
        np++;
    }
    if (n > 1) { pr[np] = n; ex[np] = 1; np++; }
    uint32_t divs[512];
    int nd = 1;
    divs[0] = 1;
    for (int k = 0; k < np; ++k) {
        int cur_nd = nd;
        uint32_t pw = 1;
        for (uint32_t e = 1; e <= ex[k]; ++e) {
            pw *= pr[k];
            for (int j = 0; j < cur_nd; ++j) {
                if (nd == 512) return -1;
                divs[nd++] = divs[j] * pw;
            }
        }
    }

    int64_t W = hi - lo + 1;
    int cnt = 0;
    FPK_R est = FPK_L(0.0);
    for (int k = 0; k < nd; ++k) {
        uint64_t d = divs[k];
        if (d < need_min) continue;
        FPK_R R2 = (FPK_R)Q * (FPK_R)d - base + margin;
        if (R2 < FPK_L(0.0)) continue;
        FPK_R R = FPK_SQRT(R2);
        R += Kerr * (R + FPK_L(2.0) * aoff) + FPK_TINY;
        int64_t dlo, dhi;
        if (FPK_N(fpk_bounds)(R, off, Uii_inv, &dlo, &dhi)) return -1;
        if (dlo < lo) dlo = lo;
        if (dhi > hi) dhi = hi;
        if (dlo > dhi) continue;

        // solve h v == -pre (mod d)
        int64_t dd = (int64_t)d;
        int64_t r0 = (int64_t)(((-pre) % dd + dd) % dd);
        int64_t hm = ((h % dd) + dd) % dd;
        int64_t g1 = (int64_t)fpk_gcd64((uint64_t)hm, (uint64_t)dd);
        if (r0 % g1) continue;
        int64_t m = dd / g1;
        int64_t v0 = 0;
        if (m > 1)
            v0 = (int64_t)(((fpk_i128)((r0 / g1) % m) * fpk_modinv((hm / g1) % m, m)) % m);
        est += (FPK_R)(dhi - dlo) / (FPK_R)m + FPK_L(1.0);
        if (est > FPK_L(0.5) * (FPK_R)W) return -1;       // not worth it

        // first v >= dlo with v == v0 (mod m)
        int64_t v = dlo + (((v0 - dlo) % m) + m) % m;
        for (; v <= dhi; v += m) {
            fpk_i128 x = pre + (fpk_i128)h * v;
            // keep v only if d(v) == d exactly (reduce |x| mod G first)
            uint64_t xr = (uint64_t)(fpk_abs128(x) % (fpk_u128)G);
            if (fpk_gcd64(G, xr) != d) continue;
            if (cnt == FPK_SPARSE_CAP) return -1;
            list[cnt++] = (int32_t)v;
        }
    }
    fpk_isort_i32(list, cnt);
    return cnt;
}

typedef struct {
    int dim, strict, use_gcd, m0_level;
    int64_t Q, qmax, linmin;
    FPK_R qmax_d, slack, Kerr, max_err;       // from FPK_N(fpk_search_consts)
    const FPK_R  *U, *Uinv;                   // U: (dim, dim)
    const FPK_R  *m0N, *m0V, *m0Va;           // M0 bound (if linvec)
    const int64_t *linvec, *mat;               // mat: exact, for the leaf
    const int64_t *Hs;                         // (nrows, dim), all rows int64
    const int     *level_start, *order;        // rows grouped by first nonzero
} FPK_N(fpk_prep);


// Status: 0 done; -2 emit asked to stop; -8 a coordinate range exceeds int32;
// -11 the error bound exceeded max_err (use a wider floating-point type).
FPK_HD static inline int FPK_N(fpk_search)(const FPK_N(fpk_prep) *S, const int32_t *prefix,
                                    int n_prefix, int stop_depth, int32_t *clist,
                                    fpk_emit_fn emit, void *ctx, fpk_counts *cnt)
{
    enum { MD = FPK_SEARCH_MAXD };
    const int dim = S->dim;
    const FPK_R slack = S->slack, Kerr = S->Kerr, qmax_d = S->qmax_d;
    const FPK_R *U = S->U;
    const int64_t *linvec = S->linvec;
    int32_t  c[MD];
    int64_t  cur[MD], hi[MD];
    FPK_R   rem[MD], off[MD], err[MD], aoff[MD];
    fpk_i128 m0p[MD];
    fpk_u128 g[MD + 1];                         // gcd of the rows determined above
    int      lmode[MD], lcnt[MD], lidx[MD];
    fpk_i128 pre_s[MD + 1];
    // Residue mode (dense levels whose parent gcd G = g[i+1] is in [1, 2^63)):
    // only gcd(g, x) matters for a row value x = pre + h v, with g | G, and
    // gcd(g, x) = gcd(g, x mod G). As v steps by 1, x mod G steps by h mod G,
    // so it is tracked with one add and compare per candidate instead of an
    // int128 multiply-add, and the gcd runs on values < G.
    uint64_t rres[MD + 1], rstep[MD + 1], rcur[MD + 1], gpar[MD];
    int      lres[MD];
    for (int j = 0; j < dim; ++j) c[j] = 0;

    // prepare level i (c[i+1:] set); returns -8 on an int32 range overflow
    #define FPK_ENTER(i)                                                        \
    do {                                                                        \
        FPK_R o_ = FPK_L(0.0), a_ = FPK_L(0.0);                                              \
        for (int j_ = (i) + 1; j_ < dim; ++j_) {                                \
            FPK_R t_ = U[(i) * dim + j_] * c[j_];                              \
            o_ += t_;                                                           \
            a_ += FPK_FABS(t_);                                                     \
        }                                                                       \
        off[(i)] = o_;                                                          \
        aoff[(i)] = a_;                                                         \
        if (S->use_gcd)                                                         \
            for (int k_ = S->level_start[(i)]; k_ < S->level_start[(i) + 1]; ++k_) { \
                int r_ = S->order[k_];                                          \
                fpk_i128 s_ = 0;                                                \
                for (int j_ = (i) + 1; j_ < dim; ++j_)                          \
                    s_ += (fpk_i128)S->Hs[r_ * dim + j_] * c[j_];               \
                pre_s[r_] = s_;                                                 \
            }                                                                   \
        FPK_R R_ = FPK_SQRT(FPK_FMAX(rem[(i)] + slack + err[(i)], FPK_L(0.0)));               \
        R_ += Kerr * (R_ + FPK_L(2.0) * a_) + FPK_TINY;                                   \
        FPK_R R0_ = R_;                                                        \
        int empty_ = 0;                                                         \
        if (S->use_gcd && g[(i) + 1] != 0 && (g[(i) + 1] >> 64) == 0) {     \
            FPK_R am_ = R_ + FPK_L(2.0) * a_;                                         \
            FPK_R Rg2_ = (FPK_R)S->Q * (FPK_R)(uint64_t)g[(i) + 1]         \
                        - (qmax_d - rem[(i)]) + slack + err[(i)]                \
                        + Kerr * (am_ * am_ + rem[(i)]);                        \
            if (Rg2_ < FPK_L(0.0)) {                                                   \
                empty_ = 1;                                                     \
            } else {                                                            \
                FPK_R Rg_ = FPK_SQRT(Rg2_);                                        \
                Rg_ += Kerr * (Rg_ + FPK_L(2.0) * a_) + FPK_TINY;                        \
                if (Rg_ < R_) R_ = Rg_;                                         \
            }                                                                   \
        }                                                                       \
        if (!empty_ && linvec && (i) >= S->m0_level && S->m0_level >= 0) {      \
            FPK_R vc_ = FPK_L(0.0), va_ = FPK_L(0.0);                                        \
            for (int j_ = (i) + 1; j_ < dim; ++j_) {                            \
                vc_ += S->m0V[(i) * dim + j_] * c[j_];                          \
                va_ += S->m0Va[(i) * dim + j_] * FPK_FABS((FPK_R)c[j_]);           \
            }                                                                   \
            FPK_R sq_ = FPK_SQRT(FPK_FMAX(S->m0N[(i)] * (rem[(i)] + slack + err[(i)]), FPK_L(0.0))); \
            FPK_R ub_ = (FPK_R)m0p[(i)] - vc_ + sq_                           \
                       + Kerr * (va_ + sq_ + FPK_FABS((FPK_R)m0p[(i)])) + FPK_G;    \
            if (ub_ < (FPK_R)S->linmin) empty_ = 1;                            \
        }                                                                       \
        int64_t lo_ = 1, hi_ = 0;                                               \
        if (!empty_ && FPK_N(fpk_bounds)(R_, off[(i)], S->Uinv[(i)], &lo_, &hi_))      \
            return -8;                                                          \
        if (!empty_ && (i) == S->m0_level) {                                    \
            fpk_i128 need_ = (fpk_i128)S->linmin - m0p[(i)];                    \
            fpk_i128 l_ = linvec[(i)];                                          \
            if (l_ > 0) {                                                       \
                fpk_i128 b_ = need_ >= 0 ? (need_ + l_ - 1) / l_ : -((-need_) / l_); \
                if (b_ > lo_) lo_ = b_ > hi_ ? hi_ + 1 : (int64_t)b_;           \
            } else {                                                            \
                fpk_i128 nl_ = -l_;                                             \
                fpk_i128 b_ = need_ <= 0 ? (-need_) / nl_ : -((need_ + nl_ - 1) / nl_); \
                if (b_ < hi_) hi_ = b_ < lo_ ? lo_ - 1 : (int64_t)b_;           \
            }                                                                   \
        }                                                                       \
        if (FPK_SEARCH_SPARSE) lmode[(i)] = 0;                                  \
        if (FPK_SEARCH_SPARSE && !empty_ && S->use_gcd && clist && hi_ - lo_ + 1 >= FPK_SPARSE_MIN_W \
                && S->level_start[(i) + 1] - S->level_start[(i)] == 1           \
                && g[(i) + 1] != 0 && g[(i) + 1] <= FPK_SPARSE_MAX_G) {     \
            int r_ = S->order[S->level_start[(i)]];                             \
            FPK_R am_ = R0_ + FPK_L(2.0) * a_;                                        \
            FPK_R mg_ = slack + err[(i)] + Kerr * (am_ * am_ + rem[(i)]);      \
            FPK_R base_ = qmax_d - rem[(i)];                                   \
            int n_ = FPK_N(fpk_sparse_candidates)(                                     \
                (uint64_t)g[(i) + 1], S->Hs[r_ * dim + (i)], pre_s[r_], lo_, hi_, \
                off[(i)], S->Uinv[(i)], base_, mg_, S->Q,                       \
                FPK_N(fpk_need)(base_ - mg_, S->Q, S->strict), Kerr, a_,               \
                &clist[(size_t)(i) * FPK_SPARSE_CAP]);                          \
            if (n_ >= 0) { lmode[(i)] = 1; lcnt[(i)] = n_; lidx[(i)] = 0; }     \
        }                                                                       \
        cur[(i)] = lo_;                                                         \
        hi[(i)]  = hi_;                                                         \
        /* forced prefix: keep only the given value, if it is a candidate */    \
        if ((i) >= dim - n_prefix) {                                            \
            int32_t pv_ = prefix[(i) - (dim - n_prefix)];                                \
            int ok_ = 0;                                                        \
            if (FPK_SEARCH_SPARSE && lmode[(i)]) {                              \
                for (int t_ = 0; t_ < lcnt[(i)]; ++t_)                          \
                    if (clist[(size_t)(i) * FPK_SPARSE_CAP + t_] == pv_) { ok_ = 1; break; } \
            } else {                                                            \
                ok_ = (lo_ <= pv_ && pv_ <= hi_);                               \
            }                                                                   \
            if (FPK_SEARCH_SPARSE) lmode[(i)] = 0;                              \
            cur[(i)] = pv_;                                                     \
            hi[(i)] = ok_ ? pv_ : (int64_t)pv_ - 1;                             \
        }                                                                       \
        lres[(i)] = 0;                                                          \
        if (S->use_gcd && !(FPK_SEARCH_SPARSE && lmode[(i)]) && hi[(i)] - cur[(i)] + 1 >= FPK_RES_MIN_W \
                && g[(i) + 1] != 0 && (g[(i) + 1] >> 63) == 0) {             \
            const uint64_t G_ = (uint64_t)g[(i) + 1];                         \
            for (int k_ = S->level_start[(i)]; k_ < S->level_start[(i) + 1]; ++k_) { \
                int r_ = S->order[k_];                                          \
                fpk_i128 x_ = (pre_s[r_] + (fpk_i128)S->Hs[r_ * dim + (i)] * cur[(i)]) % (fpk_i128)G_; \
                fpk_i128 h_ = (fpk_i128)S->Hs[r_ * dim + (i)] % (fpk_i128)G_;   \
                rres[r_]  = (uint64_t)(x_ < 0 ? x_ + (fpk_i128)G_ : x_);        \
                rstep[r_] = (uint64_t)(h_ < 0 ? h_ + (fpk_i128)G_ : h_);        \
            }                                                                   \
            gpar[(i)] = G_;                                                     \
            lres[(i)] = 1;                                                      \
        }                                                                       \
        if (cnt) cnt->n_nodes++;                                                \
    } while (0)

    int i = dim - 1;
    rem[i] = qmax_d;
    err[i] = Kerr * qmax_d;
    m0p[i] = 0;
    g[dim] = 0;
    FPK_ENTER(i);

    while (i < dim) {
        int32_t v;
        if (FPK_SEARCH_SPARSE && lmode[i]) {
            if (lidx[i] == lcnt[i]) { c[i] = 0; i++; continue; }
            v = clist[(size_t)i * FPK_SPARSE_CAP + lidx[i]++];
        } else {
            if (cur[i] > hi[i]) { c[i] = 0; i++; continue; }
            v = (int32_t)cur[i]++;
            if (lres[i])                                // residues at v; advance
                for (int k = S->level_start[i]; k < S->level_start[i + 1]; ++k) {
                    int r = S->order[k];
                    rcur[r] = rres[r];
                    uint64_t t = rres[r] + rstep[r];    // < 2 G < 2^64
                    rres[r] = t >= gpar[i] ? t - gpar[i] : t;
                }
        }
        c[i] = v;
        if (cnt) cnt->n_cand++;

        FPK_R x  = U[i * dim + i] * v + off[i];
        FPK_R nr = rem[i] - x * x;
        FPK_R ai = FPK_FABS(U[i * dim + i] * v) + aoff[i];
        FPK_R en = err[i] + Kerr * (ai * ai + rem[i]);
        if (en > S->max_err) return -11;
        if (nr < -(slack + en)) continue;

        fpk_i128 m0 = m0p[i] + (linvec ? (fpk_i128)linvec[i] * v : 0);
        if (i == S->m0_level && m0 < S->linmin) continue;

        if (S->use_gcd) {
            fpk_gval gn;
            gn.big = 0;
            gn.s = g[i + 1];
            FPK_R q_lb = (qmax_d - nr) - (slack + en);
            uint64_t need = FPK_N(fpk_need)(q_lb, S->Q, S->strict);
            int pruned = 0;
            for (int k = S->level_start[i]; k < S->level_start[i + 1]; ++k) {
                if (FPK_N(fpk_gcd_fails)(&gn, S->Q, q_lb, S->strict)) { pruned = 1; break; }
                int r = S->order[k];
                fpk_u128 a = lres[i] ? (fpk_u128)rcur[r]
                                     : fpk_abs128(pre_s[r] + (fpk_i128)S->Hs[r * dim + i] * v);
                if ((gn.s >> 64) == 0 && (a >> 64) == 0 && (gn.s | a) != 0) {
                    uint64_t gg = fpk_gcd64_ge((uint64_t)gn.s, (uint64_t)a, need);
                    if (!gg) { pruned = 1; break; }
                    gn.s = gg;
                } else {
                    gn.s = fpk_gcd128(gn.s, a);
                }
            }
            if (pruned || FPK_N(fpk_gcd_fails)(&gn, S->Q, q_lb, S->strict)) continue;
            g[i] = gn.s;
        }

        if (stop_depth > 0 && i == dim - stop_depth) {      // emit the prefix
            if (emit(ctx, 1, &c[i], stop_depth, 0)) return -2;
            continue;
        }
        if (i > 0) {
            m0p[i - 1] = m0;
            rem[i - 1] = nr;
            err[i - 1] = en;
            i--;
            FPK_ENTER(i);
            continue;
        }

        // leaf: exact checks
        if (cnt) cnt->n_leaf++;
        fpk_i128 q;
        if (fpk_exact_q128(S->mat, c, dim, &q)) {             // needs GMP
            if (emit(ctx, 2, c, dim, 0)) return -2;
            continue;
        }
        if (q > S->qmax) continue;
        if (S->use_gcd && g[0] != 0 && (g[0] >> 64) == 0) {
            fpk_i128 Qg = (fpk_i128)S->Q * (fpk_i128)(uint64_t)g[0];
            if (S->strict ? (Qg <= q) : (Qg < q)) continue;
        }
        if (emit(ctx, 0, c, dim, (int64_t)q)) return -2;
    }
    #undef FPK_ENTER
    return 0;
}


#undef FPK_R
#undef FPK_N
#undef FPK_L
#undef FPK_U
#undef FPK_G
#undef FPK_TINY
#undef FPK_I32MAX_R
#undef FPK_SQRT
#undef FPK_FABS
#undef FPK_CEIL
#undef FPK_FLOOR
#undef FPK_FMAX
