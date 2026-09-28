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
// GPU lattice setup (the counterpart of pfl_build in pfv_lattice.h).
//
// pfl_build is split where its work changes character:
//   pre_build   one thread per p: Z = kappa.p and the orthogonal lattice
//               (short serial xgcd chains);
//   tile_build  one 16-lane tile per p, state in shared memory: both LLLs
//               (93% of the setup) and the matrix products. Lanes own
//               columns of the basis and Gram updates; Gram-Schmidt is a
//               lane-parallel forward substitution. A thread-per-p LLL keeps
//               ~30 KB of state per thread in local memory, which does not
//               fit on chip at the parallelism needed, and diverges;
//   the HNF, factorization and search prep then run one thread per p again
//   (see pfvs_gpu.cu).
// Every integer operation is exact and checked, as in pfv_lattice.h; floating
// point only chooses the LLL's unimodular steps, so the basis can differ from
// the CPU's in rare ties -- any basis is valid, and the outputs (M, Kn, q)
// are basis independent. The Gram matrix is int64 (half the shared memory,
// far cheaper arithmetic); a p whose Gram entries exceed it reports
// ST_OVF64 and is redone with the int128 instantiation.
//
// Requires pfv_lattice.h (PFV_LATTICE_IMPLEMENTATION) to be included first.
#pragma once
#include <cooperative_groups.h>

namespace pfg {
namespace cg = cooperative_groups;

enum { TL = 16 };
typedef cg::thread_block_tile<TL> Tile;

// status codes (0 = ok)
enum {
    ST_OVF64 = 1,      // an int64 Gram entry overflowed: retry with int128 Gram
    ST_CPU   = 2,      // needs the host (overflow, GMP, LLL failure)
};

template <int MH, typename GGT>
struct Work {
    int64_t Binter[MH * MH], ZB[MH * MH];     // (h, d)
    int64_t mat[MH * MH];                     // (d, d)
    int64_t tmp[MH * MH];
    struct {
        int64_t B[MH * MH];                   // rows: basis (+ `last` at row m)
        GGT     GG[MH * MH];                  // exact Gram, stride MH
        double  mu[MH][MH], bb[MH], ibb[MH];
        int64_t T[MH * MH];                   // (d, d) basis change
    } l;
    int order[MH];
    int flag;                                 // status (any lane may set)
};

__device__ inline bool fits64(pfl_i128 x) { return PFL_FITS64(x); }

template <typename GGT>
__device__ inline bool gg_store(GGT *dst, pfl_i128 x);
template <>
__device__ inline bool gg_store<int64_t>(int64_t *dst, pfl_i128 x)
{
    if (!PFL_FITS64(x)) return false;
    *dst = (int64_t)x;
    return true;
}
template <>
__device__ inline bool gg_store<pfl_i128>(pfl_i128 *dst, pfl_i128 x)
{
    *dst = x;
    return true;
}

// LLL (delta 0.99) of rows 0..m-1 of W.B (length n) given their exact Gram
// W.GG; if has_last, row m is afterwards size-reduced against them (not
// swapped), as pfl_lll_core does with `last`. Returns 0, -1 (overflow; an
// int64 Gram overflow sets flag ST_OVF64), -2 (not positive), -3 (no
// convergence).
template <int MH, typename GGT>
__device__ int tile_lll(Tile t, Work<MH, GGT> *W, int m, int n, int has_last)
{
    const int lane = t.thread_rank();
    const int M = m + has_last;
    int64_t *B = W->l.B;
    GGT *GG = W->l.GG;
    double (*mu)[MH] = W->l.mu;
    double *bb = W->l.bb, *ibb = W->l.ibb;
    const double delta = 0.99;

    // Gram-Schmidt of row kk from the exact Gram (forward substitution; lane
    // j holds r_j = G_kj - sum_{l<j} mu_jl r_l)
    auto GS = [&](int kk) {
        double r = lane <= kk ? (double)GG[kk * MH + lane] : 0.0;
        double acc = 0.0;
        for (int l = 0; l < kk; ++l) {
            double rl = t.shfl(r, l);
            if (lane > l && lane < kk) r -= mu[lane][l] * rl;
            if (lane == kk) acc += rl * ibb[l] * rl;
            if (lane == 0) mu[kk][l] = rl * ibb[l];
        }
        if (lane == kk) { bb[kk] = r - acc; ibb[kk] = 1.0 / (r - acc); }
        t.sync();
    };
    // b_kk -= q b_jj on the vectors and the exact Gram
    auto SUB = [&](int kk, int jj, int64_t qi) -> int {
        int bad = 0, ovf64 = 0;
        const pfl_i128 kj = GG[kk * MH + jj], jv = GG[jj * MH + jj], kkv = GG[kk * MH + kk];
        pfl_i128 nl = 0, u = 0, tt;
        if (lane < n) {
            pfl_i128 x = (pfl_i128)B[kk * MH + lane] - (pfl_i128)qi * B[jj * MH + lane];
            if (!PFL_FITS64(x)) bad = 1; else B[kk * MH + lane] = (int64_t)x;
        }
        if (lane < M && lane != kk) {
            if (pfl_mul_ovf((pfl_i128)GG[jj * MH + lane], (pfl_i128)qi, &tt) ||
                fpk_sub_ovf((pfl_i128)GG[kk * MH + lane], tt, &nl)) bad = 1;
        }
        if (lane == 0) {              // G_kk - 2 q G_kj + q^2 G_jj
            if (pfl_mul_ovf(jv, (pfl_i128)qi, &tt) || fpk_sub_ovf(tt, 2 * kj, &tt) ||
                pfl_mul_ovf(tt, (pfl_i128)qi, &u) || fpk_add_ovf(kkv, u, &u)) bad = 1;
        }
        t.sync();                     // every read of the old Gram is done
        if (!bad && lane < M && lane != kk) {
            GGT g;
            if (gg_store<GGT>(&g, nl)) { GG[kk * MH + lane] = g; GG[lane * MH + kk] = g; }
            else ovf64 = 1;
        }
        if (!bad && lane == 0) {
            GGT g;
            if (gg_store<GGT>(&g, u)) GG[kk * MH + kk] = g; else ovf64 = 1;
        }
        t.sync();
        if (t.any(ovf64)) { if (lane == 0) W->flag = ST_OVF64; return 1; }
        return t.any(bad);
    };
    // size-reduce row k against rows 0..k-1 (mu[k] current on entry)
    auto SR = [&](int k) -> int {
        for (int pass = 0; pass < 4; ++pass) {
            int changed = 0;
            for (int j = k - 1; j >= 0; --j) {
                double q = nearbyint(mu[k][j]);
                if (q == 0.0) continue;
                if (!(fabs(q) <= 9.0e18)) return -1;
                int64_t qi = (int64_t)q;
                if (SUB(k, j, qi)) return -1;
                if (lane < j) mu[k][lane] -= q * mu[j][lane];
                if (lane == 0) mu[k][j] -= q;
                t.sync();
                changed = 1;
            }
            if (!changed) break;
            GS(k);
        }
        return 0;
    };

    if (m >= 1) { GS(0); if (!(bb[0] > 0)) return -2; }
    if (m >= 2) GS(1);
    int k = 1, iters = 0;
    while (k < m) {
        if (++iters > 100000) return -3;
        if (SR(k)) return -1;
        if (!(bb[k] > 0)) return -2;
        if (bb[k] < (delta - mu[k][k - 1] * mu[k][k - 1]) * bb[k - 1]) {
            if (lane < n) {
                int64_t x = B[k * MH + lane]; B[k * MH + lane] = B[(k - 1) * MH + lane]; B[(k - 1) * MH + lane] = x;
            }
            if (lane < M) {
                GGT x = GG[k * MH + lane]; GG[k * MH + lane] = GG[(k - 1) * MH + lane]; GG[(k - 1) * MH + lane] = x;
            }
            t.sync();
            if (lane < M) {
                GGT x = GG[lane * MH + k]; GG[lane * MH + k] = GG[lane * MH + (k - 1)]; GG[lane * MH + (k - 1)] = x;
            }
            t.sync();
            k = k > 1 ? k - 1 : 1;
            if (k == 1) GS(0);
            GS(k - 1);
            GS(k);
        } else {
            k++;
            if (k < m) GS(k);
        }
    }
    if (has_last && m > 0) {
        GS(m);                        // mu[m] = coefficients of `last`
        if (SR(m)) return -1;
    }
    return 0;
}

// C (rows x cols) = A (rows x inner) B (inner x cols), strides MH; checked.
template <int MH>
__device__ inline int tile_matmul(Tile t, const int64_t *A, const int64_t *B, int64_t *C,
                                  int rows, int inner, int cols)
{
    int bad = 0;
    for (int e = t.thread_rank(); e < rows * cols; e += TL) {
        int i = e / cols, j = e % cols;
        pfl_i128 s = 0;
        for (int l = 0; l < inner; ++l)
            if (fpk_add_ovf(s, (pfl_i128)A[i * MH + l] * B[l * MH + j], &s)) bad = 1;
        if (!PFL_FITS64(s)) bad = 1; else C[i * MH + j] = (int64_t)s;
    }
    t.sync();
    return t.any(bad);
}

// Thread-level first stage of pfl_build: Z = kappa.p (h x h, contiguous)
// and the orthogonal lattice O ((h-1) x h, contiguous). 0 or ST_CPU.
__device__ inline int pre_build(int h, const int64_t *kappa, const int64_t *Mb,
                                const int64_t *p, int64_t *Z, int64_t *O)
{
    int64_t T[PFL_MAX_H11], v[PFL_MAX_H11];
    for (int i = 0; i < h * h; ++i) {
        pfl_i128 s = 0;
        for (int k = 0; k < h; ++k)
            if (fpk_add_ovf(s, (pfl_i128)kappa[i * h + k] * p[k], &s)) return ST_CPU;
        if (!PFL_FITS64(s)) return ST_CPU;
        Z[i] = (int64_t)s;
    }
    if (pfl_matmul(Z, p, T, h, h, 1) || pfl_matmul(T, Mb, v, 1, h, h)) return ST_CPU;
    return pfl_orthogonal(v, h, O) ? ST_CPU : 0;
}

// The LLL part of pfl_build (coni, extra_lll, m0_basis) for one p, from
// Z and O (global, contiguous). On success W holds Binter, ZB, mat (strides
// MH); returns 0, else a status (ST_OVF64 / ST_CPU).
template <int MH, typename GGT>
__device__ int tile_build(Tile t, Work<MH, GGT> *W, int h, const int64_t *Z,
                          const int64_t *O, const int64_t *Mb)
{
    const int lane = t.thread_rank(), d = h - 1;
    int bad = 0;
    if (lane == 0) W->flag = 0;
    for (int e = lane; e < d * h; e += TL) W->l.B[(e / h) * MH + e % h] = O[e];
    t.sync();
    // LLL1: Euclidean Gram, then reduce
    for (int e = lane; e < d * d; e += TL) {
        int i = e / d, j = e % d;
        if (j > i) continue;
        pfl_i128 s = 0;
        for (int c = 0; c < h; ++c)
            if (fpk_add_ovf(s, (pfl_i128)W->l.B[i * MH + c] * W->l.B[j * MH + c], &s)) bad = 1;
        GGT g;
        if (!gg_store<GGT>(&g, s)) W->flag = ST_OVF64;
        W->l.GG[i * MH + j] = W->l.GG[j * MH + i] = g;
    }
    t.sync();
    if (t.any(bad)) return ST_CPU;
    if (W->flag) return W->flag;
    if (tile_lll<MH, GGT>(t, W, d, h, 0)) return W->flag ? W->flag : ST_CPU;

    // BT[a][i] = sum_l O[a][l] Mb[i][l]  (into tmp, (d, h))
    for (int e = lane; e < d * h; e += TL) {
        int a = e / h, i = e % h;
        pfl_i128 s = 0;
        for (int l = 0; l < h; ++l)
            if (fpk_add_ovf(s, (pfl_i128)Mb[i * h + l] * W->l.B[a * MH + l], &s)) bad = 1;
        if (!PFL_FITS64(s)) bad = 1; else W->tmp[a * MH + i] = (int64_t)s;
    }
    t.sync();
    if (t.any(bad)) return ST_CPU;
    // coni: columns with Binter[0] == 0 first (stable)
    if (lane == 0) {
        int no = 0;
        for (int a = 0; a < d; ++a) if (W->tmp[a * MH] == 0) W->order[no++] = a;
        for (int a = 0; a < d; ++a) if (W->tmp[a * MH] != 0) W->order[no++] = a;
    }
    t.sync();
    for (int e = lane; e < h * d; e += TL) {
        int i = e / d, a = e % d;
        W->Binter[i * MH + a] = W->tmp[W->order[a] * MH + i];
    }
    t.sync();
    // ZB = Z Binter ; mat = -Binter^T ZB
    for (int e = lane; e < h * d; e += TL) {
        int i = e / d, b = e % d;
        pfl_i128 s = 0;
        for (int k = 0; k < h; ++k)
            if (fpk_add_ovf(s, (pfl_i128)Z[i * h + k] * W->Binter[k * MH + b], &s)) bad = 1;
        if (!PFL_FITS64(s)) bad = 1; else W->ZB[i * MH + b] = (int64_t)s;
    }
    t.sync();
    if (t.any(bad)) return ST_CPU;
    for (int e = lane; e < d * d; e += TL) {
        int a = e / d, b = e % d;
        pfl_i128 s = 0;
        for (int i = 0; i < h; ++i)
            if (fpk_add_ovf(s, (pfl_i128)W->Binter[i * MH + a] * W->ZB[i * MH + b], &s)) bad = 1;
        if (!PFL_FITS64(s) || s == PFL_I64_MIN) bad = 1; else W->mat[a * MH + b] = (int64_t)(-s);
    }
    t.sync();
    if (t.any(bad)) return ST_CPU;

    // cut-aware basis: V l = (g, 0, ..., 0); K = V[1:], w = V[0]
    int nz = 0;
    for (int a = 0; a < d; ++a) nz += (W->Binter[a] != 0);
    if (nz) {
        if (lane == 0) {
            int64_t l[PFL_MAX_H11], V[PFL_MAX_H11 * PFL_MAX_H11];
            for (int a = 0; a < d; ++a) l[a] = W->Binter[a];
            if (pfl_unimodular(l, d, V)) W->flag = ST_CPU;
            else {
                // basis rows 0..d-2 = K, row d-1 = w (`last`)
                for (int a = 1; a < d; ++a)
                    for (int c = 0; c < d; ++c) W->l.B[(a - 1) * MH + c] = V[a * d + c];
                for (int c = 0; c < d; ++c) W->l.B[(d - 1) * MH + c] = V[c];
            }
        }
        t.sync();
        if (W->flag) return W->flag;
        // Gram w.r.t. mat of the d rows: tmp = B mat (d, d), GG = tmp B^T
        if (tile_matmul<MH>(t, W->l.B, W->mat, W->tmp, d, d, d)) return ST_CPU;
        for (int e = lane; e < d * d; e += TL) {
            int i = e / d, j = e % d;
            if (j > i) continue;
            pfl_i128 s = 0, tt;
            for (int c = 0; c < d; ++c) {
                if (pfl_mul_ovf((pfl_i128)W->tmp[i * MH + c], (pfl_i128)W->l.B[j * MH + c], &tt) ||
                    fpk_add_ovf(s, tt, &s)) bad = 1;
            }
            GGT g;
            if (!gg_store<GGT>(&g, s)) W->flag = ST_OVF64;
            W->l.GG[i * MH + j] = W->l.GG[j * MH + i] = g;
        }
        t.sync();
        if (t.any(bad)) return ST_CPU;
        if (W->flag) return W->flag;
        if (tile_lll<MH, GGT>(t, W, d - 1, d, 1)) return W->flag ? W->flag : ST_CPU;
        // T columns = basis rows (K..., w)
        for (int e = lane; e < d * d; e += TL) {
            int i = e / d, a = e % d;
            W->l.T[i * MH + a] = W->l.B[a * MH + i];
        }
        t.sync();
        if (tile_matmul<MH>(t, W->Binter, W->l.T, W->tmp, h, d, d)) return ST_CPU;
        for (int e = lane; e < h * d; e += TL) W->Binter[(e / d) * MH + e % d] = W->tmp[(e / d) * MH + e % d];
        t.sync();
        if (tile_matmul<MH>(t, W->ZB, W->l.T, W->tmp, h, d, d)) return ST_CPU;
        for (int e = lane; e < h * d; e += TL) W->ZB[(e / d) * MH + e % d] = W->tmp[(e / d) * MH + e % d];
        t.sync();
        // mat <- T^T (mat T)
        if (tile_matmul<MH>(t, W->mat, W->l.T, W->tmp, d, d, d)) return ST_CPU;
        for (int e = lane; e < d * d; e += TL) {
            int a = e / d, b = e % d;
            pfl_i128 s = 0;
            for (int i = 0; i < d; ++i)
                if (fpk_add_ovf(s, (pfl_i128)W->l.T[i * MH + a] * W->tmp[i * MH + b], &s)) bad = 1;
            if (!PFL_FITS64(s)) bad = 1; else W->l.B[a * MH + b] = (int64_t)s;   // (B free now)
        }
        t.sync();
        if (t.any(bad)) return ST_CPU;
        for (int e = lane; e < d * d; e += TL) W->mat[(e / d) * MH + e % d] = W->l.B[(e / d) * MH + e % d];
        t.sync();
    }

    t.sync();
    return W->flag;
}

} // namespace pfg
