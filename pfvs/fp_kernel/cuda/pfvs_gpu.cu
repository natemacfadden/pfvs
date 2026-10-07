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
// GPU version of the batched coni pipeline (_coni_batch in fp_kernel.pyx):
// for every p-vector, the lattice setup and the Fincke-Pohst search, giving
// the lattice points as (M = Binter c, Kn = Z Binter c, q = c^T mat c).
// Many geometries can share one call (each p names its geometry), which is
// what keeps the device busy when each geometry has few p-vectors.
//
// Kernels, per batch of p-vectors (all data stays on the device):
//   k_pre     thread per p: Z = kappa.p, orthogonal lattice
//   k_build   16-lane tile per p: both LLLs and the products (pfv_gpu.cuh);
//             p's whose int64 Gram overflows are redone with an int128 Gram
//   k_fin     thread per p: HNF, H rows, exact factorization, search prep
//   k_top     thread per p: the search down to depth `sd`, emitting prefixes
//   k_search  thread per prefix: the rest of that subtree, emitting points
// The search is fpk_common.h's fpk_search, in float when qmax < 2^22 (the
// GPU's FP64 rate is 1/64 of FP32; see fpk_search_impl.h for why that is
// exact) and double otherwise.
//
// Exactness is the CPU's: every decision is exact integer arithmetic. A p the
// device cannot finish exactly (an int128/int64 overflow, a factorization
// that needs GMP, an int32 coordinate range, an exact q beyond int128) is
// reported in pstat for the host to redo on the CPU path; none of its points
// are returned. Buffers that fill are grown and the batch is rerun.
//
// Points are returned grouped by p (ascending) and, within a p, sorted by
// (M, Kn) -- a canonical order, since the device's basis (hence the search
// order) can differ from the CPU's.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <mutex>

#ifndef PFL_MAX_H11
#define PFL_MAX_H11 12
#endif
// per-thread scratch sized to this build's dimension (the driver reserves
// the largest kernel stack for every resident thread)
#define FPK_SEARCH_MAXD (PFL_MAX_H11 - 1)
#define FPK_FAST_FACTOR_MAX_DIM (PFL_MAX_H11 - 1)
#define FPK_SEARCH_SPARSE 0
#define FPK_RES_MIN_W 0x7fffffff     /* residue mode: no gain on GPUs (latency-bound) */
#define PFL_NO_GMP
#define PFV_LATTICE_IMPLEMENTATION
#include "../pfv_lattice.h"
#include "pfg_compat.h"
#include "pfv_gpu.cuh"

enum { MH = PFL_MAX_H11, MD = PFL_MAX_H11 - 1 };
enum { ST_OK = 0, ST_CPU = 1, ST_EMPTY = 2, ST_PRE = -1, ST_LLL = -2 };

extern "C" {

typedef struct {
    int            n_geo;
    const int32_t *h;          // (n_geo,) h11 of each geometry (3 <= h <= pfg_max_h11())
    const int64_t *Q, *qmax, *linmin;           // (n_geo,) tadpole, floor(dil Q), ceil(M0min)
    const int64_t *kappa;      // the geometries' kappa (h^3 each), concatenated
    const int64_t *Mbasis;     // the geometries' Mbasis (h^2 each), concatenated
    int64_t        n_p;
    const int64_t *ps;         // (n_p, pfg_max_h11()) row-major; p (with its leading 0) in the first h
    const int32_t *pgeo;       // (n_p,) geometry of each p
    int            device;     // CUDA device ordinal
    int64_t        batch;      // p-vectors per batch (0: default)
    int            verbose;
} pfg_input;

typedef struct {
    int64_t  n;                // number of lattice points
    int64_t *M, *Kn;           // (n, pfg_max_h11()) row-major; first h entries valid
    int64_t *q;                // (n,) exact q
    int64_t *pidx;             // (n,) index of the p-vector
    int8_t  *pstat;            // (n_p,) 0: done here, 1: redo on the CPU path
    double   seconds_gpu;      // device time
    char     err[256];
} pfg_output;

int  pfg_max_h11(void) { return MH; }
const char *pfg_backend(void) { return PFG_HIP ? "hip" : "cuda"; }
int  pfg_device_count(void);
void pfg_release(int device);        // free the device buffers kept between calls
int  pfg_coni_batch(const pfg_input *in, pfg_output *out);
void pfg_output_free(pfg_output *out);

}  // extern "C"

namespace {

struct Geo { int h, use_f; int64_t Q, qmax, linmin; long koff, moff; };

struct PData {                 // per-p problem, device resident
    pfl_result R;
    double U[MD * MD], Uinv[MD], N[MD], V[MD * MD], Va[MD * MD];
    float Uf[MD * MD], Uinvf[MD], Nf[MD], Vf[MD * MD], Vaf[MD * MD];
    int64_t Hs[MH * MD];
    int ls[MD + 1], ord[MH];
    int m0l, status, h, geo;
};

struct Glob {
    int sd;
    const Geo *geo;
    const int64_t *kappa, *Mb, *ps;
    const int *pgeo;
    PData *pd;
    int32_t *pre; int *pre_p; unsigned long long *npre; unsigned long long pre_cap;
    int64_t *pts; int64_t *qs; int *pts_p; unsigned long long *npts; unsigned long long pts_cap;
};

struct Ctx { const Glob *G; int ip; };

using pfg::agg_inc;

// (a full buffer is not an error: the count keeps growing, the host sees it
// and reruns the batch with a larger buffer)
__device__ int emit_pre(void *vc, int kind, const int32_t *c, int n, int64_t)
{
    Ctx *X = (Ctx *)vc; const Glob *G = X->G;
    if (kind == 2) { G->pd[X->ip].status = ST_CPU; return 1; }
    unsigned long long k = agg_inc(G->npre);
    if (k >= G->pre_cap) return 0;
    for (int t = 0; t < n; ++t) G->pre[k * G->sd + t] = c[t];
    G->pre_p[k] = X->ip;
    return 0;
}

__device__ int emit_pt(void *vc, int kind, const int32_t *c, int n, int64_t q)
{
    Ctx *X = (Ctx *)vc; const Glob *G = X->G;
    PData *D = &G->pd[X->ip];
    if (kind == 2) { D->status = ST_CPU; return 1; }   // exact q needs GMP
    int64_t M[MH], K[MH];
    const int h = D->h;
    for (int i = 0; i < h; ++i) {
        fpk_i128 a = 0, b = 0;
        for (int j = 0; j < n; ++j) {
            a += (fpk_i128)D->R.Binter[i * n + j] * c[j];
            b += (fpk_i128)D->R.ZB[i * n + j] * c[j];
        }
        if (a > INT64_MAX || a < -INT64_MAX || b > INT64_MAX || b < -INT64_MAX) { D->status = ST_CPU; return 1; }
        M[i] = (int64_t)a; K[i] = (int64_t)b;
    }
    unsigned long long k = agg_inc(G->npts);
    if (k >= G->pts_cap) return 0;
    for (int i = 0; i < MH; ++i) {                     // (zero padding: the rows are compared whole)
        G->pts[k * 2 * MH + i] = i < h ? M[i] : 0;
        G->pts[k * 2 * MH + MH + i] = i < h ? K[i] : 0;
    }
    G->qs[k] = q;
    G->pts_p[k] = X->ip;
    return 0;
}

template <typename PR> struct PrepT;
template <> struct PrepT<fpk_prep> { typedef double R; };
template <> struct PrepT<fpk_prep_f> { typedef float R; };
__device__ void set_arrays(PData *D, fpk_prep *S) { S->U = D->U; S->Uinv = D->Uinv; S->m0N = D->N; S->m0V = D->V; S->m0Va = D->Va; }
__device__ void set_arrays(PData *D, fpk_prep_f *S) { S->U = D->Uf; S->Uinv = D->Uinvf; S->m0N = D->Nf; S->m0V = D->Vf; S->m0Va = D->Vaf; }
__device__ void consts(int d, int64_t qmax, fpk_prep *S) { fpk_search_consts(d, qmax, 1e-4, &S->slack, &S->Kerr, &S->max_err); }
__device__ void consts(int d, int64_t qmax, fpk_prep_f *S) { fpk_search_consts_f(d, qmax, 1e-4, &S->slack, &S->Kerr, &S->max_err); }

template <typename PR>
__device__ void make_prep(const Geo &g, PData *D, PR *S)
{
    const int d = D->h - 1;
    S->dim = d; S->strict = 1; S->use_gcd = D->R.nrows > 0; S->m0_level = D->m0l;
    S->Q = g.Q; S->qmax = g.qmax; S->linmin = g.linmin;
    S->qmax_d = (typename PrepT<PR>::R)g.qmax;
    consts(d, g.qmax, S);
    set_arrays(D, S);
    S->linvec = &D->R.Binter[0]; S->mat = D->R.mat; S->Hs = D->Hs;
    S->level_start = D->ls; S->order = D->ord;
    S->node_out = nullptr;
}

__device__ int run_search(const Glob *G, PData *D, const int32_t *prefix, int npre, int sd,
                          fpk_emit_fn emit, void *X)
{
    const Geo &g = G->geo[D->geo];
    if (g.use_f) { fpk_prep_f S; make_prep(g, D, &S); return fpk_search_f(&S, prefix, npre, sd, nullptr, emit, X, nullptr); }
    fpk_prep S; make_prep(g, D, &S); return fpk_search(&S, prefix, npre, sd, nullptr, emit, X, nullptr);
}

// depth at which a p's search is split into subtrees (at least one level
// is left to the subtree kernel)
__device__ inline int split_depth(const Glob &G, const PData *D)
{
    return G.sd < D->h - 1 ? G.sd : D->h - 2;
}

// statuses into a contiguous array (a strided copy of 4 bytes per 12 KB
// PData is slow)
__global__ void k_status(const PData *pd, int n, int *out)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip < n) out[ip] = pd[ip].status;
}

__global__ void k_pre(Glob G, int n, int64_t *Og)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n) return;
    PData *D = &G.pd[ip];
    const int gi = G.pgeo[ip];
    const Geo &g = G.geo[gi];
    D->h = g.h; D->geo = gi;
    int st = pfg::pre_build(g.h, &G.kappa[g.koff], &G.Mb[g.moff], &G.ps[(size_t)ip * MH],
                            D->R.Z, &Og[(size_t)ip * MH * MH]);
    D->status = st ? ST_CPU : ST_PRE;
}

// Lattice setup, stage by stage over retry lists: <int64 Gram, float GS>
// for every p; failures are redone with a double Gram-Schmidt (retryA), and
// int64 Gram overflows with an int128 Gram (retryB). A float LLL only ever
// fails to converge or loses a pivot's sign -- it cannot produce a wrong
// basis (all basis operations are exact) -- so the retries cost time only.
#ifndef PFG_TPB
#define PFG_TPB 2
#endif
enum { TPB = PFG_TPB };            // tiles per block (two 16-lane tiles fill a warp)
template <typename GGT, typename R>
__global__ void __launch_bounds__(TPB * pfg::TL) k_build(Glob G, int n, const int *idx, const int64_t *Og,
                                                         int64_t *Xg, int *retryA, int *nA, int *retryB, int *nB)
{
    __shared__ pfg::Work<MH, GGT, R> Ws[TPB];
    pfg::Tile t;
    const int lane = t.thread_rank(), ti = threadIdx.x / pfg::TL;
    const int k = blockIdx.x * TPB + ti;
    if (k >= n) return;                                   // uniform per tile
    const int ip = idx ? idx[k] : k;
    PData *D = &G.pd[ip];
    if (D->status != ST_PRE) return;
    const Geo &g = G.geo[D->geo];
    const int h = g.h, d = h - 1;
    pfg::Work<MH, GGT, R> &W = Ws[ti];
    int64_t *X = &Xg[(size_t)ip * 5 * MH * MH];
    W.Binter = X; W.ZB = X + MH * MH; W.mat = X + 2 * MH * MH; W.tmp = X + 3 * MH * MH; W.T = X + 4 * MH * MH;
    t.sync();
    int st = pfg::tile_build<MH, GGT, R>(t, &W, h, D->R.Z, &Og[(size_t)ip * MH * MH], &G.Mb[g.moff]);
    if (st) {
        if (lane == 0) {
            if (st == pfg::ST_OVF64 && retryB) retryB[atomicAdd(nB, 1)] = ip;
            else if (st != pfg::ST_OVF64 && retryA) retryA[atomicAdd(nA, 1)] = ip;
            else D->status = ST_CPU;
        }
        return;
    }
    for (int e = lane; e < h * d; e += pfg::TL) {
        D->R.Binter[e] = W.Binter[(e / d) * MH + e % d];
        D->R.ZB[e] = W.ZB[(e / d) * MH + e % d];
    }
    for (int e = lane; e < d * d; e += pfg::TL) D->R.mat[e] = W.mat[(e / d) * MH + e % d];
    if (lane == 0) D->status = ST_LLL;
}

__global__ void k_fin(Glob G, int n)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n) return;
    PData *D = &G.pd[ip];
    if (D->status != ST_LLL) return;
    const Geo &g = G.geo[D->geo];
    const int d = D->h - 1, nr = d;
    {
        pfl_i128 Hl[MD * MD];
        if (pfl_hnf(&D->R.ZB[d], nr, d, Hl)) { D->status = ST_CPU; return; }
        for (int e = 0; e < nr * d; ++e) {
            if (Hl[e] > INT64_MAX || Hl[e] < -INT64_MAX) { D->status = ST_CPU; return; }
            D->Hs[e] = (int64_t)Hl[e];
        }
    }
    D->R.nrows = nr;
    for (int kk = 0; kk < d * d; ++kk) D->V[kk] = D->Va[kk] = 0.0;
    if (fpk_factor_fast(D->R.mat, d, D->U, &D->R.Binter[0], D->N, D->V, D->Va)) { D->status = ST_CPU; return; }
    for (int i = 0; i < d; ++i) D->Uinv[i] = 1.0 / D->U[i * d + i];
    if (g.use_f) {
        for (int kk = 0; kk < d * d; ++kk) { D->Uf[kk] = (float)D->U[kk]; D->Vf[kk] = (float)D->V[kk]; D->Vaf[kk] = (float)D->Va[kk]; }
        for (int i = 0; i < d; ++i) { D->Nf[i] = (float)D->N[i]; D->Uinvf[i] = 1.0f / D->Uf[i * d + i]; }
    }
    D->m0l = -1;
    for (int j = 0; j < d; ++j) if (D->R.Binter[j]) { D->m0l = j; break; }
    int cnt[MD + 1] = {0}, lvl[MH];
    for (int r = 0; r < nr; ++r) {
        lvl[r] = -1;
        for (int j = 0; j < d; ++j) if (D->Hs[r * d + j]) { lvl[r] = j; break; }
        if (lvl[r] >= 0) cnt[lvl[r]]++;
    }
    D->ls[0] = 0;
    for (int l = 0; l < d; ++l) D->ls[l + 1] = D->ls[l] + cnt[l];
    int fill[MD + 1];
    for (int l = 0; l <= d; ++l) fill[l] = D->ls[l];
    for (int r = 0; r < nr; ++r) if (lvl[r] >= 0) D->ord[fill[lvl[r]]++] = r;
    D->status = (D->m0l < 0 && 0 < g.linmin) ? ST_EMPTY : ST_OK;   // M0 == 0 < linmin
}

// The search is split into subtrees in stages: k_top (thread per p) emits
// the depth-1 prefixes; k_expand (stages s < sd) extends depth-s prefixes to
// depth s + 1, or searches the rest of the subtree when s is that p's split
// depth; k_leaf (stage sd, thread per prefix) searches the subtrees. Each
// stage has short work items, so the work stays balanced.
__global__ void k_top(Glob G, int n)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n) return;
    PData *D = &G.pd[ip];
    if (D->status != ST_OK) return;
    Ctx X = {&G, ip};
    int st = split_depth(G, D) > 0 ? run_search(&G, D, nullptr, 0, 1, emit_pre, &X)
                                    : run_search(&G, D, nullptr, 0, 0, emit_pt, &X);
    if (st == -8 || st == -11) D->status = ST_CPU;   // int32 range; float precision
}

#ifndef PFG_EXP_BLOCK
#define PFG_EXP_BLOCK 128
#endif
#ifndef PFG_EXP_BOUNDS
#define PFG_EXP_BOUNDS 128
#endif
#ifndef PFG_COOP
#define PFG_COOP 1                   /* 0: k_expand searches a prefix per thread */
#endif

__global__ void __launch_bounds__(PFG_EXP_BOUNDS) k_leaf(Glob G, const int32_t *src, const int *src_p, unsigned long long n, int s)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int ip = src_p[k];
    PData *D = &G.pd[ip];
    if (D->status != ST_OK) return;
    Ctx X = {&G, ip};
    int st = run_search(&G, D, &src[k * G.sd], s, 0, emit_pt, &X);
    if (st == -8 || st == -11) D->status = ST_CPU;   // int32 range; float precision
}

// k_expand is warp-cooperative: each thread sets up its prefix's level node
// (fpk_search with node_out), then the warp's lanes share the union of their
// nodes' candidates, one candidate per lane per round (a thread per prefix
// leaves most lanes idle: the candidate counts and gcd lengths vary). Nodes
// with several H rows at the level and double geometries keep the
// thread-per-prefix search.
struct CoopSlot { fpk_node_f nd; unsigned long long k; int ip, geo, cnt, end; };

__global__ void __launch_bounds__(PFG_EXP_BOUNDS) k_expand(Glob G, const int32_t *src, const int *src_p, unsigned long long n, int s)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
#if PFG_COOP
    __shared__ CoopSlot slot[PFG_EXP_BLOCK];
    CoopSlot &my = slot[threadIdx.x];
    my.cnt = 0;
#endif
    if (k < n) {                                  // (no early return: the warp syncs below)
        int ip = src_p[k];
        PData *D = &G.pd[ip];
        if (D->status == ST_OK) {
            Ctx X = {&G, ip};
            int st;
            if (split_depth(G, D) > s) {
#if PFG_COOP
                const Geo &g = G.geo[D->geo];
                if (g.use_f) {
                    fpk_prep_f P; make_prep(g, D, &P);
                    P.node_out = &my.nd;
                    st = fpk_search_f(&P, &src[k * G.sd], s, s + 1, nullptr, emit_pre, &X, nullptr);
                    if (st == 1) {
                        my.k = k; my.ip = ip; my.geo = D->geo;
                        my.cnt = my.nd.hi >= my.nd.lo ? (int)(my.nd.hi - my.nd.lo + 1) : 0;
                        st = 0;
                    }
                } else
#endif
                st = run_search(&G, D, &src[k * G.sd], s, s + 1, emit_pre, &X);
            } else {
                st = run_search(&G, D, &src[k * G.sd], s, 0, emit_pt, &X);
            }
            if (st == -8 || st == -11) D->status = ST_CPU;   // int32 range; float precision
        }
    }
#if PFG_COOP
    // the warp's candidates: inclusive prefix sums of the counts
    // (the block size is a multiple of the warp size: whole warps)
    const int W = pfg::wave_size();
    const int lane = threadIdx.x & (W - 1), w0 = threadIdx.x & ~(W - 1);
    int incl = my.cnt;
    for (int o = 1; o < W; o <<= 1) {
        int t = pfg::wave_shfl_up(incl, o);
        if (lane >= o) incl += t;
    }
    my.end = incl;
    const int total = pfg::wave_shfl(incl, W - 1);
    pfg::wave_sync();
    for (int base = 0; base < total; base += W) {
        const int t = base + lane;
        if (t >= total) break;
        int a = 0, b = W - 1;                     // owner: first slot with end > t
        while (a < b) { int m = (a + b) >> 1; if (slot[w0 + m].end > t) b = m; else a = m + 1; }
        const CoopSlot &O = slot[w0 + a];
        const int32_t v = (int32_t)(O.nd.lo + (t - (O.end - O.cnt)));
        PData *D = &G.pd[O.ip];
        fpk_prep_f P; make_prep(G.geo[O.geo], D, &P);
        int r = fpk_node_test_f(&P, &O.nd, v);
        if (r == 1) {                             // as emit_pre: the prefix (v, src[k])
            unsigned long long kk = agg_inc(G.npre);
            if (kk < G.pre_cap) {
                G.pre[kk * G.sd] = v;
                for (int j = 0; j < s; ++j) G.pre[kk * G.sd + 1 + j] = src[O.k * G.sd + j];
                G.pre_p[kk] = O.ip;
            }
        } else if (r == -11) {
            D->status = ST_CPU;
        }
    }
#endif
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { last_err = e_; \
    snprintf(out->err, sizeof out->err, "%s (%s:%d)", cudaGetErrorString(e_), __FILE__, __LINE__); \
    goto fail; } } while (0)

// Per-device workspace: device buffers are kept between calls (allocating
// and freeing gigabytes per call costs more than the work) and regrown only
// when a call needs more; pfg_release frees them.
enum { WS_GEO, WS_K, WS_M, WS_PS, WS_PGEO, WS_PD, WS_OG, WS_XG, WS_RA, WS_RB, WS_CNT, WS_NPRE,
       WS_NPTS, WS_PRE0, WS_PREP0, WS_PRE1, WS_PREP1, WS_PTS, WS_QS, WS_PTSP, WS_STAT, WS_N };
struct Workspace { void *p[WS_N] = {}; size_t cap[WS_N] = {}; std::mutex mu; };
std::mutex ws_map_mu;
std::map<int, Workspace *> ws_map;

Workspace &workspace(int device)
{
    std::lock_guard<std::mutex> g(ws_map_mu);
    Workspace *&w = ws_map[device];
    if (!w) w = new Workspace();
    return *w;
}

cudaError_t ws_get(Workspace &w, int slot, void **ptr, size_t bytes)
{
    if (w.cap[slot] < bytes) {
        if (w.p[slot]) (void)cudaFree(w.p[slot]);
        w.p[slot] = NULL; w.cap[slot] = 0;
        size_t b = bytes + bytes / 8 + 256;            // headroom for the next call
        cudaError_t e = cudaMalloc(&w.p[slot], b);
        if (e != cudaSuccess) { w.p[slot] = NULL; return e; }
        w.cap[slot] = b;
    }
    *ptr = w.p[slot];
    return cudaSuccess;
}

size_t ws_bytes(const Workspace &w)
{
    size_t t = 0;
    for (int k = 0; k < WS_N; ++k) t += w.cap[k];
    return t;
}

void ws_release(Workspace &w)
{
    for (int k = 0; k < WS_N; ++k) {
        if (w.p[k]) (void)cudaFree(w.p[k]);
        w.p[k] = NULL; w.cap[k] = 0;
    }
}

#define WS(ptr, slot, n) ws_get(W, slot, (void **)&(ptr), (size_t)(n) * sizeof(*(ptr)) + 16)

}  // namespace

extern "C" int pfg_device_count(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

extern "C" void pfg_output_free(pfg_output *out)
{
    free(out->M); free(out->Kn); free(out->q); free(out->pidx); free(out->pstat);
    out->M = out->Kn = out->q = out->pidx = NULL; out->pstat = NULL; out->n = 0;
}

extern "C" int pfg_coni_batch(const pfg_input *in, pfg_output *out)
{
    memset(out, 0, sizeof *out);
    const long NP = (long)in->n_p;
    long batch = in->batch > 0 ? (long)in->batch : (1L << 19);
#ifndef PFG_SD
#define PFG_SD 4
#endif
    const int sd = PFG_SD;
    std::vector<Geo> geos(in->n_geo);
    std::vector<int64_t> Mo, Ko, qo, po;         // outputs, grouped by batch
    Glob G = {};
    Geo *dgeo = NULL; int64_t *dk = NULL, *dm = NULL, *dps = NULL, *Og = NULL, *Xg = NULL;
    int *dpgeo = NULL, *rA = NULL, *rB = NULL, *cnt = NULL;   // retry lists, their counts
    struct Buf { int32_t *pre; int *pre_p; unsigned long long cap; } buf[2] = {};
    long koff = 0, moff = 0;
    unsigned long long pre_cap = (unsigned long long)std::min(batch, NP) * 16, pts_cap = 1ULL << 16;
    std::vector<int> stat;
    enum { NEV = 8 };
    cudaEvent_t ev[NEV] = {};
    double tstage[NEV] = {0};
    int rc = 1;
    long off = 0;                                // p-vectors done
    cudaError_t last_err = cudaSuccess;

    out->pstat = (int8_t *)calloc(NP > 0 ? NP : 1, 1);
    if (!out->pstat) { snprintf(out->err, sizeof out->err, "out of host memory"); return 1; }
    for (int gi = 0; gi < in->n_geo; ++gi) {
        Geo &g = geos[gi];
        g.h = in->h[gi]; g.Q = in->Q[gi]; g.qmax = in->qmax[gi]; g.linmin = in->linmin[gi];
        if (g.h < 3 || g.h > MH || g.Q <= 0 || g.qmax < 0) {
            snprintf(out->err, sizeof out->err, "geometry %d: unsupported h11 = %d (3..%d), Q or qmax", gi, g.h, MH);
            return 1;
        }
        g.use_f = g.qmax < (1LL << 22);
        g.koff = koff; g.moff = moff;
        koff += (long)g.h * g.h * g.h; moff += (long)g.h * g.h;
    }
    for (long ip = 0; ip < NP; ++ip)
        if (in->pgeo[ip] < 0 || in->pgeo[ip] >= in->n_geo) { snprintf(out->err, sizeof out->err, "p %ld: bad geometry index", ip); return 1; }
    if (NP == 0) return 0;

    Workspace &W = workspace(in->device);
    std::unique_lock<std::mutex> ws_lock(W.mu);        // one call per device at a time
    CK(cudaSetDevice(in->device));
    if (in->batch <= 0) {
        // default batch: what fits in about half the free device memory
        // (per p: the problem, the setup scratch and ~16 subtree prefixes)
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        const size_t per_p = sizeof(PData) + 6 * MH * MH * 8 + MH * 8 + 16
                           + 2 * 16 * (sd * 4 + 4);
        free_b += ws_bytes(W);                         // (ours, reusable)
        batch = std::max(1L << 12, std::min(batch, (long)(free_b / 2 / per_p)));
    }
    CK(WS(dgeo, WS_GEO, geos.size())); CK(cudaMemcpy(dgeo, geos.data(), geos.size() * sizeof(Geo), cudaMemcpyHostToDevice));
    CK(WS(dk, WS_K, koff)); CK(cudaMemcpy(dk, in->kappa, koff * 8, cudaMemcpyHostToDevice));
    CK(WS(dm, WS_M, moff)); CK(cudaMemcpy(dm, in->Mbasis, moff * 8, cudaMemcpyHostToDevice));
    for (int q = 0; q < NEV; ++q) CK(cudaEventCreate(&ev[q]));
restart:                                         // (after an out-of-memory: smaller batches)
    {
        const long nb = std::min(batch, NP);
        CK(WS(dps, WS_PS, nb * MH)); CK(WS(dpgeo, WS_PGEO, nb));
        CK(WS(G.pd, WS_PD, nb)); CK(WS(Og, WS_OG, nb * MH * MH)); CK(WS(Xg, WS_XG, nb * 5 * MH * MH));
        CK(WS(rA, WS_RA, nb)); CK(WS(rB, WS_RB, nb)); CK(WS(cnt, WS_CNT, 2));
        CK(WS(G.npre, WS_NPRE, 1)); CK(WS(G.npts, WS_NPTS, 1));
        stat.resize(nb);
    }
    G.sd = sd; G.geo = dgeo; G.kappa = dk; G.Mb = dm; G.ps = dps; G.pgeo = dpgeo;
    {
    auto grow_buf = [&](int b, unsigned long long need) -> cudaError_t {
        pre_cap = std::max(pre_cap, need + need / 2);
        cudaError_t e = WS(buf[b].pre, b ? WS_PRE1 : WS_PRE0, pre_cap * sd);
        if (e == cudaSuccess) e = WS(buf[b].pre_p, b ? WS_PREP1 : WS_PREP0, pre_cap);
        if (e == cudaSuccess) buf[b].cap = pre_cap;
        return e;
    };

    while (off < NP) {
        const int n = (int)std::min(batch, NP - off);
        for (int b = 0; b < 2; ++b)
            if (!buf[b].pre) {
                CK(WS(buf[b].pre, b ? WS_PRE1 : WS_PRE0, pre_cap * sd));
                CK(WS(buf[b].pre_p, b ? WS_PREP1 : WS_PREP0, pre_cap));
                // (a buffer kept from earlier calls may be larger: use all of it)
                buf[b].cap = std::min(W.cap[b ? WS_PRE1 : WS_PRE0] / (sd * 4),
                                      W.cap[b ? WS_PREP1 : WS_PREP0] / 4) - 4;
            }
        if (!G.pts) {
            CK(WS(G.pts, WS_PTS, pts_cap * 2 * MH)); CK(WS(G.qs, WS_QS, pts_cap)); CK(WS(G.pts_p, WS_PTSP, pts_cap));
            // (buffers kept from earlier calls may be larger: use all of them)
            G.pts_cap = std::min({W.cap[WS_PTS] / (2 * MH * 8), W.cap[WS_QS] / 8, W.cap[WS_PTSP] / 4}) - 4;
        }
        CK(cudaMemcpy(dps, &in->ps[off * MH], (size_t)n * MH * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpgeo, &in->pgeo[off], (size_t)n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(G.npts, 0, 8)); CK(cudaMemset(cnt, 0, 8));

        // lattice setup
        CK(cudaEventRecord(ev[0]));
        k_pre<<<(n + 127) / 128, 128>>>(G, n, Og);
        CK(cudaEventRecord(ev[1]));
        k_build<int64_t, float><<<(n + TPB - 1) / TPB, TPB * pfg::TL>>>(G, n, nullptr, Og, Xg, rA, &cnt[0], rB, &cnt[1]);
        int nre[2];
        CK(cudaMemcpy(nre, cnt, 8, cudaMemcpyDeviceToHost));
        // int64 Gram overflows (retry list B) first get an int128 Gram with float
        // Gram-Schmidt; anything float cannot finish is then done in double
        if (nre[1]) k_build<pfl_i128, float><<<(nre[1] + TPB - 1) / TPB, TPB * pfg::TL>>>(G, nre[1], rB, Og, Xg, rA, &cnt[0], nullptr, nullptr);
        CK(cudaMemcpy(&nre[0], &cnt[0], 4, cudaMemcpyDeviceToHost));
        CK(cudaMemset(&cnt[1], 0, 4));
        if (nre[0]) k_build<int64_t, double><<<(nre[0] + TPB - 1) / TPB, TPB * pfg::TL>>>(G, nre[0], rA, Og, Xg, nullptr, nullptr, rB, &cnt[1]);
        int nre2;
        CK(cudaMemcpy(&nre2, &cnt[1], 4, cudaMemcpyDeviceToHost));
        if (nre2) k_build<pfl_i128, double><<<(nre2 + TPB - 1) / TPB, TPB * pfg::TL>>>(G, nre2, rB, Og, Xg, nullptr, nullptr, nullptr, nullptr);
        CK(cudaEventRecord(ev[2]));
        k_fin<<<(n + 127) / 128, 128>>>(G, n);
        CK(cudaEventRecord(ev[3]));

        // the search; if the points overflow their buffer (they are dropped,
        // no status changes), grow it and search again -- not the setup
        unsigned long long npts = 0, nstage[sd + 1] = {0};
        for (;;) {
            CK(cudaMemset(G.npts, 0, 8));
            // search, split into subtrees in stages (ping-pong prefix buffers). A
            // stage whose prefixes overflow its output buffer is rerun alone with
            // a larger one: its input is intact, and the points it emitted are
            // dropped by resetting the point count (status changes are idempotent).
            unsigned long long nsrc = 0, npts0 = 0;
            for (int q = 0; q <= sd; ++q) nstage[q] = 0;
            int cur = 0;
            for (;;) {
                G.pre = buf[cur].pre; G.pre_p = buf[cur].pre_p; G.pre_cap = buf[cur].cap;
                CK(cudaMemset(G.npre, 0, 8));
                k_top<<<(n + 127) / 128, 128>>>(G, n);
                CK(cudaMemcpy(&nsrc, G.npre, 8, cudaMemcpyDeviceToHost));
                if (nsrc <= buf[cur].cap) break;
                CK(grow_buf(cur, nsrc));
            }
            nstage[1] = nsrc;
            CK(cudaEventRecord(ev[4]));
            for (int s = 1; s <= sd && nsrc; ++s) {
                const int dst = cur ^ 1;
                CK(cudaMemcpy(&npts0, G.npts, 8, cudaMemcpyDeviceToHost));
                unsigned long long ndst;
                for (;;) {
                    G.pre = buf[dst].pre; G.pre_p = buf[dst].pre_p; G.pre_cap = buf[dst].cap;
                    CK(cudaMemset(G.npre, 0, 8));
                    const unsigned nb = (unsigned)((nsrc + PFG_EXP_BLOCK - 1) / PFG_EXP_BLOCK);
                    if (s < sd) k_expand<<<nb, PFG_EXP_BLOCK>>>(G, buf[cur].pre, buf[cur].pre_p, nsrc, s);
                    else k_leaf<<<nb, PFG_EXP_BLOCK>>>(G, buf[cur].pre, buf[cur].pre_p, nsrc, s);
                    if (s == sd) { ndst = 0; break; }  // the last stage emits points only
                    CK(cudaMemcpy(&ndst, G.npre, 8, cudaMemcpyDeviceToHost));
                    if (ndst <= buf[dst].cap) break;
                    CK(grow_buf(dst, ndst));
                    CK(cudaMemcpy(G.npts, &npts0, 8, cudaMemcpyHostToDevice));
                }
                if (s == sd) break;
                nstage[s + 1] = nsrc = ndst;
                cur = dst;
            }
            CK(cudaEventRecord(ev[5]));
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            CK(cudaMemcpy(&npts, G.npts, 8, cudaMemcpyDeviceToHost));
            if (npts <= G.pts_cap) break;
            pts_cap = npts + npts / 2;
            CK(WS(G.pts, WS_PTS, pts_cap * 2 * MH)); CK(WS(G.qs, WS_QS, pts_cap)); CK(WS(G.pts_p, WS_PTSP, pts_cap));
            G.pts_cap = pts_cap;
        }
        float ms[5];
        for (int q = 0; q < 5; ++q) { CK(cudaEventElapsedTime(&ms[q], ev[q], ev[q + 1])); tstage[q] += ms[q] / 1e3; }
        out->seconds_gpu += (ms[0] + ms[1] + ms[2] + ms[3] + ms[4]) / 1e3;
        {
            int *dstat = NULL;
            CK(WS(dstat, WS_STAT, n));
            k_status<<<(n + 255) / 256, 256>>>(G.pd, n, dstat);
            CK(cudaMemcpy(stat.data(), dstat, (size_t)n * 4, cudaMemcpyDeviceToHost));
        }
        {
            std::vector<int64_t> pts(npts * 2 * MH), qs(npts);
            std::vector<int> pp(npts);
            CK(cudaMemcpy(pts.data(), G.pts, npts * 2 * MH * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(qs.data(), G.qs, npts * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(pp.data(), G.pts_p, npts * 4, cudaMemcpyDeviceToHost));
            // keep the points of p's finished here; canonical order: grouped by
            // p (counting sort), then by (M, Kn) within a p (small groups)
            std::vector<unsigned long long> start(n + 1, 0);
            for (unsigned long long k = 0; k < npts; ++k)
                if (stat[pp[k]] == ST_OK) start[pp[k] + 1]++;
            for (int ip = 0; ip < n; ++ip) start[ip + 1] += start[ip];
            std::vector<unsigned long long> idx(start[n]);
            {
                std::vector<unsigned long long> fill(start.begin(), start.end() - 1);
                for (unsigned long long k = 0; k < npts; ++k)
                    if (stat[pp[k]] == ST_OK) idx[fill[pp[k]]++] = k;
            }
            auto less = [&](unsigned long long a, unsigned long long b) {
                return std::lexicographical_compare(&pts[a * 2 * MH], &pts[a * 2 * MH + 2 * MH],
                                                    &pts[b * 2 * MH], &pts[b * 2 * MH + 2 * MH]);
            };
            for (int ip = 0; ip < n; ++ip)
                if (start[ip + 1] - start[ip] > 1)
                    std::sort(idx.begin() + start[ip], idx.begin() + start[ip + 1], less);
            const size_t base = qo.size();
            Mo.resize((base + idx.size()) * MH); Ko.resize((base + idx.size()) * MH);
            qo.resize(base + idx.size()); po.resize(base + idx.size());
            for (size_t t = 0; t < idx.size(); ++t) {
                const unsigned long long k = idx[t];
                memcpy(&Mo[(base + t) * MH], &pts[k * 2 * MH], MH * 8);
                memcpy(&Ko[(base + t) * MH], &pts[k * 2 * MH + MH], MH * 8);
                qo[base + t] = qs[k];
                po[base + t] = off + pp[k];
            }
        }
        for (int ip = 0; ip < n; ++ip)
            out->pstat[off + ip] = (stat[ip] == ST_OK || stat[ip] == ST_EMPTY) ? 0 : 1;
        if (in->verbose)
            fprintf(stderr, "pfg: batch @%ld: %d p | pre %.1f, LLL %.1f (redo: %d double, %d int128), fin %.1f, "
                    "top %.1f, subtrees %.1f ms | prefixes %llu/%llu/%llu, %llu points\n",
                    off, n, ms[0], ms[1], nre[0], nre[1], ms[2], ms[3], ms[4],
                    nstage[1], nstage[2], nstage[3], npts);
        off += n;
    }
    }
    if (in->verbose) {
        double tt = 0;
        for (int q = 0; q < 5; ++q) tt += tstage[q];
        const char *nm[5] = {"pre", "LLL tiles", "fin (HNF, factor)", "top search", "subtrees"};
        for (int q = 0; q < 5; ++q) fprintf(stderr, "pfg: %-18s %8.1f ms %5.1f%%\n", nm[q], tstage[q] * 1e3, 100 * tstage[q] / tt);
    }
    {
        const size_t n = qo.size();
        out->n = (int64_t)n;
        out->M = (int64_t *)malloc(n * MH * 8 + 8); out->Kn = (int64_t *)malloc(n * MH * 8 + 8);
        out->q = (int64_t *)malloc(n * 8 + 8); out->pidx = (int64_t *)malloc(n * 8 + 8);
        if (!out->M || !out->Kn || !out->q || !out->pidx) { snprintf(out->err, sizeof out->err, "out of host memory"); goto fail; }
        if (n) {
            memcpy(out->M, Mo.data(), n * MH * 8); memcpy(out->Kn, Ko.data(), n * MH * 8);
            memcpy(out->q, qo.data(), n * 8); memcpy(out->pidx, po.data(), n * 8);
        }
    }
    rc = 0;
fail:
    if (rc && last_err == cudaErrorMemoryAllocation && batch > 4096) {
        // out of device memory (other processes, small or shared memory):
        // free the per-batch buffers, halve the batch, carry on from `off`
        (void)cudaGetLastError();
        ws_release(W);
        dps = Og = Xg = NULL; dpgeo = rA = rB = cnt = NULL; G.pd = NULL;
        G.npre = G.npts = NULL; buf[0] = buf[1] = Buf(); G.pts = G.qs = NULL; G.pts_p = NULL;
        dgeo = NULL; dk = dm = NULL;
        CK(WS(dgeo, WS_GEO, geos.size())); CK(cudaMemcpy(dgeo, geos.data(), geos.size() * sizeof(Geo), cudaMemcpyHostToDevice));
        CK(WS(dk, WS_K, koff)); CK(cudaMemcpy(dk, in->kappa, koff * 8, cudaMemcpyHostToDevice));
        CK(WS(dm, WS_M, moff)); CK(cudaMemcpy(dm, in->Mbasis, moff * 8, cudaMemcpyHostToDevice));
        batch /= 2;
        pre_cap = (unsigned long long)std::min(batch, NP) * 16;
        pts_cap = 1ULL << 16;
        if (in->verbose) fprintf(stderr, "pfg: out of device memory; batch -> %ld\n", batch);
        out->err[0] = 0;
        last_err = cudaSuccess;
        goto restart;
    }
    if (rc) pfg_output_free(out);
    for (int q = 0; q < NEV; ++q) if (ev[q]) (void)cudaEventDestroy(ev[q]);
    return rc;
}

extern "C" void pfg_release(int device)
{
    Workspace &W = workspace(device);
    std::lock_guard<std::mutex> g(W.mu);
    if (cudaSetDevice(device) == cudaSuccess) ws_release(W);
}
