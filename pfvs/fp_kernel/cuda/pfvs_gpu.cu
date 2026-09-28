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

#ifndef PFL_MAX_H11
#define PFL_MAX_H11 12
#endif
// per-thread scratch sized to this build's dimension (the driver reserves
// the largest kernel stack for every resident thread)
#define FPK_SEARCH_MAXD (PFL_MAX_H11 - 1)
#define FPK_FAST_FACTOR_MAX_DIM (PFL_MAX_H11 - 1)
#define PFL_NO_GMP
#define PFV_LATTICE_IMPLEMENTATION
#include "../pfv_lattice.h"
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
int  pfg_device_count(void);
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

// warp-aggregated slot reservation on a global counter (divergence-safe)
__device__ inline unsigned long long agg_inc(unsigned long long *ctr)
{
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
    unsigned rank = __popc(mask & ((1u << lane) - 1));
    unsigned long long base = 0;
    if (lane == leader) base = atomicAdd(ctr, (unsigned long long)__popc(mask));
    base = __shfl_sync(mask, base, leader);
    return base + rank;
}

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
    for (int i = 0; i < h; ++i) { G->pts[k * 2 * MH + i] = M[i]; G->pts[k * 2 * MH + MH + i] = K[i]; }
    G->qs[k] = q;
    G->pts_p[k] = X->ip;
    return 0;
}

template <typename PR> struct PrepT;
template <> struct PrepT<fpk_prep> { typedef double R; };
template <> struct PrepT<fpk_prep_f> { typedef float R; };
__device__ void set_arrays(PData *D, fpk_prep *S) { S->U = D->U; S->Uinv = D->Uinv; S->m0N = D->N; S->m0V = D->V; S->m0Va = D->Va; }
__device__ void set_arrays(PData *D, fpk_prep_f *S) { S->U = D->Uf; S->Uinv = D->Uinvf; S->m0N = D->Nf; S->m0V = D->Vf; S->m0Va = D->Vaf; }
__device__ void consts(int d, int64_t qmax, fpk_prep *S) { fpk_search_consts(d, qmax, 1e-4, &S->slack, &S->Kerr); }
__device__ void consts(int d, int64_t qmax, fpk_prep_f *S) { fpk_search_consts_f(d, qmax, 1e-4, &S->slack, &S->Kerr); }

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

template <typename GGT>
__global__ void __launch_bounds__(pfg::TL) k_build(Glob G, int n, const int *idx, const int64_t *Og, int *retry, int *nretry)
{
    __shared__ pfg::Work<MH, GGT> W;
    auto t = pfg::cg::tiled_partition<pfg::TL>(pfg::cg::this_thread_block());
    const int lane = t.thread_rank();
    const int k = blockIdx.x;
    if (k >= n) return;
    const int ip = idx ? idx[k] : k;
    PData *D = &G.pd[ip];
    if (D->status != ST_PRE) return;
    const Geo &g = G.geo[D->geo];
    const int h = g.h, d = h - 1;
    int st = pfg::tile_build<MH, GGT>(t, &W, h, D->R.Z, &Og[(size_t)ip * MH * MH], &G.Mb[g.moff]);
    if (st) {
        if (lane == 0) {
            if (st == pfg::ST_OVF64 && retry) retry[atomicAdd(nretry, 1)] = ip;
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

__global__ void k_top(Glob G, int n)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n) return;
    PData *D = &G.pd[ip];
    if (D->status != ST_OK) return;
    Ctx X = {&G, ip};
    int sd = split_depth(G, D);
    int st = sd > 0 ? run_search(&G, D, nullptr, 0, sd, emit_pre, &X)
                    : run_search(&G, D, nullptr, 0, 0, emit_pt, &X);
    if (st == -8) D->status = ST_CPU;
}

__global__ void k_search(Glob G, unsigned long long n)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    int ip = G.pre_p[k];
    PData *D = &G.pd[ip];
    if (D->status != ST_OK) return;
    Ctx X = {&G, ip};
    int st = run_search(&G, D, &G.pre[k * G.sd], split_depth(G, D), 0, emit_pt, &X);
    if (st == -8) D->status = ST_CPU;
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    snprintf(out->err, sizeof out->err, "%s (%s:%d)", cudaGetErrorString(e_), __FILE__, __LINE__); \
    goto fail; } } while (0)

template <typename T> static cudaError_t dalloc(T **p, size_t n) { return cudaMalloc((void **)p, n * sizeof(T) + 16); }

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
    const long batch = in->batch > 0 ? (long)in->batch : (1L << 19);
    const int sd = 3;
    std::vector<Geo> geos(in->n_geo);
    std::vector<int64_t> Mo, Ko, qo, po;         // outputs, grouped by batch
    Glob G = {};
    Geo *dgeo = NULL; int64_t *dk = NULL, *dm = NULL, *dps = NULL, *Og = NULL; int *dpgeo = NULL, *retry = NULL, *nretry = NULL;
    long koff = 0, moff = 0;
    unsigned long long pre_cap = (unsigned long long)std::min(batch, NP) * 64, pts_cap = 1ULL << 16;
    std::vector<int> stat;
    cudaEvent_t e0 = NULL, e1 = NULL;
    int rc = 1;

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

    CK(cudaSetDevice(in->device));
    CK(dalloc(&dgeo, geos.size())); CK(cudaMemcpy(dgeo, geos.data(), geos.size() * sizeof(Geo), cudaMemcpyHostToDevice));
    CK(dalloc(&dk, koff)); CK(cudaMemcpy(dk, in->kappa, koff * 8, cudaMemcpyHostToDevice));
    CK(dalloc(&dm, moff)); CK(cudaMemcpy(dm, in->Mbasis, moff * 8, cudaMemcpyHostToDevice));
    {
        const long nb = std::min(batch, NP);
        CK(dalloc(&dps, nb * MH)); CK(dalloc(&dpgeo, nb));
        CK(dalloc(&G.pd, nb)); CK(dalloc(&Og, nb * MH * MH));
        CK(dalloc(&retry, nb)); CK(dalloc(&nretry, 1));
        CK(dalloc(&G.npre, 1)); CK(dalloc(&G.npts, 1));
        stat.resize(nb);
    }
    G.sd = sd; G.geo = dgeo; G.kappa = dk; G.Mb = dm; G.ps = dps; G.pgeo = dpgeo;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    for (long off = 0; off < NP; ) {
        const int n = (int)std::min(batch, NP - off);
        if (!G.pre) { CK(dalloc(&G.pre, pre_cap * sd)); CK(dalloc(&G.pre_p, pre_cap)); G.pre_cap = pre_cap; }
        if (!G.pts) { CK(dalloc(&G.pts, pts_cap * 2 * MH)); CK(dalloc(&G.qs, pts_cap)); CK(dalloc(&G.pts_p, pts_cap)); G.pts_cap = pts_cap; }
        CK(cudaMemcpy(dps, &in->ps[off * MH], (size_t)n * MH * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpgeo, &in->pgeo[off], (size_t)n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(G.npre, 0, 8)); CK(cudaMemset(G.npts, 0, 8)); CK(cudaMemset(nretry, 0, 4));
        CK(cudaEventRecord(e0));
        k_pre<<<(n + 127) / 128, 128>>>(G, n, Og);
        k_build<int64_t><<<n, pfg::TL>>>(G, n, nullptr, Og, retry, nretry);
        int nre;
        CK(cudaMemcpy(&nre, nretry, 4, cudaMemcpyDeviceToHost));
        if (nre) k_build<pfl_i128><<<nre, pfg::TL>>>(G, nre, retry, Og, nullptr, nullptr);
        k_fin<<<(n + 127) / 128, 128>>>(G, n);
        k_top<<<(n + 127) / 128, 128>>>(G, n);
        unsigned long long npre;
        CK(cudaMemcpy(&npre, G.npre, 8, cudaMemcpyDeviceToHost));
        if (npre > G.pre_cap) {                        // grow, rerun the batch
            cudaFree(G.pre); cudaFree(G.pre_p); G.pre = NULL; G.pre_p = NULL;
            pre_cap = npre + npre / 4;
            continue;
        }
        if (npre) k_search<<<(unsigned)((npre + 127) / 128), 128>>>(G, npre);
        CK(cudaEventRecord(e1));
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        unsigned long long npts;
        CK(cudaMemcpy(&npts, G.npts, 8, cudaMemcpyDeviceToHost));
        if (npts > G.pts_cap) {
            cudaFree(G.pts); cudaFree(G.qs); cudaFree(G.pts_p); G.pts = NULL; G.qs = NULL; G.pts_p = NULL;
            pts_cap = npts + npts / 4;
            continue;
        }
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        out->seconds_gpu += ms / 1e3;
        CK(cudaMemcpy2D(stat.data(), 4, (char *)G.pd + offsetof(PData, status), sizeof(PData), 4, n, cudaMemcpyDeviceToHost));
        {
            std::vector<int64_t> pts(npts * 2 * MH), qs(npts);
            std::vector<int> pp(npts);
            CK(cudaMemcpy(pts.data(), G.pts, npts * 2 * MH * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(qs.data(), G.qs, npts * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(pp.data(), G.pts_p, npts * 4, cudaMemcpyDeviceToHost));
            // keep the points of p's finished here; canonical order
            std::vector<unsigned long long> idx;
            idx.reserve(npts);
            for (unsigned long long k = 0; k < npts; ++k)
                if (stat[pp[k]] == ST_OK) idx.push_back(k);
            std::sort(idx.begin(), idx.end(), [&](unsigned long long a, unsigned long long b) {
                if (pp[a] != pp[b]) return pp[a] < pp[b];
                return std::lexicographical_compare(&pts[a * 2 * MH], &pts[a * 2 * MH + 2 * MH],
                                                    &pts[b * 2 * MH], &pts[b * 2 * MH + 2 * MH]);
            });
            for (unsigned long long k : idx) {
                Mo.insert(Mo.end(), &pts[k * 2 * MH], &pts[k * 2 * MH + MH]);
                Ko.insert(Ko.end(), &pts[k * 2 * MH + MH], &pts[k * 2 * MH + 2 * MH]);
                qo.push_back(qs[k]);
                po.push_back(off + pp[k]);
            }
        }
        for (int ip = 0; ip < n; ++ip)
            out->pstat[off + ip] = (stat[ip] == ST_OK || stat[ip] == ST_EMPTY) ? 0 : 1;
        if (in->verbose)
            fprintf(stderr, "pfg: batch @%ld: %d p, %.1f ms, %llu subtrees, %llu points, %d int128-Gram redo\n",
                    off, n, ms, npre, npts, nre);
        off += n;
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
    if (rc) pfg_output_free(out);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    cudaFree(dgeo); cudaFree(dk); cudaFree(dm); cudaFree(dps); cudaFree(dpgeo); cudaFree(Og);
    cudaFree(retry); cudaFree(nretry); cudaFree(G.pd); cudaFree(G.npre); cudaFree(G.npts);
    cudaFree(G.pre); cudaFree(G.pre_p); cudaFree(G.pts); cudaFree(G.qs); cudaFree(G.pts_p);
    return rc;
}
