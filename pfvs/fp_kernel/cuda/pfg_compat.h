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
// Portability layer: the GPU sources build with nvcc (CUDA) and hipcc (HIP).
// cuda* runtime calls map to hip* here; 16-lane tiles and warp-aggregated
// atomics are implemented per platform, without assuming a warp width of 32.
#pragma once

#if defined(__HIPCC__) || defined(__HIP__)
#define PFG_HIP 1
#include <hip/hip_runtime.h>
#define cudaError_t               hipError_t
#define cudaSuccess               hipSuccess
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaGetErrorString        hipGetErrorString
#define cudaGetLastError          hipGetLastError
#define cudaGetDeviceCount        hipGetDeviceCount
#define cudaSetDevice             hipSetDevice
#define cudaDeviceSynchronize     hipDeviceSynchronize
#define cudaMalloc                hipMalloc
#define cudaMemGetInfo            hipMemGetInfo
#define cudaFree                  hipFree
#define cudaMemcpy                hipMemcpy
#define cudaMemcpy2D              hipMemcpy2D
#define cudaMemset                hipMemset
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaEvent_t               hipEvent_t
#define cudaEventCreate           hipEventCreate
#define cudaEventRecord           hipEventRecord
#define cudaEventElapsedTime      hipEventElapsedTime
#define cudaEventDestroy          hipEventDestroy
#else
#define PFG_HIP 0
#include <cuda_runtime.h>
#endif

namespace pfg {

enum { TL = 16 };                  // lanes per tile (one p-vector's lattice setup)

__device__ inline int lane_id()
{
#if PFG_HIP
    return __lane_id();
#else
    return threadIdx.x & 31;
#endif
}

// A tile: TL consecutive threads of a 1D block (so TL consecutive lanes of
// one warp/wavefront). All its lanes must reach each sync/shfl/any together.
struct Tile {
    __device__ int thread_rank() const { return threadIdx.x % TL; }
#if PFG_HIP
    __device__ unsigned long long mask() const { return 0xFFFFull << (lane_id() & ~(TL - 1)); }
    // a wavefront runs in lockstep: order this tile's shared-memory accesses
    __device__ void sync() const
    {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    template <typename T> __device__ T shfl(T v, int src) const { return __shfl(v, src, TL); }
    __device__ int any(int p) const { return (__ballot(p) & mask()) != 0; }
#else
    __device__ unsigned mask() const { return 0xFFFFu << (lane_id() & ~(TL - 1)); }
    __device__ void sync() const { __syncwarp(mask()); }
    template <typename T> __device__ T shfl(T v, int src) const { return __shfl_sync(mask(), v, src, TL); }
    __device__ int any(int p) const { return __any_sync(mask(), p); }
#endif
};

// Whole-warp operations: all lanes of the warp/wavefront must take part.
// wave_size(): 32 (CUDA), 32 or 64 (HIP, per target and build flags).
#if PFG_HIP
__device__ inline int wave_size() { return warpSize; }
__device__ inline void wave_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
__device__ inline int wave_shfl_up(int v, int o) { return __shfl_up(v, o); }
__device__ inline int wave_shfl(int v, int src) { return __shfl(v, src); }
#else
__device__ inline int wave_size() { return 32; }
__device__ inline void wave_sync() { __syncwarp(); }
__device__ inline int wave_shfl_up(int v, int o) { return __shfl_up_sync(0xffffffffu, v, o); }
__device__ inline int wave_shfl(int v, int src) { return __shfl_sync(0xffffffffu, v, src); }
#endif

// warp-aggregated slot reservation on a global counter (divergence-safe):
// one atomic per warp instead of per thread
__device__ inline unsigned long long agg_inc(unsigned long long *ctr)
{
    const int lane = lane_id();
#if PFG_HIP
    unsigned long long mask = __ballot(1);                 // the active lanes
    int leader = __ffsll((long long)mask) - 1;
    unsigned rank = __popcll(mask & ((1ull << lane) - 1));
    unsigned long long base = 0;
    if (lane == leader) base = atomicAdd(ctr, (unsigned long long)__popcll(mask));
    base = __shfl(base, leader);
#else
    unsigned mask = __activemask();
    int leader = __ffs(mask) - 1;
    unsigned rank = __popc(mask & ((1u << lane) - 1));
    unsigned long long base = 0;
    if (lane == leader) base = atomicAdd(ctr, (unsigned long long)__popc(mask));
    base = __shfl_sync(mask, base, leader);
#endif
    return base + rank;
}

}  // namespace pfg
