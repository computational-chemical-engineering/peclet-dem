/// @file
/// @brief dem — fused colour sweeps: one persistent kernel per sweep instead of one kernel
/// launch per colour.
///
/// The impulse step is host-submission-bound (measured ~3,300 kernel launches per step at 25k;
/// the CUDA-graph pass reduced the replay cost but the capture still re-emits ~200 kernels per
/// captured iteration). The colour-to-colour sequencing is the Gauss–Seidel dependency, so the
/// per-colour launches can be collapsed into ONE kernel that iterates the colours device-side
/// with a grid-wide barrier between them. Colour classes are body-disjoint, each work item is
/// computed by exactly one thread with the same per-item math as the launch path, and the
/// barrier reproduces the launch path's colour ordering — the fused sweep is bit-identical.
///
/// The barrier is a software grid barrier (monotone arrival counter, reset per launch), NOT
/// cooperative-groups grid sync: it needs no -rdc compilation, no cooperative-launch driver
/// support, and captures into CUDA graphs like any plain kernel. Deadlock-freedom comes from
/// launching at most cudaOccupancyMaxActiveBlocksPerMultiprocessor blocks — every launched
/// block is co-resident, exactly the invariant cooperative launch enforces.
///
/// CUDA-only; every other backend (and set_fused_sweeps('off')) keeps the per-colour launch path.
#ifndef DEM_SOLVER_FUSED_HPP
#define DEM_SOLVER_FUSED_HPP

#include <algorithm>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <vector>

#include "contact_preprocessing.hpp"  // CpExec/CpMem

namespace peclet::dem {

/// Device-side context for a fused colour sweep: the colour offsets (numColors+1, uploaded from
/// buildColorBucketsKokkos's host vector), the barrier counter, and the largest colour bucket
/// (grid sizing). maxBucket == 0 <=> inactive (fall back to per-colour launches).
struct FusedSweepCtx {
  Kokkos::View<const int*, CpMem> offsDev;
  Kokkos::View<unsigned*, CpMem> bar;
  int maxBucket = 0;
};

/// Device-side iteration loop: run up to maxIters sweeps of the colour classes inside ONE
/// kernel, evaluating the adaptive stop on-device (the same residual the host loop read back
/// between launches — the readback round trip and the CUDA-graph capture disappear entirely).
struct FusedLoopSpec {
  int maxIters;
  float tol;
  bool strictLess;  // position loop breaks on res < tol; velocity loops on res <= tol
};

/// Fused-sweep policy. `mode` is Particles::fusedSweeps (set_fused_sweeps): -1 auto, 0 off
/// (launch path everywhere), 1 on (fused everywhere it applies). Auto = fused only where
/// CUDA-graph replay is unavailable — the distributed step (ghost syncs inside the loops forbid
/// capture) and set_cuda_graphs(False) runs. Measured on the 25k/100k Dosta beds (RTX 5080):
/// graph replay beats the fused kernels by 2-4% on the solo GPU (pipelined graph-node
/// transitions are cheaper than software grid barriers once the step is GPU-bound), while
/// capture-less paths pay the full per-launch submission storm that fusion removes (25k
/// multilevel: 17.4 -> 16.2 ms/step with graphs off). Either way the arithmetic is identical.
inline bool demFusedWanted(bool graphReplayAvailable, int mode) {
#ifdef KOKKOS_ENABLE_CUDA
  return mode < 0 ? !graphReplayAvailable : mode == 1;
#else
  (void)graphReplayAvailable;
  (void)mode;
  return false;
#endif
}

#ifdef KOKKOS_ENABLE_CUDA

/// Software grid barrier #k of this launch (k must be identical across all blocks — uniform
/// control flow only, and every block must reach every executed barrier in the same order).
/// Monotone arrival counter: barrier k completes once (k+1)*gridDim.x arrivals are visible;
/// the counter is memset to 0 before each launch. A stale (too-small) volatile read only spins
/// one more poll — arrivals can never be over-counted, so the read-before-arrive hazard of
/// generation/sense-reversal barriers does not exist. Tight spin, no backoff: at most
/// ~occupancy-max blocks contend, and the sweep phases are microseconds long — wakeup latency
/// costs more than the spin traffic. (A contention-free per-block-flag variant with a block-0
/// scan was measured SLOWER: the extra arrive -> scan -> publish hop adds an L2 round trip to
/// the release latency.)
__device__ inline void demGridBarrier(unsigned* count, unsigned k) {
  __syncthreads();  // this block's work items are done
  if (threadIdx.x == 0) {
    __threadfence();  // publish this block's writes before declaring arrival
    atomicAdd(count, 1u);
    const unsigned target = (k + 1u) * gridDim.x;
    while (*reinterpret_cast<volatile unsigned*>(count) < target) {
    }
    __threadfence();  // acquire the other blocks' writes
  }
  __syncthreads();  // release the block into the next colour
}

/// Barrier scratch layout: 8 unsigneds (one 32B sector) per block + the release word. The
/// pooled view bounds the launchable grid; memset the used span before every fused launch.
inline constexpr int kFusedBarWords = 32769;  // 4096 blocks * 8 + 1

/// One fused colour sweep: iterate the colour classes in order, grid-stride within each class,
/// grid barrier between classes. Sweep must expose solveOne(int idx) with the launch-path
/// semantics (concurrent-safe within a body-disjoint colour class). Empty classes skip their
/// barrier (uniform: every block reads the same offsets).
template <class Sweep>
__global__ void demFusedColorSweepK(Sweep f, Kokkos::View<const int*, CpMem> perm,
                                    Kokkos::View<const int*, CpMem> offs, int numColors,
                                    unsigned* bar) {
  const int stride = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned k = 0;
  for (int c = 0; c < numColors; ++c) {
    const int b = offs(c), e = offs(c + 1);
    if (b == e)
      continue;
    for (int i = b + tid; i < e; i += stride)
      f.solveOne(perm(i));
    if (c + 1 < numColors)
      demGridBarrier(bar, k++);
  }
}

/// A whole adaptive iteration loop as one kernel: per iteration, zero the residual, run the
/// colour sweep (grid barrier between classes), then read the residual back POST-barrier (the
/// barrier's acquire fence makes every atomic_max visible) and take the same break the host
/// loop took. The trailing barrier keeps a slow block's residual read ahead of the next
/// iteration's zero. Bit-identical to the host loop: same sweeps, same residual, same stop.
template <class Sweep>
__global__ void demFusedSweepLoopK(Sweep f, Kokkos::View<const int*, CpMem> perm,
                                   Kokkos::View<const int*, CpMem> offs, int numColors,
                                   int maxIters, float tol, bool strictLess, float* res,
                                   unsigned* bar) {
  const int stride = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned k = 0;
  for (int it = 0; it < maxIters; ++it) {
    if (tid == 0)
      *res = 0.0f;
    demGridBarrier(bar, k++);
    for (int c = 0; c < numColors; ++c) {
      const int b = offs(c), e = offs(c + 1);
      if (b == e)
        continue;
      for (int i = b + tid; i < e; i += stride)
        f.solveOne(perm(i));
      demGridBarrier(bar, k++);
    }
    const float r = *reinterpret_cast<volatile float*>(res);
    if (strictLess ? (r < tol) : (r <= tol))
      break;
    demGridBarrier(bar, k++);  // every block has read res before the next zero
  }
}

inline constexpr int kFusedBlock = 256;

/// Largest launchable co-resident grid for `kernel` at kFusedBlock threads (cached per kernel:
/// the barrier deadlocks if any launched block is not resident, so this bound is the safety
/// contract). Returns 0 when the query fails — caller falls back to per-colour launches.
template <class Kernel>
inline int demFusedMaxGrid(Kernel kernel) {
  static const int maxGrid = [kernel] {
    int perSm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kernel, kFusedBlock, 0) !=
        cudaSuccess) {
      (void)cudaGetLastError();
      return 0;
    }
    int dev = 0, sm = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
      (void)cudaGetLastError();
      return 0;
    }
    return perSm * sm;
  }();
  return maxGrid;
}

/// Launch one fused colour sweep on the space's stream (barrier reset + kernel — both plain
/// stream ops, so the pair captures into CUDA graphs unchanged). Returns false when the fused
/// path cannot run (no occupancy info) — caller falls back.
template <class Sweep>
inline bool demLaunchFusedColorSweep(CpExec& space, const Sweep& f,
                                     Kokkos::View<const int*, CpMem> perm, const FusedSweepCtx& ctx,
                                     int numColors) {
  const int maxGrid = std::min(demFusedMaxGrid(demFusedColorSweepK<Sweep>),
                               (static_cast<int>(ctx.bar.extent(0)) - 1) / 8);
  if (maxGrid <= 0 || numColors <= 0 || ctx.maxBucket <= 0)
    return false;
  const int want = (ctx.maxBucket + kFusedBlock - 1) / kFusedBlock;
  const int grid = want < maxGrid ? want : maxGrid;
  cudaStream_t str = space.cuda_stream();
  cudaMemsetAsync(ctx.bar.data(), 0, (static_cast<std::size_t>(grid) * 8 + 1) * sizeof(unsigned),
                  str);
  demFusedColorSweepK<Sweep>
      <<<grid, kFusedBlock, 0, str>>>(f, perm, ctx.offsDev, numColors, ctx.bar.data());
  return true;
}

/// Launch a whole device-side iteration loop (see demFusedSweepLoopK). res must be the SAME
/// device scalar the host loop read for its adaptive stop — the final value stays there for
/// whatever the host reads after the loop (e.g. the stabilization trigger).
template <class Sweep>
inline bool demLaunchFusedSweepLoop(CpExec& space, const Sweep& f,
                                    Kokkos::View<const int*, CpMem> perm, const FusedSweepCtx& ctx,
                                    int numColors, const FusedLoopSpec& spec, float* res) {
  const int maxGrid = std::min(demFusedMaxGrid(demFusedSweepLoopK<Sweep>),
                               (static_cast<int>(ctx.bar.extent(0)) - 1) / 8);
  if (maxGrid <= 0 || numColors <= 0 || ctx.maxBucket <= 0 || res == nullptr)
    return false;
  const int want = (ctx.maxBucket + kFusedBlock - 1) / kFusedBlock;
  const int grid = want < maxGrid ? want : maxGrid;
  cudaStream_t str = space.cuda_stream();
  cudaMemsetAsync(ctx.bar.data(), 0, (static_cast<std::size_t>(grid) * 8 + 1) * sizeof(unsigned),
                  str);
  demFusedSweepLoopK<Sweep><<<grid, kFusedBlock, 0, str>>>(f, perm, ctx.offsDev, numColors,
                                                           spec.maxIters, spec.tol, spec.strictLess,
                                                           res, ctx.bar.data());
  return true;
}

#endif  // KOKKOS_ENABLE_CUDA

/// Fill a FusedSweepCtx from buildColorBucketsKokkos's host offsets: async-upload them into the
/// pooled device view and record the largest bucket. The caller owns the policy decision
/// (demFusedWanted); no-CUDA / empty / oversize => inactive ctx.
inline FusedSweepCtx demMakeFusedCtx(CpExec& space, const std::vector<int>& offs,
                                     Kokkos::View<int*, CpMem> offsDev,
                                     Kokkos::View<unsigned*, CpMem> bar) {
  FusedSweepCtx ctx;
#ifdef KOKKOS_ENABLE_CUDA
  if (offs.size() < 2 || offs.size() > offsDev.extent(0) + static_cast<std::size_t>(0))
    return ctx;
  int maxBucket = 0;
  for (std::size_t c = 0; c + 1 < offs.size(); ++c)
    maxBucket = std::max(maxBucket, offs[c + 1] - offs[c]);
  if (maxBucket <= 0)
    return ctx;
  const Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(offs.data(),
                                                                               offs.size());
  auto d = Kokkos::subview(offsDev, Kokkos::pair<std::size_t, std::size_t>(0, offs.size()));
  Kokkos::deep_copy(space, d, h);
  ctx.offsDev = Kokkos::subview(Kokkos::View<const int*, CpMem>(offsDev),
                                Kokkos::pair<std::size_t, std::size_t>(0, offs.size()));
  ctx.bar = bar;
  ctx.maxBucket = maxBucket;
#else
  (void)space;
  (void)offs;
  (void)offsDev;
  (void)bar;
#endif
  return ctx;
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_FUSED_HPP
