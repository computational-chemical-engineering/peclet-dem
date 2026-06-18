// packing-gpu — broad-phase backed by ArborX (replaces the CUDA-only cuBQL implementation).
//
// Keeps the build_bvh / find_collisions API that simulation.cpp calls, but the BVH + queries now run
// through ArborX (dem::findCollisionsArborX), so cuBQL is no longer a dependency. Compiled as a CXX
// TU (Kokkos routes it to nvcc via its launch compiler). The particle SoA stays as raw float4*/int2*
// device arrays here; we wrap them as unmanaged Kokkos Views (and pack xyz+radius into scratch) to
// feed the portable broad-phase. Kokkos is initialized lazily and not finalized (only temporary
// Views are created here, so there is no teardown-ordering hazard).
#include "ParticleSystem.cuh"

#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>

#include "broadphase_arborx.hpp"

namespace {
float* g_pos3 = nullptr;  // packed predicted xyz (float3 layout = [N][3])
float* g_rad = nullptr;   // effective radius per particle
int g_cap = 0;

void ensureScratch(int capacity) {
  if (capacity <= g_cap) return;
  if (g_pos3) cudaFree(g_pos3);
  if (g_rad) cudaFree(g_rad);
  cudaMalloc(&g_pos3, static_cast<size_t>(capacity) * 3 * sizeof(float));
  cudaMalloc(&g_rad, static_cast<size_t>(capacity) * sizeof(float));
  g_cap = capacity;
}
}  // namespace

// The BVH is built inside find_collisions (ArborX), so this is a no-op kept for API compatibility.
void build_bvh(ParticleSystemData& ps, float global_scale) {
  (void)ps;
  (void)global_scale;
}

void find_collisions(ParticleSystemData ps, float global_scale) {
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  ensureScratch(ps.capacity);

  using Mem = dem::BpMem;
  using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  Kokkos::View<float* [3], Mem, Unmanaged> pos3(g_pos3, ps.capacity);
  Kokkos::View<float*, Mem, Unmanaged> rad(g_rad, ps.capacity);

  // Pack predicted xyz + effective radius from the raw float4 SoA (device lambda reads float4*).
  float4* pp = ps.d_pos_pred;
  float* sc = ps.d_scale;
  const float gs = global_scale;
  Kokkos::parallel_for(
      "dem::bp::pack", Kokkos::RangePolicy<dem::BpExec>(0, ps.num_particles), KOKKOS_LAMBDA(int i) {
        const float4 p = pp[i];
        pos3(i, 0) = p.x; pos3(i, 1) = p.y; pos3(i, 2) = p.z;
        rad(i) = sc[i] * gs;
      });

  // Wrap the existing output buffers (int2[N] == int[N][2]; the atomic counter scalar).
  Kokkos::View<int* [2], Mem, Unmanaged> pairs(reinterpret_cast<int*>(ps.d_potential_collisions),
                                               ps.max_potential_collisions);
  Kokkos::View<int, Mem, Unmanaged> count(ps.d_potential_count);

  const float margin = 0.1f * global_scale;
  dem::findCollisionsArborX(pos3, rad, ps.num_particles, ps.num_real, margin, pairs, count);
}
