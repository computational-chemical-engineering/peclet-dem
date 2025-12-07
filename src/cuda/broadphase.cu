#include "ParticleSystem.cuh"
#include <cuBQL/bvh.h>
#include <cuda_runtime.h>

// Wrapper to convert particles to AABBs
__global__ void particles_to_aabb_kernel(ParticleSystemData ps,
                                         cuBQL::box3f *boxes,
                                         float radius_expansion) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float3 p = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                         ps.d_pos_star[idx].z);
  // Assuming simple sphere radius for now (e.g., 0.5) + expansion
  float r = 0.5f + radius_expansion;

  boxes[idx].upper = cuBQL::vec_t<float, 3>(p.x + r, p.y + r, p.z + r);
  boxes[idx].lower = cuBQL::vec_t<float, 3>(p.x - r, p.y - r, p.z - r);
}

// NOTE: This is a placeholder for the actual cuBQL integration.
// We need to instantiate the cuBQL builder.
// For Phase 1, we might just output that we are building BVH.
void build_bvh(ParticleSystemData &ps) {
  // 1. Convert to AABBs
  // 2. cuBQL::bvh_build
  // 3. Store result
}
