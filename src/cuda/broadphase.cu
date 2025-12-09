#include "ParticleSystem.cuh"
#include "memory_utils.cuh"
#include <cuBQL/builder/cuda.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
#include <cuda_runtime.h>
#include <iostream>

// Wrapper to convert particles to AABBs
__global__ void particles_to_aabb_kernel(ParticleSystemData ps,
                                         float global_scale, float margin,
                                         cuBQL::box3f *d_aabbs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float4 p = ps.d_pos_pred[idx]; // Use predicted position
  // Effective Scale
  float s = ps.d_scale[idx] * global_scale;

  // Simple bounding box for sphere of radius s * shape_radius (assuming unit
  // sphere r=1.0)
  float radius = 1.0f * s;

  // AABB must enclose the particle + margin
  float bound = radius + margin;

  d_aabbs[idx].upper =
      cuBQL::vec_t<float, 3>(p.x + bound, p.y + bound, p.z + bound);
  d_aabbs[idx].lower =
      cuBQL::vec_t<float, 3>(p.x - bound, p.y - bound, p.z - bound);
}

// Global storage for AABBs (managed here or allocated per frame)
cuBQL::box3f *d_boxes = nullptr;
size_t boxes_capacity = 0;

void build_bvh(ParticleSystemData &ps, float global_scale) {
  // Re-alloc boxes if needed
  if ((size_t)ps.capacity > boxes_capacity) {
    if (d_boxes)
      cudaFree(d_boxes);
    CUDA_CHECK(cudaMalloc(&d_boxes, ps.capacity * sizeof(cuBQL::box3f)));
    boxes_capacity = ps.capacity;
  }

  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  // Margin for BVH build: should match narrowphase speculative margin
  // Hardcoded 0.05f * global_scale for now, or pass it?
  // Safety Margin should be a parameter.
  float safety_margin = 0.1f * global_scale;

  particles_to_aabb_kernel<<<blocks, threads>>>(ps, global_scale, safety_margin,
                                                d_boxes);

  // Build BVH
  using namespace cuBQL;
  BuildConfig config; // Default constructor
  gpuBuilder(ps.bvh, d_boxes, ps.num_particles, config);
}

__global__ void find_collisions_kernel(ParticleSystemData ps,
                                       float global_scale,
                                       float safety_margin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Only iterate Real particles.
  if (idx >= ps.num_real)
    return;

  float3 p = make_float3(ps.d_pos_pred[idx].x, ps.d_pos_pred[idx].y,
                         ps.d_pos_pred[idx].z);

  float s = ps.d_scale[idx] * global_scale;
  float radius = 1.0f * s; // Assuming sphere 1.0

  // Search Radius: My Radius + Max Neighbor Radius + Margin
  // Conservative estimate: 2.0 * My Radius + Margin?
  // Let's use 2.0 * s + margin (Diameter + Margin)
  float r_query = 1.0f * s + safety_margin; // Just radius + margin?
  // NO. Query box is P +/- Size.
  // Intersection with AABB (Center +/- (R_other + Margin)).
  // Overlap if |P - P_other| < R_self + R_other + Margin.
  // Query Box half-width W. AABB half-width H.
  // Intersect if |P - P_other| < W + H.
  // We want W + H >= R_self + R_other + Margin.
  // H = R_other + Margin.
  // So W >= R_self.
  // So Query Box Half-Size = R_self.
  // Wait, cuBQL query is conservative?
  // Let's be safe: 2.0 * R_self.

  float box_r = 1.0f * s + safety_margin; // 2x radius.

  cuBQL::box3f queryBox;
  queryBox.lower =
      cuBQL::vec_t<float, 3>(p.x - box_r, p.y - box_r, p.z - box_r);
  queryBox.upper =
      cuBQL::vec_t<float, 3>(p.x + box_r, p.y + box_r, p.z + box_r);

  auto collision_lambda = [&](int other_idx) -> int {
    if (idx == other_idx)
      return CUBQL_CONTINUE_TRAVERSAL;

    // Add to list
    int slot = atomicAdd(ps.d_potential_count, 1);
    if (slot < ps.max_potential_collisions) {
      ps.d_potential_collisions[slot] = make_int2(idx, other_idx);
    } else {
      return CUBQL_TERMINATE_TRAVERSAL; // Buffer full
    }
    return CUBQL_CONTINUE_TRAVERSAL;
  };

  cuBQL::fixedBoxQuery::forEachPrim(collision_lambda, ps.bvh, queryBox);
}

void find_collisions(ParticleSystemData ps, float global_scale) {
  CUDA_CHECK(cudaMemset(ps.d_potential_count, 0, sizeof(int)));

  int threads = 128;
  int blocks = (ps.num_real + threads - 1) / threads;

  float safety_margin = 0.1f * global_scale;
  find_collisions_kernel<<<blocks, threads>>>(ps, global_scale, safety_margin);
  CUDA_CHECK(cudaGetLastError());
}
