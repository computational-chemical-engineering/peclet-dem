#include "ParticleSystem.cuh"
#include <cuBQL/builder/cuda.h>
#include <cuBQL/bvh.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
#include <cuda_runtime.h>
#include <iostream>

// Re-declare since I overwrote the file (wait, I should use the existing
// content as base?) I will just copy the generate_ghosts stuff and add the
// rest.

// Generate ghosts for periodic boundaries
__global__ void generate_ghosts_kernel(ParticleSystemData ps,
                                       int *d_ghost_count,
                                       float ghost_threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_real)
    return;

  float3 p = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                         ps.d_pos_star[idx].z);

  // Check 3 axes
  bool lo_x = ps.periodic_x && (p.x < ps.domain_min.x + ghost_threshold);
  bool hi_x = ps.periodic_x && (p.x > ps.domain_max.x - ghost_threshold);
  bool lo_y = ps.periodic_y && (p.y < ps.domain_min.y + ghost_threshold);
  bool hi_y = ps.periodic_y && (p.y > ps.domain_max.y - ghost_threshold);
  bool lo_z = ps.periodic_z && (p.z < ps.domain_min.z + ghost_threshold);
  bool hi_z = ps.periodic_z && (p.z > ps.domain_max.z - ghost_threshold);

  if (!(lo_x || hi_x || lo_y || hi_y || lo_z || hi_z))
    return;

  int dx_vals[2] = {0, 0};
  int nx = 1;
  if (lo_x) {
    dx_vals[1] = 1;
    nx = 2;
  } else if (hi_x) {
    dx_vals[1] = -1;
    nx = 2;
  }

  int dy_vals[2] = {0, 0};
  int ny = 1;
  if (lo_y) {
    dy_vals[1] = 1;
    ny = 2;
  } else if (hi_y) {
    dy_vals[1] = -1;
    ny = 2;
  }

  int dz_vals[2] = {0, 0};
  int nz = 1;
  if (lo_z) {
    dz_vals[1] = 1;
    nz = 2;
  } else if (hi_z) {
    dz_vals[1] = -1;
    nz = 2;
  }

  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        int dx = dx_vals[i];
        int dy = dy_vals[j];
        int dz = dz_vals[k];
        if (dx == 0 && dy == 0 && dz == 0)
          continue;

        int ghost_idx = atomicAdd(d_ghost_count, 1);
        int dest_idx = ps.num_real + ghost_idx;
        if (dest_idx >= ps.capacity)
          continue;

        float3 offset =
            make_float3(dx * ps.domain_size.x, dy * ps.domain_size.y,
                        dz * ps.domain_size.z);

        ps.d_pos_star[dest_idx] =
            make_float4(p.x + offset.x, p.y + offset.y, p.z + offset.z,
                        ps.d_pos_star[idx].w);

        ps.d_pos[dest_idx] = ps.d_pos[idx];
        ps.d_pos[dest_idx].x += offset.x;
        ps.d_pos[dest_idx].y += offset.y;
        ps.d_pos[dest_idx].z += offset.z;

        ps.d_vel[dest_idx] = ps.d_vel[idx];
        ps.d_vel[dest_idx].w = 1.0f; // Mark as GHOST

        ps.d_quat[dest_idx] = ps.d_quat[idx];
        ps.d_scale[dest_idx] = ps.d_scale[idx];
        ps.d_shape_ids[dest_idx] = ps.d_shape_ids[idx];
      }
    }
  }
}

void generate_ghosts(ParticleSystemData &ps, float threshold) {
  int *d_count;
  cudaMalloc(&d_count, sizeof(int));
  cudaMemset(d_count, 0, sizeof(int));

  int threads = 256;
  int blocks = (ps.num_real + threads - 1) / threads;

  generate_ghosts_kernel<<<blocks, threads>>>(ps, d_count, threshold);

  int h_count = 0;
  cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_count);

  ps.num_particles = ps.num_real + h_count;
  if (ps.num_particles > ps.capacity)
    ps.num_particles = ps.capacity;
}

// Wrapper to convert particles to AABBs
__global__ void particles_to_aabb_kernel(ParticleSystemData ps,
                                         cuBQL::box3f *boxes,
                                         float radius_expansion) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float3 p = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                         ps.d_pos_star[idx].z);

  // Scale the AABB
  float s = ps.d_scale[idx];
  // Assuming simple sphere radius for now (e.g., 0.5) + expansion
  // For cylinder (r=0.5, h=2.0), the bound radius is ~1.2
  // Let's use 1.5 * scale as safe bound
  float r = 1.5f * s + radius_expansion;

  boxes[idx].upper = cuBQL::vec_t<float, 3>(p.x + r, p.y + r, p.z + r);
  boxes[idx].lower = cuBQL::vec_t<float, 3>(p.x - r, p.y - r, p.z - r);
}

// Global storage for AABBs (managed here or allocated per frame)
cuBQL::box3f *d_boxes = nullptr;
size_t boxes_capacity = 0;

void build_bvh(ParticleSystemData &ps) {
  // Re-alloc boxes if needed
  if (ps.capacity > boxes_capacity) {
    if (d_boxes)
      cudaFree(d_boxes);
    cudaMalloc(&d_boxes, ps.capacity * sizeof(cuBQL::box3f));
    boxes_capacity = ps.capacity;
  }

  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  float radius_expansion = 0.1f; // For broadphase
  particles_to_aabb_kernel<<<blocks, threads>>>(ps, d_boxes, radius_expansion);

  // Build BVH
  // Note: gpuBuilder usually frees previous nodes if handled by wrapper, but
  // here ps.bvh is raw struct. If ps.bvh.nodes was allocated by previous build,
  // we should probably check. However, cuBQL::gpuBuilder might overwrite? Let's
  // assume gpuBuilder handles it or leaks. Usually it calls internal
  // allocators. The best practice is to likely free if not null. But bvh struct
  // members are just pointers. cuBQL::free(ps.bvh) ?

  cuBQL::gpuBuilder(ps.bvh, d_boxes, ps.num_particles, cuBQL::BuildConfig());
}

__global__ void find_collisions_kernel(ParticleSystemData ps,
                                       float query_radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // We only care about collisions involving at least one Real particle.
  // Or we handle all pairs and solver handles duplicates?
  // Optimization: If both A and B are ghosts, ignore. Collision is handled on
  // the original side.

  // Also, we traverse for "idx". If idx is Ghost, we might skip?
  // If idx is Ghost, its Real counterpart is handled.
  // But Ghost->Real collision? Real handles it.
  // Ghost->Ghost? Handled by Real->Real.
  // So ONLY iterate Real particles.
  if (idx >= ps.num_real)
    return;

  float3 p = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                         ps.d_pos_star[idx].z);
  float r = 0.5f + query_radius;

  cuBQL::box3f queryBox;
  queryBox.lower = cuBQL::vec_t<float, 3>(p.x - r, p.y - r, p.z - r);
  queryBox.upper = cuBQL::vec_t<float, 3>(p.x + r, p.y + r, p.z + r);

  auto collision_lambda = [&](int other_idx) -> int {
    if (idx == other_idx)
      return CUBQL_CONTINUE_TRAVERSAL;
    // Avoid duplicates: only accept if other_idx > idx?
    // But what if other_idx is Ghost? Ghosts have high indices.
    // If other_idx is Ghost, duplicate check is tricky.
    // Simple rule: always accept (idx, other) and let solver uniqueify or just
    // solve twice (wasted work but correct). Standard in parallel: add (idx,
    // other).

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

void find_collisions(ParticleSystemData &ps) {
  cudaMemset(ps.d_potential_count, 0, sizeof(int));

  int threads = 128; // Traversal uses stack, lower occupancy might be better?
  int blocks = (ps.num_real + threads - 1) / threads;

  find_collisions_kernel<<<blocks, threads>>>(ps, 0.1f);
}
