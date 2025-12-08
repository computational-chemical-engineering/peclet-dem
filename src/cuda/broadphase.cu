#include "ParticleSystem.cuh"
#include <cuBQL/builder/cuda.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
#include <cuda_runtime.h>
#include <iostream>

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

        // Initialize solver buffers for ghosts
        ps.d_delta_pos[dest_idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        ps.d_constraint_counts[dest_idx] = 0;
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
                                         float global_scale,
                                         cuBQL::box3f *d_aabbs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float4 p = ps.d_pos_star[idx]; // Use predicted position
  // Effective Scale
  float s = ps.d_scale[idx] * global_scale;

  // Simple bounding box for sphere of radius s * shape_radius
  // Assuming worst case shape radius ~ 0.5 (unit diameter)
  // Expand slightly for motion or shape
  float radius = 0.5f * s;
  // For cylinder: max dimension? 2.0 height -> 1.0 half height.
  // We used 1.5 * s in solver ground check.
  // Safe bound: 1.5 * s

  float bound = 1.5f * s; // Conservative

  d_aabbs[idx].upper =
      cuBQL::vec_t<float, 3>(p.x + bound, p.y + bound, p.z + bound);
  d_aabbs[idx].lower =
      cuBQL::vec_t<float, 3>(p.x - bound, p.y - bound, p.z - bound);

  if (idx == 0) {
    // printf("DEBUG: AABB[0] p=(%f, %f, %f), s=%f, bound=%f, lower=(%f), "
    //        "upper=(%f)\n",
    //        p.x, p.y, p.z, s, bound, d_aabbs[idx].lower.x,
    //        d_aabbs[idx].upper.x);
  }
}

// Global storage for AABBs (managed here or allocated per frame)
cuBQL::box3f *d_boxes = nullptr;
size_t boxes_capacity = 0;

void build_bvh(ParticleSystemData &ps, float global_scale) {
  // Re-alloc boxes if needed
  if ((size_t)ps.capacity > boxes_capacity) {
    if (d_boxes)
      cudaFree(d_boxes);
    cudaMalloc(&d_boxes, ps.capacity * sizeof(cuBQL::box3f));
    boxes_capacity = ps.capacity;
  }

  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  particles_to_aabb_kernel<<<blocks, threads>>>(ps, global_scale, d_boxes);

  // Build BVH
  using namespace cuBQL;
  BuildConfig config; // Default constructor
  gpuBuilder(ps.bvh, d_boxes, ps.num_particles, config);
}

__global__ void find_collisions_kernel(ParticleSystemData ps,
                                       float global_scale, float query_radius) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Only iterate Real particles.
  if (idx >= ps.num_real)
    return;

  float3 p = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                         ps.d_pos_star[idx].z);

  float s = ps.d_scale[idx] * global_scale;
  // Unit sphere radius 1.0 * s + margin
  float r = 2.0f * s + query_radius;

  cuBQL::box3f queryBox;
  queryBox.lower = cuBQL::vec_t<float, 3>(p.x - r, p.y - r, p.z - r);
  queryBox.upper = cuBQL::vec_t<float, 3>(p.x + r, p.y + r, p.z + r);

  auto collision_lambda = [&](int other_idx) -> int {
    if (idx == other_idx)
      return CUBQL_CONTINUE_TRAVERSAL;

    // Add to list
    int slot = atomicAdd(ps.d_potential_count, 1);
    if (slot < ps.max_potential_collisions) {
      ps.d_potential_collisions[slot] = make_int2(idx, other_idx);
      if (slot < 5)
        printf("DEBUG: Found collision: %d and %d (slot %d)\n", idx, other_idx,
               slot);
    } else {
      return CUBQL_TERMINATE_TRAVERSAL; // Buffer full
    }
    return CUBQL_CONTINUE_TRAVERSAL;
  };

  // fixedBoxQuery is a namespace inside cuBQL
  cuBQL::fixedBoxQuery::forEachPrim(collision_lambda, ps.bvh, queryBox);
}

void find_collisions(ParticleSystemData &ps, float global_scale) {
  cudaMemset(ps.d_potential_count, 0, sizeof(int));

  int threads = 128;
  int blocks = (ps.num_real + threads - 1) / threads;

  // if (global_scale > 0.9f)
  //   printf("DEBUG: Launching find_collisions_kernel blocks=%d, threads=%d, "
  //          "num_real=%d\n",
  //          blocks, threads, ps.num_real);

  find_collisions_kernel<<<blocks, threads>>>(ps, global_scale, 0.1f);
}
