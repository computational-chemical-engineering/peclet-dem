#include "ParticleSystem.cuh"
#include "memory_utils.cuh" // For CUDA_CHECK
#include "periodicity.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dem {

// ------------------------------------------------------------------
// Kernel 1: Find Particles Near Boundary (Candidates)
// ------------------------------------------------------------------
// This kernel traverses the LOCAL BVH to find particles that overlap
// with the "skin" region of the periodic boundary.
// ------------------------------------------------------------------
// Kernel 1: Find Particles Near Boundary (Candidates)
// ------------------------------------------------------------------
// This kernel traverses the LOCAL BVH to find particles that overlap
// with the "skin" region of the periodic boundary.
__global__ void find_ghost_candidates_kernel(ParticleSystemData ps,
                                             PeriodicConfig config,
                                             int *d_candidates,
                                             int *d_candidate_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // (Comments removed for brevity)

  if (idx >= ps.num_real)
    return;

  float4 p = ps.d_pos[idx];  // Read-Only State
  float r = ps.d_scale[idx]; // Approx radius

  // Boundary Check
  BoundarySkinQuery query;
  query.inner_min = make_float3(config.min.x + config.skin_width,
                                config.min.y + config.skin_width,
                                config.min.z + config.skin_width);
  query.inner_max = make_float3(config.max.x - config.skin_width,
                                config.max.y - config.skin_width,
                                config.max.z - config.skin_width);

  if (query.is_candidate(make_float3(p.x, p.y, p.z), r)) {
    int out_idx = atomicAdd(d_candidate_count, 1);
    d_candidates[out_idx] = idx;
  }
}

// ------------------------------------------------------------------
// Kernel 2: Generate Ghost Copies
// ------------------------------------------------------------------
__global__ void generate_ghosts_bitmask_kernel(int *d_candidates,
                                               int *d_candidate_count,
                                               ParticleSystemData ps,
                                               PeriodicConfig config) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int count = *d_candidate_count;
  if (idx >= count)
    return;

  int real_idx = d_candidates[idx];
  float4 p_val = ps.d_pos[real_idx];
  float3 p = make_float3(p_val.x, p_val.y, p_val.z);

  float skin = config.skin_width;
  float3 min_b = config.min;
  float3 max_b = config.max;

  int sx = (p.x < min_b.x + skin) ? 1 : ((p.x > max_b.x - skin) ? -1 : 0);
  int sy = (p.y < min_b.y + skin) ? 1 : ((p.y > max_b.y - skin) ? -1 : 0);
  int sz = (p.z < min_b.z + skin) ? 1 : ((p.z > max_b.z - skin) ? -1 : 0);

  int active_x[2] = {0, sx};
  int count_x = (sx == 0) ? 1 : 2;

  int active_y[2] = {0, sy};
  int count_y = (sy == 0) ? 1 : 2;

  int active_z[2] = {0, sz};
  int count_z = (sz == 0) ? 1 : 2;

  for (int i = 0; i < count_x; ++i) {
    for (int j = 0; j < count_y; ++j) {
      for (int k = 0; k < count_z; ++k) {
        int ix = active_x[i];
        int iy = active_y[j];
        int iz = active_z[k];

        if (ix == 0 && iy == 0 && iz == 0)
          continue; // Skip self

        // Add Ghost
        int ghost_idx = atomicAdd(ps.d_top_ghost, 1);
        if (ghost_idx >= ps.capacity) {
          atomicSub(ps.d_top_ghost, 1);
          return;
        }

        float3 shift = make_float3(ix * config.size.x, iy * config.size.y,
                                   iz * config.size.z);

        // Copy & Shift
        // Use predicted position for ghosts to align with solver state
        float4 p_pred = ps.d_pos_pred[real_idx];

        ps.d_pos[ghost_idx] = make_float4(p_val.x + shift.x, p_val.y + shift.y,
                                          p_val.z + shift.z, p_val.w);

        // Predictors (CRITICAL: Must match Solver time)
        ps.d_pos_pred[ghost_idx] =
            make_float4(p_pred.x + shift.x, p_pred.y + shift.y,
                        p_pred.z + shift.z, p_pred.w);

        // Copy others
        ps.d_vel[ghost_idx] = ps.d_vel[real_idx];
        ps.d_vel_pred[ghost_idx] = ps.d_vel_pred[real_idx];

        ps.d_quat[ghost_idx] = ps.d_quat[real_idx];
        ps.d_quat_pred[ghost_idx] = ps.d_quat_pred[real_idx];

        ps.d_ang_vel[ghost_idx] = ps.d_ang_vel[real_idx];
        ps.d_ang_vel_pred[ghost_idx] = ps.d_ang_vel_pred[real_idx];

        ps.d_scale[ghost_idx] = ps.d_scale[real_idx];
        ps.d_shape_ids[ghost_idx] = ps.d_shape_ids[real_idx];

        // Map Ghost to Real ID
        ps.d_real_indices[ghost_idx] = real_idx;

        // Flag as Ghost
        ps.d_vel[ghost_idx].w = 1.0f;
        ps.d_vel_pred[ghost_idx].w = 1.0f;
      }
    }
  }
}

// ... Host Wrappers ...

void launch_find_ghost_candidates(ParticleSystemData ps, PeriodicConfig config,
                                  int *d_candidates, int *d_candidate_count) {
  int threads = 256;
  int blocks = (ps.num_real + threads - 1) / threads;
  find_ghost_candidates_kernel<<<blocks, threads>>>(ps, config, d_candidates,
                                                    d_candidate_count);
  CUDA_CHECK(cudaGetLastError());
}

void launch_generate_ghosts_bitmask(int *d_candidates, int *d_candidate_count,
                                    ParticleSystemData ps,
                                    PeriodicConfig config) {
  int count;
  CUDA_CHECK(cudaMemcpy(&count, d_candidate_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (count > 0) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    generate_ghosts_bitmask_kernel<<<blocks, threads>>>(
        d_candidates, d_candidate_count, ps, config);
    CUDA_CHECK(cudaGetLastError());
  }
}

} // namespace dem
