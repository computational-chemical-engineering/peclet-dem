#include "ParticleSystem.cuh"
#include <cuda_runtime.h>

__global__ void solve_constraints_kernel(ParticleSystemData ps, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Simple Ground Constraint
  float ground_y = ps.domain_min.y;
  float4 p = ps.d_pos_star[idx];
  float r = 0.5f; // Radius

  if (p.y - r < 0.0f) { // Ground at 0 for simplicity, or ps.domain_min.y
    float C = p.y - r - (-10.0f); // Ground at -10
    if (C < 0) {
      float3 grad = make_float3(0, 1, 0);
      float lambda = -C; // Simplified XPBD
      ps.d_delta_pos[idx].x += lambda * grad.x;
      ps.d_delta_pos[idx].y += lambda * grad.y;
      ps.d_delta_pos[idx].z += lambda * grad.z;
      ps.d_constraint_counts[idx]++;
    }
  }
}

void launch_solver(ParticleSystemData &ps, float dt) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  solve_constraints_kernel<<<blocks, threads>>>(ps, dt);
  cudaDeviceSynchronize();
}
