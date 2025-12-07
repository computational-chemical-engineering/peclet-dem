#include "ParticleSystem.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void predict_position_kernel(ParticleSystemData ps, float dt,
                                        float3 gravity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Load active only
  float4 vel_w = ps.d_vel[idx];
  if (vel_w.w > 0.5f)
    return; // Ghost or inactive

  float3 pos = make_float3(ps.d_pos[idx].x, ps.d_pos[idx].y, ps.d_pos[idx].z);
  float3 vel = make_float3(vel_w.x, vel_w.y, vel_w.z);
  float inv_mass = ps.d_pos[idx].w;

  // Apply gravity
  if (inv_mass > 0.0f) {
    vel.x += gravity.x * dt;
    vel.y += gravity.y * dt;
    vel.z += gravity.z * dt;
  }

  // Predict
  float3 pos_star =
      make_float3(pos.x + vel.x * dt, pos.y + vel.y * dt, pos.z + vel.z * dt);

  // Store
  ps.d_pos_star[idx] =
      make_float4(pos_star.x, pos_star.y, pos_star.z, inv_mass);
  // Prediction for orientation can be added here
}

__global__ void reset_deltas_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  ps.d_delta_pos[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  ps.d_constraint_counts[idx] = 0;
}

__global__ void update_velocity_kernel(ParticleSystemData ps, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float4 vel_w = ps.d_vel[idx];
  if (vel_w.w > 0.5f)
    return;

  float3 pos = make_float3(ps.d_pos[idx].x, ps.d_pos[idx].y, ps.d_pos[idx].z);
  float3 pos_star = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                                ps.d_pos_star[idx].z);

  // Update velocity v = (x* - x) / dt
  float3 new_vel =
      make_float3((pos_star.x - pos.x) / dt, (pos_star.y - pos.y) / dt,
                  (pos_star.z - pos.z) / dt);

  // Update position
  ps.d_pos[idx] =
      make_float4(pos_star.x, pos_star.y, pos_star.z, ps.d_pos[idx].w);
  ps.d_vel[idx] = make_float4(new_vel.x, new_vel.y, new_vel.z, vel_w.w);
}

void launch_integration(ParticleSystemData &ps, float dt, float3 gravity) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  predict_position_kernel<<<blocks, threads>>>(ps, dt, gravity);
  reset_deltas_kernel<<<blocks, threads>>>(ps);
  cudaDeviceSynchronize(); // For safety during dev
}

void launch_update(ParticleSystemData &ps, float dt) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  update_velocity_kernel<<<blocks, threads>>>(ps, dt);
  cudaDeviceSynchronize();
}
