#pragma once
#include <cuda_runtime.h>

struct ShapeData {
  cudaTextureObject_t sdf_texture; // 3D Texture for general shapes
  float4 *d_proxies;               // Coarse Level 1 Mask
  float4 *d_fine_points;           // Level 2 Point Shell
  int2 *d_tile_descriptors;        // Lookup for tiles
};

struct ParticleSystemData {
  // --- Dynamic State ---
  float4 *d_pos;     // .xyz = position, .w = inv_mass
  float4 *d_quat;    // .xyzw = quaternion
  float4 *d_vel;     // .xyz = lin_vel, .w = particle_phase (active/ghost)
  float4 *d_ang_vel; // .xyz = ang_vel, .w = particle_scale (for expansion)

  // --- XPBD Solver State ---
  float4 *d_pos_star;  // Predicted Position
  float4 *d_quat_star; // Predicted Orientation
  float4 *d_delta_pos; // Accumulator for Jacobi
  int *d_constraint_counts;

  // --- Static/Shape Data ---
  float4 *d_inv_inertia; // .xyz = diagonal inertia tensor
  int *d_shape_ids;      // Index into ShapeData

  // --- System Bounds ---
  float3 domain_min;
  float3 domain_size; // For Periodic Mapping
  int num_particles;
};
