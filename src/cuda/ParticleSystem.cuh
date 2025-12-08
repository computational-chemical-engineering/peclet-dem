#pragma once
#include <cuBQL/bvh.h>
#include <cuda_runtime.h>

struct ShapeData {
  int type; // 0=Sphere, 1=Cylinder
  float4
      params; // Generic params (e.g. Cylinder: x=radius, y=height, z=thickness)
  cudaTextureObject_t sdf_texture; // 3D Texture for general shapes
  float4 *d_proxies;               // Coarse Level 1 Mask
  float4 *d_fine_points;           // Level 2 Point Shell
  int num_points;                  // Number of points in shell
  int2 *d_tile_descriptors;        // Lookup for tiles
};

struct BVHNode {
  float3 aabb_min;
  float3 aabb_max;
  int left;
  int right;
  int parent;
  int object_id;
};

struct ParticleSystemData {
  // -------------------------------------------------------------------------
  // Particle Data (SoA) - Size = Capacity (Real + Ghosts)
  // -------------------------------------------------------------------------
  float4 *d_pos;      // .xyz = position, .w = inv_mass
  float4 *d_pos_star; // .xyz = predicted position
  float4 *d_vel;      // .xyz = velocity, .w = type/phase (0=Real, 1=Ghost)
  float4 *d_quat;     // .xyzw = quaternion (orientation)
  float *d_scale;     // Uniform scale factor
  float4 *d_ang_vel;  // .xyz = angular velocity
  int *d_shape_ids;   // Shape ID per particle

  int num_particles; // Total particles (Real + Ghosts) in current frame
  int num_real;      // Number of Real particles (index 0 to num_real-1)
  int capacity;      // Max allocated size

  // -------------------------------------------------------------------------
  // Domain & Config
  // -------------------------------------------------------------------------
  float3 domain_min;
  float3 domain_max;
  float3 domain_size; // Convenience: max - min
  bool periodic_x;
  bool periodic_y;
  bool periodic_z;
  float4 *d_quat_star; // Predicted Orientation
  float4 *d_delta_pos; // Accumulator for Jacobi
  int *d_constraint_counts;

  // --- Static/Shape Data ---
  float4 *d_inv_inertia; // .xyz = diagonal inertia tensor
  ShapeData *d_shapes;   // Array of ShapeData structures

  // --- Broadphase Data ---
  cuBQL::bvh3f bvh;
  int2 *d_potential_collisions;
  int *d_potential_count; // Single counter
  int max_potential_collisions;

  // BVH Construction Buffers (Linear BVH / Morton)
  BVHNode *d_bvh_nodes;
  int *d_bvh_indices;
  unsigned int *d_morton;
  int *d_indices_sorted;

  // Narrowphase Contacts
  int2 *d_contacts;
  int *d_num_contacts;
  int max_contacts;

  // Locks for Gauss-Seidel
  int *d_locks;

  // --- System Bounds ---
};
