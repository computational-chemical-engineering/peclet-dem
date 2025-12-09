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

// Trivial copyable check done in simulation.cpp if needed, but this is a view
// struct so it's fine.

struct ContactConstraint {
  int bodyA;
  int bodyB;
  float4 normal; // .xyz = normal, .w = padding
  float4 rA;     // .xyz = vector from COM_A to contact point
  float4 rB;     // .xyz = vector from COM_B to contact point
  float dist;    // Signed distance (Negative = Overlap, Positive = Gap)
  float friction_lambda_n; // Stored from prev frame or pos solve (for clamping)
};

struct ParticleSystemData {
  // -------------------------------------------------------------------------
  // Particle Data (SoA) - Size = Capacity (Real + Ghosts)
  // -------------------------------------------------------------------------
  // State Arrays (Read Only during Sub-step)
  float4 *d_pos;  // .xyz = position, .w = inv_mass
  float4 *d_quat; // .xyzw = quaternion (orientation)
  float4 *d_vel;  // .xyz = linear velocity, .w = type/phase (0=Real, 1=Ghost)
  float4 *d_ang_vel;     // .xyz = angular velocity
  float4 *d_inv_inertia; // .xyz = diagonal inertia, .w = padding
  float *d_scale;        // Uniform scale factor
  int *d_shape_ids;      // Shape ID per particle

  // Predicted State (The "Working" Copy)
  float4 *d_pos_pred;
  float4 *d_quat_pred;
  float4 *d_vel_pred; // Intermediate velocity state
  float4 *d_ang_vel_pred;
  float4 *d_pos_star;  // Legacy name aliasing d_pos_pred for now? No, stick to
                       // new names.
  float4 *d_quat_star; // Legacy alias

  // Jacobi Accumulators (Write Only)
  float4 *d_delta_pos;
  float4 *d_delta_quat;
  float4 *d_delta_vel;
  float4 *d_delta_ang_vel;
  int *d_constraint_counts;

  // Constraint Buffer
  ContactConstraint *d_contacts;
  int *d_contact_count;        // Atomic counter (INT)
  int *d_global_contact_count; // Alias if needed

  // Counters & Capacity
  int num_particles; // Total particles (Real + Ghosts)
  int num_real;      // Number of Real particles
  int capacity;      // Max allocated size
  int *d_top_ghost;  // Atomic counter for ghost allocation

  // Counts
  int *d_potential_count;
  int max_potential_collisions;
  int max_contacts;

  // -------------------------------------------------------------------------
  // Domain & Config
  // -------------------------------------------------------------------------
  float3 domain_min;
  float3 domain_max;
  float3 domain_size;
  bool periodic_x;
  bool periodic_y;
  bool periodic_z;

  float dt;
  float4 gravity;

  // --- Static/Shape Data ---
  ShapeData *d_shapes; // Array of ShapeData structures

  // --- Material Properties ---
  float restitution_normal;
  float restitution_tangent;
  float friction_dynamic;

  // --- Broadphase Data ---
  cuBQL::bvh3f bvh;
  int2 *d_potential_collisions; // Legacy?

  // BVH Construction Buffers
  BVHNode *d_bvh_nodes;
  int *d_bvh_indices;
  unsigned int *d_morton;
  int *d_indices_sorted;

  // Other Legacy
  float *d_contact_lambdas;
  float *d_tangent_lambdas;
  int *d_locks;
};
