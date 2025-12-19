#pragma once
#include <cuBQL/bvh.h>
#include <cuda_runtime.h>

enum ShapeType {
  SHAPE_GRID_SDF = 0,
  SHAPE_ANALYTIC_SPHERE = 1,
  SHAPE_ANALYTIC_HOLLOW_CYLINDER = 2, // The primary target
  SHAPE_ANALYTIC_BOX = 3
};

struct ShapeDescriptor {
  ShapeType type;

  // --- Type A: Grid SDF (Texture) ---
  cudaTextureObject_t sdf_texture;
  float3 aabb_min; // For UVW mapping (World -> Texture Space)
  float3 aabb_max;

  // --- Type B: Analytic Parameters ---
  // Pack parameters into float4 registers to save memory bandwidth
  // For Hollow Cylinder:
  // .x = height (h)
  // .y = outer_radius (R)
  // .z = thickness (t)
  // .w = unused (or inner_radius precalc)
  float4 params;

  // --- Common: Point Shell (for Collision Source) ---
  // Index into the global 'd_all_shell_points' buffer
  // Note: For now we kept d_fine_points in logic, but we can map it here.
  // To keep compatibility with existing code that assumes ShapeData, we'll
  // alias/adapt.
  // Actually, let's Replace ShapeData completely or update it.
  // The plan calls for "ShapeDescriptor".
  // Existing code uses ShapeData. Let's rename ShapeData to ShapeDescriptor but
  // keep legacy fields if needed or refactor them.
  // Existing fields: d_proxies, d_fine_points, num_points
  // We should keep d_fine_points pointer for now as per plan?
  // Plan says: "Index into global d_all_shell_points buffer".
  // Let's stick to the pointer for now to minimize refactor friction in
  // Step 2.1, or define both.

  float4 *d_fine_points; // Level 2 Point Shell (Pointer for now)
  int num_points;        // Number of points in shell

  // Legacy/Unused
  float4 *d_proxies;        // Coarse Level 1 Mask
  int2 *d_tile_descriptors; // Lookup for tiles
};

// Typedef for backward compatibility if needed, or just update usages.
using ShapeData = ShapeDescriptor;

struct BVHNode {
  float3 aabb_min;
  float3 aabb_max;
  int left;
  int right;
  int parent;
  int object_id;
};

struct Plane {
  float3 point;
  float3 normal;
};

// Trivial copyable check done in simulation.cpp if needed, but this is a view
// struct so it's fine.

struct ContactConstraint {
  int bodyA;
  int bodyB;
  float4 normal;           // .xyz = normal, .w = padding
  float4 rA;               // Vector from Center A to Contact Point
  float4 rB;               // Vector from Center B to Contact Point
  float dist;              // Penetration distance (negative = overlap)
  float friction_lambda_n; // Stored from prev frame or pos solve (for clamping)
  float weight;            // Rigorous Pair Weight (1/N_pairs)
};

struct ManifoldConstraint {
  int bodyA;              // Canonical (smaller ID)
  int bodyB;              // Larger ID
  float4 normal_sum;      // Sum of aligned normals
  float4 torque_armA_sum; // Sum of (rA x n_aligned)
  float4 torque_armB_sum; // Sum of (rB x -n_aligned)
  float4 rA_sum;          // Sum of rA
  float4 rB_sum;          // Sum of rB
  int num_points;         // Number of contacts in patch
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
  ManifoldConstraint *d_manifolds; // Aggregated Constraints
  int *d_contact_count;            // Atomic counter (INT)
  int *d_manifold_count;           // Number of manifolds
  int *d_global_contact_count;     // Alias if needed
  float *d_max_overlap;            // Global atomic for max penetration

  // Counters & Capacity
  int num_particles; // Total particles (Real + Ghosts)
  int num_real;      // Number of Real particles
  int capacity;      // Max allocated size
  int *d_top_ghost;  // Atomic counter for ghost allocation

  // Counts
  int *d_potential_count;
  int max_potential_collisions;
  int max_contacts;

  // Ghost Mapping
  int *d_real_indices; // Maps any index (Real or Ghost) to Real ID

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

  // --- Static Planes ---
  Plane *d_planes;
  int num_planes;

  // --- Static/Shape Data ---
  ShapeData *d_shapes; // Array of ShapeData structures

  // --- Material Properties ---
  float restitution_normal;
  float restitution_tangent;
  float friction_dynamic;

  // --- Growth Parameters ---
  float *d_scales;
  float *d_target_scales; // For Growth Mode
  float gravity_x, gravity_y, gravity_z;
  float global_scale;
  float growth_rate;
  float growth_factor; // -1.0f = Inactive

  // --- Thermostat Parameters ---
  double *d_energy_sum; // [0]=Trans KE, [1]=Rot KE
  bool thermostat_enabled;
  float thermostat_temperature;
  float thermostat_tau;
  float thermostat_kB;

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

void reduce_contacts_to_manifolds(ParticleSystemData &ps);
