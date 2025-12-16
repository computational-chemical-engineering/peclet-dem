#include "simulation.h"
#include "cpp/OutputGenerator.hpp"
#include "cuda/math_utils.cuh"
#include "cuda/memory_utils.cuh"
#include "cuda/periodicity.cuh"
#include "io/Exporter.h"
#include "shapes/point_sampler.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <vector_types.h>

Simulation::Simulation(int num_particles) : num_particles_(num_particles) {
  // Initialize pointers to nullptr
  ps_.d_pos = nullptr;
  ps_.d_quat = nullptr;
  ps_.d_vel = nullptr;
  ps_.d_scale = nullptr;
  ps_.d_ang_vel = nullptr;
  ps_.d_pos_star = nullptr;
  ps_.d_quat_star = nullptr;
  ps_.d_pos_pred = nullptr;
  ps_.d_quat_pred = nullptr;
  ps_.d_vel_pred = nullptr;
  ps_.d_ang_vel_pred = nullptr;
  // ps_.d_pos_star = nullptr; // Legacy
  // ps_.d_quat_star = nullptr; // Legacy
  ps_.d_delta_pos = nullptr;
  ps_.d_delta_vel = nullptr;
  ps_.d_delta_ang_vel = nullptr;
  ps_.d_delta_quat = nullptr;
  ps_.d_constraint_counts = nullptr;
  ps_.d_inv_inertia = nullptr;
  ps_.d_shape_ids = nullptr;
  // ps_.d_contacts = nullptr;
  ps_.d_potential_collisions = nullptr;
  ps_.d_potential_count = nullptr;
  ps_.d_shapes = nullptr;
  ps_.d_bvh_nodes = nullptr;
  ps_.d_bvh_indices = nullptr;
  ps_.d_morton = nullptr;
  ps_.d_indices_sorted = nullptr;
  ps_.d_contacts = nullptr;
  ps_.d_contact_count = nullptr;
  ps_.d_manifolds = nullptr;
  ps_.d_manifold_count = nullptr;
  ps_.d_planes = nullptr; // Initialize plane pointer
  ps_.d_max_overlap = nullptr;
  ps_.d_locks = nullptr;
  ps_.d_top_ghost = nullptr;
  ps_.d_target_scales = nullptr; // Growth

  // Initialize defaults
  gravity_ = make_float3(0, 0.0f, 0);
  global_scale_ = 1.0f;
  position_iterations_ = 10;
  velocity_iterations_ = 0;
  domain_initialized_ = false; // Flag for first set_domain call
  force_sync_ = true;          // Ensure first step is always synchronized
  base_radius_ = 1.0f;

  // Initialize Domain Defaults (prevents garbage reads)
  ps_.domain_min = make_float3(-5.0f, -5.0f, -5.0f);
  ps_.domain_max = make_float3(5.0f, 5.0f, 5.0f);
  ps_.domain_size = make_float3(10.0f, 10.0f, 10.0f);

  // Default: Periodicity disabled (until set_domain called)
  ps_.periodic_x = false;
  ps_.periodic_y = false;
  ps_.periodic_z = false;

  ps_.num_planes = 0;

  // allocate_system removed

  // cudaEventCreate(&start_event_);
  // cudaEventCreate(&stop_event_);
  // cudaEventCreate(&integration_start_);
  // cudaEventCreate(&integration_stop_);
  // cudaEventCreate(&broadphase_start_);
  // cudaEventCreate(&broadphase_stop_);
  // cudaEventCreate(&solver_start_);
  // cudaEventCreate(&solver_stop_);
}

Simulation::~Simulation() {
  free_device(ps_.d_pos);
  free_device(ps_.d_quat);
  free_device(ps_.d_scale);
  free_device(ps_.d_target_scales); // Growth
  free_device(ps_.d_vel);
  // free_device(ps_.d_vel_old); // Deprecated
  free_device(ps_.d_top_ghost);
  free_device(ps_.d_ang_vel);

  free_device(ps_.d_pos_star);
  free_device(ps_.d_quat_star);
  free_device(ps_.d_delta_pos);
  free_device(ps_.d_constraint_counts);

  free_device(ps_.d_inv_inertia);
  free_device(ps_.d_shape_ids);

  // Cleanup collision buffers if allocated
  // Cleanup collision buffers if allocated
  if (ps_.d_contacts)
    free_device(ps_.d_contacts);
  if (ps_.d_max_overlap)
    free_device(ps_.d_max_overlap);

  free_device(ps_.d_contact_count); // Renamed from d_num_contacts
  if (ps_.d_manifolds)
    free_device(ps_.d_manifolds);
  if (ps_.d_manifold_count)
    free_device(ps_.d_manifold_count);
  if (ps_.d_planes) {
    free_device(ps_.d_planes);
  }
  free_device(ps_.d_locks); // Moved from separate if
  free_device(ps_.d_real_indices);
  // free_device(ps_.d_contact_lambdas); // Not Allocated
  // free_device(ps_.d_tangent_lambdas); // Not Allocated

  // The following were originally separate if statements, but the user's
  // snippet implied they should be freed unconditionally after the
  // d_num_contacts block. To maintain logical correctness and avoid
  // double-freeing d_contacts, and given the instruction "Allocate
  // d_contact_lambdas" which implies adding a new buffer, the most faithful
  // interpretation of the snippet's intent for the destructor is to free
  // d_contact_lambdas and ensure other frees are handled correctly. The snippet
  // provided for the destructor was syntactically problematic without curly
  // braces and included lines that were already handled or out of place.
  // Reverting to the original structure for existing frees and adding the new
  // one.

  if (ps_.d_potential_collisions)
    free_device(ps_.d_potential_collisions);
  if (ps_.d_potential_count)
    free_device(ps_.d_potential_count);
  if (ps_.d_bvh_nodes)
    free_device(ps_.d_bvh_nodes);
  if (ps_.d_bvh_indices)
    free_device(ps_.d_bvh_indices);
  if (ps_.d_morton)
    free_device(ps_.d_morton);
  if (ps_.d_indices_sorted)
    free_device(ps_.d_indices_sorted);

  // cudaEventDestroy(start_event_);
  // cudaEventDestroy(stop_event_);
  // cudaEventDestroy(integration_start_);
  // cudaEventDestroy(integration_stop_);
  // cudaEventDestroy(broadphase_start_);
  // cudaEventDestroy(broadphase_stop_);
  // cudaEventDestroy(solver_start_);
  // cudaEventDestroy(solver_stop_);
}

void Simulation::enable_periodicity(bool x, bool y, bool z) {
  if (!domain_initialized_) {
    throw std::runtime_error(
        "Cannot enable periodicity before setting domain via set_domain().");
  }
  ps_.periodic_x = x;
  ps_.periodic_y = y;
  ps_.periodic_z = z;
  printf("Periodicity Configured: X=%d Y=%d Z=%d\n", x, y, z);
}
// Deprecated allocate_system removed.

// External kernel wrappers
// External kernel wrappers
extern void launch_integrate_predict_velocity(ParticleSystemData ps, float dt,
                                              float3 gravity);
extern void launch_apply_velocity_and_predict_position(ParticleSystemData ps,
                                                       float dt);
extern void launch_apply_updates(ParticleSystemData ps);

extern void launch_apply_velocity_deltas(ParticleSystemData ps);
extern void launch_compute_contact_counts(ParticleSystemData ps);
extern void launch_final_commit(ParticleSystemData ps, float dt);
void launch_generate_ghosts(ParticleSystemData ps, float margin);
extern void launch_narrowphase(ParticleSystemData ps, float global_scale);
extern void launch_velocity_solve(ParticleSystemData ps, float nu);
extern void launch_position_solve(ParticleSystemData ps);
extern void launch_update_growth_scales(ParticleSystemData &ps);
void build_bvh(ParticleSystemData &ps, float global_scale);
void find_collisions(ParticleSystemData ps, float global_scale);
void generate_ghosts(ParticleSystemData &ps, float threshold);

// 0=Sphere, 1=Cylinder
// 0=Sphere, 1=Cylinder
void Simulation::initialize(int shape_type, float radius, float height,
                            float thickness) {
  // Allocate more for Ghosts (e.g., 8.0x for robust periodicity)
  int capacity = calculate_capacity(num_particles_, ps_.domain_size,
                                    1.0f); // Init with 1.0 margin
  ps_.capacity = capacity;
  ps_.num_real = num_particles_;
  ps_.num_particles = num_particles_;

  // Store Base Radius
  base_radius_ = radius;

  // Set Domain Defaults if not already set
  if (!domain_initialized_) {
    ps_.domain_min = make_float3(-5.0f, -5.0f, -5.0f);
    ps_.domain_max = make_float3(5.0f, 5.0f, 5.0f);
    ps_.domain_size = make_float3(10.0f, 10.0f, 10.0f);
    // Periodicity defaults (false) apply
  }

  allocate_device(ps_.d_pos, capacity);
  allocate_device(ps_.d_vel, capacity);
  allocate_device(ps_.d_quat, capacity);
  allocate_device(ps_.d_scale, capacity);
  allocate_device(ps_.d_target_scales, capacity); // Growth
  allocate_device(ps_.d_ang_vel, capacity);
  allocate_device(ps_.d_shape_ids, capacity);
  allocate_device(ps_.d_top_ghost, 1);

  // Initialize Growth State
  ps_.growth_rate = 0.0f;
  ps_.growth_factor = -1.0f; // Inactive

  allocate_device(ps_.d_pos_pred, capacity);
  allocate_device(ps_.d_quat_pred, capacity);
  allocate_device(ps_.d_vel_pred, capacity);
  allocate_device(ps_.d_ang_vel_pred, capacity);

  ps_.d_pos_star = ps_.d_pos_pred; // Alias for legacy if needed
  ps_.d_quat_star = ps_.d_quat_pred;

  allocate_device(ps_.d_delta_pos, capacity);
  allocate_device(ps_.d_delta_vel, capacity);
  allocate_device(ps_.d_delta_quat, capacity);
  allocate_device(ps_.d_delta_ang_vel, capacity);

  allocate_device(ps_.d_constraint_counts, capacity);
  allocate_device(ps_.d_inv_inertia, capacity);

  // Collision Buffers
  int max_potential =
      capacity * 200; // Increased from 50 to 200 for dense packing safety
  ps_.max_potential_collisions = max_potential;
  allocate_device(ps_.d_potential_collisions, max_potential);
  allocate_device(ps_.d_potential_count, 1);

  // BVH Buffers
  allocate_device(ps_.d_bvh_nodes, (2 * capacity - 1));
  allocate_device(ps_.d_bvh_indices, capacity);
  allocate_device(ps_.d_morton, capacity);
  allocate_device(ps_.d_indices_sorted, capacity);

  // Contact Buffers
  int max_contacts = capacity * 200; // Increased to match potential
  ps_.max_contacts = max_contacts;
  allocate_device(ps_.d_contacts, max_contacts);
  allocate_device(ps_.d_contact_count, 1 * sizeof(int));

  // Manifold Buffers
  allocate_device(ps_.d_manifolds, max_contacts);
  allocate_device(ps_.d_manifold_count, 1 * sizeof(int));

  // Locks
  allocate_device(ps_.d_locks, capacity * sizeof(int));
  allocate_device(ps_.d_max_overlap, 1 * sizeof(float));
  CUDA_CHECK(cudaMemset(ps_.d_locks, 0, capacity * sizeof(int)));

  // Ghost Mapping
  allocate_device(ps_.d_real_indices, capacity);

  // Arrays for Shapes
  CylinderParams cyl_params;
  cyl_params.radius = radius;
  cyl_params.height = height;
  cyl_params.thickness = thickness;

  std::vector<float4> h_points;
  ShapeType type = (ShapeType)shape_type;
  float4 params;

  if (shape_type == 2) {
    // CYLINDER (ID=2)
    params = make_float4(cyl_params.radius, cyl_params.height,
                         cyl_params.thickness, 0);
    h_points = generate_cylinder_points(cyl_params, 0.1f);
  } else {
    // SPHERE (ID=1)
    if (type != SHAPE_ANALYTIC_SPHERE && type != SHAPE_ANALYTIC_HOLLOW_CYLINDER)
      type = SHAPE_ANALYTIC_SPHERE;

    // Use radius for Sphere Parameters (Base Geometry)
    params = make_float4(radius, 0, 0, 0);
  }

  // Create Shape via Manager
  int created_id = shape_manager_.createAnalyticShape(type, params, h_points);
  printf("Created Shape ID %d via Manager. Points: %lu\n", created_id,
         h_points.size());

  // Upload to GPU
  shape_manager_.uploadToGPU();
  ps_.d_shapes = shape_manager_.getDeviceShapes();

  printf("Simulation initialized. Seeding particles...\n");
  CUDA_CHECK(cudaMemset(ps_.d_max_overlap, 0, sizeof(float)));

  // 1. Initialize Particles
  std::vector<float4> h_pos(num_particles_);
  std::vector<float4> h_vel(num_particles_);
  std::vector<float4> h_quat(num_particles_);
  std::vector<float> h_scale(num_particles_);
  std::vector<int> h_shape_ids(num_particles_);

  int idx = 0;
  for (int i = 0; i < num_particles_; ++i) {
    float x = (float)rand() / RAND_MAX * ps_.domain_size.x + ps_.domain_min.x;
    float y = (float)rand() / RAND_MAX * ps_.domain_size.y + ps_.domain_min.y;
    float z = (float)rand() / RAND_MAX * ps_.domain_size.z + ps_.domain_min.z;

    h_pos[i] = make_float4(x, y, z, 1.0f);
    h_vel[i] = make_float4(0, 0, 0, 0.0f);
    h_quat[i] = make_float4(0, 0, 0, 1);

    // Initialize scale to 1.0 (Unit of Base Radius)
    h_scale[i] = 1.0f;

    h_shape_ids[i] = 0;
  }

  // Create default inertia
  // Sphere I = 2/5 * m * r^2
  // InvI = 5/2 * invM * 1/r^2
  // initial mass = 1.0. invM = 1.0.
  // base_radius_ is set from radius arg.

  float inv_I_xx = 1.0f;
  float inv_I_yy = 1.0f;
  float inv_I_zz = 1.0f;

  if (shape_type == 0 || shape_type == 1) { // Sphere
    if (base_radius_ > 0.0f) {
      float val = 2.5f / (base_radius_ * base_radius_);
      inv_I_xx = val;
      inv_I_yy = val;
      inv_I_zz = val;
    }
  } else if (shape_type == 2) { // Hollow Cylinder
    float r_out = base_radius_;
    float r_in = base_radius_ - thickness;
    if (r_in < 0)
      r_in = 0;
    float h = height;

    // I_zz = 0.5 * m * (rout^2 + rin^2)
    // I_xx = 1/12 * m * (3*(rout^2 + rin^2) + h^2)
    // m = 1.0

    float term_r = r_out * r_out + r_in * r_in;
    float I_zz = 0.5f * term_r;
    float I_xx = (1.0f / 12.0f) * (3.0f * term_r + h * h);

    if (I_xx > 1e-6f)
      inv_I_xx = 1.0f / I_xx;
    if (I_xx > 1e-6f)
      inv_I_yy = 1.0f / I_xx;
    if (I_zz > 1e-6f)
      inv_I_zz = 1.0f / I_zz;
  }

  std::vector<float4> h_inv_inertia(num_particles_);
  std::vector<int> h_real_indices(
      num_particles_); // Identity map for initial particles
  for (int i = 0; i < num_particles_; ++i) {
    h_inv_inertia[i] = make_float4(inv_I_xx, inv_I_yy, inv_I_zz, 0.0f);
    h_real_indices[i] = i;
  }

  CUDA_CHECK(cudaMemcpy(ps_.d_pos, h_pos.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_vel, h_vel.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_quat, h_quat.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_scale, h_scale.data(),
                        num_particles_ * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_shape_ids, h_shape_ids.data(),
                        num_particles_ * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_inv_inertia, h_inv_inertia.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_real_indices, h_real_indices.data(),
                        num_particles_ * sizeof(int), cudaMemcpyHostToDevice));
}

// -----------------------------------------------------------------------------
// Parameters & Domain
// -----------------------------------------------------------------------------
void Simulation::set_gravity(float x, float y, float z) {
  gravity_ = make_float3(x, y, z);
}

void Simulation::set_global_scale(float s) { global_scale_ = s; }

// Material Parameters
void Simulation::set_material_params(float restitution_normal,
                                     float restitution_tangent,
                                     float friction_dynamic) {
  // allocate_device(ps_.d_top_ghost, 1 * sizeof(int)); // REMOVED: Duplicate
  // allocation bug.

  // Set material params default
  ps_.restitution_normal = 0.5f;
  ps_.restitution_normal = restitution_normal;
  ps_.restitution_tangent = restitution_tangent;
  ps_.friction_dynamic = friction_dynamic;
}

void Simulation::set_domain(float3 min, float3 max) {
  ps_.domain_min = min;
  ps_.domain_max = max;
  ps_.domain_size = make_float3(max.x - min.x, max.y - min.y, max.z - min.z);

  // On first call to set_domain, enable periodicity by default
  if (!domain_initialized_) {
    ps_.periodic_x = true;
    ps_.periodic_y = true;
    ps_.periodic_z = true;
    domain_initialized_ = true;
    printf("Domain Initialized: Periodicity enabled by default.\n");
  }
}

std::tuple<float, float, float> Simulation::get_domain_min() {
  return std::make_tuple(ps_.domain_min.x, ps_.domain_min.y, ps_.domain_min.z);
}

std::tuple<float, float, float> Simulation::get_domain_max() {
  return std::make_tuple(ps_.domain_max.x, ps_.domain_max.y, ps_.domain_max.z);
}

void Simulation::set_solver_iterations(int pos_its, int vel_its) {
  position_iterations_ = pos_its;
  velocity_iterations_ = vel_its;
}

void Simulation::set_growth_params(float rate, float new_factor) {
  // 1. Snapshot d_scales -> d_target_scales if activating for first time
  if (ps_.growth_factor == -1.0f) {
    printf("Growth Mode Activated. Snapshotting current scales to target. "
           "Rate=%.4f\n",
           rate);
    CUDA_CHECK(cudaMemcpy(ps_.d_target_scales, ps_.d_scale,
                          ps_.num_particles * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Initialize factor
    if (new_factor > 0.0f) {
      ps_.growth_factor = new_factor;
    } else {
      ps_.growth_factor = 0.01f; // Default start small
    }
  } else {
    // Already active, just updating
    if (new_factor > 0.0f) {
      ps_.growth_factor = new_factor; // Restart/Jump
      printf("Growth Mode Updated. Restarting factor to %.4f.\n", new_factor);
    } else {
      printf("Growth Mode Updated. Rate=%.4f (Continuing factor %.4f)\n", rate,
             ps_.growth_factor);
    }
  }

  ps_.growth_rate = rate;

  // Apply immediate update
  launch_update_growth_scales(ps_);
}

void Simulation::step(float dt) {
  // DEBUG: Inspect Particle 0 on GPU
  // Synchronization to prevent kernel launch race conditions (Heisenbug fix)
  // Optimization: Only sync if flagged (e.g. after data upload)
  if (force_sync_) {
    CUDA_CHECK(cudaDeviceSynchronize());
    force_sync_ = false;
  }

  // Growth Logic
  float nu = 0.0f;
  if (ps_.growth_factor != -1.0f) {
    if (ps_.growth_rate != 0.0f) {
      ps_.growth_factor += ps_.growth_rate * dt;
      if (ps_.growth_factor >= 1.0f) {
        ps_.growth_factor = 1.0f;
        ps_.growth_rate = 0.0f; // Stop growth
        nu = 0.0f;
        printf("Growth Mode Completed. Factor=1.0.\n");
      } else {
        // nu = growth_rate / growth_factor
        nu = ps_.growth_rate / ps_.growth_factor;
      }
    }

    // Update Scales Kernel
    launch_update_growth_scales(ps_);
  }

  // New Pipeline (Hybrid Velocity/Position)

  // 1. Predict Velocity (Gravity Only)
  launch_integrate_predict_velocity(ps_, dt, gravity_);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2. Generate Ghosts
  if (ps_.periodic_x || ps_.periodic_y || ps_.periodic_z) {
    update_ghosts();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // 3. Broadphase & Narrowphase
  build_bvh(ps_, global_scale_);
  CUDA_CHECK(cudaDeviceSynchronize()); // Check BVH

  find_collisions(ps_, 1.0f);
  CUDA_CHECK(cudaDeviceSynchronize()); // Check Traversal

  launch_narrowphase(ps_, global_scale_);
  CUDA_CHECK(cudaDeviceSynchronize()); // Check Narrowphase

  // 4. Phase A: Velocity Solve

  // A. Pre-Pass: Count Contacts for Min-Scaling Weighting
  // (d_constraint_counts is reused to store N for each particle)
  // A. Pre-Pass: Rigorous Manifold Reduction
  // Replaces the old contact weighting
  reduce_contacts_to_manifolds(ps_);
  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure sorting/weighting is done

  // B. Iterative Solve (Velocity-First)
  for (int i = 0; i < velocity_iterations_; ++i) {
    launch_velocity_solve(ps_, nu);
    launch_apply_velocity_deltas(ps_);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 5. Apply Velocity & Re-Integrate Position
  // Note: d_delta_vel is now cleared. This function effectively acts as
  // "Predict Position from v_pred".
  launch_apply_velocity_and_predict_position(ps_, dt);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 6. Phase B: Position Solve (Projected Jacobi)
  for (int i = 0; i < position_iterations_; ++i) {
    CUDA_CHECK(cudaMemset(ps_.d_max_overlap, 0, sizeof(float)));
    launch_position_solve(ps_);
    launch_apply_updates(ps_);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 7. Final Commit
  launch_final_commit(ps_, dt);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// Python Bindings Helpers
// -----------------------------------------------------------------------------

void Simulation::set_positions_numpy(py::array_t<float> pos) {
  py::buffer_info buf = pos.request();
  if (buf.ndim != 2 || buf.shape[0] != num_particles_ || buf.shape[1] < 3) {
    throw std::runtime_error("Positions must be (N, 3) or (N, 4)");
  }

  std::vector<float4> h_pos(num_particles_);
  char *raw_ptr = (char *)buf.ptr;
  for (int i = 0; i < num_particles_; ++i) {
    char *row = raw_ptr + i * buf.strides[0];
    float x = *(float *)(row + 0 * buf.strides[1]);
    float y = *(float *)(row + 1 * buf.strides[1]);
    float z = *(float *)(row + 2 * buf.strides[1]);
    float w_val = 1.0f;
    if (buf.shape[1] >= 4) {
      w_val = *(float *)(row + 3 * buf.strides[1]);
    }

    // Safety Force
    if (w_val == 0.0f) {
      // Warning only once
      if (i == 0)
        printf(
            "WARNING: Particles have w=0.0 (infinite mass), forcing to 1.0\n");
      w_val = 1.0f;
    }

    h_pos[i] = make_float4(x, y, z, w_val);
  }

  // Remove stderr debug
  // fflush(stdout);

  CUDA_CHECK(cudaMemcpy(ps_.d_pos, h_pos.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_pos_star, ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));

  // Explicitly redundant copy to d_pos_pred to ensure consistency
  CUDA_CHECK(cudaMemcpy(ps_.d_pos_pred, ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));

  // Flag next step to synchronize
  force_sync_ = true;
}

void Simulation::set_velocities_numpy(py::array_t<float> vel) {
  py::buffer_info buf = vel.request();
  if (buf.ndim != 2 || buf.shape[0] != num_particles_ || buf.shape[1] < 3) {
    throw std::runtime_error("Velocities must be (N, 3)");
  }
  float *ptr = static_cast<float *>(buf.ptr);

  std::vector<float4> h_vel(num_particles_);
  for (int i = 0; i < num_particles_; ++i) {
    float x = ptr[i * buf.shape[1] + 0];
    float y = ptr[i * buf.shape[1] + 1];
    float z = ptr[i * buf.shape[1] + 2];
    float w = (buf.shape[1] >= 4) ? ptr[i * buf.shape[1] + 3] : 0.0f;
    h_vel[i] = make_float4(x, y, z, w);
  }
  CUDA_CHECK(cudaMemcpy(ps_.d_vel, h_vel.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
  force_sync_ = true;
}

void Simulation::set_scales_numpy(py::array_t<float> scales) {
  py::buffer_info buf = scales.request();
  if (buf.size != num_particles_) {
    throw std::runtime_error("Scales must be size N");
  }
  float *ptr = static_cast<float *>(buf.ptr);
  CUDA_CHECK(cudaMemcpy(ps_.d_scale, ptr, num_particles_ * sizeof(float),
                        cudaMemcpyHostToDevice));
  force_sync_ = true;
}

// -----------------------------------------------------------------------------
// Getters
// -----------------------------------------------------------------------------
py::array_t<float> Simulation::get_positions_numpy(bool include_ghosts) {
  int n = include_ghosts ? ps_.num_particles : num_particles_;
  // The following lines are from the instruction, but seem misplaced for
  // get_positions_numpy auto pos_ptr = get_device_ptr(ps_.d_pos); int count =
  // 0; std::vector<float4> h_pos(capacity); // 'capacity' is not defined here
  std::vector<float4> h_pos(n); // Keep original h_pos initialization
  CUDA_CHECK(cudaMemcpy(h_pos.data(), ps_.d_pos, n * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result({n, 3});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < n; ++i) {
    r(i, 0) = h_pos[i].x;
    r(i, 1) = h_pos[i].y;
    r(i, 2) = h_pos[i].z;
  }
  return result;
}

py::array_t<float> Simulation::get_velocities_numpy() {
  std::vector<float4> h_vel(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_vel.data(), ps_.d_vel,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result({num_particles_, 3});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < num_particles_; ++i) {
    r(i, 0) = h_vel[i].x;
    r(i, 1) = h_vel[i].y;
    r(i, 2) = h_vel[i].z;
  }
  return result;
}

py::array_t<float> Simulation::get_angular_velocities_numpy() {
  std::vector<float4> h_omega(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_omega.data(), ps_.d_ang_vel,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result({num_particles_, 3});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < num_particles_; ++i) {
    r(i, 0) = h_omega[i].x;
    r(i, 1) = h_omega[i].y;
    r(i, 2) = h_omega[i].z;
  }
  return result;
}

py::array_t<float> Simulation::get_inv_inertia_numpy() {
  std::vector<float4> h_inv_I(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_inv_I.data(), ps_.d_inv_inertia,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result({num_particles_, 3});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < num_particles_; ++i) {
    r(i, 0) = h_inv_I[i].x;
    r(i, 1) = h_inv_I[i].y;
    r(i, 2) = h_inv_I[i].z;
  }
  return result;
}

py::array_t<float> Simulation::get_masses_numpy() {
  std::vector<float4> h_pos(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_pos.data(), ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result(num_particles_);
  auto r = result.mutable_unchecked<1>();
  for (int i = 0; i < num_particles_; ++i) {
    float inv_mass = h_pos[i].w;
    r(i) = (inv_mass > 0.0f) ? (1.0f / inv_mass) : 0.0f; // Handle infinite mass
  }
  return result;
}

py::array_t<float> Simulation::get_quaternions_numpy() {
  std::vector<float4> h_quat(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_quat.data(), ps_.d_quat,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result({num_particles_, 4});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < num_particles_; ++i) {
    r(i, 0) = h_quat[i].x;
    r(i, 1) = h_quat[i].y;
    r(i, 2) = h_quat[i].z;
    r(i, 3) = h_quat[i].w;
  }
  return result;
}

py::array_t<float> Simulation::get_scales_numpy() {
  py::array_t<float> result(num_particles_);
  py::buffer_info buf = result.request();
  float *ptr = static_cast<float *>(buf.ptr);
  CUDA_CHECK(cudaMemcpy(ptr, ps_.d_scale, num_particles_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return result;
}

float Simulation::get_max_overlap() {
  float max_ov;
  CUDA_CHECK(cudaMemcpy(&max_ov, ps_.d_max_overlap, sizeof(float),
                        cudaMemcpyDeviceToHost));
  return max_ov;
}

float Simulation::compute_overlaps() {
  // 1. Copy current State to Predict (kernels read pred)
  CUDA_CHECK(cudaMemcpy(ps_.d_pos_pred, ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_quat_pred, ps_.d_quat,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));
  // Scale is already there.

  // 2. Build BVH
  build_bvh(ps_, global_scale_);

  // 3. Find Collisions
  find_collisions(ps_, global_scale_);

  // 4. Reset Max Overlap
  CUDA_CHECK(cudaMemset(ps_.d_max_overlap, 0, sizeof(float)));

  // 5. Run Narrowphase in "Compute Only" mode?
  // Actually, we can just run it. If it generates contacts, it's fine,
  // we just discard them. But to be safe and fast, let's just ensure
  // narrowphase updates d_max_overlap.
  // We need to modify narrowphase to update d_max_overlap.
  // And maybe reset contact count if we don't want to overflow buffer?
  CUDA_CHECK(cudaMemset(ps_.d_contact_count, 0, sizeof(int)));

  launch_narrowphase(ps_, global_scale_);

  return get_max_overlap();
}

int Simulation::get_num_contacts() {
  int count = 0;
  if (ps_.d_contact_count) {
    CUDA_CHECK(cudaMemcpy(&count, ps_.d_contact_count, sizeof(int),
                          cudaMemcpyDeviceToHost));
  }
  return count;
}

void Simulation::add_plane(float3 point, float3 normal) {
  Plane p;
  p.point = point;
  p.normal = normal;
  planes_host_.push_back(p);

  // Re-allocate device buffer
  if (ps_.d_planes) {
    free_device(ps_.d_planes);
  }

  ps_.num_planes = planes_host_.size();
  if (ps_.num_planes > 0) {
    allocate_device(ps_.d_planes, ps_.num_planes * sizeof(Plane));
    CUDA_CHECK(cudaMemcpy(ps_.d_planes, planes_host_.data(),
                          ps_.num_planes * sizeof(Plane),
                          cudaMemcpyHostToDevice));
  }
  printf("Added Plane: Point(%.2f, %.2f, %.2f) Normal(%.2f, %.2f, %.2f)\n",
         point.x, point.y, point.z, normal.x, normal.y, normal.z);
}

void Simulation::write_vtp(const std::string &filename) const {
  std::vector<float4> h_pos(num_particles_);
  std::vector<float> h_scale(num_particles_);

  CUDA_CHECK(cudaMemcpy(h_pos.data(), ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_scale.data(), ps_.d_scale,
                        num_particles_ * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"PolyData\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n";
  out << "  <PolyData>\n";
  out << "    <Piece NumberOfPoints=\"" << num_particles_
      << "\" NumberOfVerts=\"0\" "
      << "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

  // Points
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float32\" Name=\"Position\" "
         "NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int i = 0; i < num_particles_; ++i) {
    out << h_pos[i].x << " " << h_pos[i].y << " " << h_pos[i].z << " ";
  }
  out << "\n        </DataArray>\n";
  out << "      </Points>\n";

  // Point Data
  out << "      <PointData Scalars=\"Radius\">\n";
  out << "        <DataArray type=\"Float32\" Name=\"Radius\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  for (int i = 0; i < num_particles_; ++i) {
    // Radius = scale * global_scale * base_radius_
    float r = h_scale[i] * global_scale_ * base_radius_;
    out << r << " ";
  }
  out << "\n        </DataArray>\n";

  // Velocity
  std::vector<float4> h_vel(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_vel.data(), ps_.d_vel,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  out << "        <DataArray type=\"Float32\" Name=\"Velocity\" "
         "NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (int i = 0; i < num_particles_; ++i) {
    out << h_vel[i].x << " " << h_vel[i].y << " " << h_vel[i].z << " ";
  }
  out << "\n        </DataArray>\n";
  out << "      </PointData>\n";
  out << "    </Piece>\n";
  out << "  </PolyData>\n";
  out << "</VTKFile>\n";
  out.close();
  printf("Exported VTP: %s\n", filename.c_str());
}

void Simulation::export_lammps(const std::string &filename, int step) {
  // 1. Download Data
  std::vector<float3> h_pos(num_particles_);
  std::vector<float3> h_vel(num_particles_);
  std::vector<float4> h_quat(num_particles_);
  std::vector<float> h_radii(num_particles_);

  // We need intermediate buffers because d_pos is float4 (x,y,z,inv_mass)
  std::vector<float4> h_pos4(num_particles_);
  std::vector<float4> h_vel4(num_particles_);
  std::vector<float> h_scale(num_particles_);

  CUDA_CHECK(cudaMemcpy(h_pos4.data(), ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_vel4.data(), ps_.d_vel,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_quat.data(), ps_.d_quat,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_scale.data(), ps_.d_scale,
                        num_particles_ * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int i = 0; i < num_particles_; ++i) {
    h_pos[i] = make_float3(h_pos4[i].x, h_pos4[i].y, h_pos4[i].z);
    h_vel[i] = make_float3(h_vel4[i].x, h_vel4[i].y, h_vel4[i].z);
    // Radius = scale * global_scale * base_radius_
    h_radii[i] = h_scale[i] * global_scale_ * base_radius_;
  }

  // 2. Determine Bounds
  float3 bmin = make_float3(1e20f, 1e20f, 1e20f);
  float3 bmax = make_float3(-1e20f, -1e20f, -1e20f);

  // Default to computing from particles for non-periodic dims
  for (int i = 0; i < num_particles_; ++i) {
    float r = h_radii[i];
    bmin.x = std::min(bmin.x, h_pos[i].x - r);
    bmin.y = std::min(bmin.y, h_pos[i].y - r);
    bmin.z = std::min(bmin.z, h_pos[i].z - r);
    bmax.x = std::max(bmax.x, h_pos[i].x + r);
    bmax.y = std::max(bmax.y, h_pos[i].y + r);
    bmax.z = std::max(bmax.z, h_pos[i].z + r);
  }

  // Add margin for computed bounds logic (mimicking existing behavior)
  float margin = 0.05f;
  float3 computed_size =
      make_float3(bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z);

  if (!ps_.periodic_x) {
    bmin.x -= computed_size.x * margin;
    bmax.x += computed_size.x * margin;
  }
  if (!ps_.periodic_y) {
    bmin.y -= computed_size.y * margin;
    bmax.y += computed_size.y * margin;
  }
  if (!ps_.periodic_z) {
    bmin.z -= computed_size.z * margin;
    bmax.z += computed_size.z * margin;
  }

  // Override with Domain if Periodic
  bool any_periodic = false;
  if (ps_.periodic_x) {
    bmin.x = ps_.domain_min.x;
    bmax.x = ps_.domain_max.x;
    any_periodic = true;
  }
  if (ps_.periodic_y) {
    bmin.y = ps_.domain_min.y;
    bmax.y = ps_.domain_max.y;
    any_periodic = true;
  }
  if (ps_.periodic_z) {
    bmin.z = ps_.domain_min.z;
    bmax.z = ps_.domain_max.z;
    any_periodic = true;
  }

  // 3. Export
  export_lammps_dump(filename, step, h_pos, h_vel, h_quat, h_radii, &bmin,
                     &bmax, any_periodic);
}

py::dict Simulation::get_profiling_info() {
  py::dict d;
  d["broadphase"] = time_broadphase_;
  d["solver"] = time_solver_;
  d["total"] = time_integration_ + time_broadphase_ + time_solver_;
  return d;
}

// -----------------------------------------------------------------------------
// Periodicity Helpers
// -----------------------------------------------------------------------------
int Simulation::calculate_capacity(int n_real, float3 box_size,
                                   float skin_width) {
  float3 inner = make_float3(fmaxf(0.0f, box_size.x - 2.0f * skin_width),
                             fmaxf(0.0f, box_size.y - 2.0f * skin_width),
                             fmaxf(0.0f, box_size.z - 2.0f * skin_width));
  double vol_total = (double)box_size.x * box_size.y * box_size.z;
  double vol_inner = (double)inner.x * inner.y * inner.z;
  if (vol_total <= 0.0)
    return n_real * 8; // Degenerate box

  double ghost_fraction = (vol_total - vol_inner) / vol_total;

  double estimated_ghosts = 0;
  if (ghost_fraction > 0.5) {
    estimated_ghosts =
        n_real * 32.0; // Small box: Allow for full 26-neighbor periodic ghosts
  } else {
    estimated_ghosts = n_real * ghost_fraction * 4.0; // 4x Safety
  }
  return n_real + (int)estimated_ghosts + 4096; // Add extra buffer
}

void Simulation::update_ghosts() {
  using namespace dem; // Use dem namespace for this function

  // 1. Reset Logic
  ps_.num_particles = ps_.num_real;

  // 2. Build Local BVH (Real particles only)
  int old_num = ps_.num_particles;
  ps_.num_particles = ps_.num_real;
  ps_.num_particles = ps_.num_real;

  build_bvh(ps_, global_scale_);
  CUDA_CHECK(cudaDeviceSynchronize()); // Check Local BVH Build

  // 3. Find Candidates
  PeriodicConfig config;
  config.min = ps_.domain_min;
  config.max = ps_.domain_max;
  config.size = ps_.domain_size;
  config.skin_width = 1.0f * global_scale_; // Use max radius + margin

  CUDA_CHECK(cudaMemset(ps_.d_potential_count, 0, sizeof(int)));

  dem::launch_find_ghost_candidates(ps_, config, ps_.d_bvh_indices,
                                    ps_.d_potential_count);
  CUDA_CHECK(cudaDeviceSynchronize()); // Check Candidate Find

  // 4. Generate Ghosts
  CUDA_CHECK(cudaMemcpy(ps_.d_top_ghost, &ps_.num_real, sizeof(int),
                        cudaMemcpyHostToDevice));

  dem::launch_generate_ghosts_bitmask(ps_.d_bvh_indices, ps_.d_potential_count,
                                      ps_, config);

  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel done

  // 5. Update Total Count
  int total = 0;
  CUDA_CHECK(
      cudaMemcpy(&total, ps_.d_top_ghost, sizeof(int), cudaMemcpyDeviceToHost));

  if (total > ps_.capacity) {
    printf("WARNING: Ghost buffer overflow (Req: %d, Cap: %d)\n", total,
           ps_.capacity);
    total = ps_.capacity;
  }
  ps_.num_particles = total;
}

void Simulation::export_sdf(const std::string &filename,
                            std::tuple<int, int, int> resolution) {
  dem::OutputGenerator generator(this);
  auto [rx, ry, rz] = resolution;

  // Bounds: use domain min/max expanded slightly?
  // Or just exact domain if periodic.
  // If periodic, exact domain is best.
  float3 min_b = ps_.domain_min;
  float3 max_b = ps_.domain_max;

  generator.generateAndSaveVTI(filename, make_int3(rx, ry, rz), min_b, max_b);
}

py::array_t<float>
Simulation::get_sdf_grid(std::tuple<int, int, int> resolution) {
  dem::OutputGenerator generator(this);
  auto [rx, ry, rz] = resolution;
  float3 min_b = ps_.domain_min;
  float3 max_b = ps_.domain_max;

  // Generate std::vector<float>
  std::vector<float> grid =
      generator.generateSDF(make_int3(rx, ry, rz), min_b, max_b);

  // Return as numpy array (copy)
  return py::array_t<float>({rx, ry, rz}, grid.data());
}

void Simulation::set_angular_velocities_numpy(py::array_t<float> ang_vel) {
  auto r = ang_vel.unchecked<2>();
  if (r.shape(0) != num_particles_ || r.shape(1) != 3) {
    throw std::runtime_error("Angular velocities array must be (N, 3)");
  }
  std::vector<float4> h_omega(num_particles_);
  for (int i = 0; i < num_particles_; ++i) {
    h_omega[i] = make_float4(r(i, 0), r(i, 1), r(i, 2), 0.0f);
  }

  CUDA_CHECK(cudaMemcpy(ps_.d_ang_vel, h_omega.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));

  // Sync predictions/accumulators
  CUDA_CHECK(cudaMemcpy(ps_.d_ang_vel_pred, ps_.d_ang_vel,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(
      cudaMemset(ps_.d_delta_ang_vel, 0, num_particles_ * sizeof(float4)));

  force_sync_ = true;
}

void Simulation::set_quaternions_numpy(py::array_t<float> quat) {
  auto r = quat.unchecked<2>();
  if (r.shape(0) != num_particles_ || r.shape(1) != 4) {
    throw std::runtime_error("Quaternions array must be (N, 4)");
  }
  std::vector<float4> h_quat(num_particles_);
  for (int i = 0; i < num_particles_; ++i) {
    h_quat[i] = make_float4(r(i, 0), r(i, 1), r(i, 2), r(i, 3));
  }

  CUDA_CHECK(cudaMemcpy(ps_.d_quat, h_quat.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));

  // Sync predictions to ensure consistent state
  CUDA_CHECK(cudaMemcpy(ps_.d_quat_pred, ps_.d_quat,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(ps_.d_quat_star, ps_.d_quat,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToDevice));

  force_sync_ = true;
}
