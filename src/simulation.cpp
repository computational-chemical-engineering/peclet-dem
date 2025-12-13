#include "simulation.h"
#include "cpp/OutputGenerator.hpp"
#include "cuda/math_utils.cuh"
#include "cuda/memory_utils.cuh"
#include "cuda/periodicity.cuh"
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
  ps_.d_planes = nullptr; // Initialize plane pointer
  ps_.d_max_overlap = nullptr;
  ps_.d_locks = nullptr;
  ps_.d_top_ghost = nullptr;

  // Initialize defaults
  gravity_ = make_float3(0, -9.8f, 0);
  global_scale_ = 1.0f;
  position_iterations_ = 10;
  velocity_iterations_ = 0;
  domain_initialized_ = false; // Flag for first set_domain call
  force_sync_ = true;          // Ensure first step is always synchronized

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
  if (ps_.d_planes) {
    free_device(ps_.d_planes);
  }
  free_device(ps_.d_locks); // Moved from separate if
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
extern void launch_velocity_solve(ParticleSystemData ps);
extern void launch_apply_velocity_deltas(ParticleSystemData ps);
extern void launch_compute_contact_counts(ParticleSystemData ps);
extern void launch_final_commit(ParticleSystemData ps, float dt);
void launch_generate_ghosts(ParticleSystemData ps, float margin);
extern void launch_narrowphase(ParticleSystemData ps, float global_scale);
extern void launch_velocity_solve(ParticleSystemData ps);
extern void launch_position_solve(ParticleSystemData ps);
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

  // Set Domain Defaults if not already set
  if (!domain_initialized_) {
    ps_.domain_min = make_float3(-5.0f, -5.0f, -5.0f);
    ps_.domain_max = make_float3(5.0f, 5.0f, 5.0f);
    ps_.domain_size = make_float3(10.0f, 10.0f, 10.0f);
    // Periodicity defaults (false) apply
  }
  // Periodicity flags are preserved from constructor/enable_periodicity
  // Defaults (false) or User-Set (via enable_periodicity) apply.

  allocate_device(ps_.d_pos, capacity * sizeof(float4));
  allocate_device(ps_.d_vel, capacity * sizeof(float4));
  // allocate_device(ps_.d_vel_old, capacity * sizeof(float4)); // Deprecated
  allocate_device(ps_.d_quat, capacity * sizeof(float4));
  allocate_device(ps_.d_scale, capacity * sizeof(float));
  allocate_device(ps_.d_ang_vel, capacity * sizeof(float4));
  allocate_device(ps_.d_shape_ids, capacity * sizeof(int));
  allocate_device(ps_.d_top_ghost, 1 * sizeof(int));

  allocate_device(ps_.d_pos_pred, capacity * sizeof(float4));
  allocate_device(ps_.d_quat_pred, capacity * sizeof(float4));
  allocate_device(ps_.d_vel_pred, capacity * sizeof(float4));
  allocate_device(ps_.d_ang_vel_pred, capacity * sizeof(float4));

  ps_.d_pos_star = ps_.d_pos_pred; // Alias for legacy if needed
  ps_.d_quat_star = ps_.d_quat_pred;

  allocate_device(ps_.d_delta_pos, capacity * sizeof(float4));
  allocate_device(ps_.d_delta_vel, capacity * sizeof(float4));
  allocate_device(ps_.d_delta_quat, capacity * sizeof(float4));
  allocate_device(ps_.d_delta_ang_vel, capacity * sizeof(float4));

  allocate_device(ps_.d_constraint_counts, capacity * sizeof(int));
  allocate_device(ps_.d_inv_inertia, capacity * sizeof(float4));

  // Collision Buffers
  // Estimate: 50 neighbors per particle
  int max_potential = capacity * 50;
  ps_.max_potential_collisions = max_potential;
  allocate_device(ps_.d_potential_collisions, max_potential * sizeof(int2));
  allocate_device(ps_.d_potential_count, 1 * sizeof(int));

  // BVH Buffers
  allocate_device(ps_.d_bvh_nodes, (2 * capacity - 1) * sizeof(BVHNode));
  allocate_device(ps_.d_bvh_indices, capacity * sizeof(int));
  allocate_device(ps_.d_morton, capacity * sizeof(unsigned int));
  allocate_device(ps_.d_indices_sorted, capacity * sizeof(int));

  // Contact Buffers
  int max_contacts = capacity * 50; // Max contacts for solver
  ps_.max_contacts = max_contacts;
  allocate_device(ps_.d_contacts, max_contacts * sizeof(ContactConstraint));
  allocate_device(ps_.d_contact_count, 1 * sizeof(int));

  // Locks
  allocate_device(ps_.d_locks, capacity * sizeof(int));
  allocate_device(ps_.d_max_overlap, 1 * sizeof(float));
  CUDA_CHECK(cudaMemset(ps_.d_locks, 0, capacity * sizeof(int)));

  // Arrays for Shapes
  // 0=Sphere, 1=Cylinder

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
    // SPHERE (ID=1)  -- Defaulting to Sphere if not Cylinder
    // Force type to 1 if it was 0 or something else, effectively default
    if (type != SHAPE_ANALYTIC_SPHERE && type != SHAPE_ANALYTIC_HOLLOW_CYLINDER)
      type = SHAPE_ANALYTIC_SPHERE;

    // Use radius = passed radius (default 0.5? User expects 1.0 for sphere
    // usually?) Previously hardcoded to 1.0f. Let's keep 1.0f for canonical
    // sphere and let scale handle size. Or use 'radius' arg? If I change this,
    // I might break assumptions. But 'params' for sphere in `sdf_sphere` is
    // `params.x`.
    // Use radius arg for Sphere Parameters
    // params = make_float4(radius, 0, 0, 0);
    // FIX: Use Canonical Sphere R=1.0 so that 'scale' controls size 1:1.
    // Previously, we set param=0.5 AND scale=0.5 -> Effective R=0.25 (Too
    // Small).
    params = make_float4(1.0f, 0, 0, 0);
    // Wait, if I change radius here, 'generate_packing_sdf' (phi=0.45) relying
    // on R=1.0 (internal) + Scale 0.5 (external) might break? No,
    // 'generate_packing_sdf' relies on scale. Let's stick to Canonical Sphere
    // R=1.0. Generate Fibonacci sphere points
    // Generate Fibonacci sphere points
    // Optimization: Skip points for Sphere to use Analytic Collision
    // (Center-Center) int num_sphere_points = 500; float phi = 3.14159265f *
    // (3.0f - sqrtf(5.0f)); for (int i = 0; i < num_sphere_points; ++i) {
    //   float y = 1.0f - (i / (float)(num_sphere_points - 1)) * 2.0f;
    //   float radius = sqrtf(1.0f - y * y);
    //   float theta = phi * i;
    //   float x = cosf(theta) * radius;
    //   float z = sinf(theta) * radius;
    //   h_points.push_back(make_float4(x * 1.0f, y * 1.0f, z * 1.0f, 0));
    // }
    // Points vector remains empty. ShapeManager will set num_points=0.
    // Narrowphase will use 'is_sphere_A' branch.
  }

  // Create Shape via Manager
  int created_id = shape_manager_.createAnalyticShape(type, params, h_points);
  printf("Created Shape ID %d via Manager. Points: %lu\n", created_id,
         h_points.size());

  // Upload to GPU
  shape_manager_.uploadToGPU();

  // Link Device Pointer
  ps_.d_shapes = shape_manager_.getDeviceShapes();

  printf("Simulation initialized. Seeding particles...\n");
  // Initialize Max Overlap
  CUDA_CHECK(cudaMemset(ps_.d_max_overlap, 0, sizeof(float)));

  // 1. Initialize Particles
  std::vector<float4> h_pos(num_particles_);
  std::vector<float4> h_vel(num_particles_);
  std::vector<float4> h_quat(num_particles_);
  std::vector<float> h_scale(num_particles_);
  std::vector<int> h_shape_ids(num_particles_);

  // printf("DEBUG: Initialization. N=%d. Allocating buffers...\n",
  // num_particles_);

  // Grid initialization
  // Random initialization to avoid grid alignment
  int idx = 0;
  for (int i = 0; i < num_particles_; ++i) {
    float x = (float)rand() / RAND_MAX * ps_.domain_size.x + ps_.domain_min.x;
    float y = (float)rand() / RAND_MAX * ps_.domain_size.y + ps_.domain_min.y;
    float z = (float)rand() / RAND_MAX * ps_.domain_size.z + ps_.domain_min.z;

    h_pos[i] = make_float4(x, y, z, 1.0f);
    h_vel[i] = make_float4(0, 0, 0, 0.0f);
    h_quat[i] = make_float4(0, 0, 0, 1);
    if (shape_type == 2) {
      h_scale[i] = 1.0f;
    } else {
      h_scale[i] = radius;
    }
    h_shape_ids[i] = 0; // Shape ID (will be updated by Manager)
  }

  // Create default inertia (1.0)
  std::vector<float> h_inv_inertia(num_particles_, 1.0f);

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
                        num_particles_ * sizeof(float),
                        cudaMemcpyHostToDevice));
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
  allocate_device(ps_.d_top_ghost, 1 * sizeof(int));

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

void Simulation::step(float dt) {
  // DEBUG: Inspect Particle 0 on GPU
  // Synchronization to prevent kernel launch race conditions (Heisenbug fix)
  // Synchronization to prevent kernel launch race conditions (Heisenbug fix)
  // Optimization: Only sync if flagged (e.g. after data upload)
  if (force_sync_) {
    CUDA_CHECK(cudaDeviceSynchronize());
    force_sync_ = false;
  }

  // New Pipeline (Hybrid Velocity/Position)

  // 1. Predict Velocity (Gravity Only)
  launch_integrate_predict_velocity(ps_, dt, gravity_);

  // 2. Generate Ghosts
  if (ps_.periodic_x || ps_.periodic_y || ps_.periodic_z) {
    update_ghosts();
  }

  // 3. Broadphase & Narrowphase
  build_bvh(ps_, global_scale_);
  find_collisions(ps_, 1.0f);
  launch_narrowphase(ps_, global_scale_);

  // 4. Phase A: Velocity Solve

  // A. Pre-Pass: Count Contacts for Min-Scaling Weighting
  // (d_constraint_counts is reused to store N for each particle)
  CUDA_CHECK(
      cudaMemset(ps_.d_constraint_counts, 0, ps_.num_particles * sizeof(int)));
  launch_compute_contact_counts(ps_);

  // B. Iterative Solve (Velocity-First)
  for (int i = 0; i < velocity_iterations_; ++i) {
    launch_velocity_solve(ps_);
    launch_apply_velocity_deltas(ps_);
  }

  // 5. Apply Velocity & Re-Integrate Position
  // Note: d_delta_vel is now cleared. This function effectively acts as
  // "Predict Position from v_pred".
  launch_apply_velocity_and_predict_position(ps_, dt);

  // 6. Phase B: Position Solve (Projected Jacobi)
  for (int i = 0; i < position_iterations_; ++i) {
    CUDA_CHECK(cudaMemset(ps_.d_max_overlap, 0, sizeof(float)));
    launch_position_solve(ps_);
    launch_apply_updates(ps_);
  }

  // 7. Final Commit
  launch_final_commit(ps_, dt);
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
    // Radius = scale * global_scale * 1.0 (Unit Sphere)
    float r = h_scale[i] * global_scale_;
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

py::dict Simulation::get_profiling_info() {
  py::dict d;
  d["integration"] = time_integration_;
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
    estimated_ghosts = n_real * 7.0; // Small box
  } else {
    estimated_ghosts = n_real * ghost_fraction * 2.0; // 2x Safety
  }
  return n_real + (int)estimated_ghosts + 1024;
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

  // 3. Find Candidates
  PeriodicConfig config;
  config.min = ps_.domain_min;
  config.max = ps_.domain_max;
  config.size = ps_.domain_size;
  config.skin_width = 1.0f * global_scale_; // Use max radius + margin

  CUDA_CHECK(cudaMemset(ps_.d_potential_count, 0, sizeof(int)));

  dem::launch_find_ghost_candidates(ps_, config, ps_.d_bvh_indices,
                                    ps_.d_potential_count);

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

void Simulation::set_quaternions_numpy(py::array_t<float> quat) {
  auto r = quat.unchecked<2>();
  if (r.shape(0) != num_particles_ || r.shape(1) != 4) {
    throw std::runtime_error("Quaternions array must be (N, 4)");
  }
  std::vector<float4> h_quat(num_particles_);
  for (int i = 0; i < num_particles_; ++i) {
    h_quat[i] = make_float4(r(i, 0), r(i, 1), r(i, 2), r(i, 3));
  }
  allocate_device(ps_.d_quat, num_particles_ * sizeof(float4));
  CUDA_CHECK(cudaMemcpy(ps_.d_quat, h_quat.data(),
                        num_particles_ * sizeof(float4),
                        cudaMemcpyHostToDevice));
}
