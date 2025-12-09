#include "simulation.h"
#include "cuda/memory_utils.cuh"
#include "shapes/point_sampler.h"
#include "simulation.h"
#include <cmath>
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

  // Initialize defaults
  gravity_ = make_float3(0, -9.8f, 0);
  global_scale_ = 1.0f;

  allocate_system(num_particles_);

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

  free_device(ps_.d_contact_count); // Renamed from d_num_contacts
  free_device(ps_.d_locks);         // Moved from separate if
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
  if (ps_.d_shapes)
    free_device(ps_.d_shapes);
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

void Simulation::allocate_system(int num_particles) {
  // Deprecated. All allocation moved to initialize() to handle capacity
  // scaling.
  num_particles_ = num_particles;
}

// External kernel wrappers
// External kernel wrappers
extern void launch_integrate_predict(ParticleSystemData ps, float dt,
                                     float3 gravity);
extern void launch_apply_updates(ParticleSystemData ps);
extern void launch_final_commit(ParticleSystemData ps, float dt);
void launch_generate_ghosts(ParticleSystemData ps, float margin);
extern void launch_narrowphase(ParticleSystemData ps, float global_scale);
extern void launch_velocity_solve(ParticleSystemData ps);
extern void launch_position_solve(ParticleSystemData ps);
void build_bvh(ParticleSystemData &ps, float global_scale);
void find_collisions(ParticleSystemData ps, float global_scale);
void generate_ghosts(ParticleSystemData &ps, float threshold);

// 0=Sphere, 1=Cylinder
void Simulation::initialize(int shape_type) {
  // Allocate more for Ghosts (e.g., 8.0x for robust periodicity)
  int capacity = num_particles_ * 8;
  ps_.capacity = capacity;
  ps_.num_real = num_particles_;
  ps_.num_particles = num_particles_;

  // Set Domain
  ps_.domain_min = make_float3(-5.0f, -5.0f, -5.0f);
  ps_.domain_max = make_float3(5.0f, 5.0f, 5.0f);
  ps_.domain_size = make_float3(10.0f, 10.0f, 10.0f);
  ps_.periodic_x = true;
  ps_.periodic_y = true;
  ps_.periodic_z = true;

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
  allocate_device(ps_.d_inv_inertia, capacity * sizeof(float));

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
  CUDA_CHECK(cudaMemset(ps_.d_locks, 0, capacity * sizeof(int)));

  // Arrays for Shapes
  // 0=Sphere, 1=Cylinder

  CylinderParams cyl_params;
  cyl_params.radius = 0.5f;
  cyl_params.height = 2.0f;
  cyl_params.thickness = 0.2f;

  std::vector<float4> h_points;
  ShapeData h_shape;
  h_shape.type = shape_type;

  if (shape_type == 1) {
    // CYLINDER
    h_shape.params = make_float4(cyl_params.radius, cyl_params.height,
                                 cyl_params.thickness, 0);
    h_points = generate_cylinder_points(cyl_params, 0.1f);
  } else {
    // SPHERE
    // Use radius = 1.0
    h_shape.params = make_float4(1.0f, 0, 0, 0);
    // Generate Fibonacci sphere points for robust Point-SDF testing
    // Or just random points
    int num_sphere_points = 500;
    float phi = 3.14159265f * (3.0f - sqrtf(5.0f));
    for (int i = 0; i < num_sphere_points; ++i) {
      float y = 1.0f - (i / (float)(num_sphere_points - 1)) * 2.0f;
      float radius = sqrtf(1.0f - y * y);
      float theta = phi * i;
      float x = cosf(theta) * radius;
      float z = sinf(theta) * radius;
      // Scale by 1.0
      h_points.push_back(make_float4(x * 1.0f, y * 1.0f, z * 1.0f, 0));
    }
  }

  int num_points = h_points.size();
  printf("Generated %d points for shape type %d.\n", num_points, shape_type);

  // Allocate Fine Points on GPU
  float4 *d_fine_points;
  allocate_device(d_fine_points, num_points * sizeof(float4));
  CUDA_CHECK(cudaMemcpy(d_fine_points, h_points.data(),
                        num_points * sizeof(float4), cudaMemcpyHostToDevice));

  // Prepare ShapeData
  h_shape.d_fine_points = d_fine_points;
  h_shape.num_points = num_points;
  // d_proxies, sdf_texture, tile_descriptors can be null for now if we use raw
  // points

  // Allocate ShapeData Array on GPU (Size 1)
  allocate_device(ps_.d_shapes, 1 * sizeof(ShapeData));
  CUDA_CHECK(cudaMemcpy(ps_.d_shapes, &h_shape, sizeof(ShapeData),
                        cudaMemcpyHostToDevice));

  printf("Simulation initialized. Seeding particles...\n");

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
    h_scale[i] = 1.0f;
    h_shape_ids[i] = 0;
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
  ps_.periodic_x = true;
  ps_.periodic_y = true;
  ps_.periodic_z = true;
}

std::tuple<float, float, float> Simulation::get_domain_min() {
  return std::make_tuple(ps_.domain_min.x, ps_.domain_min.y, ps_.domain_min.z);
}

std::tuple<float, float, float> Simulation::get_domain_max() {
  return std::make_tuple(ps_.domain_max.x, ps_.domain_max.y, ps_.domain_max.z);
}

void Simulation::step(float dt) {
  // New Pipeline

  // 1. Integration & Prediction (Gravity)
  // Initializes pos_pred, vel_pred, quat_pred from current state + gravity * dt
  // Also CLEARS accumulators (d_delta_*) and contact_count
  // printf("DEBUG: Launch Integrate Predict. N=%d, dt=%f\n", ps_.num_particles,
  // dt);
  launch_integrate_predict(ps_, dt, gravity_);
  CUDA_CHECK(cudaDeviceSynchronize());

  // 1a. Generate Ghosts (Periodic)
  CUDA_CHECK(cudaMemcpy(ps_.d_top_ghost, &ps_.num_real, sizeof(int),
                        cudaMemcpyHostToDevice));
  float margin = 1.0f * global_scale_;
  launch_generate_ghosts(ps_, margin);

  // Read back total particles
  int total_particles = 0;
  CUDA_CHECK(cudaMemcpy(&total_particles, ps_.d_top_ghost, sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (total_particles > ps_.capacity)
    total_particles = ps_.capacity;
  ps_.num_particles = total_particles;

  // 1b. Broadphase + Narrowphase
  build_bvh(ps_, global_scale_);
  find_collisions(ps_, 1.0f);             // Generates d_potential_collisions
  launch_narrowphase(ps_, global_scale_); // Generates d_contacts

  // 2. Velocity Solver (Projected Jacobi)
  int velocity_iterations = 5; // Start small
  for (int i = 0; i < velocity_iterations; ++i) {
    launch_velocity_solve(ps_);
    // If we had multi-pass Jacobi, we'd apply accumulated deltas here or
    // accumulate to a temp buffer. Current implementation accumulates to
    // d_delta_vel directly. But Jacobi usually accumulates locally then adds.
    // My kernel accumulates to global buffer atomically.
    // Then Apply Updates adds them to pred state?
    // Wait. Velocity Solver should update Velocity PREDICTION?
    // Plan said: "Accumulate J -> d_delta_vel".
    // "Apply Updates: v_pred += d_delta_vel / M".
    // But if we iterate, we need v_pred to be updated?
    // If we don't update v_pred between iters, it's just one big step
    // (parallel). Standard Jacobi: Compute all deltas based on State K. Update
    // -> State K+1. So:
    launch_apply_updates(ps_); // Update v_pred
  }

  // 3. Position Solver (Projected Jacobi)
  int position_iterations = 10;
  for (int i = 0; i < position_iterations; ++i) {
    launch_position_solve(ps_);
    launch_apply_updates(ps_); // Update pos_pred, quat_pred
  }

  // 4. Final Commit
  launch_final_commit(ps_, dt);

  // Timing
  // cudaEventSynchronize(solver_stop_);
  // cudaEventElapsedTime(&time_integration_, integration_start_,
  //                      integration_stop_);
  // cudaEventElapsedTime(&time_broadphase_, broadphase_start_,
  // broadphase_stop_); cudaEventElapsedTime(&time_solver_, solver_start_,
  // solver_stop_);
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
}

void Simulation::set_scales_numpy(py::array_t<float> scales) {
  py::buffer_info buf = scales.request();
  if (buf.size != num_particles_) {
    throw std::runtime_error("Scales must be size N");
  }
  float *ptr = static_cast<float *>(buf.ptr);
  CUDA_CHECK(cudaMemcpy(ps_.d_scale, ptr, num_particles_ * sizeof(float),
                        cudaMemcpyHostToDevice));
}

py::array_t<float> Simulation::get_positions_numpy() {
  std::vector<float4> h_pos(num_particles_);
  CUDA_CHECK(cudaMemcpy(h_pos.data(), ps_.d_pos,
                        num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));

  py::array_t<float> result({num_particles_, 3});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < num_particles_; ++i) {
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
  printf("DEBUG: get_scales for %d particles. d_scale: %p. First val: %f\n",
         num_particles_, ps_.d_scale, ptr[0]);
  return result;
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
