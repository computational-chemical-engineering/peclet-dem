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
  ps_.d_delta_pos = nullptr;
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
  ps_.d_num_contacts = nullptr;

  // Initialize defaults
  gravity_ = make_float3(0, -9.8f, 0);
  global_scale_ = 1.0f;

  allocate_system(num_particles_);

  cudaEventCreate(&start_event_);
  cudaEventCreate(&stop_event_);
  cudaEventCreate(&integration_start_);
  cudaEventCreate(&integration_stop_);
  cudaEventCreate(&broadphase_start_);
  cudaEventCreate(&broadphase_stop_);
  cudaEventCreate(&solver_start_);
  cudaEventCreate(&solver_stop_);
}

Simulation::~Simulation() {
  free_device(ps_.d_pos);
  free_device(ps_.d_quat);
  free_device(ps_.d_scale);
  free_device(ps_.d_vel);
  free_device(ps_.d_ang_vel);

  free_device(ps_.d_pos_star);
  free_device(ps_.d_quat_star);
  free_device(ps_.d_delta_pos);
  free_device(ps_.d_constraint_counts);

  free_device(ps_.d_inv_inertia);
  free_device(ps_.d_shape_ids);

  // Cleanup collision buffers if allocated
  if (ps_.d_contacts)
    free_device(ps_.d_contacts);
  if (ps_.d_num_contacts)
    free_device(ps_.d_num_contacts);

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

  cudaEventDestroy(start_event_);
  cudaEventDestroy(stop_event_);
  cudaEventDestroy(integration_start_);
  cudaEventDestroy(integration_stop_);
  cudaEventDestroy(broadphase_start_);
  cudaEventDestroy(broadphase_stop_);
  cudaEventDestroy(solver_start_);
  cudaEventDestroy(solver_stop_);
}

void Simulation::allocate_system(int num_particles) {
  // Deprecated. All allocation moved to initialize() to handle capacity
  // scaling.
  num_particles_ = num_particles;
}

// External kernel wrappers
void launch_integration(ParticleSystemData &ps, float dt, float3 gravity);
void launch_update(ParticleSystemData &ps, float dt);
void launch_solver(ParticleSystemData &ps, float dt, float global_scale,
                   int offset);
void build_bvh(ParticleSystemData &ps, float global_scale);
void find_collisions(ParticleSystemData &ps, float global_scale);
void generate_ghosts(ParticleSystemData &ps, float threshold);

// 0=Sphere, 1=Cylinder
void Simulation::initialize(int shape_type) {
  // Allocate more for Ghosts (e.g., 2.0x)
  int capacity = num_particles_ * 2;
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
  allocate_device(ps_.d_pos_star, capacity * sizeof(float4));
  allocate_device(ps_.d_quat, capacity * sizeof(float4));
  allocate_device(ps_.d_scale, capacity * sizeof(float));
  allocate_device(ps_.d_ang_vel, capacity * sizeof(float4));
  allocate_device(ps_.d_shape_ids, capacity * sizeof(int));

  // Missing allocations added:
  allocate_device(ps_.d_quat_star, capacity * sizeof(float4));
  allocate_device(
      ps_.d_delta_pos,
      capacity *
          sizeof(float4)); // float4 for delta? Or float3? Wrapper uses float4.
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
  allocate_device(ps_.d_contacts, max_contacts * sizeof(int2));
  allocate_device(ps_.d_num_contacts, 1 * sizeof(int));

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

  // Grid initialization
  int dim = ceil(pow(num_particles_, 1.0 / 3.0));
  float spacing_grid = 2.5f; // Increased spacing for cylinders
  int idx = 0;
  for (int x = 0; x < dim; ++x) {
    for (int y = 0; y < dim; ++y) {
      for (int z = 0; z < dim; ++z) {
        if (idx >= num_particles_)
          break;
        h_pos[idx] =
            make_float4(x * spacing_grid - 4.0f, y * spacing_grid - 4.0f,
                        z * spacing_grid - 4.0f, 1.0f); // Centered
        h_vel[idx] = make_float4(0, 0, 0, 0.0f);        // .w=0 (Real)

        // Random orientation? For now Identity
        h_quat[idx] = make_float4(0, 0, 0, 1);
        h_scale[idx] = 1.0f;
        h_shape_ids[idx] = 0; // Point to Cylinder
        idx++;
      }
    }
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

void Simulation::set_domain(float3 min, float3 max) {
  ps_.domain_min = min;
  ps_.domain_max = max;
  ps_.domain_size = make_float3(max.x - min.x, max.y - min.y, max.z - min.z);
}

std::tuple<float, float, float> Simulation::get_domain_min() {
  return std::make_tuple(ps_.domain_min.x, ps_.domain_min.y, ps_.domain_min.z);
}

std::tuple<float, float, float> Simulation::get_domain_max() {
  return std::make_tuple(ps_.domain_max.x, ps_.domain_max.y, ps_.domain_max.z);
}

void Simulation::step(float dt) {
  float3 gravity = make_float3(0.0f, -9.8f, 0.0f);

  ps_.num_particles = ps_.num_real;

  // 1. Integration
  cudaEventRecord(integration_start_);
  launch_integration(ps_, dt, gravity_);
  // Ghost Generation
  generate_ghosts(ps_, 1.0f);
  cudaEventRecord(integration_stop_);

  // 3. Broadphase
  cudaEventRecord(broadphase_start_);
  build_bvh(ps_, global_scale_);

  // Clear contacts
  cudaMemset(ps_.d_contacts, 0, ps_.max_contacts * sizeof(int2));
  cudaMemset(ps_.d_num_contacts, 0, sizeof(int));

  find_collisions(ps_, 1.0f); // search_radius = 1.0 (Unit Sphere)
  cudaEventRecord(broadphase_stop_);

  // 4. Solver
  cudaEventRecord(solver_start_);
  int substeps = 50; // Increased to 50 for stability at 0.64 density
  float sub_dt = dt / substeps;

  launch_solver(ps_, sub_dt, substeps, global_scale_);
  cudaEventRecord(solver_stop_);

  // 5. Update Velocity (Finalize)
  launch_update(ps_, dt);

  // Profile Sync
  cudaEventSynchronize(solver_stop_);
  cudaEventElapsedTime(&time_integration_, integration_start_,
                       integration_stop_);
  cudaEventElapsedTime(&time_broadphase_, broadphase_start_, broadphase_stop_);
  cudaEventElapsedTime(&time_solver_, solver_start_, solver_stop_);
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
