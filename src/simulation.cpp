#include "simulation.h"
#include "cuda/memory_utils.cuh"
#include "shapes/point_sampler.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector_functions.h>
#include <vector_types.h>

Simulation::Simulation(int num_particles) : num_particles_(num_particles) {
  // Initialize pointers to nullptr
  ps_.d_pos = nullptr;
  ps_.d_quat = nullptr;
  ps_.d_vel = nullptr;
  ps_.d_ang_vel = nullptr;
  ps_.d_pos_star = nullptr;
  ps_.d_quat_star = nullptr;
  ps_.d_delta_pos = nullptr;
  ps_.d_constraint_counts = nullptr;
  ps_.d_inv_inertia = nullptr;
  ps_.d_shape_ids = nullptr;

  allocate_state();
}

Simulation::~Simulation() { free_state(); }

void Simulation::allocate_state() {
  allocate_device(ps_.d_pos, num_particles_);
  allocate_device(ps_.d_quat, num_particles_);
  allocate_device(ps_.d_scale, num_particles_);
  allocate_device(ps_.d_vel, num_particles_);
  allocate_device(ps_.d_ang_vel, num_particles_);

  allocate_device(ps_.d_pos_star, num_particles_);
  allocate_device(ps_.d_quat_star, num_particles_);
  allocate_device(ps_.d_delta_pos, num_particles_);
  allocate_device(ps_.d_constraint_counts, num_particles_);

  allocate_device(ps_.d_inv_inertia, num_particles_);
  allocate_device(ps_.d_shape_ids, num_particles_);

  ps_.num_particles = num_particles_;
  // Default domain - can be exposed later
  ps_.domain_min = make_float3(-10.0f, -10.0f, -10.0f);
  ps_.domain_size = make_float3(20.0f, 20.0f, 20.0f);

  printf("Allocated memory for %d particles.\n", num_particles_);
}

void Simulation::free_state() {
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
}

// External kernel wrappers
void launch_integration(ParticleSystemData &ps, float dt, float3 gravity);
void launch_update(ParticleSystemData &ps, float dt);
void launch_solver(ParticleSystemData &ps, float dt);
void build_bvh(ParticleSystemData &ps); // Placeholder
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

  // Collision Buffers
  // Estimate: 50 neighbors per particle
  int max_potential = capacity * 50;
  ps_.max_potential_collisions = max_potential;
  allocate_device(ps_.d_potential_collisions, max_potential * sizeof(int2));
  allocate_device(ps_.d_potential_count, 1 * sizeof(int));

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
    // Use radius = 0.5
    h_shape.params = make_float4(0.5f, 0, 0, 0);
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
      // Scale by 0.5
      h_points.push_back(make_float4(x * 0.5f, y * 0.5f, z * 0.5f, 0));
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

void Simulation::step(float dt) {
  float3 gravity = make_float3(0.0f, -9.8f, 0.0f);

  // 1. Integration (Predict) - ONLY Real particles
  // We need to ensure kernels respect ps.num_real
  // But currently integration wrapper might use ps.num_particles.
  // Ideally we pass strict count, or update ps.num_particles to num_real
  // before.

  // Reset for Integration
  ps_.num_particles = ps_.num_real;
  launch_integration(ps_, dt, gravity);

  // 2. Generate Ghosts
  generate_ghosts(ps_, 1.0f); // Threshold = interaction radius (approx)

  // 3. Broadphase (Build BVH with ALL particles)
  build_bvh(ps_);

  // 4. Solver
  int solver_substeps = 10;
  for (int i = 0; i < solver_substeps; ++i) {
    launch_solver(ps_, dt / solver_substeps);
  }

  // 5. Update Velocity - ONLY Real
  // Note: Solver might have moved Ghosts. We don't care. We only update Real.
  // But if we used Ghosts in constraints, the forces/deltas on Real particles
  // are what matters. XPBD modifies d_pos_star. The solver should update Real
  // d_pos_star based on contacts with Ghosts.

  // Ideally update just real
  ps_.num_particles = ps_.num_real;
  launch_update(ps_, dt);
}

void Simulation::get_positions_numpy(unsigned long h_ptr, int max_size) {
  if (max_size < num_particles_ * 4) { // float4 components
    fprintf(stderr, "Buffer too small for positions\n");
    return;
  }
  // Copy d_pos to host pointer
  float *host_ptr = reinterpret_cast<float *>(h_ptr);
  CUDA_CHECK(cudaMemcpy(host_ptr, ps_.d_pos, num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
}

void Simulation::get_quaternions_numpy(unsigned long h_ptr, int max_size) {
  if (max_size < num_particles_ * 4) {
    fprintf(stderr, "Buffer too small for quaternions\n");
    return;
  }
  float *host_ptr = reinterpret_cast<float *>(h_ptr);
  CUDA_CHECK(cudaMemcpy(host_ptr, ps_.d_quat, num_particles_ * sizeof(float4),
                        cudaMemcpyDeviceToHost));
}

void Simulation::get_scales_numpy(unsigned long h_ptr, int max_size) {
  if (max_size < num_particles_) {
    fprintf(stderr, "Buffer too small for scales\n");
    return;
  }
  float *host_ptr = reinterpret_cast<float *>(h_ptr);
  CUDA_CHECK(cudaMemcpy(host_ptr, ps_.d_scale, num_particles_ * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void Simulation::set_scales_numpy(unsigned long h_ptr, int max_size) {
  if (max_size < num_particles_) {
    fprintf(stderr, "Buffer too small for setting scales\n");
    return;
  }
  float *host_ptr = reinterpret_cast<float *>(h_ptr);
  CUDA_CHECK(cudaMemcpy(ps_.d_scale, host_ptr, num_particles_ * sizeof(float),
                        cudaMemcpyHostToDevice));
}
