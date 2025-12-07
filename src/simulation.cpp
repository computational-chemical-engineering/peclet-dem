#include "simulation.h"
#include "cuda/memory_utils.cuh"
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

void Simulation::initialize() {
  printf("Simulation initialized. Seeding particles...\n");

  std::vector<float4> h_pos(num_particles_);
  std::vector<float4> h_vel(num_particles_);

  // Grid initialization
  int dim = ceil(pow(num_particles_, 1.0 / 3.0));
  float spacing = 1.2f;
  int idx = 0;
  for (int x = 0; x < dim; ++x) {
    for (int y = 0; y < dim; ++y) {
      for (int z = 0; z < dim; ++z) {
        if (idx >= num_particles_)
          break;
        h_pos[idx] = make_float4(x * spacing - 5.0f, y * spacing + 5.0f,
                                 z * spacing - 5.0f, 1.0f); // w=inv_mass=1
        h_vel[idx] = make_float4(0, 0, 0, 1.0f);            // w=active
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
}

void Simulation::step(float dt) {
  float3 gravity = make_float3(0.0f, -9.8f, 0.0f);

  launch_integration(ps_, dt, gravity);

  // Broadphase (Placeholder)
  // build_bvh(ps_);

  // Solver
  int solver_substeps = 1; // Just 1 for now
  for (int i = 0; i < solver_substeps; ++i) {
    launch_solver(ps_, dt / solver_substeps);
  }

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
