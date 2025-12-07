#pragma once
#include "cuda/ParticleSystem.cuh"
#include <memory>
#include <vector>

class Simulation {
public:
  Simulation(int num_particles);
  ~Simulation();

  void initialize(int shape_type = 1); // 0=Sphere, 1=Cylinder
  void step(float dt);
  int num_particles() const { return num_particles_; }

  // Helpers // Python Bindings
  void get_positions_numpy(unsigned long h_ptr, int max_size);
  void get_quaternions_numpy(unsigned long h_ptr, int max_size);
  void get_scales_numpy(unsigned long h_ptr, int max_size);
  void set_scales_numpy(unsigned long h_ptr, int max_size);

private:
  int num_particles_;
  ParticleSystemData ps_; // Device pointers

  // Host side tracking if needed, or simply keeping pointers
  void allocate_state();
  void free_state();
};
