#pragma once
#include "cuda/ParticleSystem.cuh"
#include <memory>
#include <vector>

class Simulation {
public:
  Simulation(int num_particles);
  ~Simulation();

  void initialize();
  void step(float dt);

  // Helpers for Python binding
  void get_positions_numpy(unsigned long h_ptr, int max_size);

private:
  int num_particles_;
  ParticleSystemData ps_; // Device pointers

  // Host side tracking if needed, or simply keeping pointers
  void allocate_state();
  void free_state();
};
