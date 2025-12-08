#pragma once
#include "cuda/ParticleSystem.cuh"
#include <memory>
#include <pybind11/numpy.h>
#include <vector>
#include <vector_types.h> // For float3

namespace py = pybind11;

class Simulation {
public:
  Simulation(int num_particles);
  ~Simulation();

  void initialize(int shape_type = 1); // 0=Sphere, 1=Cylinder
  void step(float dt);
  int num_particles() const { return num_particles_; }

  // Helpers // Python Bindings
  // Getters (copy to host)
  py::array_t<float> get_positions_numpy();
  py::array_t<float> get_quaternions_numpy();
  py::array_t<float> get_scales_numpy();
  py::array_t<float> get_velocities_numpy(); // New

  // Setters (copy from host)
  void set_scales_numpy(py::array_t<float> scales);
  void set_positions_numpy(py::array_t<float> pos);
  void set_velocities_numpy(py::array_t<float> vel);
  void set_gravity(float x, float y, float z);
  void set_global_scale(float s);

  // Domain Configuration
  void set_domain(float3 min, float3 max);
  std::tuple<float, float, float> get_domain_min();
  std::tuple<float, float, float> get_domain_max();

private:
  ParticleSystemData ps_;
  int num_particles_;
  float3 gravity_;
  float global_scale_;

  // Helper to re-allocate if needed or just zero out
  void allocate_system(int num_particles);
};
