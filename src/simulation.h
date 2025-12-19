#pragma once
#include "cuda/ParticleSystem.cuh"
#include "shapes/ShapeManager.hpp" // Added
#include <memory>
#include <pybind11/numpy.h>
#include <vector>
#include <vector_types.h> // For float3

namespace py = pybind11;
namespace dem {
class OutputGenerator;
}

class Simulation {
public:
  friend class dem::OutputGenerator; // Allow access to private members
  Simulation(int num_particles);
  ~Simulation();

  void initialize(int shape_type, float radius = 0.5f, float height = 2.0f,
                  float thickness = 0.2f);
  void step(float dt);
  int num_particles(bool include_ghosts = false) const {
    if (include_ghosts)
      return ps_.num_particles;
    return num_particles_;
  }

  // Helpers // Python Bindings
  // Getters (copy to host)
  py::array_t<float> get_positions_numpy(bool include_ghosts = false);
  py::array_t<float> get_quaternions_numpy();
  py::array_t<float> get_scales_numpy();
  py::array_t<float> get_velocities_numpy();
  py::array_t<float> get_angular_velocities_numpy(); // New
  py::array_t<float> get_inv_inertia_numpy();        // New
  py::array_t<float> get_masses_numpy();             // New

  // Setters (copy from host)
  void set_scales_numpy(py::array_t<float> scales);
  void set_positions_numpy(py::array_t<float> pos);
  void set_velocities_numpy(py::array_t<float> vel);
  void set_angular_velocities_numpy(py::array_t<float> ang_vel); // Added
  void set_quaternions_numpy(py::array_t<float> quat);           // Added
  void set_gravity(float x, float y, float z);
  void set_global_scale(float s);

  // Material Parameters
  void set_material_params(float restitution_normal, float restitution_tangent,
                           float friction_dynamic);
  void set_solver_iterations(int pos_its, int vel_its);

  // Growth Mode
  void set_growth_params(float rate, float new_factor = -1.0f);
  float get_growth_rate() const { return ps_.growth_rate; }
  float get_growth_factor() const { return ps_.growth_factor; }

  // Periodicity
  void enable_periodicity(bool x, bool y, bool z);
  void add_plane(float3 point, float3 normal); // Explicit Planes

  // Domain Configuration
  void set_domain(float3 min, float3 max);
  std::tuple<float, float, float> get_domain_min();
  std::tuple<float, float, float> get_domain_max();

  // Visualization
  void write_vtp(const std::string &filename) const;

  // Profiling
  // Profiling
  py::dict get_profiling_info();

  // Export SDF
  void export_sdf(const std::string &filename,
                  std::tuple<int, int, int> resolution);
  py::array_t<float> get_sdf_grid(std::tuple<int, int, int> resolution);

  // Export LAMMPS (Hybrid Bounds)
  void export_lammps(const std::string &filename, int step);

  float get_max_overlap();
  float compute_overlaps(); // Re-runs collision detection on current state
  int get_num_contacts();   // Returns number of contacts from last step
  int get_num_manifolds();  // Returns number of manifolds from last step

private:
  ParticleSystemData ps_;
  int num_particles_;
  float3 gravity_;
  float global_scale_;
  dem::ShapeManager shape_manager_;
  int position_iterations_;
  int velocity_iterations_;

  // Helper to re-allocate if needed or just zero out
  // Helpers

  // Profiling Events
  cudaEvent_t start_event_, stop_event_;
  cudaEvent_t integration_start_, integration_stop_;
  cudaEvent_t broadphase_start_, broadphase_stop_;
  cudaEvent_t solver_start_, solver_stop_;

  float time_integration_ = 0.0f;
  float time_broadphase_ = 0.0f;
  float time_solver_ = 0.0f;

  // Periodic Helpers
  int calculate_capacity(int n_real, float3 box_size, float skin_width);
  void update_ghosts();

  // Data for planes
  std::vector<Plane> planes_host_;

  // State
  bool domain_initialized_;
  bool force_sync_; // Flag to force synchronization on next step (e.g. after
                    // set_positions)
  float base_radius_;
};
