#include "simulation.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>  // ArborX broad-phase backend (lifecycle managed below)

#include "cuda/sdf_wrapper.h"
#include "io/Exporter.h"

namespace py = pybind11;

// Helper to convert numpy array to vector of float3
std::vector<float3> array_to_float3(py::array_t<float> arr) {
  std::vector<float3> result;
  auto r = arr.unchecked<2>();
  result.reserve(r.shape(0));
  for (ssize_t i = 0; i < r.shape(0); i++) {
    result.push_back(make_float3(r(i, 0), r(i, 1), r(i, 2)));
  }
  return result;
}

// Helper to convert numpy array to vector of float4
std::vector<float4> array_to_float4(py::array_t<float> arr) {
  std::vector<float4> result;
  auto r = arr.unchecked<2>();
  result.reserve(r.shape(0));
  for (ssize_t i = 0; i < r.shape(0); i++) {
    result.push_back(make_float4(r(i, 0), r(i, 1), r(i, 2), r(i, 3)));
  }
  return result;
}

// Helper for radii (1D)
std::vector<float> array_to_float(py::array_t<float> arr) {
  std::vector<float> result;
  auto r = arr.unchecked<1>();
  result.reserve(r.shape(0));
  for (ssize_t i = 0; i < r.shape(0); i++) {
    result.push_back(r(i));
  }
  return result;
}

PYBIND11_MODULE(demgpu, m) {
  m.doc() = "DEM-GPU: XPBD Engine for Granular Dynamics on GPU";

  // The ArborX broad-phase runs on Kokkos. Initialize at import and finalize via Python atexit so
  // Kokkos shuts down while the CUDA driver is still up (avoids cudaErrorCudartUnloading at exit).
  // The Simulation holds no Kokkos Views (only per-call broad-phase temporaries), so this is safe.
  if (!Kokkos::is_initialized()) Kokkos::initialize();
  py::module_::import("atexit").attr("register")(py::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }));

  m.def(
      "export_lammps",
      [](std::string filename, int step, py::array_t<float> pos_np,
         py::array_t<float> vel_np, py::array_t<float> quats_np,
         py::array_t<float> radii_np,
         std::optional<std::tuple<float, float, float>> box_min,
         std::optional<std::tuple<float, float, float>> box_max,
         bool pbc_enabled) {
        auto pos = array_to_float3(pos_np);
        auto vel = array_to_float3(vel_np);
        auto quats = array_to_float4(quats_np);
        auto radii = array_to_float(radii_np);

        float3 bmin_storage, bmax_storage;
        float3 *bmin_ptr = nullptr;
        float3 *bmax_ptr = nullptr;

        if (box_min.has_value()) {
          bmin_storage =
              make_float3(std::get<0>(*box_min), std::get<1>(*box_min),
                          std::get<2>(*box_min));
          bmin_ptr = &bmin_storage;
        }
        if (box_max.has_value()) {
          bmax_storage =
              make_float3(std::get<0>(*box_max), std::get<1>(*box_max),
                          std::get<2>(*box_max));
          bmax_ptr = &bmax_storage;
        }

        export_lammps_dump(filename, step, pos, vel, quats, radii, bmin_ptr,
                           bmax_ptr, pbc_enabled);
      },
      py::arg("filename"), py::arg("step"), py::arg("pos"), py::arg("vel"),
      py::arg("quats"), py::arg("radii"), py::arg("box_min") = py::none(),
      py::arg("box_max") = py::none(), py::arg("pbc_enabled") = false);

  // SDF Wrappers
  m.def(
      "sdf_hollow_cylinder",
      [](std::tuple<float, float, float> p,
         std::tuple<float, float, float, float> params) {
        float3 p_vec =
            make_float3(std::get<0>(p), std::get<1>(p), std::get<2>(p));
        float4 p_params = make_float4(std::get<0>(params), std::get<1>(params),
                                      std::get<2>(params), std::get<3>(params));
        return evaluate_sdf_hollow_cylinder(p_vec, p_params);
      },
      "Evaluate Hollow Cylinder SDF", py::arg("p"), py::arg("params"));

  m.def(
      "sdf_sphere",
      [](std::tuple<float, float, float> p,
         std::tuple<float, float, float, float> params) {
        float3 p_vec =
            make_float3(std::get<0>(p), std::get<1>(p), std::get<2>(p));
        float4 p_params = make_float4(std::get<0>(params), std::get<1>(params),
                                      std::get<2>(params), std::get<3>(params));
        return evaluate_sdf_sphere(p_vec, p_params);
      },
      "Evaluate Sphere SDF", py::arg("p"), py::arg("params"));

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<int>(), py::arg("num_particles") = 1000)
      .def_static("set_cuda_device", &Simulation::set_cuda_device, py::arg("device"),
                  "Bind this process to a CUDA device (map MPI local rank -> GPU before init)")
      .def_static("cuda_device_count", &Simulation::cuda_device_count,
                  "Number of visible CUDA devices")
      .def("initialize", &Simulation::initialize, py::arg("shape_type"),
           py::arg("radius") = 0.5f, py::arg("height") = 2.0f,
           py::arg("thickness") = 0.2f)
      .def("set_positions", &Simulation::set_positions_numpy)
      .def("set_inv_mass", &Simulation::set_inv_mass_numpy)
      .def("get_inv_mass", &Simulation::get_inv_mass_numpy)
      .def("set_velocities", &Simulation::set_velocities_numpy)
      .def("set_angular_velocities", &Simulation::set_angular_velocities_numpy)
      .def("get_velocities", &Simulation::get_velocities_numpy)
      .def("get_angular_velocities", &Simulation::get_angular_velocities_numpy)
      .def("get_inv_inertia", &Simulation::get_inv_inertia_numpy)
      .def("get_masses", &Simulation::get_masses_numpy)
      .def("set_quaternions", &Simulation::set_quaternions_numpy)
      .def("set_scales", &Simulation::set_scales_numpy)
      .def("set_material_params", &Simulation::set_material_params,
           py::arg("restitution_normal"), py::arg("restitution_tangent"),
           py::arg("friction_dynamic"))
      .def("set_gravity", &Simulation::set_gravity)
      .def("set_material_params", &Simulation::set_material_params)
      .def("set_solver_iterations", &Simulation::set_solver_iterations)
      .def("set_global_scale", &Simulation::set_global_scale)
      .def(
          "set_domain",
          [](Simulation &s, std::tuple<float, float, float> min,
             std::tuple<float, float, float> max) {
            s.set_domain(make_float3(std::get<0>(min), std::get<1>(min),
                                     std::get<2>(min)),
                         make_float3(std::get<0>(max), std::get<1>(max),
                                     std::get<2>(max)));
          },
          "Set simulation domain min and max", py::arg("min"), py::arg("max"))
      .def("get_domain_min", &Simulation::get_domain_min)
      .def("get_domain_max", &Simulation::get_domain_max)
      .def("step", &Simulation::step)
      .def("set_growth_params", &Simulation::set_growth_params, py::arg("rate"),
           py::arg("new_factor") = -1.0f)
      .def("set_thermostat", &Simulation::set_thermostat,
           py::arg("temperature"), py::arg("tau"), py::arg("kB") = 1.0f,
           "Enable Berendsen thermostat. Set tau=0 to disable.")
      .def("get_growth_rate", &Simulation::get_growth_rate)
      .def("get_growth_factor", &Simulation::get_growth_factor)
      .def("num_particles", &Simulation::num_particles,
           py::arg("include_ghosts") = false)
      .def("get_positions", &Simulation::get_positions_numpy,
           py::arg("include_ghosts") = false)
      .def("get_quaternions", &Simulation::get_quaternions_numpy)
      .def("get_scales", &Simulation::get_scales_numpy)
      .def("write_vtp", &Simulation::write_vtp, py::arg("filename"))
      .def("export_sdf", &Simulation::export_sdf, py::arg("filename"),
           py::arg("resolution"),
           "Export SDF field to VTI. Resolution is (rx, ry, rz)")
      .def("get_sdf_grid", &Simulation::get_sdf_grid, py::arg("resolution"),
           "Get SDF grid as numpy array (rx, ry, rz)")
      .def("export_lammps", &Simulation::export_lammps, py::arg("filename"),
           py::arg("step"), "Export LAMMPS Dump with fixed bounds")
      .def("get_num_contacts", &Simulation::get_num_contacts,
           "Get number of contacts")
      .def("get_num_manifolds", &Simulation::get_num_manifolds,
           "Get number of manifolds")
      .def("get_max_overlap", &Simulation::get_max_overlap)
      .def("compute_overlaps", &Simulation::compute_overlaps,
           "Compute overlap of current state")
      .def("add_plane",
           [](Simulation &s, std::tuple<float, float, float> point,
              std::tuple<float, float, float> normal) {
             s.add_plane(make_float3(std::get<0>(point), std::get<1>(point),
                                     std::get<2>(point)),
                         make_float3(std::get<0>(normal), std::get<1>(normal),
                                     std::get<2>(normal)));
           })
      .def("enable_periodicity", &Simulation::enable_periodicity, py::arg("x"),
           py::arg("y"), py::arg("z"), "Enable periodic boundaries on axes")
#ifdef DEMGPU_HAVE_TPX
      .def("mpi_init", &Simulation::mpi_init, py::arg("origin"), py::arg("size"),
           py::arg("gsize"), py::arg("periodic"),
           "Set up the transport-core block decomposition over the global domain (MPI-aware step)")
      .def("mpi_build_halo", &Simulation::mpi_build_halo, py::arg("rcut"),
           "Build the owner<->ghost correspondence over current owned positions; returns ghost count")
      .def("enable_mpi_step", &Simulation::enable_mpi_step, py::arg("rcut"),
           py::arg("sync_every") = 1, py::arg("forward_rotation") = true,
           "Enable the MPI-aware step: gather ghosts (real mass) + owner->ghost forward during the "
           "solve. sync_every=1 is EXACT (refresh every iteration); M>1 refreshes every M iterations "
           "(+last). forward_rotation=False skips ghost quaternion forwards (valid for spheres)")
#endif
      .def("get_profiling_info", &Simulation::get_profiling_info);
}
