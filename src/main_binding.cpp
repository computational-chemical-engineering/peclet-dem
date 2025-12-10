#include "simulation.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(demgpu, m) {
  m.doc() = "DEM-GPU: XPBD Engine for Granular Dynamics on GPU";

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<int>(), py::arg("num_particles") = 1000)
      .def("initialize", &Simulation::initialize, py::arg("shape_type"),
           py::arg("radius") = 0.5f, py::arg("height") = 2.0f,
           py::arg("thickness") = 0.2f)
      .def("set_positions", &Simulation::set_positions_numpy)
      .def("set_velocities", &Simulation::set_velocities_numpy)
      .def("get_velocities", &Simulation::get_velocities_numpy)
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
      .def("get_positions", &Simulation::get_positions_numpy)
      .def("get_quaternions", &Simulation::get_quaternions_numpy)
      .def("get_scales", &Simulation::get_scales_numpy)
      .def("write_vtp", &Simulation::write_vtp, py::arg("filename"))
      .def("export_sdf", &Simulation::export_sdf, py::arg("filename"),
           py::arg("resolution"),
           "Export SDF field to VTI. Resolution is (rx, ry, rz)")
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
      .def("get_profiling_info", &Simulation::get_profiling_info);
}
