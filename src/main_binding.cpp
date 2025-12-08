#include "simulation.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(demgpu, m) {
  m.doc() = "DEM-GPU: XPBD Engine for Granular Dynamics on GPU";

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<int>(), py::arg("num_particles") = 1000)
      .def("initialize", &Simulation::initialize, py::arg("shape_type") = 1)
      .def("set_positions", &Simulation::set_positions_numpy)
      .def("set_velocities", &Simulation::set_velocities_numpy)
      .def("get_velocities", &Simulation::get_velocities_numpy)
      .def("set_scales", &Simulation::set_scales_numpy)
      .def("set_gravity", &Simulation::set_gravity)
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
      .def("get_scales", &Simulation::get_scales_numpy);
}
