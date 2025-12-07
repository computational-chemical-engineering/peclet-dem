#include "simulation.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(demgpu, m) {
  m.doc() = "DEM-GPU: XPBD Engine for Granular Dynamics on GPU";

  py::class_<Simulation>(m, "Simulation")
      .def(py::init<int>(), py::arg("num_particles") = 1000)
      .def("initialize", &Simulation::initialize, py::arg("shape_type") = 1)
      .def("step", &Simulation::step)
      .def("get_positions",
           [](Simulation &s) {
             // Return numpy array copy
             int n = s.num_particles();
             auto result = py::array_t<float>(n * 4);
             py::buffer_info buf = result.request();
             s.get_positions_numpy((unsigned long)buf.ptr, n * 4);
             return result;
           })
      .def("get_quaternions",
           [](Simulation &s) {
             int n = s.num_particles();
             auto result = py::array_t<float>(n * 4);
             py::buffer_info buf = result.request();
             s.get_quaternions_numpy((unsigned long)buf.ptr, n * 4);
             return result;
           })
      .def("get_scales",
           [](Simulation &s) {
             int n = s.num_particles();
             auto result = py::array_t<float>(n);
             py::buffer_info buf = result.request();
             s.get_scales_numpy((unsigned long)buf.ptr, n);
             return result;
           })
      .def("set_scales", [](Simulation &s, py::array_t<float> scales) {
        int n = s.num_particles();
        py::buffer_info buf = scales.request();
        s.set_scales_numpy((unsigned long)buf.ptr, buf.size);
      });
}
