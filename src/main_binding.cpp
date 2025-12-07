#include <pybind11/pybind11.h>
#include "simulation.h"

namespace py = pybind11;

PYBIND11_MODULE(demgpu, m) {
    m.doc() = "DEM-GPU: XPBD Engine for Granular Dynamics on GPU";

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<int>(), py::arg("num_particles") = 1000)
        .def("initialize", &Simulation::initialize)
        .def("step", &Simulation::step, py::arg("dt"))
        .def("get_positions", &Simulation::get_positions_numpy, "Get particle positions as numpy array")
        ;
}
