// engine_binding.cpp - pybind11 binding exposing a minimal DEMGPU engine
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace demgpu {

struct EngineConfig {
    int particleCount = 0;
    int solverIterations = 5;
};

class Engine {
public:
    explicit Engine(EngineConfig cfg) : cfg_(cfg) {}
    int run_step() {
        // Placeholder logic: return iterations * particle count
        return cfg_.solverIterations * cfg_.particleCount;
    }
private:
    EngineConfig cfg_;
};

} // namespace demgpu

PYBIND11_MODULE(demgpu, m) {
    using namespace demgpu;
    py::class_<EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("particleCount", &EngineConfig::particleCount)
        .def_readwrite("solverIterations", &EngineConfig::solverIterations);

    py::class_<Engine>(m, "Engine")
        .def(py::init<EngineConfig>())
        .def("run_step", &Engine::run_step);
}
