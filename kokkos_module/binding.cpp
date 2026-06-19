// packing-gpu — pybind11 module for the Kokkos-backed DEM simulation (demgpu_kokkos).
//
// Additive to the CUDA demgpu.so: exposes dem::KokkosSim (the portable Kokkos+ArborX pipeline) with
// the essential sphere-packing API. Kokkos is initialized at import and finalized via Python atexit.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>

#include "sim_kokkos.hpp"

namespace py = pybind11;
using dem::KokkosSim;

static std::vector<float> to_vec(py::array_t<float, py::array::c_style | py::array::forcecast> a) {
  std::vector<float> v(static_cast<size_t>(a.size()));
  std::memcpy(v.data(), a.data(), v.size() * sizeof(float));
  return v;
}

PYBIND11_MODULE(demgpu_kokkos, m) {
  m.doc() = "DEM-GPU (Kokkos + ArborX): portable XPBD granular dynamics";

  if (!Kokkos::is_initialized()) Kokkos::initialize();
  // Kokkos lifetime: finalize via Python atexit so Kokkos shuts down while the CUDA driver is still
  // up (avoids cudaErrorCudartUnloading), but the caller must release Simulation objects (their
  // Kokkos Views) BEFORE interpreter exit — `del sim; gc.collect()` — or use finalize() explicitly,
  // otherwise a View would be freed after finalize. m.finalize() exposes it for deterministic teardown.
  m.def("finalize", []() { if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize(); });
  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() {
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  }));
  m.attr("execution_space") = py::str(Kokkos::DefaultExecutionSpace::name());

  py::class_<KokkosSim>(m, "Simulation")
      .def(py::init<int>(), py::arg("capacity"))
      .def("set_sphere_shape", &KokkosSim::setSphereShape, py::arg("radius"))
      .def("set_domain", &KokkosSim::setDomain, py::arg("lx"), py::arg("ly"), py::arg("lz"),
           py::arg("px") = true, py::arg("py") = true, py::arg("pz") = false)
      .def("set_gravity", &KokkosSim::setGravity)
      .def("set_solver_iterations", &KokkosSim::setSolverIterations, py::arg("pos"), py::arg("vel"))
      .def("set_global_scale", &KokkosSim::setGlobalScale)
      .def("set_dt", &KokkosSim::setDt)
      .def("set_material_params", &KokkosSim::setMaterialParams,
           py::arg("restitution_normal"), py::arg("restitution_tangent") = 0.0f, py::arg("friction") = 0.0f)
      .def("add_plane", &KokkosSim::addPlane)
      .def("set_positions", [](KokkosSim& s, py::array_t<float> a) { s.setPositions(to_vec(a)); })
      .def("set_velocities", [](KokkosSim& s, py::array_t<float> a) { s.setVelocities(to_vec(a)); })
      .def("set_scales_uniform", &KokkosSim::setScalesUniform)
      .def("set_scales", [](KokkosSim& s, py::array_t<float> a) { s.setScales(to_vec(a)); })
      .def("set_growth_params", &KokkosSim::setGrowthParams, py::arg("rate"), py::arg("new_factor") = -1.0f)
      .def("get_growth_factor", &KokkosSim::growthFactor)
      .def("get_positions",
           [](const KokkosSim& s) {
             auto v = s.getPositions();
             const int n = static_cast<int>(v.size() / 3);
             py::array_t<float> out({n, 3});
             std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
             return out;
           })
      .def("get_velocities", [](const KokkosSim& s) {
             auto v = s.getVelocities(); const int n = (int)(v.size() / 3);
             py::array_t<float> out({n, 3}); std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float)); return out; })
      .def("get_quaternions", [](const KokkosSim& s) {
             auto v = s.getQuaternions(); const int n = (int)(v.size() / 4);
             py::array_t<float> out({n, 4}); std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float)); return out; })
      .def("get_scales", [](const KokkosSim& s) {
             auto v = s.getScales(); py::array_t<float> out((py::ssize_t)v.size());
             std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float)); return out; })
      .def("step", &KokkosSim::step, py::arg("nsteps") = 1)
      .def("num_particles", &KokkosSim::numParticles)
      .def("num_contacts", &KokkosSim::numContacts)
      .def("max_overlap", &KokkosSim::maxOverlap);
}
