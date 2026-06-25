/// @file
/// @brief pybind11 module `dem` — the Kokkos + ArborX XPBD granular-dynamics simulation.
///
/// Exposes dem::KokkosSim (the portable Kokkos+ArborX pipeline) with the essential sphere-packing API.
/// Kokkos is initialized at import and finalized via Python atexit.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Kokkos_Core.hpp>

#include <cstring>
#include <optional>
#include <tuple>

#ifdef DEM_MPI
#include <mpi.h>
#endif

#include "sim.hpp"

namespace py = pybind11;
using dem::KokkosSim;

static std::vector<float> to_vec(py::array_t<float, py::array::c_style | py::array::forcecast> a) {
  std::vector<float> v(static_cast<size_t>(a.size()));
  std::memcpy(v.data(), a.data(), v.size() * sizeof(float));
  return v;
}

PYBIND11_MODULE(dem, m) {
  m.doc() = "DEM-GPU (Kokkos + ArborX): portable XPBD granular dynamics";

  if (!Kokkos::is_initialized()) Kokkos::initialize();
  // Kokkos lifetime: finalize via Python atexit so Kokkos shuts down while the CUDA driver is still
  // up (avoids cudaErrorCudartUnloading), but the caller must release Simulation objects (their
  // Kokkos Views) BEFORE interpreter exit — `del sim; gc.collect()` — or use finalize() explicitly,
  // otherwise a View would be freed after finalize. m.finalize() exposes it for deterministic teardown.
  // Release every live Simulation's Kokkos Views, THEN finalize (so callers need not del+gc manually).
  auto shutdown = []() {
    KokkosSim::releaseAll();
    if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
  };
  m.def("finalize", shutdown,
        "Release all live Simulations and finalize Kokkos (deterministic teardown; otherwise done at exit).");
  py::module_::import("atexit").attr("register")(py::cpp_function(shutdown));
  m.attr("execution_space") = py::str(Kokkos::DefaultExecutionSpace::name());

  py::class_<KokkosSim>(m, "Simulation")
      .def(py::init<int>(), py::arg("capacity"))
      .def("set_sphere_shape", &KokkosSim::setSphereShape, py::arg("radius"), "Use a uniform sphere of the given radius for all particles.")
      .def("initialize_shape", &KokkosSim::initializeShape, py::arg("shape_type"),
           py::arg("radius"), py::arg("height") = 0.0f, py::arg("thickness") = 0.0f,
           "Select the particle shape (sphere/cylinder/ring/...) and its dimensions.")
      // CUDA-API alias: initialize(shape_type, radius, height, thickness).
      .def("initialize", &KokkosSim::initializeShape, py::arg("shape_type"),
           py::arg("radius") = 0.5f, py::arg("height") = 2.0f, py::arg("thickness") = 0.0f,
           "CUDA-API alias for initialize_shape.")
      .def("set_domain", &KokkosSim::setDomain, py::arg("lx"), py::arg("ly"), py::arg("lz"),
           py::arg("px") = true, py::arg("py") = true, py::arg("pz") = false,
           "Set the box size (lx,ly,lz) and per-axis periodicity.")
      // CUDA-API overload: set_domain(min, max) tuples (arbitrary origin); keeps current periodicity.
      .def("set_domain",
           [](KokkosSim& s, std::tuple<float, float, float> mn, std::tuple<float, float, float> mx) {
             s.setDomainMinMax(dem::F3{std::get<0>(mn), std::get<1>(mn), std::get<2>(mn)},
                               dem::F3{std::get<0>(mx), std::get<1>(mx), std::get<2>(mx)});
           },
           py::arg("min"), py::arg("max"),
           "Set the domain by (min, max) corner tuples (arbitrary origin); keeps current periodicity.")
      .def("enable_periodicity", &KokkosSim::enablePeriodicity, py::arg("x"), py::arg("y"), py::arg("z"), "Enable periodic boundaries per axis (x, y, z).")
      .def("get_domain_min", &KokkosSim::getDomainMin, "Return the domain minimum corner (x, y, z).")
      .def("get_domain_max", &KokkosSim::getDomainMax, "Return the domain maximum corner (x, y, z).")
      .def("set_gravity", &KokkosSim::setGravity, "Set the gravitational acceleration vector (gx, gy, gz).")
      .def("set_thermostat", &KokkosSim::setThermostat,
           py::arg("temperature"), py::arg("tau"), py::arg("kB") = 1.0f,
           "Enable a Berendsen-style velocity thermostat (target temperature, coupling time tau).")
      .def("set_solver_iterations", &KokkosSim::setSolverIterations, py::arg("pos"), py::arg("vel"), "Set the XPBD position- and velocity-solve iteration counts.")
      .def("set_global_scale", &KokkosSim::setGlobalScale, "Set a global length scale applied to all particles.")
      .def("set_dt", &KokkosSim::setDt, "Set the time step dt.")
      .def("set_material_params", &KokkosSim::setMaterialParams,
           py::arg("restitution_normal"), py::arg("restitution_tangent") = 0.0f, py::arg("friction") = 0.0f,
           "Set normal/tangential restitution and the Coulomb friction coefficient.")
      .def("add_plane", &KokkosSim::addPlane, "Add a boundary wall plane (px,py,pz, nx,ny,nz).")
      // CUDA-API overload: add_plane(point, normal) as 3-sequences.
      .def("add_plane",
           [](KokkosSim& s, std::tuple<float, float, float> p, std::tuple<float, float, float> n) {
             s.addPlane(std::get<0>(p), std::get<1>(p), std::get<2>(p),
                        std::get<0>(n), std::get<1>(n), std::get<2>(n));
           },
           py::arg("point"), py::arg("normal"),
           "Add a boundary wall plane from a point and a normal (3-sequences).")
      // Accepts (N,3) or (N,4) like CUDA set_positions; column 3 (if present) is inv_mass (w==0 -> 1.0).
      .def("set_positions",
           [](KokkosSim& s, py::array_t<float, py::array::c_style | py::array::forcecast> a) {
             auto b = a.request();
             if (b.ndim == 2 && (b.shape[1] == 3 || b.shape[1] == 4)) {
               const int n = (int)b.shape[0], k = (int)b.shape[1];
               const float* p = static_cast<const float*>(b.ptr);
               std::vector<float> xyz((size_t)n * 3), im;
               const bool hasMass = (k == 4);
               if (hasMass) im.resize(n);
               for (int i = 0; i < n; ++i) {
                 xyz[3 * i] = p[k * i]; xyz[3 * i + 1] = p[k * i + 1]; xyz[3 * i + 2] = p[k * i + 2];
                 if (hasMass) { float w = p[k * i + 3]; im[i] = (w == 0.0f) ? 1.0f : w; }
               }
               s.setPositions(xyz);
               if (hasMass) s.setInvMass(im);
             } else {
               s.setPositions(to_vec(a));  // flat [n*3] fallback
             }
           },
           "Set particle positions from an (N,3) array, or (N,4) where column 3 is inverse mass.")
      .def("set_velocities", [](KokkosSim& s, py::array_t<float> a) { s.setVelocities(to_vec(a)); }, "Set particle velocities from an (N,3) array.")
      .def("set_quaternions", [](KokkosSim& s, py::array_t<float> a) { s.setQuaternions(to_vec(a)); }, "Set particle orientation quaternions from an (N,4) array.")
      .def("set_angular_velocities", [](KokkosSim& s, py::array_t<float> a) { s.setAngularVelocities(to_vec(a)); }, "Set particle angular velocities from an (N,3) array.")
      .def("set_inv_inertia", [](KokkosSim& s, py::array_t<float> a) { s.setInvInertia(to_vec(a)); }, "Set per-particle inverse inertia from an (N,3) array.")
      .def("set_inv_mass", [](KokkosSim& s, py::array_t<float> a) { s.setInvMass(to_vec(a)); }, "Set per-particle inverse mass (0 => fixed/immovable).")
      .def("get_angular_velocities", [](const KokkosSim& s) {
             auto v = s.getAngularVelocities(); const int n = (int)(v.size()/3);
             py::array_t<float> o({n,3}); std::memcpy(o.mutable_data(), v.data(), v.size()*sizeof(float)); return o; })
      .def("get_inv_inertia", [](const KokkosSim& s) {
             auto v = s.getInvInertia(); const int n = (int)(v.size()/3);
             py::array_t<float> o({n,3}); std::memcpy(o.mutable_data(), v.data(), v.size()*sizeof(float)); return o; })
      .def("set_scales_uniform", &KokkosSim::setScalesUniform, "Set a single uniform scale for all particles.")
      .def("set_scales", [](KokkosSim& s, py::array_t<float> a) { s.setScales(to_vec(a)); }, "Set per-particle scales from an array.")
      .def("set_growth_params", &KokkosSim::setGrowthParams, py::arg("rate"), py::arg("new_factor") = -1.0f, "Set the particle growth rate and target size factor.")
      .def("get_growth_factor", &KokkosSim::growthFactor, "Return the current particle growth factor.")
      .def("get_growth_rate", &KokkosSim::getGrowthRate, "Return the particle growth rate.")
      .def("get_masses", [](const KokkosSim& s) {
             auto v = s.getMasses(); py::array_t<float> o((py::ssize_t)v.size());
             std::memcpy(o.mutable_data(), v.data(), v.size() * sizeof(float)); return o; })
      .def("get_positions",
           [](const KokkosSim& s) {
             auto v = s.getPositions();
             const int n = static_cast<int>(v.size() / 3);
             py::array_t<float> out({n, 3});
             std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
             return out;
           }, "Return particle positions as an (N,3) numpy array.")
      .def("get_velocities", [](const KokkosSim& s) {
             auto v = s.getVelocities(); const int n = (int)(v.size() / 3);
             py::array_t<float> out({n, 3}); std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float)); return out; },
             "Return particle velocities as an (N,3) numpy array.")
      .def("get_quaternions", [](const KokkosSim& s) {
             auto v = s.getQuaternions(); const int n = (int)(v.size() / 4);
             py::array_t<float> out({n, 4}); std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float)); return out; },
             "Return particle orientation quaternions as an (N,4) numpy array.")
      .def("get_scales", [](const KokkosSim& s) {
             auto v = s.getScales(); py::array_t<float> out((py::ssize_t)v.size());
             std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float)); return out; },
             "Return per-particle scales as a numpy array.")
      .def("step", &KokkosSim::step, py::arg("dt") = 0.0f, "Advance the simulation one step (dt=0 uses the configured time step).")
      .def("get_sdf_grid", [](KokkosSim& s, std::tuple<int, int, int> res) {
             auto [rx, ry, rz] = res; auto g = s.getSdfGrid(rx, ry, rz);
             return py::array_t<float>({rx, ry, rz}, g.data()); }, py::arg("resolution"),
           "Reconstruct a packed-bed SDF on a (rx,ry,rz) grid (the get_sdf_grid pipeline for CFD).")
      .def("write_vtp", &KokkosSim::writeVtp, py::arg("filename"), "Write particle state to a VTP file (ParaView/Ovito).")
      .def("num_particles", &KokkosSim::numParticles, "Return the number of particles.")
      .def("num_contacts", &KokkosSim::numContacts, "Return the number of broad-phase contacts.")
      .def("num_manifolds", &KokkosSim::numManifolds, "Return the number of contact manifolds.")
      .def("max_overlap", &KokkosSim::maxOverlap, "Return the maximum particle-particle overlap.")
      // CUDA-API parity: overlap measurement + LAMMPS/SDF export + profiling.
      .def("get_num_contacts", &KokkosSim::numContacts)        // CUDA-API alias
      .def("get_num_manifolds", &KokkosSim::numManifolds)      // CUDA-API alias
      .def("get_max_overlap", &KokkosSim::maxOverlap)          // CUDA-API alias
      .def("compute_overlaps", &KokkosSim::computeOverlaps, "Recompute particle overlaps.")
      .def("export_lammps", &KokkosSim::exportLammps, py::arg("filename"), py::arg("step"), "Export particle state to a LAMMPS dump file.")
      .def("export_sdf",
           [](KokkosSim& s, const std::string& filename, std::tuple<int, int, int> res) {
             auto [rx, ry, rz] = res; s.exportSdf(filename, rx, ry, rz);
           },
           py::arg("filename"), py::arg("resolution"),
           "Reconstruct and write the packed-bed SDF on a (rx,ry,rz) grid to a VTI file.")
      .def("get_profiling_info", [](KokkosSim& s) {
             py::dict d;
             d["num_particles"] = s.numParticles();
             d["num_contacts"] = s.numContacts();
             d["num_manifolds"] = s.numManifolds();
             d["max_overlap"] = s.maxOverlap();
             return d; }, "Return a dict of particle/contact/manifold counts and the max overlap.")
#ifdef DEM_MPI
      // Gated MPI step (mirrors the CUDA dem MPI binding); built only with -DDEM_MPI.
      .def("init_mpi",
           [](KokkosSim& s, std::tuple<double, double, double> origin,
              std::tuple<double, double, double> size, std::tuple<long, long, long> gsize,
              std::tuple<bool, bool, bool> periodic) {
             int inited = 0; MPI_Initialized(&inited);
             if (!inited) { int argc = 0; char** argv = nullptr; MPI_Init(&argc, &argv); }
             s.initMpi(origin, size, gsize, periodic, MPI_COMM_WORLD);
           },
           py::arg("origin"), py::arg("size"), py::arg("gsize"), py::arg("periodic"),
           "Set up the ORB block decomposition + transport-core particle halo for the distributed step.")
      .def("enable_mpi_step", &KokkosSim::enableMpiStep,
           py::arg("rcut"), py::arg("sync_every") = 1, py::arg("forward_rotation") = true,
           py::arg("rebalance_every") = 0,
           "Enable the distributed step: ghost cutoff rcut, sync cadence, rotation forwarding, and the "
           "load-rebalance interval in steps (0 = fixed decomposition).")
      .def("step_mpi", &KokkosSim::stepMpi, py::arg("nsteps") = 1, "Advance the distributed (MPI) simulation by nsteps with halo exchange.")
      .def("rebalance", &KokkosSim::rebalance,
           "Re-decompose by particle count and migrate ownership now; returns this rank's new owned count.")
      .def("rank", &KokkosSim::rank, "Return this rank's MPI index.")
      .def("num_ghost", &KokkosSim::numGhost, "Return the number of ghost particles on this rank.")
#endif
      ;

  // CUDA-API parity: module-level export_lammps(filename, step, pos, vel, quats, radii, box_min, box_max, pbc).
  m.def("export_lammps",
        [](const std::string& filename, int step, py::array_t<float, py::array::c_style | py::array::forcecast> pos,
           py::array_t<float, py::array::c_style | py::array::forcecast> vel,
           py::array_t<float, py::array::c_style | py::array::forcecast> quats,
           py::array_t<float, py::array::c_style | py::array::forcecast> radii,
           std::optional<std::tuple<float, float, float>> box_min,
           std::optional<std::tuple<float, float, float>> box_max, bool pbc_enabled) {
          float bmin[3], bmax[3]; const float *pmn = nullptr, *pmx = nullptr;
          if (box_min) { bmin[0] = std::get<0>(*box_min); bmin[1] = std::get<1>(*box_min); bmin[2] = std::get<2>(*box_min); pmn = bmin; }
          if (box_max) { bmax[0] = std::get<0>(*box_max); bmax[1] = std::get<1>(*box_max); bmax[2] = std::get<2>(*box_max); pmx = bmax; }
          dem::writeLammpsDump(filename, step, to_vec(pos), to_vec(vel), to_vec(quats), to_vec(radii),
                               pmn, pmx, pbc_enabled);
        },
        py::arg("filename"), py::arg("step"), py::arg("pos"), py::arg("vel"), py::arg("quats"),
        py::arg("radii"), py::arg("box_min") = py::none(), py::arg("box_max") = py::none(),
        py::arg("pbc_enabled") = false,
        "Module-level LAMMPS dump writer from raw arrays (filename, step, pos, vel, quats, radii, box, pbc).");
}
