/// @file
/// @brief nanobind module `dem` — the Kokkos + ArborX XPBD granular-dynamics simulation.
///
/// Exposes peclet::dem::Simulation (the portable Kokkos+ArborX pipeline) with the essential
/// sphere-packing API. Kokkos is initialized at import and left initialized for the interpreter's
/// lifetime. Particle arrays cross the boundary through the shared peclet::core::python bridge
/// (transport-core): inputs read as flat C-order buffers, getters move the result into the NumPy
/// array's backing store (no extra copy), and under a GPU backend the bridge's device path lets
/// CuPy arrays flow in/out zero-copy.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <optional>
#include <tuple>
#include <vector>

#ifdef PECLET_DEM_MPI
#include <mpi.h>
#endif

#include "peclet/core/python/ndarray_interop.hpp"
#include "sim.hpp"

namespace nb = nanobind;
using peclet::dem::Simulation;

// A contiguous numpy array -> flat C-order host vector (the bridge handles host copy / device
// wrap).
static std::vector<float> to_vec(nb::ndarray<float, nb::c_contig> a) {
  return peclet::core::python::ndarray_to_vector<float>(nb::ndarray<>(a));
}

// A flat C-order vector -> (N,cols) numpy array, moved into the array's backing store (no extra
// copy).
static nb::ndarray<nb::numpy, float> rows(std::vector<float>&& v, int cols) {
  const std::size_t n = v.size() / static_cast<std::size_t>(cols);
  return peclet::core::python::vector_to_ndarray(std::move(v), {n, (std::size_t)cols},
                                                 {(std::int64_t)cols, 1});
}
static nb::ndarray<nb::numpy, float> flat(std::vector<float>&& v) {
  const std::size_t n = v.size();
  return peclet::core::python::vector_to_ndarray(std::move(v), {n}, {1});
}

NB_MODULE(_dem, m) {
  m.attr("__doc__") = "DEM-GPU (Kokkos + ArborX): portable XPBD granular dynamics";

  if (!Kokkos::is_initialized())
    Kokkos::initialize();
  // Teardown order matters on CUDA: releaseAll() drops every live Simulation's Views FIRST (so none
  // outlive finalize -> no "deallocated after Kokkos::finalize"), THEN Kokkos::finalize() runs from
  // a Python atexit hook while the CUDA driver is still up (so no cudaErrorCudartUnloading). Doing
  // only one of the two aborts on CUDA. Returned arrays are backed by host std::vectors (no device
  // Views).
  auto shutdown = []() {
    Simulation::releaseAll();
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
  };
  m.def("finalize", shutdown,
        "Release all live Simulations and finalize Kokkos (deterministic teardown; also run at "
        "exit).");
  nb::module_::import_("atexit").attr("register")(nb::cpp_function(shutdown));
  m.attr("execution_space") = nb::str(Kokkos::DefaultExecutionSpace::name());

  nb::class_<Simulation>(m, "Simulation")
      .def(nb::init<int>(), nb::arg("capacity"))
      .def("set_sphere_shape", &Simulation::setSphereShape, nb::arg("radius"),
           "Use a uniform sphere of the given radius for all particles.")
      .def("initialize_shape", &Simulation::initializeShape, nb::arg("shape_type"),
           nb::arg("radius"), nb::arg("height") = 0.0f, nb::arg("thickness") = 0.0f,
           "Select the particle shape (sphere/cylinder/ring/...) and its dimensions.")
      // CUDA-API alias: initialize(shape_type, radius, height, thickness).
      .def("initialize", &Simulation::initializeShape, nb::arg("shape_type"),
           nb::arg("radius") = 0.5f, nb::arg("height") = 2.0f, nb::arg("thickness") = 0.0f,
           "CUDA-API alias for initialize_shape.")
      // Import a general particle as a grid SDF + surface point shell (all particles share it).
      // grid: flat [nx*ny*nz] signed-distance samples, x-fastest (idx = x + y*nx + z*nx*ny), at
      // nodes origin + (x,y,z)*spacing (negative inside). shell: (M,3) surface points. inv_inertia:
      // unit-mass principal-frame diagonal inverse inertia. bounding_radius: canonical enclosing
      // radius. See peclet.dem.particle_builder for the SDF -> (grid, shell, inertia) helper.
      .def(
          "set_sdf_shape",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> grid, int nx, int ny, int nz,
             std::tuple<float, float, float> origin, std::tuple<float, float, float> spacing,
             nb::ndarray<float, nb::c_contig> shell, std::tuple<float, float, float> inv_inertia,
             float bounding_radius) {
            s.setSdfShape(
                to_vec(grid), nx, ny, nz,
                peclet::dem::F3{std::get<0>(origin), std::get<1>(origin), std::get<2>(origin)},
                peclet::dem::F3{std::get<0>(spacing), std::get<1>(spacing), std::get<2>(spacing)},
                to_vec(shell),
                peclet::dem::F3{std::get<0>(inv_inertia), std::get<1>(inv_inertia),
                                std::get<2>(inv_inertia)},
                bounding_radius);
          },
          nb::arg("grid"), nb::arg("nx"), nb::arg("ny"), nb::arg("nz"), nb::arg("origin"),
          nb::arg("spacing"), nb::arg("shell"), nb::arg("inv_inertia"), nb::arg("bounding_radius"),
          "Import a general particle: grid SDF (flat nx*ny*nz, x-fastest), surface point shell "
          "(M,3), unit-mass principal diagonal inverse inertia, and canonical bounding radius.")
      .def("set_domain", &Simulation::setDomain, nb::arg("lx"), nb::arg("ly"), nb::arg("lz"),
           nb::arg("px") = true, nb::arg("py") = true, nb::arg("pz") = false,
           "Set the box size (lx,ly,lz) and per-axis periodicity.")
      // CUDA-API overload: set_domain(min, max) tuples (arbitrary origin); keeps current
      // periodicity.
      .def(
          "set_domain",
          [](Simulation& s, std::tuple<float, float, float> mn,
             std::tuple<float, float, float> mx) {
            s.setDomainMinMax(peclet::dem::F3{std::get<0>(mn), std::get<1>(mn), std::get<2>(mn)},
                              peclet::dem::F3{std::get<0>(mx), std::get<1>(mx), std::get<2>(mx)});
          },
          nb::arg("min"), nb::arg("max"),
          "Set the domain by (min, max) corner tuples (arbitrary origin); keeps current "
          "periodicity.")
      .def("enable_periodicity", &Simulation::enablePeriodicity, nb::arg("x"), nb::arg("y"),
           nb::arg("z"), "Enable periodic boundaries per axis (x, y, z).")
      .def("get_domain_min", &Simulation::getDomainMin,
           "Return the domain minimum corner (x, y, z).")
      .def("get_domain_max", &Simulation::getDomainMax,
           "Return the domain maximum corner (x, y, z).")
      .def("set_gravity", &Simulation::setGravity,
           "Set the gravitational acceleration vector (gx, gy, gz).")
      .def("set_thermostat", &Simulation::setThermostat, nb::arg("temperature"), nb::arg("tau"),
           nb::arg("kB") = 1.0f,
           "Enable a Berendsen-style velocity thermostat (target temperature, coupling time tau).")
      .def("set_solver_iterations", &Simulation::setSolverIterations, nb::arg("pos"),
           nb::arg("vel"), "Set the XPBD position- and velocity-solve iteration counts.")
      .def("set_global_scale", &Simulation::setGlobalScale,
           "Set a global length scale applied to all particles.")
      .def("set_dt", &Simulation::setDt, "Set the time step dt.")
      .def("set_material_params", &Simulation::setMaterialParams, nb::arg("restitution_normal"),
           nb::arg("restitution_tangent") = 0.0f, nb::arg("friction") = 0.0f,
           "Set normal/tangential restitution and the Coulomb friction coefficient.")
      .def("add_plane", &Simulation::addPlane, "Add a boundary wall plane (px,py,pz, nx,ny,nz).")
      // CUDA-API overload: add_plane(point, normal) as 3-sequences.
      .def(
          "add_plane",
          [](Simulation& s, std::tuple<float, float, float> p, std::tuple<float, float, float> n) {
            s.addPlane(std::get<0>(p), std::get<1>(p), std::get<2>(p), std::get<0>(n),
                       std::get<1>(n), std::get<2>(n));
          },
          nb::arg("point"), nb::arg("normal"),
          "Add a boundary wall plane from a point and a normal (3-sequences).")
      // Accepts (N,3) or (N,4) like CUDA set_positions; column 3 (if present) is inv_mass (w==0
      // -> 1.0).
      .def(
          "set_positions",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            if (a.ndim() == 2 && (a.shape(1) == 3 || a.shape(1) == 4)) {
              const int n = (int)a.shape(0), k = (int)a.shape(1);
              const float* p = static_cast<const float*>(a.data());
              std::vector<float> xyz((size_t)n * 3), im;
              const bool hasMass = (k == 4);
              if (hasMass)
                im.resize(n);
              for (int i = 0; i < n; ++i) {
                xyz[3 * i] = p[k * i];
                xyz[3 * i + 1] = p[k * i + 1];
                xyz[3 * i + 2] = p[k * i + 2];
                if (hasMass) {
                  float w = p[k * i + 3];
                  im[i] = (w == 0.0f) ? 1.0f : w;
                }
              }
              s.setPositions(xyz);
              if (hasMass)
                s.setInvMass(im);
            } else {
              s.setPositions(to_vec(a));  // flat [n*3] fallback
            }
          },
          "Set particle positions from an (N,3) array, or (N,4) where column 3 is inverse mass.")
      .def(
          "set_velocities",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) { s.setVelocities(to_vec(a)); },
          "Set particle velocities from an (N,3) array.")
      .def(
          "set_quaternions",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) { s.setQuaternions(to_vec(a)); },
          "Set particle orientation quaternions from an (N,4) array.")
      .def(
          "set_angular_velocities",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setAngularVelocities(to_vec(a));
          },
          "Set particle angular velocities from an (N,3) array.")
      .def(
          "set_inv_inertia",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) { s.setInvInertia(to_vec(a)); },
          "Set per-particle inverse inertia from an (N,3) array.")
      .def(
          "set_inv_mass",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) { s.setInvMass(to_vec(a)); },
          "Set per-particle inverse mass (0 => fixed/immovable).")
      .def("get_angular_velocities",
           [](const Simulation& s) { return rows(s.getAngularVelocities(), 3); })
      .def("get_inv_inertia", [](const Simulation& s) { return rows(s.getInvInertia(), 3); })
      .def("set_scales_uniform", &Simulation::setScalesUniform,
           "Set a single uniform scale for all particles.")
      .def(
          "set_scales",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) { s.setScales(to_vec(a)); },
          "Set per-particle scales from an array.")
      .def("set_growth_params", &Simulation::setGrowthParams, nb::arg("rate"),
           nb::arg("new_factor") = -1.0f, "Set the particle growth rate and target size factor.")
      .def("get_growth_factor", &Simulation::growthFactor,
           "Return the current particle growth factor.")
      .def("get_growth_rate", &Simulation::getGrowthRate, "Return the particle growth rate.")
      .def("get_masses", [](const Simulation& s) { return flat(s.getMasses()); })
      .def(
          "get_positions", [](const Simulation& s) { return rows(s.getPositions(), 3); },
          "Return particle positions as an (N,3) numpy array.")
      .def(
          "get_velocities", [](const Simulation& s) { return rows(s.getVelocities(), 3); },
          "Return particle velocities as an (N,3) numpy array.")
      // Zero-copy device export (H2): the returned (N,3) array REFERENCES the device particle Views
      // — a NumPy view on a host backend, a DLPack/__cuda_array_interface__ array (consume with
      // cupy.from_dlpack / torch.from_dlpack) on CUDA/HIP — so a GPU-resident analysis chain never
      // pays the device->host copy. The array keeps the (ref-counted) View alive.
      .def(
          "get_positions_view",
          [](const Simulation& s) {
            return peclet::core::python::view_to_ndarray(Kokkos::subview(
                s.positionsView(), Kokkos::make_pair(0, s.numParticles()), Kokkos::ALL));
          },
          "Zero-copy (N,3) device array of positions (NumPy view on host, DLPack/CuPy on GPU).")
      .def(
          "get_velocities_view",
          [](const Simulation& s) {
            return peclet::core::python::view_to_ndarray(Kokkos::subview(
                s.velocitiesView(), Kokkos::make_pair(0, s.numParticles()), Kokkos::ALL));
          },
          "Zero-copy (N,3) device array of velocities (NumPy view on host, DLPack/CuPy on GPU).")
      .def(
          "get_quaternions", [](const Simulation& s) { return rows(s.getQuaternions(), 4); },
          "Return particle orientation quaternions as an (N,4) numpy array.")
      .def(
          "get_scales", [](const Simulation& s) { return flat(s.getScales()); },
          "Return per-particle scales as a numpy array.")
      .def("step", &Simulation::step, nb::arg("dt") = 0.0f,
           "Advance the simulation one step (dt=0 uses the configured time step).")
      .def(
          "get_sdf_grid",
          [](Simulation& s, std::tuple<int, int, int> res) {
            auto [rx, ry, rz] = res;
            // C-order (rx,ry,rz) float array, matching the prior py::array_t<float>({rx,ry,rz},
            // ...).
            return peclet::core::python::vector_to_ndarray(
                s.getSdfGrid(rx, ry, rz), {(std::size_t)rx, (std::size_t)ry, (std::size_t)rz},
                {(std::int64_t)ry * rz, (std::int64_t)rz, 1});
          },
          nb::arg("resolution"),
          "Reconstruct a packed-bed SDF on a (rx,ry,rz) grid (the get_sdf_grid pipeline for CFD).")
      .def("write_vtp", &Simulation::writeVtp, nb::arg("filename"),
           "Write particle state to a VTP file (ParaView/Ovito).")
      .def("num_particles", &Simulation::numParticles, "Return the number of particles.")
      .def("num_contacts", &Simulation::numContacts, "Return the number of broad-phase contacts.")
      .def("num_manifolds", &Simulation::numManifolds, "Return the number of contact manifolds.")
      .def("max_overlap", &Simulation::maxOverlap, "Return the maximum particle-particle overlap.")
      // CUDA-API parity: overlap measurement + LAMMPS/SDF export + profiling.
      .def("get_num_contacts", &Simulation::numContacts)    // CUDA-API alias
      .def("get_num_manifolds", &Simulation::numManifolds)  // CUDA-API alias
      .def("get_max_overlap", &Simulation::maxOverlap)      // CUDA-API alias
      .def("compute_overlaps", &Simulation::computeOverlaps, "Recompute particle overlaps.")
      .def("export_lammps", &Simulation::exportLammps, nb::arg("filename"), nb::arg("step"),
           "Export particle state to a LAMMPS dump file.")
      .def(
          "export_sdf",
          [](Simulation& s, const std::string& filename, std::tuple<int, int, int> res) {
            auto [rx, ry, rz] = res;
            s.exportSdf(filename, rx, ry, rz);
          },
          nb::arg("filename"), nb::arg("resolution"),
          "Reconstruct and write the packed-bed SDF on a (rx,ry,rz) grid to a VTI file.")
      .def(
          "get_profiling_info",
          [](Simulation& s) {
            nb::dict d;
            d["num_particles"] = s.numParticles();
            d["num_contacts"] = s.numContacts();
            d["num_manifolds"] = s.numManifolds();
            d["max_overlap"] = s.maxOverlap();
            return d;
          },
          "Return a dict of particle/contact/manifold counts and the max overlap.")
#ifdef PECLET_DEM_MPI
      // Gated MPI step (mirrors the CUDA dem MPI binding); built only with -DDEM_MPI.
      .def(
          "init_mpi",
          [](Simulation& s, std::tuple<double, double, double> origin,
             std::tuple<double, double, double> size, std::tuple<long, long, long> gsize,
             std::tuple<bool, bool, bool> periodic) {
            int inited = 0;
            MPI_Initialized(&inited);
            if (!inited) {
              int argc = 0;
              char** argv = nullptr;
              MPI_Init(&argc, &argv);
            }
            s.initMpi(origin, size, gsize, periodic, MPI_COMM_WORLD);
          },
          nb::arg("origin"), nb::arg("size"), nb::arg("gsize"), nb::arg("periodic"),
          "Set up the ORB block decomposition + transport-core particle halo for the distributed "
          "step.")
      .def("enable_mpi_step", &Simulation::enableMpiStep, nb::arg("rcut"),
           nb::arg("sync_every") = 1, nb::arg("forward_rotation") = true,
           nb::arg("rebalance_every") = 0, nb::arg("verlet_skin") = 0.0,
           "Enable the distributed step: ghost cutoff rcut, sync cadence, rotation forwarding, the "
           "load-rebalance interval in steps (0 = fixed decomposition), and the Verlet ghost-reuse "
           "skin "
           "(0 = rebuild the halo topology every substep; >0 = reuse it until a particle moves > "
           "skin).")
      .def("step_mpi", &Simulation::stepMpi, nb::arg("nsteps") = 1,
           "Advance the distributed (MPI) simulation by nsteps with halo exchange.")
      .def("rebalance", &Simulation::rebalance,
           "Re-decompose by particle count and migrate ownership now; returns this rank's new "
           "owned count.")
      .def("rank", &Simulation::rank, "Return this rank's MPI index.")
      .def("num_ghost", &Simulation::numGhost, "Return the number of ghost particles on this rank.")
      .def("mpi_rebuilds", &Simulation::mpiRebuilds,
           "Cumulative halo topology-rebuild count (Verlet-skin path); pair with mpi_gathers() for "
           "the ghost-reuse ratio.")
      .def("mpi_gathers", &Simulation::mpiGathers,
           "Cumulative ghost gather() count across distributed steps.")
#endif
      ;

  // CUDA-API parity: module-level export_lammps(filename, step, pos, vel, quats, radii, box_min,
  // box_max, pbc).
  m.def(
      "export_lammps",
      [](const std::string& filename, int step, nb::ndarray<float, nb::c_contig> pos,
         nb::ndarray<float, nb::c_contig> vel, nb::ndarray<float, nb::c_contig> quats,
         nb::ndarray<float, nb::c_contig> radii,
         std::optional<std::tuple<float, float, float>> box_min,
         std::optional<std::tuple<float, float, float>> box_max, bool pbc_enabled) {
        float bmin[3], bmax[3];
        const float *pmn = nullptr, *pmx = nullptr;
        if (box_min) {
          bmin[0] = std::get<0>(*box_min);
          bmin[1] = std::get<1>(*box_min);
          bmin[2] = std::get<2>(*box_min);
          pmn = bmin;
        }
        if (box_max) {
          bmax[0] = std::get<0>(*box_max);
          bmax[1] = std::get<1>(*box_max);
          bmax[2] = std::get<2>(*box_max);
          pmx = bmax;
        }
        peclet::dem::writeLammpsDump(filename, step, to_vec(pos), to_vec(vel), to_vec(quats),
                                     to_vec(radii), pmn, pmx, pbc_enabled);
      },
      nb::arg("filename"), nb::arg("step"), nb::arg("pos"), nb::arg("vel"), nb::arg("quats"),
      nb::arg("radii"), nb::arg("box_min") = std::nullopt, nb::arg("box_max") = std::nullopt,
      nb::arg("pbc_enabled") = false,
      "Module-level LAMMPS dump writer from raw arrays (filename, step, pos, vel, quats, radii, "
      "box, pbc).");
}
