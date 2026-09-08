/// @file
/// @brief nanobind module `dem` — the Kokkos + ArborX XPBD granular-dynamics simulation.
///
/// Exposes peclet::dem::Simulation (the portable Kokkos+ArborX pipeline) in two tiers
/// (suite/docs/QUALITY_PLAN.md D2): the PUBLIC surface on `Simulation` is what a user needs to set
/// up, run and read out a simulation; everything a developer uses to inspect, ablate or profile
/// lives on `Simulation.diagnostics` (a `Diagnostics` view holding a reference to the simulation,
/// no copies of state). Names follow suite/docs/NAMING.md: `set_<thing>` setters, stored scalars
/// as properties, computed scalars as bare methods, array copies as `get_*`, `set_dt` + `step(n)`,
/// string modes, 3-tuples for every triple, one 3-D `(nx, ny, nz)` Fortran-order array for every
/// grid.
///
/// Particle arrays cross the boundary through the shared peclet::core::python bridge (core):
/// inputs read as flat C-order buffers, getters move the result into the NumPy array's backing
/// store (no extra copy), and under a GPU backend the bridge's device path lets CuPy arrays flow
/// in/out zero-copy.
///
/// Kokkos teardown follows the suite-wide pattern of peclet/core/python/kokkos_teardown.hpp: Kokkos
/// is initialized at import; Simulation keeps its own live registry (sim.hpp,
/// Simulation::releaseAll drops every live Simulation's Views) which is plugged in as a release
/// hook, and every zero-copy array (`get_*_view`) is a Releasable capsule; the module's single
/// atexit hook (also `dem.finalize()`) releases all of them and THEN calls Kokkos::finalize, so a
/// Simulation or array still referenced at interpreter exit (script globals, a Jupyter/Quarto
/// kernel) can no longer be destroyed after finalize -- which is a Kokkos::abort (SIGABRT / exit
/// 134, on OpenMP as on CUDA).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef PECLET_DEM_MPI
#include <mpi.h>
#endif

#include "peclet/core/python/kokkos_teardown.hpp"
#include "peclet/core/python/ndarray_interop.hpp"
#include "sim.hpp"

namespace nb = nanobind;
using peclet::dem::Simulation;

namespace {

/// A triple: any sequence of three floats (tuple, list, (3,) array).
using F3Arg = std::array<float, 3>;
peclet::dem::F3 f3(const F3Arg& a) {
  return peclet::dem::F3{a[0], a[1], a[2]};
}

/// A grid SDF: ONE 3-D `(nx, ny, nz)` array indexed `[x, y, z]` (Fortran order = x-fastest, the
/// suite's grid layout, CONVENTIONS.md §6); nanobind converts a C-order input. Never a flat
/// buffer plus `nx, ny, nz`.
using Grid3 = nb::ndarray<float, nb::ndim<3>, nb::f_contig>;

// A contiguous numpy array -> flat C-order host vector (the bridge handles host copy / device
// wrap).
std::vector<float> to_vec(nb::ndarray<float, nb::c_contig> a) {
  return peclet::core::python::ndarray_to_vector<float>(nb::ndarray<>(a));
}
std::vector<float> grid_to_vec(Grid3 g) {
  return peclet::core::python::ndarray_to_vector<float>(nb::ndarray<>(g));
}

/// Per-particle `(N, cols)` input: the shape is checked against the particle set `set_positions`
/// sized, so a wrong shape or a call before `set_positions` raises instead of silently writing a
/// prefix (the sim.hpp setters clamp to the shorter of the two).
std::vector<float> rows_of(nb::ndarray<float, nb::c_contig> a, int cols, const Simulation& s,
                           const char* who) {
  if (a.ndim() != 2 || static_cast<int>(a.shape(1)) != cols)
    throw std::invalid_argument(std::string(who) + ": expected an (N, " + std::to_string(cols) +
                                ") array");
  if (static_cast<int>(a.shape(0)) != s.numParticles())
    throw std::invalid_argument(std::string(who) + ": " + std::to_string(a.shape(0)) +
                                " rows but num_particles = " + std::to_string(s.numParticles()) +
                                " (set_positions sizes the particle set; call it first)");
  return to_vec(a);
}
std::vector<float> flat_of(nb::ndarray<float, nb::c_contig> a, const Simulation& s,
                           const char* who) {
  if (a.ndim() != 1)
    throw std::invalid_argument(std::string(who) + ": expected an (N,) array");
  if (static_cast<int>(a.shape(0)) != s.numParticles())
    throw std::invalid_argument(std::string(who) + ": " + std::to_string(a.shape(0)) +
                                " entries but num_particles = " + std::to_string(s.numParticles()) +
                                " (set_positions sizes the particle set; call it first)");
  return to_vec(a);
}

// A flat C-order vector -> (N,cols) numpy array, moved into the array's backing store (no extra
// copy).
nb::ndarray<nb::numpy, float> rows(std::vector<float>&& v, int cols) {
  const std::size_t n = v.size() / static_cast<std::size_t>(cols);
  return peclet::core::python::vector_to_ndarray(std::move(v), {n, (std::size_t)cols},
                                                 {(std::int64_t)cols, 1});
}
nb::ndarray<nb::numpy, float> flat(std::vector<float>&& v) {
  const std::size_t n = v.size();
  return peclet::core::python::vector_to_ndarray(std::move(v), {n}, {1});
}

/// Analytic shape names -> peclet::dem::ShapeKind.
int shape_kind(const std::string& name, const char* who) {
  if (name == "sphere")
    return peclet::dem::SPHERE;
  if (name == "hollow_cylinder")
    return peclet::dem::HOLLOW_CYLINDER;
  if (name == "box")
    return peclet::dem::BOX;
  throw std::invalid_argument(
      std::string(who) + ": expected 'sphere', 'hollow_cylinder' or 'box', got '" + name + "'");
}

/// The developer tier: a view onto one Simulation (no state of its own). Bound as
/// `Simulation.diagnostics`; the property keeps the Simulation alive.
struct Diagnostics {
  Simulation* s;
};

}  // namespace

NB_MODULE(_dem, m) {
  m.attr("__doc__") = "peclet.dem (Kokkos + ArborX): performance-portable XPBD granular dynamics";

  // Kokkos init + the release-then-finalize atexit hook + finalize() + execution_space: the
  // suite-wide pattern (file comment; peclet/core/python/kokkos_teardown.hpp). Simulation's own
  // registry (sim.hpp) is released through a hook; the get_*_view capsules register themselves.
  peclet::core::python::add_release_hook(&Simulation::releaseAll);
  peclet::core::python::install(m);

  nb::class_<Diagnostics>(
      m, "Diagnostics",
      "Developer tier of a Simulation, reached as `sim.diagnostics`: instruments, ablations and "
      "execution-policy switches. Nothing here is needed to set up, run or read out a "
      "simulation; the setters that CHANGE RESULTS say so in their docstring.")
      .def(
          "set_stabilization",
          [](Diagnostics& d, const std::string& mode) { d.s->setStabilizationMode(mode); },
          nb::arg("mode"),
          "The full stabilization mode set: the three production modes of "
          "Simulation.set_stabilization plus the two measurement modes -- 'escalate' (extra "
          "symmetric sweeps up to 256; diagnostic/fallback) and 'ordered' (level-ordered symmetric "
          "sweeps; known insufficient for deep columns).")
      .def(
          "set_velocity_solver",
          [](Diagnostics& d, const std::string& name) {
            if (name == "gauss_seidel")
              d.s->setVelocityUseGS(true);
            else if (name == "jacobi")
              d.s->setVelocityUseGS(false);
            else
              throw std::invalid_argument(
                  "set_velocity_solver: expected 'gauss_seidel' or 'jacobi', got '" + name + "'");
          },
          nb::arg("name"),
          "A/B switch of the single-GPU collision solves: 'gauss_seidel' (default; colored "
          "Gauss-Seidel, correct multi-contact dissipation) or 'jacobi' (count-averaged Jacobi, "
          "the exact-redundant legacy scheme). CHANGES RESULTS.")
      .def_prop_ro(
          "velocity_solver",
          [](Diagnostics& d) {
            return std::string(d.s->velocityUseGS() ? "gauss_seidel" : "jacobi");
          },
          "'gauss_seidel' or 'jacobi'.")
      .def(
          "set_cuda_graphs", [](Diagnostics& d, bool on) { d.s->setCudaGraphs(on); },
          nb::arg("enabled"),
          "CUDA-graph replay of the solver's iteration loops (default True): capture collapses "
          "each iteration's launch storm into one replay. Results are bit-identical either way; "
          "inert on non-CUDA backends and on the distributed step.")
      .def_prop_ro(
          "cuda_graphs", [](Diagnostics& d) { return d.s->cudaGraphs(); },
          "Whether CUDA-graph replay of the solver loops is enabled.")
      .def(
          "set_fused_sweeps",
          [](Diagnostics& d, const std::string& mode) { d.s->setFusedSweeps(mode); },
          nb::arg("mode"),
          "Fused color sweeps (CUDA): a whole sweep -- and where eligible the whole adaptive "
          "iteration loop -- as ONE kernel behind software grid barriers. 'auto' (default) uses "
          "them exactly where graph replay is unavailable (the distributed step, or "
          "set_cuda_graphs(False)); 'on'/'off' force. Bit-identical results either way.")
      .def_prop_ro(
          "fused_sweeps", [](Diagnostics& d) { return d.s->fusedSweeps(); },
          "Fused-sweep policy: 'auto', 'on' or 'off'.")
      .def(
          "coloring_conflicts", [](Diagnostics& d) { return d.s->debugColoringConflicts(); },
          "(velocity, position) graph-coloring-invariant violations in the last substep; a valid "
          "coloring returns (0, 0). Test hook for the incremental warm-start path.")
      .def(
          "rest_orphan_stats", [](Diagnostics& d) { return d.s->restOrphanStats(); },
          "Poisson-restitution instrument: (sum, max, n_bodies>0) of the per-body orphaned event "
          "budget (physical impulse units).")
      .def(
          "rest_bank_stats", [](Diagnostics& d) { return d.s->restBankStats(); },
          "Poisson-restitution instrument: (sum, max, n_pairs>0) of the per-pair owed separation "
          "impulse committed last substep (physical impulse units).")
      .def(
          "wall_sdf_at",
          [](Diagnostics& d, int wall_index, nb::ndarray<float, nb::c_contig> pts) {
            if (pts.ndim() != 2 || pts.shape(1) != 3)
              throw std::invalid_argument("wall_sdf_at: points must be an (M, 3) array");
            return flat(d.s->wallSdfAt(wall_index, to_vec(pts)));
          },
          nb::arg("wall_index"), nb::arg("points"),
          "The wall's signed distance at world points: (M,3) in -> (M,) out, POSITIVE in the void "
          "where the grains live -- exactly what the narrow phase reads, so it is the honest check "
          "of a set_wall_transform placement and the way to draw a stirrer.")
      .def(
          "profiling_info",
          [](Diagnostics& d) {
            nb::dict r;
            r["num_particles"] = d.s->numParticles();
            r["num_contacts"] = d.s->numContacts();
            r["num_manifolds"] = d.s->numManifolds();
            r["max_overlap"] = d.s->maxOverlap();
            return r;
          },
          "Dict of the particle / broad-phase pair / manifold counts and the last-step max "
          "overlap (the same four values the Simulation properties carry).")
#ifdef PECLET_DEM_MPI
      .def_prop_ro(
          "mpi_rebuilds", [](Diagnostics& d) { return d.s->mpiRebuilds(); },
          "Cumulative halo topology-rebuild count (Verlet-skin path); with mpi_gathers this is "
          "the ghost-reuse ratio.")
      .def_prop_ro(
          "mpi_gathers", [](Diagnostics& d) { return d.s->mpiGathers(); },
          "Cumulative ghost gather() count across distributed steps.")
#endif
      ;

  nb::class_<Simulation>(
      m, "Simulation",
      "One granular-dynamics simulation (XPBD impulse engine + explicit Hertz-Mindlin engine, "
      "the same particle set). Set up shapes -> domain -> positions -> materials, `set_dt`, then "
      "`step(n)`; read out with the `get_*` arrays. Developer instruments live on "
      "`.diagnostics`.")
      .def(nb::init<int>(), nb::arg("capacity"),
           "Allocate for up to `capacity` particles (owned + periodic ghost slots); set_positions "
           "refuses more.")
      .def_prop_ro("capacity", &Simulation::capacity, "The particle capacity this was built with.")
      .def_prop_ro(
          "diagnostics", [](Simulation& s) { return Diagnostics{&s}; }, nb::keep_alive<0, 1>(),
          "The developer tier (Diagnostics) of this simulation.")
      // ---- shapes -------------------------------------------------------------------------
      .def(
          "initialize_shape",
          [](Simulation& s, const std::string& shape, float radius, float height, float thickness) {
            s.initializeShape(shape_kind(shape, "initialize_shape"), radius, height, thickness);
          },
          nb::arg("shape"), nb::arg("radius"), nb::arg("height") = 0.0f,
          nb::arg("thickness") = 0.0f,
          "Select the single particle shape and its dimensions: 'sphere' (radius), "
          "'hollow_cylinder' (radius, height, thickness) or 'box' (half-extent = radius). RESETS "
          "the shape registry to this one shape (use add_shape for a mixture) and records the "
          "unit-mass inverse inertia that set_positions applies to every particle -- so call it "
          "BEFORE set_positions.")
      .def(
          "set_sdf_shape",
          [](Simulation& s, Grid3 grid, F3Arg origin, F3Arg spacing,
             nb::ndarray<float, nb::c_contig> shell, F3Arg inv_inertia, float bounding_radius) {
            const int nx = (int)grid.shape(0), ny = (int)grid.shape(1), nz = (int)grid.shape(2);
            s.setSdfShape(grid_to_vec(grid), nx, ny, nz, f3(origin), f3(spacing), to_vec(shell),
                          f3(inv_inertia), bounding_radius);
          },
          nb::arg("grid"), nb::arg("origin"), nb::arg("spacing"), nb::arg("shell"),
          nb::arg("inv_inertia"), nb::arg("bounding_radius"),
          "Import a general particle: `grid` is the body-frame signed distance on an (nx, ny, nz) "
          "array indexed [x, y, z] (negative inside), sampled at nodes origin + (x, y, z) * "
          "spacing; `shell` the (M,3) surface point shell (EMPTY = sample the field's own zero "
          "level set); `inv_inertia` the unit-mass principal diagonal inverse inertia; "
          "`bounding_radius` the canonical enclosing radius. RESETS the registry to this one "
          "shape. See peclet.dem.build_particle for the SDF -> (grid, shell, inertia) helper.")
      .def(
          "add_shape",
          [](Simulation& s, const std::string& shape, float radius, float height, float thickness) {
            return s.addShape(shape_kind(shape, "add_shape"), radius, height, thickness);
          },
          nb::arg("shape"), nb::arg("radius"), nb::arg("height") = 0.0f,
          nb::arg("thickness") = 0.0f,
          "Append an analytic shape ('sphere' | 'hollow_cylinder' | 'box', dimensions as "
          "initialize_shape) to the registry and return its index, for a simulation with a "
          "MIXTURE of shapes. initialize_shape stays the single-shape entry point (it RESETS the "
          "registry). Assign the returned index with set_shape_ids.")
      .def(
          "add_sdf_shape",
          [](Simulation& s, Grid3 grid, F3Arg origin, F3Arg spacing,
             nb::ndarray<float, nb::c_contig> shell, F3Arg inv_inertia, float bounding_radius) {
            const int nx = (int)grid.shape(0), ny = (int)grid.shape(1), nz = (int)grid.shape(2);
            return s.addSdfShape(grid_to_vec(grid), nx, ny, nz, f3(origin), f3(spacing),
                                 to_vec(shell), f3(inv_inertia), bounding_radius);
          },
          nb::arg("grid"), nb::arg("origin"), nb::arg("spacing"), nb::arg("shell"),
          nb::arg("inv_inertia"), nb::arg("bounding_radius"),
          "Append a grid-SDF shape (arguments as set_sdf_shape) and return its index. "
          "set_sdf_shape stays the single-shape entry point (it RESETS the registry).")
      .def(
          "add_scene_shape",
          [](Simulation& s, nb::ndarray<int, nb::c_contig> ni, nb::ndarray<float, nb::c_contig> nr,
             int root, nb::ndarray<float, nb::c_contig> shell, F3Arg inv_inertia,
             float bounding_radius) {
            return s.addSceneShape(std::vector<int>(ni.data(), ni.data() + ni.size()),
                                   std::vector<float>(nr.data(), nr.data() + nr.size()), root,
                                   std::vector<float>(shell.data(), shell.data() + shell.size()),
                                   f3(inv_inertia), bounding_radius);
          },
          nb::arg("node_ints"), nb::arg("node_reals"), nb::arg("root"), nb::arg("shell"),
          nb::arg("inv_inertia"), nb::arg("bounding_radius"),
          "Register a COMPOSED analytic particle shape from core's flat node encoding (the arrays "
          "peclet.core.geom.SceneBuilder.encode() returns; CSG of the full leaf vocabulary): the "
          "collision field is the exact tree, evaluated in canonical body space. shell: (M,3) "
          "surface probe points (bake the tree and run the shell path -- the point-shell model "
          "still needs probes). inv_inertia: unit-mass principal diagonal inverse inertia; "
          "bounding_radius: canonical enclosing radius. THE CANONICAL FRAME MUST BE THE PRINCIPAL "
          "INERTIA FRAME (SceneBuilder.principal_frame emits exactly that); a non-principal tree "
          "runs the diagonal-inertia rotational update on the wrong frame, silently. Returns the "
          "shape id for set_shape_ids.")
      .def(
          "set_shape_ids",
          [](Simulation& s, nb::ndarray<int, nb::c_contig> ids) {
            s.setShapeIds(std::vector<int>(ids.data(), ids.data() + ids.size()));
          },
          nb::arg("ids"),
          "Per-particle shape index (one int per particle, each < num_shapes). Refreshes each "
          "particle's inverse inertia from its new shape. Call it AFTER set_positions: "
          "set_positions resets every particle to shape 0 (and sizes the particle set this call "
          "is checked against).")
      .def_prop_ro("num_shapes", &Simulation::numShapes, "Number of registered shapes.")
      // ---- domain ---------------------------------------------------------------------------
      // Corner overload: set_domain(min, max) 3-tuples (arbitrary origin); keeps the current
      // periodicity.
      .def(
          "set_domain",
          [](Simulation& s, F3Arg mn, F3Arg mx) { s.setDomainMinMax(f3(mn), f3(mx)); },
          nb::arg("min"), nb::arg("max"),
          "Set the domain by (min, max) corner tuples (arbitrary origin); keeps current "
          "periodicity.")
      // The suite-canonical form (suite/docs/NAMING.md 1.1): keyword `extent` / `origin` /
      // `periodic`, the same three names flow.Solver, peclet.voro and the AMR octree take. Bound
      // AFTER the corner overload, so a two-positional call still resolves to (min, max);
      // `extent=` selects this one.
      .def(
          "set_domain",
          [](Simulation& s, F3Arg extent, F3Arg origin, std::array<bool, 3> periodic) {
            s.setDomainCanonical(f3(extent), f3(origin), periodic[0], periodic[1], periodic[2]);
          },
          nb::arg("extent"), nb::arg("origin") = F3Arg{0, 0, 0},
          nb::arg("periodic") = std::array<bool, 3>{true, true, false},
          "Set the domain the suite-canonical way: `extent` is the box SIZE (not the far corner), "
          "`origin` the lower corner, `periodic` the per-axis flags. Equivalent to "
          "`set_domain(min=origin, max=origin+extent)` plus `set_periodic(*periodic)`. Either "
          "form resets the broad-phase skin to 0.1 x the global scale (as set_global_scale does).")
      .def("set_periodic", &Simulation::enablePeriodicity, nb::arg("x"), nb::arg("y"), nb::arg("z"),
           "Set periodic boundaries per axis (x, y, z); read back with the `periodic` property "
           "(suite/docs/NAMING.md 1.4).")
      .def_prop_ro("origin", &Simulation::domainOrigin,
                   "The domain's lower corner (x, y, z) — read-only; set it with `set_domain`.")
      .def_prop_ro("extent", &Simulation::domainExtent,
                   "The domain's SIZE (Lx, Ly, Lz) — read-only, and note it is a size and not the "
                   "far corner, which is `origin + extent` (suite/docs/NAMING.md 1.1).")
      .def_prop_ro("periodic", &Simulation::domainPeriodic,
                   "Per-axis periodicity (x, y, z) — read-only; set it with `set_periodic`.")
      // ---- physics settings ---------------------------------------------------------------
      .def(
          "set_gravity", [](Simulation& s, F3Arg g) { s.setGravity(g[0], g[1], g[2]); },
          nb::arg("gravity"),
          "Set the gravitational acceleration (gx, gy, gz) -- any sequence of three floats. The "
          "default is ZERO.")
      .def_prop_ro("gravity", &Simulation::gravity, "The gravitational acceleration (gx, gy, gz).")
      .def("set_thermostat", &Simulation::setThermostat, nb::arg("temperature"), nb::arg("tau"),
           nb::arg("kB") = 1.0f,
           "Enable a Berendsen-style velocity thermostat (target temperature, coupling time tau; "
           "tau = 0 disables).")
      .def("set_solver_iterations", &Simulation::setSolverIterations, nb::arg("pos"),
           nb::arg("vel"),
           "Set the XPBD position- and velocity-solve iteration counts. `vel` defaults to 0 (no "
           "velocity solve, hence no restitution).")
      .def("set_hertz_material", &Simulation::setHertzMaterial, nb::arg("mat"), nb::arg("youngs"),
           nb::arg("poisson"),
           "Per-material Young's modulus and Poisson ratio for the soft-sphere Hertz-Mindlin "
           "engine (material ids as in set_material_ids).")
      .def(
          "set_stabilization",
          [](Simulation& s, const std::string& mode) {
            if (mode != "off" && mode != "onesided" && mode != "multilevel")
              throw std::invalid_argument(
                  "set_stabilization: expected 'off', 'onesided' or 'multilevel', got '" + mode +
                  "' ('escalate' / 'ordered' are measurement modes: "
                  "diagnostics.set_stabilization)");
            s.setStabilizationMode(mode);
          },
          nb::arg("mode"),
          "Stabilization pass of the staged velocity solve: 'off' (pure symmetric PGS -- exact "
          "ballistic response, but a deep static column mid-collapse cannot be arrested within the "
          "iteration budget), 'onesided' (default: held-lower-side grounded impulses -- arrests "
          "any collapse but is a momentum sink), 'multilevel' (GraphMG contact-graph aggregation: "
          "coarse inelastic solves at super-body masses -- momentum-conserving transport "
          "acceleration).")
      .def_prop_ro("stabilization", &Simulation::stabilizationMode,
                   "The stabilization mode: 'off', 'onesided', 'multilevel' (or a diagnostics "
                   "mode).")
      .def("set_restitution_model", &Simulation::setRestitutionModel, nb::arg("model"),
           "Restitution model of the PGS velocity solve: 'newton' (default; per-substep "
           "restitution on the pre-solve approach) or 'poisson' (event-level: each pair banks its "
           "kinetic compression impulse and releases e x the bank as a budget-capped "
           "separation-velocity target during unloading -- restores the multi-substep-impact "
           "rebound per-substep Newton cannot return).")
      .def_prop_ro("restitution_model", &Simulation::restitutionModel, "'newton' or 'poisson'.")
      .def("set_global_scale", &Simulation::setGlobalScale, nb::arg("scale"),
           "Set a global length scale applied to all particles (effective radius = scale x "
           "global_scale x base radius). Resets the broad-phase skin, as set_domain does -- call "
           "it BEFORE set_domain if you rely on the skin.")
      .def_prop_ro("global_scale", &Simulation::globalScale, "The global length scale.")
      .def("set_dt", &Simulation::setDt, nb::arg("dt"),
           "Set the time step (> 0) every stepper uses: step, step_hertz, step_mpi, "
           "step_hertz_mpi. There is no default -- a step before set_dt raises.")
      .def_prop_ro("dt", &Simulation::dt, "The time step set by set_dt.")
      .def("set_material_params", &Simulation::setMaterialParams, nb::arg("restitution_normal"),
           nb::arg("restitution_tangent") = 0.0f, nb::arg("friction") = 0.0f,
           "Set the BODY-BODY normal/tangential restitution and Coulomb friction coefficient. The "
           "default friction is ZERO, and add_analytic_wall / add_sdf_wall set the particle-WALL "
           "material only -- so a bed more than a few layers deep run with the defaults behaves "
           "like a liquid: it transmits full hydrostatic pressure to the container and the "
           "position solve squeezes grains through the boundary. That failure is silent and looks "
           "like a solver-convergence bug (raising the position iterations and halving dt do not "
           "move it). Set a non-zero friction for any deep bed.")
      .def("set_material_ids", &Simulation::setMaterialIds, nb::arg("ids"),
           "Per-particle material ids (0..7). Pair (e, mu) values come from set_pair_material; "
           "without any set_pair_material call the global material applies everywhere.")
      .def("set_pair_material", &Simulation::setPairMaterial, nb::arg("a"), nb::arg("b"),
           nb::arg("restitution"), nb::arg("friction"),
           "Symmetric pair material (restitution, friction) for material ids (a, b). The first "
           "call seeds every pair from the current global material.")
      .def("set_wall_material_id", &Simulation::setWallMaterialId, nb::arg("wid"), nb::arg("mat"),
           "Give an SDF wall a material id so particle-wall (e, mu) resolves via the pair table "
           "instead of the wall's binary material.")
      // ---- walls ----------------------------------------------------------------------------
      .def(
          "add_analytic_wall",
          [](Simulation& s, nb::ndarray<int, nb::c_contig> node_ints,
             nb::ndarray<float, nb::c_contig> node_reals, int root, bool invert, float restitution,
             float friction) {
            return s.addAnalyticWall(
                std::vector<int>(node_ints.data(), node_ints.data() + node_ints.size()),
                to_vec(node_reals), root, invert, restitution, friction);
          },
          nb::arg("node_ints"), nb::arg("node_reals"), nb::arg("root"), nb::arg("invert"),
          nb::arg("restitution") = 0.0f, nb::arg("friction") = 0.0f,
          "Add an ANALYTIC wall from a core shape tree in the flat node encoding (3 ints + 16 "
          "reals per node). Exact at every scale, with no voxel grid to replicate per rank. "
          "invert=False for a stirrer/obstacle (grains outside the solid); invert=True for a "
          "container, built from a SOLID body (a solid cylinder for a drum). The wall is "
          "positioned by the node TRANSFORM -- an identity transform sits at the origin. Returns "
          "the wall index.")
      .def(
          "add_plane",
          [](Simulation& s, F3Arg p, F3Arg n) { s.addPlane(p[0], p[1], p[2], n[0], n[1], n[2]); },
          nb::arg("point"), nb::arg("normal"),
          "Add an infinite boundary wall plane through `point`, with `normal` pointing into the "
          "half-space the grains occupy (both 3-sequences). XPBD engine only; the Hertz engine "
          "takes SDF walls.")
      .def(
          "add_sdf_wall",
          [](Simulation& s, Grid3 grid, F3Arg origin, F3Arg spacing, float restitution,
             float friction) {
            const int nx = (int)grid.shape(0), ny = (int)grid.shape(1), nz = (int)grid.shape(2);
            return s.addSdfWall(grid_to_vec(grid), nx, ny, nz, f3(origin), f3(spacing), restitution,
                                friction);
          },
          nb::arg("grid"), nb::arg("origin"), nb::arg("spacing"), nb::arg("restitution") = 0.0f,
          nb::arg("friction") = 0.0f,
          "Add a static world-space SDF wall/container (drum barrel, hopper, vibrating tray): "
          "`grid` is the signed distance on an (nx, ny, nz) array indexed [x, y, z] at world nodes "
          "origin + (x, y, z) * spacing -- POSITIVE in the void where grains live, NEGATIVE in the "
          "solid wall; restitution / friction are the binary particle-wall material. Returns the "
          "wall index (for set_wall_velocity). See peclet.dem.build_wall_sdf for the SDF -> "
          "(grid, origin, spacing) helper.")
      // Rigid-body surface-velocity field of a wall: v(x) = linVel + angVel × (x − center).
      // Rotating drum: set angVel about the axis point `center`. Vibrating wall: drive linVel each
      // step.
      .def(
          "set_wall_velocity",
          [](Simulation& s, int wall_index, F3Arg lin, F3Arg ang, F3Arg center) {
            s.setWallVelocity(wall_index, f3(lin), f3(ang), f3(center));
          },
          nb::arg("wall_index"), nb::arg("lin_vel") = F3Arg{0, 0, 0},
          nb::arg("ang_vel") = F3Arg{0, 0, 0}, nb::arg("center") = F3Arg{0, 0, 0},
          "Set a wall's rigid-body surface velocity v(x) = lin_vel + ang_vel × (x − center) (felt "
          "by grains in contact even though the geometry is static). Cheap; call every step for a "
          "vibrating wall.")
      .def(
          "set_wall_transform",
          [](Simulation& s, int wall_index, F3Arg translation, std::array<float, 4> quat) {
            s.setWallTransform(wall_index, f3(translation), quat[0], quat[1], quat[2], quat[3]);
          },
          nb::arg("wall_index"), nb::arg("translation") = F3Arg{0, 0, 0},
          nb::arg("quat") = std::array<float, 4>{0, 0, 0, 1},
          "Place an ANALYTIC wall (add_analytic_wall) rigidly in the world: quat (x,y,z,w) then "
          "translation, composed onto the AUTHORED root transform, so calls are absolute and never "
          "compound. This moves the GEOMETRY -- a stirrer blade sweeps -- where set_wall_velocity "
          "only gives the static surface a velocity field (enough for an axisymmetric drum). Drive "
          "both together each step: integrate ang_vel into the quaternion here and pass the same "
          "ang_vel to set_wall_velocity. Grid-SDF walls have no tree and are refused.")
      // ---- particle state -------------------------------------------------------------------
      .def(
          "set_positions",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            if (a.ndim() != 2 || (a.shape(1) != 3 && a.shape(1) != 4))
              throw std::invalid_argument("set_positions: expected an (N, 3) or (N, 4) array");
            const int n = (int)a.shape(0), k = (int)a.shape(1);
            if (n > s.capacity())
              throw std::invalid_argument(
                  "set_positions: " + std::to_string(n) +
                  " particles exceed Simulation(capacity=" + std::to_string(s.capacity()) + ")");
            const std::vector<float> buf = to_vec(a);
            std::vector<float> xyz((size_t)n * 3), im;
            const bool hasMass = (k == 4);
            if (hasMass)
              im.resize(n);
            for (int i = 0; i < n; ++i) {
              xyz[3 * i] = buf[k * i];
              xyz[3 * i + 1] = buf[k * i + 1];
              xyz[3 * i + 2] = buf[k * i + 2];
              if (hasMass) {
                const float w = buf[k * i + 3];
                im[i] = (w == 0.0f) ? 1.0f : w;
              }
            }
            s.setPositions(xyz);
            if (hasMass)
              s.setInvMass(im);
          },
          nb::arg("positions"),
          "Set particle positions from an (N, 3) array, or (N, 4) whose column 3 is the INVERSE "
          "mass (0 is remapped to 1; use set_inv_mass for a fixed body). N may not exceed "
          "`capacity`. This (re)sizes the particle set and RESETS every particle's quaternion, "
          "scale, inverse mass, shape id (to 0), velocities and gid -- so every other per-particle "
          "setter goes AFTER it.")
      .def(
          "set_velocities",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setVelocities(rows_of(a, 3, s, "set_velocities"));
          },
          nb::arg("velocities"), "Set particle velocities from an (N, 3) array.")
      .def(
          "set_external_forces",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setExternalForces(rows_of(a, 3, s, "set_external_forces"));
          },
          nb::arg("forces"),
          "Set the per-particle external FORCE (e.g. fluid drag) from an (N, 3) array. Applied "
          "each step as dv = F*invMass*dt; persists until re-set or cleared.")
      .def("clear_external_forces", &Simulation::clearExternalForces,
           "Zero all per-particle external forces.")
      .def(
          "set_external_torques",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setExternalTorques(rows_of(a, 3, s, "set_external_torques"));
          },
          nb::arg("torques"),
          "Set the per-particle external TORQUE in the WORLD frame from an (N, 3) array (the "
          "resolved-CFD-DEM hydrodynamic torque, a magnetic couple, ...). Applied each step in the "
          "angular predictor as Euler's equation in the body frame, dw = invI*(tau_body - w x I "
          "w)*dt, alongside the gyroscopic term that is already there; persists until re-set or "
          "cleared. Only bodies with a finite inertia respond -- a torque on a body whose "
          "invInertia is zero is inert, exactly as the gyroscopic term is. Sleeping is disabled "
          "while a torque is set, as it is for external forces.")
      .def("clear_external_torques", &Simulation::clearExternalTorques,
           "Zero all per-particle external torques (and re-enable island sleeping).")
      .def(
          "set_quaternions",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setQuaternions(rows_of(a, 4, s, "set_quaternions"));
          },
          nb::arg("quaternions"),
          "Set particle orientation quaternions (x, y, z, w) from an (N, 4) array.")
      .def(
          "set_angular_velocities",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setAngularVelocities(rows_of(a, 3, s, "set_angular_velocities"));
          },
          nb::arg("angular_velocities"),
          "Set particle angular velocities (body frame) from an (N, 3) array.")
      .def(
          "set_inv_inertia",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setInvInertia(rows_of(a, 3, s, "set_inv_inertia"));
          },
          nb::arg("inv_inertia"),
          "Set the per-particle principal-frame diagonal inverse inertia from an (N, 3) array.")
      .def(
          "set_inv_mass",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setInvMass(flat_of(a, s, "set_inv_mass"));
          },
          nb::arg("inv_mass"), "Set per-particle inverse mass, an (N,) array (0 = fixed body).")
      .def("set_scales_uniform", &Simulation::setScalesUniform, nb::arg("scale"),
           "Set a single uniform scale for all particles.")
      .def(
          "set_scales",
          [](Simulation& s, nb::ndarray<float, nb::c_contig> a) {
            s.setScales(flat_of(a, s, "set_scales"));
          },
          nb::arg("scales"), "Set per-particle scales from an (N,) array (the growth target).")
      .def("set_growth_params", &Simulation::setGrowthParams, nb::arg("rate"),
           nb::arg("new_factor") = -1.0f,
           "Set the particle growth rate (factor *= exp(rate*dt) per step, capped at 1) and, if "
           "new_factor > 0, the current growth factor (a negative value keeps it; 0.01 if growth "
           "was inactive).")
      .def_prop_ro("growth_factor", &Simulation::growthFactor,
                   "The current particle growth factor (-1 = growth inactive).")
      .def_prop_ro("growth_rate", &Simulation::getGrowthRate, "The particle growth rate.")
      .def(
          "get_masses", [](const Simulation& s) { return flat(s.getMasses()); },
          "Return per-particle masses (1 / inverse mass; 0 for fixed bodies) as an (N,) array.")
      .def(
          "get_positions", [](const Simulation& s) { return rows(s.getPositions(), 3); },
          "Return particle positions as an (N, 3) numpy array.")
      .def(
          "get_velocities", [](const Simulation& s) { return rows(s.getVelocities(), 3); },
          "Return particle velocities as an (N, 3) numpy array.")
      .def(
          "get_angular_velocities",
          [](const Simulation& s) { return rows(s.getAngularVelocities(), 3); },
          "Return particle angular velocities (body frame) as an (N, 3) numpy array.")
      .def(
          "get_inv_inertia", [](const Simulation& s) { return rows(s.getInvInertia(), 3); },
          "Return the per-particle principal-frame diagonal inverse inertia as an (N, 3) array.")
      .def(
          "get_quaternions", [](const Simulation& s) { return rows(s.getQuaternions(), 4); },
          "Return particle orientation quaternions (x, y, z, w) as an (N, 4) numpy array.")
      .def(
          "get_scales", [](const Simulation& s) { return flat(s.getScales()); },
          "Return per-particle scales as an (N,) numpy array.")
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
          "get_external_forces_view",
          [](const Simulation& s) {
            return peclet::core::python::view_to_ndarray(Kokkos::subview(
                s.externalForcesView(), Kokkos::make_pair(0, s.numParticles()), Kokkos::ALL));
          },
          "Zero-copy (N,3) device array of the per-particle external force (NumPy view on host, "
          "DLPack/CuPy on GPU) — write fluid drag here directly to avoid a host round-trip.")
      .def(
          "get_inv_mass_view",
          [](const Simulation& s) {
            return peclet::core::python::view_to_ndarray(
                Kokkos::subview(s.invMassView(), Kokkos::make_pair(0, s.numParticles())));
          },
          "Zero-copy (N,) device array of per-particle inverse mass (NumPy view on host, "
          "DLPack/CuPy on GPU) — read-only use; needed for stiff-safe drag integration.")
      // ---- stepping -------------------------------------------------------------------------
      .def("step", &Simulation::step, nb::arg("n") = 1,
           "Advance `n` XPBD substeps of the time step set by set_dt (raises if set_dt was never "
           "called).")
      .def("relax", &Simulation::relax, nb::arg("n") = 1,
           "Run `n` dynamics-free RELAXATION substeps: overlap removal only, no gravity and no "
           "velocity update -- the growth-packing protocol's settle move (dt = 0 inside; the "
           "stored dt is untouched and set_dt is not required).")
      .def("step_hertz", &Simulation::stepHertz, nb::arg("substeps") = 1,
           nb::arg("skin_frac") = 0.3f,
           "Advance `substeps` explicit soft-sphere Hertz-Mindlin steps of the time step set by "
           "set_dt (spheres, SDF walls, non-periodic): viscoelastic Hertz normal force + Mindlin "
           "shear-history spring, Coulomb-clamped; (e, mu) from the pair-material tables, "
           "stiffness from set_hertz_material. skin_frac is the Verlet pair-list skin as a "
           "fraction of the radius.")
      // ---- read-out -------------------------------------------------------------------------
      .def_prop_ro("num_particles", &Simulation::numParticles, "The number of particles.")
      .def_prop_ro(
          "num_contacts", [](Simulation& s) { return s.numContacts(); },
          "Broad-phase candidate pairs found in the last step (ArborX BVH query).")
      .def_prop_ro(
          "num_manifolds", [](Simulation& s) { return s.numManifolds(); },
          "Narrow-phase contact manifolds (touching pairs) resolved in the last step.")
      .def_prop_ro(
          "max_overlap", [](Simulation& s) { return s.maxOverlap(); },
          "Maximum pair interpenetration recorded by the position solver in the last step (its "
          "last-iteration residual, so it under-reports the committed overlap -- see "
          "docs/packing_investigation.md). compute_overlaps() measures the committed state.")
      .def("compute_overlaps", &Simulation::computeOverlaps,
           "Measure and return the maximum pair interpenetration of the current committed state.")
      .def(
          "get_sdf_grid",
          [](Simulation& s, std::tuple<int, int, int> res) {
            auto [rx, ry, rz] = res;
            // x-fastest samples -> an (rx, ry, rz) array indexed [x, y, z] (Fortran order), the
            // suite's grid layout and what set_sdf_shape / add_sdf_wall take back.
            return peclet::core::python::vector_to_ndarray(
                s.getSdfGrid(rx, ry, rz), {(std::size_t)rx, (std::size_t)ry, (std::size_t)rz},
                {1, (std::int64_t)rx, (std::int64_t)rx * ry});
          },
          nb::arg("resolution"),
          "Reconstruct the packed-bed SDF (negative inside solid) over the domain on an (rx, ry, "
          "rz) grid; returns an (rx, ry, rz) array indexed [x, y, z].")
      .def(
          "export_sdf",
          [](Simulation& s, const std::string& filename, std::tuple<int, int, int> res) {
            auto [rx, ry, rz] = res;
            s.exportSdf(filename, rx, ry, rz);
          },
          nb::arg("filename"), nb::arg("resolution"),
          "Reconstruct and write the packed-bed SDF on an (rx, ry, rz) grid to a VTI file.")
      .def("write_vtp", &Simulation::writeVtp, nb::arg("filename"),
           "Write particle state to a VTP file (ParaView/Ovito).")
      .def("export_lammps", &Simulation::exportLammps, nb::arg("filename"), nb::arg("step"),
           "Export particle state to a LAMMPS dump file.")
      // ---- execution policy -----------------------------------------------------------------
      .def("set_sleeping", &Simulation::setSleeping, nb::arg("enabled"),
           nb::arg("threshold_scale") = 2.0f, nb::arg("consecutive") = 64,
           nb::arg("wake_scale") = 40.0f, nb::arg("wake_on_lost_contact") = false,
           nb::arg("immovable_frac") = 0.01f,
           "Island sleeping (single-GPU statics, default ON): freeze grounded bodies whose motion "
           "stays below threshold_scale x the resting floor for `consecutive` substeps; wake only "
           "above wake_scale x that floor (hysteresis vs residual jitter). wake_on_lost_contact "
           "additionally wakes a sleeper whose support disappeared; immovable_frac is a sleeper's "
           "effective inverse-mass fraction in the solve (0 = exactly immovable). Requires gravity "
           "and no external force; inert under MPI.")
      .def_prop_ro("sleeping", &Simulation::sleeping, "Whether island sleeping is enabled.")
      .def_prop_ro(
          "num_asleep", [](Simulation& s) { return s.numAsleep(); },
          "Number of currently-sleeping real bodies.")
      .def("set_verlet_skin", &Simulation::setVerletSkin, nb::arg("skin_frac"),
           "Enable the Verlet-cached impulse broadphase (single-GPU, non-periodic, default OFF, "
           "i.e. skin_frac = 0): skip the ArborX rebuild while nothing moved more than skin/2 "
           "(skin = skin_frac x max grain radius).")
      .def_prop_ro("verlet_skin", &Simulation::verletSkin,
                   "Broadphase-skin fraction of the max grain radius (0 = rebuild every step).")
      .def("set_incremental_coloring", &Simulation::setIncrementalColoring, nb::arg("enabled"),
           "Incremental (warm-started) graph coloring of the contact manifolds (default True): "
           "reuse last substep's colors and repair only the conflicts. This CHANGES RESULTS -- "
           "the coloring fixes the Gauss-Seidel sweep order -- so False reproduces the "
           "pre-incremental behaviour rather than merely running slower.")
      .def_prop_ro("incremental_coloring", &Simulation::incrementalColoring,
                   "Whether incremental (warm-started) coloring is enabled.")
#ifdef PECLET_DEM_MPI
      // ---- distributed step; built only with -DPECLET_DEM_MPI=ON ---------------------------
      .def(
          "init_mpi",
          [](Simulation& s, std::tuple<double, double, double> origin,
             std::tuple<double, double, double> extent, std::tuple<long, long, long> cells,
             std::tuple<bool, bool, bool> periodic) {
            int inited = 0;
            MPI_Initialized(&inited);
            if (!inited) {
              int argc = 0;
              char** argv = nullptr;
              MPI_Init(&argc, &argv);
            }
            s.initMpi(origin, extent, cells, periodic, MPI_COMM_WORLD);
          },
          nb::arg("origin"), nb::arg("extent"), nb::arg("cells"), nb::arg("periodic"),
          "Set up the ORB block decomposition + core particle halo for the distributed step on "
          "the global domain (`origin` lower corner, `extent` SIZE, `cells` per axis for the ORB "
          "grid, `periodic` per axis -- the suite's domain quartet).")
      .def("enable_mpi_step", &Simulation::enableMpiStep, nb::arg("rcut"),
           nb::arg("sync_every") = 1, nb::arg("forward_rotation") = true,
           nb::arg("rebalance_every") = 0, nb::arg("verlet_skin") = 0.0,
           "Enable the distributed step: ghost cutoff rcut, sync cadence, rotation forwarding, the "
           "load-rebalance interval in steps (0 = fixed decomposition), and the Verlet ghost-reuse "
           "skin (0 = rebuild the halo topology every substep; >0 = reuse it until a particle "
           "moves > skin).")
      .def("step_mpi", &Simulation::stepMpi, nb::arg("n") = 1,
           "Advance the distributed (MPI) simulation by `n` steps of the time step set by set_dt, "
           "with halo exchange.")
      .def("step_hertz_mpi", &Simulation::stepHertzMpi, nb::arg("substeps") = 1,
           nb::arg("skin_frac") = 0.3f,
           "Advance `substeps` distributed explicit Hertz-Mindlin (force-based) steps of the time "
           "step set by set_dt — the MPI counterpart of step_hertz on the init_mpi/enable_mpi_step "
           "decomposition. rebalance_every counts CALLS of this method; migration carries the "
           "Mindlin history.")
      .def("rebalance", &Simulation::rebalance,
           "Re-decompose by particle count and migrate ownership now; returns this rank's new "
           "owned count.")
      .def("migrate_to_weights", &Simulation::migrateToWeights, nb::arg("weights"),
           "Co-rebalance: migrate ownership onto the weighted ORB of per-cell weights (global "
           "x-fastest, matching the ORB grid) -- the SAME partition the coupled flow solver "
           "redistributes onto from the same weight field. Returns this rank's new owned count.")
      .def_prop_ro("rank", &Simulation::rank, "This rank's MPI index.")
      .def_prop_ro("num_ghost", &Simulation::numGhost,
                   "The number of ghost particles on this rank.")
#endif
      ;

  // Module-level LAMMPS dump writer from raw arrays: export_lammps(filename, step, pos, vel,
  // quats, radii, box_min, box_max, periodic).
  m.def(
      "export_lammps",
      [](const std::string& filename, int step, nb::ndarray<float, nb::c_contig> pos,
         nb::ndarray<float, nb::c_contig> vel, nb::ndarray<float, nb::c_contig> quats,
         nb::ndarray<float, nb::c_contig> radii, std::optional<F3Arg> box_min,
         std::optional<F3Arg> box_max, bool periodic) {
        const float* pmn = box_min ? box_min->data() : nullptr;
        const float* pmx = box_max ? box_max->data() : nullptr;
        peclet::dem::writeLammpsDump(filename, step, to_vec(pos), to_vec(vel), to_vec(quats),
                                     to_vec(radii), pmn, pmx, periodic);
      },
      nb::arg("filename"), nb::arg("step"), nb::arg("pos"), nb::arg("vel"), nb::arg("quats"),
      nb::arg("radii"), nb::arg("box_min") = std::nullopt, nb::arg("box_max") = std::nullopt,
      nb::arg("periodic") = false,
      "Module-level LAMMPS dump writer from raw arrays (filename, step, pos, vel, quats, radii, "
      "box corners, periodic flag).");
}
