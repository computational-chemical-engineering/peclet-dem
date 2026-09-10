/// @file
/// @brief dem — the `Simulation` facade: the host-facing, binding-agnostic std::vector API the
/// nanobind module (dem_bindings.cpp) drives from Python.
///
/// Derives from ShapeRegistry (shape_registry.hpp: shapes, shells, inertias, the particle SoA
/// `P_`) and adds the domain / material / solver setters, walls and planes, the per-particle
/// state setters and getters, the steppers (`step` / `relax` / `step_hertz` and, gated
/// PECLET_DEM_MPI, `step_mpi` / `step_hertz_mpi` + rebalancing) and the exports. The step
/// drivers themselves are the free functions of step_solve.hpp / step_solve_mpi.hpp.
#ifndef DEM_SIM_HPP
#define DEM_SIM_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "broadphase_arborx.hpp"
#include "contact_preprocessing.hpp"
#include "integration.hpp"
#include "io.hpp"
#include "narrowphase.hpp"
#include "output_sdf.hpp"
#include "particles.hpp"
#include "peclet/core/common/view.hpp"  // peclet::core::toVector — single-copy device View -> host std::vector (S2a)
#include "peclet/core/geom/scene_builder.hpp"
#include "periodicity.hpp"
#include "shape_registry.hpp"  // ShapeRegistry (the base class: shapes, shells, P_)
#include "shapes_portable.hpp"
#include "sleeping.hpp"            // island sleeping / freezing (single-GPU statics)
#include "solve_driver.hpp"        // demSolveContacts + SoloSolveHooks + readInt/readFloat
#include "solve_driver_force.hpp"  // demStepForce + HertzMindlinLaw + demStepHertz
#include "solver_friction.hpp"
#include "solver_hertz.hpp"
#include "solver_multilevel.hpp"
#include "solver_position.hpp"
#include "solver_velocity.hpp"
#include "step_solve.hpp"  // demStep / computeOverlapsKokkos / growContactBuffers

#ifdef PECLET_DEM_MPI
#include "mpi_halo.hpp"        // ParticleHalo (gated; default module never includes it)
#include "step_solve_mpi.hpp"  // demStepMpi / demStepHertzMpi + the MPI hook policies
#endif

namespace peclet::dem {

/// Host-facing facade with std::vector setters/getters (binding-agnostic).
class Simulation : public ShapeRegistry {
 public:
  explicit Simulation(int capacity) {
    registry().push_back(this);
    P_.allocate(capacity, capacity * 64, capacity * 16, /*shapes*/ 1, /*shell*/ 1, /*planes*/ 8);
    // default sphere shape (radius 1) + identity-ish defaults
    setSphereShape(1.0f);
  }
  ~Simulation() {
    auto& r = registry();
    r.erase(std::remove(r.begin(), r.end(), this), r.end());
  }

  // Teardown safety: the Particles SoA holds Kokkos Views, so they MUST be freed before
  // Kokkos::finalize (else "deallocated after finalize" aborts). releaseAll() (called from the
  // module's atexit, before finalize) frees every live Sim's Views, so callers need not `del sim;
  // gc.collect()` themselves.
  void releaseViews() {
    P_ = Particles{};
#ifdef PECLET_DEM_MPI
    halo_.reset();  // the halo also owns Kokkos Views (gather/forward buffers + its core
                    // sub-objects') that must be freed before Kokkos::finalize, else "deallocated
                    // after finalize" aborts. Destroying it via the unique_ptr frees them all.
#endif
  }
  static void releaseAll() {
    for (auto* s : registry())
      s->releaseViews();
  }
  static std::vector<Simulation*>& registry() {
    static std::vector<Simulation*> r;
    return r;
  }

  /// Add an ANALYTIC wall from a core shape tree in the flat encoding (Layer 1). Replaces the
  /// voxelised container/stirrer: exact at every scale, and no per-rank replicated sample pool.
  ///
  /// `nodeInts` / `nodeReals` are core's flat node encoding (geom/scene_builder.hpp: 3 ints and 16
  /// reals per node, kind/aux/params/transform).
  ///
  /// SIGN. A wall must read POSITIVE in the void where the grains live and NEGATIVE in wall
  /// material; core's leaves are negative-inside-solid. So:
  ///   * a STIRRER, a paddle, an obstacle -- grains are OUTSIDE the tree's solid: invert = FALSE.
  ///   * a CONTAINER -- grains are INSIDE: describe the container as a SOLID body (a solid
  ///     cylinder for a drum: kHollowCylinder with thickness = 2*rOuter) and pass invert = TRUE.
  ///     Inverting a solid is what makes everything beyond the barrel read as wall, so a grain
  ///     that escapes is pushed back.
  ///
  /// Do NOT build a container from a thin TUBE and invert it: a tube's bore is already void, so
  /// inverting makes the bore read as solid and the solver pushes every grain out through it.
  ///
  /// POSITION. An analytic wall is placed by its node TRANSFORM. Unlike add_sdf_wall, where the
  /// grid origin positions the field implicitly, a leaf with an identity transform sits at the
  /// ORIGIN -- which is a domain corner in the usual [0,L]^3 setup, not the centre. Getting this
  /// wrong looks exactly like a sign error: grains start outside the body, read "solid", and are
  /// driven out of the domain.
  ///
  /// Grid leaves are not supported inside an analytic wall tree -- use add_sdf_wall for a sampled
  /// container.
  int addAnalyticWall(const std::vector<int>& nodeInts, const std::vector<float>& nodeReals,
                      int root, bool invert, float restitution, float friction) {
    namespace g = peclet::core::geom;
    if (nodeInts.size() % g::kNodeIntStride || nodeReals.size() % g::kNodeRealStride)
      throw std::runtime_error("add_analytic_wall: node arrays are not a whole number of records");
    const int n = static_cast<int>(nodeInts.size() / g::kNodeIntStride);
    if (n == 0 || static_cast<int>(nodeReals.size() / g::kNodeRealStride) != n)
      throw std::runtime_error("add_analytic_wall: node int/real counts disagree");
    if (root < 0 || root >= n)
      throw std::runtime_error("add_analytic_wall: root out of range");
    const int base = static_cast<int>(wallNodesHost_.size());
    for (int i = 0; i < n; ++i) {
      g::ShapeNode<float> nd = g::decodeNode<float>(&nodeInts[i * g::kNodeIntStride],
                                                    &nodeReals[i * g::kNodeRealStride]);
      if (nd.kind == g::kGrid)
        throw std::runtime_error("add_analytic_wall: grid leaves are not supported here");
      if (nd.kind >= g::kCsgBase) {  // rebase child indices into the shared pool
        nd.aux0 += base;
        nd.aux1 += base;
      }
      wallNodesHost_.push_back(nd);
    }
    P_.wallNodes = Kokkos::View<g::ShapeNode<float>*, CpMem>("wallNodes", wallNodesHost_.size());
    auto hn = Kokkos::create_mirror_view(P_.wallNodes);
    for (std::size_t i = 0; i < wallNodesHost_.size(); ++i)
      hn(i) = wallNodesHost_[i];
    Kokkos::deep_copy(P_.wallNodes, hn);

    WallSdf w{};
    w.shapeRoot = base + root;
    w.nodeCount = static_cast<int>(wallNodesHost_.size());
    // R1: the AUTHORED root transform, kept so setWallTransform composes onto it instead of onto
    // the previous frame's result (which would compound and drift).
    wallRootTf_[base + root] = wallNodesHost_[(std::size_t)(base + root)].transform;
    w.sign = invert ? -1.0f : 1.0f;
    w.restitution = restitution;
    w.friction = friction;
    const int idx = static_cast<int>(wallsHost_.size());
    wallsHost_.push_back(w);
    // Every wall's node pointer must be refreshed: the pool View was just reallocated.
    for (auto& wh : wallsHost_)
      if (wh.shapeRoot >= 0)
        wh.nodes = P_.wallNodes.data();
    P_.numWalls = static_cast<int>(wallsHost_.size());
    uploadWalls();
    ensureContactCapacity();
    return idx;
  }

  void setDomain(float lx, float ly, float lz, bool px, bool py, bool pz) {
    P_.domain = Domain{F3{0, 0, 0}, F3{lx, ly, lz}, F3{lx, ly, lz}, px, py, pz};
    P_.skin = 0.1f * P_.globalScale;
  }
  // CUDA Simulation::set_domain(min, max): arbitrary origin; keeps the current periodicity flags.
  void setDomainMinMax(F3 mn, F3 mx) {
    P_.domain = Domain{mn,
                       mx,
                       F3{mx.x - mn.x, mx.y - mn.y, mx.z - mn.z},
                       P_.domain.periodic_x,
                       P_.domain.periodic_y,
                       P_.domain.periodic_z};
    P_.skin = 0.1f * P_.globalScale;
  }
  void enablePeriodicity(bool x, bool y, bool z) {
    P_.domain.periodic_x = x;
    P_.domain.periodic_y = y;
    P_.domain.periodic_z = z;
  }
  // The suite-canonical domain quartet (suite/docs/NAMING.md 1.1): `origin` is the lower corner,
  // `extent` the SIZE, `periodic` the per-axis flags — the same four names flow, voro and the AMR
  // octree use. `setDomainCanonical` is what the keyword form of `set_domain` binds to.
  void setDomainCanonical(F3 extent, F3 origin, bool px, bool py, bool pz) {
    P_.domain = Domain{origin, F3{origin.x + extent.x, origin.y + extent.y, origin.z + extent.z},
                       extent, px,
                       py,     pz};
    P_.skin = 0.1f * P_.globalScale;
  }
  std::tuple<float, float, float> domainOrigin() const {
    return {P_.domain.min.x, P_.domain.min.y, P_.domain.min.z};
  }
  std::tuple<float, float, float> domainExtent() const {
    return {P_.domain.size.x, P_.domain.size.y, P_.domain.size.z};
  }
  std::tuple<bool, bool, bool> domainPeriodic() const {
    return {P_.domain.periodic_x, P_.domain.periodic_y, P_.domain.periodic_z};
  }
  void setGravity(float gx, float gy, float gz) { P_.gravity = F3{gx, gy, gz}; }
  std::tuple<float, float, float> gravity() const {
    return {P_.gravity.x, P_.gravity.y, P_.gravity.z};
  }
  void setThermostat(float temperature, float tau, float kB) {  // Berendsen; tau=0 disables
    P_.thermostatTemp = temperature;
    P_.thermostatTau = tau;
    P_.thermostatKB = kB;
  }
  void setSolverIterations(int pos, int vel) {
    P_.positionIterations = pos;
    P_.velocityIterations = vel;
  }
  // Select the single-GPU collision solves: true (default) = colored Gauss–Seidel for both the
  // restitution and the overlap solve, false = count-averaged Jacobi (legacy). For A/B validation.
  void setVelocityUseGS(bool useGS) { P_.velocityUseGS = useGS; }
  bool velocityUseGS() const { return P_.velocityUseGS; }
  /// Select the stabilization pass of the staged velocity solve: "off" (pure symmetric PGS),
  /// "onesided" (default: held-lower-side grounded impulses -- arrests any collapse but is a
  /// momentum sink), "multilevel" (GraphMG contact-graph aggregation: coarse inelastic solves
  /// at super-body masses -- momentum-conserving transport acceleration), "escalate" (extra
  /// symmetric sweeps up to 256; diagnostic/fallback), "ordered" (level-ordered symmetric
  /// sweeps; measurement mode, known insufficient for deep columns).
  void setStabilizationMode(const std::string& mode) {
    if (mode == "off")
      P_.stabilizationMode = 0;
    else if (mode == "onesided")
      P_.stabilizationMode = 1;
    else if (mode == "multilevel")
      P_.stabilizationMode = 2;
    else if (mode == "escalate")
      P_.stabilizationMode = 3;
    else if (mode == "ordered")
      P_.stabilizationMode = 4;
    else
      throw std::invalid_argument(
          "set_stabilization: expected 'off', 'onesided', 'multilevel', 'escalate' or "
          "'ordered'");
  }
  std::string stabilizationMode() const {
    static const char* const names[] = {"off", "onesided", "multilevel", "escalate", "ordered"};
    return names[P_.stabilizationMode];
  }
  /// Restitution model of the PGS velocity solve: "newton" (default; per-substep restitution on
  /// the pre-solve approach — the pre-existing behaviour) or "poisson" (event-level: each pair
  /// banks its kinetic compression impulse and releases e x the bank as a budget-capped
  /// separation-velocity target during unloading — restores the multi-substep-impact rebound that
  /// per-substep Newton structurally cannot return).
  void setRestitutionModel(const std::string& model) {
    if (model == "newton")
      P_.restitutionModel = 0;
    else if (model == "poisson")
      P_.restitutionModel = 1;
    else
      throw std::invalid_argument("set_restitution_model: expected 'newton' or 'poisson'");
  }
  std::string restitutionModel() const { return P_.restitutionModel == 1 ? "poisson" : "newton"; }
  /// Island sleeping / freezing (single-GPU statics, default ON; `enabled=False` disables). A
  /// REAL body whose linear AND
  /// angular motion stays below `scale` x the resting floor (2 dt |g|) for K substeps while
  /// grounded is put to sleep: velocity zeroed, integration skipped, and a manifold whose BOTH
  /// endpoints are asleep (a static wall counts) is excluded from the colouring / sweeps /
  /// multilevel hierarchy — so a settled bed collapses to the broad/narrow-phase floor. A sleeper
  /// keeps a small POSITIVE effective inverse mass in the solve (sleepImmovableFrac x its own;
  /// heavy but not perfectly rigid), so an awake body wedged between sleepers relieves against them
  /// instead of the PGS normal impulse diverging; its velocity is re-zeroed each substep so no
  /// momentum accumulates. It wakes only when disturbed (fast approaching neighbour, contact-set
  /// change, moving wall).
  /// Requires gravity on and no external (CFD-DEM drag) force; inert under MPI.
  /// wake_on_lost_contact additionally wakes a sleeper whose support disappeared (rule (b));
  /// immovable_frac is the sleeper's effective inverse-mass fraction in the solve (0 = exactly
  /// immovable; the default 0.01 keeps it 100x a grain's mass but not infinitely rigid).
  void setSleeping(bool enabled, float threshold_scale = 2.0f, int consecutive = 64,
                   float wake_scale = 40.0f, bool wake_on_lost_contact = false,
                   float immovable_frac = 0.01f) {
    P_.sleepingEnabled = enabled;
    if (threshold_scale > 0.0f)
      P_.sleepScale = threshold_scale;
    if (consecutive > 0)
      P_.sleepK = consecutive;
    if (wake_scale > 0.0f)
      P_.wakeScale = wake_scale;  // hysteresis: wake only well above the residual settling jitter
    P_.sleepWakeLostContact = wake_on_lost_contact;
    if (immovable_frac >= 0.0f)
      P_.sleepImmovableFrac = immovable_frac;
  }
  /// True while island sleeping is enabled.
  bool sleeping() const { return P_.sleepingEnabled; }
  /// Verlet-cached impulse broadphase (single-GPU, non-periodic; default OFF). skin_frac is the
  /// broadphase-skin fraction of the max grain radius: the ArborX rebuild is skipped while no
  /// particle has moved more than skin/2, so between rebuilds the candidate list is a superset and
  /// the narrowphase yields identical contacts. Composes with sleeping (a frozen bed never
  /// rebuilds).
  void setVerletSkin(float skin_frac) {
    P_.verletSkinFrac = skin_frac;
    P_.impNumPairs = -1;  // invalidate the cache
  }
  float verletSkin() const { return P_.verletSkinFrac; }
  /// CUDA-graph replay of the solver's iteration loops (single-rank CUDA; default ON). Capture
  /// collapses each iteration's launch storm into one replay; the arithmetic, the residual
  /// readback and the adaptive stop are unchanged, so results are bit-identical either way.
  /// Inert on every non-CUDA backend and on the distributed step (ghost syncs forbid capture).
  void setCudaGraphs(bool enabled) { P_.cudaGraphs = enabled; }
  bool cudaGraphs() const { return P_.cudaGraphs; }
  /// Fused colour sweeps (CUDA): run a whole sweep — and where eligible the whole adaptive
  /// iteration loop — as ONE kernel iterating device-side behind software grid barriers.
  /// "auto" (default) uses them exactly where graph replay is unavailable (the distributed step,
  /// or set_cuda_graphs(False)); "on"/"off" force. Same colour ordering, bit-identical results.
  void setFusedSweeps(const std::string& mode) {
    if (mode == "auto")
      P_.fusedSweeps = -1;
    else if (mode == "off")
      P_.fusedSweeps = 0;
    else if (mode == "on")
      P_.fusedSweeps = 1;
    else
      throw std::invalid_argument("set_fused_sweeps: expected 'auto', 'on' or 'off'");
  }
  std::string fusedSweeps() const {
    return P_.fusedSweeps < 0 ? "auto" : (P_.fusedSweeps == 1 ? "on" : "off");
  }
  /// Incremental (warm-started) graph colouring of the contact manifolds (default ON): reuse
  /// last substep's colours and repair only the conflicts instead of recolouring from scratch.
  /// This CHANGES RESULTS -- the colouring fixes the Gauss-Seidel sweep order -- so it is a
  /// numerical option, not just a speed one; False reproduces the pre-incremental behaviour.
  void setIncrementalColoring(bool enabled) { P_.incrementalColoring = enabled; }
  bool incrementalColoring() const { return P_.incrementalColoring; }
  /// Number of currently-sleeping real bodies (diagnostics / tests).
  int numAsleep() {
    int n = 0;
    auto a = P_.asleep;
    Kokkos::parallel_reduce(
        "peclet::dem::count_asleep", Kokkos::RangePolicy<CpExec>(0, P_.numReal),
        KOKKOS_LAMBDA(int i, int& acc) { acc += a(i) ? 1 : 0; }, n);
    return n;
  }
  /// Per-material Young's modulus + Poisson ratio for the Hertz-Mindlin engine (material ids as
  /// in setMaterialIds; without ids every particle is material 0).
  void setHertzMaterial(int mat, float youngs, float poisson) {
    if (mat < 0 || mat >= 8)
      throw std::invalid_argument("material id out of range");
    auto he = Kokkos::create_mirror_view(P_.hertzE);
    auto hn = Kokkos::create_mirror_view(P_.hertzNu);
    Kokkos::deep_copy(he, P_.hertzE);
    Kokkos::deep_copy(hn, P_.hertzNu);
    he(mat) = youngs;
    hn(mat) = poisson;
    Kokkos::deep_copy(P_.hertzE, he);
    Kokkos::deep_copy(P_.hertzNu, hn);
  }
  /// Advance `substeps` explicit soft-sphere Hertz-Mindlin steps of the stored dt (device-side
  /// loop; the (e, mu) pairs come from the impulse solver's material tables).
  void stepHertz(int substeps, float skin_frac) {
    requireDt("step_hertz");
    demStepHertz(P_, P_.dt, substeps, skin_frac);
  }
  void setGlobalScale(float s) {
    P_.globalScale = s;
    P_.skin = 0.1f * s;
  }
  float globalScale() const { return P_.globalScale; }
  /// The time step every stepper uses (suite/docs/NAMING.md 1.5). Must be > 0; a step called
  /// before set_dt raises instead of silently running on a default.
  void setDt(float dt) {
    if (!(dt > 0.0f))
      throw std::invalid_argument("set_dt: dt must be > 0 (relax() is the dynamics-free substep)");
    P_.dt = dt;
    dtSet_ = true;
  }
  float dt() const { return P_.dt; }
  int capacity() const { return P_.capacity; }
  // (restitution_normal, restitution_tangent, friction) to match CUDA set_material_params; the
  // Kokkos pipeline currently carries normal restitution + dynamic friction (tangential restitution
  // unused).
  // NOTE (found via peclet-examples/stirred-column, 2026-08-30): the DEFAULT body-body material is
  // frictionless. add_analytic_wall / add_sdf_wall set the particle-WALL material only, so a bed
  // more than a few layers deep with the default body-body friction behaves like a liquid -- it
  // transmits full hydrostatic pressure to the container and the position solve squeezes grains
  // through the boundary. The failure is silent and looks exactly like a solver-convergence bug
  // (raising the position iterations 4,4 -> 24,12 and halving dt changed the measured leakage not
  // at all). Set a non-zero friction here for any deep bed.
  void setMaterialParams(float restitution_normal, float restitution_tangent, float friction) {
    P_.restitutionNormal = restitution_normal;
    P_.frictionDynamic = friction;
    P_.restitutionTangent = restitution_tangent;  // Walton beta (impact law, cone-clamped)
  }
  /// Per-particle material ids (0..kMaxMaterials-1); pair (e, mu) values come from
  /// setPairMaterial. Ids default to 0; without any setPairMaterial call the global material
  /// applies everywhere.
  void setMaterialIds(const std::vector<int>& ids) {
    auto h = Kokkos::create_mirror_view(P_.materialId);
    Kokkos::deep_copy(h, P_.materialId);
    for (int i = 0; i < P_.numReal && i < (int)ids.size(); ++i)
      h(i) = static_cast<unsigned char>(ids[i]);
    Kokkos::deep_copy(P_.materialId, h);
  }
  /// Set the symmetric pair material (restitution, friction) for material ids (a, b). The first
  /// call allocates the pair table, initialised from the current global material for every pair.
  void setPairMaterial(int a, int b, float restitution, float friction) {
    if (a < 0 || b < 0 || a >= kMaxMaterials || b >= kMaxMaterials)
      throw std::invalid_argument("material id out of range");
    if (P_.pairMaterials.extent(0) == 0) {
      P_.pairMaterials =
          Kokkos::View<float*, CpMem>("pairMaterials", kMaxMaterials * kMaxMaterials * 2);
      auto h0 = Kokkos::create_mirror_view(P_.pairMaterials);
      for (int i = 0; i < kMaxMaterials * kMaxMaterials; ++i) {
        h0(2 * i) = P_.restitutionNormal;
        h0(2 * i + 1) = P_.frictionDynamic;
      }
      Kokkos::deep_copy(P_.pairMaterials, h0);
    }
    auto h = Kokkos::create_mirror_view(P_.pairMaterials);
    Kokkos::deep_copy(h, P_.pairMaterials);
    for (auto xy : {std::pair<int, int>{a, b}, std::pair<int, int>{b, a}}) {
      h((xy.first * kMaxMaterials + xy.second) * 2) = restitution;
      h((xy.first * kMaxMaterials + xy.second) * 2 + 1) = friction;
    }
    Kokkos::deep_copy(P_.pairMaterials, h);
  }
  /// Give an SDF wall a material id so particle-wall (e, mu) also resolves via the pair table.
  void setWallMaterialId(int wid, int mat) {
    if (wid < 0 || wid >= P_.numWalls)
      throw std::invalid_argument("wall index out of range");
    auto h = Kokkos::create_mirror_view(P_.walls);
    Kokkos::deep_copy(h, P_.walls);
    h(wid).materialId = mat;
    Kokkos::deep_copy(P_.walls, h);
  }
  void addPlane(float px, float py, float pz, float nx, float ny, float nz) {
    auto h = Kokkos::create_mirror_view(P_.planes);
    Kokkos::deep_copy(h, P_.planes);
    if (P_.numPlanes < static_cast<int>(P_.planes.extent(0)))
      h(P_.numPlanes++) = PlaneP{F3{px, py, pz}, F3{nx, ny, nz}};
    Kokkos::deep_copy(P_.planes, h);
  }

  // Add a static, world-space SDF wall/container the grains collide against (a drum barrel, hopper,
  // vibrating tray). `grid` is a flat [nx*ny*nz] signed-distance field, x-fastest (idx = x + y*nx +
  // z*nx*ny), sampled at world nodes origin + (x,y,z)*spacing — POSITIVE in the void where the
  // grains live, NEGATIVE inside the solid wall (so a grain surface point reads the penetration
  // depth and the outward gradient is the push-out normal). restitution/friction are the binary
  // particle–wall material (independent of the body-body material). The wall is motionless but
  // carries a rigid-body surface-velocity field (set via setWallVelocity) so a grain touching it
  // feels the wall's motion. Returns the wall's index (for setWallVelocity). Add walls before
  // stepping.
  int addSdfWall(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                 float restitution, float friction) {
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("addSdfWall: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("addSdfWall: grid.size() must equal nx*ny*nz");

    WallSdf w{};
    w.grid.nx = nx;
    w.grid.ny = ny;
    w.grid.nz = nz;
    w.grid.offset = static_cast<int>(wallGridHost_.size());
    w.grid.origin = toCoreVec(origin);
    w.grid.invSpacing = peclet::core::Vec3<float>{spacing.x > 0 ? 1.0f / spacing.x : 0.0f,
                                                  spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
                                                  spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
    // A CONTAINER: beyond the stored box is wall-side, so the off-grid residual is SUBTRACTED.
    w.grid.extension = peclet::core::geom::GridExtension::kContainer;
    w.restitution = restitution;
    w.friction = friction;
    const int idx = static_cast<int>(wallsHost_.size());
    wallsHost_.push_back(w);
    wallGridHost_.insert(wallGridHost_.end(), grid.begin(), grid.end());
    // Upload the concatenated grid samples (only changes when a wall is added, not per step).
    P_.wallGrid = Kokkos::View<float*, CpMem>("wallGrid", wallGridHost_.size());
    auto hg = Kokkos::create_mirror_view(P_.wallGrid);
    for (size_t i = 0; i < wallGridHost_.size(); ++i)
      hg(i) = wallGridHost_[i];
    Kokkos::deep_copy(P_.wallGrid, hg);
    uploadWalls();
    P_.numWalls = static_cast<int>(wallsHost_.size());
    if (friction > P_.wallFrictionMax)
      P_.wallFrictionMax = friction;
    ensureContactCapacity();
    return idx;
  }

  // Set a wall's rigid-body surface-velocity field v(x) = linVel + angVel × (x − center). A grain
  // in contact feels this velocity even though the geometry never moves: set angVel for a rotating
  // drum (about `center` on the axis), or drive linVel sinusoidally each step for a vibrating wall.
  // Cheap (a few host scalars); safe to call every step.
  void setWallVelocity(int wallIndex, F3 linVel, F3 angVel, F3 center) {
    if (wallIndex < 0 || wallIndex >= static_cast<int>(wallsHost_.size()))
      throw std::runtime_error("setWallVelocity: wall index out of range");
    wallsHost_[wallIndex].linVel = linVel;
    wallsHost_[wallIndex].angVel = angVel;
    wallsHost_[wallIndex].center = center;
    uploadWalls();
  }

  // R1: RIGID-BODY PLACEMENT of an analytic wall -- rotate/translate the GEOMETRY, not just its
  // surface-velocity field. setWallVelocity alone is enough for an axisymmetric wall (a drum
  // barrel looks the same at every angle), but a stirrer blade has to actually move. The world
  // transform W is composed ONTO THE AUTHORED root transform (kept from add_analytic_wall), so
  // repeated calls place the wall absolutely and never compound; the KB-sized node pool is
  // re-uploaded, which is the whole cost.
  //
  // W is the placement of the authored frame in the world: a probe point p is evaluated as
  // eval_authored(toCanonical(W, p)). Drive it together with setWallVelocity (v = linVel +
  // angVel x (p - center)) so the surface velocity a grain feels matches the geometry it touches;
  // nothing here infers one from the other.
  void setWallTransform(int wallIndex, F3 translation, float qx, float qy, float qz, float qw) {
    namespace g = peclet::core::geom;
    if (wallIndex < 0 || wallIndex >= static_cast<int>(wallsHost_.size()))
      throw std::runtime_error("setWallTransform: wall index out of range");
    const int root = wallsHost_[(std::size_t)wallIndex].shapeRoot;
    if (root < 0)
      throw std::runtime_error(
          "setWallTransform: this wall is a sampled GRID SDF, which has no shape tree to place -- "
          "only analytic walls (add_analytic_wall) can be moved");
    const auto it = wallRootTf_.find(root);
    if (it == wallRootTf_.end())
      throw std::runtime_error("setWallTransform: authored root transform missing (internal)");
    g::Transform<float> W;
    W.translation = peclet::core::Vec3<float>{translation.x, translation.y, translation.z};
    W.rotation = peclet::core::Quat<float>{qx, qy, qz, qw};
    W.scale = 1.0f;
    wallNodesHost_[(std::size_t)root].transform =
        g::SceneBuilder<float>::composeTransform(W, it->second);
    auto hn = Kokkos::create_mirror_view(P_.wallNodes);
    for (std::size_t i = 0; i < wallNodesHost_.size(); ++i)
      hn(i) = wallNodesHost_[i];
    Kokkos::deep_copy(P_.wallNodes, hn);
  }

  // Diagnostic/authoring probe: the wall's own SDF at world points (flat [m*3]), sign included --
  // POSITIVE in the void where the grains live. This is exactly what the narrow phase reads
  // (sampleWallSdf), so it is the honest way to check a placement or draw a stirrer.
  std::vector<float> wallSdfAt(int wallIndex, const std::vector<float>& pts) const {
    if (wallIndex < 0 || wallIndex >= static_cast<int>(wallsHost_.size()))
      throw std::runtime_error("wall_sdf_at: wall index out of range");
    const int m = static_cast<int>(pts.size() / 3);
    std::vector<float> out((std::size_t)m, 0.0f);
    const WallSdf w = wallsHost_[(std::size_t)wallIndex];
    Kokkos::View<float*, CpMem> d("wallProbe", std::max(1, m));
    Kokkos::View<float*, CpMem> q("wallProbePts", std::max(1, 3 * m));
    auto hq = Kokkos::create_mirror_view(q);
    for (int i = 0; i < 3 * m; ++i)
      hq(i) = pts[(std::size_t)i];
    Kokkos::deep_copy(q, hq);
    auto grid = P_.wallGrid;
    Kokkos::parallel_for(
        "peclet::dem::wall_sdf_probe", Kokkos::RangePolicy<CpExec>(0, m), KOKKOS_LAMBDA(int i) {
          d(i) = sampleWallSdf(F3{q(3 * i), q(3 * i + 1), q(3 * i + 2)}, w, grid);
        });
    Kokkos::fence();
    auto hd = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(hd, d);
    for (int i = 0; i < m; ++i)
      out[(std::size_t)i] = hd(i);
    return out;
  }

  // positions: flat [n*3]; (re)sets the real-particle count and default state.
  void setPositions(const std::vector<float>& xyz) {
    const int n = static_cast<int>(xyz.size() / 3);
    if (n > P_.capacity)
      throw std::invalid_argument(
          "set_positions: " + std::to_string(n) +
          " particles exceed Simulation(capacity=" + std::to_string(P_.capacity) + ")");
    P_.numReal = n;
    P_.numParticles = n;
    P_.hertzNumPairs = -1;  // particle indices changed: invalidate the hertz pair cache
    P_.hertzPrevCount = 0;
    P_.prevPairCount = 0;  // stale persistent-pair ledger must not warm-start the new set
#ifdef PECLET_DEM_MPI
    mpiGidsGlobal_ = false;  // new particle set -> re-base the global ids at the next stepMpi
#endif
    auto pos = Kokkos::create_mirror_view(P_.pos);
    auto q = Kokkos::create_mirror_view(P_.quat);
    auto im = Kokkos::create_mirror_view(P_.invMass);
    auto sc = Kokkos::create_mirror_view(P_.scale);
    auto ii = Kokkos::create_mirror_view(P_.invInertia);
    auto sid = Kokkos::create_mirror_view(P_.shapeId);
    auto vel = Kokkos::create_mirror_view(P_.vel);
    auto av = Kokkos::create_mirror_view(P_.angVel);
    auto gi = Kokkos::create_mirror_view(P_.gid);
    for (int i = 0; i < n; ++i) {
      gi(i) = i;  // identity global id; the MPI enable re-bases it to a global Exscan offset
      pos(i, 0) = xyz[3 * i];
      pos(i, 1) = xyz[3 * i + 1];
      pos(i, 2) = xyz[3 * i + 2];
      q(i, 0) = 0;
      q(i, 1) = 0;
      q(i, 2) = 0;
      q(i, 3) = 1;
      im(i) = 1.0f;
      sc(i) = 1.0f;
      sid(i) = 0;
      ii(i, 0) = defaultInvI_.x;
      ii(i, 1) = defaultInvI_.y;
      ii(i, 2) = defaultInvI_.z;
      vel(i, 0) = vel(i, 1) = vel(i, 2) = 0;
      av(i, 0) = av(i, 1) = av(i, 2) = 0;
    }
    Kokkos::deep_copy(P_.pos, pos);
    Kokkos::deep_copy(P_.quat, q);
    Kokkos::deep_copy(P_.invMass, im);
    Kokkos::deep_copy(P_.scale, sc);
    Kokkos::deep_copy(P_.invInertia, ii);
    Kokkos::deep_copy(P_.shapeId, sid);
    Kokkos::deep_copy(P_.vel, vel);
    Kokkos::deep_copy(P_.angVel, av);
    Kokkos::deep_copy(P_.gid, gi);
    Kokkos::deep_copy(P_.targetScale, P_.scale);  // unscaled growth target = the set scale
  }
  void setScalesUniform(float s) {
    auto sc = Kokkos::create_mirror_view(P_.scale);
    for (int i = 0; i < P_.numReal; ++i)
      sc(i) = s;
    Kokkos::deep_copy(P_.scale, sc);
    Kokkos::deep_copy(P_.targetScale, P_.scale);
  }
  // per-particle scales (growth target): flat [n]. scale starts at target (growth factor applies in
  // step).
  void setScales(const std::vector<float>& s) {
    auto tsc = Kokkos::create_mirror_view(P_.targetScale);
    for (int i = 0; i < P_.numReal && i < (int)s.size(); ++i)
      tsc(i) = s[i];
    Kokkos::deep_copy(P_.targetScale, tsc);
    Kokkos::deep_copy(P_.scale, P_.targetScale);
  }
  void setVelocities(const std::vector<float>& v) {
    auto vel = Kokkos::create_mirror_view(P_.vel);
    for (int i = 0; i < P_.numReal && 3 * i + 2 < (int)v.size(); ++i) {
      vel(i, 0) = v[3 * i];
      vel(i, 1) = v[3 * i + 1];
      vel(i, 2) = v[3 * i + 2];
    }
    Kokkos::deep_copy(P_.vel, vel);
  }
  // Per-particle external FORCE (fluid drag etc.), an (N,3) flat array. Applied in the next
  // step()'s velocity predict as dv = F*invMass*dt. Persists across steps until re-set or cleared.
  void setExternalForces(const std::vector<float>& f) {
    auto ef = Kokkos::create_mirror_view(P_.extForce);
    for (int i = 0; i < P_.numReal && 3 * i + 2 < (int)f.size(); ++i) {
      ef(i, 0) = f[3 * i];
      ef(i, 1) = f[3 * i + 1];
      ef(i, 2) = f[3 * i + 2];
    }
    Kokkos::deep_copy(P_.extForce, ef);
  }
  void clearExternalForces() { Kokkos::deep_copy(P_.extForce, 0.0f); }
  // Per-particle external TORQUE in the WORLD frame, an (N,3) flat array. Applied in the next
  // step()'s angular predict as the body-frame Euler update dw = invI*(tau_body - w x I w)*dt.
  // Persists across steps until re-set or cleared. Only bodies with a finite inertia (invInertia
  // > 0, i.e. a registered non-degenerate shape) respond -- a torque on a point mass is silently
  // inert, exactly as the gyroscopic term is.
  void setExternalTorques(const std::vector<float>& t) {
    auto et = Kokkos::create_mirror_view(P_.extTorque);
    for (int i = 0; i < P_.numReal && 3 * i + 2 < (int)t.size(); ++i) {
      et(i, 0) = t[3 * i];
      et(i, 1) = t[3 * i + 1];
      et(i, 2) = t[3 * i + 2];
    }
    Kokkos::deep_copy(P_.extTorque, et);
    P_.extTorqueActive = true;
  }
  void clearExternalTorques() {
    Kokkos::deep_copy(P_.extTorque, 0.0f);
    P_.extTorqueActive = false;
  }
  const V3& externalTorquesView() const { return P_.extTorque; }
  const V3& externalForcesView() const { return P_.extForce; }
  const Vf& invMassView() const { return P_.invMass; }
  // rigid-body rotation state (the pipeline integrates the gyroscopic Euler term + quaternion
  // already)
  void setQuaternions(const std::vector<float>& q) {
    auto h = Kokkos::create_mirror_view(P_.quat);
    for (int i = 0; i < P_.numReal && 4 * i + 3 < (int)q.size(); ++i) {
      h(i, 0) = q[4 * i];
      h(i, 1) = q[4 * i + 1];
      h(i, 2) = q[4 * i + 2];
      h(i, 3) = q[4 * i + 3];
    }
    Kokkos::deep_copy(P_.quat, h);
  }
  void setAngularVelocities(const std::vector<float>& w) {
    auto h = Kokkos::create_mirror_view(P_.angVel);
    for (int i = 0; i < P_.numReal && 3 * i + 2 < (int)w.size(); ++i) {
      h(i, 0) = w[3 * i];
      h(i, 1) = w[3 * i + 1];
      h(i, 2) = w[3 * i + 2];
    }
    Kokkos::deep_copy(P_.angVel, h);
  }
  void setInvInertia(const std::vector<float>& ii) {
    auto h = Kokkos::create_mirror_view(P_.invInertia);
    for (int i = 0; i < P_.numReal && 3 * i + 2 < (int)ii.size(); ++i) {
      h(i, 0) = ii[3 * i];
      h(i, 1) = ii[3 * i + 1];
      h(i, 2) = ii[3 * i + 2];
    }
    Kokkos::deep_copy(P_.invInertia, h);
  }
  void setInvMass(const std::vector<float>& im) {
    auto h = Kokkos::create_mirror_view(P_.invMass);
    for (int i = 0; i < P_.numReal && i < (int)im.size(); ++i)
      h(i) = im[i];
    Kokkos::deep_copy(P_.invMass, h);
  }
  std::vector<float> getAngularVelocities() const {
    return peclet::core::toVector(
        Kokkos::subview(P_.angVel, Kokkos::make_pair(0, P_.numReal), Kokkos::ALL));
  }
  std::vector<float> getInvInertia() const {
    return peclet::core::toVector(
        Kokkos::subview(P_.invInertia, Kokkos::make_pair(0, P_.numReal), Kokkos::ALL));
  }
  // growth: factor *= exp(rate*dt) per step (capped at 1); new_factor<0 keeps/initialises (0.01 if
  // inactive).
  void setGrowthParams(float rate, float new_factor) {
    if (P_.growthFactor == -1.0f)
      P_.growthFactor = (new_factor > 0.0f) ? new_factor : 0.01f;
    else if (new_factor > 0.0f)
      P_.growthFactor = new_factor;
    P_.growthRate = rate;
    if (P_.growthFactor > 0.0f)
      updateGrowthScalesKokkos(P_.numReal, P_.scale, P_.targetScale, P_.growthFactor);
  }
  float growthFactor() const { return P_.growthFactor; }
  float getGrowthRate() const { return P_.growthRate; }
  // per-particle mass = 1/invMass (0 for fixed/infinite-mass particles), CUDA
  // Simulation::get_masses.
  std::vector<float> getMasses() const {
    auto im = Kokkos::create_mirror_view(P_.invMass);
    Kokkos::deep_copy(im, P_.invMass);
    std::vector<float> out(P_.numReal);
    for (int i = 0; i < P_.numReal; ++i)
      out[i] = (im(i) > 0.0f) ? (1.0f / im(i)) : 0.0f;
    return out;
  }

  std::vector<float> getPositions() const {
    return peclet::core::toVector(
        Kokkos::subview(P_.pos, Kokkos::make_pair(0, P_.numReal), Kokkos::ALL));
  }
  std::vector<float> getVelocities() const {
    return peclet::core::toVector(
        Kokkos::subview(P_.vel, Kokkos::make_pair(0, P_.numReal), Kokkos::ALL));
  }
  std::vector<float> getQuaternions() const {
    return peclet::core::toVector(
        Kokkos::subview(P_.quat, Kokkos::make_pair(0, P_.numReal), Kokkos::ALL));
  }
  std::vector<float> getScales() const {
    return peclet::core::toVector(Kokkos::subview(P_.scale, Kokkos::make_pair(0, P_.numReal)));
  }

  /// Advance `n` XPBD substeps of the stored dt (set_dt first; suite/docs/NAMING.md 1.5).
  void step(int n = 1) {
    requireDt("step");
    for (int i = 0; i < n; ++i)
      demStep(P_);
  }
  /// `n` dynamics-free RELAXATION substeps (overlap removal only, no gravity / velocity update):
  /// the growth-packing protocol's "settle" move. Runs the pipeline with dt = 0 and restores the
  /// stored dt afterwards; does not require set_dt.
  void relax(int n = 1) {
    const float dt = P_.dt;
    P_.dt = 0.0f;
    for (int i = 0; i < n; ++i)
      demStep(P_);
    P_.dt = dt;
  }

  // Max pair interpenetration on the current committed state (CUDA Simulation::compute_overlaps).
  float computeOverlaps() { return computeOverlapsKokkos(P_); }

  // LAMMPS "dump custom" of the current committed state (CUDA Simulation::export_lammps). Radius =
  // scale*globalScale*baseRadius; bounds computed from the particle AABBs.
  void exportLammps(const std::string& filename, int step) const {
    const std::vector<float> pos = getPositions(), vel = getVelocities(), quat = getQuaternions();
    auto sc = Kokkos::create_mirror_view(P_.scale);
    Kokkos::deep_copy(sc, P_.scale);
    std::vector<float> radii(P_.numReal);
    for (int i = 0; i < P_.numReal; ++i)
      radii[i] = sc(i) * P_.globalScale * baseRadius_;
    const bool pbc = P_.domain.periodic_x || P_.domain.periodic_y || P_.domain.periodic_z;
    peclet::dem::writeLammpsDump(filename, step, pos, vel, quat, radii, nullptr, nullptr, pbc);
  }

  // SDF field over the domain -> ImageData VTI (CUDA Simulation::export_sdf).
  void exportSdf(const std::string& filename, int rx, int ry, int rz) {
    const std::vector<float> grid = getSdfGrid(rx, ry, rz);
    const float mn[3] = {P_.domain.min.x, P_.domain.min.y, P_.domain.min.z};
    const float mx[3] = {P_.domain.max.x, P_.domain.max.y, P_.domain.max.z};
    peclet::dem::writeSdfVti(filename, grid, rx, ry, rz, mn, mx);
  }

#ifdef PECLET_DEM_MPI
  // Block decomposition over the GLOBAL domain (once); the per-block solver stays non-periodic, the
  // halo supplies the periodic wrap. gsize is the ORB cell grid. Mirror of Simulation::mpi_init.
  void initMpi(std::tuple<double, double, double> origin, std::tuple<double, double, double> size,
               std::tuple<long, long, long> gsize, std::tuple<bool, bool, bool> periodic,
               MPI_Comm comm) {
    halo_->initMpi({std::get<0>(origin), std::get<1>(origin), std::get<2>(origin)},
                   {std::get<0>(size), std::get<1>(size), std::get<2>(size)},
                   {std::get<0>(gsize), std::get<1>(gsize), std::get<2>(gsize)},
                   {std::get<0>(periodic), std::get<1>(periodic), std::get<2>(periodic)}, comm);
  }
  // Enable the distributed step. rcut is the ghost-band width (default = 1.0*globalScale, the
  // periodic skin used by the single-GPU path); sync_every is the owner->ghost refresh interval (1
  // = EXACT). rebalance_every: re-decompose by particle count + migrate ownership every N
  // distributed steps to keep the per-rank load even as a packing densifies (0 = never; the
  // partition is then fixed at the initial decomposition, as before). A pure redistribution — the
  // physics result is unchanged.
  void enableMpiStep(double rcut, int sync_every = 1, bool forward_rotation = true,
                     int rebalance_every = 0, double verlet_skin = 0.0) {
    mpiRcut_ = rcut;
    mpiSyncEvery_ = sync_every < 1 ? 1 : sync_every;
    mpiForwardRotation_ = forward_rotation;
    mpiRebalanceEvery_ = rebalance_every < 0 ? 0 : rebalance_every;
    // Verlet-skin ghost reuse (D2): rebuild the halo topology only when a particle has moved >
    // skin, instead of every substep. 0 (default) keeps the exact per-substep rebuild.
    halo_->setVerletSkin(static_cast<float>(verlet_skin));
  }
  // Halo rebuild stats (D2): topology rebuilds vs total gather() calls (for benchmarking).
  long mpiRebuilds() const { return halo_->numRebuilds(); }
  long mpiGathers() const { return halo_->numGathers(); }
  // Migrate ownership now so each rank holds a near-equal particle count. Safe to call at a step
  // boundary; returns this rank's new owned count. Exposed for manual / adaptive balancing.
  int rebalance() { return halo_->rebalance(P_); }
  // Co-rebalance: migrate ownership onto the weighted ORB of per-cell weights `w` (the SAME
  // partition the coupled flow solver redistributes onto from the same weight field). Returns new
  // owned count.
  int migrateToWeights(const std::vector<peclet::core::Real>& w) {
    return halo_->migrateToWeights(P_, w);
  }
  // Globally-unique particle ids (persistent-pair and Mindlin-history keys are gid-based):
  // re-base each rank's identity ids by an exclusive scan of the owned counts, once per particle
  // set. Migration and rebalance carry gids, so the ids stay stable afterwards; set_positions
  // resets the flag. Any pre-MPI ledger was keyed with the identity ids, which the re-base makes
  // stale — cold-start both engines' histories (one soft restart, negligible).
  void ensureGlobalGids() {
    if (mpiGidsGlobal_)
      return;
    long base = 0, mine = P_.numReal;
    MPI_Exscan(&mine, &base, 1, MPI_LONG, MPI_SUM, halo_->comm());
    if (halo_->rank() == 0)
      base = 0;  // MPI_Exscan leaves rank 0's recvbuf undefined
    fillGidBaseKokkos(P_.gid, P_.numReal, static_cast<int>(base));
    P_.prevPairCount = 0;
    P_.hertzPrevCount = 0;
    P_.hertzNumPairs = -1;
    mpiGidsGlobal_ = true;
  }
  void stepMpi(int nsteps) {
    requireDt("step_mpi");
    const double rcut = (mpiRcut_ > 0.0) ? mpiRcut_ : maxOwnedRadius(P_);
    ensureGlobalGids();
    for (int s = 0; s < nsteps; ++s) {
      if (mpiRebalanceEvery_ > 0 && mpiStepCount_ % mpiRebalanceEvery_ == 0)
        halo_->rebalance(P_);
      demStepMpi(P_, *halo_, rcut, mpiSyncEvery_, mpiForwardRotation_);
      ++mpiStepCount_;
    }
  }
  /// Advance `substeps` distributed explicit Hertz–Mindlin (force-based) steps of the stored dt —
  /// the MPI counterpart of step_hertz, on the same halo/decomposition as step_mpi (init_mpi +
  /// enable_mpi_step first). rebalance_every counts CALLS of this method (each call = one
  /// Rayleigh-limited inner batch); the migration carries the Mindlin pair/wall history.
  void stepHertzMpi(int substeps, float skin_frac) {
    requireDt("step_hertz_mpi");
    ensureGlobalGids();
    if (mpiRebalanceEvery_ > 0 && mpiHertzCalls_ % mpiRebalanceEvery_ == 0)
      halo_->rebalance(P_);
    ++mpiHertzCalls_;
    demStepHertzMpi(P_, *halo_, P_.dt, substeps, skin_frac);
  }
  int rank() const { return halo_->rank(); }
  int numGhost() const { return halo_->numGhost(); }
#endif  // PECLET_DEM_MPI

  // SDF grid (get_sdf_grid): Eikonal reconstruction over the domain, flat x-fastest, negative
  // inside solid.
  std::vector<float> getSdfGrid(int rx, int ry, int rz) {
    return peclet::dem::generateSdfKokkos(rx, ry, rz, P_.domain.min, P_.domain.max, P_.numReal,
                                          P_.pos, P_.quat, P_.scale, P_.shapeId, P_.shapes,
                                          P_.domain.periodic_x, P_.domain.periodic_y,
                                          P_.domain.periodic_z, P_.sdfGrid, P_.globalScale);
  }

  int numParticles() const { return P_.numReal; }
  // Live device Views of the owned particle state, for the zero-copy device-array export (H2): the
  // binding wraps a [0,numReal) subview as a DLPack/__cuda_array_interface__ array (or a NumPy view
  // on a host backend) referencing this memory — no device->host copy.
  const V3& positionsView() const { return P_.pos; }
  const V3& velocitiesView() const { return P_.vel; }
  int numContacts() { return readInt(P_.contactCount); }
  int numManifolds() { return readInt(P_.manifoldCount); }
  // TEST-ONLY colouring self-check: over the LAST solved substep's colourings, count how many
  // (manifold, contact) pairs violate the graph-colouring invariant "no two same-colour items share
  // a body" — must be exactly (0, 0) for a valid colouring (the incremental warm-start path must
  // never import a conflict). Returns {velocity-colour conflicts, position-colour conflicts}.
  std::pair<int, int> debugColoringConflicts() {
    using peclet::dem::CpExec;
    using peclet::dem::CpMem;
    CpExec space;
    const int nm = readInt(P_.manifoldCount);
    const int nc = readInt(P_.contactCount);
    const int nb = std::max(P_.numParticles, P_.numReal) + 1;
    Kokkos::View<std::uint64_t*, CpMem> seen("dbg_color_seen", nb);
    int velConf = 0, posConf = 0;
    if (nm > 0) {
      Kokkos::deep_copy(space, seen, std::uint64_t(0));
      auto manifolds = P_.manifolds;
      auto realIdx = P_.realIndices;
      auto mColor = P_.manifoldColor;
      Kokkos::parallel_reduce(
          "peclet::dem::dbg_vel_color", Kokkos::RangePolicy<CpExec>(space, 0, nm),
          KOKKOS_LAMBDA(int idx, int& acc) {
            const int c = mColor(idx);
            if (c < 0)
              return;
            const auto m = manifolds(idx);
            const std::uint64_t bit = std::uint64_t(1) << c;
            if ((Kokkos::atomic_fetch_or(&seen(realIdx(m.bodyA)), bit) >> c) & 1)
              acc += 1;
            if (m.bodyB >= 0 && ((Kokkos::atomic_fetch_or(&seen(realIdx(m.bodyB)), bit) >> c) & 1))
              acc += 1;
          },
          velConf);
    }
    if (nc > 0) {
      Kokkos::deep_copy(space, seen, std::uint64_t(0));
      auto contacts = P_.contacts;
      auto cColor = P_.contactColor;
      Kokkos::parallel_reduce(
          "peclet::dem::dbg_pos_color", Kokkos::RangePolicy<CpExec>(space, 0, nc),
          KOKKOS_LAMBDA(int idx, int& acc) {
            const int c = cColor(idx);
            if (c < 0)
              return;
            const auto ct = contacts(idx);
            const std::uint64_t bit = std::uint64_t(1) << c;
            if ((Kokkos::atomic_fetch_or(&seen(ct.bodyA), bit) >> c) & 1)
              acc += 1;
            if (ct.bodyB >= 0 && ((Kokkos::atomic_fetch_or(&seen(ct.bodyB), bit) >> c) & 1))
              acc += 1;
          },
          posConf);
    }
    space.fence();
    return {velConf, posConf};
  }
  float maxOverlap() {
    float h;
    Kokkos::deep_copy(h, P_.maxOverlap);
    return h;
  }
  /// Poisson-restitution diagnostics: (sum, max, count>0) of the committed per-pair owed
  /// separation impulse (prevRestBank[0:prevPairCount], physical units).
  std::tuple<double, float, int> restBankStats() {
    return restBankStatsKokkos(P_.prevRestBank, P_.prevPairCount);
  }
  /// Orphan-account diagnostics: (sum, max, count>0) of the per-body orphaned budget.
  std::tuple<double, float, int> restOrphanStats() {
    return restBankStatsKokkos(P_.bodyOrphan, P_.numReal);
  }

  // ParaView PolyData (points + Radius + Velocity), faithful to CUDA Simulation::write_vtp:
  // Radius = scale * globalScale * baseRadius.
  void writeVtp(const std::string& filename) const {
    auto pos = Kokkos::create_mirror_view(P_.pos);
    Kokkos::deep_copy(pos, P_.pos);
    auto sc = Kokkos::create_mirror_view(P_.scale);
    Kokkos::deep_copy(sc, P_.scale);
    auto vel = Kokkos::create_mirror_view(P_.vel);
    Kokkos::deep_copy(vel, P_.vel);
    const int n = P_.numReal;

    std::ofstream out(filename);
    if (!out)
      throw std::runtime_error("Could not open file for writing: " + filename);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << n << "\" NumberOfVerts=\"0\" "
        << "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Position\" NumberOfComponents=\"3\" "
           "format=\"ascii\">\n";
    for (int i = 0; i < n; ++i)
      out << pos(i, 0) << " " << pos(i, 1) << " " << pos(i, 2) << " ";
    out << "\n        </DataArray>\n";
    out << "      </Points>\n";
    out << "      <PointData Scalars=\"Radius\">\n";
    out << "        <DataArray type=\"Float32\" Name=\"Radius\" NumberOfComponents=\"1\" "
           "format=\"ascii\">\n";
    for (int i = 0; i < n; ++i)
      out << sc(i) * P_.globalScale * baseRadius_ << " ";
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" "
           "format=\"ascii\">\n";
    for (int i = 0; i < n; ++i)
      out << vel(i, 0) << " " << vel(i, 1) << " " << vel(i, 2) << " ";
    out << "\n        </DataArray>\n";
    out << "      </PointData>\n";
    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
    out.close();
    std::printf("Exported VTP: %s\n", filename.c_str());
  }

 private:
  bool dtSet_ = false;  // set_dt was called: every stepper checks it (no silent default)
  void requireDt(const char* who) const {
    if (!dtSet_)
      throw std::runtime_error(std::string(who) +
                               ": call set_dt(dt) first (there is no default time step)");
  }
  // (Re)upload just the small WallSdf array (velocity fields change every step for a vibrating
  // wall; the grid samples are uploaded once in addSdfWall).
  void uploadWalls() {
    const int n = std::max<int>(1, static_cast<int>(wallsHost_.size()));
    if (static_cast<int>(P_.walls.extent(0)) < n)
      P_.walls = Kokkos::View<WallSdf*, CpMem>("walls", n);
    auto h = Kokkos::create_mirror_view(P_.walls);
    for (size_t i = 0; i < wallsHost_.size(); ++i)
      h(i) = wallsHost_[i];
    Kokkos::deep_copy(P_.walls, h);
  }

  std::vector<WallSdf> wallsHost_;
  std::vector<float> wallGridHost_;
  std::vector<peclet::core::geom::ShapeNode<float>> wallNodesHost_;
  // R1: authored root transform per analytic-wall root node (see setWallTransform).
  std::map<int, peclet::core::geom::Transform<float>> wallRootTf_;
#ifdef PECLET_DEM_MPI
  std::unique_ptr<ParticleHalo> halo_ = std::make_unique<ParticleHalo>();
  double mpiRcut_ = 0.0;
  int mpiSyncEvery_ = 1;
  bool mpiForwardRotation_ = true;
  int mpiRebalanceEvery_ = 0;
  long mpiStepCount_ = 0;
  long mpiHertzCalls_ =
      0;  // step_hertz_mpi call count (rebalance_every cadence for the force path)
  bool mpiGidsGlobal_ = false;  // gids re-based to a global Exscan offset (once per particle set)
#endif
};

}  // namespace peclet::dem

#endif  // DEM_SIM_HPP
