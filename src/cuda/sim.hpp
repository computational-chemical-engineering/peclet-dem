// packing-gpu — portable (Kokkos) Simulation facade: the demgpu flip's host-facing driver.
//
// Owns a dem::Particles SoA and runs the full XPBD DEM step by composing the ported kernels in the
// simulation.cpp step() order. Exposes a small std::vector-based API (binding-agnostic) so a pybind
// module can drive it from Python (set/get arrays, step). Sphere shapes + analytic planes for now.
#ifndef DEM_SIM_HPP
#define DEM_SIM_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "broadphase_arborx.hpp"
#include "contact_preprocessing.hpp"
#include "integration.hpp"
#include "io.hpp"
#include "narrowphase.hpp"
#include "output_sdf.hpp"
#include "particles.hpp"
#include "periodicity.hpp"
#include "shapes_portable.hpp"
#include "solver_friction.hpp"
#include "solver_position.hpp"
#include "solver_velocity.hpp"

#ifdef DEMGPU_MPI
#include "mpi_halo.hpp"  // KokkosParticleHalo (gated; default module never includes it)
#endif

namespace dem {

inline int readInt(Kokkos::View<int, CpMem> v) { int h; Kokkos::deep_copy(h, v); return h; }

/// One full XPBD DEM substep over the particle SoA (mirrors simulation.cpp Simulation::step()).
inline void demStep(Particles& P) {
  CpExec space;
  const float margin = 0.1f * P.globalScale;

  // growth ramp (faithful to CUDA Simulation::step): factor *= exp(rate*dt), capped at 1, then
  // scale = targetScale * factor. (CUDA stores the unscaled target in d_target_scales.)
  if (P.growthFactor != -1.0f && P.growthRate != 0.0f) {
    P.growthFactor *= std::exp(P.growthRate * P.dt);
    if (P.growthFactor > 1.0f) P.growthFactor = 1.0f;
  }
  if (P.growthFactor > 0.0f) updateGrowthScalesKokkos(P.numReal, P.scale, P.targetScale, P.growthFactor);

  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt);

  { auto ri = P.realIndices;
    Kokkos::parallel_for("self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
                         KOKKOS_LAMBDA(int i) { ri(i) = i; }); }
  Kokkos::deep_copy(space, P.topGhost, P.numReal);
  // periodic ghost band = max radius + margin (CUDA config.skin_width = 1.0*global_scale), NOT the small
  // Verlet broadphase margin -- else particles farther than the Verlet skin from a periodic face are not
  // ghosted and cross-boundary contacts are missed.
  const float ghostBand = 1.0f * P.globalScale;
  generateGhostsKokkos(P.numReal, P.capacity, P.domain, ghostBand, P.pos, P.invMass, P.posPred, P.vel,
                       P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale, P.shapeId,
                       P.realIndices, P.topGhost);
  P.numParticles = readInt(P.topGhost);

  { auto sc = P.scale; auto rad = P.rad; float gs = P.globalScale;
    Kokkos::parallel_for("rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
                         KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs; }); }
  // Collision detection runs on the PREDICTED state (speculative positions/orientations), matching
  // the CUDA solver — the position solve then corrects posPred against these contacts.
  findCollisionsArborX(P.posPred, P.crad(), P.numParticles, P.numReal, margin, P.pairs, P.pairCount);
  const int np = readInt(P.pairCount);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                       P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  const int nc = readInt(P.contactCount);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount);
  const int nm = readInt(P.manifoldCount);

  const bool friction = (P.frictionDynamic > 0.0f);
  if (friction)
    computePlaneLoadKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred, P.planeFriction);

  for (int it = 0; it < P.velocityIterations; ++it) {
    if (friction)
      accumulateNormalImpulseKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                                    P.realIndices, P.growthRate);
    solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                        P.realIndices, P.growthRate, P.restitutionNormal, P.deltaVel, P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
  }
  if (friction) {
    countFrictionContactsKokkos(P.contacts, nc, P.realIndices, P.planeFriction);
    solveContactFrictionKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                               P.realIndices, P.planeFriction, P.frictionDynamic, P.deltaVel, P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
  }

  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);

  for (int it = 0; it < P.positionIterations; ++it) {
    solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                        P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
    applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel, P.constraintCounts);
  }

  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);

  // Berendsen thermostat at the end of the step (CUDA Simulation::step), tau>0 enables.
  if (P.thermostatTau > 0.0f && P.dt > 0.0f)
    applyThermostatKokkos(P.numReal, P.vel, P.invMass, P.angVel, P.invInertia, P.quat,
                          P.thermostatKB, P.thermostatTau, P.thermostatTemp, P.dt);
}

/// Max pair interpenetration on the *committed* state (faithful to CUDA Simulation::compute_overlaps):
/// copy committed pos/quat into the predicted buffers, regenerate the periodic ghosts from that state, then
/// run the same broad/narrow phase as demStep and return the recorded max overlap. No solve.
inline float computeOverlapsKokkos(Particles& P) {
  CpExec space;
  const float margin = 0.1f * P.globalScale;
  P.numParticles = P.numReal;
  Kokkos::deep_copy(P.posPred, P.pos);
  Kokkos::deep_copy(P.quatPred, P.quat);
  { auto ri = P.realIndices;
    Kokkos::parallel_for("self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
                         KOKKOS_LAMBDA(int i) { ri(i) = i; }); }
  Kokkos::deep_copy(space, P.topGhost, P.numReal);
  const float ghostBand = 1.0f * P.globalScale;
  generateGhostsKokkos(P.numReal, P.capacity, P.domain, ghostBand, P.pos, P.invMass, P.posPred, P.vel,
                       P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale, P.shapeId,
                       P.realIndices, P.topGhost);
  P.numParticles = readInt(P.topGhost);
  { auto sc = P.scale; auto rad = P.rad; float gs = P.globalScale;
    Kokkos::parallel_for("rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
                         KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs; }); }
  findCollisionsArborX(P.posPred, P.crad(), P.numParticles, P.numReal, margin, P.pairs, P.pairCount);
  const int np = readInt(P.pairCount);
  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                       P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  P.numParticles = P.numReal;
  float h; Kokkos::deep_copy(h, P.maxOverlap); return h;
}

#ifdef DEMGPU_MPI
/// One distributed XPBD DEM substep (faithful port of Simulation::step_mpi). Identical to demStep
/// EXCEPT: the periodic ghost generation is replaced by a cross-rank gather (halo.gather, ghosts
/// carrying REAL mass), and the owners refresh their ghost copies (velPred/angVelPred, then
/// posPred/quatPred) every `syncEvery` solver iterations (and the last). Each owned particle thus
/// sees all its neighbours -- owned or ghost -- and computes its full serial delta locally; the
/// ghost deltas land on self-mapped slots and are discarded. Mirrors the CUDA distributed scheme,
/// which (like step_mpi) carries NO body-body friction passes -- the velocity solve is pure normal
/// restitution. `forwardRotation`=false (spheres) skips the angular/quaternion forwards.
///
/// PERIODICITY: cross-rank ghosts supply the wrap on DECOMPOSED axes; LOCAL periodic self-ghosts
/// (KokkosParticleHalo build with includePeriodicSelf) supply it on UNDECOMPOSED periodic axes (a "x1"
/// ORB axis, e.g. z of a 2x2x1 layout, or np=1). Correct for any layout, including np=1 fully periodic
/// (it matches the single-GPU demStep to ~roundoff). CAPACITY: a periodic box needs a thick ghost
/// boundary layer -- a fully periodic box at this rcut needs ~no + (boundary layer) ghost slots, well
/// above the no*2 the closed case wants -- so size the Simulation capacity for the worst-case ghost
/// band; gather() throws on overflow rather than corrupting the SoA.
inline void demStepMpi(Particles& P, KokkosParticleHalo& halo, double rcut, int syncEvery,
                       bool forwardRotation) {
  CpExec space;
  const float margin = 0.1f * P.globalScale;

  if (P.growthFactor != -1.0f && P.growthRate != 0.0f) {
    P.growthFactor *= std::exp(P.growthRate * P.dt);
    if (P.growthFactor > 1.0f) P.growthFactor = 1.0f;
  }
  if (P.growthFactor > 0.0f) updateGrowthScalesKokkos(P.numReal, P.scale, P.targetScale, P.growthFactor);

  // 1. Predict velocity on the owned set (no ghosts yet -> numParticles == numReal).
  P.numParticles = P.numReal;
  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt);

  // 2. Gather ghosts (real mass) from owners over the halo: full state into the ghost slots; sets
  //    P.numParticles = numReal + numGhost and self-maps realIndices.
  halo.gather(P, rcut);

  { auto sc = P.scale; auto rad = P.rad; float gs = P.globalScale;
    Kokkos::parallel_for("rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
                         KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs; }); }

  // 3. Broad/narrow phase + manifold reduction over owned + ghosts.
  findCollisionsArborX(P.posPred, P.crad(), P.numParticles, P.numReal, margin, P.pairs, P.pairCount);
  const int np = readInt(P.pairCount);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                       P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  const int nc = readInt(P.contactCount);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount);
  const int nm = readInt(P.manifoldCount);

  // 4. Velocity solve (normal restitution); refresh ghost velocities every syncEvery iters (+ last).
  for (int it = 0; it < P.velocityIterations; ++it) {
    solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                        P.realIndices, P.growthRate, P.restitutionNormal, P.deltaVel, P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
    if ((it + 1) % syncEvery == 0 || it == P.velocityIterations - 1) {
      halo.forward(P.velPred);
      if (forwardRotation) halo.forward(P.angVelPred);
    }
  }

  // 5. Apply velocity & predict position, then refresh ghost predicted positions (+ pose if rotating).
  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);
  halo.forwardPositions(P.posPred);
  if (forwardRotation) halo.forward4(P.quatPred);

  // 6. Position solve (Projected Jacobi); refresh ghost predicted pose every syncEvery iters (+ last).
  for (int it = 0; it < P.positionIterations; ++it) {
    solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                        P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
    applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel, P.constraintCounts);
    if ((it + 1) % syncEvery == 0 || it == P.positionIterations - 1) {
      halo.forwardPositions(P.posPred);
      if (forwardRotation) halo.forward4(P.quatPred);
    }
  }

  // 7. Commit (owned results kept; ghosts discarded, re-gathered next substep).
  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);

  if (P.thermostatTau > 0.0f && P.dt > 0.0f)
    applyThermostatKokkos(P.numReal, P.vel, P.invMass, P.angVel, P.invInertia, P.quat,
                          P.thermostatKB, P.thermostatTau, P.thermostatTemp, P.dt);

  P.numParticles = P.numReal;  // restore owned-only active count for getters
}
#endif  // DEMGPU_MPI

/// Host-facing facade with std::vector setters/getters (binding-agnostic).
class KokkosSim {
 public:
  explicit KokkosSim(int capacity) {
    registry().push_back(this);
    P_.allocate(capacity, capacity * 64, capacity * 16, /*shapes*/ 1, /*shell*/ 1, /*planes*/ 8);
    // default sphere shape (radius 1) + identity-ish defaults
    setSphereShape(1.0f);
  }
  ~KokkosSim() { auto& r = registry(); r.erase(std::remove(r.begin(), r.end(), this), r.end()); }

  // Teardown safety: the Particles SoA holds Kokkos Views, so they MUST be freed before Kokkos::finalize
  // (else "deallocated after finalize" aborts). releaseAll() (called from the module's atexit, before
  // finalize) frees every live Sim's Views, so callers need not `del sim; gc.collect()` themselves.
  void releaseViews() { P_ = Particles{}; }
  static void releaseAll() { for (auto* s : registry()) s->releaseViews(); }
  static std::vector<KokkosSim*>& registry() { static std::vector<KokkosSim*> r; return r; }

  void setSphereShape(float radius) {
    initializeShape(SPHERE, radius, 0.0f, 0.0f);
  }

  // Mirror of CUDA Simulation::initialize(shape_type, radius, height, thickness): builds shape 0's
  // descriptor + surface point shell (cylinder/box) and records the per-shape base radius and
  // (uniform-mass=1) inverse inertia applied to every particle by setPositions. shape_type uses the
  // dem::ShapeKind values (SPHERE=1, HOLLOW_CYLINDER=2, BOX=3).
  void initializeShape(int shape_type, float radius, float height, float thickness) {
    baseRadius_ = radius;
    F4 params{radius, 0, 0, 0};
    std::vector<F3> shell;

    if (shape_type == HOLLOW_CYLINDER) {
      params = F4{radius, height, thickness, 0};
      // Dynamic spacing (faithful to CUDA): >=4 pts across thickness, >=20 around circumference.
      float min_dim = std::min(radius, thickness);
      if (min_dim < 1e-4f) min_dim = radius;  // safety if thickness 0
      float spacing = std::min(radius * 0.3f, min_dim * 0.5f);
      if (spacing < 1e-3f) spacing = 1e-3f;
      shell = genCylinderShell(radius, height, thickness, spacing);
    } else if (shape_type == BOX) {
      // Cube with half-extent = radius (side = 2*radius).
      params = F4{radius, radius, radius, 0};
      float spacing = std::max(radius * 0.5f, 1e-3f);
      shell = genBoxShell(radius, radius, radius, spacing);
    } else {
      shape_type = SPHERE;
      params = F4{radius, 0, 0, 0};  // sphere: analytic single-probe, no shell
    }

    // Upload the shell (resize the View; numPoints==0 => analytic single-probe like the sphere).
    const int nPts = static_cast<int>(shell.size());
    if (nPts > 0) {
      P_.shell = Kokkos::View<float* [3], CpMem>("shell", nPts);
      auto hs = Kokkos::create_mirror_view(P_.shell);
      for (int i = 0; i < nPts; ++i) { hs(i, 0) = shell[i].x; hs(i, 1) = shell[i].y; hs(i, 2) = shell[i].z; }
      Kokkos::deep_copy(P_.shell, hs);
    }

    auto h = Kokkos::create_mirror_view(P_.shapes);
    h(0) = ShapeDesc{shape_type, params, 0, nPts};
    Kokkos::deep_copy(P_.shapes, h);

    // Per-shape inverse inertia (mass=1), faithful to CUDA Simulation::initialize.
    float ix = 1.0f, iy = 1.0f, iz = 1.0f;
    if (shape_type == SPHERE) {
      if (baseRadius_ > 0.0f) { float v = 2.5f / (baseRadius_ * baseRadius_); ix = iy = iz = v; }
    } else if (shape_type == HOLLOW_CYLINDER) {
      float r_out = baseRadius_, r_in = baseRadius_ - thickness;
      if (r_in < 0) r_in = 0;
      float term_r = r_out * r_out + r_in * r_in;
      float I_zz = 0.5f * term_r;
      float I_xx = (1.0f / 12.0f) * (3.0f * term_r + height * height);
      if (I_xx > 1e-6f) { ix = 1.0f / I_xx; iy = 1.0f / I_xx; }
      if (I_zz > 1e-6f) iz = 1.0f / I_zz;
    } else if (shape_type == BOX) {
      float L = 2.0f * baseRadius_;
      float I = (1.0f / 6.0f) * L * L;
      if (I > 1e-6f) { ix = iy = iz = 1.0f / I; }
    }
    defaultInvI_ = F3{ix, iy, iz};
  }
  void setDomain(float lx, float ly, float lz, bool px, bool py, bool pz) {
    P_.domain = Domain{F3{0, 0, 0}, F3{lx, ly, lz}, F3{lx, ly, lz}, px, py, pz};
    P_.skin = 0.1f * P_.globalScale;
  }
  // CUDA Simulation::set_domain(min, max): arbitrary origin; keeps the current periodicity flags.
  void setDomainMinMax(F3 mn, F3 mx) {
    P_.domain = Domain{mn, mx, F3{mx.x - mn.x, mx.y - mn.y, mx.z - mn.z},
                       P_.domain.periodic_x, P_.domain.periodic_y, P_.domain.periodic_z};
    P_.skin = 0.1f * P_.globalScale;
  }
  void enablePeriodicity(bool x, bool y, bool z) {
    P_.domain.periodic_x = x; P_.domain.periodic_y = y; P_.domain.periodic_z = z;
  }
  std::tuple<float, float, float> getDomainMin() const { return {P_.domain.min.x, P_.domain.min.y, P_.domain.min.z}; }
  std::tuple<float, float, float> getDomainMax() const { return {P_.domain.max.x, P_.domain.max.y, P_.domain.max.z}; }
  void setGravity(float gx, float gy, float gz) { P_.gravity = F3{gx, gy, gz}; }
  void setThermostat(float temperature, float tau, float kB) {  // Berendsen; tau=0 disables
    P_.thermostatTemp = temperature; P_.thermostatTau = tau; P_.thermostatKB = kB;
  }
  void setSolverIterations(int pos, int vel) { P_.positionIterations = pos; P_.velocityIterations = vel; }
  void setGlobalScale(float s) { P_.globalScale = s; P_.skin = 0.1f * s; }
  void setDt(float dt) { P_.dt = dt; }
  // (restitution_normal, restitution_tangent, friction) to match CUDA set_material_params; the Kokkos
  // pipeline currently carries normal restitution + dynamic friction (tangential restitution unused).
  void setMaterialParams(float restitution_normal, float restitution_tangent, float friction) {
    P_.restitutionNormal = restitution_normal; P_.frictionDynamic = friction; (void)restitution_tangent;
  }
  void addPlane(float px, float py, float pz, float nx, float ny, float nz) {
    auto h = Kokkos::create_mirror_view(P_.planes);
    Kokkos::deep_copy(h, P_.planes);
    if (P_.numPlanes < static_cast<int>(P_.planes.extent(0)))
      h(P_.numPlanes++) = PlaneP{F3{px, py, pz}, F3{nx, ny, nz}};
    Kokkos::deep_copy(P_.planes, h);
  }

  // positions: flat [n*3]; (re)sets the real-particle count and default state.
  void setPositions(const std::vector<float>& xyz) {
    const int n = static_cast<int>(xyz.size() / 3);
    P_.numReal = n; P_.numParticles = n;
    auto pos = Kokkos::create_mirror_view(P_.pos);
    auto q = Kokkos::create_mirror_view(P_.quat);
    auto im = Kokkos::create_mirror_view(P_.invMass);
    auto sc = Kokkos::create_mirror_view(P_.scale);
    auto ii = Kokkos::create_mirror_view(P_.invInertia);
    auto sid = Kokkos::create_mirror_view(P_.shapeId);
    auto vel = Kokkos::create_mirror_view(P_.vel);
    auto av = Kokkos::create_mirror_view(P_.angVel);
    for (int i = 0; i < n; ++i) {
      pos(i, 0) = xyz[3*i]; pos(i, 1) = xyz[3*i+1]; pos(i, 2) = xyz[3*i+2];
      q(i, 0) = 0; q(i, 1) = 0; q(i, 2) = 0; q(i, 3) = 1;
      im(i) = 1.0f; sc(i) = 1.0f; sid(i) = 0;
      ii(i, 0) = defaultInvI_.x; ii(i, 1) = defaultInvI_.y; ii(i, 2) = defaultInvI_.z;
      vel(i, 0) = vel(i, 1) = vel(i, 2) = 0; av(i, 0) = av(i, 1) = av(i, 2) = 0;
    }
    Kokkos::deep_copy(P_.pos, pos); Kokkos::deep_copy(P_.quat, q); Kokkos::deep_copy(P_.invMass, im);
    Kokkos::deep_copy(P_.scale, sc); Kokkos::deep_copy(P_.invInertia, ii); Kokkos::deep_copy(P_.shapeId, sid);
    Kokkos::deep_copy(P_.vel, vel); Kokkos::deep_copy(P_.angVel, av);
    Kokkos::deep_copy(P_.targetScale, P_.scale);  // unscaled growth target = the set scale
  }
  void setScalesUniform(float s) {
    auto sc = Kokkos::create_mirror_view(P_.scale);
    for (int i = 0; i < P_.numReal; ++i) sc(i) = s;
    Kokkos::deep_copy(P_.scale, sc); Kokkos::deep_copy(P_.targetScale, P_.scale);
  }
  // per-particle scales (growth target): flat [n]. scale starts at target (growth factor applies in step).
  void setScales(const std::vector<float>& s) {
    auto tsc = Kokkos::create_mirror_view(P_.targetScale);
    for (int i = 0; i < P_.numReal && i < (int)s.size(); ++i) tsc(i) = s[i];
    Kokkos::deep_copy(P_.targetScale, tsc); Kokkos::deep_copy(P_.scale, P_.targetScale);
  }
  void setVelocities(const std::vector<float>& v) {
    auto vel = Kokkos::create_mirror_view(P_.vel);
    for (int i = 0; i < P_.numReal && 3*i+2 < (int)v.size(); ++i) { vel(i,0)=v[3*i]; vel(i,1)=v[3*i+1]; vel(i,2)=v[3*i+2]; }
    Kokkos::deep_copy(P_.vel, vel);
  }
  // rigid-body rotation state (the pipeline integrates the gyroscopic Euler term + quaternion already)
  void setQuaternions(const std::vector<float>& q) {
    auto h = Kokkos::create_mirror_view(P_.quat);
    for (int i = 0; i < P_.numReal && 4*i+3 < (int)q.size(); ++i) { h(i,0)=q[4*i]; h(i,1)=q[4*i+1]; h(i,2)=q[4*i+2]; h(i,3)=q[4*i+3]; }
    Kokkos::deep_copy(P_.quat, h);
  }
  void setAngularVelocities(const std::vector<float>& w) {
    auto h = Kokkos::create_mirror_view(P_.angVel);
    for (int i = 0; i < P_.numReal && 3*i+2 < (int)w.size(); ++i) { h(i,0)=w[3*i]; h(i,1)=w[3*i+1]; h(i,2)=w[3*i+2]; }
    Kokkos::deep_copy(P_.angVel, h);
  }
  void setInvInertia(const std::vector<float>& ii) {
    auto h = Kokkos::create_mirror_view(P_.invInertia);
    for (int i = 0; i < P_.numReal && 3*i+2 < (int)ii.size(); ++i) { h(i,0)=ii[3*i]; h(i,1)=ii[3*i+1]; h(i,2)=ii[3*i+2]; }
    Kokkos::deep_copy(P_.invInertia, h);
  }
  void setInvMass(const std::vector<float>& im) {
    auto h = Kokkos::create_mirror_view(P_.invMass);
    for (int i = 0; i < P_.numReal && i < (int)im.size(); ++i) h(i) = im[i];
    Kokkos::deep_copy(P_.invMass, h);
  }
  std::vector<float> getAngularVelocities() const {
    auto w = Kokkos::create_mirror_view(P_.angVel); Kokkos::deep_copy(w, P_.angVel);
    std::vector<float> out((size_t)P_.numReal * 3);
    for (int i = 0; i < P_.numReal; ++i) { out[3*i]=w(i,0); out[3*i+1]=w(i,1); out[3*i+2]=w(i,2); }
    return out;
  }
  std::vector<float> getInvInertia() const {
    auto ii = Kokkos::create_mirror_view(P_.invInertia); Kokkos::deep_copy(ii, P_.invInertia);
    std::vector<float> out((size_t)P_.numReal * 3);
    for (int i = 0; i < P_.numReal; ++i) { out[3*i]=ii(i,0); out[3*i+1]=ii(i,1); out[3*i+2]=ii(i,2); }
    return out;
  }
  // growth: factor *= exp(rate*dt) per step (capped at 1); new_factor<0 keeps/initialises (0.01 if inactive).
  void setGrowthParams(float rate, float new_factor) {
    if (P_.growthFactor == -1.0f) P_.growthFactor = (new_factor > 0.0f) ? new_factor : 0.01f;
    else if (new_factor > 0.0f) P_.growthFactor = new_factor;
    P_.growthRate = rate;
    if (P_.growthFactor > 0.0f) updateGrowthScalesKokkos(P_.numReal, P_.scale, P_.targetScale, P_.growthFactor);
  }
  float growthFactor() const { return P_.growthFactor; }
  float getGrowthRate() const { return P_.growthRate; }
  // per-particle mass = 1/invMass (0 for fixed/infinite-mass particles), CUDA Simulation::get_masses.
  std::vector<float> getMasses() const {
    auto im = Kokkos::create_mirror_view(P_.invMass); Kokkos::deep_copy(im, P_.invMass);
    std::vector<float> out(P_.numReal);
    for (int i = 0; i < P_.numReal; ++i) out[i] = (im(i) > 0.0f) ? (1.0f / im(i)) : 0.0f;
    return out;
  }

  std::vector<float> getPositions() const {
    auto pos = Kokkos::create_mirror_view(P_.pos);
    Kokkos::deep_copy(pos, P_.pos);
    std::vector<float> out(static_cast<size_t>(P_.numReal) * 3);
    for (int i = 0; i < P_.numReal; ++i) { out[3*i]=pos(i,0); out[3*i+1]=pos(i,1); out[3*i+2]=pos(i,2); }
    return out;
  }
  std::vector<float> getVelocities() const {
    auto v = Kokkos::create_mirror_view(P_.vel); Kokkos::deep_copy(v, P_.vel);
    std::vector<float> out((size_t)P_.numReal * 3);
    for (int i = 0; i < P_.numReal; ++i) { out[3*i]=v(i,0); out[3*i+1]=v(i,1); out[3*i+2]=v(i,2); }
    return out;
  }
  std::vector<float> getQuaternions() const {
    auto q = Kokkos::create_mirror_view(P_.quat); Kokkos::deep_copy(q, P_.quat);
    std::vector<float> out((size_t)P_.numReal * 4);
    for (int i = 0; i < P_.numReal; ++i) { out[4*i]=q(i,0); out[4*i+1]=q(i,1); out[4*i+2]=q(i,2); out[4*i+3]=q(i,3); }
    return out;
  }
  std::vector<float> getScales() const {
    auto s = Kokkos::create_mirror_view(P_.scale); Kokkos::deep_copy(s, P_.scale);
    return std::vector<float>(s.data(), s.data() + P_.numReal);
  }

  // One XPBD substep (CUDA Simulation::step(dt) semantics): dt>0 sets the timestep; dt==0 is a
  // dynamics-free relaxation step (overlap removal only). Drive the loop from Python.
  void step(float dt) { P_.dt = dt; demStep(P_); }

  // Max pair interpenetration on the current committed state (CUDA Simulation::compute_overlaps).
  float computeOverlaps() { return computeOverlapsKokkos(P_); }

  // LAMMPS "dump custom" of the current committed state (CUDA Simulation::export_lammps). Radius =
  // scale*globalScale*baseRadius; bounds computed from the particle AABBs.
  void exportLammps(const std::string& filename, int step) const {
    const std::vector<float> pos = getPositions(), vel = getVelocities(), quat = getQuaternions();
    auto sc = Kokkos::create_mirror_view(P_.scale); Kokkos::deep_copy(sc, P_.scale);
    std::vector<float> radii(P_.numReal);
    for (int i = 0; i < P_.numReal; ++i) radii[i] = sc(i) * P_.globalScale * baseRadius_;
    const bool pbc = P_.domain.periodic_x || P_.domain.periodic_y || P_.domain.periodic_z;
    dem::writeLammpsDump(filename, step, pos, vel, quat, radii, nullptr, nullptr, pbc);
  }

  // SDF field over the domain -> ImageData VTI (CUDA Simulation::export_sdf).
  void exportSdf(const std::string& filename, int rx, int ry, int rz) {
    const std::vector<float> grid = getSdfGrid(rx, ry, rz);
    const float mn[3] = {P_.domain.min.x, P_.domain.min.y, P_.domain.min.z};
    const float mx[3] = {P_.domain.max.x, P_.domain.max.y, P_.domain.max.z};
    dem::writeSdfVti(filename, grid, rx, ry, rz, mn, mx);
  }

#ifdef DEMGPU_MPI
  // Block decomposition over the GLOBAL domain (once); the per-block solver stays non-periodic, the
  // halo supplies the periodic wrap. gsize is the ORB cell grid. Mirror of Simulation::mpi_init.
  void initMpi(std::tuple<double, double, double> origin, std::tuple<double, double, double> size,
               std::tuple<long, long, long> gsize, std::tuple<bool, bool, bool> periodic,
               MPI_Comm comm) {
    halo_.initMpi({std::get<0>(origin), std::get<1>(origin), std::get<2>(origin)},
                  {std::get<0>(size), std::get<1>(size), std::get<2>(size)},
                  {std::get<0>(gsize), std::get<1>(gsize), std::get<2>(gsize)},
                  {std::get<0>(periodic), std::get<1>(periodic), std::get<2>(periodic)}, comm);
  }
  // Enable the distributed step. rcut is the ghost-band width (default = 1.0*globalScale, the periodic
  // skin used by the single-GPU path); sync_every is the owner->ghost refresh interval (1 = EXACT).
  void enableMpiStep(double rcut, int sync_every = 1, bool forward_rotation = true) {
    mpiRcut_ = rcut; mpiSyncEvery_ = sync_every < 1 ? 1 : sync_every; mpiForwardRotation_ = forward_rotation;
  }
  void stepMpi(int nsteps) {
    const double rcut = (mpiRcut_ > 0.0) ? mpiRcut_ : 1.0 * P_.globalScale;
    for (int s = 0; s < nsteps; ++s) demStepMpi(P_, halo_, rcut, mpiSyncEvery_, mpiForwardRotation_);
  }
  int rank() const { return halo_.rank(); }
  int numGhost() const { return halo_.numGhost(); }
#endif  // DEMGPU_MPI

  // SDF grid (get_sdf_grid): Eikonal reconstruction over the domain, flat x-fastest, negative inside solid.
  std::vector<float> getSdfGrid(int rx, int ry, int rz) {
    return dem::generateSdfKokkos(rx, ry, rz, P_.domain.min, P_.domain.max, P_.numReal, P_.pos, P_.quat,
                                  P_.scale, P_.shapeId, P_.shapes, P_.domain.periodic_x, P_.domain.periodic_y,
                                  P_.domain.periodic_z);
  }

  int numParticles() const { return P_.numReal; }
  int numContacts() { return readInt(P_.contactCount); }
  int numManifolds() { return readInt(P_.manifoldCount); }
  float maxOverlap() { float h; Kokkos::deep_copy(h, P_.maxOverlap); return h; }

  // ParaView PolyData (points + Radius + Velocity), faithful to CUDA Simulation::write_vtp:
  // Radius = scale * globalScale * baseRadius.
  void writeVtp(const std::string& filename) const {
    auto pos = Kokkos::create_mirror_view(P_.pos);   Kokkos::deep_copy(pos, P_.pos);
    auto sc = Kokkos::create_mirror_view(P_.scale);  Kokkos::deep_copy(sc, P_.scale);
    auto vel = Kokkos::create_mirror_view(P_.vel);   Kokkos::deep_copy(vel, P_.vel);
    const int n = P_.numReal;

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Could not open file for writing: " + filename);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << n << "\" NumberOfVerts=\"0\" "
        << "NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Position\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < n; ++i) out << pos(i, 0) << " " << pos(i, 1) << " " << pos(i, 2) << " ";
    out << "\n        </DataArray>\n";
    out << "      </Points>\n";
    out << "      <PointData Scalars=\"Radius\">\n";
    out << "        <DataArray type=\"Float32\" Name=\"Radius\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int i = 0; i < n; ++i) out << sc(i) * P_.globalScale * baseRadius_ << " ";
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < n; ++i) out << vel(i, 0) << " " << vel(i, 1) << " " << vel(i, 2) << " ";
    out << "\n        </DataArray>\n";
    out << "      </PointData>\n";
    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
    out.close();
    std::printf("Exported VTP: %s\n", filename.c_str());
  }

 private:
  Particles P_;
  float baseRadius_ = 1.0f;
  F3 defaultInvI_{2.5f, 2.5f, 2.5f};
#ifdef DEMGPU_MPI
  KokkosParticleHalo halo_;
  double mpiRcut_ = 0.0;
  int mpiSyncEvery_ = 1;
  bool mpiForwardRotation_ = true;
#endif
};

}  // namespace dem

#endif  // DEM_SIM_HPP
