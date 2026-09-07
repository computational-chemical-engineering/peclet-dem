/// @file
/// @brief dem — portable (Kokkos) Simulation facade: the dem flip's host-facing driver.
///
/// Owns a peclet::dem::Particles SoA and runs the full XPBD DEM step by composing the ported
/// kernels in the simulation.cpp step() order. Exposes a small std::vector-based API
/// (binding-agnostic) so a pybind module can drive it from Python (set/get arrays, step). Sphere
/// shapes + analytic planes for now.
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
#include "periodicity.hpp"
#include "shapes_portable.hpp"
#include "peclet/core/geom/scene_builder.hpp"
#include "sleeping.hpp"            // island sleeping / freezing (single-GPU statics)
#include "solve_driver.hpp"        // demSolveContacts + SoloSolveHooks + readInt/readFloat
#include "solve_driver_force.hpp"  // demStepForce + HertzMindlinLaw + demStepHertz
#include "solver_friction.hpp"
#include "solver_hertz.hpp"
#include "solver_multilevel.hpp"
#include "solver_position.hpp"
#include "solver_velocity.hpp"

#ifdef PECLET_DEM_MPI
#include "mpi_halo.hpp"  // ParticleHalo (gated; default module never includes it)
#endif

namespace peclet::dem {

/// Grow every maxContacts-sized collision buffer to match the CURRENT particle capacity.
///
/// Sized from `capacity`, not `numReal`: periodic ghost slots take part in the narrow phase, so a
/// buffer sized when capacity was smaller is an out-of-bounds write the moment the ghost band
/// grows it. That is exactly what happened -- `demStep` calls `ensureCapacity` every step (the
/// ghost headroom is `numReal + estGhosts + 4096`), so a Simulation constructed with a small
/// capacity had its contact buffers frozen at the construction size while the particle SoA grew
/// by an order of magnitude around them. Found via peclet-examples/pall-ring-packing, where it
/// showed up twice: boundary contacts silently dropped (scene-shape grains falling through an
/// add_plane floor) and heap corruption inside step() at ~2000 steps.
///
/// `nBodies` is the count the setup-time bound is taken over (`capacity`, exactly as the original
/// sizing did, so every existing run is byte-identical). `floorWant` is the per-step path: the
/// narrow phase knows the contact count it actually needs and asks for it directly. A
/// per-PARTICLE bound is not safe for a many-probe composed shape -- a pair can contribute up to
/// shellPoints contacts and a grain has several neighbours -- and sizing from the ghost capacity
/// is worse still, since calculateGhostCapacity's +4096 slots of slack alone ask for 13 million
/// contacts (several GB across the ~40 buffers) with a 1625-probe particle. See narrowPhaseGrow.
inline void growContactBuffers(Particles& P, long nBodies, long floorWant = 0) {
  const int perParticle = std::max(16, P.shellPoints);
    const long want = std::max(
        floorWant, nBodies * perParticle + nBodies * std::max(1, P.shellPoints) * P.numWalls);
    if (want > P.maxContacts) {
      P.maxContacts = static_cast<int>(want);
      // Reallocate EVERY maxContacts-sized view, not just contacts/manifolds: the solve writes all
      // of them up to the live contact/manifold count, so any view left at the old size is an
      // out-of-bounds write once the count grows past it (silent device corruption on GPU, heap
      // corruption on host backends). Warm-start history is cleared by the fresh zeroed views —
      // growth happens at setup (shape/wall registration), so nothing warm is lost mid-run.
      P.contacts = Kokkos::View<ContactC*, CpMem>("contacts", want);
      P.manifolds = Kokkos::View<ManifoldC*, CpMem>("manifolds", want);
      P.manifoldColor = Kokkos::View<int*, CpMem>("manifoldColor", want);
      P.pairKeys = Kokkos::View<unsigned long long*, CpMem>("pairKeys", want);
      P.prevPairKeys = Kokkos::View<unsigned long long*, CpMem>("prevPairKeys", want);
      P.manifoldPersistent = Kokkos::View<unsigned char*, CpMem>("manifoldPersistent", want);
      P.contactColor = Kokkos::View<int*, CpMem>("contactColor", want);
      P.lambdaAcc = Kokkos::View<float*, CpMem>("lambdaAcc", want);
      P.lambdaT = Kokkos::View<float* [3], CpMem>("lambdaT", want);
      P.posLambdaContact = Kokkos::View<float*, CpMem>("posLambdaContact", want);
      P.posImpulse = Kokkos::View<float*, CpMem>("posImpulse", want);
      P.prevPosImpulse = Kokkos::View<float*, CpMem>("prevPosImpulse", want);
      P.restBank = Kokkos::View<float*, CpMem>("restBank", want);
      P.prevRestBank = Kokkos::View<float*, CpMem>("prevRestBank", want);
      P.restRel = Kokkos::View<float*, CpMem>("restRel", want);
      P.restVPeak = Kokkos::View<float*, CpMem>("restVPeak", want);
      P.prevRestVPeak = Kokkos::View<float*, CpMem>("prevRestVPeak", want);
      P.prevMatched = Kokkos::View<unsigned char*, CpMem>("prevMatched", want);
      P.velPerm = Kokkos::View<int*, CpMem>("velPerm", want);
      P.commitPerm = Kokkos::View<int*, CpMem>("commitPerm", want);
      P.sideFlags = Kokkos::View<unsigned char*, CpMem>("sideFlags", want);
      P.prevLambdaT = Kokkos::View<float* [3], CpMem>("prevLambdaT", want);
      P.contactSlot = Kokkos::View<int*, CpMem>("contactSlot", want);
      P.prevLambda = Kokkos::View<float*, CpMem>("prevLambda", want);
      P.vn0 = Kokkos::View<float*, CpMem>("vn0", want);
      P.vt0 = Kokkos::View<float* [3], CpMem>("vt0", want);
      P.levelKey = Kokkos::View<int*, CpMem>("levelKey", want);
      P.levelPerm = Kokkos::View<int*, CpMem>("levelPerm", want);
      P.mlColorPacked = Kokkos::View<long long*, CpMem>("mlColorPacked", want);
      // Incremental-colouring ledgers + fused position permutation + sleeping masks are all
      // maxContacts-sized and the solve indexes them up to the live contact/manifold count too —
      // they MUST grow with the buffer or nc > extent is an out-of-bounds write (NaN / heap
      // corruption in dense multi-contact scenes such as the statics column/pour).
      P.prevManifoldColor = Kokkos::View<int*, CpMem>("prevManifoldColor", want);
      P.contactKeys = Kokkos::View<unsigned long long*, CpMem>("contactKeys", want);
      P.prevContactKeys = Kokkos::View<unsigned long long*, CpMem>("prevContactKeys", want);
      P.prevContactColor = Kokkos::View<int*, CpMem>("prevContactColor", want);
      P.posCommitPerm = Kokkos::View<int*, CpMem>("posCommitPerm", want);
      P.posPerm = Kokkos::View<int*, CpMem>("posPerm", want);
      P.manifoldSleep = Kokkos::View<unsigned char*, CpMem>("manifoldSleep", want);
      P.contactSleep = Kokkos::View<unsigned char*, CpMem>("contactSleep", want);
      P.posPrevContactCount = 0;  // the cleared position ledger must not be gathered against
      P.prevPairCount = 0;        // the cleared prevPairKeys must not be gathered against
    }
}

/// Narrow phase with an automatically-grown contact buffer.
///
/// The three detect kernels guard their contact WRITES at the buffer extent, but `P.contactCount`
/// is the RAW number of contacts found -- and every consumer downstream (manifold reduction, the
/// sleeping masks, the whole solve) uses that count as a loop bound over maxContacts-sized views.
/// When the raw count exceeds the buffer, those loops walk off the end: "corrupted double-linked
/// list" on a host backend, cudaErrorIllegalAddress on a device one. Exactly the failure mode
/// findCollisionsGrow already handles for the broad-phase pair buffer, and handled the same way --
/// detect the overflow, grow with headroom, re-run once so no contact is silently dropped, then
/// clamp defensively so the returned count is ALWAYS <= the extent.
///
/// This is what actually bounds the contact buffer. Sizing it from the particle count (capacity or
/// numParticles) was only ever a proxy: a pair can contribute up to shellPoints contacts and a
/// grain has several neighbours, so no per-particle bound is safe for a many-probe composed shape.
/// Found via peclet-examples/pall-ring-packing (48 rings x 1625 probes).
inline int narrowPhaseGrow(Particles& P, int np, float margin) {
  CpExec space;
  auto detect = [&]() {
    Kokkos::deep_copy(space, P.contactCount, 0);
    detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                         P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap,
                         P.sdfGrid, P.materialId, P.pairMaterials);
    detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId,
                         P.shapes, P.shell, P.planes, P.globalScale, margin, P.contacts,
                         P.contactCount, P.maxOverlap);
    if (P.numWalls > 0)
      detectWallSdfKokkos(P.numReal, P.numWalls, P.posPred, P.quatPred, P.scale, P.shapeId,
                          P.shapes, P.shell, P.walls, P.wallGrid, P.globalScale, margin,
                          P.contacts, P.contactCount, P.maxOverlap, P.materialId, P.pairMaterials);
    return readInt(P.contactCount);
  };
  int nc = detect();
  if (nc > P.maxContacts) {
    growContactBuffers(P, 0, static_cast<long>(nc) + nc / 2 + 64);  // 1.5x + slack
    nc = detect();
  }
  return std::min(nc, P.maxContacts);
}

/// One full XPBD DEM substep over the particle SoA (mirrors simulation.cpp Simulation::step()).
inline void demStep(Particles& P) {
  CpExec space;

  // growth ramp (faithful to CUDA Simulation::step): factor *= exp(rate*dt), capped at 1, then
  // scale = targetScale * factor. (CUDA stores the unscaled target in d_target_scales.)
  if (P.growthFactor != -1.0f && P.growthRate != 0.0f) {
    P.growthFactor *= std::exp(P.growthRate * P.dt);
    if (P.growthFactor > 1.0f)
      P.growthFactor = 1.0f;
  }
  if (P.growthFactor > 0.0f)
    updateGrowthScalesKokkos(P.numReal, P.scale, P.targetScale, P.growthFactor);

  // ghost band + broadphase margin sized off the ACTUAL max grain radius (post-growth), so SI-unit
  // particles just work; identical to the old 0.1/1.0*globalScale when globalScale ~ the grain
  // size.
  const float maxRad = maxOwnedRadius(P);
  const float margin = 0.1f * maxRad;

  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt, P.extForce,
                        P.extTorque);

  // Island sleeping (single-GPU statics, opt-in, gravity on, no external drag): freeze the
  // currently-asleep bodies so gravity/prediction do not move them; the wake pass + both-asleep
  // exclusion + effective-inverse-mass swap happen after the narrow phase (below). See
  // sleeping.hpp.
  const float gMag =
      std::sqrt(P.gravity.x * P.gravity.x + P.gravity.y * P.gravity.y + P.gravity.z * P.gravity.z);
  const bool sleepStep =
      P.sleepingEnabled && gMag > 0.0f && !P.extForceActive && !P.extTorqueActive;
  const float sleepVRest = 2.0f * P.dt * gMag;
  if (sleepStep)
    freezeAsleepKokkos(P.numReal, P.asleep, P.pos, P.quat, P.posPred, P.quatPred, P.velPred,
                       P.angVelPred);

  {
    auto ri = P.realIndices;
    Kokkos::parallel_for(
        "self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
        KOKKOS_LAMBDA(int i) { ri(i) = i; });
  }
  Kokkos::deep_copy(space, P.topGhost, P.numReal);
  // periodic ghost band = max grain radius: the CLOSER particle of any cross-boundary contacting
  // pair is within one radius of the face, so a band of maxRad ghosts it (sufficient for
  // sphere-sphere).
  const float ghostBand = maxRad;
  // Size the SoA for the ghost boundary layer BEFORE emitting (CUDA did this in initialize() via
  // calculate_capacity). Without it a Simulation(numReal) leaves capacity==numReal, so every ghost
  // overflows P.capacity in generateGhostsKokkos and cross-boundary contacts are never detected.
  P.ensureCapacity(calculateGhostCapacity(P.numReal, P.domain, ghostBand));
  generateGhostsKokkos(P.numReal, P.capacity, P.domain, ghostBand, P.pos, P.invMass, P.posPred,
                       P.vel, P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale,
                       P.shapeId, P.realIndices, P.topGhost, P.gid, P.materialId);
  P.numParticles = readInt(P.topGhost);

  {
    auto sc = P.scale;
    auto rad = P.rad;
    float gs = P.globalScale, bR = P.baseRadius;
    Kokkos::parallel_for(
        "rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
        KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs * bR; });
  }
  // Collision detection runs on the PREDICTED state (speculative positions/orientations), matching
  // the CUDA solver — the position solve then corrects posPred against these contacts.
  // findCollisionsGrow fences + reads the pair count back to host and guarantees np ≤ P.pairs
  // extent (growing the buffer on overflow) so the narrowphase never reads P.pairs out of bounds.
  // Verlet-cached broadphase (opt-in, non-periodic only: periodic ghosts are regenerated per step
  // with unstable slot ids, which cached pairs would reference). Skips the ArborX rebuild while
  // nothing moved more than skin/2 — composes with sleeping (a frozen bed never rebuilds).
  const bool verletOK = P.verletSkinFrac > 0.0f && !P.domain.periodic_x && !P.domain.periodic_y &&
                        !P.domain.periodic_z && P.numParticles == P.numReal;
  const int np = verletOK ? findCollisionsVerlet(P, margin, maxRad) : findCollisionsGrow(P, margin);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  const int nc = narrowPhaseGrow(P, np, margin);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount, P.contactSlot);
  const int nm = readInt(P.manifoldCount);

  // Island sleeping: wake any sleeper actually disturbed (fast approaching neighbour, moving wall,
  // or a change in its contact set), then flag the frozen (both-asleep) manifolds/contacts and give
  // every sleeper effective inverse mass 0 for the solve. Swapping P.invMass -> P.invMassEff makes
  // a sleeper immovable everywhere in the driver (both-asleep constraints become no-ops; the
  // exclusion just skips their now-wasted colouring/sweeps/hierarchy work), so the solve is
  // unchanged for awake bodies and the ledger still carries the frozen force network.
  Vf savedInvMass = P.invMass;
  if (sleepStep) {
    wakeDisturbedKokkos(P.manifolds, nm, P.realIndices, P.velPred, P.wakeScale * sleepVRest,
                        P.asleep, P.sleepCounter, P.sleepMovingWall, P.sleepCurCount, P.numReal);
    wakeContactChangeKokkos(P.numReal, P.sleepCurCount, P.sleepPrevCount, P.sleepMovingWall,
                            P.asleep, P.sleepCounter, P.sleepWakeLostContact);
    computeManifoldSleepKokkos(P.manifolds, nm, P.realIndices, P.asleep, P.manifoldSleep);
    computeContactSleepKokkos(P.contacts, nc, P.realIndices, P.asleep, P.contactSleep);
    buildInvMassEffKokkos(P.numParticles, P.asleep, P.realIndices, P.invMass, P.invMassEff,
                          P.sleepImmovableFrac);
    P.invMass = P.invMassEff;
  }

  // Full modern velocity + position solve (warm-started colored PGS, gravity statics /
  // stabilization, friction, colored-GS overlap projection) — shared with the distributed step.
  demSolveContacts(P, nc, nm, P.numReal, P.realIndices, SoloSolveHooks{});

  if (sleepStep)
    P.invMass = savedInvMass;  // restore the real inverse mass for the commit / next step
  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);

  // Sleep detection: a grounded body whose motion stayed below the resting floor for K substeps
  // goes to sleep (velocity zeroed). Runs on the committed velocities + the solve's grounded
  // levels.
  if (sleepStep)
    updateSleepKokkos(P.numReal, P.vel, P.angVel, P.rad, P.groundedLevel, P.sleepMovingWall,
                      P.asleep, P.sleepCounter, P.sleepScale * sleepVRest, P.sleepK, P.vel,
                      P.angVel);

  // Berendsen thermostat at the end of the step (CUDA Simulation::step), tau>0 enables.
  if (P.thermostatTau > 0.0f && P.dt > 0.0f)
    applyThermostatKokkos(P.numReal, P.vel, P.invMass, P.angVel, P.invInertia, P.quat,
                          P.thermostatKB, P.thermostatTau, P.thermostatTemp, P.dt);
}

/// Max pair interpenetration on the *committed* state (faithful to CUDA
/// Simulation::compute_overlaps): copy committed pos/quat into the predicted buffers, regenerate
/// the periodic ghosts from that state, then run the same broad/narrow phase as demStep and return
/// the recorded max overlap. No solve.
inline float computeOverlapsKokkos(Particles& P) {
  CpExec space;
  const float maxRad = maxOwnedRadius(P);
  const float margin = 0.1f * maxRad;
  P.numParticles = P.numReal;
  Kokkos::deep_copy(P.posPred, P.pos);
  Kokkos::deep_copy(P.quatPred, P.quat);
  {
    auto ri = P.realIndices;
    Kokkos::parallel_for(
        "self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
        KOKKOS_LAMBDA(int i) { ri(i) = i; });
  }
  Kokkos::deep_copy(space, P.topGhost, P.numReal);
  const float ghostBand = maxRad;
  // Match demStep: ensure ghost-boundary-layer headroom so cross-boundary overlaps are counted (a
  // Simulation(numReal) otherwise has capacity==numReal and every ghost overflows). See demStep.
  P.ensureCapacity(calculateGhostCapacity(P.numReal, P.domain, ghostBand));
  generateGhostsKokkos(P.numReal, P.capacity, P.domain, ghostBand, P.pos, P.invMass, P.posPred,
                       P.vel, P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale,
                       P.shapeId, P.realIndices, P.topGhost, P.gid, P.materialId);
  P.numParticles = readInt(P.topGhost);
  {
    auto sc = P.scale;
    auto rad = P.rad;
    float gs = P.globalScale, bR = P.baseRadius;
    Kokkos::parallel_for(
        "rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
        KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs * bR; });
  }
  const int np = findCollisionsGrow(P, margin);
  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  (void)narrowPhaseGrow(P, np, margin);
  P.numParticles = P.numReal;
  float h;
  Kokkos::deep_copy(h, P.maxOverlap);
  return h;
}

#ifdef PECLET_DEM_MPI
/// MPI hooks for the shared contact-solve driver (demSolveContacts): the distributed step is the
/// processor-block Gauss-Seidel form of the SAME modern sequence the single-GPU step runs. The
/// colouring and the sweeps stay rank-local over owned + ghost bodies (ghost pairs are solved
/// redundantly on both owners -- each rank keeps only its owned bodies' updates, the ghost copies
/// are overwritten at the next refresh), the owners re-publish their ghost state every `syncEvery`
/// solver iterations plus once after every solve phase, and each adaptive-stop residual is
/// Allreduce-MAXed so all ranks take the same break (the refreshes are collective -- a rank-local
/// break would deadlock them).
struct MpiSolveHooks {
  static constexpr bool distributed = true;
  ParticleHalo& halo;
  int syncEvery;
  bool forwardRotation;
  float allMax(float v) const {
    float g = v;
    MPI_Allreduce(&v, &g, 1, MPI_FLOAT, MPI_MAX, halo.comm());
    return g;
  }
  bool syncPoint(int it) const { return (it + 1) % syncEvery == 0; }
  void syncVelocities(Particles& P) const {
    halo.forward(P.velPred);
    if (forwardRotation)
      halo.forward(P.angVelPred);
  }
  void syncPositions(Particles& P) const {
    halo.forwardPositions(P.posPred);
    if (forwardRotation)
      halo.forward4(P.quatPred);
  }
};

/// One distributed XPBD DEM substep. The periodic ghost generation of the single-rank step is
/// replaced by a cross-rank gather (halo.gather, ghosts carrying REAL mass + the owner's gid /
/// material / grounded level), then the FULL modern solve sequence runs through demSolveContacts
/// with MpiSolveHooks -- graph-colored Gauss-Seidel restitution, warm-started PGS with
/// persistent contacts (pair keys built from GLOBAL ids, so they survive halo rebuilds and
/// ownership migration), gravity statics (grounded shock propagation / stabilization passes),
/// friction cone, colored-GS overlap projection and the adaptive stops -- identical physics to
/// the single-GPU demStep, same fixed point, not bit-exact (rank-local sweep order differs).
/// `forwardRotation`=false (spheres) skips the angular/quaternion forwards.
///
/// PERIODICITY: cross-rank ghosts supply the wrap on DECOMPOSED axes; LOCAL periodic self-ghosts
/// (ParticleHalo build with includePeriodicSelf) supply it on UNDECOMPOSED periodic axes (a "x1"
/// ORB axis, e.g. z of a 2x2x1 layout, or np=1). Correct for any layout, including np=1 fully
/// periodic. CAPACITY: a periodic box needs a thick ghost boundary layer -- size the Simulation
/// capacity for the worst-case ghost band; gather() throws on overflow rather than corrupting the
/// SoA.
inline void demStepMpi(Particles& P, ParticleHalo& halo, double rcut, int syncEvery,
                       bool forwardRotation) {
  CpExec space;
  const float margin = 0.1f * maxOwnedRadius(P);

  if (P.growthFactor != -1.0f && P.growthRate != 0.0f) {
    P.growthFactor *= std::exp(P.growthRate * P.dt);
    if (P.growthFactor > 1.0f)
      P.growthFactor = 1.0f;
  }
  if (P.growthFactor > 0.0f)
    updateGrowthScalesKokkos(P.numReal, P.scale, P.targetScale, P.growthFactor);

  // 1. Predict velocity on the owned set (no ghosts yet -> numParticles == numReal).
  P.numParticles = P.numReal;
  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt, P.extForce,
                        P.extTorque);

  // 2. Gather ghosts (real mass) from owners over the halo: full state -- including gid,
  //    materialId and the warm grounded level -- into the ghost slots; sets
  //    P.numParticles = numReal + numGhost and self-maps realIndices.
  halo.gather(P, rcut);

  {
    auto sc = P.scale;
    auto rad = P.rad;
    float gs = P.globalScale, bR = P.baseRadius;
    Kokkos::parallel_for(
        "rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
        KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs * bR; });
  }

  // 3. Broad/narrow phase + manifold reduction over owned + ghosts (contactSlot map included:
  // the PGS friction bound and the position-channel Coulomb carry read through it).
  // findCollisionsGrow fences + reads the pair count back to host and guarantees np <= P.pairs
  // extent (growing the buffer on overflow) so the narrowphase never reads P.pairs out of bounds.
  const int np = findCollisionsGrow(P, margin);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  const int nc = narrowPhaseGrow(P, np, margin);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount, P.contactSlot);
  const int nm = readInt(P.manifoldCount);

  // 4-6. The shared modern velocity + position solve, distributed: rank-local colouring over the
  // owned + ghost body slots (nBodies = numParticles; realIndices are self-mapped, so ghost
  // copies evolve in place between refreshes), persistent-pair keys from the global ids.
  demSolveContacts(P, nc, nm, P.numParticles, P.gid,
                   MpiSolveHooks{halo, syncEvery < 1 ? 1 : syncEvery, forwardRotation});

  // 7. Commit (owned results kept; ghosts discarded, re-gathered next substep).
  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);

  if (P.thermostatTau > 0.0f && P.dt > 0.0f)
    applyThermostatKokkos(P.numReal, P.vel, P.invMass, P.angVel, P.invInertia, P.quat,
                          P.thermostatKB, P.thermostatTau, P.thermostatTemp, P.dt);

  P.numParticles = P.numReal;  // restore owned-only active count for getters
}

/// MPI hooks for the force-based driver (demStepForce): domain-decomposed explicit DEM in the
/// classical MD mold. One halo gather (fresh topology, band = pair cutoff + skin) per Verlet
/// pair-list rebuild; between rebuilds only the ghost STATE is forwarded owner->ghost each step.
/// Forces on ghost slots are discarded (the neighbour rank computes the mirrored pair itself);
/// the skin / rebuild / cache-validity decisions are Allreduced so the collective schedule is
/// identical on all ranks.
struct MpiForceHooks {
  static constexpr bool distributed = true;
  ParticleHalo& halo;
  double band;
  float allMax(float v) const {
    float g = v;
    MPI_Allreduce(&v, &g, 1, MPI_FLOAT, MPI_MAX, halo.comm());
    return g;
  }
  float allMin(float v) const {
    float g = v;
    MPI_Allreduce(&v, &g, 1, MPI_FLOAT, MPI_MIN, halo.comm());
    return g;
  }
  void gatherGhosts(Particles& P) const {
    halo.invalidateTopology();  // fresh band + fresh positions at every pair rebuild
    halo.gather(P, band);
    fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numParticles);
  }
  void refreshGhostState(Particles& P, bool needQuat) const {
    halo.forwardPositions(P.pos);
    halo.forward(P.vel);
    halo.forward(P.angVel);
    if (needQuat)
      halo.forward4(P.quat);
  }
  void clearGhostScratch(Particles& P) const {
    zeroForceScratchKokkos(P.deltaVel, P.deltaAngVel, P.numReal, P.numReal + halo.numGhost());
  }
};

/// `nsteps` distributed force-based (Hertz–Mindlin) steps — the MPI instantiation of
/// demStepForce. Per-pair Mindlin history is gid-keyed (stable across halo rebuilds/migration);
/// non-periodic domains only (matching the single-GPU engine).
inline void demStepHertzMpi(Particles& P, ParticleHalo& halo, float dt, int nsteps,
                            float skinFrac) {
  // Ghost band = worst-case pair cutoff (2 R_max,global) + skin. The driver's skin is
  // skinFrac * R_min,global <= skinFrac * R_max,global, so this band bounds it.
  float maxR = maxOwnedRadius(P), maxRg = maxR;
  MPI_Allreduce(&maxR, &maxRg, 1, MPI_FLOAT, MPI_MAX, halo.comm());
  const double band = (2.0 + skinFrac) * static_cast<double>(maxRg);
  demStepForce(P, dt, nsteps, skinFrac, HertzMindlinLaw{}, MpiForceHooks{halo, band});
  P.numParticles = P.numReal;  // restore owned-only active count for getters
}
#endif  // PECLET_DEM_MPI

/// Host-facing facade with std::vector setters/getters (binding-agnostic).
class Simulation {
 public:
  explicit Simulation(int capacity) {
    registry().push_back(this);
    P_.allocate(capacity, capacity * 64, capacity * 16, /*shapes*/ 1, /*shell*/ 1, /*planes*/ 8);
    // default sphere shape (radius 1) + identity-ish defaults
    setSphereShape(1.0f);
    // A/B hook for the validation battery (mirrors PECLET_DEM_STAB_MODE's role): switch the
    // restitution model without touching driver scripts.
    if (const char* e = std::getenv("PECLET_DEM_REST_MODEL"); e && *e)
      setRestitutionModel(e);
    // Island sleeping A/B env (default ON): PECLET_DEM_SLEEP=1 enables, =0 disables; the scales /
    // K / wake threshold have their own overrides for the tuning battery.
    if (const char* e = std::getenv("PECLET_DEM_SLEEP"); e && *e)
      P_.sleepingEnabled = std::atoi(e) != 0;
    if (const char* e = std::getenv("PECLET_DEM_SLEEP_SCALE"); e && *e)
      P_.sleepScale = std::atof(e);
    if (const char* e = std::getenv("PECLET_DEM_SLEEP_K"); e && *e)
      P_.sleepK = std::atoi(e);
    if (const char* e = std::getenv("PECLET_DEM_WAKE_SCALE"); e && *e)
      P_.wakeScale = std::atof(e);
    if (const char* e = std::getenv("PECLET_DEM_SLEEP_WAKELOST"); e && *e)
      P_.sleepWakeLostContact = std::atoi(e) != 0;
    if (const char* e = std::getenv("PECLET_DEM_SLEEP_INVMASS_FRAC"); e && *e)
      P_.sleepImmovableFrac = std::atof(e);
    // Verlet-cached impulse broadphase (default OFF): PECLET_DEM_VERLET_SKIN = skin fraction of the
    // max grain radius (e.g. 0.3). 0 = rebuild every step.
    if (const char* e = std::getenv("PECLET_DEM_VERLET_SKIN"); e && *e)
      P_.verletSkinFrac = std::atof(e);
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

  void setSphereShape(float radius) { initializeShape(SPHERE, radius, 0.0f, 0.0f); }

  // Mirror of CUDA Simulation::initialize(shape_type, radius, height, thickness): builds shape 0's
  // descriptor + surface point shell (cylinder/box) and records the per-shape base radius and
  // (uniform-mass=1) inverse inertia applied to every particle by setPositions. shape_type uses the
  // peclet::dem::ShapeKind values (SPHERE=1, HOLLOW_CYLINDER=2, BOX=3).
  void initializeShape(int shape_type, float radius, float height, float thickness) {
    baseRadius_ = radius;
    P_.baseRadius =
        radius;  // effective radius = scale*globalScale*baseRadius (broadphase + ghost band)
    F4 params{radius, 0, 0, 0};
    std::vector<F3> shell;

    if (shape_type == HOLLOW_CYLINDER) {
      params = F4{radius, height, thickness, 0};
      // Dynamic spacing (faithful to CUDA): >=4 pts across thickness, >=20 around circumference.
      float min_dim = std::min(radius, thickness);
      if (min_dim < 1e-4f)
        min_dim = radius;  // safety if thickness 0
      float spacing = std::min(radius * 0.3f, min_dim * 0.5f);
      if (spacing < 1e-3f)
        spacing = 1e-3f;
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

    const int nPts = static_cast<int>(shell.size());

    // Per-shape inverse inertia (mass=1), faithful to CUDA Simulation::initialize.
    float ix = 1.0f, iy = 1.0f, iz = 1.0f;
    if (shape_type == SPHERE) {
      if (baseRadius_ > 0.0f) {
        float v = 2.5f / (baseRadius_ * baseRadius_);
        ix = iy = iz = v;
      }
    } else if (shape_type == HOLLOW_CYLINDER) {
      float r_out = baseRadius_, r_in = baseRadius_ - thickness;
      if (r_in < 0)
        r_in = 0;
      float term_r = r_out * r_out + r_in * r_in;
      float I_zz = 0.5f * term_r;
      float I_xx = (1.0f / 12.0f) * (3.0f * term_r + height * height);
      if (I_xx > 1e-6f) {
        ix = 1.0f / I_xx;
        iy = 1.0f / I_xx;
      }
      if (I_zz > 1e-6f)
        iz = 1.0f / I_zz;
    } else if (shape_type == BOX) {
      float L = 2.0f * baseRadius_;
      float I = (1.0f / 6.0f) * L * L;
      if (I > 1e-6f) {
        ix = iy = iz = 1.0f / I;
      }
    }
    // RESET the registry to this one shape -- initializeShape's documented behaviour is "shape 0
    // becomes this". Use addShape() to build a mixture instead.
    clearShapes();
    ShapeDesc sd{shape_type, params, 0, nPts};
    appendShape(sd, shell, F3{ix, iy, iz}, radius);
    uploadShapes();
    ensureContactCapacity();
  }

  /// Append an analytic shape and return its index, for simulations with a MIXTURE of shapes.
  /// Same arguments as initializeShape, which stays the "single shape" entry point (it resets the
  /// registry). Assign the returned index to particles with setShapeIds().
  int addShape(int shape_type, float radius, float height, float thickness) {
    const std::size_t before = shapesHost_.size();
    if (before == 0)
      throw std::runtime_error("addShape: call initialize_shape/set_sphere_shape first");
    // Reuse initializeShape's descriptor + shell construction by running it into a scratch
    // registry, then splice the result back on top of the existing shapes.
    std::vector<ShapeDesc> keepS = shapesHost_;
    std::vector<F3> keepShell = shellHost_;
    std::vector<F3> keepInvI = invIHost_;
    std::vector<float> keepBase = baseRadiusHost_;
    std::vector<float> keepSamples = sdfSamplesHost_;  // initializeShape clears these too
    initializeShape(shape_type, radius, height, thickness);  // leaves exactly one shape
    ShapeDesc added = shapesHost_[0];
    const std::vector<F3> addedShell = shellHost_;
    const F3 addedInvI = invIHost_[0];
    const float addedBase = baseRadiusHost_[0];
    shapesHost_ = keepS;
    shellHost_ = keepShell;
    invIHost_ = keepInvI;
    baseRadiusHost_ = keepBase;
    sdfSamplesHost_ = keepSamples;
    appendShape(added, addedShell, addedInvI, addedBase);
    uploadShapes();
    ensureContactCapacity();
    return static_cast<int>(shapesHost_.size()) - 1;
  }

  /// Per-particle shape assignment. Also refreshes each particle's inverse inertia from its new
  /// shape, so the order of setPositions/setShapeIds does not matter.
  void setShapeIds(const std::vector<int>& ids) {
    if (static_cast<int>(ids.size()) != P_.numReal)
      throw std::runtime_error("set_shape_ids: expected one id per particle");
    const int nShapes = static_cast<int>(shapesHost_.size());
    for (int v : ids)
      if (v < 0 || v >= nShapes)
        throw std::runtime_error("set_shape_ids: shape id out of range");
    auto sid = Kokkos::create_mirror_view(P_.shapeId);
    auto ii = Kokkos::create_mirror_view(P_.invInertia);
    Kokkos::deep_copy(sid, P_.shapeId);
    Kokkos::deep_copy(ii, P_.invInertia);
    for (int i = 0; i < P_.numReal; ++i) {
      sid(i) = ids[i];
      ii(i, 0) = invIHost_[ids[i]].x;
      ii(i, 1) = invIHost_[ids[i]].y;
      ii(i, 2) = invIHost_[ids[i]].z;
    }
    Kokkos::deep_copy(P_.shapeId, sid);
    Kokkos::deep_copy(P_.invInertia, ii);
  }

  /// Append a grid-SDF shape (the general non-spherical particle) to the mixture and return its
  /// index. Same arguments as set_sdf_shape, which stays the single-shape entry point. The samples
  /// are appended to the shared pool and the descriptor's offset set accordingly.
  int addSdfShape(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                  const std::vector<float>& shellFlat, F3 invInertia, float boundingRadius) {
    if (shapesHost_.empty())
      throw std::runtime_error("add_sdf_shape: call initialize_shape/set_sdf_shape first");
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("add_sdf_shape: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("add_sdf_shape: grid.size() must equal nx*ny*nz");
    const std::vector<float> shellUse =
        shellFlat.empty() ? autoShell(grid, nx, ny, nz, origin, spacing, boundingRadius)
                          : shellFlat;
    const int nPts = static_cast<int>(shellUse.size() / 3);
    if (nPts <= 0)
      throw std::runtime_error("add_sdf_shape: surface point shell is empty and could not be "
                               "generated from the field");
    ShapeDesc sd{};
    sd.type = SHAPE_GRID_SDF;
    sd.params = F4{boundingRadius, 0, 0, 0};
    sd.grid.nx = nx;
    sd.grid.ny = ny;
    sd.grid.nz = nz;
    sd.grid.offset = static_cast<int>(sdfSamplesHost_.size());  // append to the shared pool
    sd.grid.origin = toCoreVec(origin);
    sd.grid.invSpacing = peclet::core::Vec3<float>{
        spacing.x > 0 ? 1.0f / spacing.x : 0.0f, spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
        spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
    sd.grid.extension = peclet::core::geom::GridExtension::kObject;
    sdfSamplesHost_.insert(sdfSamplesHost_.end(), grid.begin(), grid.end());
    std::vector<F3> shellPts(nPts);
    for (int i = 0; i < nPts; ++i)
      shellPts[i] = F3{shellUse[3 * i], shellUse[3 * i + 1], shellUse[3 * i + 2]};
    appendShape(sd, shellPts, invInertia, boundingRadius);
    uploadShapes();
    ensureContactCapacity();
    return static_cast<int>(shapesHost_.size()) - 1;
  }

  int numShapes() const { return static_cast<int>(shapesHost_.size()); }

  // Import a general particle as a grid SDF (canonical, unit-scale space) + a surface point shell +
  // its unit-mass principal-frame diagonal inverse inertia. Replaces shape 0, so every particle
  // becomes an instance of this shape (per-particle position/quaternion/scale still apply). `grid`
  // holds nx*ny*nz signed-distance samples, x-fastest (idx = x + y*nx + z*nx*ny), at lattice nodes
  // q = origin + (x,y,z)*spacing (negative inside). `shellFlat` is the flat [nPts*3] surface point
  // set the collision probes body A against. boundingRadius is the canonical radius enclosing the
  // surface (broad-phase + VTI splat bound). invInertia is the per-unit-mass diagonal inverse
  // inertia in the body (principal) frame; like the analytic shapes it becomes the default applied
  // to every particle by setPositions (override afterwards with set_inv_inertia / set_inv_mass for
  // a real density). See peclet.dem.particle_builder for the Python side that produces these arrays
  // from an implicit-solid SDF (marching-cubes shell + voxel-integrated mass properties).
  void setSdfShape(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                   const std::vector<float>& shellFlat, F3 invInertia, float boundingRadius) {
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("setSdfShape: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("setSdfShape: grid.size() must equal nx*ny*nz");
    // An EMPTY shell means "generate one from the field itself" (Layer 1).
    const std::vector<float> shellUse =
        shellFlat.empty() ? autoShell(grid, nx, ny, nz, origin, spacing, boundingRadius)
                          : shellFlat;
    const int nPts = static_cast<int>(shellUse.size() / 3);
    if (nPts <= 0)
      throw std::runtime_error("setSdfShape: surface point shell is empty and could not be "
                               "generated from the field");

    // Upload the signed-distance samples.
    P_.sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", grid.size());
    auto hg = Kokkos::create_mirror_view(P_.sdfGrid);
    for (size_t i = 0; i < grid.size(); ++i)
      hg(i) = grid[i];
    Kokkos::deep_copy(P_.sdfGrid, hg);

    // Upload the surface point shell.
    P_.shell = Kokkos::View<float* [3], CpMem>("shell", nPts);
    auto hs = Kokkos::create_mirror_view(P_.shell);
    for (int i = 0; i < nPts; ++i) {
      hs(i, 0) = shellFlat[3 * i];
      hs(i, 1) = shellFlat[3 * i + 1];
      hs(i, 2) = shellFlat[3 * i + 2];
    }
    Kokkos::deep_copy(P_.shell, hs);

    // Shape descriptor: grid SDF for the field, shell points for the probes.
    ShapeDesc sd{};
    sd.type = SHAPE_GRID_SDF;
    sd.params = F4{boundingRadius, 0, 0, 0};
    sd.shellOffset = 0;
    sd.numPoints = nPts;
    sd.grid.offset = 0;
    sd.grid.nx = nx;
    sd.grid.ny = ny;
    sd.grid.nz = nz;
    sd.grid.origin = toCoreVec(origin);
    sd.grid.invSpacing = peclet::core::Vec3<float>{
        spacing.x > 0 ? 1.0f / spacing.x : 0.0f, spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
        spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
    sd.grid.extension = peclet::core::geom::GridExtension::kObject;  // a body

    std::vector<F3> shellPts(nPts);
    for (int i = 0; i < nPts; ++i)
      shellPts[i] = F3{shellUse[3 * i], shellUse[3 * i + 1], shellUse[3 * i + 2]};
    clearShapes();
    sdfSamplesHost_ = grid;   // this shape owns the whole pool when it is the only shape
    sd.grid.offset = 0;
    appendShape(sd, shellPts, invInertia, boundingRadius);
    uploadShapes();
    ensureContactCapacity();
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
  // Composed-analytic particle shape (SHAPE_SCENE): decode core's flat node encoding into the
  // shared per-Simulation pool (child indices rebased to ABSOLUTE pool indices, exactly like
  // addAnalyticWall), register a ShapeDesc whose tree pointer is patched at uploadShapes. The
  // canonical frame is trusted to be the PRINCIPAL frame -- SceneBuilder.principal_frame emits
  // that; passing a non-principal tree runs the diagonal-inertia rotational update on the wrong
  // frame, silently, which is why the docstring shouts about it.
  int addSceneShape(const std::vector<int>& nodeInts, const std::vector<float>& nodeReals,
                    int root, const std::vector<float>& shellFlat, F3 invInertia,
                    float boundingRadius) {
    namespace g = peclet::core::geom;
    if (nodeInts.size() % g::kNodeIntStride || nodeReals.size() % g::kNodeRealStride)
      throw std::runtime_error("add_scene_shape: node arrays are not a whole number of records");
    const int n = static_cast<int>(nodeInts.size() / g::kNodeIntStride);
    if (n == 0 || static_cast<int>(nodeReals.size() / g::kNodeRealStride) != n)
      throw std::runtime_error("add_scene_shape: node int/real counts disagree");
    if (root < 0 || root >= n)
      throw std::runtime_error("add_scene_shape: root out of range");
    const int base = static_cast<int>(shapeNodesHost_.size());
    for (int i = 0; i < n; ++i) {
      g::ShapeNode<float> nd = g::decodeNode<float>(&nodeInts[i * g::kNodeIntStride],
                                                    &nodeReals[i * g::kNodeRealStride]);
      if (nd.kind == g::kGrid)
        throw std::runtime_error(
            "add_scene_shape: grid leaves are not supported in particle trees (use set_sdf_shape "
            "for sampled shapes)");
      if (nd.kind >= g::kCsgBase) {
        nd.aux0 += base;
        nd.aux1 += base;
      }
      shapeNodesHost_.push_back(nd);
    }
    std::vector<F3> shellPts;
    if (shellFlat.size() % 3)
      throw std::runtime_error("add_scene_shape: shell must be (M,3)");
    shellPts.reserve(shellFlat.size() / 3);
    for (std::size_t i = 0; i + 2 < shellFlat.size(); i += 3)
      shellPts.push_back(F3{shellFlat[i], shellFlat[i + 1], shellFlat[i + 2]});
    if (shellPts.empty())
      throw std::runtime_error(
          "add_scene_shape: an empty shell would make the body invisible to contacts -- bake the "
          "tree and generate one (SceneBuilder.bake + the marching-cubes shell path)");
    ShapeDesc sd{};
    sd.type = SHAPE_SCENE;
    sd.params = F4{boundingRadius, 0, 0, 0};
    sd.nodeCount = 0;   // patched at uploadShapes (whole-pool count; root is absolute)
    sd.shapeRoot = base + root;
    appendShape(sd, shellPts, invInertia, boundingRadius);
    uploadShapes();
    ensureContactCapacity();  // every other shape adder does this; its absence dropped contacts
    return static_cast<int>(shapesHost_.size()) - 1;
  }


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

  /// Generate a collision shell for a grid SDF by sampling its own zero level set, so a caller
  /// does not have to supply one. Layer 1: previously every shape needed a hand-written generator
  /// (there were only two, for the cylinder and the box) or a marching-cubes shell computed in
  /// Python; core's surfacePoints() is driven by the SDF itself and works for any geometry.
  std::vector<float> autoShell(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin,
                               F3 spacing, float boundingRadius) const {
    namespace g = peclet::core::geom;
    g::SceneBuilder<float> b;
    const int node = b.addGrid(grid, nx, ny, nz,
                               peclet::core::Vec3<float>{origin.x, origin.y, origin.z},
                               peclet::core::Vec3<float>{spacing.x, spacing.y, spacing.z},
                               g::GridExtension::kObject);
    // Pitch: fine enough that the shell resolves the body, coarse enough not to explode the
    // contact buffers -- ~1/12 of the bounding radius matches the density of the hand-written
    // cylinder/box shells.
    const float pitch = std::max(boundingRadius / 12.0f, 1e-4f);
    const float r = boundingRadius * 1.05f;
    const std::vector<peclet::core::Vec3<float>> pts = g::surfacePoints<float>(
        b.view(), node, pitch, peclet::core::Vec3<float>{-r, -r, -r},
        peclet::core::Vec3<float>{r, r, r});
    std::vector<float> flat;
    flat.reserve(pts.size() * 3);
    for (const auto& q : pts) {
      flat.push_back(q.x);
      flat.push_back(q.y);
      flat.push_back(q.z);
    }
    return flat;
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
    P_.domain = Domain{origin,
                       F3{origin.x + extent.x, origin.y + extent.y, origin.z + extent.z},
                       extent,
                       px,
                       py,
                       pz};
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
  /// Enable/disable the stabilization pass (default on). Off = pure momentum-conserving PGS
  /// everywhere -- exact ballistic response, but deep static columns mid-collapse cannot be
  /// arrested within the iteration budget. Backward-compatible boolean form of
  /// setStabilizationMode: true selects "onesided", false selects "off".
  void setStabilization(bool enabled) { P_.stabilizationMode = enabled ? 1 : 0; }
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
          "set_stabilization_mode: expected 'off', 'onesided', 'multilevel', 'escalate' or "
          "'ordered'");
  }
  /// Restitution model of the PGS velocity solve: "newton" (default; per-substep restitution on
  /// the pre-solve approach — the pre-existing behaviour) or "poisson" (event-level: each pair
  /// banks its kinetic compression impulse and releases e x the bank as a budget-capped
  /// separation-velocity target during unloading — restores the multi-substep-impact rebound that
  /// per-substep Newton structurally cannot return). PECLET_DEM_REST_MODEL overrides at startup.
  void setRestitutionModel(const std::string& model) {
    if (model == "newton")
      P_.restitutionModel = 0;
    else if (model == "poisson")
      P_.restitutionModel = 1;
    else
      throw std::invalid_argument("set_restitution_model: expected 'newton' or 'poisson'");
  }
  /// Island sleeping / freezing (single-GPU statics, default ON; PECLET_DEM_SLEEP=0 disables). A
  /// REAL body whose linear AND
  /// angular motion stays below `scale` x the resting floor (2 dt |g|) for K substeps while
  /// grounded is put to sleep: velocity zeroed, integration skipped, and a manifold whose BOTH
  /// endpoints are asleep (a static wall counts) is excluded from the colouring / sweeps /
  /// multilevel hierarchy — so a settled bed collapses to the broad/narrow-phase floor. A sleeper
  /// keeps a small POSITIVE effective inverse mass in the solve (sleepImmovableFrac x its own; heavy
  /// but not perfectly rigid), so an awake body wedged between sleepers relieves against them
  /// instead of the PGS normal impulse diverging; its velocity is re-zeroed each substep so no
  /// momentum accumulates. It wakes only when disturbed (fast approaching neighbour, contact-set
  /// change, moving wall).
  /// Requires gravity on and no external (CFD-DEM drag) force; inert under MPI. PECLET_DEM_SLEEP
  /// overrides at startup.
  void setSleeping(bool enabled, float threshold_scale = 2.0f, int consecutive = 64,
                   float wake_scale = 40.0f) {
    P_.sleepingEnabled = enabled;
    if (threshold_scale > 0.0f)
      P_.sleepScale = threshold_scale;
    if (consecutive > 0)
      P_.sleepK = consecutive;
    if (wake_scale > 0.0f)
      P_.wakeScale = wake_scale;  // hysteresis: wake only well above the residual settling jitter
  }
  /// Verlet-cached impulse broadphase (single-GPU, non-periodic; default OFF). skin_frac is the
  /// broadphase-skin fraction of the max grain radius: the ArborX rebuild is skipped while no
  /// particle has moved more than skin/2, so between rebuilds the candidate list is a superset and
  /// the narrowphase yields identical contacts. Composes with sleeping (a frozen bed never
  /// rebuilds).
  void setVerletSkin(float skin_frac) {
    P_.verletSkinFrac = skin_frac;
    P_.impNumPairs = -1;  // invalidate the cache
  }
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
  /// Advance `substeps` explicit soft-sphere Hertz-Mindlin steps of size dt (device-side loop;
  /// the (e, mu) pairs come from the impulse solver's material tables).
  void stepHertz(float dt, int substeps, float skin_frac) {
    P_.dt = dt;
    demStepHertz(P_, dt, substeps, skin_frac);
  }
  void setGlobalScale(float s) {
    P_.globalScale = s;
    P_.skin = 0.1f * s;
  }
  void setDt(float dt) { P_.dt = dt; }
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
    w.grid.invSpacing = peclet::core::Vec3<float>{
        spacing.x > 0 ? 1.0f / spacing.x : 0.0f, spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
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

  // One XPBD substep (CUDA Simulation::step(dt) semantics): dt>0 sets the timestep; dt==0 is a
  // dynamics-free relaxation step (overlap removal only). Drive the loop from Python.
  void step(float dt) {
    P_.dt = dt;
    demStep(P_);
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
    const double rcut = (mpiRcut_ > 0.0) ? mpiRcut_ : maxOwnedRadius(P_);
    ensureGlobalGids();
    for (int s = 0; s < nsteps; ++s) {
      if (mpiRebalanceEvery_ > 0 && mpiStepCount_ % mpiRebalanceEvery_ == 0)
        halo_->rebalance(P_);
      demStepMpi(P_, *halo_, rcut, mpiSyncEvery_, mpiForwardRotation_);
      ++mpiStepCount_;
    }
  }
  /// Advance `substeps` distributed explicit Hertz–Mindlin (force-based) steps of size dt — the
  /// MPI counterpart of step_hertz, on the same halo/decomposition as step_mpi (init_mpi +
  /// enable_mpi_step first). rebalance_every counts CALLS of this method (each call = one
  /// Rayleigh-limited inner batch); the migration carries the Mindlin pair/wall history.
  void stepHertzMpi(float dt, int substeps, float skin_frac) {
    P_.dt = dt;
    ensureGlobalGids();
    if (mpiRebalanceEvery_ > 0 && mpiHertzCalls_ % mpiRebalanceEvery_ == 0)
      halo_->rebalance(P_);
    ++mpiHertzCalls_;
    demStepHertzMpi(P_, *halo_, dt, substeps, skin_frac);
  }
  int rank() const { return halo_->rank(); }
  int numGhost() const { return halo_->numGhost(); }
#endif  // PECLET_DEM_MPI

  // SDF grid (get_sdf_grid): Eikonal reconstruction over the domain, flat x-fastest, negative
  // inside solid.
  std::vector<float> getSdfGrid(int rx, int ry, int rz) {
    return peclet::dem::generateSdfKokkos(
        rx, ry, rz, P_.domain.min, P_.domain.max, P_.numReal, P_.pos, P_.quat, P_.scale, P_.shapeId,
        P_.shapes, P_.domain.periodic_x, P_.domain.periodic_y, P_.domain.periodic_z, P_.sdfGrid,
        P_.globalScale);
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

  // Size the contact/manifold buffers so no contact is dropped: a shell point sits inside at most
  // one neighbour (body-body ~ capacity*shellPoints) plus one per wall it touches
  // (capacity*shellPoints per wall). Boundary/wall contacts are appended AFTER body-body ones, so
  // an undersized buffer silently drops them and grains tunnel through walls. Floored at the
  // analytic default; grows only.
  void ensureContactCapacity() {
    P_.shellPoints = shellPoints_;
    growContactBuffers(P_, P_.capacity);
  }

  Particles P_;
  float baseRadius_ = 1.0f;
  // --- host-side shape registry (Layer 1) -------------------------------------------------
  // The device Views (P_.shapes / P_.shell / P_.sdfGrid) are rebuilt from these by uploadShapes(),
  // which is what assigns each shape its shellOffset and grid offset into the shared pools. Keeping
  // the authoritative copy on the host is what makes a MIXTURE of shapes possible at all: the old
  // code wrote descriptor slot 0 directly and had nowhere to put a second shell.
  std::vector<ShapeDesc> shapesHost_;
  std::vector<F3> shellHost_;         // concatenated shells; ShapeDesc::shellOffset indexes it
  std::vector<F3> invIHost_;          // per-shape unit-mass inverse inertia
  std::vector<float> baseRadiusHost_;  // per-shape canonical bounding radius
  std::vector<float> sdfSamplesHost_;  // concatenated grid-SDF samples

  void clearShapes() {
    shapesHost_.clear();
    shellHost_.clear();
    invIHost_.clear();
    baseRadiusHost_.clear();
    sdfSamplesHost_.clear();
  }

  /// Append one shape, taking ownership of its shell. Fills in shellOffset from the running
  /// concatenation; the caller has already set the grid fields (if any).
  void appendShape(ShapeDesc sd, const std::vector<F3>& shell, F3 invI, float baseRadius) {
    sd.shellOffset = static_cast<int>(shellHost_.size());
    sd.numPoints = static_cast<int>(shell.size());
    shellHost_.insert(shellHost_.end(), shell.begin(), shell.end());
    shapesHost_.push_back(sd);
    invIHost_.push_back(invI);
    baseRadiusHost_.push_back(baseRadius);
  }

  /// Rebuild the device Views from the host registry.
  void uploadShapes() {
    const int nShapes = static_cast<int>(shapesHost_.size());
    if (nShapes == 0)
      return;
    // Composed-analytic node pool first, so the ShapeDesc pointers below can refer to it. Rebuilt
    // whole (KBs); every SHAPE_SCENE descriptor points at the pool BASE with an absolute root, so
    // one View serves every tree (the wallNodes pattern).
    if (!shapeNodesHost_.empty()) {
      P_.shapeNodes = Kokkos::View<peclet::core::geom::ShapeNode<float>*, CpMem>(
          "shapeNodes", shapeNodesHost_.size());
      auto hn = Kokkos::create_mirror_view(P_.shapeNodes);
      for (std::size_t i = 0; i < shapeNodesHost_.size(); ++i)
        hn(i) = shapeNodesHost_[i];
      Kokkos::deep_copy(P_.shapeNodes, hn);
    }
    if (static_cast<int>(P_.shapes.extent(0)) < nShapes)
      P_.shapes = Kokkos::View<ShapeDesc*, CpMem>("shapes", nShapes);
    auto hs = Kokkos::create_mirror_view(P_.shapes);
    for (int i = 0; i < nShapes; ++i) {
      hs(i) = shapesHost_[i];
      if (hs(i).type == SHAPE_SCENE) {
        hs(i).nodes = P_.shapeNodes.data();
        hs(i).nodeCount = static_cast<int>(P_.shapeNodes.extent(0));
      }
    }
    Kokkos::deep_copy(P_.shapes, hs);

    const int nPts = static_cast<int>(shellHost_.size());
    if (nPts > 0) {
      P_.shell = Kokkos::View<float* [3], CpMem>("shell", nPts);
      auto hsh = Kokkos::create_mirror_view(P_.shell);
      for (int i = 0; i < nPts; ++i) {
        hsh(i, 0) = shellHost_[i].x;
        hsh(i, 1) = shellHost_[i].y;
        hsh(i, 2) = shellHost_[i].z;
      }
      Kokkos::deep_copy(P_.shell, hsh);
    }

    if (!sdfSamplesHost_.empty()) {
      P_.sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", sdfSamplesHost_.size());
      auto hg = Kokkos::create_mirror_view(P_.sdfGrid);
      for (std::size_t i = 0; i < sdfSamplesHost_.size(); ++i)
        hg(i) = sdfSamplesHost_[i];
      Kokkos::deep_copy(P_.sdfGrid, hg);
    }

    // Broad-phase band and contact-buffer sizing must cover the LARGEST shape present, not the
    // most recently added one.
    float rmax = 0.0f;
    int shellMax = 0;
    for (int i = 0; i < nShapes; ++i) {
      rmax = std::max(rmax, baseRadiusHost_[i]);
      shellMax = std::max(shellMax, shapesHost_[i].numPoints);
    }
    baseRadius_ = rmax;
    P_.baseRadius = rmax;
    defaultInvI_ = invIHost_[0];
    shellPoints_ = shellMax;
  }

  int shellPoints_ = 0;  // LARGEST surface-shell size across shapes (contact-buffer sizing)
  std::vector<WallSdf> wallsHost_;
  std::vector<float> wallGridHost_;
  std::vector<peclet::core::geom::ShapeNode<float>> wallNodesHost_;
  // R1: authored root transform per analytic-wall root node (see setWallTransform).
  std::map<int, peclet::core::geom::Transform<float>> wallRootTf_;
  std::vector<peclet::core::geom::ShapeNode<float>> shapeNodesHost_;  // SHAPE_SCENE pool
  F3 defaultInvI_{2.5f, 2.5f, 2.5f};
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
