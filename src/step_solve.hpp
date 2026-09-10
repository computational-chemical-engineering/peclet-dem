/// @file
/// @brief dem — the single-rank step drivers over the particle SoA: the XPBD substep (`demStep`),
/// its committed-state overlap probe (`computeOverlapsKokkos`), and the contact-buffer sizing the
/// narrow phase relies on (`growContactBuffers` / `narrowPhaseGrow`). Free functions; the
/// `Simulation` facade in sim.hpp calls them, the distributed drivers in step_solve_mpi.hpp share
/// the buffer sizing and the narrow phase.
#ifndef DEM_STEP_SOLVE_HPP
#define DEM_STEP_SOLVE_HPP

#include <algorithm>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"
#include "integration.hpp"
#include "narrowphase.hpp"
#include "particles.hpp"
#include "periodicity.hpp"
#include "sleeping.hpp"      // island sleeping / freezing (single-GPU statics)
#include "solve_driver.hpp"  // demSolveContacts + SoloSolveHooks + readInt/readFloat

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
                         P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap, P.sdfGrid,
                         P.materialId, P.pairMaterials);
    detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId,
                         P.shapes, P.shell, P.planes, P.globalScale, margin, P.contacts,
                         P.contactCount, P.maxOverlap);
    if (P.numWalls > 0)
      detectWallSdfKokkos(P.numReal, P.numWalls, P.posPred, P.quatPred, P.scale, P.shapeId,
                          P.shapes, P.shell, P.walls, P.wallGrid, P.globalScale, margin, P.contacts,
                          P.contactCount, P.maxOverlap, P.materialId, P.pairMaterials);
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

  fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numParticles);
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
  fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numParticles);
  const int np = findCollisionsGrow(P, margin);
  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  (void)narrowPhaseGrow(P, np, margin);
  P.numParticles = P.numReal;
  float h;
  Kokkos::deep_copy(h, P.maxOverlap);
  return h;
}

}  // namespace peclet::dem

#endif  // DEM_STEP_SOLVE_HPP
