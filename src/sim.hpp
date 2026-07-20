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
#include <memory>
#include <stdexcept>
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
#include "solver_friction.hpp"
#include "solver_position.hpp"
#include "solver_velocity.hpp"

#ifdef PECLET_DEM_MPI
#include "mpi_halo.hpp"  // ParticleHalo (gated; default module never includes it)
#endif

namespace peclet::dem {

inline int readInt(Kokkos::View<int, CpMem> v) {
  int h;
  Kokkos::deep_copy(h, v);
  return h;
}
inline float readFloat(Kokkos::View<float, CpMem> v) {
  float h;
  Kokkos::deep_copy(h, v);
  return h;
}

/// Largest effective particle radius over the owned set (= max scale × globalScale, growth included).
/// The ghost band + broadphase margin are sized off THIS, not globalScale directly, so they scale with
/// the actual grain size — set particles in SI (`radius = 1e-3`, globalScale left at 1) and the halo
/// layer follows automatically. For the usual convention (globalScale ≈ grain size, scale ≈ 1) it is
/// numerically identical to the old `1.0*globalScale`.
inline float maxOwnedRadius(const Particles& P) {
  if (P.numReal <= 0)
    return P.globalScale * P.baseRadius;
  float mx = 0.0f;
  auto sc = P.scale;
  Kokkos::parallel_reduce(
      "peclet::dem::max_scale", Kokkos::RangePolicy<CpExec>(0, P.numReal),
      KOKKOS_LAMBDA(int i, float& m) { m = sc(i) > m ? sc(i) : m; }, Kokkos::Max<float>(mx));
  return mx * P.globalScale * P.baseRadius;
}

/// Broad phase with an automatically-grown pair buffer.
///
/// `findCollisionsArborX` guards its pair WRITES at `maxPairs` but returns the RAW candidate count,
/// which can exceed `P.pairs`' capacity once a bed compacts (more neighbour pairs than the buffer
/// holds — e.g. a fluidized bed driven denser by the CFD-DEM drag). Feeding that raw count straight
/// into `detectContactsKokkos` as its loop bound makes the narrowphase read `P.pairs` out of bounds
/// → `cudaErrorIllegalAddress`. Here we detect the overflow, reallocate `P.pairs` (with headroom so
/// an oscillating count doesn't realloc every step) and re-run once so no candidate pair is silently
/// dropped, then clamp defensively so the returned count is ALWAYS ≤ the buffer extent — the
/// narrowphase can never walk off the end regardless.
inline int findCollisionsGrow(Particles& P, float margin) {
  const float boxCap = std::max(std::max(P.domain.size.x, P.domain.size.y), P.domain.size.z);
  int np = findCollisionsArborX(P.posPred, P.crad(), P.numParticles, P.numReal, margin, P.pairs,
                                P.pairCount, boxCap);
  if (np > static_cast<int>(P.pairs.extent(0))) {
    const int grown = np + np / 2 + 64;  // 1.5× + slack
    Kokkos::realloc(Kokkos::WithoutInitializing, P.pairs, grown);
    P.maxPairs = grown;
    np = findCollisionsArborX(P.posPred, P.crad(), P.numParticles, P.numReal, margin, P.pairs,
                              P.pairCount, boxCap);
  }
  return std::min(np, static_cast<int>(P.pairs.extent(0)));
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
  // particles just work; identical to the old 0.1/1.0*globalScale when globalScale ~ the grain size.
  const float maxRad = maxOwnedRadius(P);
  const float margin = 0.1f * maxRad;

  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt, P.extForce);

  {
    auto ri = P.realIndices;
    Kokkos::parallel_for(
        "self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
        KOKKOS_LAMBDA(int i) { ri(i) = i; });
  }
  Kokkos::deep_copy(space, P.topGhost, P.numReal);
  // periodic ghost band = max grain radius: the CLOSER particle of any cross-boundary contacting pair
  // is within one radius of the face, so a band of maxRad ghosts it (sufficient for sphere-sphere).
  const float ghostBand = maxRad;
  // Size the SoA for the ghost boundary layer BEFORE emitting (CUDA did this in initialize() via
  // calculate_capacity). Without it a Simulation(numReal) leaves capacity==numReal, so every ghost
  // overflows P.capacity in generateGhostsKokkos and cross-boundary contacts are never detected.
  P.ensureCapacity(calculateGhostCapacity(P.numReal, P.domain, ghostBand));
  generateGhostsKokkos(P.numReal, P.capacity, P.domain, ghostBand, P.pos, P.invMass, P.posPred,
                       P.vel, P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale,
                       P.shapeId, P.realIndices, P.topGhost);
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
  // findCollisionsGrow fences + reads the pair count back to host and guarantees np ≤ P.pairs extent
  // (growing the buffer on overflow) so the narrowphase never reads P.pairs out of bounds.
  const int np = findCollisionsGrow(P, margin);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap, P.sdfGrid,
                       P.materialId, P.pairMaterials);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                       P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount,
                       P.maxOverlap);
  if (P.numWalls > 0)
    detectWallSdfKokkos(P.numReal, P.numWalls, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                        P.shell, P.walls, P.wallGrid, P.globalScale, margin, P.contacts,
                        P.contactCount, P.maxOverlap, P.materialId, P.pairMaterials);
  const int nc = readInt(P.contactCount);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount, P.contactSlot);
  const int nm = readInt(P.manifoldCount);

  // A frictional wall drives friction even when the body-body material is frictionless.
  const bool friction = (P.frictionDynamic > 0.0f || P.wallFrictionMax > 0.0f);
  const bool usePersistPre = (P.gravity.x != 0.0f || P.gravity.y != 0.0f || P.gravity.z != 0.0f);
  const bool legacyFriction = friction && !(usePersistPre && P.velocityUseGS);
  if (legacyFriction)
    computePlaneLoadKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                           P.planeFriction);

  // Colour the manifold graph ONCE (topology-only; reused across the sweeps), then normal
  // restitution as colored Gauss–Seidel: correct multi-contact dissipation with no count-averaging
  // (see solver_velocity.hpp). count==1 binary collisions are identical to the old Jacobi path.
  int velLeftover = 0;
  const int numColors =
      P.velocityUseGS ? colorManifoldsKokkos(P.manifolds, nm, P.realIndices, P.numReal,
                                             P.manifoldColor, P.bodyWinner, P.bodyColorMask,
                                             velLeftover)
                      : 0;
  // Persistent-contact restitution, gravity-gated (|g| = 0 leaves behaviour untouched: growth
  // packing / HCS bit-identical). A pair already in contact LAST substep is loaded, not impacting:
  // it gets e = 0 (the impulse still cancels the approach — pure inelastic support), so the
  // velocity solve carries a pile's static weight through impulse chains and a settling column
  // actually cools; material/wall restitution stays reserved for newly formed contacts.
  const bool usePersist = (P.gravity.x != 0.0f || P.gravity.y != 0.0f || P.gravity.z != 0.0f);
  // Warm-started PGS velocity solve (|g| > 0): gather each manifold's previous-substep converged
  // push impulse by pair key, record the pre-solve approach (restitution bias), and apply the warm
  // impulses up front -- a static pile's force network is re-established in ~one sweep. g = 0
  // keeps the original one-shot colored-GS path bit-identical (HCS, growth packing).
  const bool usePGS = usePersist && P.velocityUseGS;
  const float gMagP = Kokkos::sqrt(P.gravity.x * P.gravity.x + P.gravity.y * P.gravity.y +
                                   P.gravity.z * P.gravity.z);
  const F3 gHat = usePersist ? F3{P.gravity.x / gMagP, P.gravity.y / gMagP, P.gravity.z / gMagP}
                             : F3{0, 0, 0};
  if (usePGS) {
    gatherWarmLambdaKokkos(P.manifolds, nm, P.realIndices, P.prevPairKeys, P.prevLambda,
                           P.prevLambdaT, P.prevPairCount, P.pairKeys, P.lambdaAcc, P.lambdaT);
    markPersistentManifoldsKokkos(P.manifolds, nm, P.realIndices, P.prevPairKeys, P.prevPairCount,
                                  P.pairKeys, P.manifoldPersistent);
    updateGroundedLevelsKokkos(P.manifolds, nm, P.realIndices, P.posPred, gHat, P.groundedLevel,
                               P.numReal, /*sweeps*/ 8, /*decay*/ 8);
    // STAGED SOLVE (Guendelman): the main sweeps are fully momentum-conserving (side flags all
    // zero) -- ballistic impact, discharge and shear see correct physics. One-sided grounding is
    // reserved for the STABILIZATION pass below, which runs only if the main sweeps leave an
    // unconverged residual (a deep column mid-collapse that symmetric GS cannot arrest within the
    // iteration budget).
    {
      auto flags = Kokkos::subview(P.sideFlags, Kokkos::pair<int, int>(0, nm));
      Kokkos::deep_copy(flags, static_cast<unsigned char>(0));
    }
    computeVn0Kokkos(P.manifolds, nm, P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.vn0);
    warmStartApplyKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                         P.realIndices, P.lambdaAcc, P.lambdaT);
  }
  for (int it = 0; it < P.velocityIterations; ++it) {
    if (legacyFriction)
      accumulateNormalImpulseKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred,
                                    P.angVelPred, P.realIndices, P.growthRate);
    // Restitution threshold ~ the speed one substep of free fall gains: below it a contact is
    // RESTING and bounces with e=0 (see solveVelocityKokkos — dense-pile energy-bomb guard).
    const float vRest = 2.0f * P.dt *
                        Kokkos::sqrt(P.gravity.x * P.gravity.x + P.gravity.y * P.gravity.y +
                                     P.gravity.z * P.gravity.z);
    if (P.velocityUseGS) {
      Kokkos::deep_copy(P.maxApproach, 0.0f);
      if (usePGS)
        solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia,
                               P.quat, P.velPred, P.angVelPred, P.realIndices, P.growthRate,
                               P.restitutionNormal, vRest, P.maxApproach, P.lambdaAcc, P.vn0,
                               Kokkos::View<const unsigned char*, CpMem>(P.sideFlags),
                               P.lambdaT, P.frictionDynamic);
      else
        solveVelocityColoredGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                     P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                     P.growthRate, P.restitutionNormal, vRest, P.maxApproach);
      // Colour-mask saturation fallback (interpenetration degree > 62): the manifolds the colouring
      // could not place are applied with the count-averaged Jacobi pass — stable, and only active
      // in pathologically crushed regions; without it those manifolds were silently skipped and
      // deep overlap could never resolve.
      if (velLeftover > 0) {
        solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred,
                            P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRest,
                            P.deltaVel, P.deltaAngVel, P.constraintCounts,
                            Kokkos::View<const int*, CpMem>(P.manifoldColor), -1);
        applyVelocityDeltasAveragedKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel,
                                          P.deltaAngVel, P.constraintCounts);
      }
      // Adaptive stop. One-shot GS: end once no pair approaches above the resting threshold. PGS:
      // maxApproach records the largest APPLIED correction, and meaningful increments are ~g dt
      // (they propagate a chain one link per sweep), so the tolerance must sit well below vRest or
      // the stop starves deep-chain convergence permanently (measured: a 113-layer pile plateaued
      // at vz ~ -5 with the vRest stop). Once the warm-started network is converged the first
      // sweep's correction is ~0 and the loop still exits immediately.
      if (readFloat(P.maxApproach) <= (usePGS ? 0.02f * vRest : vRest))
        break;
    } else {
      solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                          P.realIndices, P.growthRate, P.restitutionNormal, vRest, P.deltaVel,
                          P.deltaAngVel, P.constraintCounts);
      applyVelocityDeltasAveragedKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel,
                                        P.deltaAngVel, P.constraintCounts);
    }
  }
  // STABILIZATION PASS: if the symmetric sweeps could not drain the residual (a collapsing
  // column needs ~one sweep per layer to carry its weight to the floor -- unaffordable), arrest
  // the remaining quasi-static approach with grounded one-sided sweeps. In dynamic scenes the
  // residual is below the threshold and this pass never runs, so impact/discharge/shear keep
  // pure momentum-conserving physics. (PECLET_DEM_SYMMETRIC_PGS=1 disables the pass -- sandbox
  // A/B toggle.)
  if (usePGS) {
    const float vRestS = 2.0f * P.dt * gMagP;
    if (P.stabilization && readFloat(P.maxApproach) > vRestS) {
      computeSideFlagsKokkos(P.manifolds, nm, P.realIndices,
                             Kokkos::View<const unsigned char*, CpMem>(P.manifoldPersistent),
                             Kokkos::View<const unsigned char*, CpMem>(P.groundedLevel), P.posPred,
                             P.velPred, gHat, 8.0f * P.dt * gMagP, P.sideFlags, P.vn0,
                             8.0f * P.dt * gMagP);
      // Arrest budget: 2x the main budget -- the pass must out-pace a violent collapse, and it
      // only ever runs when the residual says one is happening (adaptive stop ends it early).
      for (int it = 0; it < 2 * P.velocityIterations; ++it) {
        Kokkos::deep_copy(P.maxApproach, 0.0f);
        solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                               P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                               P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                               P.lambdaAcc, P.vn0,
                               Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                               P.frictionDynamic);
        if (readFloat(P.maxApproach) <= vRestS)
          break;
      }
    }
  }
  if (usePGS) {  // save the converged force network for next substep's warm start
    commitPairKeysLambdaKokkos(P.pairKeys, P.lambdaAcc, P.lambdaT, P.prevPairKeys, P.prevLambda,
                               P.prevLambdaT, nm);
    P.prevPairCount = nm;
  }
  if (legacyFriction) {
    countFrictionContactsKokkos(P.contacts, nc, P.realIndices, P.planeFriction);
    solveContactFrictionKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                               P.realIndices, P.planeFriction, P.frictionDynamic, P.deltaVel,
                               P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
  }

  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);

  // Colour the contact graph ONCE (topology-only; reused across the position sweeps), then remove
  // overlap with colored Gauss–Seidel (true sequential projection, no count-averaging softening).
  int posLeftover = 0;
  const int numPosColors =
      P.velocityUseGS
          ? colorContactsKokkos(P.contacts, nc, P.numParticles, P.contactColor, P.bodyWinner,
                                P.bodyColorMask, posLeftover)
          : 0;
  // Overlap resolved once the deepest penetration falls below ~0.01% of a particle radius.
  const float posTol = 1e-4f * P.baseRadius * P.globalScale;
  for (int it = 0; it < P.positionIterations; ++it) {
    if (P.velocityUseGS) {
      Kokkos::deep_copy(P.maxOverlap, 0.0f);  // per-sweep so the readback is this sweep's residual
      solvePositionColoredGSKokkos(P.contacts, nc, P.contactColor, numPosColors, P.invMass,
                                   P.posPred, P.quatPred, P.quat, P.invInertia, P.maxOverlap);
      // Colour-mask saturation fallback: contacts the colouring could not place (degree > 62 in
      // crushed regions) get the count-averaged Jacobi projection so deep overlap still resolves.
      if (posLeftover > 0) {
        solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                            P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap,
                            Kokkos::View<const int*, CpMem>(P.contactColor), -1);
        applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel,
                           P.constraintCounts);
      }
      // Adaptive stop: end once no contact overlaps by more than posTol. Fixed positionIterations
      // is the cap.
      if (readFloat(P.maxOverlap) < posTol)
        break;
    } else {
      solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                          P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
      applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel,
                         P.constraintCounts);
    }
  }

  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);

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
                       P.shapeId, P.realIndices, P.topGhost);
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
  detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap, P.sdfGrid,
                       P.materialId, P.pairMaterials);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                       P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount,
                       P.maxOverlap);
  if (P.numWalls > 0)
    detectWallSdfKokkos(P.numReal, P.numWalls, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                        P.shell, P.walls, P.wallGrid, P.globalScale, margin, P.contacts,
                        P.contactCount, P.maxOverlap, P.materialId, P.pairMaterials);
  P.numParticles = P.numReal;
  float h;
  Kokkos::deep_copy(h, P.maxOverlap);
  return h;
}

#ifdef PECLET_DEM_MPI
/// One distributed XPBD DEM substep. The periodic ghost generation of the single-rank step is
/// replaced by a cross-rank gather (halo.gather, ghosts carrying REAL mass), and the owners
/// refresh their ghost copies (velPred/angVelPred, then posPred/quatPred) every `syncEvery`
/// solver iterations (and the last). Each owned particle thus sees all its neighbours -- owned
/// or ghost -- and computes its full serial delta locally; the ghost deltas land on self-mapped
/// slots and are discarded. Friction (wall + body-body Coulomb) IS carried, same kernels as the
/// single-rank step. `forwardRotation`=false (spheres) skips the angular/quaternion forwards.
///
/// SOLVER PARITY GAP (open): this path still runs the older count-averaged JACOBI velocity and
/// position solves (solveVelocityKokkos/solvePositionKokkos). The newer single-rank solver stack
/// -- graph-colored Gauss-Seidel (colorManifoldsKokkos + solveVelocityColoredGSKokkos /
/// solvePositionColoredGSKokkos), warm-started PGS with persistent contacts
/// (prevPairKeys/prevLambda), gravity statics (grounded shock propagation), and the adaptive
/// stop -- is NOT wired in here, and MigratePack does not carry the persistent-contact state
/// across a rebalance. Distributing that stack (rank-local coloring + the syncEvery ghost
/// refresh) is the main remaining dem MPI work item.
///
/// PERIODICITY: cross-rank ghosts supply the wrap on DECOMPOSED axes; LOCAL periodic self-ghosts
/// (ParticleHalo build with includePeriodicSelf) supply it on UNDECOMPOSED periodic axes (a "x1"
/// ORB axis, e.g. z of a 2x2x1 layout, or np=1). Correct for any layout, including np=1 fully
/// periodic (it matches the single-GPU demStep to ~roundoff). CAPACITY: a periodic box needs a
/// thick ghost boundary layer -- a fully periodic box at this rcut needs ~no + (boundary layer)
/// ghost slots, well above the no*2 the closed case wants -- so size the Simulation capacity for
/// the worst-case ghost band; gather() throws on overflow rather than corrupting the SoA.
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
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt, P.extForce);

  // 2. Gather ghosts (real mass) from owners over the halo: full state into the ghost slots; sets
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

  // 3. Broad/narrow phase + manifold reduction over owned + ghosts.
  // findCollisionsGrow fences + reads the pair count back to host and guarantees np ≤ P.pairs extent
  // (growing the buffer on overflow) so the narrowphase never reads P.pairs out of bounds.
  const int np = findCollisionsGrow(P, margin);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  detectContactsKokkos(P.pairs, np, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap, P.sdfGrid,
                       P.materialId, P.pairMaterials);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                       P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount,
                       P.maxOverlap);
  if (P.numWalls > 0)
    detectWallSdfKokkos(P.numReal, P.numWalls, P.posPred, P.quatPred, P.scale, P.shapeId, P.shapes,
                        P.shell, P.walls, P.wallGrid, P.globalScale, margin, P.contacts,
                        P.contactCount, P.maxOverlap, P.materialId, P.pairMaterials);
  const int nc = readInt(P.contactCount);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount);
  const int nm = readInt(P.manifoldCount);

  // 4. Velocity solve: normal restitution + Coulomb friction; refresh ghost velocities every
  // syncEvery iters (+ last). Unlike the original CUDA distributed scheme (pure normal restitution),
  // the friction cluster runs here too, so a frictional/moving wall drives the distributed step —
  // e.g. a rotating drum lifts its bed under MPI exactly as single-rank. WALL friction (boundary
  // contacts, bodyB<0) touches only the OWNED particle, so it is exact across ranks; body-body
  // friction near a rank boundary uses the ghost's LOCAL contact count in the count-average, so it is
  // a close approximation there (bounded by the Coulomb clamp), not bit-exact. The gathered ghosts
  // carry current velPred (forwarded each syncEvery iter), so the force-chain load sees neighbours.
  const bool friction = (P.frictionDynamic > 0.0f || P.wallFrictionMax > 0.0f);
  if (friction)
    computePlaneLoadKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                           P.planeFriction);

  for (int it = 0; it < P.velocityIterations; ++it) {
    if (friction)
      accumulateNormalImpulseKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred,
                                    P.angVelPred, P.realIndices, P.growthRate);
    // Restitution threshold ~ the speed one substep of free fall gains: below it a contact is
    // RESTING and bounces with e=0 (see solveVelocityKokkos — dense-pile energy-bomb guard).
    const float vRest = 2.0f * P.dt *
                        Kokkos::sqrt(P.gravity.x * P.gravity.x + P.gravity.y * P.gravity.y +
                                     P.gravity.z * P.gravity.z);
    solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                        P.realIndices, P.growthRate, P.restitutionNormal, vRest, P.deltaVel,
                        P.deltaAngVel, P.constraintCounts);
    applyVelocityDeltasAveragedKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel,
                                      P.deltaAngVel, P.constraintCounts);
    if ((it + 1) % syncEvery == 0 || it == P.velocityIterations - 1) {
      halo.forward(P.velPred);
      if (forwardRotation)
        halo.forward(P.angVelPred);
    }
  }
  if (friction) {
    countFrictionContactsKokkos(P.contacts, nc, P.realIndices, P.planeFriction);
    solveContactFrictionKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                               P.realIndices, P.planeFriction, P.frictionDynamic, P.deltaVel,
                               P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
    halo.forward(P.velPred);  // publish the friction velocity update to the ghosts
    if (forwardRotation)
      halo.forward(P.angVelPred);
  }

  // 5. Apply velocity & predict position, then refresh ghost predicted positions (+ pose if
  // rotating).
  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);
  halo.forwardPositions(P.posPred);
  if (forwardRotation)
    halo.forward4(P.quatPred);

  // 6. Position solve (Projected Jacobi); refresh ghost predicted pose every syncEvery iters (+
  // last).
  for (int it = 0; it < P.positionIterations; ++it) {
    solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                        P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
    applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel,
                       P.constraintCounts);
    if ((it + 1) % syncEvery == 0 || it == P.positionIterations - 1) {
      halo.forwardPositions(P.posPred);
      if (forwardRotation)
        halo.forward4(P.quatPred);
    }
  }

  // 7. Commit (owned results kept; ghosts discarded, re-gathered next substep).
  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);

  if (P.thermostatTau > 0.0f && P.dt > 0.0f)
    applyThermostatKokkos(P.numReal, P.vel, P.invMass, P.angVel, P.invInertia, P.quat,
                          P.thermostatKB, P.thermostatTau, P.thermostatTemp, P.dt);

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
    halo_.reset();  // the halo also owns Kokkos Views (gather/forward buffers + its core sub-objects')
                    // that must be freed before Kokkos::finalize, else "deallocated after finalize"
                    // aborts. Destroying it via the unique_ptr frees them all.
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
    P_.baseRadius = radius;  // effective radius = scale*globalScale*baseRadius (broadphase + ghost band)
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

    // Upload the shell (resize the View; numPoints==0 => analytic single-probe like the sphere).
    const int nPts = static_cast<int>(shell.size());
    shellPoints_ = nPts;
    if (nPts > 0) {
      P_.shell = Kokkos::View<float* [3], CpMem>("shell", nPts);
      auto hs = Kokkos::create_mirror_view(P_.shell);
      for (int i = 0; i < nPts; ++i) {
        hs(i, 0) = shell[i].x;
        hs(i, 1) = shell[i].y;
        hs(i, 2) = shell[i].z;
      }
      Kokkos::deep_copy(P_.shell, hs);
    }

    auto h = Kokkos::create_mirror_view(P_.shapes);
    h(0) = ShapeDesc{shape_type, params, 0, nPts};
    Kokkos::deep_copy(P_.shapes, h);

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
    defaultInvI_ = F3{ix, iy, iz};
  }

  // Import a general particle as a grid SDF (canonical, unit-scale space) + a surface point shell +
  // its unit-mass principal-frame diagonal inverse inertia. Replaces shape 0, so every particle
  // becomes an instance of this shape (per-particle position/quaternion/scale still apply). `grid`
  // holds nx*ny*nz signed-distance samples, x-fastest (idx = x + y*nx + z*nx*ny), at lattice nodes
  // q = origin + (x,y,z)*spacing (negative inside). `shellFlat` is the flat [nPts*3] surface point
  // set the collision probes body A against. boundingRadius is the canonical radius enclosing the
  // surface (broad-phase + VTI splat bound). invInertia is the per-unit-mass diagonal inverse
  // inertia in the body (principal) frame; like the analytic shapes it becomes the default applied
  // to every particle by setPositions (override afterwards with set_inv_inertia / set_inv_mass for a
  // real density). See peclet.dem.particle_builder for the Python side that produces these arrays
  // from an implicit-solid SDF (marching-cubes shell + voxel-integrated mass properties).
  void setSdfShape(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                   const std::vector<float>& shellFlat, F3 invInertia, float boundingRadius) {
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("setSdfShape: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("setSdfShape: grid.size() must equal nx*ny*nz");
    const int nPts = static_cast<int>(shellFlat.size() / 3);
    if (nPts <= 0)
      throw std::runtime_error("setSdfShape: empty surface point shell");

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
    sd.gridOffset = 0;
    sd.nx = nx;
    sd.ny = ny;
    sd.nz = nz;
    sd.gridOrigin = origin;
    sd.gridInvSpacing = F3{spacing.x > 0 ? 1.0f / spacing.x : 0.0f,
                           spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
                           spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
    auto h = Kokkos::create_mirror_view(P_.shapes);
    h(0) = sd;
    Kokkos::deep_copy(P_.shapes, h);

    baseRadius_ = boundingRadius;
    P_.baseRadius = boundingRadius;
    defaultInvI_ = invInertia;
    shellPoints_ = nPts;
    ensureContactCapacity();
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
  std::tuple<float, float, float> getDomainMin() const {
    return {P_.domain.min.x, P_.domain.min.y, P_.domain.min.z};
  }
  std::tuple<float, float, float> getDomainMax() const {
    return {P_.domain.max.x, P_.domain.max.y, P_.domain.max.z};
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
  /// Enable/disable the one-sided grounded stabilization pass (default on). Off = pure
  /// momentum-conserving PGS everywhere -- exact ballistic response, but deep static columns
  /// mid-collapse cannot be arrested within the iteration budget.
  void setStabilization(bool enabled) { P_.stabilization = enabled; }
  void setGlobalScale(float s) {
    P_.globalScale = s;
    P_.skin = 0.1f * s;
  }
  void setDt(float dt) { P_.dt = dt; }
  // (restitution_normal, restitution_tangent, friction) to match CUDA set_material_params; the
  // Kokkos pipeline currently carries normal restitution + dynamic friction (tangential restitution
  // unused).
  void setMaterialParams(float restitution_normal, float restitution_tangent, float friction) {
    P_.restitutionNormal = restitution_normal;
    P_.frictionDynamic = friction;
    (void)restitution_tangent;
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
  // z*nx*ny), sampled at world nodes origin + (x,y,z)*spacing — POSITIVE in the void where the grains
  // live, NEGATIVE inside the solid wall (so a grain surface point reads the penetration depth and
  // the outward gradient is the push-out normal). restitution/friction are the binary particle–wall
  // material (independent of the body-body material). The wall is motionless but carries a rigid-body
  // surface-velocity field (set via setWallVelocity) so a grain touching it feels the wall's motion.
  // Returns the wall's index (for setWallVelocity). Add walls before stepping.
  int addSdfWall(const std::vector<float>& grid, int nx, int ny, int nz, F3 origin, F3 spacing,
                 float restitution, float friction) {
    if (nx < 2 || ny < 2 || nz < 2)
      throw std::runtime_error("addSdfWall: grid dims must be >= 2 on each axis");
    if (static_cast<long>(nx) * ny * nz != static_cast<long>(grid.size()))
      throw std::runtime_error("addSdfWall: grid.size() must equal nx*ny*nz");

    WallSdf w{};
    w.nx = nx;
    w.ny = ny;
    w.nz = nz;
    w.gridOffset = static_cast<int>(wallGridHost_.size());
    w.origin = origin;
    w.invSpacing = F3{spacing.x > 0 ? 1.0f / spacing.x : 0.0f, spacing.y > 0 ? 1.0f / spacing.y : 0.0f,
                      spacing.z > 0 ? 1.0f / spacing.z : 0.0f};
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

  // Set a wall's rigid-body surface-velocity field v(x) = linVel + angVel × (x − center). A grain in
  // contact feels this velocity even though the geometry never moves: set angVel for a rotating drum
  // (about `center` on the axis), or drive linVel sinusoidally each step for a vibrating wall. Cheap
  // (a few host scalars); safe to call every step.
  void setWallVelocity(int wallIndex, F3 linVel, F3 angVel, F3 center) {
    if (wallIndex < 0 || wallIndex >= static_cast<int>(wallsHost_.size()))
      throw std::runtime_error("setWallVelocity: wall index out of range");
    wallsHost_[wallIndex].linVel = linVel;
    wallsHost_[wallIndex].angVel = angVel;
    wallsHost_[wallIndex].center = center;
    uploadWalls();
  }

  // positions: flat [n*3]; (re)sets the real-particle count and default state.
  void setPositions(const std::vector<float>& xyz) {
    const int n = static_cast<int>(xyz.size() / 3);
    P_.numReal = n;
    P_.numParticles = n;
    auto pos = Kokkos::create_mirror_view(P_.pos);
    auto q = Kokkos::create_mirror_view(P_.quat);
    auto im = Kokkos::create_mirror_view(P_.invMass);
    auto sc = Kokkos::create_mirror_view(P_.scale);
    auto ii = Kokkos::create_mirror_view(P_.invInertia);
    auto sid = Kokkos::create_mirror_view(P_.shapeId);
    auto vel = Kokkos::create_mirror_view(P_.vel);
    auto av = Kokkos::create_mirror_view(P_.angVel);
    for (int i = 0; i < n; ++i) {
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
  // Per-particle external FORCE (fluid drag etc.), an (N,3) flat array. Applied in the next step()'s
  // velocity predict as dv = F*invMass*dt. Persists across steps until re-set or cleared.
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
  // Co-rebalance: migrate ownership onto the weighted ORB of per-cell weights `w` (the SAME partition
  // the coupled flow solver redistributes onto from the same weight field). Returns new owned count.
  int migrateToWeights(const std::vector<peclet::core::Real>& w) {
    return halo_->migrateToWeights(P_, w);
  }
  void stepMpi(int nsteps) {
    const double rcut = (mpiRcut_ > 0.0) ? mpiRcut_ : maxOwnedRadius(P_);
    for (int s = 0; s < nsteps; ++s) {
      if (mpiRebalanceEvery_ > 0 && mpiStepCount_ % mpiRebalanceEvery_ == 0)
        halo_->rebalance(P_);
      demStepMpi(P_, *halo_, rcut, mpiSyncEvery_, mpiForwardRotation_);
      ++mpiStepCount_;
    }
  }
  int rank() const { return halo_->rank(); }
  int numGhost() const { return halo_->numGhost(); }
#endif  // PECLET_DEM_MPI

  // SDF grid (get_sdf_grid): Eikonal reconstruction over the domain, flat x-fastest, negative
  // inside solid.
  std::vector<float> getSdfGrid(int rx, int ry, int rz) {
    return peclet::dem::generateSdfKokkos(
        rx, ry, rz, P_.domain.min, P_.domain.max, P_.numReal, P_.pos, P_.quat, P_.scale, P_.shapeId,
        P_.shapes, P_.domain.periodic_x, P_.domain.periodic_y, P_.domain.periodic_z, P_.sdfGrid);
  }

  int numParticles() const { return P_.numReal; }
  // Live device Views of the owned particle state, for the zero-copy device-array export (H2): the
  // binding wraps a [0,numReal) subview as a DLPack/__cuda_array_interface__ array (or a NumPy view
  // on a host backend) referencing this memory — no device->host copy.
  const V3& positionsView() const { return P_.pos; }
  const V3& velocitiesView() const { return P_.vel; }
  int numContacts() { return readInt(P_.contactCount); }
  int numManifolds() { return readInt(P_.manifoldCount); }
  float maxOverlap() {
    float h;
    Kokkos::deep_copy(h, P_.maxOverlap);
    return h;
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
  // (Re)upload just the small WallSdf array (velocity fields change every step for a vibrating wall;
  // the grid samples are uploaded once in addSdfWall).
  void uploadWalls() {
    const int n = std::max<int>(1, static_cast<int>(wallsHost_.size()));
    if (static_cast<int>(P_.walls.extent(0)) < n)
      P_.walls = Kokkos::View<WallSdf*, CpMem>("walls", n);
    auto h = Kokkos::create_mirror_view(P_.walls);
    for (size_t i = 0; i < wallsHost_.size(); ++i)
      h(i) = wallsHost_[i];
    Kokkos::deep_copy(P_.walls, h);
  }

  // Size the contact/manifold buffers so no contact is dropped: a shell point sits inside at most one
  // neighbour (body-body ~ capacity*shellPoints) plus one per wall it touches (capacity*shellPoints
  // per wall). Boundary/wall contacts are appended AFTER body-body ones, so an undersized buffer
  // silently drops them and grains tunnel through walls. Floored at the analytic default; grows only.
  void ensureContactCapacity() {
    const int perParticle = std::max(16, shellPoints_);
    const long want = static_cast<long>(P_.capacity) * perParticle +
                      static_cast<long>(P_.capacity) * std::max(1, shellPoints_) * P_.numWalls;
    if (want > P_.maxContacts) {
      P_.maxContacts = static_cast<int>(want);
      P_.contacts = Kokkos::View<ContactC*, CpMem>("contacts", want);
      P_.manifolds = Kokkos::View<ManifoldC*, CpMem>("manifolds", want);
    }
  }

  Particles P_;
  float baseRadius_ = 1.0f;
  int shellPoints_ = 0;  // surface-shell size of the active shape (contact-buffer sizing)
  std::vector<WallSdf> wallsHost_;
  std::vector<float> wallGridHost_;
  F3 defaultInvI_{2.5f, 2.5f, 2.5f};
#ifdef PECLET_DEM_MPI
  std::unique_ptr<ParticleHalo> halo_ = std::make_unique<ParticleHalo>();
  double mpiRcut_ = 0.0;
  int mpiSyncEvery_ = 1;
  bool mpiForwardRotation_ = true;
  int mpiRebalanceEvery_ = 0;
  long mpiStepCount_ = 0;
#endif
};

}  // namespace peclet::dem

#endif  // DEM_SIM_HPP
