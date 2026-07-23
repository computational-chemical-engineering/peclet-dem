/// @file
/// @brief dem — the shared contact-solve driver: the full modern velocity + position solve
/// sequence (warm-started colored PGS with persistent contacts, gravity statics / stabilization
/// passes, friction, colored-GS overlap projection, adaptive stops), extracted from demStep so the
/// single-GPU and the distributed (MPI) steps run ONE driver instead of drifting copies.
///
/// The two callers differ only through the `Hooks` policy:
///   * SoloSolveHooks (single-GPU): every hook is a no-op / identity — the driver compiles to
///     exactly the pre-extraction demStep sequence (validated bit-for-bit on the Serial backend).
///   * MpiSolveHooks (sim.hpp, PECLET_DEM_MPI): processor-block Gauss–Seidel — the colouring and
///     the sweeps stay rank-local over owned + ghost bodies (ghost pairs are solved redundantly on
///     both owners; ghost deltas are discarded at the next refresh), `syncVelocities` /
///     `syncPositions` refresh the ghost copies owner->ghost every `syncEvery` iterations plus
///     once after every solve phase, and `allMax` turns each adaptive-stop residual into a global
///     MPI_Allreduce(MAX) so all ranks take the same break (a rank-local break would desynchronise
///     the collective ghost refreshes and deadlock).
///
/// `nBodies` is the body-slot span of the solve graph: numReal on the single-GPU path (ghost slots
/// are realIndices-mapped onto their owners), numReal + numGhost under MPI (ghosts are self-mapped
/// slots solved in place). `keyIdx` maps a body slot to the identity used in the persistent-pair
/// keys: realIndices on the single-GPU path, the global particle id under MPI (local slots are not
/// stable across halo rebuilds / migration).
#ifndef DEM_SOLVE_DRIVER_HPP
#define DEM_SOLVE_DRIVER_HPP

#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <utility>
#include <vector>

#include "contact_preprocessing.hpp"
#include "particles.hpp"
#include "solver_friction.hpp"
#include "solver_multilevel.hpp"
#include "solver_position.hpp"
#include "solver_velocity.hpp"

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

/// Single-GPU hooks: no ghost refresh, residuals are already global. Everything inlines away.
struct SoloSolveHooks {
  static constexpr bool distributed = false;
  float allMax(float v) const { return v; }
  bool syncPoint(int) const { return false; }
  void syncVelocities(Particles&) const {}
  void syncPositions(Particles&) const {}
};

/// One full velocity + position contact solve over the already-built contacts/manifolds (see file
/// comment). Runs between the narrow phase and finalCommit; the caller owns ghost construction,
/// broad/narrow phase and the commit.
template <class Hooks>
inline void demSolveContacts(Particles& P, int nc, int nm, int nBodies,
                             Kokkos::View<const int*, CpMem> keyIdx, const Hooks& hooks) {
  CpExec space;

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
      P.velocityUseGS ? colorManifoldsKokkos(P.manifolds, nm, P.realIndices, nBodies,
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
    gatherWarmLambdaKokkos(P.manifolds, nm, P.realIndices, keyIdx, P.prevPairKeys, P.prevLambda,
                           P.prevLambdaT, P.prevPosImpulse, P.prevPairCount, P.pairKeys,
                           P.lambdaAcc, P.lambdaT, P.posImpulse);
    markPersistentManifoldsKokkos(P.manifolds, nm, P.realIndices, keyIdx, P.prevPairKeys,
                                  P.prevPairCount, P.pairKeys, P.manifoldPersistent);
    updateGroundedLevelsKokkos(P.manifolds, nm, P.realIndices, P.posPred, gHat, P.groundedLevel,
                               nBodies, /*sweeps*/ 8, /*decay*/ 8);
    // STAGED SOLVE (Guendelman): the main sweeps are fully momentum-conserving (side flags all
    // zero) -- ballistic impact, discharge and shear see correct physics. One-sided grounding is
    // reserved for the STABILIZATION pass below, which runs only if the main sweeps leave an
    // unconverged residual (a deep column mid-collapse that symmetric GS cannot arrest within the
    // iteration budget).
    {
      auto flags = Kokkos::subview(P.sideFlags, Kokkos::pair<int, int>(0, nm));
      Kokkos::deep_copy(flags, static_cast<unsigned char>(0));
    }
    computeVn0Kokkos(P.manifolds, nm, P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.vn0,
                     P.vt0);
    warmStartApplyKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                         P.realIndices, P.lambdaAcc, P.lambdaT);
    // The warm impulses just moved every owned AND ghost body: re-publish the owners' velocities
    // so the first sweep reads a consistent ghost state.
    if constexpr (Hooks::distributed)
      hooks.syncVelocities(P);
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
                               P.lambdaT, P.frictionDynamic, P.vt0, P.restitutionTangent,
                               Kokkos::View<const float*, CpMem>(P.posImpulse));
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
      // sweep's correction is ~0 and the loop still exits immediately. Distributed: the residual
      // is Allreduce-MAXed so every rank takes the same break (collective-refresh consistency).
      if (hooks.allMax(readFloat(P.maxApproach)) <= (usePGS ? 0.02f * vRest : vRest))
        break;
    } else {
      solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                          P.realIndices, P.growthRate, P.restitutionNormal, vRest, P.deltaVel,
                          P.deltaAngVel, P.constraintCounts);
      applyVelocityDeltasAveragedKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel,
                                        P.deltaAngVel, P.constraintCounts);
    }
    if constexpr (Hooks::distributed) {
      if (hooks.syncPoint(it))
        hooks.syncVelocities(P);
    }
  }
  if constexpr (Hooks::distributed)
    hooks.syncVelocities(P);  // final owner->ghost refresh of the main velocity phase
  // STABILIZATION PASS: if the symmetric sweeps could not drain the residual (a collapsing
  // column needs ~one sweep per layer to carry its weight to the floor -- unaffordable), arrest
  // the remaining quasi-static approach with grounded one-sided sweeps. In dynamic scenes the
  // residual is below the threshold and this pass never runs, so impact/discharge/shear keep
  // pure momentum-conserving physics. (PECLET_DEM_SYMMETRIC_PGS=1 disables the pass -- sandbox
  // A/B toggle.)
  if (usePGS) {
    const float vRestS = 2.0f * P.dt * gMagP;
    const int smode = P.stabilizationMode;
    if (smode != 0 && hooks.allMax(readFloat(P.maxApproach)) > vRestS) {
      if (smode == 1) {  // ONE-SIDED grounded pass (default): held-lower-side impulses
        computeSideFlagsKokkos(P.manifolds, nm, P.realIndices,
                               Kokkos::View<const unsigned char*, CpMem>(P.manifoldPersistent),
                               Kokkos::View<const unsigned char*, CpMem>(P.groundedLevel),
                               P.posPred, P.velPred, gHat, 8.0f * P.dt * gMagP, P.sideFlags,
                               P.vn0, 8.0f * P.dt * gMagP);
        // Arrest budget: 2x the main budget -- the pass must out-pace a violent collapse, and it
        // only ever runs when the residual says one is happening (adaptive stop ends it early).
        for (int it = 0; it < 2 * P.velocityIterations; ++it) {
          Kokkos::deep_copy(P.maxApproach, 0.0f);
          solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                 P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                 P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                                 P.lambdaAcc, P.vn0,
                                 Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                                 P.frictionDynamic, P.vt0, P.restitutionTangent,
                                 Kokkos::View<const float*, CpMem>(P.posImpulse));
          if (hooks.allMax(readFloat(P.maxApproach)) <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              hooks.syncVelocities(P);
          }
        }
      } else if (smode == 2) {
        // MULTILEVEL (GraphMG) pass: never deletes momentum, only accelerates its transport.
        // Greedy pairwise aggregation over the quasi-static contact graph builds super-bodies
        // (summed mass, momentum-weighted velocity); the fine manifolds crossing aggregate
        // boundaries are re-solved with the AGGREGATE masses -- the supported chain's genuinely
        // huge inertia plays the role the held lower side faked, so a wall contact drains a
        // whole column's momentum in one coarse impulse while every impulse stays symmetric.
        // Ballistic pairs (|vn0| > qsThr) never aggregate: an impactor keeps its fine-level,
        // momentum-conserving physics and its rebound. Coarse lambda shares the fine
        // accumulator, so the force-network ledger stays consistent for next substep's warm
        // start and the friction cone's Coulomb bound. See solver_multilevel.hpp. Distributed:
        // the hierarchy is built rank-locally over owned + ghost bodies (aggregates never cross
        // a rank boundary beyond the ghost band; the syncEvery refresh reconciles).
        const float qsThr = 8.0f * P.dt * gMagP;
        // Eligibility gates (mldetail::kGate*): slip is the production default -- it keeps the
        // pass off sustained shear (silo bulk) without starving a crushing bed's aggregation.
        // PECLET_DEM_ML_GATES overrides the mask for A/B measurement.
        static const int mlGates = [] {
          const char* e = std::getenv("PECLET_DEM_ML_GATES");
          return e ? std::atoi(e) : mldetail::kGateSlip;
        }();
        MlScratch S{P.mlColorPacked, P.mlParent, P.mlInvMassG, P.mlVelG,
                    P.mlVelG0,       P.mlMassG,  P.mlGrp,      P.mlMate};
        const ContactHierarchy H = buildContactHierarchyKokkos(
            P.manifolds, nm, P.realIndices, Kokkos::View<const int*, CpMem>(P.manifoldColor),
            Kokkos::View<const float*, CpMem>(P.vn0),
            Kokkos::View<const float* [3], CpMem>(P.vt0),
            Kokkos::View<const unsigned char*, CpMem>(P.manifoldPersistent), P.posPred, gHat,
            Kokkos::View<const float*, CpMem>(P.invMass), qsThr, mlGates, nBodies, S,
            P.bodyWinner, P.bodyColorMask);
        // The loop's stop criterion is the QUASI-STATIC residual (fine corrections on contacts
        // with |vn0| <= 4 vRest, plus every coarse correction): the fine sweep's full residual
        // is dominated by ballistic contacts in flowing scenes (a discharging silo never gets
        // below vRest there), and gating on it burns the full budget of extra fine sweeps every
        // substep -- an over-convergence brake on discharge (the escalate effect, measured -7%).
        // The pass exists to converge the quasi-static network; once that is done, it is done.
        for (int it = 0; it < 2 * P.velocityIterations; ++it) {
          Kokkos::deep_copy(P.maxApproachQS, 0.0f);
          // fine smoothing sweep (full physics: friction cone, restitution gates)
          solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                 P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                 P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                                 P.lambdaAcc, P.vn0,
                                 Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                                 P.frictionDynamic, P.vt0, P.restitutionTangent,
                                 Kokkos::View<const float*, CpMem>(P.posImpulse),
                                 P.maxApproachQS);
          // coarse leg: fine -> coarse, translation-only inelastic PGS at aggregate masses
          if (H.numLevels > 0)
            multilevelCoarseCycleKokkos(P.manifolds, nm, P.realIndices,
                                        Kokkos::View<const float*, CpMem>(P.invMass), P.velPred,
                                        P.lambdaAcc, P.maxApproachQS, nBodies, H, S,
                                        /*coarseSweeps*/ 2);
          if (hooks.allMax(readFloat(P.maxApproachQS)) <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              hooks.syncVelocities(P);
          }
        }
      } else if (smode == 4) {
        // ORDERED (level-ordered symmetric sweeps; measurement mode): fresh height-from-floor
        // BFS levels order the manifolds bottom-up + top-down. Fully symmetric, but a pairwise
        // inelastic impulse only EQUALIZES velocities, so a deep column still cools one halving
        // per cycle -- measured insufficient on the statics battery (kept for A/B comparison
        // against the multilevel pass).
        computeHeightLevelsKokkos(P.manifolds, nm, P.realIndices, P.posPred, gHat, P.heightLevel,
                                  nBodies);
        std::vector<std::pair<int, int>> buckets;
        buildLevelColorBucketsKokkos(P.manifolds, nm, P.realIndices,
                                     Kokkos::View<const int*, CpMem>(P.manifoldColor),
                                     Kokkos::View<const int*, CpMem>(P.heightLevel), P.levelKey,
                                     P.levelPerm, buckets);
        const PGSManifoldSweep sweep{P.manifolds,
                                     P.invMass,
                                     P.invInertia,
                                     P.quat,
                                     P.velPred,
                                     P.angVelPred,
                                     P.realIndices,
                                     P.growthRate,
                                     P.restitutionNormal,
                                     vRestS,
                                     P.maxApproach,
                                     P.maxApproach,
                                     P.lambdaAcc,
                                     P.vn0,
                                     Kokkos::View<const unsigned char*, CpMem>(P.sideFlags),
                                     P.lambdaT,
                                     P.frictionDynamic,
                                     P.vt0,
                                     P.restitutionTangent,
                                     Kokkos::View<const float*, CpMem>(P.posImpulse)};
        for (int it = 0; it < 2 * P.velocityIterations; ++it) {
          Kokkos::deep_copy(P.maxApproach, 0.0f);
          solveVelocityPGSBucketsKokkos(sweep, Kokkos::View<const int*, CpMem>(P.levelPerm),
                                        buckets, /*topDown*/ false);
          solveVelocityPGSBucketsKokkos(sweep, Kokkos::View<const int*, CpMem>(P.levelPerm),
                                        buckets, /*topDown*/ true);
          if (hooks.allMax(readFloat(P.maxApproach)) <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              hooks.syncVelocities(P);
          }
        }
      } else if (smode == 3) {
        // ESCALATION (diagnostic/fallback): keep running plain symmetric colored sweeps until
        // the residual drains or the 256-sweep cap -- provably correct physics, and the sweep
        // count it needs bounds what the ordered pass must deliver.
        for (int it = 0; it < 256; ++it) {
          Kokkos::deep_copy(P.maxApproach, 0.0f);
          solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                 P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                 P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                                 P.lambdaAcc, P.vn0,
                                 Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                                 P.frictionDynamic, P.vt0, P.restitutionTangent,
                                 Kokkos::View<const float*, CpMem>(P.posImpulse));
          if (hooks.allMax(readFloat(P.maxApproach)) <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              hooks.syncVelocities(P);
          }
        }
      }
      if constexpr (Hooks::distributed)
        hooks.syncVelocities(P);  // final refresh of the stabilization phase
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
    if constexpr (Hooks::distributed)
      hooks.syncVelocities(P);  // publish the friction velocity update to the ghosts
  }

  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);
  if constexpr (Hooks::distributed)
    hooks.syncPositions(P);

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
  {
    auto pc = Kokkos::subview(P.posLambdaContact, Kokkos::pair<int, int>(0, nc));
    Kokkos::deep_copy(pc, 0.0f);
  }
  for (int it = 0; it < P.positionIterations; ++it) {
    if (P.velocityUseGS) {
      Kokkos::deep_copy(P.maxOverlap, 0.0f);  // per-sweep so the readback is this sweep's residual
      solvePositionColoredGSKokkos(P.contacts, nc, P.contactColor, numPosColors, P.invMass,
                                   P.posPred, P.quatPred, P.quat, P.invInertia, P.maxOverlap,
                                   P.posLambdaContact);
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
      // is the cap. Distributed: Allreduce-MAXed so all ranks break together.
      if (hooks.allMax(readFloat(P.maxOverlap)) < posTol)
        break;
    } else {
      solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                          P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
      applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel,
                         P.constraintCounts);
    }
    if constexpr (Hooks::distributed) {
      if (hooks.syncPoint(it))
        hooks.syncPositions(P);
    }
  }
  if constexpr (Hooks::distributed)
    hooks.syncPositions(P);  // final owner->ghost refresh of the position phase
  // Position-channel Coulomb-bound carry (PGS path): next substep's friction cone sees
  // mu * (velocity-impulse channel + this position-channel load). Without it a jostled bed's
  // bound under-counts the true normal force and stick leaks (measured: 99% sliding wall
  // contacts in the benchmark drum while the Hertz reference sticks).
  if (usePGS && nm > 0)
    commitPosImpulseKokkos(P.posLambdaContact, nc, P.contactSlot, P.manifolds, nm, P.pairKeys,
                           P.prevPairKeys, P.prevPairCount, P.dt, P.vn0, P.prevPosImpulse);
  (void)space;
}

}  // namespace peclet::dem

#endif  // DEM_SOLVE_DRIVER_HPP
