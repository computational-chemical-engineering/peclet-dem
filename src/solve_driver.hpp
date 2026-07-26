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

#include <algorithm>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <utility>
#include <vector>

#include "broadphase_arborx.hpp"
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

/// gid(i) = base + i over [0, n) — the per-rank global-id re-base of the distributed step
/// (namespace scope: nvcc forbids KOKKOS_LAMBDA in member functions).
inline void fillGidBaseKokkos(Vi gid, int n, int base) {
  Kokkos::parallel_for(
      "peclet::dem::gid_base", Kokkos::RangePolicy<CpExec>(0, n),
      KOKKOS_LAMBDA(int i) { gid(i) = base + i; });
  Kokkos::fence();
}

/// Largest effective particle radius over the owned set (= max scale × globalScale, growth
/// included). The ghost band + broadphase margin are sized off THIS, not globalScale directly, so
/// they scale with the actual grain size — set particles in SI (`radius = 1e-3`, globalScale left
/// at 1) and the halo layer follows automatically. For the usual convention (globalScale ≈ grain
/// size, scale ≈ 1) it is numerically identical to the old `1.0*globalScale`.
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

#ifdef KOKKOS_ENABLE_CUDA
/// Capture-once / replay-N for the iteration loops. Each velocity / stabilization / position
/// iteration re-submits an IDENTICAL sequence of tiny kernels (colour sweeps, coarse cycles) —
/// measured 3,300 launches per step at 25k, ~11 ms of pure host submission, the step's actual
/// bound. A CUDA graph replays the whole iteration as ONE launch; the residual readback (which
/// must stay on the host) runs between replays exactly as it ran between submissions, so the
/// numerics are the submission-path's, bit for bit. Falls back cleanly (returns false) when the
/// stream refuses capture; MPI paths never capture (ghost syncs inside the loop).
struct CudaIterGraph {
  cudaGraphExec_t exec = nullptr;
  template <class F>
  bool capture(CpExec& space, F&& emit, void*& cacheSlot) {
    cudaStream_t str = space.cuda_stream();
    if (cudaStreamBeginCapture(str, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
      (void)cudaGetLastError();
      return false;
    }
    emit();
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture(str, &g) != cudaSuccess || g == nullptr) {
      (void)cudaGetLastError();
      return false;
    }
    // Instantiation dominates the per-step graph cost (~10 us/node); the topology is stable
    // step-to-step, so refresh last step's executable in place and only re-instantiate when the
    // structure genuinely changed (bucket emptiness pattern / colour count shifts).
    if (cacheSlot != nullptr) {
      exec = static_cast<cudaGraphExec_t>(cacheSlot);
      cudaGraphExecUpdateResultInfo ri;
      if (cudaGraphExecUpdate(exec, g, &ri) != cudaSuccess) {
        (void)cudaGetLastError();
        cudaGraphExecDestroy(exec);
        exec = nullptr;
        cacheSlot = nullptr;
      }
    }
    if (exec == nullptr) {
      if (cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        exec = nullptr;
      }
      cacheSlot = exec;
    }
    cudaGraphDestroy(g);
    return exec != nullptr;
  }
  void launch(CpExec& space) { cudaGraphLaunch(exec, space.cuda_stream()); }
  // the executable is owned by the cross-step cache slot, not this per-step handle
};
#define PECLET_DEM_GRAPH_LOOP(useVar, graphVar, emitVar, slotVar)             \
  if constexpr (!Hooks::distributed) {                                        \
    static const bool gOff = std::getenv("PECLET_DEM_NO_GRAPH") != nullptr;   \
    if (!gOff)                                                                \
      useVar = graphVar.capture(space, emitVar, slotVar);                     \
  }
#else
struct CudaIterGraph {
  template <class F>
  bool capture(CpExec&, F&&, void*&) {
    return false;
  }
  void launch(CpExec&) {}
};
#define PECLET_DEM_GRAPH_LOOP(useVar, graphVar, emitVar, slotVar) (void)graphVar;
#endif

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
  // Dense colour buckets for the PGS sweeps (bit-identical: colour classes are body-disjoint).
  std::vector<int> velOffs;
  const bool velBuckets = P.velocityUseGS && nm > 0 && numColors > 0;
  if (velBuckets)
    buildColorBucketsKokkos(Kokkos::View<const int*, CpMem>(P.manifoldColor), nm, numColors,
                            P.velPerm, P.bucketCursor, velOffs);
  const Kokkos::View<const int*, CpMem> velPermC(P.velPerm);
  const std::vector<int>* velOffsP = velBuckets ? &velOffs : nullptr;
  // Fused colour sweeps (CUDA): the whole PGS sweep — and where eligible the whole adaptive
  // iteration loop — as ONE kernel iterating device-side (see solver_fused.hpp); same colour
  // ordering, bit-identical. Default policy: on exactly where CUDA-graph replay is unavailable
  // (the distributed step, PECLET_DEM_NO_GRAPH); PECLET_DEM_FUSED / PECLET_DEM_NO_FUSED force.
  static const bool graphEnvOff = std::getenv("PECLET_DEM_NO_GRAPH") != nullptr;
  const bool wantFused = demFusedWanted(!Hooks::distributed && !graphEnvOff);
  const FusedSweepCtx velFused = (velBuckets && wantFused)
                                     ? demMakeFusedCtx(space, velOffs, P.velOffsDev, P.fusedBar)
                                     : FusedSweepCtx{};
  const FusedSweepCtx* velFusedP = velFused.maxBucket > 0 ? &velFused : nullptr;
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
  // Event-level (Poisson) restitution: per-pair banked compression budget, released as a
  // budget-capped separation-velocity target during unloading (see updateRestitutionBankKokkos /
  // PGSManifoldSweep). Off (empty views) the sweeps run the per-substep Newton path verbatim.
  const bool poisson = usePGS && P.restitutionModel == 1;
  const Kokkos::View<float*, CpMem> bankV = poisson ? P.restBank : Kokkos::View<float*, CpMem>();
  const Kokkos::View<float*, CpMem> relV = poisson ? P.restRel : Kokkos::View<float*, CpMem>();
  const Kokkos::View<const unsigned char*, CpMem> persC =
      poisson ? Kokkos::View<const unsigned char*, CpMem>(P.manifoldPersistent)
              : Kokkos::View<const unsigned char*, CpMem>();
  const Kokkos::View<const float*, CpMem> vpkC =
      poisson ? Kokkos::View<const float*, CpMem>(P.restVPeak) : Kokkos::View<const float*, CpMem>();
  const Kokkos::View<const unsigned char*, CpMem> grdC =
      poisson ? Kokkos::View<const unsigned char*, CpMem>(P.groundedLevel)
              : Kokkos::View<const unsigned char*, CpMem>();
  const Kokkos::View<float*, CpMem> orphV =
      poisson ? P.bodyOrphan : Kokkos::View<float*, CpMem>();
  const Kokkos::View<const float*, CpMem> orphPk =
      poisson ? Kokkos::View<const float*, CpMem>(P.bodyOrphanVPeak)
              : Kokkos::View<const float*, CpMem>();
  // A/B measurement toggles for the Poisson channel (default: Newton alive + symmetric release —
  // the measured-best config on the 25k Dosta impact).
  static const bool restNewtonOff = [] {
    const char* e2 = std::getenv("PECLET_DEM_REST_NEWTON_OFF");
    return e2 && std::atoi(e2) != 0;
  }();
  static const bool restOneSided = [] {
    const char* e2 = std::getenv("PECLET_DEM_REST_ONESIDED");
    return e2 && std::atoi(e2) != 0;
  }();
  if (usePGS) {
    if (poisson && P.prevPairCount > 0) {  // reset the prev-ledger survival flags for the gather
      auto mt = Kokkos::subview(P.prevMatched, Kokkos::pair<int, int>(0, P.prevPairCount));
      Kokkos::deep_copy(mt, static_cast<unsigned char>(0));
    }
    gatherWarmLambdaKokkos(P.manifolds, nm, P.realIndices, keyIdx, P.prevPairKeys, P.prevLambda,
                           P.prevLambdaT, P.prevPosImpulse, P.prevRestBank, P.prevRestVPeak,
                           P.prevPairCount, P.pairKeys, P.lambdaAcc, P.lambdaT, P.posImpulse,
                           P.restBank, P.restVPeak,
                           poisson ? P.prevMatched : Kokkos::View<unsigned char*, CpMem>());
    if (poisson) {
      {  // per-substep release accumulator starts from zero every substep
        auto rr = Kokkos::subview(P.restRel, Kokkos::pair<int, int>(0, nm));
        Kokkos::deep_copy(rr, 0.0f);
      }
      // Orphan transfer: age the body accounts (owned range; MPI ghosts are mirrored), then
      // settle dead pairs' remaining budgets onto their endpoint bodies. Under MPI the pair-key
      // identities are gids, so the scatter resolves them through a sorted gid -> slot map built
      // over owned + ghost slots (a ghost-side credit is overwritten by the next owner mirror —
      // the owner's redundant ledger copy applies the same credit authoritatively).
      decayBodyOrphanKokkos(P.bodyOrphan, P.bodyOrphanVPeak, P.numReal, 2.0f * P.dt * gMagP);
      if (P.prevPairCount > 0) {
        Kokkos::View<const int*, CpMem> gidSorted, slotSorted;
        if constexpr (Hooks::distributed) {
          Kokkos::View<int*, CpMem> gs(
              Kokkos::view_alloc(space, "peclet::dem::orphan_gids", Kokkos::WithoutInitializing),
              nBodies);
          Kokkos::View<int*, CpMem> ss(
              Kokkos::view_alloc(space, "peclet::dem::orphan_slots", Kokkos::WithoutInitializing),
              nBodies);
          auto gid = P.gid;
          Kokkos::parallel_for(
              "peclet::dem::orphan_gid_map", Kokkos::RangePolicy<CpExec>(space, 0, nBodies),
              KOKKOS_LAMBDA(int i) {
                gs(i) = gid(i);
                ss(i) = i;
              });
          Kokkos::Experimental::sort_by_key(space, gs, ss);
          gidSorted = gs;
          slotSorted = ss;
        }
        scatterOrphanBanksKokkos(
            Kokkos::View<const unsigned long long*, CpMem>(P.prevPairKeys),
            Kokkos::View<const float*, CpMem>(P.prevRestBank),
            Kokkos::View<const float*, CpMem>(P.prevRestVPeak),
            Kokkos::View<const unsigned char*, CpMem>(P.prevMatched), P.prevPairCount,
            Kokkos::View<const float*, CpMem>(P.invMass), P.bodyOrphan, P.bodyOrphanVPeak,
            gidSorted, slotSorted);
      }
    }
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
  // Restitution threshold ~ the speed one substep of free fall gains: below it a contact is
  // RESTING and bounces with e=0 (see solveVelocityKokkos — dense-pile energy-bomb guard).
  const float vRest = 2.0f * P.dt * gMagP;
  // One PGS velocity iteration (async residual zero + full colour sweep). Captured as a CUDA
  // graph and replayed per iteration on the single-GPU path: the step is host-submission-bound
  // (measured 3,300 launches / ~11 ms per step at 25k), and replay collapses each iteration's
  // launch storm into one. The residual readback between replays is unchanged, so the adaptive
  // stop — and the physics — are bit-identical to the submission path.
  auto emitVelIter = [&] {
    Kokkos::deep_copy(space, P.maxApproach, 0.0f);
    solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia,
                           P.quat, P.velPred, P.angVelPred, P.realIndices, P.growthRate,
                           P.restitutionNormal, vRest, P.maxApproach, P.lambdaAcc, P.vn0,
                           Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                           P.frictionDynamic, P.vt0, P.restitutionTangent,
                           Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV, persC,
                           vpkC, gHat, grdC, restNewtonOff, restOneSided, orphV, orphPk, velPermC,
                           velOffsP, velFusedP);
  };
  // Device-side iteration loop (CUDA, single-rank): the whole adaptive velocity loop as ONE
  // kernel — same sweeps, same residual, same stop; the per-iteration readback and the graph
  // capture disappear. The final residual stays in P.maxApproach for the stabilization trigger.
  bool velLoopDone = false;
  if constexpr (!Hooks::distributed) {
    if (usePGS && P.velocityUseGS && velLeftover == 0 && velFusedP) {
      const FusedLoopSpec spec{P.velocityIterations, 0.02f * vRest, false};
      velLoopDone = solveVelocityPGSKokkos(
          P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia, P.quat, P.velPred,
          P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRest, P.maxApproach,
          P.lambdaAcc, P.vn0, Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
          P.frictionDynamic, P.vt0, P.restitutionTangent,
          Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV, persC, vpkC, gHat,
          grdC, restNewtonOff, restOneSided, orphV, orphPk, velPermC, velOffsP, velFusedP, &spec);
    }
  }
  bool graphVel = false;
  CudaIterGraph gVel;
  if (!velLoopDone && usePGS && P.velocityUseGS && velLeftover == 0) {
    PECLET_DEM_GRAPH_LOOP(graphVel, gVel, emitVelIter, P.graphCache[0])
  }
  for (int it = 0; !velLoopDone && it < P.velocityIterations; ++it) {
    if (legacyFriction)
      accumulateNormalImpulseKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred,
                                    P.angVelPred, P.realIndices, P.growthRate);
    if (P.velocityUseGS) {
      if (usePGS) {
        if (graphVel)
          gVel.launch(space);
        else
          emitVelIter();
      } else
      {
        Kokkos::deep_copy(space, P.maxApproach, 0.0f);
        solveVelocityColoredGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                     P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                     P.growthRate, P.restitutionNormal, vRest, P.maxApproach);
      }
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
        auto emitOsIter = [&] {
          Kokkos::deep_copy(space, P.maxApproach, 0.0f);
          solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                 P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                 P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                                 P.lambdaAcc, P.vn0,
                                 Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                                 P.frictionDynamic, P.vt0, P.restitutionTangent,
                                 Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV,
                                 persC, vpkC, gHat, grdC, restNewtonOff, restOneSided, orphV,
                                 orphPk, velPermC, velOffsP, velFusedP);
        };
        bool osLoopDone = false;
        if constexpr (!Hooks::distributed) {
          if (velLeftover == 0 && velFusedP) {
            const FusedLoopSpec spec{2 * P.velocityIterations, vRestS, false};
            osLoopDone = solveVelocityPGSKokkos(
                P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia, P.quat,
                P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRestS,
                P.maxApproach, P.lambdaAcc, P.vn0,
                Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                P.frictionDynamic, P.vt0, P.restitutionTangent,
                Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV, persC, vpkC,
                gHat, grdC, restNewtonOff, restOneSided, orphV, orphPk, velPermC, velOffsP,
                velFusedP, &spec);
          }
        }
        bool graphOs = false;
        CudaIterGraph gOs;
        if (!osLoopDone && velLeftover == 0) {
          PECLET_DEM_GRAPH_LOOP(graphOs, gOs, emitOsIter, P.graphCache[1])
        }
        for (int it = 0; !osLoopDone && it < 2 * P.velocityIterations; ++it) {
          if (graphOs)
            gOs.launch(space);
          else
            emitOsIter();
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
        // Dense per-(level, colour) buckets, built once per hierarchy (see solver_multilevel.hpp).
        std::vector<std::vector<int>> mlOffs;
        if (H.numLevels > 0) {
          if (static_cast<int>(P.mlBucketPerm.extent(0)) < H.numLevels * nm)
            Kokkos::realloc(Kokkos::WithoutInitializing, P.mlBucketPerm, H.numLevels * nm);
          buildCoarseBucketsKokkos(H, S, nm, P.levelKey, P.mlBucketPerm, P.bucketCursor, mlOffs);
        }
        // Fused coarse cycle (CUDA): the whole per-iteration coarse leg as ONE kernel.
        const MlFusedCtx mlFused =
            (H.numLevels > 0 && wantFused)
                ? demMakeMlFusedCtx(space, H, mlOffs, nm, nBodies, /*coarseSweeps*/ 2,
                                    P.mlOffsDev, P.fusedBar)
                : MlFusedCtx{};
        const MlFusedCtx* mlFusedP = mlFused.maxWork > 0 ? &mlFused : nullptr;
        // The loop's stop criterion is the QUASI-STATIC residual (fine corrections on contacts
        // with |vn0| <= 4 vRest, plus every coarse correction): the fine sweep's full residual
        // is dominated by ballistic contacts in flowing scenes (a discharging silo never gets
        // below vRest there), and gating on it burns the full budget of extra fine sweeps every
        // substep -- an over-convergence brake on discharge (the escalate effect, measured -7%).
        // The pass exists to converge the quasi-static network; once that is done, it is done.
        // One stabilization iteration (async QS-residual zero + fine sweep + coarse cycle),
        // graph-captured on the single-GPU path — this loop is THE launch storm (up to 16
        // iterations x [colour sweeps + per-(level, colour) coarse kernels] per substep).
        auto emitMlIter = [&] {
          Kokkos::deep_copy(space, P.maxApproachQS, 0.0f);
          solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, P.invMass,
                                 P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                 P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                                 P.lambdaAcc, P.vn0,
                                 Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                                 P.frictionDynamic, P.vt0, P.restitutionTangent,
                                 Kokkos::View<const float*, CpMem>(P.posImpulse),
                                 P.maxApproachQS, bankV, relV, persC, vpkC, gHat, grdC,
                                 restNewtonOff, restOneSided, orphV, orphPk,
                                 velPermC, velOffsP, velFusedP);
          if (H.numLevels > 0)
            multilevelCoarseCycleKokkos(P.manifolds, nm, P.realIndices,
                                        Kokkos::View<const float*, CpMem>(P.invMass), P.velPred,
                                        P.lambdaAcc, P.maxApproachQS, nBodies, H, S,
                                        /*coarseSweeps*/ 2,
                                        Kokkos::View<const float*, CpMem>(relV), &mlOffs,
                                        Kokkos::View<const int*, CpMem>(P.mlBucketPerm), mlFusedP);
        };
        // Device-side stabilization loop (CUDA, single-rank): fine sweep + coarse cycle +
        // adaptive stop, all iterations in ONE kernel (see demFusedMlLoopK).
        bool mlLoopDone = false;
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr (!Hooks::distributed) {
          if (velLeftover == 0 && velFusedP) {
            if (H.numLevels > 0 && mlFusedP) {
              const PGSManifoldSweep fStab = makePGSManifoldSweep(
                  P.manifolds, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                  P.realIndices, P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                  P.maxApproachQS, P.lambdaAcc, P.vn0,
                  Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                  P.frictionDynamic, P.vt0, P.restitutionTangent,
                  Kokkos::View<const float*, CpMem>(P.posImpulse), bankV, relV, persC, vpkC,
                  gHat, grdC, restNewtonOff, restOneSided, orphV, orphPk);
              mlLoopDone = demLaunchFusedMlLoop(
                  space, fStab, velPermC, *velFusedP, numColors, P.manifolds, P.realIndices,
                  Kokkos::View<const float*, CpMem>(P.invMass), P.velPred, P.lambdaAcc,
                  P.maxApproachQS, Kokkos::View<const float*, CpMem>(relV), S,
                  Kokkos::View<const int*, CpMem>(P.mlBucketPerm), *mlFusedP,
                  2 * P.velocityIterations, vRestS);
            } else if (H.numLevels == 0) {
              // aggregation found nothing: the loop is plain fine sweeps on the QS residual
              const FusedLoopSpec spec{2 * P.velocityIterations, vRestS, false};
              mlLoopDone = solveVelocityPGSKokkos(
                  P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia, P.quat,
                  P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal,
                  vRestS, P.maxApproach, P.lambdaAcc, P.vn0,
                  Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                  P.frictionDynamic, P.vt0, P.restitutionTangent,
                  Kokkos::View<const float*, CpMem>(P.posImpulse), P.maxApproachQS, bankV, relV,
                  persC, vpkC, gHat, grdC, restNewtonOff, restOneSided, orphV, orphPk, velPermC,
                  velOffsP, velFusedP, &spec);
            }
          }
        }
#endif
        bool graphMl = false;
        CudaIterGraph gMl;
        if (!mlLoopDone && velLeftover == 0) {
          PECLET_DEM_GRAPH_LOOP(graphMl, gMl, emitMlIter, P.graphCache[2])
        }
        for (int it = 0; !mlLoopDone && it < 2 * P.velocityIterations; ++it) {
          if (graphMl)
            gMl.launch(space);
          else
            emitMlIter();
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
                                     Kokkos::View<const float*, CpMem>(P.posImpulse),
                                     bankV,
                                     relV,
                                     persC,
                                     vpkC,
                                     gHat,
                                     grdC,
                                     restNewtonOff,
                                     restOneSided,
                                     orphV,
                                     orphPk};
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
                                 Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV,
                                 relV, persC, vpkC, gHat, grdC, restNewtonOff,
                                 restOneSided, orphV, orphPk, velPermC, velOffsP, velFusedP);
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
  // Poisson bookkeeping runs once per substep on the FINAL velocity state (after every phase and
  // ghost refresh): bank this substep's kinetic compression, deduct what was returned/released.
  if (poisson)
    updateRestitutionBankKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred,
                                P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal,
                                2.0f * P.dt * gMagP, P.vn0, P.lambdaAcc,
                                Kokkos::View<const float*, CpMem>(P.restRel), P.restBank,
                                P.restVPeak);
  if (usePGS) {  // save the converged force network for next substep's warm start
    commitPairKeysLambdaKokkos(P.pairKeys, P.lambdaAcc, P.lambdaT,
                               Kokkos::View<const float*, CpMem>(P.restBank),
                               Kokkos::View<const float*, CpMem>(P.restVPeak), P.prevPairKeys,
                               P.prevLambda, P.prevLambdaT, P.prevRestBank, P.prevRestVPeak,
                               P.commitPerm, nm);
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
  // Dense colour buckets + fused sweep for the position projection (same precedent as the
  // velocity sweeps: colour classes are body-disjoint => bit-identical; uncoloured leftovers
  // keep the Jacobi fallback below).
  std::vector<int> posOffs;
  const bool posBuckets = P.velocityUseGS && nc > 0 && numPosColors > 0;
  if (posBuckets)
    buildColorBucketsKokkos(Kokkos::View<const int*, CpMem>(P.contactColor), nc, numPosColors,
                            P.posPerm, P.bucketCursor, posOffs);
  const Kokkos::View<const int*, CpMem> posPermC(P.posPerm);
  const std::vector<int>* posOffsP = posBuckets ? &posOffs : nullptr;
  const FusedSweepCtx posFused = (posBuckets && wantFused)
                                     ? demMakeFusedCtx(space, posOffs, P.posOffsDev, P.fusedBar)
                                     : FusedSweepCtx{};
  const FusedSweepCtx* posFusedP = posFused.maxBucket > 0 ? &posFused : nullptr;
  // One position iteration (async residual zero + colored overlap sweep), graph-captured on
  // the single-GPU path like the velocity loops.
  auto emitPosIter = [&] {
    Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
    solvePositionColoredGSKokkos(P.contacts, nc, P.contactColor, numPosColors, P.invMass,
                                 P.posPred, P.quatPred, P.quat, P.invInertia, P.maxOverlap,
                                 P.posLambdaContact, posPermC, posOffsP, posFusedP);
  };
  // Device-side position loop (CUDA, single-rank): all overlap-projection iterations + the
  // adaptive stop in ONE kernel. Leftover contacts (colour-mask saturation) need the host
  // loop's per-iteration Jacobi fallback, so they keep the launch path.
  bool posLoopDone = false;
  if constexpr (!Hooks::distributed) {
    if (P.velocityUseGS && posLeftover == 0 && posFusedP) {
      const FusedLoopSpec spec{P.positionIterations, posTol, true};
      posLoopDone = solvePositionColoredGSKokkos(P.contacts, nc, P.contactColor, numPosColors,
                                                 P.invMass, P.posPred, P.quatPred, P.quat,
                                                 P.invInertia, P.maxOverlap, P.posLambdaContact,
                                                 posPermC, posOffsP, posFusedP, &spec);
    }
  }
  bool graphPos = false;
  CudaIterGraph gPos;
  if (!posLoopDone && P.velocityUseGS && posLeftover == 0) {
    PECLET_DEM_GRAPH_LOOP(graphPos, gPos, emitPosIter, P.graphCache[3])
  }
  for (int it = 0; !posLoopDone && it < P.positionIterations; ++it) {
    if (P.velocityUseGS) {
      if (graphPos)
        gPos.launch(space);
      else
        emitPosIter();
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
