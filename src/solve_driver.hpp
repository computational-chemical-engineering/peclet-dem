/// @file
/// @brief dem — the shared contact-solve driver: the full modern velocity + position solve
/// sequence (warm-started colored PGS with persistent contacts, gravity statics / stabilization
/// passes, friction, colored-GS overlap projection, adaptive stops), extracted from demStep so the
/// single-GPU and the distributed (MPI) steps run ONE driver instead of drifting copies.
///
/// The two callers differ only through the `Hooks` policy:
///   * SoloSolveHooks (single-GPU): every hook is a no-op / identity — the driver compiles to
///     exactly the pre-extraction demStep sequence (validated bit-for-bit on the Serial backend).
///   * MpiSolveHooks (step_solve_mpi.hpp, PECLET_DEM_MPI): processor-block Gauss–Seidel with a
///     SINGLE owner per contact (docs/mpi_momentum_conservation.md) — `nc` / `nm` are the counts
///     of the contacts / manifolds this rank owns, and only they are swept, over owned + ghost
///     body slots; the partner half of every impulse lands in the ghost slot. `beginSolve` marks
///     the ghosts' velocity baselines; `syncVelocities` / `syncPositions` reverse-accumulate each
///     ghost's change since its baseline onto its owner, then refresh owner->ghost (reverse, then
///     forward), every `syncEvery` iterations plus once after every solve phase;
///     `publishPositions` opens the position phase with a forward only (the integration of a
///     ghost is not an interaction); `syncFrictionCounts` makes the legacy-friction counts and
///     `syncContactCounts` the Jacobi count-averaging counts the serial ones.
///     `visibleManifolds(nm)` is the whole visible range, read only by the label passes
///     (warm-ledger match, grounded / height levels). `allMax` turns each adaptive-stop residual
///     into a global MPI_Allreduce(MAX) so all ranks take the same break (a rank-local break
///     would desynchronise the collective syncs and deadlock); `allMaxAny` folds a rank-local
///     vote (the colouring-invariant check, §12 S3) into that same Allreduce.
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "broadphase_arborx.hpp"
#include "contact_preprocessing.hpp"
#include "particles.hpp"
#include "solve_copies.hpp"
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

/// World radii rad(i) = scale(i) * globalScale * baseRadius over [0, n): the owned span before a
/// step, owned + ghosts after periodic ghost generation or a halo gather. Every step driver (XPBD,
/// overlap probe, force-based, their distributed forms) fills the radii through this one kernel.
/// Fences the default instance: the callers' next launch on that instance is stream-ordered
/// anyway, so the fence changes nothing about the result.
inline void fillWorldRadiiKokkos(Vf scale, Vf rad, float gs, float bR, int n) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::world_radii", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { rad(i) = scale(i) * gs * bR; });
  space.fence();
}

/// Broad phase with an automatically-grown pair buffer.
///
/// `findCollisionsArborX` guards its pair WRITES at `maxPairs` but returns the RAW candidate count,
/// which can exceed `P.pairs`' capacity once a bed compacts (more neighbour pairs than the buffer
/// holds — e.g. a fluidized bed driven denser by the CFD-DEM drag). Feeding that raw count straight
/// into `detectContactsKokkos` as its loop bound makes the narrowphase read `P.pairs` out of bounds
/// → `cudaErrorIllegalAddress`. Here we detect the overflow, reallocate `P.pairs` (with headroom so
/// an oscillating count doesn't realloc every step) and re-run once so no candidate pair is
/// silently dropped, then clamp defensively so the returned count is ALWAYS ≤ the buffer extent —
/// the narrowphase can never walk off the end regardless.
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

/// Verlet-cached impulse broadphase (single-GPU, non-periodic). Reuse the last ArborX candidate
/// list while no particle's predicted position has moved more than skin/2 since the build; the list
/// is built at `margin + skin`, so every within-`margin` pair is still present between rebuilds — a
/// SUPERSET, so the narrowphase yields identical contacts (order differs -> run-scatter, like the
/// colouring/fused paths). Rebuilds on an invalid cache, growth beyond the skin, or the
/// displacement bound. Composes with sleeping: a frozen bed has zero displacement, so it never
/// rebuilds. The caller guarantees non-periodic (no per-step ghost-slot churn) and verletSkinFrac >
/// 0.
inline int findCollisionsVerlet(Particles& P, float margin, float maxRad) {
  const float skin = P.verletSkinFrac * maxRad;
  bool rebuild = (P.impNumPairs < 0);
  if (!rebuild) {
    float md2 = 0.0f;
    auto pp = P.posPred;
    auto rp = P.impRefPos;
    Kokkos::parallel_reduce(
        "peclet::dem::verlet_disp", Kokkos::RangePolicy<CpExec>(0, P.numReal),
        KOKKOS_LAMBDA(int i, float& acc) {
          const float dx = pp(i, 0) - rp(i, 0), dy = pp(i, 1) - rp(i, 1), dz = pp(i, 2) - rp(i, 2);
          const float d = dx * dx + dy * dy + dz * dz;
          if (d > acc)
            acc = d;
        },
        Kokkos::Max<float>(md2));
    Kokkos::fence();
    // Growth grows radii between rebuilds; a pair can close by ~2x the radius growth with no CoM
    // motion, so fold it into the displacement budget.
    const float grow = std::max(0.0f, maxRad - P.impRefMaxRad);
    if (std::sqrt(md2) + 2.0f * grow > 0.5f * skin)
      rebuild = true;
  }
  if (rebuild) {
    const int np = findCollisionsGrow(P, margin + skin);  // candidate list at margin + skin
    const auto rng = Kokkos::pair<int, int>(0, P.numReal);
    Kokkos::deep_copy(Kokkos::subview(P.impRefPos, rng, Kokkos::ALL),
                      Kokkos::subview(P.posPred, rng, Kokkos::ALL));
    P.impNumPairs = np;
    P.impRefMaxRad = maxRad;
    return np;
  }
  return P.impNumPairs;  // reuse the cached candidate list (P.pairs is unchanged)
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
/// Capture `emitVar`'s launches as a CUDA graph into `slotVar` (single-rank only). `P` is the
/// enclosing scope's Particles: P.cudaGraphs is the set_cuda_graphs() policy.
#define PECLET_DEM_GRAPH_LOOP(useVar, graphVar, emitVar, slotVar) \
  if constexpr (!Hooks::distributed) {                            \
    if (P.cudaGraphs)                                             \
      useVar = graphVar.capture(space, emitVar, slotVar);         \
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

/// Rank-level M (docs/contact_solve_framework.md §13.3) as the hooks expose it to the driver:
/// whether this rank reconciles copies with neighbours at all (it sends or receives ghosts). The
/// per-slot counts themselves live in Particles (aVel / aPos from the driver's activity pass,
/// kVel / kPos from the halo's openings) and feed the solve-view builder
/// (buildSolveViewsRankKKokkos) and the coarse vertex masses.
struct RankK {
  bool exchanges = false;
  // Rank-level X (§1.4, WO-6): this rank's colour col(r), the number of colours C of the rank
  // graph, and the sync interval (solver iterations per sync) that the holder cycle steps with.
  int color = 0;
  int numColors = 1;
  int syncInterval = 1;
};

/// Single-GPU hooks: no ghost refresh, residuals are already global. Everything inlines away.
struct SoloSolveHooks {
  static constexpr bool distributed = false;
  float allMax(float v) const { return v; }
  float allMaxAny(float v, bool&) const { return v; }
  bool syncPoint(int) const { return false; }
  int visibleManifolds(int nm) const { return nm; }
  void beginSolve(Particles&) const {}
  void syncVelocities(Particles&) const {}
  void publishPositions(Particles&) const {}
  void syncPositions(Particles&) const {}
  void syncFrictionCounts(Particles&) const {}
  void syncContactCounts(Particles&) const {}
  // Rank-level M (docs/contact_solve_framework.md §13.3): a single rank has no remote copies, so
  // the opening is "step 0" only -- k = max(1, a(own)) = WO-4's groupK, which the driver uses
  // directly -- and there is no rank-level k to expose.
  RankK rankK() const { return {}; }
  void openVelocityPhase(Particles&, bool) const {}
  void openPositionCounts(Particles&, bool) const {}
  void restoreOrphanBalance(Particles&) const {}
};

/// Diagnostics (docs/contact_solve_framework.md §12 S12): the device scalar a fused main loop
/// writes its iteration count to (slot 0 velocity, 1 position), allocated on first use.
inline int* demIterCountSlot(Particles& P, int slot) {
  if (P.iterCountDev.extent(0) < 2)
    P.iterCountDev = Kokkos::View<int*, CpMem>("peclet::dem::iter_count", 2);
  return P.iterCountDev.data() + slot;
}
/// The count a fused main loop wrote (one small readback, only with P.iterCounters on); -1 with
/// the counters off (no fence, no copy).
inline int demReadIterCount(const Particles& P, int slot) {
  if (!P.iterCounters || P.iterCountDev.extent(0) < 2)
    return -1;
  int v = -1;
  Kokkos::deep_copy(v, Kokkos::subview(P.iterCountDev, slot));
  return v;
}

/// One full velocity + position contact solve over the already-built contacts/manifolds (see file
/// comment). Runs between the narrow phase and finalCommit; the caller owns ghost construction,
/// broad/narrow phase and the commit.
template <class Hooks>
inline void demSolveContacts(Particles& P, int nc, int nm, int nBodies,
                             Kokkos::View<const int*, CpMem> keyIdx, const Hooks& hooks) {
  CpExec space;
  hooks.beginSolve(P);
  // Rank-level X (§1.4): identical on every rank; each substep starts the holder cycle one step
  // later, so no rank is systematically first. Incremented on every rank and every call (np 1
  // included, where it is read by nothing).
  ++P.solveEpoch;
  Kokkos::Profiling::pushRegion("dem::solve::prep");
  const int nmVisible = hooks.visibleManifolds(nm);

  // A frictional wall drives friction even when the body-body material is frictionless.
  const bool friction = (P.frictionDynamic > 0.0f || P.wallFrictionMax > 0.0f);
  const bool usePersistPre = (P.gravity.x != 0.0f || P.gravity.y != 0.0f || P.gravity.z != 0.0f);
  // Incremental (warm-started) colouring (set_incremental_coloring, default on; single-GPU PGS
  // path). Off forces the full per-substep recolour — the pre-incremental behaviour, kept for
  // validation. The colouring decides the Gauss-Seidel sweep order, so this DOES change results.
  const bool incrColorOff = !P.incrementalColoring;
  // Island sleeping (single-GPU statics, gravity on, no external drag): both-asleep manifolds /
  // contacts are excluded from the colouring / sweeps / multilevel hierarchy (their masks were
  // filled by the caller). Empty views leave every colouring bit-identical to the sleeping-off
  // path. The caller has already swapped P.invMass to the effective (sleeper -> 0) inverse mass.
  const bool sleepOn = P.sleepingEnabled && !Hooks::distributed && usePersistPre &&
                       !P.extForceActive && !P.extTorqueActive;
  const Kokkos::View<const unsigned char*, CpMem> mSleep =
      sleepOn ? Kokkos::View<const unsigned char*, CpMem>(P.manifoldSleep)
              : Kokkos::View<const unsigned char*, CpMem>();
  const Kokkos::View<const unsigned char*, CpMem> cSleep =
      sleepOn ? Kokkos::View<const unsigned char*, CpMem>(P.contactSleep)
              : Kokkos::View<const unsigned char*, CpMem>();
  const bool legacyFriction = friction && !(usePersistPre && P.velocityUseGS);
  if (legacyFriction)
    computePlaneLoadKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                           P.planeFriction);

  // Colour the manifold graph ONCE (topology-only; reused across the sweeps), then normal
  // restitution as colored Gauss–Seidel: correct multi-contact dissipation with no count-averaging
  // (see solver_velocity.hpp). count==1 binary collisions are identical to the old Jacobi path.
  int velLeftover = 0;
  int numColors = 0;
  // Complete colourings (docs/contact_solve_framework.md §4.2 item 4): colour each phase graph
  // once; if an edge found no free colour, split every vertex above kHubEdgeBudget edges into hub
  // copies (§4.4) and recolour that phase from scratch on the copy vertices -- guaranteed to
  // succeed (§1.6). Anything still uncoloured violates the colouring invariant (§12 S3): a single
  // rank throws here, the distributed step votes it in its first stop Allreduce and every rank
  // throws. Copy slots are appended at [slotBase, slotBase + nCopies); both phases reuse the range.
  PhaseCopies& VC = P.velCopies;
  PhaseCopies& PC = P.posCopies;
  VC.nHubs = VC.nCopies = VC.nGroups = 0;
  PC.nHubs = PC.nCopies = PC.nGroups = 0;
  const int slotBase = P.numParticles;
  VC.slotBase = PC.slotBase = slotBase;
  {  // mlHubAggregated is the max over the step call's substeps (reset by the step entry points)
    const int mlAgg = P.splitStats.mlHubAggregated;
    const long long drift = P.splitStats.driftMigrations;  // cumulative (the drift vote, §5.1)
    P.splitStats = SplitStats{};
    P.splitStats.mlHubAggregated = mlAgg;
    P.splitStats.driftMigrations = drift;
  }
  P.mlLast.numLevels = 0;
  Kokkos::View<int*, CpMem> edgeA, edgeB;  // a failed phase's colouring edges (scratch)
  auto growSlots = [&](int nCopies) {
    if (slotBase + nCopies > P.capacity)
      P.ensureCapacity(slotBase + nCopies + 64);
  };
  std::string invariantMsg;
  auto noteInvariant = [&](const char* phase, Kokkos::View<const int*, CpMem> color, int n) {
    // Name the first uncolourable edge's vertices and their degrees (host readback, error path).
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), color);
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), edgeA);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), edgeB);
    auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.vertexDegree);
    int bad = 0, first = -1;
    for (int e = 0; e < n; ++e)
      if (hc(e) == -1 || hc(e) == kColorUncolourable) {
        ++bad;
        if (first < 0)
          first = e;
      }
    std::string m = std::string("colouring invariant violated: ") + std::to_string(bad) + " " +
                    phase + " uncolourable edges after hub splitting";
    if (first >= 0) {
      m += " (first: vertex " + std::to_string(ha(first)) + ", degree " +
           std::to_string(ha(first) >= 0 ? hd(ha(first)) : 0);
      if (hb(first) >= 0)
        m += "; vertex " + std::to_string(hb(first)) + ", degree " + std::to_string(hd(hb(first)));
      m += ")";
    }
    invariantMsg += (invariantMsg.empty() ? "" : "; ") + m;
  };
  if (P.velocityUseGS) {
    // Incremental colouring (single-GPU PGS path only): carry surviving pairs' colours across
    // substeps, re-arbitrating only the new manifolds. Bit-identical colouring FORM to the full
    // recolour; forced full on the fresh substep and when the colour count creeps > 1.3x the last
    // full recolour (compaction keeps the serialized-phase count bounded). Under MPI or g == 0 the
    // full recolour stays (a carried colour can cross a rank boundary it never arbitrated against).
    const bool incrColor = usePersistPre && P.velocityUseGS && !Hooks::distributed && !incrColorOff;
    if (incrColor) {
      // A substep coloured through hub copies carried per-COPY colours: recolour in full.
      bool didFull = (P.prevPairCount <= 0) || P.velCopiesLastSubstep;
      numColors = colorManifoldsIncrementalKokkos(
          P.manifolds, nm, P.realIndices, nBodies,
          Kokkos::View<const unsigned long long*, CpMem>(P.prevPairKeys),
          Kokkos::View<const int*, CpMem>(P.prevManifoldColor), P.prevPairCount, P.manifoldColor,
          P.bodyWinner, P.bodyColorMask, velLeftover, /*forceFull*/ didFull, mSleep);
      if (!didFull && P.velLastFullColors > 0 && numColors > (P.velLastFullColors * 13) / 10) {
        numColors = colorManifoldsIncrementalKokkos(
            P.manifolds, nm, P.realIndices, nBodies,
            Kokkos::View<const unsigned long long*, CpMem>(P.prevPairKeys),
            Kokkos::View<const int*, CpMem>(P.prevManifoldColor), P.prevPairCount, P.manifoldColor,
            P.bodyWinner, P.bodyColorMask, velLeftover, /*forceFull*/ true, mSleep);
        didFull = true;
      }
      if (didFull)
        P.velLastFullColors = numColors;
    } else {
      numColors = colorManifoldsKokkos(P.manifolds, nm, P.realIndices, nBodies, P.manifoldColor,
                                       P.bodyWinner, P.bodyColorMask, velLeftover, mSleep, {},
                                       /*dedupTwins*/ !Hooks::distributed);
    }
    if (velLeftover > 0 && nm > 0) {
      // Attempt failed: hub copies over the vertices realIdx(body) (§4.4), full recolour.
      edgeA = Kokkos::View<int*, CpMem>("peclet::dem::copies_eA", nm);
      edgeB = Kokkos::View<int*, CpMem>("peclet::dem::copies_eB", nm);
      velocityEdgeEndsKokkos(P.manifolds, nm, P.realIndices,
                             Kokkos::View<const int*, CpMem>(P.manifoldColor), edgeA, edgeB);
      growCopyView(P.vertexDegree, static_cast<std::size_t>(nBodies), "peclet::dem::vertexDegree");
      vertexDegreeKokkos(edgeA, edgeB, nm, nBodies, P.vertexDegree);
      buildHubCopiesKokkos(edgeA, edgeB, nm, nBodies, slotBase, P.vertexDegree, VC);
      growSlots(VC.nCopies);
      const SlotOverride ovColor{VC.slotA, VC.slotB, {}, 1.0f};
      numColors = colorManifoldsKokkos(P.manifolds, nm, P.realIndices, nBodies, P.manifoldColor,
                                       P.bodyWinner, P.bodyColorMask, velLeftover, mSleep, ovColor,
                                       /*dedupTwins*/ !Hooks::distributed, slotBase + VC.nCopies);
      if (incrColor)
        P.velLastFullColors = numColors;
      buildCopyGroupsKokkos(P.realIndices, nBodies, P.numReal, P.vertexDegree, {}, VC,
                            /*identityBodies*/ true);
      P.splitStats.lightHubs = countLightHubsKokkos(VC, edgeA, edgeB, nm, nBodies,
                                                    Kokkos::View<const float*, CpMem>(P.invMass));
      if (velLeftover > 0)
        noteInvariant("velocity", Kokkos::View<const int*, CpMem>(P.manifoldColor), nm);
    }
    P.velCopiesLastSubstep = VC.nCopies > 0;
  }
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
  // ordering, bit-identical. Default policy ("auto"): on exactly where CUDA-graph replay is
  // unavailable — the distributed step, or set_cuda_graphs(False); set_fused_sweeps("on"/"off")
  // forces either way.
  const bool wantFused = demFusedWanted(!Hooks::distributed && P.cudaGraphs, P.fusedSweeps);
  const FusedSweepCtx velFused = (velBuckets && wantFused)
                                     ? demMakeFusedCtx(space, velOffs, P.velOffsDev, P.fusedBar)
                                     : FusedSweepCtx{};
  const FusedSweepCtx* velFusedP = velFused.maxBucket > 0 ? &velFused : nullptr;
  // Position colouring, moved here from after the velocity phase (docs/contact_solve_framework.md
  // §4.2 item 3): both phases are coloured at the top of the solve. Its inputs (contacts, sleep
  // masks, the previous substep's ledger) do not change in between, so the move is byte-identical.
  // The overlap projection is coloured and swept per position UNIT (§4.3): all contacts of one
  // body pair (or of one body and one wall, §12 S6) in one work item, in ascending contact order.
  // A one-point unit is exactly the old per-contact edge, key and arithmetic.
  int posLeftover = 0;
  int numPosColors = 0;
  int numPosUnits = 0;
  PosUnits posUnits;
  P.numPosUnits = 0;
  // Incremental position colouring, same single-GPU PGS gate + creep policy as the velocity path.
  const bool incrPosColor =
      usePersistPre && P.velocityUseGS && !Hooks::distributed && !incrColorOff;
  bool posDidFull = false;
  if (P.velocityUseGS) {
    numPosUnits = buildPositionUnitsKokkos(P.contacts, nc, P.unitStart, P.unitContacts, nm);
    P.numPosUnits = numPosUnits;
    // All singletons (spheres): the unit map is the identity -- the empty PosUnits, so the
    // sweeps and passes index contacts directly (no two-level indirection per contact).
    if (numPosUnits != nc)
      posUnits = PosUnits{Kokkos::subview(Kokkos::View<const int*, CpMem>(P.unitStart),
                                          Kokkos::pair<int, int>(0, numPosUnits + 1)),
                          Kokkos::subview(Kokkos::View<const int*, CpMem>(P.unitContacts),
                                          Kokkos::pair<int, int>(0, nc))};
    if (incrPosColor) {
      posDidFull = (P.posPrevContactCount <= 0);
      numPosColors = colorContactsIncrementalKokkos(
          P.contacts, numPosUnits, P.numParticles,
          Kokkos::View<const unsigned long long*, CpMem>(P.prevContactKeys),
          Kokkos::View<const int*, CpMem>(P.prevContactColor), P.posPrevContactCount, P.unitColor,
          P.contactKeys, P.bodyWinner, P.bodyColorMask, posLeftover,
          /*forceFull*/ posDidFull, cSleep, posUnits);
      if (!posDidFull && P.posLastFullColors > 0 &&
          numPosColors > (P.posLastFullColors * 13) / 10) {
        numPosColors = colorContactsIncrementalKokkos(
            P.contacts, numPosUnits, P.numParticles,
            Kokkos::View<const unsigned long long*, CpMem>(P.prevContactKeys),
            Kokkos::View<const int*, CpMem>(P.prevContactColor), P.posPrevContactCount, P.unitColor,
            P.contactKeys, P.bodyWinner, P.bodyColorMask, posLeftover,
            /*forceFull*/ true, cSleep, posUnits);
        posDidFull = true;
      }
      if (posDidFull)
        P.posLastFullColors = numPosColors;
    } else {
      numPosColors =
          colorContactsKokkos(P.contacts, numPosUnits, P.numParticles, P.unitColor, P.bodyWinner,
                              P.bodyColorMask, posLeftover, cSleep, posUnits);
    }
    // Position vertices are raw slots. demStep's periodic images are local copies of their body
    // (§WO-4 item 2): they join its fold group, which needs the per-slot degree even without hubs.
    const bool images = !Hooks::distributed && P.numParticles > P.numReal;
    if ((posLeftover > 0 || images) && numPosUnits > 0) {
      edgeA = Kokkos::View<int*, CpMem>("peclet::dem::copies_eA", numPosUnits);
      edgeB = Kokkos::View<int*, CpMem>("peclet::dem::copies_eB", numPosUnits);
      positionEdgeEndsKokkos(P.contacts, posUnits, numPosUnits,
                             Kokkos::View<const int*, CpMem>(P.unitColor), edgeA, edgeB);
      growCopyView(P.vertexDegree, static_cast<std::size_t>(P.numParticles),
                   "peclet::dem::vertexDegree");
      vertexDegreeKokkos(edgeA, edgeB, numPosUnits, P.numParticles, P.vertexDegree);
      if (posLeftover > 0) {
        // Attempt failed: hub copies over the raw slots (§4.4), full recolour (the incremental
        // path falls back to this full path with copies).
        buildHubCopiesKokkos(edgeA, edgeB, numPosUnits, P.numParticles, slotBase, P.vertexDegree,
                             PC);
        growSlots(PC.nCopies);
        const SlotOverride ovColor{PC.slotA, PC.slotB, {}, 1.0f};
        numPosColors = colorContactsKokkos(P.contacts, numPosUnits, P.numParticles, P.unitColor,
                                           P.bodyWinner, P.bodyColorMask, posLeftover, cSleep,
                                           posUnits, ovColor, slotBase + PC.nCopies);
        if (incrPosColor) {
          posDidFull = true;
          P.posLastFullColors = numPosColors;
        }
        if (posLeftover > 0)
          noteInvariant("position", Kokkos::View<const int*, CpMem>(P.unitColor), numPosUnits);
      }
      buildCopyGroupsKokkos(P.realIndices, P.numParticles, P.numReal, P.vertexDegree,
                            images ? Kokkos::View<const float* [3], CpMem>(P.imageShift)
                                   : Kokkos::View<const float* [3], CpMem>(),
                            PC, /*identityBodies*/ Hooks::distributed);
    }
    // Every contact carries its unit's colour: the per-contact ledger commit below and the
    // diagnostics read it.
    expandUnitColorsKokkos(posUnits, numPosUnits, Kokkos::View<const int*, CpMem>(P.unitColor),
                           P.contactColor);
    if (incrPosColor) {
      // Commit this substep's (contact key, colour) for next substep's warm gather.
      commitContactColorKokkos(Kokkos::View<const unsigned long long*, CpMem>(P.contactKeys),
                               Kokkos::View<const int*, CpMem>(P.contactColor), P.prevContactKeys,
                               P.prevContactColor, P.posCommitPerm, nc);
      P.posPrevContactCount = nc;
    }
  }
  // Rank-level M (docs/contact_solve_framework.md §13.3): on a rank that exchanges with a
  // neighbour, every body touched through a ghost is a mass-split copy set across ranks. The
  // activity pass gives each slot its local active copy count a per phase (0 or 1 outside a hub
  // group, the group's count at a hub base) from the final colourings; the openings make the
  // global k. A rank that neither sends nor receives (np 1 closed, an isolated block) keeps WO-4's
  // single-rank path (k = groupK), bit for bit: §13.3 "step 0".
  const bool rankM = Hooks::distributed && hooks.rankK().exchanges;
  if (rankM && P.velocityUseGS) {
    const int nAll = P.numParticles + (VC.nCopies > PC.nCopies ? VC.nCopies : PC.nCopies);
    growCopyView(P.aVel, static_cast<std::size_t>(nAll), "peclet::dem::aVel");
    growCopyView(P.aPos, static_cast<std::size_t>(nAll), "peclet::dem::aPos");
    growCopyView(P.kVel, static_cast<std::size_t>(nAll), "peclet::dem::kVel");
    growCopyView(P.kPos, static_cast<std::size_t>(nAll), "peclet::dem::kPos");
    growCopyView(P.activityHit, static_cast<std::size_t>(nAll), "peclet::dem::activityHit");
    if (!usePersistPre) {  // rank-level X (§1.4): the g = 0 one-shot's masks and gates
      growCopyView(P.velMask, static_cast<std::size_t>(nAll), "peclet::dem::velMask");
      growCopyView(P.xGate, static_cast<std::size_t>(nm > 0 ? nm : 1), "peclet::dem::xGate");
    }
    const SlotOverride vOv =
        VC.nHubs > 0 ? SlotOverride{VC.slotA, VC.slotB, {}, 1.0f} : SlotOverride{};
    activityHitVelocityKokkos(P.manifolds, nm, P.realIndices,
                              Kokkos::View<const int*, CpMem>(P.manifoldColor), vOv, P.activityHit,
                              P.numParticles + VC.nCopies);
    finishActivityKokkos(Kokkos::View<const unsigned char*, CpMem>(P.activityHit), P.numParticles,
                         VC, P.aVel);
    const SlotOverride pOv =
        PC.nHubs > 0 ? SlotOverride{PC.slotA, PC.slotB, {}, 1.0f} : SlotOverride{};
    activityHitPositionKokkos(P.contacts, posUnits, numPosUnits,
                              Kokkos::View<const int*, CpMem>(P.unitColor), pOv, P.activityHit,
                              P.numParticles + PC.nCopies);
    finishActivityKokkos(Kokkos::View<const unsigned char*, CpMem>(P.activityHit), P.numParticles,
                         PC, P.aPos);
  }
  P.splitStats.velHubCopies = VC.nCopies;
  P.splitStats.posHubCopies = PC.nCopies;
  P.splitStats.splitBodiesVel = countSplitBodiesKokkos(VC);
  P.splitStats.splitBodiesPos = countSplitBodiesKokkos(PC);
  // The colouring invariant (§4.2 item 4, §12 S3): a single rank throws now; the distributed step
  // votes the flag in its first stop Allreduce and every rank throws there (a rank-local throw
  // would leave the others waiting in a collective).
  bool invariantPending = !invariantMsg.empty();
  if constexpr (!Hooks::distributed) {
    if (invariantPending)
      throw std::runtime_error(invariantMsg);
  }
  auto voteInvariant = [&](float res) {
    bool bad = invariantPending;
    invariantPending = false;
    const float r = hooks.allMaxAny(res, bad);
    if (bad)
      throw std::runtime_error(invariantMsg.empty()
                                   ? std::string("colouring invariant violated on another rank")
                                   : invariantMsg);
    return r;
  };
  // Dense colour buckets + fused sweep for the position projection (same precedent as the
  // velocity sweeps: colour classes are body-disjoint => bit-identical). The buckets list unit
  // indices.
  std::vector<int> posOffs;
  const bool posBuckets = P.velocityUseGS && numPosUnits > 0 && numPosColors > 0;
  if (posBuckets)
    buildColorBucketsKokkos(Kokkos::View<const int*, CpMem>(P.unitColor), numPosUnits, numPosColors,
                            P.posPerm, P.bucketCursor, posOffs);
  const Kokkos::View<const int*, CpMem> posPermC(P.posPerm);
  const std::vector<int>* posOffsP = posBuckets ? &posOffs : nullptr;
  const FusedSweepCtx posFused = (posBuckets && wantFused)
                                     ? demMakeFusedCtx(space, posOffs, P.posOffsDev, P.fusedBar)
                                     : FusedSweepCtx{};
  const FusedSweepCtx* posFusedP = posFused.maxBucket > 0 ? &posFused : nullptr;
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
  // Rank-level X (§1.4, WO-6): the g = 0 one-shot on a rank that exchanges, when the rank graph
  // has more than one colour (with one colour no two ranks are adjacent, so no body is split
  // across ranks and every contact fires).
  const RankK rk = hooks.rankK();
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 4
  const bool xOn = false;  // G13 mutant 4: every rank fires every contact
#else
  const bool xOn = rankM && !usePGS && P.velocityUseGS && rk.numColors > 1;
#endif
  const float gMagP = Kokkos::sqrt(P.gravity.x * P.gravity.x + P.gravity.y * P.gravity.y +
                                   P.gravity.z * P.gravity.z);
  const F3 gHat =
      usePersist ? F3{P.gravity.x / gMagP, P.gravity.y / gMagP, P.gravity.z / gMagP} : F3{0, 0, 0};
  // Event-level (Poisson) restitution: per-pair banked compression budget, released as a
  // budget-capped separation-velocity target during unloading (see updateRestitutionBankKokkos /
  // PGSManifoldSweep). Off (empty views) the sweeps run the per-substep Newton path verbatim.
  const bool poisson = usePGS && P.restitutionModel == 1;
  const Kokkos::View<float*, CpMem> bankV = poisson ? P.restBank : Kokkos::View<float*, CpMem>();
  const Kokkos::View<float*, CpMem> relV = poisson ? P.restRel : Kokkos::View<float*, CpMem>();
  const Kokkos::View<const float*, CpMem> vpkC =
      poisson ? Kokkos::View<const float*, CpMem>(P.restVPeak)
              : Kokkos::View<const float*, CpMem>();
  const Kokkos::View<float*, CpMem> orphV = poisson ? P.bodyOrphan : Kokkos::View<float*, CpMem>();
  const Kokkos::View<const float*, CpMem> orphPk =
      poisson ? Kokkos::View<const float*, CpMem>(P.bodyOrphanVPeak)
              : Kokkos::View<const float*, CpMem>();
  if (usePGS) {
    if (poisson && P.prevPairCount > 0) {  // reset the prev-ledger survival flags for the gather
      auto mt = Kokkos::subview(P.prevMatched, Kokkos::pair<int, int>(0, P.prevPairCount));
      Kokkos::deep_copy(mt, static_cast<unsigned char>(0));
    }
    gatherWarmLambdaKokkos(P.manifolds, nmVisible, P.realIndices, keyIdx, P.prevPairKeys,
                           P.prevLambda, P.prevLambdaT, P.prevPosImpulse, P.prevRestBank,
                           P.prevRestVPeak, P.prevPairCount, P.pairKeys, P.lambdaAcc, P.lambdaT,
                           P.posImpulse, P.restBank, P.restVPeak,
                           poisson ? P.prevMatched : Kokkos::View<unsigned char*, CpMem>(),
                           /*dedupTwins*/ !Hooks::distributed);
    if (poisson) {
      {  // per-substep release accumulator starts from zero every substep
        auto rr = Kokkos::subview(P.restRel, Kokkos::pair<int, int>(0, nm));
        Kokkos::deep_copy(rr, 0.0f);
      }
      // Orphan transfer: age the body accounts (owned range; MPI ghosts are mirrored), then
      // settle dead pairs' remaining budgets onto their endpoint bodies. Under MPI the pair-key
      // identities are gids, so the scatter resolves them through a sorted gid -> slot map built
      // over owned + ghost slots (a ghost-side credit is delivered to the owner by the velocity
      // reverse at the next sync; the pair's bank lives only in its owning rank's ledger).
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
        scatterOrphanBanksKokkos(Kokkos::View<const unsigned long long*, CpMem>(P.prevPairKeys),
                                 Kokkos::View<const float*, CpMem>(P.prevRestBank),
                                 Kokkos::View<const float*, CpMem>(P.prevRestVPeak),
                                 Kokkos::View<const unsigned char*, CpMem>(P.prevMatched),
                                 P.prevPairCount, Kokkos::View<const float*, CpMem>(P.invMass),
                                 P.bodyOrphan, P.bodyOrphanVPeak, gidSorted, slotSorted,
                                 Hooks::distributed ? P.numReal : -1);
      }
    }
    markPersistentManifoldsKokkos(P.manifolds, nm, P.realIndices, keyIdx, P.prevPairKeys,
                                  P.prevPairCount, P.pairKeys, P.manifoldPersistent);
    updateGroundedLevelsKokkos(P.manifolds, nmVisible, P.realIndices, P.posPred, gHat,
                               P.groundedLevel, nBodies, /*sweeps*/ 8, /*decay*/ 8);
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
                     P.vt0, /*dedupTwins*/ !Hooks::distributed);
    warmStartApplyKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                         P.realIndices, P.lambdaAcc, P.lambdaT, /*dedupTwins*/ !Hooks::distributed);
    // The warm impulses just moved every owned AND ghost body: the opening (§13.3 step 5)
    // reconciles them raw (known impulses), makes the global k_vel / k_pos and forwards the
    // owners' state, balance and k, so the first sweep reads a consistent ghost state. (An
    // isolated rank: a no-op, as the sync it replaces.)
    if constexpr (Hooks::distributed)
      hooks.openVelocityPhase(P, poisson);
  } else if (P.velocityUseGS) {
    // The g = 0 one-shot path: the opening makes the position phase's k_pos (§13.3) and, in the
    // same round, the velocity activity masks of rank-level X (§1.4, WO-6). The velocity phase's
    // reconciliation stays c771e07's raw reverse + forward: under X at most one copy of a body
    // changes per sync interval, so the raw sum is exact.
    if constexpr (Hooks::distributed)
      hooks.openPositionCounts(P, /*velocityMask=*/rankM);
  }
  // Velocity-phase copies (§4.4 step 6, §4.5): seeded once the warm start -- known impulses on
  // true masses -- has been applied (and, under MPI, published). The sweeps, the stabilization
  // passes and the multilevel cycle take the solve views and the slot overrides; the fold runs in
  // every iteration after the sweep; every rank sync re-marks the seeds (syncVel).
  const bool velCopiesOn = VC.nGroups > 0;
  // Rank-level M in the velocity phase (§13.3): the PGS path on an exchanging rank. The solve
  // views carry each slot's global k_vel (a hub copy its base's); the local fold still divides by
  // the local count a (groupK); the orphan shares B / k are already set by the opening.
  const bool velM = rankM && usePGS;
  Kokkos::View<const float*, CpMem> invMassVel = P.invMass;
  Kokkos::View<const float* [3], CpMem> invInertiaVel = P.invInertia;
  SlotOverride velOv{};
  const Kokkos::View<float*, CpMem> orphPkW =
      poisson ? P.bodyOrphanVPeak : Kokkos::View<float*, CpMem>();
  if (velM) {
    buildSolveViewsRankKKokkos(P, VC, slotBase + (velCopiesOn ? VC.nCopies : 0), P.invMass,
                               P.invInertia, Kokkos::View<const int*, CpMem>(P.kVel));
    invMassVel = P.invMassSolve;
    invInertiaVel = P.invInertiaSolve;
    velOv = velCopiesOn ? SlotOverride{VC.slotA, VC.slotB, P.splitSlot, kSplitOmegaVelocity}
                        : SlotOverride{{}, {}, P.splitSlot, kSplitOmegaVelocity};
    if (velCopiesOn) {
      seedCopySlotStateKokkos(P, VC);
      markCopySeedsKokkos(VC, P.velPred, P.angVelPred, orphV, orphPkW, /*shareFromBase*/ true);
    }
  } else if (velCopiesOn) {
    buildSolveViewsKokkos(P, VC, slotBase + VC.nCopies, P.invMass, P.invInertia);
    seedCopySlotStateKokkos(P, VC);
    markCopySeedsKokkos(VC, P.velPred, P.angVelPred, orphV, orphPkW);
    invMassVel = P.invMassSolve;
    invInertiaVel = P.invInertiaSolve;
    velOv = SlotOverride{VC.slotA, VC.slotB, P.splitSlot, kSplitOmegaVelocity};
  }
  auto foldVel = [&] {
    if (velCopiesOn)
      foldCopiesKokkos(space, VC, P.velPred, P.angVelPred, orphV, orphPkW, P.maxConsensus);
  };
  // §12 S14: a velocity phase with copies (local hub copies, or rank-level M) has not converged
  // while copies of one body disagree. Its stop votes fold in the largest consensus correction
  // since the previous vote (the local folds and the M syncs atomic-max it into P.maxConsensus),
  // through the SAME Allreduce-MAX; it is zeroed after each read. Without copies nothing is read
  // or written (np 1 without copies stays byte-identical).
  const bool velCons = velCopiesOn || velM;
  auto withConsensus = [&](float res, bool cons) {
    if (!cons)
      return res;
    const float c = readFloat(P.maxConsensus);
    Kokkos::deep_copy(P.maxConsensus, 0.0f);
    return std::max(res, c);
  };
  // §12 S13 (test-only): with P.noAdaptiveStop every loop runs its cap; the votes still run.
  const bool stopsOn = !P.noAdaptiveStop;
  const float fusedOff = -1.0f;  // a fused device loop's tolerance that never stops
  // A rank sync inside the velocity phase: orphan accounts back to balances, the raw reverse +
  // forward, then the copies' seeds re-marked from the reconciled bases.
  // Under rank-level M (velM) the sync itself is M (the halo's weighted apply; the bases hold
  // their shares, so no unfold first) and the copies re-seed from their bases' shares.
  auto syncVel = [&] {
    if constexpr (Hooks::distributed) {
      if (velM) {
        hooks.syncVelocities(P);
        if (velCopiesOn)
          markCopySeedsKokkos(VC, P.velPred, P.angVelPred, orphV, orphPkW, /*shareFromBase*/ true);
        return;
      }
      if (velCopiesOn)
        unfoldOrphanKokkos(VC, orphV);
      hooks.syncVelocities(P);
      if (velCopiesOn)
        markCopySeedsKokkos(VC, P.velPred, P.angVelPred, orphV, orphPkW);
    }
  };
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
    solveVelocityPGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, invMassVel, invInertiaVel,
                           P.quat, P.velPred, P.angVelPred, P.realIndices, P.growthRate,
                           P.restitutionNormal, vRest, P.maxApproach, P.lambdaAcc, P.vn0,
                           Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                           P.frictionDynamic, P.vt0, P.restitutionTangent,
                           Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV, vpkC,
                           orphV, orphPk, velPermC, velOffsP, velFusedP, nullptr, velOv);
    foldVel();
  };
  // Device-side iteration loop (CUDA, single-rank): the whole adaptive velocity loop as ONE
  // kernel — same sweeps, same residual, same stop; the per-iteration readback and the graph
  // capture disappear. The final residual stays in P.maxApproach for the stabilization trigger.
  bool velLoopDone = false;
  if constexpr (!Hooks::distributed) {
    if (usePGS && P.velocityUseGS && !velCopiesOn && velFusedP) {
      FusedLoopSpec spec{P.velocityIterations, stopsOn ? 0.02f * vRest : fusedOff, false};
      if (P.iterCounters)  // diagnostics (§12 S12): the loop writes its count on the device
        spec.iters = demIterCountSlot(P, 0);
      velLoopDone = solveVelocityPGSKokkos(
          P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia, P.quat, P.velPred,
          P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRest, P.maxApproach,
          P.lambdaAcc, P.vn0, Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
          P.frictionDynamic, P.vt0, P.restitutionTangent,
          Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV, vpkC, orphV, orphPk,
          velPermC, velOffsP, velFusedP, &spec);
    }
  }
  bool graphVel = false;
  CudaIterGraph gVel;
  if (!velLoopDone && usePGS && P.velocityUseGS) {
    PECLET_DEM_GRAPH_LOOP(graphVel, gVel, emitVelIter, P.graphCache[0])
  }
  P.splitStats.velItersUsed = velLoopDone ? demReadIterCount(P, 0) : 0;
  if (velCons)
    Kokkos::deep_copy(P.maxConsensus, 0.0f);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("dem::solve::vel");
  for (int it = 0; !velLoopDone && it < P.velocityIterations; ++it) {
    ++P.splitStats.velItersUsed;
    if (legacyFriction)
      accumulateNormalImpulseKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred,
                                    P.angVelPred, P.realIndices, P.growthRate,
                                    /*dedupTwins*/ !Hooks::distributed);
    if (P.velocityUseGS) {
      if (usePGS) {
        if (graphVel)
          gVel.launch(space);
        else
          emitVelIter();
      } else {
        Kokkos::deep_copy(space, P.maxApproach, 0.0f);
        // Rank-level X (§1.4, WO-6): in sync interval t a contact fires only if this rank holds
        // every rank-split endpoint; a non-firing contact still records its approach for the stop.
        XGateSpec xGate;  // evaluated inline in the sweep (no separate pass)
        if (xOn) {
          const long long t = P.solveEpoch + it / rk.syncInterval;
          xGate = XGateSpec{Kokkos::View<const unsigned long long*, CpMem>(P.velMask), rk.color,
                            rk.numColors, t};
          if (P.iterCounters) {  // diagnostics only: the explicit gate pass, to count the unfired
            computeXGateKokkos(P.manifolds, nm, P.realIndices,
                               Kokkos::View<const unsigned long long*, CpMem>(P.velMask), P.xGate,
                               rk.color, rk.numColors, t);
            P.splitStats.unfiredSplitContacts += countUnfiredKokkos(P.xGate, nm);
          }
        }
        solveVelocityColoredGSKokkos(P.manifolds, nm, P.manifoldColor, numColors, invMassVel,
                                     invInertiaVel, P.quat, P.velPred, P.angVelPred, P.realIndices,
                                     P.growthRate, P.restitutionNormal, vRest, P.maxApproach, {},
                                     {}, {}, {}, velOv, xGate);
        foldVel();
      }
      // Every active manifold is coloured (§4.2): the count-averaged fallback is gone. The stop's
      // Allreduce carries the colouring-invariant vote (voteInvariant; no extra message).
      const float velRes = voteInvariant(withConsensus(readFloat(P.maxApproach), velCons));
      // Adaptive stop. One-shot GS: end once no pair approaches above the resting threshold. PGS:
      // maxApproach records the largest APPLIED correction, and meaningful increments are ~g dt
      // (they propagate a chain one link per sweep), so the tolerance must sit well below vRest or
      // the stop starves deep-chain convergence permanently (measured: a 113-layer pile plateaued
      // at vz ~ -5 with the vRest stop). Once the warm-started network is converged the first
      // sweep's correction is ~0 and the loop still exits immediately. Distributed: the residual
      // is Allreduce-MAXed so every rank takes the same break (collective-refresh consistency).
      if (stopsOn && velRes <= (usePGS ? 0.02f * vRest : vRest))
        break;
    } else {
      // Mass-split Jacobi (velocityUseGS off, a global setting; docs/contact_solve_framework.md
      // §3.1): every manifold is its own copy. Count first, make the counts global (every copy of
      // a body must see the serial count), solve each manifold against copies of mass m / n with
      // true-mass deltas, and add them with factor 1 -- conservative at every iterate.
      countVelocityJacobiKokkos(P.manifolds, nm, P.realIndices, P.constraintCounts,
                                /*dedupTwins*/ !Hooks::distributed);
      hooks.syncContactCounts(P);
      solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                          P.realIndices, P.growthRate, P.restitutionNormal, vRest, P.deltaVel,
                          P.deltaAngVel, P.constraintCounts, {}, 0, {}, {}, {}, {},
                          /*massSplit=*/true, /*dedupTwins*/ !Hooks::distributed);
      applyVelocityDeltasAveragedKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel,
                                        P.deltaAngVel, P.constraintCounts, /*averaged=*/false);
    }
    if constexpr (Hooks::distributed) {
      if (hooks.syncPoint(it))
        syncVel();
    }
  }
  if constexpr (Hooks::distributed)
    syncVel();  // final owner->ghost refresh of the main velocity phase
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("dem::solve::stab_fric");
  // STABILIZATION PASS: if the symmetric sweeps could not drain the residual (a collapsing
  // column needs ~one sweep per layer to carry its weight to the floor -- unaffordable), arrest
  // the remaining quasi-static approach with grounded one-sided sweeps. In dynamic scenes the
  // residual is below the threshold and this pass never runs, so impact/discharge/shear keep
  // pure momentum-conserving physics. (set_stabilization_mode('off') disables the pass.)
  if (usePGS) {
    const float vRestS = 2.0f * P.dt * gMagP;
    const int smode = P.stabilizationMode;
    if (smode != 0 && hooks.allMax(readFloat(P.maxApproach)) > vRestS) {
      if (smode == 1) {  // ONE-SIDED grounded pass (default): held-lower-side impulses
        computeSideFlagsKokkos(P.manifolds, nm, P.realIndices,
                               Kokkos::View<const unsigned char*, CpMem>(P.manifoldPersistent),
                               Kokkos::View<const unsigned char*, CpMem>(P.groundedLevel),
                               P.posPred, P.velPred, gHat, 8.0f * P.dt * gMagP, P.sideFlags, P.vn0,
                               8.0f * P.dt * gMagP, /*dedupTwins*/ !Hooks::distributed);
        // Arrest budget: 2x the main budget -- the pass must out-pace a violent collapse, and it
        // only ever runs when the residual says one is happening (adaptive stop ends it early).
        auto emitOsIter = [&] {
          Kokkos::deep_copy(space, P.maxApproach, 0.0f);
          solveVelocityPGSKokkos(
              P.manifolds, nm, P.manifoldColor, numColors, invMassVel, invInertiaVel, P.quat,
              P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRestS,
              P.maxApproach, P.lambdaAcc, P.vn0,
              Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT, P.frictionDynamic,
              P.vt0, P.restitutionTangent, Kokkos::View<const float*, CpMem>(P.posImpulse), {},
              bankV, relV, vpkC, orphV, orphPk, velPermC, velOffsP, velFusedP, nullptr, velOv);
          foldVel();
        };
        bool osLoopDone = false;
        if constexpr (!Hooks::distributed) {
          if (!velCopiesOn && velFusedP) {
            const FusedLoopSpec spec{2 * P.velocityIterations, stopsOn ? vRestS : fusedOff, false};
            osLoopDone = solveVelocityPGSKokkos(
                P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia, P.quat,
                P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRestS,
                P.maxApproach, P.lambdaAcc, P.vn0,
                Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                P.frictionDynamic, P.vt0, P.restitutionTangent,
                Kokkos::View<const float*, CpMem>(P.posImpulse), {}, bankV, relV, vpkC, orphV,
                orphPk, velPermC, velOffsP, velFusedP, &spec);
          }
        }
        bool graphOs = false;
        CudaIterGraph gOs;
        if (!osLoopDone) {
          PECLET_DEM_GRAPH_LOOP(graphOs, gOs, emitOsIter, P.graphCache[1])
        }
        if (velCons)
          Kokkos::deep_copy(P.maxConsensus, 0.0f);
        for (int it = 0; !osLoopDone && it < 2 * P.velocityIterations; ++it) {
          if (graphOs)
            gOs.launch(space);
          else
            emitOsIter();
          const float osRes = hooks.allMax(withConsensus(readFloat(P.maxApproach), velCons));
          if (stopsOn && osRes <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              syncVel();
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
        // Eligibility gates (mldetail::kGate*): slip keeps the pass off sustained shear (silo
        // bulk) without starving a crushing bed's aggregation. (The env A/B over the mask was
        // retired in 1.0.0; the measured-best mask is the only one that ships.)
        constexpr int mlGates = mldetail::kGateSlip;
        // Coarse vertex masses (docs/contact_solve_framework.md §13.2): after the fold the a(q)
        // active local copies of vertex q move together, so the coarse problem sees ONE vertex of
        // mass a m / k -- invMassCoarse = invMass k / a. On a single rank (and for WO-4's interim
        // rank-local hubs) a = k at every vertex, so invMassCoarse is P.invMass itself: the true
        // mass at a folded hub, not the solve view's m / k (which gained (1 - 1/s) m dV per
        // coarse cycle at a hub that aggregates).
        // Under rank-level M a rank-split vertex has a < k: invMassCoarse = invMass k / max(1, a).
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 6
        Kokkos::View<const float*, CpMem> invMassCoarse = P.invMassSolve;  // G13 mutant 6
#else
        Kokkos::View<const float*, CpMem> invMassCoarse = P.invMass;
#endif
        if (velM) {
          growCopyView(P.invMassCoarse, static_cast<std::size_t>(nBodies),
                       "peclet::dem::invMassCoarse");
          buildInvMassCoarseKokkos(P.invMass, Kokkos::View<const int*, CpMem>(P.kVel),
                                   Kokkos::View<const int*, CpMem>(P.aVel), nBodies,
                                   P.invMassCoarse);
          invMassCoarse = P.invMassCoarse;
        }
        MlScratch S{P.mlColorPacked, P.mlParent, P.mlInvMassG, P.mlVelG,
                    P.mlVelG0,       P.mlMassG,  P.mlGrp,      P.mlMate};
        const ContactHierarchy H = buildContactHierarchyKokkos(
            P.manifolds, nm, P.realIndices, Kokkos::View<const int*, CpMem>(P.manifoldColor),
            Kokkos::View<const float*, CpMem>(P.vn0), Kokkos::View<const float* [3], CpMem>(P.vt0),
            Kokkos::View<const unsigned char*, CpMem>(P.manifoldPersistent), P.posPred, gHat,
            invMassCoarse, qsThr, mlGates, nBodies, S, P.bodyWinner, P.bodyColorMask,
            /*excludeImmovable*/ sleepOn,
            sleepOn ? Kokkos::View<const unsigned char*, CpMem>(P.asleep)
                    : Kokkos::View<const unsigned char*, CpMem>());
        P.mlLast.numLevels = H.numLevels;
#ifndef NDEBUG
        // Debug build (§13.5 WO-5 item 4): no multi-member coarse group holds an inactive vertex.
        if (velM && H.numLevels > 0 &&
            countInactiveAggregatedKokkos(Kokkos::View<const int*, CpMem>(S.parent), H.parentOff[0],
                                          nBodies, H.numGroups[0],
                                          Kokkos::View<const int*, CpMem>(P.aVel)) > 0)
          throw std::logic_error("rank-level M: a coarse group holds a vertex with a = 0");
#endif
        if (H.numLevels > 0 && (velCopiesOn || velM))  // §13.2's positive control (diagnostic)
          P.splitStats.mlHubAggregated =
              std::max(P.splitStats.mlHubAggregated,
                       countSplitAggregatedKokkos(
                           Kokkos::View<const int*, CpMem>(S.parent), H.parentOff[0], nBodies,
                           H.numGroups[0], Kokkos::View<const unsigned char*, CpMem>(P.splitSlot)));
        P.mlLast.numManifolds = nm;
        P.mlLast.numBodies = nBodies;
        P.mlLast.parentOff = H.parentOff;
        P.mlLast.numGroups = H.numGroups;
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
                ? demMakeMlFusedCtx(space, H, mlOffs, nm, nBodies, /*coarseSweeps*/ 2, P.mlOffsDev,
                                    P.fusedBar)
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
        // With hub copies (§4.4): fold between the fine sweep and the coarse cycle (the coarse
        // leg reads the bases, [0, nBodies)), and re-seed the copies from the bases after it.
        auto emitMlIter = [&] {
          Kokkos::deep_copy(space, P.maxApproachQS, 0.0f);
          solveVelocityPGSKokkos(
              P.manifolds, nm, P.manifoldColor, numColors, invMassVel, invInertiaVel, P.quat,
              P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRestS,
              P.maxApproach, P.lambdaAcc, P.vn0,
              Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT, P.frictionDynamic,
              P.vt0, P.restitutionTangent, Kokkos::View<const float*, CpMem>(P.posImpulse),
              P.maxApproachQS, bankV, relV, vpkC, orphV, orphPk, velPermC, velOffsP, velFusedP,
              nullptr, velOv);
          foldVel();
          if (H.numLevels > 0) {
            multilevelCoarseCycleKokkos(P.manifolds, nm, P.realIndices, invMassCoarse, P.velPred,
                                        P.lambdaAcc, P.maxApproachQS, nBodies, H, S,
                                        /*coarseSweeps*/ 2, Kokkos::View<const float*, CpMem>(relV),
                                        &mlOffs, Kokkos::View<const int*, CpMem>(P.mlBucketPerm),
                                        mlFusedP);
            if (velCopiesOn)
              reseedCopiesKokkos(space, VC, P.velPred, P.angVelPred);
          }
        };
        // Device-side stabilization loop (CUDA, single-rank): fine sweep + coarse cycle +
        // adaptive stop, all iterations in ONE kernel (see demFusedMlLoopK).
        bool mlLoopDone = false;
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr (!Hooks::distributed) {
          if (!velCopiesOn && velFusedP) {
            if (H.numLevels > 0 && mlFusedP) {
              const PGSManifoldSweep fStab = makePGSManifoldSweep(
                  P.manifolds, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                  P.realIndices, P.growthRate, P.restitutionNormal, vRestS, P.maxApproach,
                  P.maxApproachQS, P.lambdaAcc, P.vn0,
                  Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                  P.frictionDynamic, P.vt0, P.restitutionTangent,
                  Kokkos::View<const float*, CpMem>(P.posImpulse), bankV, relV, vpkC, orphV,
                  orphPk);
              mlLoopDone = demLaunchFusedMlLoop(
                  space, fStab, velPermC, *velFusedP, numColors, P.manifolds, P.realIndices,
                  Kokkos::View<const float*, CpMem>(P.invMass), P.velPred, P.lambdaAcc,
                  P.maxApproachQS, Kokkos::View<const float*, CpMem>(relV), S,
                  Kokkos::View<const int*, CpMem>(P.mlBucketPerm), *mlFusedP,
                  2 * P.velocityIterations, stopsOn ? vRestS : fusedOff);
            } else if (H.numLevels == 0) {
              // aggregation found nothing: the loop is plain fine sweeps on the QS residual
              const FusedLoopSpec spec{2 * P.velocityIterations, stopsOn ? vRestS : fusedOff,
                                       false};
              mlLoopDone = solveVelocityPGSKokkos(
                  P.manifolds, nm, P.manifoldColor, numColors, P.invMass, P.invInertia, P.quat,
                  P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRestS,
                  P.maxApproach, P.lambdaAcc, P.vn0,
                  Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT,
                  P.frictionDynamic, P.vt0, P.restitutionTangent,
                  Kokkos::View<const float*, CpMem>(P.posImpulse), P.maxApproachQS, bankV, relV,
                  vpkC, orphV, orphPk, velPermC, velOffsP, velFusedP, &spec);
            }
          }
        }
#endif
        bool graphMl = false;
        CudaIterGraph gMl;
        if (!mlLoopDone) {
          PECLET_DEM_GRAPH_LOOP(graphMl, gMl, emitMlIter, P.graphCache[2])
        }
        if (velCons)
          Kokkos::deep_copy(P.maxConsensus, 0.0f);
        for (int it = 0; !mlLoopDone && it < 2 * P.velocityIterations; ++it) {
          if (graphMl)
            gMl.launch(space);
          else
            emitMlIter();
          const float mlRes = hooks.allMax(withConsensus(readFloat(P.maxApproachQS), velCons));
          if (stopsOn && mlRes <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              syncVel();
          }
        }
      } else if (smode == 4) {
        // ORDERED (level-ordered symmetric sweeps; measurement mode): fresh height-from-floor
        // BFS levels order the manifolds bottom-up + top-down. Fully symmetric, but a pairwise
        // inelastic impulse only EQUALIZES velocities, so a deep column still cools one halving
        // per cycle -- measured insufficient on the statics battery (kept for A/B comparison
        // against the multilevel pass).
        computeHeightLevelsKokkos(P.manifolds, nmVisible, P.realIndices, P.posPred, gHat,
                                  P.heightLevel, nBodies);
        std::vector<std::pair<int, int>> buckets;
        buildLevelColorBucketsKokkos(
            P.manifolds, nm, P.realIndices, Kokkos::View<const int*, CpMem>(P.manifoldColor),
            Kokkos::View<const int*, CpMem>(P.heightLevel), P.levelKey, P.levelPerm, buckets);
        const PGSManifoldSweep sweep{P.manifolds,
                                     invMassVel,
                                     invInertiaVel,
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
                                     vpkC,
                                     orphV,
                                     orphPk,
                                     velOv};
        if (velCons)
          Kokkos::deep_copy(P.maxConsensus, 0.0f);
        for (int it = 0; it < 2 * P.velocityIterations; ++it) {
          Kokkos::deep_copy(P.maxApproach, 0.0f);
          solveVelocityPGSBucketsKokkos(sweep, Kokkos::View<const int*, CpMem>(P.levelPerm),
                                        buckets, /*topDown*/ false);
          solveVelocityPGSBucketsKokkos(sweep, Kokkos::View<const int*, CpMem>(P.levelPerm),
                                        buckets, /*topDown*/ true);
          foldVel();
          const float orRes = hooks.allMax(withConsensus(readFloat(P.maxApproach), velCons));
          if (stopsOn && orRes <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              syncVel();
          }
        }
      } else if (smode == 3) {
        // ESCALATION (diagnostic/fallback): keep running plain symmetric colored sweeps until
        // the residual drains or the 256-sweep cap -- provably correct physics, and the sweep
        // count it needs bounds what the ordered pass must deliver.
        if (velCons)
          Kokkos::deep_copy(P.maxConsensus, 0.0f);
        for (int it = 0; it < 256; ++it) {
          Kokkos::deep_copy(P.maxApproach, 0.0f);
          solveVelocityPGSKokkos(
              P.manifolds, nm, P.manifoldColor, numColors, invMassVel, invInertiaVel, P.quat,
              P.velPred, P.angVelPred, P.realIndices, P.growthRate, P.restitutionNormal, vRestS,
              P.maxApproach, P.lambdaAcc, P.vn0,
              Kokkos::View<const unsigned char*, CpMem>(P.sideFlags), P.lambdaT, P.frictionDynamic,
              P.vt0, P.restitutionTangent, Kokkos::View<const float*, CpMem>(P.posImpulse), {},
              bankV, relV, vpkC, orphV, orphPk, velPermC, velOffsP, velFusedP, nullptr, velOv);
          foldVel();
          const float esRes = hooks.allMax(withConsensus(readFloat(P.maxApproach), velCons));
          if (stopsOn && esRes <= vRestS)
            break;
          if constexpr (Hooks::distributed) {
            if (hooks.syncPoint(it))
              syncVel();
          }
        }
      }
      if constexpr (Hooks::distributed)
        syncVel();  // final refresh of the stabilization phase
    }
  }
  // End of the velocity phase: the orphan accounts back to balances (the bookkeeping, the legacy
  // friction pass and the next substep read the bases; copy slots are dead from here on).
  // Under rank-level M the owner rows of k_vel > 1 bodies get their balance back (§13.3 step 8;
  // the copies' shares are dead from here on, the next gather forwards B again).
  if (velM) {
    if (poisson)
      hooks.restoreOrphanBalance(P);
  } else if (velCopiesOn) {
    unfoldOrphanKokkos(VC, orphV);
  }
  // Poisson bookkeeping runs once per substep on the FINAL velocity state (after every phase and
  // ghost refresh): bank this substep's kinetic compression, deduct what was returned/released.
  if (poisson)
    updateRestitutionBankKokkos(
        P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred, P.realIndices,
        P.growthRate, P.restitutionNormal, 2.0f * P.dt * gMagP, P.vn0, P.lambdaAcc,
        Kokkos::View<const float*, CpMem>(P.restRel), P.restBank, P.restVPeak,
        /*dedupTwins*/ !Hooks::distributed);
  if (usePGS) {  // save the converged force network for next substep's warm start
    commitPairKeysLambdaKokkos(
        P.pairKeys, P.lambdaAcc, P.lambdaT, Kokkos::View<const float*, CpMem>(P.restBank),
        Kokkos::View<const float*, CpMem>(P.restVPeak), P.prevPairKeys, P.prevLambda, P.prevLambdaT,
        P.prevRestBank, P.prevRestVPeak, P.commitPerm, nm,
        // Carry the per-manifold colour by pair key (single-GPU incremental
        // colouring warm start; empty on the distributed path).
        Hooks::distributed ? Kokkos::View<const int*, CpMem>()
                           : Kokkos::View<const int*, CpMem>(P.manifoldColor),
        Hooks::distributed ? Kokkos::View<int*, CpMem>() : P.prevManifoldColor);
    P.prevPairCount = nm;
  }
  if (legacyFriction) {
    countFrictionContactsKokkos(P.contacts, nc, P.realIndices, P.planeFriction);
    hooks.syncFrictionCounts(P);
    // World-frame inverse inertia at the phase's frozen orientation P.quat (§12 S15).
    solveContactFrictionKokkos(P.contacts, nc, P.invMass, P.invInertia, P.quat, P.velPred,
                               P.angVelPred, P.realIndices, P.planeFriction, P.frictionDynamic,
                               P.deltaVel, P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
    if constexpr (Hooks::distributed)
      hooks.syncVelocities(P);  // publish the friction velocity update to the ghosts
  }

  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);
  if constexpr (Hooks::distributed)
    hooks.publishPositions(P);

  // Position-phase copies (§4.4, §4.5; demStep's periodic images, §WO-4 item 2): seeded after the
  // predict (and, under MPI, the publish). An image is re-seeded to its body's predicted position
  // plus its generation shift, so its increments are relative to the body.
  const bool posCopiesOn = PC.nGroups > 0;
  Kokkos::View<const float*, CpMem> invMassPos = P.invMass;
  Kokkos::View<const float* [3], CpMem> invInertiaPos = P.invInertia;
  SlotOverride posOv{};
  // Rank-level M in the position phase (§13.3 step 9): an exchanging rank on the GS paths (both
  // openings made k_pos). The solve views carry each raw slot's global k_pos (a hub copy its
  // base's); the local fold still divides by the local count a; the syncs are M in the halo.
  const bool posM = rankM && P.velocityUseGS;
  if (posM) {
    buildSolveViewsRankKKokkos(P, PC, slotBase + (posCopiesOn ? PC.nCopies : 0), P.invMass,
                               P.invInertia, Kokkos::View<const int*, CpMem>(P.kPos));
    invMassPos = P.invMassSolve;
    invInertiaPos = P.invInertiaSolve;
    if (posCopiesOn) {
      seedCopySlotStateKokkos(P, PC);
      markCopySeedsKokkos(PC, P.posPred, {}, {}, {});
      if (PC.nHubs > 0)
        posOv = SlotOverride{PC.slotA, PC.slotB, {}, 1.0f};
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 7
      posOv = SlotOverride{PC.slotA, PC.slotB, P.splitSlot, 1.5f};  // G13 mutant 7
#endif
    }
  } else if (posCopiesOn) {
    buildSolveViewsKokkos(P, PC, slotBase + PC.nCopies, P.invMass, P.invInertia);
    seedCopySlotStateKokkos(P, PC);
    markCopySeedsKokkos(PC, P.posPred, {}, {}, {});
    invMassPos = P.invMassSolve;
    invInertiaPos = P.invInertiaSolve;
    // The slot overrides only: the overlap projection is never relaxed (§13.1).
    if (PC.nHubs > 0)
      posOv = SlotOverride{PC.slotA, PC.slotB, {}, 1.0f};
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 7
    posOv = SlotOverride{PC.slotA, PC.slotB, P.splitSlot, 1.5f};  // G13 mutant 7
#endif
  }
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("dem::solve::pos");
  auto syncPos = [&] {
    if constexpr (Hooks::distributed) {
      hooks.syncPositions(P);
      if (posCopiesOn)
        markCopySeedsKokkos(PC, P.posPred, {}, {}, {});
    }
  };

  // Overlap resolved once the deepest penetration falls below ~0.01% of a particle radius.
  const float posTol = 1e-4f * P.baseRadius * P.globalScale;
  {
    auto pc = Kokkos::subview(P.posLambdaContact, Kokkos::pair<int, int>(0, nc));
    Kokkos::deep_copy(pc, 0.0f);
  }

  // One position iteration (async residual zero + colored overlap sweep), graph-captured on
  // the single-GPU path like the velocity loops.
  // WO-12: the accumulated, retractable projection (§13.5) at omega = P.positionOmega; the stop
  // reads the position change (posResidual), maxOverlap the largest violation seen.
  auto emitPosIter = [&] {
    Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
    Kokkos::deep_copy(space, P.posResidual, 0.0f);
    solvePositionColoredGSKokkos(P.contacts, posUnits, numPosUnits, P.unitColor, numPosColors,
                                 invMassPos, P.posPred, P.quatPred, P.quat, invInertiaPos,
                                 P.maxOverlap, P.posLambdaContact, posPermC, posOffsP, posFusedP,
                                 nullptr, posOv, P.posResidual, P.positionOmega);
    if (posCopiesOn)
      foldCopiesKokkos(space, PC, P.posPred, {}, {}, {}, P.maxConsensus);
  };
  // §12 S14: the position phase folds its consensus corrections (|dx| of a local fold or an M
  // sync, absolute like maxOverlap) into its stop vote, exactly as the velocity phase does.
  const bool posCons = posCopiesOn || posM;
  // Device-side position loop (CUDA, single-rank): all overlap-projection iterations + the
  // adaptive stop in ONE kernel. Disabled while the phase has copies (the fold runs between
  // iterations on the host-driven path).
  bool posLoopDone = false;
  if constexpr (!Hooks::distributed) {
    if (P.velocityUseGS && !posCopiesOn && posFusedP) {
      FusedLoopSpec spec{P.positionIterations, stopsOn ? posTol : fusedOff, true};
      if (P.iterCounters)  // diagnostics (§12 S12)
        spec.iters = demIterCountSlot(P, 1);
      Kokkos::deep_copy(space, P.maxOverlap, 0.0f);  // the largest violation over the device loop
      posLoopDone = solvePositionColoredGSKokkos(
          P.contacts, posUnits, numPosUnits, P.unitColor, numPosColors, P.invMass, P.posPred,
          P.quatPred, P.quat, P.invInertia, P.maxOverlap, P.posLambdaContact, posPermC, posOffsP,
          posFusedP, &spec, {}, P.posResidual, P.positionOmega);
    }
  }
  bool graphPos = false;
  CudaIterGraph gPos;
  if (!posLoopDone && P.velocityUseGS) {
    PECLET_DEM_GRAPH_LOOP(graphPos, gPos, emitPosIter, P.graphCache[3])
  }
  P.splitStats.posItersUsed = posLoopDone ? demReadIterCount(P, 1) : 0;
  if (posCons)
    Kokkos::deep_copy(P.maxConsensus, 0.0f);
  for (int it = 0; !posLoopDone && it < P.positionIterations; ++it) {
    ++P.splitStats.posItersUsed;
    if (P.velocityUseGS) {
      if (graphPos)
        gPos.launch(space);
      else
        emitPosIter();
      // Every active unit is coloured (§4.2): the count-averaged fallback is gone; the stop's
      // Allreduce carries the colouring-invariant vote if the velocity loop did not.
      const float posRes = voteInvariant(withConsensus(readFloat(P.posResidual), posCons));
      // Adaptive stop (WO-12): end once no contact's projection moves it by more than posTol
      // (|d| w, retractions included). Fixed positionIterations is the cap. Distributed:
      // Allreduce-MAXed so all ranks break together.
      if (stopsOn && posRes < posTol)
        break;
    } else {
      // Mass-split Jacobi (see the velocity branch): count, global counts, solve, add.
      countPositionJacobiKokkos(P.contacts, nc, P.constraintCounts);
      hooks.syncContactCounts(P);
      solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                          P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap, {}, 0,
                          /*massSplit=*/true);
      applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel,
                         P.constraintCounts, /*averaged=*/false);
    }
    if constexpr (Hooks::distributed) {
      if (hooks.syncPoint(it))
        syncPos();
    }
  }
  if constexpr (Hooks::distributed)
    syncPos();  // final owner->ghost refresh of the position phase
  // Position-channel Coulomb-bound carry (PGS path): next substep's friction cone sees
  // mu * (velocity-impulse channel + this position-channel load). Without it a jostled bed's
  // bound under-counts the true normal force and stick leaks (measured: 99% sliding wall
  // contacts in the benchmark drum while the Hertz reference sticks).
  if (usePGS && nm > 0)
    commitPosImpulseKokkos(P.posLambdaContact, nc, P.contactSlot, P.manifolds, nm, P.pairKeys,
                           P.prevPairKeys, P.prevPairCount, P.dt, P.vn0, P.prevPosImpulse);
  Kokkos::Profiling::popRegion();
  (void)space;
}

}  // namespace peclet::dem

#endif  // DEM_SOLVE_DRIVER_HPP
