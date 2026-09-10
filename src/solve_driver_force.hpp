/// @file
/// @brief dem — the force-based DEM step driver: explicit soft-contact time stepping (the
/// engine family that computes contact FORCES from overlap and integrates them, in contrast to the
/// impulse/constraint XPBD engine in solve_driver.hpp).
///
/// The driver owns the engine-generic machinery — the cached Verlet pair list with key-carried
/// per-pair history, the skin/rebuild cadence (rebuild when any particle moved skin/2), the wall
/// candidate list, symplectic-Euler integration, displacement tracking and profiling — while the
/// contact FORCE LAW is a policy: `HertzMindlinLaw` (viscoelastic Hertz normal + history Mindlin
/// tangential, the Dosta-2024 / LIGGGHTS reference model) is the first implementation; other
/// explicit laws (linear spring-dashpot, cohesive/JKR, bonded) slot in as further policies with
/// their own per-pair state, reusing the same driver, halo choreography and tests.
///
/// Like the impulse driver, the same sequence runs single-GPU and distributed through a `Hooks`
/// policy:
///   * SoloForceHooks — every hook a no-op/identity; `demStepHertz` compiles to the historical
///     single-GPU engine (validated bit-for-bit on the Serial backend).
///   * MpiForceHooks (step_solve_mpi.hpp, PECLET_DEM_MPI) — domain-decomposed explicit DEM in the
///   classical
///     MD mold: at every pair-list rebuild the halo re-gathers ghosts in a band of
///     (pair cutoff + skin); between rebuilds only the ghost STATE (pos/vel/angVel/quat) is
///     forwarded owner->ghost each step. Every pair touching an owned particle is present
///     rank-locally (broadphase emits owned-ghost pairs once), forces on ghost slots are discarded
///     (the neighbour rank computes the mirrored pair itself — the same exact-redundant pattern as
///     the impulse engine), and only owned particles integrate. The rebuild decision, the skin and
///     the initial cache-validity flag are Allreduced so the collective gather/forward schedule is
///     identical on all ranks.
///
/// Per-pair history (e.g. the Mindlin shear spring xi) is keyed by GLOBAL particle id — stable
/// across halo rebuilds and ownership migration; the migration pack carries each particle's slice
/// of it (see mpi_halo.hpp).
#ifndef DEM_SOLVE_DRIVER_FORCE_HPP
#define DEM_SOLVE_DRIVER_FORCE_HPP

#include <cstdio>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <stdexcept>

#include "particles.hpp"
#include "solve_driver.hpp"  // findCollisionsGrow, maxOwnedRadius, readInt/readFloat
#include "solver_hertz.hpp"

namespace peclet::dem {

/// Rebuild the Hertz cached pair list with margin `skin`, carrying the Mindlin shear history
/// across by pair key (permutation sort of the OLD list, binary search from the NEW). Keys are
/// built from GLOBAL ids (identity == local index single-rank, so single-GPU behaviour is
/// unchanged; under MPI ghost slots are not stable identities, gids are).
inline void hertzRebuildPairs(Particles& P, float skin) {
  CpExec space;
  Kokkos::deep_copy(space, P.posPred, P.pos);  // broadphase reads posPred
  const int np = findCollisionsGrow(P, skin);
  const size_t need = P.pairs.extent(0);
  if (P.hertzXi.extent(0) < need) {
    P.hertzXi = Kokkos::View<float* [3], CpMem>("hertzXi", need);
    P.hertzSnPair = Kokkos::View<float*, CpMem>("hertzSnPair", need);
    P.hertzKeys = Kokkos::View<unsigned long long*, CpMem>("hertzKeys", need);
    P.hertzPrevKeys = Kokkos::View<unsigned long long*, CpMem>("hertzPrevKeys", need);
    P.hertzPrevXi = Kokkos::View<float* [3], CpMem>("hertzPrevXi", need);
  }
  // keys for the new list + history carry from the previous sorted list
  {
    auto pairs = P.pairs;
    auto keys = P.hertzKeys;
    auto xi = P.hertzXi;
    auto pk = P.hertzPrevKeys;
    auto px = P.hertzPrevXi;
    auto gid = P.gid;
    const int pc = P.hertzPrevCount;
    Kokkos::parallel_for(
        "peclet::dem::hertz_carry", Kokkos::RangePolicy<CpExec>(space, 0, np),
        KOKKOS_LAMBDA(int idx) {
          const unsigned long long k =
              pairKeyFromGids((unsigned)gid(pairs(idx, 0)), (unsigned)gid(pairs(idx, 1)));
          keys(idx) = k;
          const int l = lowerBoundKey(pk, pc, k);
          const bool hit = (l < pc && pk(l) == k);
          xi(idx, 0) = hit ? px(l, 0) : 0.0f;
          xi(idx, 1) = hit ? px(l, 1) : 0.0f;
          xi(idx, 2) = hit ? px(l, 2) : 0.0f;
        });
  }
  P.hertzNumPairs = np;
  {  // lagged patch-stiffness store: reset at rebuild (one mis-damped step, harmless)
    auto sn = Kokkos::subview(P.hertzSnPair, Kokkos::pair<int, int>(0, np));
    Kokkos::deep_copy(space, sn, 0.0f);
  }
  Kokkos::deep_copy(space, P.hertzRefPos, P.pos);
  Kokkos::deep_copy(space, P.hertzDispMax, 0.0f);
  space.fence();
}

/// Save the current list (sorted by key) so the NEXT rebuild can carry the history.
inline void hertzCommitHistory(Particles& P) {
  const int n = P.hertzNumPairs;
  if (n < 0)
    return;  // no valid LIVE list (fresh sim, or just after an ownership migration): the prev
             // store may hold carried history for the coming rebuild — do not wipe it.
  if (n == 0) {
    P.hertzPrevCount = 0;
    return;
  }
  CpExec space;
  const auto rng = Kokkos::pair<int, int>(0, n);
  auto kd = Kokkos::subview(P.hertzPrevKeys, rng);
  Kokkos::deep_copy(space, kd, Kokkos::subview(P.hertzKeys, rng));
  Kokkos::View<int*, CpMem> perm(
      Kokkos::view_alloc(space, "peclet::dem::hertz_perm", Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "peclet::dem::hertz_iota", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { perm(i) = i; });
  Kokkos::Experimental::sort_by_key(space, kd, perm);
  auto xi = P.hertzXi;
  auto px = P.hertzPrevXi;
  Kokkos::parallel_for(
      "peclet::dem::hertz_gather", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        const int j = perm(i);
        px(i, 0) = xi(j, 0);
        px(i, 1) = xi(j, 1);
        px(i, 2) = xi(j, 2);
      });
  P.hertzPrevCount = n;
  space.fence();
}

/// Hertz–Mindlin force-law policy (the first force law; see file comment). Owns the law-specific
/// parts: pair-list/history rebuild semantics, wall candidates, and the force kernels. Per-pair
/// state: Mindlin shear history xi (+ lagged patch stiffness snPair, reset at rebuild); per-
/// (particle, wall) state: xiWall/snWall.
struct HertzMindlinLaw {
  void commitHistory(Particles& P) const { hertzCommitHistory(P); }
  void rebuild(Particles& P, float skin) const {
    hertzRebuildPairs(P, skin);
    if (P.numWalls > 0)
      P.hertzNumWallCand =
          hertzBuildWallCandidatesKokkos(P.numReal, P.numWalls, P.walls, P.wallGrid, P.pos, P.rad,
                                         skin, P.hertzWallCand, P.hertzWallCandCount);
  }
  void pairForces(Particles& P, float dt, bool hasShapes) const {
    if (hasShapes)
      hertzShapePairForcesKokkos(P.pairs, P.hertzNumPairs, P.pos, P.quat, P.vel, P.angVel, P.scale,
                                 P.shapeId, P.shapes, P.shell, P.sdfGrid, P.globalScale,
                                 P.hertzContactRadiusFrac, P.invMass, MatIdView(P.materialId),
                                 PairTableView(P.pairMaterials), P.restitutionNormal,
                                 P.frictionDynamic, P.hertzE, P.hertzNu, dt, P.hertzXi,
                                 P.hertzSnPair, P.deltaVel, P.deltaAngVel);
    else
      hertzPairForcesKokkos(P.pairs, P.hertzNumPairs, P.pos, P.vel, P.angVel, P.rad, P.invMass,
                            MatIdView(P.materialId), PairTableView(P.pairMaterials),
                            P.restitutionNormal, P.frictionDynamic, P.hertzE, P.hertzNu, dt,
                            P.hertzXi, P.deltaVel, P.deltaAngVel);
  }
  void wallForces(Particles& P, float dt, bool hasShapes) const {
    if (P.numWalls > 0 && P.hertzNumWallCand > 0)
      hertzWallForcesKokkos(
          Kokkos::View<const int*, CpMem>(P.hertzWallCand), P.hertzNumWallCand, P.walls, P.wallGrid,
          P.pos, P.vel, P.angVel, P.rad, P.invMass, MatIdView(P.materialId),
          PairTableView(P.pairMaterials), P.restitutionNormal, P.frictionDynamic, P.hertzE,
          P.hertzNu, dt, P.hertzXiWall, Particles::kHertzMaxWalls, P.deltaVel, P.deltaAngVel,
          Kokkos::View<const float* [4], CpMem>(P.quat), Kokkos::View<const float*, CpMem>(P.scale),
          P.shapeId, P.shapes, P.shell, P.globalScale, P.hertzContactRadiusFrac, hasShapes,
          P.hertzSnWall);
  }
  /// Engine-support restrictions of this law (probed once per call).
  void validate(const Particles& P) const {
    if (P.domain.periodic_x || P.domain.periodic_y || P.domain.periodic_z)
      throw std::runtime_error("step_hertz: periodic domains not supported");
    if (P.numPlanes > 0)
      throw std::runtime_error("step_hertz: analytic planes not supported (use an SDF wall)");
    if (P.numWalls > Particles::kHertzMaxWalls)
      throw std::runtime_error("step_hertz: too many SDF walls");
  }
};

/// Zero the force/torque accumulator rows [lo, hi) — under MPI the pair kernels atomically
/// accumulate onto ghost slots too (Newton's third law), but only owned rows are consumed and
/// cleared by the integrator; without this the ghost rows grow without bound.
inline void zeroForceScratchKokkos(V3 dv, V3 dw, int lo, int hi) {
  if (hi <= lo)
    return;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::force_zero_ghost", Kokkos::RangePolicy<CpExec>(space, lo, hi),
      KOKKOS_LAMBDA(int i) {
        for (int c = 0; c < 3; ++c) {
          dv(i, c) = 0.0f;
          dw(i, c) = 0.0f;
        }
      });
  space.fence();
}

/// Single-GPU hooks: no ghosts, reductions are already global. Everything inlines away.
struct SoloForceHooks {
  static constexpr bool distributed = false;
  float allMax(float v) const { return v; }
  float allMin(float v) const { return v; }
  void gatherGhosts(Particles&) const {}
  void refreshGhostState(Particles&, bool) const {}
  void clearGhostScratch(Particles&) const {}
};

/// `nsteps` explicit force-based DEM steps of size `dt` with force law `Law` (see file comment).
/// The force/torque accumulators reuse the impulse engine's deltaVel/deltaAngVel scratch.
template <class Law, class Hooks>
inline void demStepForce(Particles& P, float dt, int nsteps, float skinFrac, const Law& law,
                         const Hooks& hooks) {
  law.validate(P);
  // world radii of the owned span (a halo gather re-fills owned + ghosts)
  fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numReal);
  float minRad = 0.0f;
  {
    CpExec space;
    auto rad = P.rad;
    float m = 3.4e38f;
    Kokkos::parallel_reduce(
        "peclet::dem::hertz_minrad", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
        KOKKOS_LAMBDA(int i, float& acc) {
          if (rad(i) < acc)
            acc = rad(i);
        },
        Kokkos::Min<float>(m));
    space.fence();
    minRad = m;
  }
  // Skin off the SMALLEST radius (globally under MPI: every rank must use the same skin, so the
  // rebuild threshold — and thus the collective gather schedule — is identical): a max-radius skin
  // inflates every small-grain pair cutoff by the big body's margin (measured ~15x more cached
  // pairs in the ball-impact case).
  const float skin = skinFrac * hooks.allMin(minRad);
  const float rebuildAt = 0.25f * skin * skin;  // (skin/2)^2 on |dx|^2
  const F3 g = P.gravity;
  // Cache-validity must be a COLLECTIVE decision too: after a migration every rank invalidates,
  // but a single desynchronised flag would desynchronise the gather below.
  bool rebuild = hooks.allMax(P.hertzNumPairs < 0 ? 1.0f : 0.0f) > 0.0f;
  // Non-spherical dispatch: any shape with a point shell routes pairs through the per-point
  // Hertz kernel and enables orientation integration.
  bool hasShapes = false;
  {
    auto hs = Kokkos::create_mirror_view(P.shapes);
    Kokkos::deep_copy(hs, P.shapes);
    for (size_t si = 0; si < hs.extent(0); ++si)
      if (hs(si).numPoints > 0)
        hasShapes = true;
  }
  static const bool profile = std::getenv("PECLET_DEM_HERTZ_PROFILE") != nullptr;
  double tPair = 0, tWall = 0, tInt = 0, tRebuild = 0, tCheck = 0;
  int nRebuilds = 0;
  Kokkos::Timer timer;
  for (int k = 0; k < nsteps; ++k) {
    if (rebuild) {
      law.commitHistory(P);
      // Distributed: re-gather the ghost band (pair cutoff + skin) with a fresh topology, THEN
      // build the rank-local pair list over owned + ghosts. Between rebuilds the Verlet argument
      // covers the ghosts too: nothing outside the band can reach a cached pair before the next
      // displacement-triggered rebuild.
      hooks.gatherGhosts(P);
      law.rebuild(P, skin);
      rebuild = false;
      ++nRebuilds;
      if (profile) {
        tRebuild += timer.seconds();
        timer.reset();
      }
    } else {
      // Ghosts must track their owners every step (positions/velocities changed by integrate).
      hooks.refreshGhostState(P, hasShapes);
    }
    hooks.clearGhostScratch(P);  // ghost force/torque slots: accumulated but never consumed
    law.pairForces(P, dt, hasShapes);
    if (profile) {
      tPair += timer.seconds();
      timer.reset();
    }
    law.wallForces(P, dt, hasShapes);
    if (profile) {
      tWall += timer.seconds();
      timer.reset();
    }
    hertzIntegrateKokkos(P.numReal, P.deltaVel, P.deltaAngVel, P.invMass, P.invInertia, g, dt,
                         P.vel, P.angVel, P.pos, P.quat, hasShapes);
    if (profile) {
      tInt += timer.seconds();
      timer.reset();
    }
    if ((k & 15) == 15 || k == nsteps - 1) {
      if (hooks.allMax(hertzMaxDisp2Kokkos(P.numReal, P.pos, P.hertzRefPos)) > rebuildAt)
        rebuild = true;
      if (profile) {
        tCheck += timer.seconds();
        timer.reset();
      }
    }
  }
  if (profile)
    std::printf(
        "[hertz profile] steps=%d pairs=%d wallcand=%d rebuilds=%d | pair %.3fs wall %.3fs "
        "integrate %.3fs check %.3fs rebuild %.3fs\n",
        nsteps, P.hertzNumPairs, P.hertzNumWallCand, nRebuilds, tPair, tWall, tInt, tCheck,
        tRebuild);
  CpExec space;
  Kokkos::deep_copy(space, P.posPred, P.pos);  // keep the impulse-path views coherent
  Kokkos::deep_copy(space, P.velPred, P.vel);
  space.fence();
}

/// `nsteps` of the soft-sphere Hertz-Mindlin engine — the single-GPU instantiation of the
/// force-based driver (kept as the historical entry point; see demStepForce).
inline void demStepHertz(Particles& P, float dt, int nsteps, float skinFrac) {
  demStepForce(P, dt, nsteps, skinFrac, HertzMindlinLaw{}, SoloForceHooks{});
}

}  // namespace peclet::dem

#endif  // DEM_SOLVE_DRIVER_FORCE_HPP
