/// @file
/// @brief dem — the distributed step drivers (gated `PECLET_DEM_MPI`): the hook policies that turn
/// the shared solve drivers into their processor-block forms (`MpiSolveHooks`, `MpiForceHooks`)
/// and the two distributed substeps, `demStepMpi` (XPBD) and `demStepHertzMpi` (Hertz–Mindlin).
/// The default (non-MPI) module never includes mpi_halo.hpp: the whole file is inert without the
/// macro.
#ifndef DEM_STEP_SOLVE_MPI_HPP
#define DEM_STEP_SOLVE_MPI_HPP

#ifdef PECLET_DEM_MPI

#include <algorithm>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "mpi_halo.hpp"  // ParticleHalo
#include "particles.hpp"
#include "solve_driver.hpp"        // demSolveContacts, maxOwnedRadius
#include "solve_driver_force.hpp"  // demStepForce + HertzMindlinLaw + fillWorldRadiiKokkos
#include "step_solve.hpp"          // narrowPhaseGrow

namespace peclet::dem {

/// MPI hooks for the shared contact-solve driver (demSolveContacts): the distributed step is the
/// processor-block Gauss-Seidel form of the SAME modern sequence the single-GPU step runs
/// (docs/mpi_momentum_conservation.md). Every contact is solved by EXACTLY ONE rank
/// (ContactOwnership: the only owner that sees it, else the lower-gid body's owner); each rank
/// sweeps only the contacts and manifolds it owns, over owned + ghost body slots, and writes the
/// partner's half of every impulse into the partner's ghost slot. At every sync the ghosts'
/// accumulated changes are reverse-accumulated onto their owners, then the owners republish
/// (reverse, then forward), so each impulse and each position correction is applied once, equal
/// and opposite -- no redundant two-owner solve. Syncs run every `syncEvery` solver iterations
/// plus once after every solve phase; each adaptive-stop residual is Allreduce-MAXed so all ranks
/// take the same break (the syncs are collective -- a rank-local break would deadlock them).
/// `numManifoldsVisible` is the whole visible manifold range, read only by the label passes
/// (warm-ledger match, grounded / height levels).
struct MpiSolveHooks {
  static constexpr bool distributed = true;
  ParticleHalo& halo;
  int syncEvery;
  bool forwardRotation;
  int numManifoldsVisible;
  float allMax(float v) const {
    float g = v;
    MPI_Allreduce(&v, &g, 1, MPI_FLOAT, MPI_MAX, halo.comm());
    return g;
  }
  /// allMax of a residual plus a vote in the SAME Allreduce: `any` becomes true on every rank if
  /// it was true on any (a rank-local fallback made collective at no extra message).
  float allMaxAny(float v, bool& any) const {
    const float in[2] = {v, any ? 1.0f : 0.0f};
    float out[2] = {v, in[1]};
    MPI_Allreduce(in, out, 2, MPI_FLOAT, MPI_MAX, halo.comm());
    any = out[1] > 0.0f;
    return out[0];
  }
  bool syncPoint(int it) const { return (it + 1) % syncEvery == 0; }
  int visibleManifolds(int) const { return numManifoldsVisible; }
  void beginSolve(Particles& P) const { halo.markVelocityBaseline(P); }
  void syncVelocities(Particles& P) const { halo.syncVelocities(P, forwardRotation); }
  void publishPositions(Particles& P) const { halo.publishPositions(P, forwardRotation); }
  void syncPositions(Particles& P) const { halo.syncPositions(P, forwardRotation); }
  void syncFrictionCounts(Particles& P) const { halo.syncFrictionCounts(P); }
  void syncContactCounts(Particles& P) const { halo.syncContactCounts(P); }
  // Rank-level M (docs/contact_solve_framework.md §13.3).
  RankK rankK() const {
    return RankK{halo.exchanges(), halo.rankColor(), halo.numRankColors(), syncEvery};
  }
  void openVelocityPhase(Particles& P, bool poisson) const {
    halo.openVelocityPhase(P, forwardRotation, poisson);
  }
  void openPositionCounts(Particles& P, bool velocityMask) const {
    halo.openPositionCounts(P, velocityMask);
  }
  void restoreOrphanBalance(Particles& P) const { halo.restoreOrphanBalance(P); }
};

/// Drift slack S = kDriftSlack * R_max (docs/contact_solve_framework.md §5.1, R-F1): the ghost
/// band carries S beyond the contact reach, and a particle that has drifted S beyond its owner's
/// block triggers a collective migrateToBlocks.
inline constexpr float kDriftSlack = 0.25f;

/// The drift vote's rank-local maxima over the owned set (§5.1):
///   d_i = |v_i| dt + |g + extForce_i invMass_i| dt^2  (bound on predict's centre displacement)
///   E_i = dist(x_i, this rank's block, minimum image on periodic axes) + d_i.
/// `dt` = 0 gives the pure distance (the Hertz rebuild vote on committed positions, §5.5).
inline void driftVoteLocalKokkos(const Particles& P, const ParticleHalo::BlockBox box, float dt,
                                 float& eMax, float& dMax) {
  const F3 g = P.gravity;
  const auto pos = P.pos;
  const auto vel = P.vel;
  const auto ext = P.extForce;
  const auto invM = P.invMass;
  float e = 0.0f, d = 0.0f;
  Kokkos::parallel_reduce(
      "peclet::dem::drift_vote", Kokkos::RangePolicy<CpExec>(0, P.numReal),
      KOKKOS_LAMBDA(int i, float& em, float& dm) {
        const float ax = g.x + ext(i, 0) * invM(i), ay = g.y + ext(i, 1) * invM(i),
                    az = g.z + ext(i, 2) * invM(i);
        const float vv = Kokkos::sqrt(vel(i, 0) * vel(i, 0) + vel(i, 1) * vel(i, 1) +
                                      vel(i, 2) * vel(i, 2));
        const float di = vv * dt + Kokkos::sqrt(ax * ax + ay * ay + az * az) * dt * dt;
        const float x[3] = {pos(i, 0), pos(i, 1), pos(i, 2)};
        const float lo[3] = {box.lo.x, box.lo.y, box.lo.z}, hi[3] = {box.hi.x, box.hi.y, box.hi.z},
                    L[3] = {box.period.x, box.period.y, box.period.z};
        float d2 = 0.0f;
        for (int k = 0; k < 3; ++k) {
          auto gap = [&](float xx) { return Kokkos::fmax(0.0f, Kokkos::fmax(lo[k] - xx, xx - hi[k])); };
          float gk = gap(x[k]);
          if (L[k] > 0.0f)
            gk = Kokkos::fmin(gk, Kokkos::fmin(gap(x[k] - L[k]), gap(x[k] + L[k])));
          d2 += gk * gk;
        }
        const float ei = Kokkos::sqrt(d2) + di;
        if (ei > em)
          em = ei;
        if (di > dm)
          dm = di;
      },
      Kokkos::Max<float>(e), Kokkos::Max<float>(d));
  eMax = e;
  dMax = d;
}

/// Largest particle radius over ALL ranks (growth included) -- the halo band and the contact
/// reach must not depend on which grains a rank happens to own. An empty rank contributes nothing
/// (maxOwnedRadius's scale-1 fallback would inflate the maximum of a run of smaller grains).
inline float globalMaxRadius(const Particles& P, MPI_Comm comm) {
  float r = P.numReal > 0 ? maxOwnedRadius(P) : 0.0f, g = r;
  MPI_Allreduce(&r, &g, 1, MPI_FLOAT, MPI_MAX, comm);
  return g > 0.0f ? g : maxOwnedRadius(P);
}



/// The XPBD narrow phase's reach: it reports a pair while the gap is below the broadphase margin
/// (0.1 R_max), i.e. at centre distance < r_i + r_j + margin, so a partner of a body across a
/// block face can sit up to 2 R_max + 0.1 R_max from it. The single-rank periodic ghost band
/// (demStep) is the same number.
inline double xpbdContactReach(float rMax) {
  return 2.1 * static_cast<double>(rMax);
}

/// TEST-ONLY (Simulation::debugCaptureContacts): host copies of the owned contacts [0, ncOwned)
/// -- global ids, each slot's periodic image (the halo's shift of a ghost slot in box lengths),
/// dist -- and of the owned predicted positions + radii, into P.debugCaptured. Reads only.
inline void debugCaptureOwnedContacts(Particles& P, const ParticleHalo& halo, int ncOwned) {
  DebugContactCapture& cap = P.debugCaptured;
  const int no = P.numReal;
  auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.contacts);
  auto hg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.gid);
  auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.posPred);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), P.rad);
  const std::vector<float> shift = halo.debugGhostShiftsBySlot();
  const double box[3] = {static_cast<double>(P.domain.max.x) - P.domain.min.x,
                         static_cast<double>(P.domain.max.y) - P.domain.min.y,
                         static_cast<double>(P.domain.max.z) - P.domain.min.z};
  auto image = [&](int slot, std::vector<int>& out) {
    for (int d = 0; d < 3; ++d) {
      const float sh = slot < no ? 0.0f : shift[3 * static_cast<std::size_t>(slot - no) + d];
      out.push_back(sh == 0.0f ? 0 : static_cast<int>(std::lround(sh / box[d])));
    }
  };
  cap = DebugContactCapture{};
  for (int i = 0; i < ncOwned; ++i) {
    const ContactC& c = hc(i);
    cap.gidA.push_back(hg(c.bodyA));
    cap.gidB.push_back(c.bodyB >= 0 ? hg(c.bodyB) : -1);
    image(c.bodyA, cap.imageA);
    if (c.bodyB >= 0)
      image(c.bodyB, cap.imageB);
    else
      cap.imageB.insert(cap.imageB.end(), {0, 0, 0});
    cap.dist.push_back(c.dist);
  }
  for (int i = 0; i < no; ++i) {
    cap.gid.push_back(hg(i));
    for (int d = 0; d < 3; ++d)
      cap.posPred.push_back(hp(i, d));
    cap.rad.push_back(hr(i));
  }
}

/// The XPBD drift vote (docs/contact_solve_framework.md §5.1), run at the top of every
/// distributed substep: ONE Allreduce(MAX) of {R_max owned, E_max, d_max}; if some particle is S
/// beyond its owner's block (or would be after predict), every particle moves to the rank whose
/// current block holds it (collective migrateToBlocks). Everything after it depends only on the
/// reduced values, so the schedule is identical on every rank.
struct MpiDriftVote {
  float rMax;    // global max radius (pre-growth)
  float margin;  // 0.1 R_max: the broad/narrow-phase margin
  float slack;   // S = kDriftSlack R_max
  float dMax;    // global max predicted displacement of the substep
};
inline MpiDriftVote mpiDriftVote(Particles& P, ParticleHalo& halo) {
  float vote[3] = {P.numReal > 0 ? maxOwnedRadius(P) : 0.0f, 0.0f, 0.0f};
  driftVoteLocalKokkos(P, halo.blockBox(), P.dt, vote[1], vote[2]);
  float g[3];
  MPI_Allreduce(vote, g, 3, MPI_FLOAT, MPI_MAX, halo.comm());
  const MpiDriftVote v{g[0], 0.1f * g[0], kDriftSlack * g[0], g[2]};
  int commSize = 1;
  MPI_Comm_size(halo.comm(), &commSize);
  // One rank owns everything: there is no other owner to move a particle to (a migration would be
  // a host round trip that changes nothing).
  if (commSize > 1 && g[1] >= v.slack) {
    halo.migrateToBlocks(P);
    ++P.splitStats.driftMigrations;
  }
  return v;
}
/// The XPBD ghost band: reach + S + P, P the Verlet skin when on (the halo adds it itself), else
/// the substep's predicted displacement d_max (the topology is rebuilt every gather then). `rcut`
/// stays a lower bound. Reads the post-growth global R_max (one Allreduce, as before).
inline double mpiXpbdBand(const Particles& P, ParticleHalo& halo, double rcut,
                          const MpiDriftVote& v) {
  return std::max(rcut, xpbdContactReach(globalMaxRadius(P, halo.comm())) +
                            static_cast<double>(v.slack) +
                            (halo.verletSkin() > 0.0f ? 0.0 : static_cast<double>(v.dMax)));
}

/// One distributed XPBD DEM substep. The periodic ghost generation of the single-rank step is
/// replaced by a cross-rank gather (halo.gather, ghosts carrying REAL mass + the owner's gid /
/// material / grounded level), then the FULL modern solve sequence runs through demSolveContacts
/// with MpiSolveHooks -- graph-colored Gauss-Seidel restitution, warm-started PGS with
/// persistent contacts (pair keys built from GLOBAL ids, so they survive halo rebuilds and
/// ownership migration), gravity statics (grounded shock propagation / stabilization passes),
/// friction cone, colored-GS overlap projection and the adaptive stops -- over the contacts this
/// rank OWNS (docs/mpi_momentum_conservation.md). Linear momentum and the centre of mass are
/// conserved to round-off at any np, thread count and sync_every. Interior contacts keep the
/// serial Gauss-Seidel order; a contact across a rank face sees its far body as of the last
/// reconciliation, so trajectories agree with single-rank statistically, not bit for bit.
/// np = 1 on a closed domain is the single-rank sequence bit for bit; np = 1 on a periodic
/// domain solves each wrap pair once (the twin single-rank keeps), not twice.
/// `forwardRotation`=false (spheres) skips the angular/quaternion forwards.
///
/// PERIODICITY: cross-rank ghosts supply the wrap on DECOMPOSED axes; LOCAL periodic self-ghosts
/// (ParticleHalo build with includePeriodicSelf) supply it on UNDECOMPOSED periodic axes (a "x1"
/// ORB axis, e.g. z of a 2x2x1 layout, or np=1). Correct for any layout, including np=1 fully
/// periodic. CAPACITY: a periodic box needs a thick ghost boundary layer -- size the Simulation
/// capacity for the worst-case ghost band; gather() throws on overflow rather than corrupting the
/// SoA.
///
/// BAND: the contact reach over the GLOBAL maximum radius after this substep's growth
/// (xpbdContactReach), or the caller's `rcut` when that is wider. A narrower band cannot be
/// right: it drops a partner of a cross-face pair from its neighbour's view (a band of one
/// rank-local radius did; so does rcut = 2r, which misses the margin).
inline void demStepMpi(Particles& P, ParticleHalo& halo, double rcut, int syncEvery,
                       bool forwardRotation) {
  CpExec space;
  const MpiDriftVote vote = mpiDriftVote(P, halo);
  // Broadphase / narrow-phase margin: 0.1 R_max over ALL ranks, so both owners of a cross-face
  // pair report it at the same gap (a rank-local R_max dropped the pair on the rank whose own
  // grains are small while the other kept it).
  const float margin = vote.margin;

  if (P.growthFactor != -1.0f && P.growthRate != 0.0f) {
    P.growthFactor *= std::exp(P.growthRate * P.dt);
    if (P.growthFactor > 1.0f)
      P.growthFactor = 1.0f;
  }
  if (P.growthFactor > 0.0f)
    updateGrowthScalesKokkos(P.numReal, P.scale, P.targetScale, P.growthFactor);
  const double band = mpiXpbdBand(P, halo, rcut, vote);

  // 1. Predict velocity on the owned set (no ghosts yet -> numParticles == numReal).
  P.numParticles = P.numReal;
  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt, P.extForce,
                        P.extTorque);

  // 2. Gather ghosts (real mass) from owners over the halo: full state -- including gid,
  //    materialId and the warm grounded level -- into the ghost slots; sets
  //    P.numParticles = numReal + numGhost and self-maps realIndices.
  halo.gather(P, band);
  // The velocity-phase slot map (docs/contact_solve_framework.md §6.2): one velocity slot per body
  // per rank -- a periodic self image maps to the owned slot, other images of one body to its
  // lowest ghost slot. Here, not in gather (the Hertz engine shares gather). Positions stay per
  // slot.
  halo.mapVelocitySlots(P);

  fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numParticles);

  // 3. Broad/narrow phase + manifold reduction over owned + ghosts (contactSlot map included:
  // the PGS friction bound and the position-channel Coulomb carry read through it).
  // findCollisionsGrow fences + reads the pair count back to host and guarantees np <= P.pairs
  // extent (growing the buffer on overflow) so the narrowphase never reads P.pairs out of bounds.
  const int np = findCollisionsGrow(P, margin);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  const int nc = narrowPhaseGrow(P, np, margin);

  // Contact ownership (docs/mpi_momentum_conservation.md §2.1-2.2): owned contacts first, then
  // the manifolds reduced owned-first. The solve runs on [0, ncOwned) / [0, nmOwned); the
  // visible counts stay in P.contactCount / P.manifoldCount for the getters.
  const int ncOwned = partitionContactsKokkos(P.contacts, nc, halo.contactOwnership(P));
  if (P.debugCapture)
    debugCaptureOwnedContacts(P, halo, ncOwned);
  int nmOwned = 0;
  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount, P.contactSlot,
                                  ncOwned, &nmOwned);
  const int nmVisible = readInt(P.manifoldCount);

  // 4-6. The shared modern velocity + position solve, distributed: rank-local colouring over the
  // owned + ghost body slots (nBodies = numParticles; realIndices are self-mapped, so ghost
  // copies evolve in place between reconciliations), persistent-pair keys from the global ids.
  demSolveContacts(P, ncOwned, nmOwned, P.numParticles, P.gid,
                   MpiSolveHooks{halo, syncEvery < 1 ? 1 : syncEvery, forwardRotation, nmVisible});

  // 7. Commit (owned results kept; the ghost slots, reconciled by the final sync, are dropped and
  // re-gathered next substep).
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
  float slack;  // drift slack S = kDriftSlack R_max (§5.5)
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
    // The drift vote at every pair-list rebuild (docs/contact_solve_framework.md §5.5): if some
    // particle sits S or more beyond its owner's block (committed positions; the list criterion
    // covers motion until the next rebuild), move every particle to its current block first. The
    // live Mindlin springs ride the migration pack (gid-keyed) exactly as in a rebalance.
    float e = 0.0f, d = 0.0f;
    driftVoteLocalKokkos(P, halo.blockBox(), 0.0f, e, d);
    int commSize = 1;
    MPI_Comm_size(halo.comm(), &commSize);
    if (allMax(e) >= slack && commSize > 1) {
      halo.migrateToBlocks(P);
      ++P.splitStats.driftMigrations;
      // The owned force accumulators are zeroed as they are consumed (integrate), the ghost ones
      // only at the start of each step (clearGhostScratch) and then left holding the discarded
      // ghost halves. A migration turns former ghost slots into owned ones: clear the owned range
      // so no stale ghost force is integrated (measured: dP 1e-8 -> 9e-4 at the first migration).
      zeroForceScratchKokkos(P.deltaVel, P.deltaAngVel, 0, P.numReal);
    }
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
  // Ghost band = worst-case pair cutoff (2 R_max,global) + skin + drift slack S (§5.5). The
  // driver's skin is skinFrac * R_min,global <= skinFrac * R_max,global, so this band bounds it;
  // S covers a partner up to S beyond its owner's block (the rebuild vote keeps it there).
  float maxR = P.numReal > 0 ? maxOwnedRadius(P) : 0.0f, maxRg = maxR;
  MPI_Allreduce(&maxR, &maxRg, 1, MPI_FLOAT, MPI_MAX, halo.comm());
  const float slack = kDriftSlack * maxRg;
  const double band = (2.0 + skinFrac) * static_cast<double>(maxRg) + static_cast<double>(slack);
  demStepForce(P, dt, nsteps, skinFrac, HertzMindlinLaw{}, MpiForceHooks{halo, band, slack});
  P.numParticles = P.numReal;  // restore owned-only active count for getters
}

}  // namespace peclet::dem

#endif  // PECLET_DEM_MPI
#endif  // DEM_STEP_SOLVE_MPI_HPP
