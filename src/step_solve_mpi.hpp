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
};

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
  // Broadphase / narrow-phase margin: 0.1 R_max over ALL ranks, so both owners of a cross-face
  // pair report it at the same gap (a rank-local R_max dropped the pair on the rank whose own
  // grains are small while the other kept it).
  const float margin = 0.1f * globalMaxRadius(P, halo.comm());

  if (P.growthFactor != -1.0f && P.growthRate != 0.0f) {
    P.growthFactor *= std::exp(P.growthRate * P.dt);
    if (P.growthFactor > 1.0f)
      P.growthFactor = 1.0f;
  }
  if (P.growthFactor > 0.0f)
    updateGrowthScalesKokkos(P.numReal, P.scale, P.targetScale, P.growthFactor);
  const double band = std::max(rcut, xpbdContactReach(globalMaxRadius(P, halo.comm())));

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

}  // namespace peclet::dem

#endif  // PECLET_DEM_MPI
#endif  // DEM_STEP_SOLVE_MPI_HPP
