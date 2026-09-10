/// @file
/// @brief dem — the distributed step drivers (gated `PECLET_DEM_MPI`): the hook policies that turn
/// the shared solve drivers into their processor-block forms (`MpiSolveHooks`, `MpiForceHooks`)
/// and the two distributed substeps, `demStepMpi` (XPBD) and `demStepHertzMpi` (Hertz–Mindlin).
/// The default (non-MPI) module never includes mpi_halo.hpp: the whole file is inert without the
/// macro.
#ifndef DEM_STEP_SOLVE_MPI_HPP
#define DEM_STEP_SOLVE_MPI_HPP

#ifdef PECLET_DEM_MPI

#include <cmath>
#include <Kokkos_Core.hpp>

#include "mpi_halo.hpp"  // ParticleHalo
#include "particles.hpp"
#include "solve_driver.hpp"        // demSolveContacts, maxOwnedRadius
#include "solve_driver_force.hpp"  // demStepForce + HertzMindlinLaw + fillWorldRadiiKokkos
#include "step_solve.hpp"          // narrowPhaseGrow

namespace peclet::dem {

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

}  // namespace peclet::dem

#endif  // PECLET_DEM_MPI
#endif  // DEM_STEP_SOLVE_MPI_HPP
