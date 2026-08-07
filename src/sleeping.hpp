/// @file
/// @brief dem — island sleeping / freezing for the single-GPU PGS statics path.
///
/// A settled granular bed is quasi-static: its contact network re-solves to the SAME frozen force
/// balance every substep, and the multilevel stabilization pass rebuilds its whole hierarchy
/// (~540 launches/substep at 25k) to re-arrest a pile that is not moving. Sleeping removes that
/// waste: a REAL body whose motion has stayed below the solver's own resting floor for K substeps
/// while grounded is put to sleep (velocity zeroed, integration skipped). A manifold whose BOTH
/// endpoints are asleep (a static wall counts as asleep) is excluded from the colouring, the
/// sweeps and the multilevel hierarchy — so a fully-settled bed collapses to the broad/narrow-phase
/// floor. A sleeping body is given effective inverse mass 0 for the solve, so any awake–asleep
/// contact treats the sleeper as immovable (kinematic) — exactly the role the grounded one-sided
/// pass fakes — which keeps the solve correct WITHOUT waking the sleeper. Waking is therefore a
/// physics decision, not a solver-correctness one: a sleeper is woken only when actually disturbed
/// (a fast approaching neighbour, a change in its contact set, or a moving wall).
///
/// Single-GPU only (gravity on, no external force). Under MPI / CFD-DEM drag the caller leaves
/// sleeping disabled, so this whole path is inert. Default OFF (set_sleeping / PECLET_DEM_SLEEP).
#ifndef DEM_SLEEPING_HPP
#define DEM_SLEEPING_HPP

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ManifoldC, CpExec/CpMem
#include "dem_portable.hpp"

namespace peclet::dem {

/// After predictVelocity: re-freeze the currently-asleep bodies so gravity/prediction do not move
/// them — velPred = angVelPred = 0, posPred = pos, quatPred = quat.
inline void freezeAsleepKokkos(int numReal, Kokkos::View<const unsigned char*, CpMem> asleep,
                               Kokkos::View<const float* [3], CpMem> pos,
                               Kokkos::View<const float* [4], CpMem> quat,
                               Kokkos::View<float* [3], CpMem> posPred,
                               Kokkos::View<float* [4], CpMem> quatPred,
                               Kokkos::View<float* [3], CpMem> velPred,
                               Kokkos::View<float* [3], CpMem> angVelPred) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::freeze_asleep", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        if (!asleep(i))
          return;
        for (int c = 0; c < 3; ++c) {
          posPred(i, c) = pos(i, c);
          velPred(i, c) = 0.0f;
          angVelPred(i, c) = 0.0f;
        }
        for (int c = 0; c < 4; ++c)
          quatPred(i, c) = quat(i, c);
      });
  space.fence();
}

/// Effective inverse mass for the solve: a sleeping body (real, or a periodic ghost whose real is
/// asleep) is immovable — invMassEff = 0. Everything else keeps its real invMass. Built over the
/// full body-slot span (owned + ghosts); the driver's swap points every solve kernel at it.
inline void buildInvMassEffKokkos(int numBodies, Kokkos::View<const unsigned char*, CpMem> asleep,
                                  Kokkos::View<const int*, CpMem> realIdx,
                                  Kokkos::View<const float*, CpMem> invMass,
                                  Kokkos::View<float*, CpMem> invMassEff) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::inv_mass_eff", Kokkos::RangePolicy<CpExec>(space, 0, numBodies),
      KOKKOS_LAMBDA(int i) { invMassEff(i) = asleep(realIdx(i)) ? 0.0f : invMass(i); });
  space.fence();
}

/// Per-manifold "both endpoints asleep" flag (a static wall, bodyB < 0, counts as asleep). Such
/// manifolds are excluded from colouring / sweeps / the multilevel hierarchy but STILL carry their
/// frozen force network through the warm-start ledger (the caller never skips the ledger kernels).
inline void computeManifoldSleepKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                       int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                       Kokkos::View<const unsigned char*, CpMem> asleep,
                                       Kokkos::View<unsigned char*, CpMem> manifoldSleep) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::manifold_sleep", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        unsigned char s = 0;
        if (m.num_points > 0) {
          const bool aA = asleep(realIdx(m.bodyA)) != 0;
          const bool aB = (m.bodyB < 0) ? true : (asleep(realIdx(m.bodyB)) != 0);
          s = (aA && aB) ? 1 : 0;
        }
        manifoldSleep(idx) = s;
      });
  space.fence();
}

/// Per-contact twin of computeManifoldSleepKokkos for the position colouring.
inline void computeContactSleepKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                      int numContacts, Kokkos::View<const int*, CpMem> realIdx,
                                      Kokkos::View<const unsigned char*, CpMem> asleep,
                                      Kokkos::View<unsigned char*, CpMem> contactSleep) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::contact_sleep", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        const ContactC c = contacts(idx);
        const bool aA = asleep(realIdx(c.bodyA)) != 0;
        const bool aB = (c.bodyB < 0) ? true : (asleep(realIdx(c.bodyB)) != 0);
        contactSleep(idx) = (aA && aB) ? 1 : 0;
      });
  space.fence();
}

/// Wake pass (before the solve): a sleeper is woken when actually disturbed. Over this substep's
/// manifolds — (a) a manifold pairing it with an AWAKE body whose speed exceeds wakeSpeed wakes it;
/// (c) a contact with a MOVING wall (wallVel != 0) wakes it (and flags it never-sleep this
/// substep). Also tallies each real body's live contact count into curCount for the caller's
/// contact-set-change wake rule (b). velPred is the post-predict / post-freeze velocity (awake
/// bodies carry their real speed, sleepers carry 0).
inline void wakeDisturbedKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
                                Kokkos::View<const int*, CpMem> realIdx,
                                Kokkos::View<const float* [3], CpMem> velPred, float wakeSpeed,
                                Kokkos::View<unsigned char*, CpMem> asleep,
                                Kokkos::View<unsigned char*, CpMem> sleepCounter,
                                Kokkos::View<unsigned char*, CpMem> movingWall,
                                Kokkos::View<int*, CpMem> curCount, int numReal) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::wake_reset", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        movingWall(i) = 0;
        curCount(i) = 0;
      });
  const float wakeSpeed2 = wakeSpeed * wakeSpeed;
  Kokkos::parallel_for(
      "peclet::dem::wake_disturbed", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;
        const int ra = realIdx(m.bodyA);
        Kokkos::atomic_add(&curCount(ra), 1);
        int rb = -1;
        if (m.bodyB >= 0) {
          rb = realIdx(m.bodyB);
          if (rb == ra)
            return;  // periodic self-pair
          Kokkos::atomic_add(&curCount(rb), 1);
        }
        // Moving wall (boundary manifold with nonzero wall velocity): never sleeps.
        if (m.bodyB < 0) {
          const F4 wv = m.wallVel_sum;
          if (wv.x != 0.0f || wv.y != 0.0f || wv.z != 0.0f) {
            movingWall(ra) = 1;
            if (asleep(ra)) {
              asleep(ra) = 0;
              sleepCounter(ra) = 0;
            }
          }
          return;
        }
        const bool sA = asleep(ra) != 0, sB = asleep(rb) != 0;
        if (sA == sB)
          return;  // both awake or both asleep: no cross-wake here
        // Exactly one asleep: wake it if the awake side moves faster than wakeSpeed.
        const int awake = sA ? rb : ra, sleeper = sA ? ra : rb;
        const float vx = velPred(awake, 0), vy = velPred(awake, 1), vz = velPred(awake, 2);
        if (vx * vx + vy * vy + vz * vz > wakeSpeed2) {
          asleep(sleeper) = 0;
          sleepCounter(sleeper) = 0;
        }
      });
  space.fence();
}

/// Contact-set-change wake rule (b): wake any still-asleep body whose live contact count differs
/// from the stored one (a support gained or lost), then refresh the store. Also clears the counter
/// of a moving-wall body so it can never accumulate toward sleep.
inline void wakeContactChangeKokkos(int numReal, Kokkos::View<const int*, CpMem> curCount,
                                    Kokkos::View<int*, CpMem> prevCount,
                                    Kokkos::View<const unsigned char*, CpMem> movingWall,
                                    Kokkos::View<unsigned char*, CpMem> asleep,
                                    Kokkos::View<unsigned char*, CpMem> sleepCounter,
                                    bool wakeOnChange) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::wake_contact_change", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        // Rule (b): a change in the live contact count. In a near-threshold settled bed contacts
        // FLICKER in and out every substep, so waking on any change stops the bed ever freezing;
        // waking only on a LOST contact (support removed), and only when enabled, is the robust
        // form. A gained contact is extra support and never a reason to wake.
        if (wakeOnChange && asleep(i) && curCount(i) < prevCount(i)) {
          asleep(i) = 0;
          sleepCounter(i) = 0;
        }
        if (movingWall(i))
          sleepCounter(i) = 0;
        prevCount(i) = curCount(i);
      });
  space.fence();
}

/// Sleep detection (after the commit): an AWAKE, grounded body whose linear AND angular motion has
/// stayed below sleepSpeed for K consecutive substeps is put to sleep (velocity + spin zeroed).
/// Any body at/above the threshold, ungrounded, or on a moving wall resets its counter. `vel` /
/// `angVel` are the committed velocities of this substep.
inline void updateSleepKokkos(int numReal, Kokkos::View<const float* [3], CpMem> vel,
                              Kokkos::View<const float* [3], CpMem> angVel,
                              Kokkos::View<const float*, CpMem> rad,
                              Kokkos::View<const unsigned char*, CpMem> grounded,
                              Kokkos::View<const unsigned char*, CpMem> movingWall,
                              Kokkos::View<unsigned char*, CpMem> asleep,
                              Kokkos::View<unsigned char*, CpMem> sleepCounter, float sleepSpeed,
                              int K, Kokkos::View<float* [3], CpMem> velOut,
                              Kokkos::View<float* [3], CpMem> angVelOut) {
  CpExec space;
  const float s2 = sleepSpeed * sleepSpeed;
  Kokkos::parallel_for(
      "peclet::dem::update_sleep", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        if (asleep(i))
          return;  // stays asleep until an explicit wake
        const float vx = vel(i, 0), vy = vel(i, 1), vz = vel(i, 2);
        const float wx = angVel(i, 0), wy = angVel(i, 1), wz = angVel(i, 2);
        const float r = rad(i);
        const bool lowLin = (vx * vx + vy * vy + vz * vz) < s2;
        const bool lowAng = (wx * wx + wy * wy + wz * wz) * r * r < s2;
        if (lowLin && lowAng && grounded(i) > 0 && !movingWall(i)) {
          int c = static_cast<int>(sleepCounter(i)) + 1;
          if (c >= K) {
            asleep(i) = 1;
            for (int k = 0; k < 3; ++k) {
              velOut(i, k) = 0.0f;
              angVelOut(i, k) = 0.0f;
            }
            c = K;  // saturate the counter
          }
          sleepCounter(i) = static_cast<unsigned char>(c > 255 ? 255 : c);
        } else {
          sleepCounter(i) = 0;
        }
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SLEEPING_HPP
