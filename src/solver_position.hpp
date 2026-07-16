/// @file
/// @brief dem — portable (Kokkos) XPBD position solve (pure overlap removal).
///
/// Kokkos port of solve_position_jacobi_kernel (solver_position.cu): one thread per contact
/// evaluates the linearized non-penetration constraint C(x) (using the delta-rotated lever arms /
/// normal), computes the XPBD position correction, and atomically scatters delta_pos / delta_quat
/// and bumps the per-body constraint count (the caller averages by it). Faithful copy of the CUDA
/// math over the SoA Views. Friction is a separate cluster (ported separately); the position solve
/// does overlap only.
#ifndef DEM_SOLVER_POSITION_HPP
#define DEM_SOLVER_POSITION_HPP

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ContactC, CpExec/CpMem
#include "dem_portable.hpp"

namespace peclet::dem {

namespace detail {
KOKKOS_INLINE_FUNCTION float computeW(F3 r, F3 dir, float invM, F3 invI) {
  const F3 rn = cross3v(r, dir);
  return invM + rn.x * rn.x * invI.x + rn.y * rn.y * invI.y + rn.z * rn.z * invI.z;
}
KOKKOS_INLINE_FUNCTION F4 deltaQuat(F3 dTheta, F4 q) {
  return F4{0.5f * (dTheta.x * q.w + dTheta.y * q.z - dTheta.z * q.y),
            0.5f * (dTheta.y * q.w + dTheta.z * q.x - dTheta.x * q.z),
            0.5f * (dTheta.z * q.w + dTheta.x * q.y - dTheta.y * q.x),
            0.5f * (-dTheta.x * q.x - dTheta.y * q.y - dTheta.z * q.z)};
}
}  // namespace detail

/// Accumulate XPBD position corrections for `numContacts` contacts.
inline void solvePositionKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int numContacts,
                                Kokkos::View<const float*, CpMem> invMass,
                                Kokkos::View<const float* [3], CpMem> posPred,
                                Kokkos::View<const float* [4], CpMem> quatPred,
                                Kokkos::View<const float* [4], CpMem> quatStatic,
                                Kokkos::View<const float* [3], CpMem> invInertia,
                                Kokkos::View<float* [3], CpMem> deltaPos,
                                Kokkos::View<float* [4], CpMem> deltaQuat,
                                Kokkos::View<int*, CpMem> constraintCounts,
                                Kokkos::View<float, CpMem> maxOverlap,
                                Kokkos::View<const int*, CpMem> onlyColor = {},
                                int colorFilter = 0) {
  using detail::computeW;
  CpExec space;
  const bool filt = onlyColor.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::solve_position", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        if (filt && onlyColor(idx) != colorFilter)
          return;  // Jacobi fallback pass: only the contacts the colouring could not place
        const ContactC c = contacts(idx);
        const int idA = c.bodyA, idB = c.bodyB;
        const float invMassA = invMass(idA);
        const float invMassB = (idB >= 0) ? invMass(idB) : 0.0f;

        const F3 pA = ldF3(posPred, idA);
        const F4 qA = ldF4(quatPred, idA);
        F3 pB{0, 0, 0};
        F4 qB{0, 0, 0, 1};
        if (idB >= 0) {
          pB = ldF3(posPred, idB);
          qB = ldF4(quatPred, idB);
        }

        // Delta-rotate the stored lever arms / normal from the static frame to the predicted one.
        const F4 qAdelta = quatMult(qA, quatInverse(ldF4(quatStatic, idA)));
        F3 rA = rotateVector(qAdelta, F3{c.rA.x, c.rA.y, c.rA.z});
        F3 rB{c.rB.x, c.rB.y, c.rB.z};
        F3 n{c.normal.x, c.normal.y, c.normal.z};
        if (idB >= 0) {
          const F4 qBdelta = quatMult(qB, quatInverse(ldF4(quatStatic, idB)));
          rB = rotateVector(qBdelta, rB);
          n = rotateVector(qBdelta, n);
        }

        float C;
        if (idB < 0) {
          n = F3{c.normal.x, c.normal.y, c.normal.z};  // wall normal is static
          const F3 pAsurf = add3(pA, rA);
          C = dot3(sub3(pAsurf, F3{c.rB.x, c.rB.y, c.rB.z}), n);
        } else {
          const F3 pAc = add3(pA, rA);
          const F3 pBc = add3(pB, rB);
          C = dot3(sub3(pAc, pBc), n);
        }
        if (C >= 0.0f)
          return;

        const F3 invIA = ldF3(invInertia, idA);
        const F3 invIB = (idB >= 0) ? ldF3(invInertia, idB) : F3{0, 0, 0};
        const float wTotal = computeW(rA, n, invMassA, invIA) + computeW(rB, n, invMassB, invIB);
        if (wTotal < 1e-6f)
          return;

        const float dLambda = -C / wTotal;

        // Linear + angular correction on A.
        Kokkos::atomic_add(&deltaPos(idA, 0), n.x * dLambda * invMassA);
        Kokkos::atomic_add(&deltaPos(idA, 1), n.y * dLambda * invMassA);
        Kokkos::atomic_add(&deltaPos(idA, 2), n.z * dLambda * invMassA);
        {
          const F3 rn = cross3v(rA, n);
          const F3 dTheta{rn.x * invIA.x * dLambda, rn.y * invIA.y * dLambda,
                          rn.z * invIA.z * dLambda};
          const F4 dq = detail::deltaQuat(dTheta, qA);
          Kokkos::atomic_add(&deltaQuat(idA, 0), dq.x);
          Kokkos::atomic_add(&deltaQuat(idA, 1), dq.y);
          Kokkos::atomic_add(&deltaQuat(idA, 2), dq.z);
          Kokkos::atomic_add(&deltaQuat(idA, 3), dq.w);
        }
        if (idB >= 0) {
          Kokkos::atomic_add(&deltaPos(idB, 0), -n.x * dLambda * invMassB);
          Kokkos::atomic_add(&deltaPos(idB, 1), -n.y * dLambda * invMassB);
          Kokkos::atomic_add(&deltaPos(idB, 2), -n.z * dLambda * invMassB);
          const F3 rn = cross3v(rB, n);
          const F3 dTheta{-rn.x * invIB.x * dLambda, -rn.y * invIB.y * dLambda,
                          -rn.z * invIB.z * dLambda};
          const F4 dq = detail::deltaQuat(dTheta, qB);
          Kokkos::atomic_add(&deltaQuat(idB, 0), dq.x);
          Kokkos::atomic_add(&deltaQuat(idB, 1), dq.y);
          Kokkos::atomic_add(&deltaQuat(idB, 2), dq.z);
          Kokkos::atomic_add(&deltaQuat(idB, 3), dq.w);
          Kokkos::atomic_add(&constraintCounts(idB), 1);
        }
        Kokkos::atomic_add(&constraintCounts(idA), 1);

        if (C < 0.0f)
          Kokkos::atomic_max(&maxOverlap(), -C);
      });
  space.fence();
}

// ============================ colored Gauss–Seidel position solve ============================
// The overlap solve above is the position-side twin of the Jacobi restitution solve: one thread per
// contact scatters an XPBD non-penetration correction, and the caller relaxes the per-body SUM by the
// contact count (applyUpdatesKokkos, 1/count). Same trade-off — stable but under-converged, so a body
// wedged by many neighbours keeps residual overlap. The colored path removes the averaging: graph-
// colour the CONTACT graph (vertices = body slots, edges = contacts; raw bodyA/bodyB, the same
// indices the solve writes) so no two contacts sharing a body get one colour, then sweep colour-by-
// colour applying each correction IN PLACE. Within a colour the contacts are an independent set, so
// the read-modify-write is race-free without atomics, and each sweep sees the previous colours'
// moves — a true sequential projection that resolves stacked contacts far better per iteration.
// Translation only, matching the Jacobi path (applyUpdatesKokkos applies deltaPos, not deltaQuat; the
// angular contact response lives in the velocity solve), and it never touches velocity — overlap
// removal stays decoupled from the velocity update.

/// Greedy graph-colour the contacts (raw bodyA/bodyB): no two contacts sharing a body get the same
/// colour. Round-based max-index arbitration, identical machinery to colorManifoldsKokkos but over
/// the per-contact graph with direct (non-realIdx) body slots; every contact is active. `numBodies`
/// is the body-slot count (numParticles, incl. periodic ghosts, which the position solve corrects
/// independently). Returns the number of colours used.
inline int colorContactsKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int numContacts,
                               int numBodies, Kokkos::View<int*, CpMem> cColor,
                               Kokkos::View<long long*, CpMem> bodyWinner,
                               Kokkos::View<std::uint64_t*, CpMem> bodyMask, int& leftover) {
  leftover = 0;
  CpExec space;
  if (numContacts <= 0 || numBodies <= 0)
    return 0;
  Kokkos::parallel_for(
      "peclet::dem::pcolor_init_bodies", Kokkos::RangePolicy<CpExec>(space, 0, numBodies),
      KOKKOS_LAMBDA(int i) { bodyMask(i) = 0; });
  Kokkos::parallel_for(
      "peclet::dem::pcolor_init_contacts", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) { cColor(idx) = -1; });

  int remaining = 1, prevRemaining = -1;
  const int maxRounds = numBodies + 2;
  for (int round = 0; round < maxRounds && remaining > 0; ++round) {
    Kokkos::parallel_for(
        "peclet::dem::pcolor_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, numBodies),
        KOKKOS_LAMBDA(int i) { bodyWinner(i) = -1; });
    Kokkos::parallel_for(
        "peclet::dem::pcolor_contend", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx) {
          if (cColor(idx) != -1)
            return;
          const ContactC c = contacts(idx);
          const long long key = colorKey(idx);  // hashed priority (see solver_velocity.hpp)
          Kokkos::atomic_max(&bodyWinner(c.bodyA), key);
          if (c.bodyB >= 0)
            Kokkos::atomic_max(&bodyWinner(c.bodyB), key);
        });
    int rem = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::pcolor_commit", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx, int& acc) {
          if (cColor(idx) != -1)
            return;
          const ContactC c = contacts(idx);
          const int ea = c.bodyA;
          const int eb = c.bodyB;  // <0 for a wall/boundary contact
          const long long key = colorKey(idx);
          if (bodyWinner(ea) != key || (eb >= 0 && bodyWinner(eb) != key)) {
            acc += 1;
            return;
          }
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int col = 0;
          while (col < 62 && (forbidden & (std::uint64_t(1) << col)))
            ++col;
          cColor(idx) = col;
          const std::uint64_t bit = std::uint64_t(1) << col;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    // Stall detection: a body whose 62-colour mask fills (interpenetration degree > 62) can never
    // host a new colour — without this break the loop would spin maxRounds (~numBodies) times doing
    // nothing. The stuck contacts stay -1 and are handled by the Jacobi fallback in the solve.
    if (rem == prevRemaining)
      break;
    prevRemaining = rem;
    remaining = rem;
  }

  int maxc = -1;
  Kokkos::parallel_reduce(
      "peclet::dem::pcolor_max", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx, int& mx) {
        if (cColor(idx) > mx)
          mx = cColor(idx);
      },
      Kokkos::Max<int>(maxc));
  Kokkos::parallel_reduce(
      "peclet::dem::pcolor_leftover", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx, int& acc) {
        if (cColor(idx) == -1)
          acc += 1;
      },
      leftover);
  space.fence();
  return maxc + 1;
}

/// Colored Gauss–Seidel XPBD overlap solve: sweep the `numColors` colour classes in order, applying
/// each contact's non-penetration correction directly to posPred (in place, translation only). Same
/// per-contact math as solvePositionKokkos — only the write-back differs (in-place RMW instead of
/// atomic-accumulate + count-average). Race-free because a colour is an independent set of contacts.
/// One outer call = one full sweep over all colours; the caller loops it positionIterations times.
inline void solvePositionColoredGSKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                         int numContacts, Kokkos::View<const int*, CpMem> cColor,
                                         int numColors, Kokkos::View<const float*, CpMem> invMass,
                                         Kokkos::View<float* [3], CpMem> posPred,
                                         Kokkos::View<const float* [4], CpMem> quatPred,
                                         Kokkos::View<const float* [4], CpMem> quatStatic,
                                         Kokkos::View<const float* [3], CpMem> invInertia,
                                         Kokkos::View<float, CpMem> maxOverlap) {
  using detail::computeW;
  CpExec space;
  for (int color = 0; color < numColors; ++color) {
    Kokkos::parallel_for(
        "peclet::dem::solve_position_gs", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx) {
          if (cColor(idx) != color)
            return;
          const ContactC c = contacts(idx);
          const int idA = c.bodyA, idB = c.bodyB;
          const float invMassA = invMass(idA);
          const float invMassB = (idB >= 0) ? invMass(idB) : 0.0f;

          const F3 pA = ldF3(posPred, idA);
          const F4 qA = ldF4(quatPred, idA);
          F3 pB{0, 0, 0};
          F4 qB{0, 0, 0, 1};
          if (idB >= 0) {
            pB = ldF3(posPred, idB);
            qB = ldF4(quatPred, idB);
          }

          const F4 qAdelta = quatMult(qA, quatInverse(ldF4(quatStatic, idA)));
          F3 rA = rotateVector(qAdelta, F3{c.rA.x, c.rA.y, c.rA.z});
          F3 rB{c.rB.x, c.rB.y, c.rB.z};
          F3 n{c.normal.x, c.normal.y, c.normal.z};
          if (idB >= 0) {
            const F4 qBdelta = quatMult(qB, quatInverse(ldF4(quatStatic, idB)));
            rB = rotateVector(qBdelta, rB);
            n = rotateVector(qBdelta, n);
          }

          float C;
          if (idB < 0) {
            n = F3{c.normal.x, c.normal.y, c.normal.z};
            const F3 pAsurf = add3(pA, rA);
            C = dot3(sub3(pAsurf, F3{c.rB.x, c.rB.y, c.rB.z}), n);
          } else {
            const F3 pAc = add3(pA, rA);
            const F3 pBc = add3(pB, rB);
            C = dot3(sub3(pAc, pBc), n);
          }
          if (C >= 0.0f)
            return;

          const F3 invIA = ldF3(invInertia, idA);
          const F3 invIB = (idB >= 0) ? ldF3(invInertia, idB) : F3{0, 0, 0};
          const float wTotal = computeW(rA, n, invMassA, invIA) + computeW(rB, n, invMassB, invIB);
          if (wTotal < 1e-6f)
            return;
          const float dLambda = -C / wTotal;

          // Translation-only correction, in place (rotation discarded to match applyUpdatesKokkos).
          posPred(idA, 0) += n.x * dLambda * invMassA;
          posPred(idA, 1) += n.y * dLambda * invMassA;
          posPred(idA, 2) += n.z * dLambda * invMassA;
          if (idB >= 0) {
            posPred(idB, 0) += -n.x * dLambda * invMassB;
            posPred(idB, 1) += -n.y * dLambda * invMassB;
            posPred(idB, 2) += -n.z * dLambda * invMassB;
          }
          Kokkos::atomic_max(&maxOverlap(), -C);
        });
    // Stream-ordered on the device, so colour c+1 already sees colour c's moves — no host fence per
    // colour (that would only stall the host). One fence after the full sweep.
  }
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_POSITION_HPP
