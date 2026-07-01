/// @file
/// @brief dem — portable (Kokkos) XPBD position solve (pure overlap removal).
///
/// Kokkos port of solve_position_jacobi_kernel (solver_position.cu): one thread per contact evaluates
/// the linearized non-penetration constraint C(x) (using the delta-rotated lever arms / normal),
/// computes the XPBD position correction, and atomically scatters delta_pos / delta_quat and bumps the
/// per-body constraint count (the caller averages by it). Faithful copy of the CUDA math over the SoA
/// Views. Friction is a separate cluster (ported separately); the position solve does overlap only.
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
                                Kokkos::View<float, CpMem> maxOverlap) {
  using detail::computeW;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::solve_position", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
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
        if (C >= 0.0f) return;

        const F3 invIA = ldF3(invInertia, idA);
        const F3 invIB = (idB >= 0) ? ldF3(invInertia, idB) : F3{0, 0, 0};
        const float wTotal =
            computeW(rA, n, invMassA, invIA) + computeW(rB, n, invMassB, invIB);
        if (wTotal < 1e-6f) return;

        const float dLambda = -C / wTotal;

        // Linear + angular correction on A.
        Kokkos::atomic_add(&deltaPos(idA, 0), n.x * dLambda * invMassA);
        Kokkos::atomic_add(&deltaPos(idA, 1), n.y * dLambda * invMassA);
        Kokkos::atomic_add(&deltaPos(idA, 2), n.z * dLambda * invMassA);
        {
          const F3 rn = cross3v(rA, n);
          const F3 dTheta{rn.x * invIA.x * dLambda, rn.y * invIA.y * dLambda, rn.z * invIA.z * dLambda};
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

        if (C < 0.0f) Kokkos::atomic_max(&maxOverlap(), -C);
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_POSITION_HPP
