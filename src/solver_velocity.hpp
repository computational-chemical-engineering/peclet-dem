/// @file
/// @brief dem — portable (Kokkos) manifold velocity solve (normal restitution impulse).
///
/// Kokkos port of solve_velocity_jacobi_kernel (solver_velocity.cu): one thread per manifold
/// computes the aggregate normal impulse (with growth-velocity term and restitution) and atomically
/// scatters the linear/angular velocity deltas onto the two bodies' REAL indices. Faithful copy of
/// the CUDA math; runs on the particle SoA expressed as Kokkos Views. Friction is a separate
/// per-contact pass (solver_position) and is not done here, matching the original.
#ifndef DEM_SOLVER_VELOCITY_HPP
#define DEM_SOLVER_VELOCITY_HPP

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ManifoldC, CpExec/CpMem
#include "dem_portable.hpp"

namespace peclet::dem {

namespace detail {
KOKKOS_INLINE_FUNCTION F3 ld3(Kokkos::View<const float* [3], CpMem> v, int i) {
  return F3{v(i, 0), v(i, 1), v(i, 2)};
}
// v^T I_world^-1 v with I_world^-1 = R I_local^-1 R^T  ->  (R^T v) diag(invI_local) (R^T v).
KOKKOS_INLINE_FUNCTION float genInvMass(F3 tau, F3 invIlocal, F4 q) {
  const F3 t = invRotateVector(q, tau);
  return t.x * t.x * invIlocal.x + t.y * t.y * invIlocal.y + t.z * t.z * invIlocal.z;
}
}  // namespace detail

/// Accumulate normal-restitution velocity deltas for `numManifolds` manifolds.
inline void solveVelocityKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
                                Kokkos::View<const float*, CpMem> invMass,
                                Kokkos::View<const float* [3], CpMem> invInertia,
                                Kokkos::View<const float* [4], CpMem> quat,
                                Kokkos::View<const float* [3], CpMem> velPred,
                                Kokkos::View<const float* [3], CpMem> angVelPred,
                                Kokkos::View<const int*, CpMem> realIdx, float growthRate,
                                float restitutionNormal, Kokkos::View<float* [3], CpMem> deltaVel,
                                Kokkos::View<float* [3], CpMem> deltaAngVel) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::solve_velocity", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;

        const int idA = m.bodyA, idB = m.bodyB;
        const int realA = realIdx(idA);
        int realB = idB;
        if (idB >= 0) {
          realB = realIdx(idB);
          if (realA > realB)
            return;  // periodic dedup
        }

        const float invMassA = invMass(realA);
        const float invMassB = (idB >= 0) ? invMass(realB) : 0.0f;
        const F3 invIA = ld3(invInertia, realA);
        const F3 invIB = (idB >= 0) ? ld3(invInertia, realB) : F3{0, 0, 0};
        const F4 qA = F4{quat(realA, 0), quat(realA, 1), quat(realA, 2), quat(realA, 3)};
        const F4 qB = (idB >= 0)
                          ? F4{quat(realB, 0), quat(realB, 1), quat(realB, 2), quat(realB, 3)}
                          : F4{0, 0, 0, 1};

        const F3 vA = ld3(velPred, realA), wA = ld3(angVelPred, realA);
        F3 vB{0, 0, 0}, wB{0, 0, 0};
        if (idB >= 0) {
          vB = ld3(velPred, realB);
          wB = ld3(angVelPred, realB);
        }

        const F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
        const F3 TauA{m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z};
        const F3 TauB{m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z};

        const float invN = 1.0f / static_cast<float>(m.num_points);
        const F3 rAavg = scale3(F3{m.rA_sum.x, m.rA_sum.y, m.rA_sum.z}, invN);
        const F3 rBavg = scale3(F3{m.rB_sum.x, m.rB_sum.y, m.rB_sum.z}, invN);

        const float lenN = Kokkos::sqrt(dot3(Nsum, Nsum));
        if (lenN < 1e-9f)
          return;

        const F3 diffCenters = sub3(rAavg, rBavg);
        const F3 vGrowth = scale3(diffCenters, growthRate);

        float vn = dot3(vA, Nsum) + dot3(wA, TauA) + dot3(vB, F3{-Nsum.x, -Nsum.y, -Nsum.z}) +
                   dot3(wB, TauB);
        vn += dot3(vGrowth, Nsum);

        const float alignment = dot3(Nsum, diffCenters);
        if (alignment > 0.0f) {
          if (vn < 0.0f)
            return;  // sphere convention: approaching if vn>0
        } else {
          if (vn > 0.0f)
            return;  // inverted convention
        }

        const float Nsq = dot3(Nsum, Nsum);
        const float wA_n = Nsq * invMassA + genInvMass(TauA, invIA, qA);
        const float wB_n = Nsq * invMassB + genInvMass(TauB, invIB, qB);
        const float wTotal = wA_n + wB_n;
        if (wTotal <= 0.0f)
          return;

        const float lambda = (-restitutionNormal * vn - vn) / wTotal;

        const F3 Jlin = scale3(Nsum, lambda);
        const F3 JangA = scale3(TauA, lambda);
        const F3 JangB = scale3(TauB, lambda);

        // Linear delta on A.
        Kokkos::atomic_add(&deltaVel(realA, 0), Jlin.x * invMassA);
        Kokkos::atomic_add(&deltaVel(realA, 1), Jlin.y * invMassA);
        Kokkos::atomic_add(&deltaVel(realA, 2), Jlin.z * invMassA);
        // Angular delta on A: dw_world = R (invI_local * (R^T Jang)).
        {
          const F3 Jl = invRotateVector(qA, JangA);
          const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
          const F3 dww = rotateVector(qA, dwl);
          Kokkos::atomic_add(&deltaAngVel(realA, 0), dww.x);
          Kokkos::atomic_add(&deltaAngVel(realA, 1), dww.y);
          Kokkos::atomic_add(&deltaAngVel(realA, 2), dww.z);
        }
        if (idB >= 0) {
          Kokkos::atomic_add(&deltaVel(realB, 0), -Jlin.x * invMassB);
          Kokkos::atomic_add(&deltaVel(realB, 1), -Jlin.y * invMassB);
          Kokkos::atomic_add(&deltaVel(realB, 2), -Jlin.z * invMassB);
          const F3 Jl = invRotateVector(qB, JangB);
          const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
          const F3 dww = rotateVector(qB, dwl);
          Kokkos::atomic_add(&deltaAngVel(realB, 0), dww.x);
          Kokkos::atomic_add(&deltaAngVel(realB, 1), dww.y);
          Kokkos::atomic_add(&deltaAngVel(realB, 2), dww.z);
        }
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_VELOCITY_HPP
