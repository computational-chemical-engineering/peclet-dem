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
                                float restitutionNormal, float restVelThreshold,
                                Kokkos::View<float* [3], CpMem> deltaVel,
                                Kokkos::View<float* [3], CpMem> deltaAngVel,
                                Kokkos::View<int*, CpMem> velCounts) {
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

        // Moving-wall boundary (idB<0): the static "body B" carries the wall's surface velocity
        // (count-averaged over the contact patch) so restitution is against the wall's motion, and
        // its own restitution (a < 0 average keeps the global material — planes, body-body).
        float restitution = restitutionNormal;
        if (idB < 0) {
          vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
          const float ra = m.restitution_sum * invN;
          if (ra >= 0.0f)
            restitution = ra;
        }
        const F3 rAavg = scale3(F3{m.rA_sum.x, m.rA_sum.y, m.rA_sum.z}, invN);
        const F3 rBavg = scale3(F3{m.rB_sum.x, m.rB_sum.y, m.rB_sum.z}, invN);

        const float lenN = Kokkos::sqrt(dot3(Nsum, Nsum));
        if (lenN < 1e-9f)
          return;

        // Separation vector for the approaching-sign gate + growth velocity. For a BOUNDARY (idB<0),
        // rBavg is the ABSOLUTE wall contact point (kept absolute for the position solve's plane
        // linearisation), NOT a body-relative lever — so rAavg - rBavg would depend on where the
        // contact sits in world space, flipping `alignment` (and thus the approaching test) around a
        // curved wall / a wall far from the origin and injecting energy (grains "jump" on the way
        // down a rotating drum). The grain's own contact lever rAavg is the meaningful relative
        // vector (dot(Nsum, rAavg) = radius > 0, a consistent convention). Body-body keeps rAavg-rBavg.
        const F3 diffCenters = (idB < 0) ? rAavg : sub3(rAavg, rBavg);
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

        // Resting-contact regularization (the standard PBD/XPBD restitution threshold): bounce only
        // when the physical approach speed |vn|/|Nsum| exceeds ~2 g dt (what one substep of free
        // fall gains). Below it the contact is RESTING — its vn is integration noise, and bouncing
        // it every substep across a dense pile's contact chains (impulses Jacobi-SUMMED per body,
        // no mass splitting) pumps energy without bound: a settled 180k glass-bead bed switched to
        // e=0.8 reached |v| ~ 3e6 cells/s within 15 substeps. With e=0 the impulse still cancels
        // the approach velocity — exactly the quasi-static dissipation a resting pile needs.
        if (Kokkos::fabs(vn) < restVelThreshold * lenN)
          restitution = 0.0f;

        const float lambda = (-restitution * vn - vn) / wTotal;

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
          Kokkos::atomic_add(&velCounts(realB), 1);
        }
        Kokkos::atomic_add(&velCounts(realA), 1);
      });
  space.fence();
}

/// Apply the accumulated velocity deltas AVERAGED by the per-body manifold count — the velocity-
/// solve twin of the position solve's constraint-count averaging (applyUpdatesKokkos). A body in a
/// dense pile receives one full-strength impulse per touching manifold; the raw Jacobi SUM
/// overshoots by ~the contact count and diverges hard for e ≳ 0.5 (a settled 180k glass-bead bed
/// switched to e=0.8 grew |v| by ~5x per substep). A lone binary collision (count 1) is unchanged.
template <class V3, class Vi>
inline void applyVelocityDeltasAveragedKokkos(int n, V3 velPred, V3 angVelPred, V3 deltaVel,
                                              V3 deltaAngVel, Vi velCounts) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::apply_vel_avg", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        const int count = velCounts(i);
        if (count <= 0)
          return;
        // Over-relaxed average: omega=2 halves the convergence loss of plain 1/count averaging
        // (the crush-dissipation rate) while staying far below the raw-sum overshoot (omega=count)
        // that detonates a resting pile at e=0.8. count==1 (binary collision) stays exact.
        const float f = Kokkos::fmin(1.0f, 2.0f / static_cast<float>(count));
        for (int c = 0; c < 3; ++c) {
          velPred(i, c) += deltaVel(i, c) * f;
          angVelPred(i, c) += deltaAngVel(i, c) * f;
          deltaVel(i, c) = 0.0f;
          deltaAngVel(i, c) = 0.0f;
        }
        velCounts(i) = 0;
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_VELOCITY_HPP
