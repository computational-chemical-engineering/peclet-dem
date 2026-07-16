/// @file
/// @brief dem — portable (Kokkos) time integration kernels (integration.cu).
///
/// The per-particle element-wise steps of the XPBD step over the particle SoA Views: velocity
/// predict (gravity + gyroscopic Euler term), re-integration (apply velocity deltas, trapezoidal
/// position predict, quaternion integrate+normalize), iterative delta apply, Jacobi count-averaged
/// update, final commit with periodic wrap, contact-count pre-pass, and growth-scale update.
/// Faithful copies of the CUDA kernels. Ghost generation and the thermostat are separate units.
#ifndef DEM_INTEGRATION_HPP
#define DEM_INTEGRATION_HPP

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ContactC, CpExec/CpMem
#include "dem_portable.hpp"

namespace peclet::dem {

struct Domain {
  F3 min, max, size;
  bool periodic_x, periodic_y, periodic_z;
};

using V3 = Kokkos::View<float* [3], CpMem>;
using V4 = Kokkos::View<float* [4], CpMem>;
using Vf = Kokkos::View<float*, CpMem>;
using Vi = Kokkos::View<int*, CpMem>;

namespace detail {
KOKKOS_INLINE_FUNCTION void st3(const V3& v, int i, F3 a) {
  v(i, 0) = a.x;
  v(i, 1) = a.y;
  v(i, 2) = a.z;
}
KOKKOS_INLINE_FUNCTION void st4(const V4& v, int i, F4 a) {
  v(i, 0) = a.x;
  v(i, 1) = a.y;
  v(i, 2) = a.z;
  v(i, 3) = a.w;
}
}  // namespace detail

/// Predict velocity (gravity + gyroscopic precession), speculative position, and clear all deltas.
inline void predictVelocityKokkos(int n, V3 pos, Vf invMass, V3 vel, V4 quat, V3 angVel,
                                  V3 invInertia, V3 posPred, V4 quatPred, V3 velPred, V3 angVelPred,
                                  V3 deltaPos, V4 deltaQuat, V3 deltaVel, V3 deltaAngVel,
                                  Vi constraintCounts, F3 gravity, float dt, V3 extForce) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::predict_velocity", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        const float invM = invMass(i);
        F3 v = ldF3(vel, i);
        if (invM > 0.0f) {
          v = add3(v, scale3(gravity, dt));
          // external force (fluid drag, etc.): F=ma => dv = F*invMass*dt (extForce is a FORCE, so it
          // scales with 1/mass; gravity above is already an acceleration).
          v = add3(v, scale3(ldF3(extForce, i), invM * dt));
        }
        detail::st3(velPred, i, v);

        F3 wpred = ldF3(angVel, i);
        if (invM > 0.0f) {
          const F3 invI = ldF3(invInertia, i);
          if (invI.x > 0.0f || invI.y > 0.0f || invI.z > 0.0f) {
            const F4 q = ldF4(quat, i);
            F3 wb = invRotateVector(q, ldF3(angVel, i));
            const F3 Ib{(invI.x > 1e-9f) ? 1.0f / invI.x : 0.0f,
                        (invI.y > 1e-9f) ? 1.0f / invI.y : 0.0f,
                        (invI.z > 1e-9f) ? 1.0f / invI.z : 0.0f};
            const F3 Lb{Ib.x * wb.x, Ib.y * wb.y, Ib.z * wb.z};
            const F3 wxL = cross3v(wb, Lb);
            const F3 alpha{-invI.x * wxL.x, -invI.y * wxL.y, -invI.z * wxL.z};
            wb = add3(wb, scale3(alpha, dt));
            wpred = rotateVector(q, wb);
          }
        }
        detail::st3(angVelPred, i, wpred);

        F3 x = ldF3(pos, i);
        if (invM > 0.0f)
          x = add3(x, scale3(v, dt));
        detail::st3(posPred, i, x);
        detail::st4(quatPred, i, ldF4(quat, i));

        detail::st3(deltaVel, i, F3{0, 0, 0});
        detail::st3(deltaAngVel, i, F3{0, 0, 0});
        detail::st3(deltaPos, i, F3{0, 0, 0});
        detail::st4(deltaQuat, i, F4{0, 0, 0, 0});
        constraintCounts(i) = 0;
      });
  space.fence();
}

/// Add accumulated velocity/angular deltas onto the predicted velocity, then clear the delta
/// buffers.
inline void applyVelocityDeltasKokkos(int n, V3 velPred, V3 angVelPred, V3 deltaVel,
                                      V3 deltaAngVel) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::apply_velocity_deltas", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        detail::st3(velPred, i, add3(ldF3(velPred, i), ldF3(deltaVel, i)));
        detail::st3(angVelPred, i, add3(ldF3(angVelPred, i), ldF3(deltaAngVel, i)));
        detail::st3(deltaVel, i, F3{0, 0, 0});
        detail::st3(deltaAngVel, i, F3{0, 0, 0});
      });
  space.fence();
}

/// Re-integration: persist solved velocity, trapezoidal position predict, quaternion integrate.
inline void applyVelocityAndPredictPositionKokkos(int n, V3 pos, Vf invMass, V3 vel, V4 quat,
                                                  V3 velPred, V3 angVelPred, V3 posPred,
                                                  V4 quatPred, V3 angVel, float dt) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::apply_vel_predict_pos", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        const F3 vStart = ldF3(vel, i);
        const F3 vFinal = ldF3(velPred, i);  // already includes deltas
        detail::st3(velPred, i, vFinal);
        detail::st3(vel, i, vFinal);  // persist v_{n+1}

        const float invM = invMass(i);
        F3 x = ldF3(pos, i);
        if (invM > 0.0f)
          x = add3(x, scale3(add3(vStart, vFinal), 0.5f * dt));
        detail::st3(posPred, i, x);

        const F4 q = ldF4(quat, i);
        const F3 omega = ldF3(angVelPred, i);
        detail::st3(angVel, i, omega);
        F4 qp = q;
        if (invM > 0.0f) {
          qp.x += 0.5f * dt * (omega.x * q.w + omega.y * q.z - omega.z * q.y);
          qp.y += 0.5f * dt * (omega.y * q.w + omega.z * q.x - omega.x * q.z);
          qp.z += 0.5f * dt * (omega.z * q.w + omega.x * q.y - omega.y * q.x);
          qp.w += 0.5f * dt * (-omega.x * q.x - omega.y * q.y - omega.z * q.z);
          const float len = Kokkos::sqrt(qp.x * qp.x + qp.y * qp.y + qp.z * qp.z + qp.w * qp.w);
          if (len > 1e-9f) {
            const float s = 1.0f / len;
            qp = F4{qp.x * s, qp.y * s, qp.z * s, qp.w * s};
          }
        }
        detail::st4(quatPred, i, qp);
      });
  space.fence();
}

/// Per-contact constraint-count pre-pass (active contacts, dist <= 0).
inline void computeContactCountsKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                       int numContacts, Vi constraintCounts) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::contact_counts", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int i) {
        const ContactC c = contacts(i);
        if (c.dist > 0.0f)
          return;
        Kokkos::atomic_add(&constraintCounts(c.bodyA), 1);
        if (c.bodyB >= 0)
          Kokkos::atomic_add(&constraintCounts(c.bodyB), 1);
      });
  space.fence();
}

/// Jacobi count-averaged apply of position/velocity deltas, then clear deltas + counts.
inline void applyUpdatesKokkos(int n, V3 posPred, V3 velPred, V3 deltaPos, V3 deltaVel,
                               Vi constraintCounts) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::apply_updates", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        const int count = constraintCounts(i);
        if (count <= 0)
          return;
        const float f = 1.0f / static_cast<float>(count);
        detail::st3(posPred, i, add3(ldF3(posPred, i), scale3(ldF3(deltaPos, i), f)));
        detail::st3(velPred, i, add3(ldF3(velPred, i), scale3(ldF3(deltaVel, i), f)));
        detail::st3(deltaPos, i, F3{0, 0, 0});
        detail::st3(deltaVel, i, F3{0, 0, 0});
        constraintCounts(i) = 0;
      });
  space.fence();
}

/// Static-support velocity back-coupling: feed the position solve's net displacement back into the
/// committed velocity, clamped to the quasi-static scale (a few g dt). Without it a gravity-loaded
/// pile keeps a phantom collective fall: neighbours co-move (zero relative approach, so the velocity
/// solve has nothing to dissipate), the position solve silently lifts the column back each step, and
/// hard floor approaches re-bounce at full restitution instead of resting -- measured as a settled
/// 113-layer bed whose velocity field stays at free-fall magnitudes and crushes its bottom layers.
/// Statics need exactly +g dt per step, far inside the clamp; ballistic impacts produce dx/dt of
/// impact-speed size and are clamped to ~capMag, so no energy is injected (restitution stays owned
/// by the velocity solve). g = 0 => capMag 0 => disabled: growth-driven packing and HCS unchanged.
inline void applyStaticSupportKokkos(int numReal, V3 vel, Vf invMass, V3 posPred, V3 posPreSolve,
                                     float dt, float capMag) {
  if (!(capMag > 0.0f) || !(dt > 0.0f))
    return;
  CpExec space;
  const float idt = 1.0f / dt;
  Kokkos::parallel_for(
      "peclet::dem::static_support", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        if (invMass(i) <= 0.0f)
          return;
        F3 dv = scale3(sub3(ldF3(posPred, i), ldF3(posPreSolve, i)), idt);
        const float m2 = dot3(dv, dv);
        if (m2 > capMag * capMag)
          dv = scale3(dv, capMag / Kokkos::sqrt(m2));
        detail::st3(vel, i, add3(ldF3(vel, i), dv));
      });
  space.fence();
}

/// Final commit: periodic wrap of the predicted position into the domain, commit position + quat.
inline void finalCommitKokkos(int n, V3 pos, Vf invMass, V3 posPred, V4 quat, V4 quatPred,
                              Domain dom) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::final_commit", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        F3 p = ldF3(posPred, i);
        if (dom.periodic_x) {
          if (p.x < dom.min.x)
            p.x += dom.size.x;
          else if (p.x >= dom.max.x)
            p.x -= dom.size.x;
        }
        if (dom.periodic_y) {
          if (p.y < dom.min.y)
            p.y += dom.size.y;
          else if (p.y >= dom.max.y)
            p.y -= dom.size.y;
        }
        if (dom.periodic_z) {
          if (p.z < dom.min.z)
            p.z += dom.size.z;
          else if (p.z >= dom.max.z)
            p.z -= dom.size.z;
        }
        detail::st3(pos, i, p);
        detail::st4(quat, i, ldF4(quatPred, i));
      });
  space.fence();
}

/// Berendsen thermostat: reduce translational + rotational KE over real particles, compute the two
/// scaling factors, and rescale linear/angular velocities. Port of compute_energy_reduction_kernel
/// + apply_berendsen_thermostat_kernel (the scaling factors are computed on the host from the
/// reduced energies rather than in a shared-memory kernel — same result, simpler/portable).
struct KE2 {
  double t = 0.0, r = 0.0;
  KOKKOS_INLINE_FUNCTION KE2& operator+=(const KE2& o) {
    t += o.t;
    r += o.r;
    return *this;
  }
};
inline void applyThermostatKokkos(int numReal, V3 vel, Vf invMass, V3 angVel, V3 invInertia,
                                  V4 quat, double kB, double tau, double Ttarget, float dt) {
  CpExec space;
  KE2 sum;
  Kokkos::parallel_reduce(
      "peclet::dem::thermo_energy", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i, KE2& acc) {
        const float invM = invMass(i);
        const F3 v = ldF3(vel, i);
        if (invM > 0.0f)
          acc.t += 0.5 * (1.0 / invM) * (v.x * v.x + v.y * v.y + v.z * v.z);
        const F3 invI = ldF3(invInertia, i);
        if (invI.x > 0.0f || invI.y > 0.0f || invI.z > 0.0f) {
          const F3 wb = invRotateVector(ldF4(quat, i), ldF3(angVel, i));
          const double Ix = (invI.x > 0.0f) ? 1.0 / invI.x : 0.0;
          const double Iy = (invI.y > 0.0f) ? 1.0 / invI.y : 0.0;
          const double Iz = (invI.z > 0.0f) ? 1.0 / invI.z : 0.0;
          acc.r += 0.5 * (Ix * wb.x * wb.x + Iy * wb.y * wb.y + Iz * wb.z * wb.z);
        }
      },
      sum);

  const double ndof = 3.0 * numReal;
  auto lambda = [&](double ke) {
    if (ndof <= 0 || kB <= 0)
      return 1.0f;
    const double Tcur = (2.0 * ke) / (ndof * kB);
    if (Tcur < 1e-6)
      return 1.0f;
    double f = 1.0 + (dt / tau) * (Ttarget / Tcur - 1.0);
    if (f < 0.0)
      f = 0.0;
    return static_cast<float>(Kokkos::sqrt(f));
  };
  const float lt = lambda(sum.t), lr = lambda(sum.r);

  Kokkos::parallel_for(
      "peclet::dem::thermo_scale", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        detail::st3(vel, i, scale3(ldF3(vel, i), lt));
        detail::st3(angVel, i, scale3(ldF3(angVel, i), lr));
      });
  space.fence();
}

/// Growth mode: scale = target * factor (when active).
inline void updateGrowthScalesKokkos(int n, Vf scale, Vf targetScale, float factor) {
  if (!(factor > 0.0f))
    return;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::growth", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { scale(i) = targetScale(i) * factor; });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_INTEGRATION_HPP
