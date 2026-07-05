/// @file
/// @brief dem — portable (Kokkos) Coulomb friction cluster (the single dissipative friction path).
///
/// Kokkos port of the four friction kernels in solver_position.cu. Friction lives in the velocity
/// solve (all dissipation there; the position solve does pure overlap removal):
///   computePlaneLoad   (once, before the velocity loop) — plane (idB<0) one-shot normal load.
///   accumulateNormalImpulse (each velocity iteration)   — body-body force-chain normal load.
///   countFrictionContacts (once, after the loop)        — per-body active-contact count (Jacobi
///   split). solveContactFriction (friction sweep)               — tangential impulse,
///   Coulomb-clamped by the
///                                                         normal load, count-averaged.
/// Faithful copy of the CUDA math over the SoA Views; the contact's friction_lambda_n is
/// read/written in place (per-contact, no cross-thread race). See packing-velocity-position-split
/// for the design.
#ifndef DEM_SOLVER_FRICTION_HPP
#define DEM_SOLVER_FRICTION_HPP

#include <Kokkos_Core.hpp>

#include "dem_portable.hpp"
#include "solver_position.hpp"  // ContactC, detail::computeW, CpExec/CpMem

namespace peclet::dem {

using FrManifoldCounts = Kokkos::View<float* [2], CpMem>;  // per body: .x = plane load, .y = count

/// Body-body force-chain normal load: contacts(idx).friction_lambda_n += approach / w_n.
inline void accumulateNormalImpulseKokkos(Kokkos::View<ContactC*, CpMem> contacts, int numContacts,
                                          Kokkos::View<const float*, CpMem> invMass,
                                          Kokkos::View<const float* [3], CpMem> invInertia,
                                          Kokkos::View<const float* [3], CpMem> velPred,
                                          Kokkos::View<const float* [3], CpMem> angVelPred,
                                          Kokkos::View<const int*, CpMem> realIdx,
                                          float growthRate) {
  using detail::computeW;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::fric_accum", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        ContactC c = contacts(idx);
        if (c.bodyB < 0)
          return;
        const int realA = realIdx(c.bodyA), realB = realIdx(c.bodyB);
        if (realA > realB)
          return;
        const float invMA = invMass(realA), invMB = invMass(realB);
        const F3 invIA = ldF3(invInertia, realA), invIB = ldF3(invInertia, realB);
        const F3 rA{c.rA.x, c.rA.y, c.rA.z}, rB{c.rB.x, c.rB.y, c.rB.z};
        const F3 n{c.normal.x, c.normal.y, c.normal.z};
        const F3 vAc = add3(ldF3(velPred, realA), cross3v(ldF3(angVelPred, realA), rA));
        const F3 vBc = add3(ldF3(velPred, realB), cross3v(ldF3(angVelPred, realB), rB));
        const F3 vrel = add3(sub3(vAc, vBc), scale3(sub3(rA, rB), growthRate));
        const float approach = -dot3(vrel, n);
        if (approach <= 0.0f)
          return;
        const float w_n = computeW(rA, n, invMA, invIA) + computeW(rB, n, invMB, invIB);
        if (w_n > 1e-6f)
          contacts(idx).friction_lambda_n += approach / w_n;
      });
  space.fence();
}

/// Plane (idB<0) one-shot loads. Zeros planeFriction first (both columns), then sets the
/// per-contact friction_lambda_n and the per-body max plane load (.x).
inline void computePlaneLoadKokkos(Kokkos::View<ContactC*, CpMem> contacts, int numContacts,
                                   Kokkos::View<const float*, CpMem> invMass,
                                   Kokkos::View<const float* [3], CpMem> invInertia,
                                   Kokkos::View<const float* [3], CpMem> velPred,
                                   Kokkos::View<const float* [3], CpMem> angVelPred,
                                   FrManifoldCounts planeFriction) {
  using detail::computeW;
  CpExec space;
  Kokkos::deep_copy(space, planeFriction, 0.0f);
  Kokkos::parallel_for(
      "peclet::dem::fric_plane_load", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        ContactC c = contacts(idx);
        if (c.bodyB >= 0)
          return;
        const int idA = c.bodyA;
        const float invMA = invMass(idA);
        if (invMA <= 0.0f)
          return;
        const F3 invIA = ldF3(invInertia, idA);
        const F3 rA{c.rA.x, c.rA.y, c.rA.z}, n{c.normal.x, c.normal.y, c.normal.z};
        const F3 vAc = add3(ldF3(velPred, idA), cross3v(ldF3(angVelPred, idA), rA));
        // Relative to the wall's surface velocity (0 for a static plane), so a wall pressing into a
        // grain along its normal (a vibrating tray) registers the normal load that bounds friction.
        const F3 vWall{c.boundaryVel.x, c.boundaryVel.y, c.boundaryVel.z};
        const float approach = -dot3(sub3(vAc, vWall), n);
        if (approach <= 0.0f)
          return;
        const float w_n = computeW(rA, n, invMA, invIA);
        contacts(idx).friction_lambda_n = (w_n > 1e-6f) ? approach / w_n : 0.0f;
        Kokkos::atomic_max(&planeFriction(idA, 0), approach / invMA);
      });
  space.fence();
}

/// Per-body active-contact count into planeFriction(:,1).
inline void countFrictionContactsKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                        int numContacts, Kokkos::View<const int*, CpMem> realIdx,
                                        FrManifoldCounts planeFriction) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::fric_count", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        const ContactC c = contacts(idx);
        if (c.friction_lambda_n <= 0.0f)
          return;
        Kokkos::atomic_add(&planeFriction(realIdx(c.bodyA), 1), 1.0f);
        if (c.bodyB >= 0)
          Kokkos::atomic_add(&planeFriction(realIdx(c.bodyB), 1), 1.0f);
      });
  space.fence();
}

/// One count-averaged Coulomb friction sweep. Zeros the delta arrays first, then accumulates the
/// tangential impulse (clamped by mu*normal-load, divided by the larger contact count) into them.
inline void solveContactFrictionKokkos(
    Kokkos::View<const ContactC*, CpMem> contacts, int numContacts,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<const float* [3], CpMem> invInertia,
    Kokkos::View<const float* [3], CpMem> velPred, Kokkos::View<const float* [3], CpMem> angVelPred,
    Kokkos::View<const int*, CpMem> realIdx, FrManifoldCounts planeFriction, float frictionDynamic,
    Kokkos::View<float* [3], CpMem> deltaVel, Kokkos::View<float* [3], CpMem> deltaAngVel) {
  CpExec space;
  Kokkos::deep_copy(space, deltaVel, 0.0f);
  Kokkos::deep_copy(space, deltaAngVel, 0.0f);
  Kokkos::parallel_for(
      "peclet::dem::fric_solve", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        const ContactC c = contacts(idx);
        const float lambda_n = c.friction_lambda_n;
        if (lambda_n <= 0.0f)
          return;
        const int idA = c.bodyA, idB = c.bodyB;
        const int realA = realIdx(idA);
        const int realB = (idB >= 0) ? realIdx(idB) : -1;
        const float invMA = invMass(realA);
        const float invMB = (idB >= 0) ? invMass(realB) : 0.0f;
        const F3 invIA = ldF3(invInertia, realA);
        const F3 invIB = (idB >= 0) ? ldF3(invInertia, realB) : F3{0, 0, 0};
        const F3 rA{c.rA.x, c.rA.y, c.rA.z}, rB{c.rB.x, c.rB.y, c.rB.z};
        const F3 n{c.normal.x, c.normal.y, c.normal.z};

        const F3 vAc = add3(ldF3(velPred, realA), cross3v(ldF3(angVelPred, realA), rA));
        F3 vBc{0, 0, 0};
        if (idB >= 0)
          vBc = add3(ldF3(velPred, realB), cross3v(ldF3(angVelPred, realB), rB));
        else
          vBc = F3{c.boundaryVel.x, c.boundaryVel.y, c.boundaryVel.z};  // moving wall drags the grain
        const F3 vrel = sub3(vAc, vBc);
        const float vn = dot3(vrel, n);
        const F3 vt = sub3(vrel, scale3(n, vn));
        const float vt_len = Kokkos::sqrt(dot3(vt, vt));
        if (vt_len < 1e-7f)
          return;
        const F3 t = scale3(vt, 1.0f / vt_len);

        const F3 rnA = cross3v(rA, t), rnB = cross3v(rB, t);
        const float w_t = invMA + invMB + rnA.x * rnA.x * invIA.x + rnA.y * rnA.y * invIA.y +
                          rnA.z * rnA.z * invIA.z + rnB.x * rnB.x * invIB.x +
                          rnB.y * rnB.y * invIB.y + rnB.z * rnB.z * invIB.z;
        if (w_t < 1e-6f)
          return;

        const float bound = (idB < 0) ? planeFriction(idA, 0) : lambda_n;
        const float nA = planeFriction(realA, 1);
        const float nB = (idB >= 0) ? planeFriction(realB, 1) : 0.0f;
        const float inv_n = 1.0f / Kokkos::fmax(Kokkos::fmax(nA, nB), 1.0f);
        float lt = -vt_len / w_t;
        // Per-wall friction for a boundary contact (a < 0 sentinel keeps the global material — planes
        // and body-body); grain-grain contacts always use the global coefficient.
        const float mu = (idB < 0 && c.boundaryFriction >= 0.0f) ? c.boundaryFriction : frictionDynamic;
        const float maxf = mu * bound;
        if (lt < -maxf)
          lt = -maxf;
        lt *= inv_n;

        Kokkos::atomic_add(&deltaVel(realA, 0), t.x * lt * invMA);
        Kokkos::atomic_add(&deltaVel(realA, 1), t.y * lt * invMA);
        Kokkos::atomic_add(&deltaVel(realA, 2), t.z * lt * invMA);
        Kokkos::atomic_add(&deltaAngVel(realA, 0), rnA.x * invIA.x * lt);
        Kokkos::atomic_add(&deltaAngVel(realA, 1), rnA.y * invIA.y * lt);
        Kokkos::atomic_add(&deltaAngVel(realA, 2), rnA.z * invIA.z * lt);
        if (idB >= 0) {
          Kokkos::atomic_add(&deltaVel(realB, 0), -t.x * lt * invMB);
          Kokkos::atomic_add(&deltaVel(realB, 1), -t.y * lt * invMB);
          Kokkos::atomic_add(&deltaVel(realB, 2), -t.z * lt * invMB);
          Kokkos::atomic_add(&deltaAngVel(realB, 0), -rnB.x * invIB.x * lt);
          Kokkos::atomic_add(&deltaAngVel(realB, 1), -rnB.y * invIB.y * lt);
          Kokkos::atomic_add(&deltaAngVel(realB, 2), -rnB.z * invIB.z * lt);
        }
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_FRICTION_HPP
