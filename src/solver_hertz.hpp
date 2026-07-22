#pragma once
/// @file
/// @brief Soft-sphere Hertz–Mindlin DEM (reference force model) for SPHERES.
///
/// The benchmark contact model of Dosta et al. (2024): viscoelastic Hertz normal force with
/// restitution-matched damping + "no-slip" Mindlin tangential spring with history, Coulomb-clamped
/// (LIGGGHTS `gran model hertz tangential history` formulas). This is an explicit force model —
/// time step limited by the Rayleigh time — living beside the impulse (XPBD/NSCD) solver as a
/// physics reference and cross-check. Spheres only, non-periodic domains, SDF walls (no planes).
///
/// Structure: a cached Verlet pair list (broadphase margin = skin, rebuilt when any particle has
/// moved skin/2) carries the per-pair tangential history xi through its slots; on rebuild the
/// history is carried over by pair key. Wall contacts are evaluated per particle per wall each
/// step with per-(particle, wall) history. The inner time loop runs device-side per call.

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "contact_preprocessing.hpp"
#include "narrowphase.hpp"
#include "particles.hpp"

namespace peclet::dem {

/// Pairwise (e, mu) lookup shared with the narrowphase convention: pair table if present,
/// global fallback otherwise.
KOKKOS_INLINE_FUNCTION void hertzPairEMu(int matA, int matB, PairTableView pairTable,
                                         float eGlobal, float muGlobal, float& e, float& mu) {
  if (pairTable.extent(0) > 0) {
    const int t = (matA * kMaxMaterials + matB) * 2;
    e = pairTable(t);
    mu = pairTable(t + 1);
  } else {
    e = eGlobal;
    mu = muGlobal;
  }
}

/// Restitution -> damping ratio beta_d = ln e / sqrt(ln^2 e + pi^2)  (e in (0, 1]; e <= 0 -> -1,
/// the critically-damped limit used by LIGGGHTS for e -> 0).
KOKKOS_INLINE_FUNCTION float hertzBetaD(float e) {
  if (e <= 0.0f)
    return -1.0f;
  if (e >= 1.0f)
    return 0.0f;
  const float le = Kokkos::log(e);
  return le / Kokkos::sqrt(le * le + 9.8696044f);
}

/// One Hertz–Mindlin force evaluation for a sphere-sphere or sphere-wall contact.
/// Returns the force ON body A applied at the contact point (linear) and updates xi in place.
/// nhat points from B to A (push direction on A). vrel = contact velocity of A relative to B.
KOKKOS_INLINE_FUNCTION F3 hertzForce(float delta, F3 nhat, F3 vrel, float rStar, float mStar,
                                     float eStar, float gStar, float e, float mu, float dt,
                                     F3& xi) {
  const float sq = Kokkos::sqrt(rStar * delta);
  const float sn = 2.0f * eStar * sq;
  const float st = 8.0f * gStar * sq;
  const float kn = (4.0f / 3.0f) * eStar * sq;  // Fn_spring = kn * delta
  const float bd = hertzBetaD(e);
  const float etaN = -2.0f * 0.9128709f * bd * Kokkos::sqrt(sn * mStar);  // sqrt(5/6)
  const float etaT = -2.0f * 0.9128709f * bd * Kokkos::sqrt(st * mStar);

  const float vn = dot3(vrel, nhat);  // > 0: separating (A moving away from B)
  const float fnSpring = kn * delta;
  const float fn = fnSpring - etaN * vn;  // + etaN * ddelta/dt: damping opposes approach
  const F3 vt = sub3(vrel, scale3(nhat, vn));

  // Mindlin history spring: accumulate tangential displacement, keep it in the tangent plane.
  xi = add3(xi, scale3(vt, dt));
  xi = sub3(xi, scale3(nhat, dot3(xi, nhat)));
  F3 ft = add3(scale3(xi, -st), scale3(vt, -etaT));
  const float ftLen = Kokkos::sqrt(dot3(ft, ft));
  const float ftMax = mu * Kokkos::fabs(fnSpring);
  if (ftLen > ftMax) {
    // sliding: clamp AND rescale the history so the spring alone carries the clamped force
    // (the LIGGGHTS shear-history rescale).
    const float s = (ftLen > 1e-20f) ? ftMax / ftLen : 0.0f;
    ft = scale3(ft, s);
    if (st > 1e-20f)
      xi = scale3(add3(ft, scale3(vt, etaT)), -1.0f / st);
  }
  return add3(scale3(nhat, fn), ft);
}

/// Cached-pair forces: overlap from current positions; history resets when a cached pair is
/// currently separated. Forces/torques accumulate atomically (a body appears in many pairs).
inline void hertzPairForcesKokkos(Kokkos::View<const int* [2], CpMem> pairs, int numPairs,
                                  Kokkos::View<const float* [3], CpMem> pos,
                                  Kokkos::View<const float* [3], CpMem> vel,
                                  Kokkos::View<const float* [3], CpMem> angVel,
                                  Kokkos::View<const float*, CpMem> rad,
                                  Kokkos::View<const float*, CpMem> invMass,
                                  MatIdView matId, PairTableView pairTable, float eGlobal,
                                  float muGlobal, Kokkos::View<const float*, CpMem> hertzE,
                                  Kokkos::View<const float*, CpMem> hertzNu, float dt,
                                  Kokkos::View<float* [3], CpMem> xi,
                                  Kokkos::View<float* [3], CpMem> force,
                                  Kokkos::View<float* [3], CpMem> torque) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::hertz_pairs", Kokkos::RangePolicy<CpExec>(space, 0, numPairs),
      KOKKOS_LAMBDA(int idx) {
        const int i = pairs(idx, 0), j = pairs(idx, 1);
        const F3 pi = loadF3(pos, i), pj = loadF3(pos, j);
        const F3 dx = sub3(pi, pj);
        const float d = Kokkos::sqrt(dot3(dx, dx));
        const float ri = rad(i), rj = rad(j);
        const float delta = ri + rj - d;
        if (delta <= 0.0f || d < 1e-12f) {
          xi(idx, 0) = xi(idx, 1) = xi(idx, 2) = 0.0f;  // contact open: history resets
          return;
        }
        const F3 nhat = scale3(dx, 1.0f / d);  // B(j) -> A(i)
        const F3 rA = scale3(nhat, -(ri - 0.5f * delta));
        const F3 rB = scale3(nhat, (rj - 0.5f * delta));
        const F3 vrel = sub3(add3(loadF3(vel, i), cross3v(loadF3(angVel, i), rA)),
                             add3(loadF3(vel, j), cross3v(loadF3(angVel, j), rB)));
        const int ma = matId(i), mb = matId(j);
        float e, mu;
        hertzPairEMu(ma, mb, pairTable, eGlobal, muGlobal, e, mu);
        const float Ei = hertzE(ma), Ej = hertzE(mb);
        const float ni = hertzNu(ma), nj = hertzNu(mb);
        const float eStar = 1.0f / ((1.0f - ni * ni) / Ei + (1.0f - nj * nj) / Ej);
        const float gStar =
            1.0f / (2.0f * (2.0f - ni) * (1.0f + ni) / Ei + 2.0f * (2.0f - nj) * (1.0f + nj) / Ej);
        const float rStar = ri * rj / (ri + rj);
        const float imSum = invMass(i) + invMass(j);
        const float mStar = (imSum > 0.0f) ? 1.0f / imSum : 0.0f;
        F3 x{xi(idx, 0), xi(idx, 1), xi(idx, 2)};
        const F3 f = hertzForce(delta, nhat, vrel, rStar, mStar, eStar, gStar, e, mu, dt, x);
        xi(idx, 0) = x.x;
        xi(idx, 1) = x.y;
        xi(idx, 2) = x.z;
        const F3 tA = cross3v(rA, f), tB = cross3v(rB, scale3(f, -1.0f));
        for (int c = 0; c < 3; ++c) {
          Kokkos::atomic_add(&force(i, c), (&f.x)[c]);
          Kokkos::atomic_add(&force(j, c), -(&f.x)[c]);
          Kokkos::atomic_add(&torque(i, c), (&tA.x)[c]);
          Kokkos::atomic_add(&torque(j, c), (&tB.x)[c]);
        }
      });
  space.fence();
}

/// SDF-wall forces (per particle x wall). Gradient by central differences at half a grid cell.
/// The wall's rigid surface velocity enters vrel; per-(particle, wall) history in xiWall.
inline void hertzWallForcesKokkos(int numReal, int numWalls,
                                  Kokkos::View<const WallSdf*, CpMem> walls, GridView wallGrid,
                                  Kokkos::View<const float* [3], CpMem> pos,
                                  Kokkos::View<const float* [3], CpMem> vel,
                                  Kokkos::View<const float* [3], CpMem> angVel,
                                  Kokkos::View<const float*, CpMem> rad,
                                  Kokkos::View<const float*, CpMem> invMass,
                                  MatIdView matId, PairTableView pairTable, float eGlobal,
                                  float muGlobal, Kokkos::View<const float*, CpMem> hertzE,
                                  Kokkos::View<const float*, CpMem> hertzNu, float dt,
                                  Kokkos::View<float* [3], CpMem> xiWall, int maxWalls,
                                  Kokkos::View<float* [3], CpMem> force,
                                  Kokkos::View<float* [3], CpMem> torque) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::hertz_walls", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const F3 p = loadF3(pos, i);
        const float r = rad(i);
        for (int wi = 0; wi < numWalls; ++wi) {
          const WallSdf w = walls(wi);
          const int slot = i * maxWalls + wi;
          const float s = sampleWallSdf(p, w, wallGrid);
          const float delta = r - s;
          if (delta <= 0.0f) {
            xiWall(slot, 0) = xiWall(slot, 1) = xiWall(slot, 2) = 0.0f;
            continue;
          }
          const float hx = 0.5f / w.invSpacing.x;
          F3 g{(sampleWallSdf(F3{p.x + hx, p.y, p.z}, w, wallGrid) -
                sampleWallSdf(F3{p.x - hx, p.y, p.z}, w, wallGrid)),
               (sampleWallSdf(F3{p.x, p.y + hx, p.z}, w, wallGrid) -
                sampleWallSdf(F3{p.x, p.y - hx, p.z}, w, wallGrid)),
               (sampleWallSdf(F3{p.x, p.y, p.z + hx}, w, wallGrid) -
                sampleWallSdf(F3{p.x, p.y, p.z - hx}, w, wallGrid))};
          const float gl = Kokkos::sqrt(dot3(g, g));
          if (gl < 1e-12f)
            continue;
          const F3 nhat = scale3(g, 1.0f / gl);  // into the void = push direction on the grain
          const F3 rA = scale3(nhat, -(r - 0.5f * delta));
          const F3 cpt = add3(p, rA);
          const F3 vWall = add3(w.linVel, cross3v(w.angVel, sub3(cpt, w.center)));
          const F3 vrel = sub3(add3(loadF3(vel, i), cross3v(loadF3(angVel, i), rA)), vWall);
          const int ma = matId(i);
          float e, mu;
          if (w.materialId >= 0 && pairTable.extent(0) > 0) {
            hertzPairEMu(ma, w.materialId, pairTable, eGlobal, muGlobal, e, mu);
          } else {
            e = w.restitution;
            mu = w.friction;
          }
          const int mw = (w.materialId >= 0) ? w.materialId : ma;
          const float Ei = hertzE(ma), Ew = hertzE(mw);
          const float ni = hertzNu(ma), nw = hertzNu(mw);
          const float eStar = 1.0f / ((1.0f - ni * ni) / Ei + (1.0f - nw * nw) / Ew);
          const float gStar = 1.0f / (2.0f * (2.0f - ni) * (1.0f + ni) / Ei +
                                      2.0f * (2.0f - nw) * (1.0f + nw) / Ew);
          const float mStar = (invMass(i) > 0.0f) ? 1.0f / invMass(i) : 0.0f;
          F3 x{xiWall(slot, 0), xiWall(slot, 1), xiWall(slot, 2)};
          const F3 f = hertzForce(delta, nhat, vrel, r, mStar, eStar, gStar, e, mu, dt, x);
          xiWall(slot, 0) = x.x;
          xiWall(slot, 1) = x.y;
          xiWall(slot, 2) = x.z;
          const F3 tA = cross3v(rA, f);
          for (int c = 0; c < 3; ++c) {
            Kokkos::atomic_add(&force(i, c), (&f.x)[c]);
            Kokkos::atomic_add(&torque(i, c), (&tA.x)[c]);
          }
        }
      });
  space.fence();
}

/// Symplectic-Euler kick-drift (MUSEN-style) + displacement tracking for the Verlet rebuild.
/// Spheres: isotropic inertia, angular update needs no orientation.
inline void hertzIntegrateKokkos(int numReal, Kokkos::View<float* [3], CpMem> force,
                                 Kokkos::View<float* [3], CpMem> torque,
                                 Kokkos::View<const float*, CpMem> invMass,
                                 Kokkos::View<const float* [3], CpMem> invInertia, F3 gravity,
                                 float dt, Kokkos::View<float* [3], CpMem> vel,
                                 Kokkos::View<float* [3], CpMem> angVel,
                                 Kokkos::View<float* [3], CpMem> pos) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::hertz_integrate", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const float im = invMass(i);
        if (im > 0.0f) {
          for (int c = 0; c < 3; ++c) {
            vel(i, c) += (force(i, c) * im + (&gravity.x)[c]) * dt;
            angVel(i, c) += torque(i, c) * invInertia(i, 0) * dt;  // isotropic (spheres)
            pos(i, c) += vel(i, c) * dt;
          }
        }
        force(i, 0) = force(i, 1) = force(i, 2) = 0.0f;
        torque(i, 0) = torque(i, 1) = torque(i, 2) = 0.0f;
      });
  space.fence();
}

/// Max squared displacement since the last pair build (called only at rebuild checks --
/// keeping this out of the integrate kernel avoids a per-step single-address atomic).
inline float hertzMaxDisp2Kokkos(int numReal, Kokkos::View<const float* [3], CpMem> pos,
                                 Kokkos::View<const float* [3], CpMem> refPos) {
  CpExec space;
  float m = 0.0f;
  Kokkos::parallel_reduce(
      "peclet::dem::hertz_maxdisp", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i, float& acc) {
        const F3 d = sub3(loadF3(pos, i), loadF3(refPos, i));
        const float dd = dot3(d, d);
        if (dd > acc)
          acc = dd;
      },
      Kokkos::Max<float>(m));
  space.fence();
  return m;
}

}  // namespace peclet::dem
