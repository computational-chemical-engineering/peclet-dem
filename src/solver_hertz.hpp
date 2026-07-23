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
inline void hertzWallForcesKokkos(Kokkos::View<const int*, CpMem> candSlots, int numCand,
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
                                  Kokkos::View<float* [3], CpMem> torque,
                                  Kokkos::View<const float* [4], CpMem> quat = {},
                                  Kokkos::View<const float*, CpMem> scale = {},
                                  ScalarI shapeId = {},
                                  Kokkos::View<const ShapeDesc*, CpMem> shapes = {},
                                  ShellView shell = {}, float globalScale = 1.0f,
                                  float contactRadiusFrac = 0.5f, bool hasShapes = false,
                                  Kokkos::View<float*, CpMem> snWall = {}) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::hertz_walls", Kokkos::RangePolicy<CpExec>(space, 0, numCand),
      KOKKOS_LAMBDA(int ci) {
        const int enc = candSlots(ci);
        const int i = enc / 8;
        const int wi = enc % 8;
        const F3 p = loadF3(pos, i);
        const float r = rad(i);
        const WallSdf w = walls(wi);
        const int slot = i * maxWalls + wi;
        const int ma = matId(i);
        if (hasShapes && shapes(shapeId(i)).numPoints > 0) {
          // Non-spherical: per-point Hertz springs of the shell against the wall SDF, one
          // Mindlin patch history per (particle, wall).
          const ShapeDesc dS = shapes(shapeId(i));
          const F4 qP = F4{quat(i, 0), quat(i, 1), quat(i, 2), quat(i, 3)};
          const float effScale = scale(i) * globalScale;
          const float rc = contactRadiusFrac * dS.params.x * effScale;
          float e, mu;
          if (w.materialId >= 0 && pairTable.extent(0) > 0)
            hertzPairEMu(ma, w.materialId, pairTable, eGlobal, muGlobal, e, mu);
          else {
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
          const float bd = hertzBetaD(e);
          const F3 vP = loadF3(vel, i), wP = loadF3(angVel, i);
          const float snTotPrev = (snWall.extent(0) > 0) ? snWall(slot) : 0.0f;
          const float dampC =
              (mStar > 0.0f) ? -2.0f * 0.9128709f * bd * Kokkos::sqrt(mStar) : 0.0f;
          float fnSum = 0.0f, snSum = 0.0f, dMax = 0.0f;
          F3 cSum{0, 0, 0}, nSum{0, 0, 0}, fAcc{0, 0, 0}, tAcc{0, 0, 0};
          for (int k = 0; k < dS.numPoints; ++k) {
            const int sIdx = dS.shellOffset + k;
            const F3 pW = add3(
                p, rotateVector(qP, scale3(F3{shell(sIdx, 0), shell(sIdx, 1), shell(sIdx, 2)},
                                           effScale)));
            const float sd = sampleWallSdf(pW, w, wallGrid);
            if (sd >= 0.0f)
              continue;
            const float delta = -sd;
            const float hx = 0.5f / w.invSpacing.x;
            F3 g{(sampleWallSdf(F3{pW.x + hx, pW.y, pW.z}, w, wallGrid) -
                  sampleWallSdf(F3{pW.x - hx, pW.y, pW.z}, w, wallGrid)),
                 (sampleWallSdf(F3{pW.x, pW.y + hx, pW.z}, w, wallGrid) -
                  sampleWallSdf(F3{pW.x, pW.y - hx, pW.z}, w, wallGrid)),
                 (sampleWallSdf(F3{pW.x, pW.y, pW.z + hx}, w, wallGrid) -
                  sampleWallSdf(F3{pW.x, pW.y, pW.z - hx}, w, wallGrid))};
            const float gl = len3(g);
            if (gl < 1e-12f)
              continue;
            const F3 nhat = scale3(g, 1.0f / gl);
            const F3 rA = sub3(pW, p);
            const F3 vWall = add3(w.linVel, cross3v(w.angVel, sub3(pW, w.center)));
            const F3 vrelP = sub3(add3(vP, cross3v(wP, rA)), vWall);
            const float vnP = dot3(vrelP, nhat);
            const float sq = Kokkos::sqrt(rc * delta);
            const float fnSpring = (4.0f / 3.0f) * eStar * sq * delta;
            const float snI = 2.0f * eStar * sq;
            const float snTot = (snTotPrev > snI) ? snTotPrev : snI;
            const float etaI = dampC * snI / Kokkos::sqrt(snTot);
            const F3 f = scale3(nhat, fnSpring - etaI * vnP);
            fAcc = add3(fAcc, f);
            tAcc = add3(tAcc, cross3v(rA, f));
            fnSum += fnSpring;
            snSum += 2.0f * eStar * sq;
            cSum = add3(cSum, scale3(pW, fnSpring));
            nSum = add3(nSum, scale3(nhat, fnSpring));
            if (delta > dMax)
              dMax = delta;
          }
          if (fnSum <= 0.0f) {
            xiWall(slot, 0) = xiWall(slot, 1) = xiWall(slot, 2) = 0.0f;
            if (snWall.extent(0) > 0)
              snWall(slot) = 0.0f;
            return;
          }
          const F3 cpt = scale3(cSum, 1.0f / fnSum);
          F3 nhat = nSum;
          const float nl = len3(nhat);
          if (nl > 1e-9f)
            nhat = scale3(nhat, 1.0f / nl);
          const F3 rA = sub3(cpt, p);
          const F3 vWall = add3(w.linVel, cross3v(w.angVel, sub3(cpt, w.center)));
          const F3 vrel = sub3(add3(vP, cross3v(wP, rA)), vWall);
          if (snWall.extent(0) > 0)
            snWall(slot) = snSum;
          const F3 vt = sub3(vrel, scale3(nhat, dot3(vrel, nhat)));
          const float st = 8.0f * gStar * Kokkos::sqrt(rc * dMax);
          const float etaT = -2.0f * 0.9128709f * bd * Kokkos::sqrt(st * mStar);
          F3 x{xiWall(slot, 0), xiWall(slot, 1), xiWall(slot, 2)};
          x = add3(x, scale3(vt, dt));
          x = sub3(x, scale3(nhat, dot3(x, nhat)));
          F3 ft = add3(scale3(x, -st), scale3(vt, -etaT));
          const float ftLen = len3(ft);
          const float ftMax = mu * fnSum;
          if (ftLen > ftMax) {
            const float sc2 = (ftLen > 1e-20f) ? ftMax / ftLen : 0.0f;
            ft = scale3(ft, sc2);
            if (st > 1e-20f)
              x = scale3(add3(ft, scale3(vt, etaT)), -1.0f / st);
          }
          xiWall(slot, 0) = x.x;
          xiWall(slot, 1) = x.y;
          xiWall(slot, 2) = x.z;
          fAcc = add3(fAcc, ft);
          tAcc = add3(tAcc, cross3v(rA, ft));
          for (int c = 0; c < 3; ++c) {
            Kokkos::atomic_add(&force(i, c), (&fAcc.x)[c]);
            Kokkos::atomic_add(&torque(i, c), (&tAcc.x)[c]);
          }
          return;
        }
        {
          const float s = sampleWallSdf(p, w, wallGrid);
          const float delta = r - s;
          if (delta <= 0.0f) {
            xiWall(slot, 0) = xiWall(slot, 1) = xiWall(slot, 2) = 0.0f;
            return;
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
            return;
          const F3 nhat = scale3(g, 1.0f / gl);  // into the void = push direction on the grain
          const F3 rA = scale3(nhat, -(r - 0.5f * delta));
          const F3 cpt = add3(p, rA);
          const F3 vWall = add3(w.linVel, cross3v(w.angVel, sub3(cpt, w.center)));
          const F3 vrel = sub3(add3(loadF3(vel, i), cross3v(loadF3(angVel, i), rA)), vWall);
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

/// Build the wall candidate list: particles within (radius + skin) of any wall's zero level.
/// Between pair rebuilds nothing else can reach a wall (same Verlet-skin argument as the pair
/// list), so the per-step wall pass only visits these slots. Encodes i * maxWalls + wallIdx.
inline int hertzBuildWallCandidatesKokkos(int numReal, int numWalls,
                                          Kokkos::View<const WallSdf*, CpMem> walls,
                                          GridView wallGrid,
                                          Kokkos::View<const float* [3], CpMem> pos,
                                          Kokkos::View<const float*, CpMem> rad, float skin,
                                          Kokkos::View<int*, CpMem> outSlots,
                                          Kokkos::View<int, CpMem> outCount) {
  CpExec space;
  Kokkos::deep_copy(space, outCount, 0);
  const int cap = static_cast<int>(outSlots.extent(0));
  Kokkos::parallel_for(
      "peclet::dem::hertz_wall_cand", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const F3 p{pos(i, 0), pos(i, 1), pos(i, 2)};
        for (int wi = 0; wi < numWalls; ++wi) {
          const float s = sampleWallSdf(p, walls(wi), wallGrid);
          if (s < rad(i) + skin) {
            const int at = Kokkos::atomic_fetch_add(&outCount(), 1);
            if (at < cap)
              outSlots(at) = i * 8 + wi;
          }
        }
      });
  space.fence();
  int n = 0;
  Kokkos::deep_copy(n, outCount);
  return n < cap ? n : cap;
}

/// Symplectic-Euler kick-drift (MUSEN-style) + displacement tracking for the Verlet rebuild.
/// Spheres: isotropic inertia, angular update needs no orientation.
inline void hertzIntegrateKokkos(int numReal, Kokkos::View<float* [3], CpMem> force,
                                 Kokkos::View<float* [3], CpMem> torque,
                                 Kokkos::View<const float*, CpMem> invMass,
                                 Kokkos::View<const float* [3], CpMem> invInertia, F3 gravity,
                                 float dt, Kokkos::View<float* [3], CpMem> vel,
                                 Kokkos::View<float* [3], CpMem> angVel,
                                 Kokkos::View<float* [3], CpMem> pos,
                                 Kokkos::View<float* [4], CpMem> quat = {},
                                 bool integrateOrientation = false) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::hertz_integrate", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const float im = invMass(i);
        if (im > 0.0f) {
          const F3 iI{invInertia(i, 0), invInertia(i, 1), invInertia(i, 2)};
          F3 dw;
          if (iI.x == iI.y && iI.y == iI.z) {  // isotropic: no frame rotation needed
            dw = F3{torque(i, 0) * iI.x, torque(i, 1) * iI.x, torque(i, 2) * iI.x};
          } else {  // world torque -> principal frame -> back
            const F4 q = loadF4(quat, i);
            const F3 tl = invRotateVector(q, F3{torque(i, 0), torque(i, 1), torque(i, 2)});
            dw = rotateVector(q, F3{tl.x * iI.x, tl.y * iI.y, tl.z * iI.z});
          }
          for (int c = 0; c < 3; ++c) {
            vel(i, c) += (force(i, c) * im + (&gravity.x)[c]) * dt;
            angVel(i, c) += (&dw.x)[c] * dt;
            pos(i, c) += vel(i, c) * dt;
          }
          if (integrateOrientation) {
            F4 q = loadF4(quat, i);
            const F3 w{angVel(i, 0), angVel(i, 1), angVel(i, 2)};
            // q += dt/2 * (0, w) x q, renormalized
            F4 dq{0.5f * dt * (w.x * q.w + w.y * q.z - w.z * q.y),
                  0.5f * dt * (w.y * q.w + w.z * q.x - w.x * q.z),
                  0.5f * dt * (w.z * q.w + w.x * q.y - w.y * q.x),
                  0.5f * dt * (-w.x * q.x - w.y * q.y - w.z * q.z)};
            q = F4{q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w};
            const float qn = Kokkos::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            if (qn > 1e-12f) {
              quat(i, 0) = q.x / qn;
              quat(i, 1) = q.y / qn;
              quat(i, 2) = q.z / qn;
              quat(i, 3) = q.w / qn;
            }
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


/// Non-spherical pairs: per-point Hertz springs over A's point shell against B's SDF (the same
/// one-sided sampling convention as the narrowphase), ONE Mindlin patch history per pair.
/// Per shell point k: penetration delta_k from B's SDF, its own Hertz normal spring + dashpot
/// applied AT THE POINT (correct face/edge force distribution and torques); the tangential
/// spring acts at the load-weighted patch centroid, bounded by mu * (sum of point spring forces).
/// R* combines per-shape CONTACT radii (Hertz curvature is undefined at faces/edges -- this is
/// the standard point-spring approximation; spheres use their true radii via params.x).
inline void hertzShapePairForcesKokkos(
    Kokkos::View<const int* [2], CpMem> pairs, int numPairs,
    Kokkos::View<const float* [3], CpMem> pos, Kokkos::View<const float* [4], CpMem> quat,
    Kokkos::View<const float* [3], CpMem> vel, Kokkos::View<const float* [3], CpMem> angVel,
    Kokkos::View<const float*, CpMem> scale, ScalarI shapeId,
    Kokkos::View<const ShapeDesc*, CpMem> shapes, ShellView shell, GridView sdfGrid,
    float globalScale, float contactRadiusFrac, Kokkos::View<const float*, CpMem> invMass,
    MatIdView matId, PairTableView pairTable, float eGlobal, float muGlobal,
    Kokkos::View<const float*, CpMem> hertzE, Kokkos::View<const float*, CpMem> hertzNu, float dt,
    Kokkos::View<float* [3], CpMem> xi, Kokkos::View<float*, CpMem> snPrev,
    Kokkos::View<float* [3], CpMem> force, Kokkos::View<float* [3], CpMem> torque) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::hertz_shape_pairs", Kokkos::RangePolicy<CpExec>(space, 0, numPairs),
      KOKKOS_LAMBDA(int idx) {
        const int idA = pairs(idx, 0), idB = pairs(idx, 1);
        const ShapeDesc dA = shapes(shapeId(idA));
        const ShapeDesc dB = shapes(shapeId(idB));
        const F3 posA = loadF3(pos, idA), posB = loadF3(pos, idB);
        const F4 qA = loadF4(quat, idA), qB = loadF4(quat, idB);
        const float effScaleA = scale(idA) * globalScale, effScaleB = scale(idB) * globalScale;
        const int countA = dA.numPoints;
        const bool sphereA = (dA.type == SPHERE);
        const int iter = (countA > 0) ? countA : 1;

        const int ma = matId(idA), mb = matId(idB);
        float e, mu;
        hertzPairEMu(ma, mb, pairTable, eGlobal, muGlobal, e, mu);
        const float Ei = hertzE(ma), Ej = hertzE(mb);
        const float ni = hertzNu(ma), nj = hertzNu(mb);
        const float eStar = 1.0f / ((1.0f - ni * ni) / Ei + (1.0f - nj * nj) / Ej);
        const float gStar =
            1.0f / (2.0f * (2.0f - ni) * (1.0f + ni) / Ei + 2.0f * (2.0f - nj) * (1.0f + nj) / Ej);
        // contact radii: spheres exact, shapes = frac * bounding radius
        // params.x = sphere radius or (grid SDF) canonical bounding radius
        const float rcA = (sphereA ? 1.0f : contactRadiusFrac) * dA.params.x * effScaleA;
        const float rcB =
            ((dB.type == SPHERE) ? 1.0f : contactRadiusFrac) * dB.params.x * effScaleB;
        const float rStar = rcA * rcB / (rcA + rcB);
        const float imSum = invMass(idA) + invMass(idB);
        const float mStar = (imSum > 0.0f) ? 1.0f / imSum : 0.0f;
        const float bd = hertzBetaD(e);

        const F3 vA = loadF3(vel, idA), wA = loadF3(angVel, idA);
        const F3 vB = loadF3(vel, idB), wB = loadF3(angVel, idB);

        float fnSum = 0.0f, snSum = 0.0f;
        F3 cSum{0, 0, 0}, nSum{0, 0, 0};
        float dMax = 0.0f;
        F3 fAcc{0, 0, 0}, tAccA{0, 0, 0}, tAccB{0, 0, 0};
        // Per-point dashpots weighted by stiffness against LAST step's patch stiffness sum:
        // eta_i = 2 beta sn_i sqrt(m*/snTot). Sums to the exact two-body damping AND gives every
        // patch mode (incl. face rocking) a uniform damping ratio; the patch-level dashpot alone
        // left rocking modes undamped (measured: cubes tumbling uphill on an incline).
        const float snTotPrev = snPrev(idx);
        const float dampC =
            (mStar > 0.0f) ? -2.0f * 0.9128709f * bd * Kokkos::sqrt(mStar) : 0.0f;

        for (int k = 0; k < iter; ++k) {
          F3 pLocalA{0, 0, 0};
          float pointRadius = 0.0f;
          if (countA > 0) {
            const int sIdx = dA.shellOffset + k;
            pLocalA = F3{shell(sIdx, 0), shell(sIdx, 1), shell(sIdx, 2)};
          } else if (sphereA) {
            pointRadius = dA.params.x * effScaleA;
          }
          const F3 pWorld = add3(posA, rotateVector(qA, scale3(pLocalA, effScaleA)));
          const F3 pCanB = scale3(invRotateVector(qB, sub3(pWorld, posB)), 1.0f / effScaleB);
          const float dist = sdfEvalShape(pCanB, dB, sdfGrid) * effScaleB;
          const float effDist = dist - pointRadius;
          if (effDist >= 0.0f)
            continue;
          const float delta = -effDist;
          const float eps = 1e-4f;
          F3 nLoc{sdfEvalShape(F3{pCanB.x + eps, pCanB.y, pCanB.z}, dB, sdfGrid) -
                      sdfEvalShape(F3{pCanB.x - eps, pCanB.y, pCanB.z}, dB, sdfGrid),
                  sdfEvalShape(F3{pCanB.x, pCanB.y + eps, pCanB.z}, dB, sdfGrid) -
                      sdfEvalShape(F3{pCanB.x, pCanB.y - eps, pCanB.z}, dB, sdfGrid),
                  sdfEvalShape(F3{pCanB.x, pCanB.y, pCanB.z + eps}, dB, sdfGrid) -
                      sdfEvalShape(F3{pCanB.x, pCanB.y, pCanB.z - eps}, dB, sdfGrid)};
          const float len = len3(nLoc);
          nLoc = (len > 1e-9f) ? scale3(nLoc, 1.0f / len) : F3{0, 1, 0};
          const F3 nW = rotateVector(qB, nLoc);  // out of B = push direction on A
          const F3 pSurfA = sub3(pWorld, scale3(nW, pointRadius));
          const F3 rA = sub3(pSurfA, posA);
          const F3 rB = sub3(add3(pSurfA, scale3(nW, delta)), posB);

          const F3 vrelP = sub3(add3(vA, cross3v(wA, rA)), add3(vB, cross3v(wB, rB)));
          const float vnP = dot3(vrelP, nW);
          const float sq = Kokkos::sqrt(rStar * delta);
          const float fnSpring = (4.0f / 3.0f) * eStar * sq * delta;
          const float snI = 2.0f * eStar * sq;
          const float snTot = (snTotPrev > snI) ? snTotPrev : snI;  // first-contact guard
          const float etaI = dampC * snI / Kokkos::sqrt(snTot);
          const F3 f = scale3(nW, fnSpring - etaI * vnP);
          fAcc = add3(fAcc, f);
          tAccA = add3(tAccA, cross3v(rA, f));
          tAccB = add3(tAccB, cross3v(rB, scale3(f, -1.0f)));
          fnSum += fnSpring;
          snSum += 2.0f * eStar * sq;  // patch tangent stiffness = sum of point stiffnesses
          cSum = add3(cSum, scale3(pSurfA, fnSpring));
          nSum = add3(nSum, scale3(nW, fnSpring));
          if (delta > dMax)
            dMax = delta;
        }

        if (fnSum <= 0.0f) {
          xi(idx, 0) = xi(idx, 1) = xi(idx, 2) = 0.0f;
          snPrev(idx) = 0.0f;
          return;
        }
        // Patch level: ONE normal dashpot and ONE Mindlin tangential spring at the
        // load-weighted centroid.
        const F3 cpt = scale3(cSum, 1.0f / fnSum);
        F3 nhat = nSum;
        const float nl = len3(nhat);
        if (nl > 1e-9f)
          nhat = scale3(nhat, 1.0f / nl);
        const F3 rA = sub3(cpt, posA), rB = sub3(cpt, posB);
        const F3 vrel = sub3(add3(vA, cross3v(wA, rA)), add3(vB, cross3v(wB, rB)));
        snPrev(idx) = snSum;
        const F3 vt = sub3(vrel, scale3(nhat, dot3(vrel, nhat)));
        const float st = 8.0f * gStar * Kokkos::sqrt(rStar * dMax);
        const float etaT = -2.0f * 0.9128709f * bd * Kokkos::sqrt(st * mStar);
        F3 x{xi(idx, 0), xi(idx, 1), xi(idx, 2)};
        x = add3(x, scale3(vt, dt));
        x = sub3(x, scale3(nhat, dot3(x, nhat)));
        F3 ft = add3(scale3(x, -st), scale3(vt, -etaT));
        const float ftLen = len3(ft);
        const float ftMax = mu * fnSum;
        if (ftLen > ftMax) {
          const float sc = (ftLen > 1e-20f) ? ftMax / ftLen : 0.0f;
          ft = scale3(ft, sc);
          if (st > 1e-20f)
            x = scale3(add3(ft, scale3(vt, etaT)), -1.0f / st);
        }
        xi(idx, 0) = x.x;
        xi(idx, 1) = x.y;
        xi(idx, 2) = x.z;
        fAcc = add3(fAcc, ft);
        tAccA = add3(tAccA, cross3v(rA, ft));
        tAccB = add3(tAccB, cross3v(rB, scale3(ft, -1.0f)));

        for (int c = 0; c < 3; ++c) {
          Kokkos::atomic_add(&force(idA, c), (&fAcc.x)[c]);
          Kokkos::atomic_add(&force(idB, c), -(&fAcc.x)[c]);
          Kokkos::atomic_add(&torque(idA, c), (&tAccA.x)[c]);
          Kokkos::atomic_add(&torque(idB, c), (&tAccB.x)[c]);
        }
      });
  space.fence();
}

}  // namespace peclet::dem
