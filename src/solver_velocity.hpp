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
                                Kokkos::View<int*, CpMem> velCounts,
                                Kokkos::View<const int*, CpMem> onlyColor = {},
                                int colorFilter = 0) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  const bool filt = onlyColor.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::solve_velocity", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        if (filt && onlyColor(idx) != colorFilter)
          return;  // Jacobi fallback pass: only the manifolds the colouring could not place
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

// ============================ colored Gauss–Seidel velocity solve ============================
// The Jacobi solve above sums every touching manifold's impulse onto a body, then relaxes the sum by
// the contact count to stay stable — a stable but UNDER-converged approximation that under-dissipates
// in dense multi-contact regions (effective restitution rises above the prescribed e). The colored
// Gauss–Seidel path removes that approximation: graph-colour the manifolds so no two sharing a real
// body share a colour, then sweep colour-by-colour applying each impulse IN PLACE (read the current
// velocity, apply, write) — a body sees the updates of every previously-solved contact in the same
// sweep. Within a colour the manifolds are an independent set (no shared body), so the in-place
// read-modify-write is race-free WITHOUT atomics or averaging, and the fixed-point is the true
// coupled multi-contact solution, so the dissipation is correct by construction. count==1 (a binary
// collision) is identical to the Jacobi path; the difference is confined to dense clusters.

/// Greedy graph-colour the manifolds: no two manifolds sharing a real body get the same colour.
/// Round-based max-index (Jones–Plassmann) arbitration, no adjacency lists: each round every still-
/// uncoloured manifold contends for its endpoint bodies via atomicMax(bodyWinner, idx); a manifold
/// that wins BOTH endpoints has no uncoloured conflict this round, so it commits the lowest colour
/// free at either endpoint (a per-body bitmask) — and, as the unique winner of those bodies, updates
/// the masks race-free. Inactive manifolds (empty, or the periodic-dedup duplicate realA>realB) are
/// tagged -2 and skipped by the solve. Runs once per step; the colouring is reused across the
/// velocity sweeps. Returns the number of colours used (0 if no active manifolds).
inline int colorManifoldsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
                                Kokkos::View<const int*, CpMem> realIdx, int numReal,
                                Kokkos::View<int*, CpMem> mColor,
                                Kokkos::View<long long*, CpMem> bodyWinner,
                                Kokkos::View<std::uint64_t*, CpMem> bodyMask, int& leftover) {
  leftover = 0;
  CpExec space;
  if (numManifolds <= 0 || numReal <= 0)
    return 0;
  Kokkos::parallel_for(
      "peclet::dem::color_init_bodies", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) { bodyMask(i) = 0; });
  // -2 inactive (skip forever), -1 uncoloured, >=0 committed colour.
  Kokkos::parallel_for(
      "peclet::dem::color_init_manifolds", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0) {
          mColor(idx) = -2;
          return;
        }
        if (m.bodyB >= 0 && realIdx(m.bodyA) > realIdx(m.bodyB)) {
          mColor(idx) = -2;  // periodic dedup: the (realA<=realB) twin carries this contact
          return;
        }
        mColor(idx) = -1;
      });

  int remaining = 1, prevRemaining = -1;
  const int maxRounds = numReal + 2;  // safety bound; converges in ~max-degree rounds in practice
  for (int round = 0; round < maxRounds && remaining > 0; ++round) {
    Kokkos::parallel_for(
        "peclet::dem::color_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
        KOKKOS_LAMBDA(int i) { bodyWinner(i) = -1; });
    Kokkos::parallel_for(
        "peclet::dem::color_contend", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) {
          if (mColor(idx) != -1)
            return;
          const ManifoldC m = manifolds(idx);
          const long long key = colorKey(idx);
          Kokkos::atomic_max(&bodyWinner(realIdx(m.bodyA)), key);
          if (m.bodyB >= 0)
            Kokkos::atomic_max(&bodyWinner(realIdx(m.bodyB)), key);
        });
    int rem = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::color_commit", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx, int& acc) {
          if (mColor(idx) != -1)
            return;
          const ManifoldC m = manifolds(idx);
          const int ea = realIdx(m.bodyA);
          const int eb = (m.bodyB >= 0) ? realIdx(m.bodyB) : -1;
          // Winner iff it holds BOTH its endpoints -> no uncoloured conflict, sole writer of ea/eb.
          const long long key = colorKey(idx);
          if (bodyWinner(ea) != key || (eb >= 0 && bodyWinner(eb) != key)) {
            acc += 1;
            return;
          }
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int c = 0;
          while (c < 62 && (forbidden & (std::uint64_t(1) << c)))
            ++c;  // lowest free colour (cap 63; dense sphere degree ~12, far below)
          mColor(idx) = c;
          const std::uint64_t bit = std::uint64_t(1) << c;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    if (rem == prevRemaining)
      break;  // colour-mask saturation (degree > 62): leftovers stay -1, Jacobi fallback applies them
    prevRemaining = rem;
    remaining = rem;
  }

  int maxc = -1;
  Kokkos::parallel_reduce(
      "peclet::dem::color_max", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx, int& mx) {
        if (mColor(idx) > mx)
          mx = mColor(idx);
      },
      Kokkos::Max<int>(maxc));
  Kokkos::parallel_reduce(
      "peclet::dem::color_leftover", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx, int& acc) {
        if (mColor(idx) == -1)
          acc += 1;
      },
      leftover);
  space.fence();
  return maxc + 1;
}

/// Colored Gauss–Seidel normal-restitution solve: sweep the `numColors` colour classes in order,
/// applying each manifold's impulse directly to velPred/angVelPred (in place). Same per-manifold
/// impulse math as solveVelocityKokkos (growth-velocity term, approach gate, resting-contact e=0
/// threshold, and the R·(invI_local·(Rᵀ J))·… world-space angular update) — only the write-back
/// differs (in-place RMW instead of atomic-accumulate + count-average). Race-free because a colour is
/// an independent set of manifolds. One outer call = one full sweep over all colours; the caller
/// loops it velocityIterations times.
inline void solveVelocityColoredGSKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                         int numManifolds,
                                         Kokkos::View<const int*, CpMem> mColor, int numColors,
                                         Kokkos::View<const float*, CpMem> invMass,
                                         Kokkos::View<const float* [3], CpMem> invInertia,
                                         Kokkos::View<const float* [4], CpMem> quat,
                                         Kokkos::View<float* [3], CpMem> velPred,
                                         Kokkos::View<float* [3], CpMem> angVelPred,
                                         Kokkos::View<const int*, CpMem> realIdx, float growthRate,
                                         float restitutionNormal, float restVelThreshold,
                                         Kokkos::View<float, CpMem> maxApproach) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  for (int color = 0; color < numColors; ++color) {
    Kokkos::parallel_for(
        "peclet::dem::solve_velocity_gs", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) {
          if (mColor(idx) != color)
            return;
          const ManifoldC m = manifolds(idx);

          const int idA = m.bodyA, idB = m.bodyB;
          const int realA = realIdx(idA);
          const int realB = (idB >= 0) ? realIdx(idB) : idB;

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

          const F3 diffCenters = (idB < 0) ? rAavg : sub3(rAavg, rBavg);
          const F3 vGrowth = scale3(diffCenters, growthRate);

          float vn = dot3(vA, Nsum) + dot3(wA, TauA) + dot3(vB, F3{-Nsum.x, -Nsum.y, -Nsum.z}) +
                     dot3(wB, TauB);
          vn += dot3(vGrowth, Nsum);

          const float alignment = dot3(Nsum, diffCenters);
          if (alignment > 0.0f) {
            if (vn < 0.0f)
              return;
          } else {
            if (vn > 0.0f)
              return;
          }

          const float Nsq = dot3(Nsum, Nsum);
          const float wA_n = Nsq * invMassA + genInvMass(TauA, invIA, qA);
          const float wB_n = Nsq * invMassB + genInvMass(TauB, invIB, qB);
          const float wTotal = wA_n + wB_n;
          if (wTotal <= 0.0f)
            return;

          // Record this approaching pair's physical approach speed for the adaptive stop: the caller
          // ends the velocity loop once no manifold approaches faster than the resting threshold.
          Kokkos::atomic_max(&maxApproach(), Kokkos::fabs(vn) / lenN);

          if (Kokkos::fabs(vn) < restVelThreshold * lenN)
            restitution = 0.0f;

          const float lambda = (-restitution * vn - vn) / wTotal;

          const F3 Jlin = scale3(Nsum, lambda);
          const F3 JangA = scale3(TauA, lambda);
          const F3 JangB = scale3(TauB, lambda);

          // Linear + angular delta on A (dw_world = R (invI_local * (R^T Jang))), applied in place.
          velPred(realA, 0) += Jlin.x * invMassA;
          velPred(realA, 1) += Jlin.y * invMassA;
          velPred(realA, 2) += Jlin.z * invMassA;
          {
            const F3 Jl = invRotateVector(qA, JangA);
            const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
            const F3 dww = rotateVector(qA, dwl);
            angVelPred(realA, 0) += dww.x;
            angVelPred(realA, 1) += dww.y;
            angVelPred(realA, 2) += dww.z;
          }
          if (idB >= 0) {
            velPred(realB, 0) += -Jlin.x * invMassB;
            velPred(realB, 1) += -Jlin.y * invMassB;
            velPred(realB, 2) += -Jlin.z * invMassB;
            const F3 Jl = invRotateVector(qB, JangB);
            const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
            const F3 dww = rotateVector(qB, dwl);
            angVelPred(realB, 0) += dww.x;
            angVelPred(realB, 1) += dww.y;
            angVelPred(realB, 2) += dww.z;
          }
        });
    // No host fence here: consecutive parallel_for on one execution space are stream-ordered on the
    // device, so colour c+1's kernel already observes colour c's in-place writes (the Gauss–Seidel
    // dependency). A per-colour fence would only stall the host. One fence after the sweep suffices.
  }
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_VELOCITY_HPP
