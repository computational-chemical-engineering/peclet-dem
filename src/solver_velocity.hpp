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
                                int colorFilter = 0,
                                Kokkos::View<const unsigned char*, CpMem> persistent = {},
                                Kokkos::View<const float* [3], CpMem> posPred = {}, F3 gHat = {},
                                Kokkos::View<const unsigned char*, CpMem> grounded = {}) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  const bool filt = onlyColor.extent(0) > 0;
  const bool usePersist = persistent.extent(0) > 0;
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
        if (idB < 0)
          vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
        {  // per-wall AND per-pair material override (a < 0 average keeps the global material)
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
        float wA_n = Nsq * invMassA + genInvMass(TauA, invIA, qA);
        float wB_n = Nsq * invMassB + genInvMass(TauB, invIB, qB);
        // Shock propagation (Guendelman et al. 2003) for persistent LOADED body-body contacts: an
        // inelastic pairwise solve conserves momentum, so a deep column merely homogenises its fall
        // (the floor drains one layer per sweep and the pile never cools -- the phantom-fall state).
        // Treating the LOWER body of a loaded contact as static drains the column's momentum
        // through the support chain into the ground: the upper body is corrected, the lower keeps
        // its (already supported) velocity. Near-horizontal pairs stay symmetric; new contacts and
        // g = 0 runs are untouched (momentum-conserving impacts with material restitution).
        // ... but ONLY when the lower body is not moving UPWARD: correcting the upper body against
        // a static or FALLING support strictly removes momentum (monotone drainage into the
        // ground), while one-sidedness against a RISING support copies its bounce velocity up the
        // chain with no mass penalty and fountains the whole column. Rising supports (floor
        // bounces, bubble eruptions) therefore keep the symmetric momentum-conserving impulse.
        bool applyA = true, applyB = true;
        if (usePersist && persistent(idx) != 0 && idB >= 0) {
          const F3 dx = sub3(ldF3(posPred, idA), ldF3(posPred, idB));  // ghost-aware pair geometry
          const float up = -(dx.x * gHat.x + dx.y * gHat.y + dx.z * gHat.z);  // >0: A above B
          const float thr = 0.3f * Kokkos::sqrt(dot3(dx, dx));
          const float riseThr = 4.0f * restVelThreshold;  // rise = -v.gHat (gHat points down-gravity)
          // ... and the support must be GROUNDED (contact path to the floor): a gas-borne emulsion
          // or lifted slug keeps symmetric momentum-conserving impulses, so its weight stays on
          // the gas -- only genuinely supported chains drain into the ground.
          if (up > thr && -dot3(vB, gHat) <= riseThr && grounded(realB) > 0) {
            wB_n = 0.0f;
            applyB = false;
            restitution = 0.0f;  // shock pass is inelastic: e > 0 one-sided would bounce bodies
                                 // off the ground with unpaid momentum
          } else if (up < -thr && -dot3(vA, gHat) <= riseThr && grounded(realA) > 0) {
            wA_n = 0.0f;
            applyA = false;
            restitution = 0.0f;
          }
        }
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
        if (applyA) {
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
        }
        if (idB >= 0 && applyB) {
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
        if (applyA)
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

/// ---- Warm-started projected Gauss-Seidel (PGS) velocity solve ----
/// Nonsmooth contact dynamics (Moreau-Jean; Stewart-Trinkle LCP): per manifold an ACCUMULATED
/// push impulse p >= 0 along the manifold normal, updated by colored GS sweeps and projected to
/// p >= 0 (a contact may push, never pull; over-push is retracted in later sweeps). At convergence
/// the p's are the contact force network (x dt): a resting pile's chains carry exactly the weight
/// above them, all velocities -> 0 -- statics with no notion of "lower body" or gravity direction,
/// momentum-conserving at every contact (the wall provides the reaction). Deep-pile convergence is
/// bought by WARM STARTING: p is seeded from the previous substep's converged value (matched by
/// pair key) and applied up front, so a static network is re-established in ~1 sweep per substep.
/// Restitution enters as the target relative velocity -e*vn0 on the PRE-SOLVE approach vn0 (with
/// the resting threshold), so fresh binary impacts reproduce the one-shot impulse exactly.

/// Pre-solve approach velocity per manifold (the restitution bias), measured BEFORE the warm-start
/// application. Same kinematics as the sweep kernel.
inline void computeVn0Kokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
                             Kokkos::View<const float* [3], CpMem> velPred,
                             Kokkos::View<const float* [3], CpMem> angVelPred,
                             Kokkos::View<const int*, CpMem> realIdx, float growthRate,
                             Kokkos::View<float*, CpMem> vn0) {
  using detail::ld3;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::pgs_vn0", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;
        const int idA = m.bodyA, idB = m.bodyB;
        const int realA = realIdx(idA);
        if (idB >= 0 && realA > realIdx(idB))
          return;  // periodic dedup
        const float invN = 1.0f / static_cast<float>(m.num_points);
        const F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
        const F3 TauA{m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z};
        const F3 TauB{m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z};
        const F3 vA = ld3(velPred, realA), wA = ld3(angVelPred, realA);
        F3 vB{0, 0, 0}, wB{0, 0, 0};
        if (idB >= 0) {
          vB = ld3(velPred, realIdx(idB));
          wB = ld3(angVelPred, realIdx(idB));
        } else {
          vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
        }
        const F3 rAavg = scale3(F3{m.rA_sum.x, m.rA_sum.y, m.rA_sum.z}, invN);
        const F3 rBavg = scale3(F3{m.rB_sum.x, m.rB_sum.y, m.rB_sum.z}, invN);
        const F3 diffCenters = (idB < 0) ? rAavg : sub3(rAavg, rBavg);
        const F3 vGrowth = scale3(diffCenters, growthRate);
        float vn = dot3(vA, Nsum) + dot3(wA, TauA) + dot3(vB, F3{-Nsum.x, -Nsum.y, -Nsum.z}) +
                   dot3(wB, TauB);
        vn += dot3(vGrowth, Nsum);
        vn0(idx) = vn;
      });
  space.fence();
}

/// Decide each persistent contact's treatment ONCE per substep (before any impulse is applied):
/// 0 = symmetric momentum-conserving PGS, 1 = one-sided with B as the held ground side, 2 = A held.
/// One-sided requires the ground side grounded (contact path to the floor) and not rising. Flagged
/// contacts get their warm impulse ZEROED: their ground side is held externally (recursively down
/// to the floor), so the correct per-substep impulse is only the ~m g dt refill, accumulated from
/// zero -- warm-applying last substep's chain impulse one-sidedly would inject the whole column
/// weight as upward velocity (measured: instant crush + churn from the inconsistent ledger).
inline void computeSideFlagsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                   int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                   Kokkos::View<const unsigned char*, CpMem> persistent,
                                   Kokkos::View<const unsigned char*, CpMem> grounded,
                                   Kokkos::View<const float* [3], CpMem> posPred,
                                   Kokkos::View<const float* [3], CpMem> velPred, F3 gHat,
                                   float riseThr, Kokkos::View<unsigned char*, CpMem> sideFlag,
                                   Kokkos::View<const float*, CpMem> vn0, float approachThr) {
  using detail::ld3;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::pgs_side_flags", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const unsigned char wasPersistent = persistent(idx);  // read BEFORE the write: the caller
        sideFlag(idx) = 0;  // may alias persistent and sideFlag (flag reuses the persistence view)
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0 || m.bodyB < 0 || wasPersistent == 0)
          return;
        // BALLISTIC GATE: one-sided grounding is a statics device -- a pair whose pre-solve
        // relative normal speed exceeds the quasi-static scale (a few substeps of free fall)
        // is shock-loaded or shearing and must stay momentum-conserving. Without this, a fast
        // impactor meets an infinite-mass bed (measured: a 5 m/s ball stops at the surface of a
        // 25k bed) and flowing regions over-resist (silo discharge -24%).
        if (Kokkos::fabs(vn0(idx)) > approachThr)
          return;
        const int realA = realIdx(m.bodyA), realB = realIdx(m.bodyB);
        if (realA > realB)
          return;  // periodic dedup
        const F3 dx = sub3(ldF3(posPred, m.bodyA), ldF3(posPred, m.bodyB));
        const float up = -(dx.x * gHat.x + dx.y * gHat.y + dx.z * gHat.z);  // >0: A above B
        const float thr = 0.3f * Kokkos::sqrt(dot3(dx, dx));
        if (up > thr && -dot3(ld3(velPred, realB), gHat) <= riseThr && grounded(realB) > 0) {
          sideFlag(idx) = 1;
        } else if (up < -thr && -dot3(ld3(velPred, realA), gHat) <= riseThr &&
                   grounded(realA) > 0) {
          sideFlag(idx) = 2;
        }
      });
  space.fence();
}

/// Apply the warm-start impulses up front (order-independent: fixed impulses, atomic adds).
inline void warmStartApplyKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
                                 Kokkos::View<const float*, CpMem> invMass,
                                 Kokkos::View<const float* [3], CpMem> invInertia,
                                 Kokkos::View<const float* [4], CpMem> quat,
                                 Kokkos::View<float* [3], CpMem> velPred,
                                 Kokkos::View<float* [3], CpMem> angVelPred,
                                 Kokkos::View<const int*, CpMem> realIdx,
                                 Kokkos::View<const float*, CpMem> warmP,
                                 Kokkos::View<float* [3], CpMem> warmT) {
  using detail::ld3;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::pgs_warm_apply", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const float p = warmP(idx);
        F3 lt{warmT(idx, 0), warmT(idx, 1), warmT(idx, 2)};
        const bool hasT = (lt.x != 0.0f || lt.y != 0.0f || lt.z != 0.0f);
        if (p == 0.0f && !hasT)
          return;
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;
        const int idA = m.bodyA, idB = m.bodyB;
        const int realA = realIdx(idA);
        const int realB = (idB >= 0) ? realIdx(idB) : idB;
        if (idB >= 0 && realA > realB)
          return;  // periodic dedup (warmP is 0 for dups anyway)
        const float invN = 1.0f / static_cast<float>(m.num_points);
        const F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
        const F3 TauA{m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z};
        const F3 TauB{m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z};
        const F3 rAavg = scale3(F3{m.rA_sum.x, m.rA_sum.y, m.rA_sum.z}, invN);
        const F3 rBavg = scale3(F3{m.rB_sum.x, m.rB_sum.y, m.rB_sum.z}, invN);
        const F3 diffCenters = (idB < 0) ? rAavg : sub3(rAavg, rBavg);
        const float alignment = dot3(Nsum, diffCenters);
        const float sgn = (alignment > 0.0f) ? 1.0f : -1.0f;
        const float lambda = -sgn * p;
        const F3 Jlin = scale3(Nsum, lambda);
        const float invMassA = invMass(realA);
        const float invMassB = (idB >= 0) ? invMass(realB) : 0.0f;
        Kokkos::atomic_add(&velPred(realA, 0), Jlin.x * invMassA);
        Kokkos::atomic_add(&velPred(realA, 1), Jlin.y * invMassA);
        Kokkos::atomic_add(&velPred(realA, 2), Jlin.z * invMassA);
        {
          const F4 qA = F4{quat(realA, 0), quat(realA, 1), quat(realA, 2), quat(realA, 3)};
          const F3 Jl = invRotateVector(qA, scale3(TauA, lambda));
          const F3 invIA = ld3(invInertia, realA);
          const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
          const F3 dww = rotateVector(qA, dwl);
          Kokkos::atomic_add(&angVelPred(realA, 0), dww.x);
          Kokkos::atomic_add(&angVelPred(realA, 1), dww.y);
          Kokkos::atomic_add(&angVelPred(realA, 2), dww.z);
        }
        if (idB >= 0) {
          Kokkos::atomic_add(&velPred(realB, 0), -Jlin.x * invMassB);
          Kokkos::atomic_add(&velPred(realB, 1), -Jlin.y * invMassB);
          Kokkos::atomic_add(&velPred(realB, 2), -Jlin.z * invMassB);
          const F4 qB = F4{quat(realB, 0), quat(realB, 1), quat(realB, 2), quat(realB, 3)};
          const F3 Jl = invRotateVector(qB, scale3(TauB, lambda));
          const F3 invIB = ld3(invInertia, realB);
          const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
          const F3 dww = rotateVector(qB, dwl);
          Kokkos::atomic_add(&angVelPred(realB, 0), dww.x);
          Kokkos::atomic_add(&angVelPred(realB, 1), dww.y);
          Kokkos::atomic_add(&angVelPred(realB, 2), dww.z);
        }
        // Tangential warm impulse: project the stored world-frame accumulator onto the CURRENT
        // tangent plane (the normal moved a little between substeps), write it back so the sweep
        // accumulator starts consistent, then apply +lt on A / -lt on B at the averaged arms.
        if (hasT) {
          const float lenN2 = Kokkos::sqrt(dot3(Nsum, Nsum));
          if (lenN2 > 1e-9f) {
            const F3 nhat = scale3(Nsum, 1.0f / lenN2);
            lt = sub3(lt, scale3(nhat, dot3(lt, nhat)));
          }
          warmT(idx, 0) = lt.x;
          warmT(idx, 1) = lt.y;
          warmT(idx, 2) = lt.z;
          Kokkos::atomic_add(&velPred(realA, 0), lt.x * invMassA);
          Kokkos::atomic_add(&velPred(realA, 1), lt.y * invMassA);
          Kokkos::atomic_add(&velPred(realA, 2), lt.z * invMassA);
          {
            const F4 qA = F4{quat(realA, 0), quat(realA, 1), quat(realA, 2), quat(realA, 3)};
            const F3 Jl = invRotateVector(qA, cross3v(rAavg, lt));
            const F3 invIA = ld3(invInertia, realA);
            const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
            const F3 dww = rotateVector(qA, dwl);
            Kokkos::atomic_add(&angVelPred(realA, 0), dww.x);
            Kokkos::atomic_add(&angVelPred(realA, 1), dww.y);
            Kokkos::atomic_add(&angVelPred(realA, 2), dww.z);
          }
          if (idB >= 0) {
            Kokkos::atomic_add(&velPred(realB, 0), -lt.x * invMassB);
            Kokkos::atomic_add(&velPred(realB, 1), -lt.y * invMassB);
            Kokkos::atomic_add(&velPred(realB, 2), -lt.z * invMassB);
            const F4 qB = F4{quat(realB, 0), quat(realB, 1), quat(realB, 2), quat(realB, 3)};
            const F3 Jl = invRotateVector(qB, cross3v(rBavg, scale3(lt, -1.0f)));
            const F3 invIB = ld3(invInertia, realB);
            const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
            const F3 dww = rotateVector(qB, dwl);
            Kokkos::atomic_add(&angVelPred(realB, 0), dww.x);
            Kokkos::atomic_add(&angVelPred(realB, 1), dww.y);
            Kokkos::atomic_add(&angVelPred(realB, 2), dww.z);
          }
        }
      });
  space.fence();
}

/// One full colored PGS sweep. Per manifold: current approach vtil = s*vn, restitution target
/// -e*max(vtil0,0) (e via the resting threshold on vn0), incremental impulse dp = (vtil-target)/w,
/// accumulator projection p := max(0, p+dp), apply the applied difference in place. maxApproach
/// records the largest applied velocity correction (physical units) for the adaptive stop.
inline void solveVelocityPGSKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                   int numManifolds, Kokkos::View<const int*, CpMem> mColor,
                                   int numColors, Kokkos::View<const float*, CpMem> invMass,
                                   Kokkos::View<const float* [3], CpMem> invInertia,
                                   Kokkos::View<const float* [4], CpMem> quat,
                                   Kokkos::View<float* [3], CpMem> velPred,
                                   Kokkos::View<float* [3], CpMem> angVelPred,
                                   Kokkos::View<const int*, CpMem> realIdx, float growthRate,
                                   float restitutionNormal, float restVelThreshold,
                                   Kokkos::View<float, CpMem> maxApproach,
                                   Kokkos::View<float*, CpMem> lambdaAcc,
                                   Kokkos::View<const float*, CpMem> vn0,
                                   Kokkos::View<const unsigned char*, CpMem> sideFlag,
                                   Kokkos::View<float* [3], CpMem> lambdaT,
                                   float frictionDynamic) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  for (int color = 0; color < numColors; ++color) {
    Kokkos::parallel_for(
        "peclet::dem::solve_velocity_pgs", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
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
          const float invN = 1.0f / static_cast<float>(m.num_points);
          float restitution = restitutionNormal;
          if (idB >= 0) {
            vB = ld3(velPred, realB);
            wB = ld3(angVelPred, realB);
          } else {
            vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
          }
          {  // per-wall AND per-pair material override (a < 0 average keeps the global material)
            const float ra = m.restitution_sum * invN;
            if (ra >= 0.0f)
              restitution = ra;
          }
          const F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
          const F3 TauA{m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z};
          const F3 TauB{m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z};
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
          const float sgn = (alignment > 0.0f) ? 1.0f : -1.0f;
          const float Nsq = dot3(Nsum, Nsum);
          float wA_n = Nsq * invMassA + genInvMass(TauA, invIA, qA);
          float wB_n = Nsq * invMassB + genInvMass(TauB, invIB, qB);
          // Shock propagation INSIDE the PGS sweep (Guendelman staged solve, per-contact form):
          // sidedness was decided ONCE this substep (computeSideFlagsKokkos) so the warm start,
          // accumulator and every sweep share one consistent ledger. The held ground side absorbs
          // the reaction (recursively down to the floor); e is forced 0 on one-sided contacts.
          bool applyA = true, applyB = true;
          const unsigned char sf = sideFlag(idx);
          if (sf == 1) {
            wB_n = 0.0f;
            applyB = false;
            restitution = 0.0f;
          } else if (sf == 2) {
            wA_n = 0.0f;
            applyA = false;
            restitution = 0.0f;
          }
          const float wTotal = wA_n + wB_n;
          if (wTotal <= 0.0f)
            return;
          // Restitution bias on the PRE-SOLVE approach (resting threshold as in the one-shot path).
          const float v0til = sgn * vn0(idx);
          if (Kokkos::fabs(vn0(idx)) < restVelThreshold * lenN)
            restitution = 0.0f;
          const float target = (v0til > 0.0f) ? -restitution * v0til : 0.0f;
          const float vtil = sgn * vn;
          const float dp = (vtil - target) / wTotal;
          const float pOld = lambdaAcc(idx);
          float pNew = pOld + dp;
          if (pNew < 0.0f)
            pNew = 0.0f;
          const float dApplied = pNew - pOld;
          if (dApplied != 0.0f) {
          lambdaAcc(idx) = pNew;
          Kokkos::atomic_max(&maxApproach(), Kokkos::fabs(dApplied) * wTotal / lenN);
          const float lambda = -sgn * dApplied;
          const F3 Jlin = scale3(Nsum, lambda);
          const F3 JangA = scale3(TauA, lambda);
          const F3 JangB = scale3(TauB, lambda);
          if (applyA) {
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
          }
          if (idB >= 0 && applyB) {
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
          }  // dApplied != 0

          // ---- Friction cone (sequential tangential impulse) ----
          // Accumulated world-frame tangential impulse lambdaT on body A, updated by the same
          // colored nonlinear GS: incremental impulse -vt/w_t along the current slip direction,
          // then projection onto the Coulomb disc |lambdaT| <= mu * lambdaN (physical impulse:
          // the normal accumulator is scaled by |Nsum|). Static stick falls out naturally: at
          // vt = 0 the accumulator holds whatever tangential load the cone admits. Sidedness
          // mirrors the normal solve (a held side neither moves nor adds compliance).
          {
            float mu = frictionDynamic;
            const float fa = m.friction_sum * invN;
            if (fa >= 0.0f)
              mu = fa;
            F3 ltOld{lambdaT(idx, 0), lambdaT(idx, 1), lambdaT(idx, 2)};
            const bool haveOld =
                (ltOld.x != 0.0f || ltOld.y != 0.0f || ltOld.z != 0.0f);
            if (mu > 0.0f || haveOld) {
              const F3 nhat = scale3(Nsum, 1.0f / lenN);
              const F3 vA2 = ld3(velPred, realA), wA2 = ld3(angVelPred, realA);
              F3 vB2{0, 0, 0}, wB2{0, 0, 0};
              if (idB >= 0) {
                vB2 = ld3(velPred, realB);
                wB2 = ld3(angVelPred, realB);
              } else {
                vB2 = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
              }
              const F3 vrel = sub3(add3(vA2, cross3v(wA2, rAavg)),
                                   add3(vB2, cross3v(wB2, rBavg)));
              const F3 vt = sub3(vrel, scale3(nhat, dot3(vrel, nhat)));
              const float vtLen = Kokkos::sqrt(dot3(vt, vt));
              F3 ltNew = ltOld;
              float wT = invMassA + invMassB;
              if (vtLen > 1e-9f) {
                const F3 that = scale3(vt, 1.0f / vtLen);
                float wAt =
                    (sf == 2) ? 0.0f
                              : invMassA + genInvMass(cross3v(rAavg, that), invIA, qA);
                float wBt = (sf == 1 || idB < 0)
                                ? 0.0f
                                : invMassB + genInvMass(cross3v(rBavg, that), invIB, qB);
                wT = wAt + wBt;
                if (wT > 1e-9f)
                  ltNew = add3(ltOld, scale3(that, -vtLen / wT));
              }
              ltNew = sub3(ltNew, scale3(nhat, dot3(ltNew, nhat)));
              const float bound = mu * Kokkos::fmax(lambdaAcc(idx), 0.0f) * lenN;
              const float ltLen = Kokkos::sqrt(dot3(ltNew, ltNew));
              if (ltLen > bound)
                ltNew = (bound > 0.0f) ? scale3(ltNew, bound / ltLen) : F3{0, 0, 0};
              const F3 dApp = sub3(ltNew, ltOld);
              if (dApp.x != 0.0f || dApp.y != 0.0f || dApp.z != 0.0f) {
                lambdaT(idx, 0) = ltNew.x;
                lambdaT(idx, 1) = ltNew.y;
                lambdaT(idx, 2) = ltNew.z;
                const float dLen = Kokkos::sqrt(dot3(dApp, dApp));
                if (wT > 1e-9f)
                  Kokkos::atomic_max(&maxApproach(), dLen * wT);
                if (applyA) {
                  velPred(realA, 0) += dApp.x * invMassA;
                  velPred(realA, 1) += dApp.y * invMassA;
                  velPred(realA, 2) += dApp.z * invMassA;
                  const F3 Jl = invRotateVector(qA, cross3v(rAavg, dApp));
                  const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
                  const F3 dww = rotateVector(qA, dwl);
                  angVelPred(realA, 0) += dww.x;
                  angVelPred(realA, 1) += dww.y;
                  angVelPred(realA, 2) += dww.z;
                }
                if (idB >= 0 && applyB) {
                  velPred(realB, 0) += -dApp.x * invMassB;
                  velPred(realB, 1) += -dApp.y * invMassB;
                  velPred(realB, 2) += -dApp.z * invMassB;
                  const F3 Jl =
                      invRotateVector(qB, cross3v(rBavg, scale3(dApp, -1.0f)));
                  const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
                  const F3 dww = rotateVector(qB, dwl);
                  angVelPred(realB, 0) += dww.x;
                  angVelPred(realB, 1) += dww.y;
                  angVelPred(realB, 2) += dww.z;
                }
              }
            }
          }
        });
  }
  space.fence();
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
                                         Kokkos::View<float, CpMem> maxApproach,
                                         Kokkos::View<const unsigned char*, CpMem> persistent = {},
                                         Kokkos::View<const float* [3], CpMem> posPred = {},
                                         F3 gHat = {},
                                         Kokkos::View<const unsigned char*, CpMem> grounded = {}) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  const bool usePersist = persistent.extent(0) > 0;
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
          if (idB < 0)
            vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
          {  // per-wall AND per-pair material override (a < 0 average keeps the global material)
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
          float wA_n = Nsq * invMassA + genInvMass(TauA, invIA, qA);
          float wB_n = Nsq * invMassB + genInvMass(TauB, invIB, qB);
          // Shock propagation for persistent loaded contacts -- see solveVelocityKokkos.
          bool applyA = true, applyB = true;
          if (usePersist && persistent(idx) != 0 && idB >= 0) {
            const F3 dx = sub3(ldF3(posPred, idA), ldF3(posPred, idB));
            const float up = -(dx.x * gHat.x + dx.y * gHat.y + dx.z * gHat.z);
            const float thr = 0.3f * Kokkos::sqrt(dot3(dx, dx));
            const float riseThr = 4.0f * restVelThreshold;  // non-rising + grounded support gate
            if (up > thr && -dot3(vB, gHat) <= riseThr && grounded(realB) > 0) {
              wB_n = 0.0f;
              applyB = false;
              restitution = 0.0f;  // shock pass is inelastic — see solveVelocityKokkos
            } else if (up < -thr && -dot3(vA, gHat) <= riseThr && grounded(realA) > 0) {
              wA_n = 0.0f;
              applyA = false;
              restitution = 0.0f;
            }
          }
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
          if (applyA) {
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
          }
          if (idB >= 0 && applyB) {
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
