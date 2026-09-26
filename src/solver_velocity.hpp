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

#include <climits>
#include <Kokkos_Core.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "contact_preprocessing.hpp"  // ManifoldC, CpExec/CpMem
#include "dem_portable.hpp"
#include "solver_fused.hpp"

namespace peclet::dem {

namespace detail {
KOKKOS_INLINE_FUNCTION F3 ld3(Kokkos::View<const float* [3], CpMem> v, int i) {
  return F3{v(i, 0), v(i, 1), v(i, 2)};
}
// v^T I_world^-1 v with I_world^-1 = R I_local^-1 R^T  ->  (R^T v) diag(invI_local) (R^T v).
KOKKOS_INLINE_FUNCTION float genInvMass(F3 tau, F3 invIlocal, F4 q) {
  // Isotropic inertia (spheres): the principal frame is irrelevant -- skip both rotations.
  if (invIlocal.x == invIlocal.y && invIlocal.y == invIlocal.z)
    return dot3(tau, tau) * invIlocal.x;
  const F3 t = invRotateVector(q, tau);
  return t.x * t.x * invIlocal.x + t.y * t.y * invIlocal.y + t.z * t.z * invIlocal.z;
}
}  // namespace detail

/// Mass-split Jacobi count pass (docs/contact_solve_framework.md §3.1, D3): per body (REAL index),
/// the number of manifolds the Jacobi velocity solve visits -- num_points > 0, one manifold per
/// periodic pair (the solve's realA > realB dedup); a wall side counts nothing. Accumulates into
/// `counts`, which the caller hands over zeroed (the apply clears it), exactly as the solve's own
/// count did. Under MPI the caller then makes the counts global (syncContactCounts).
/// dedupTwins (docs/contact_solve_framework.md §4.2 item 2, §6.2): the single-rank periodic twin
/// rule (skip realIdx(A) > realIdx(B)); off under MPI, where ownership already excludes the
/// non-owned twin and the velocity slot map would make the two rules disagree.
inline void countVelocityJacobiKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                      int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                      Kokkos::View<int*, CpMem> counts, bool dedupTwins = true) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::count_velocity_jacobi", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;
        const int realA = realIdx(m.bodyA);
        if (m.bodyB >= 0) {
          const int realB = realIdx(m.bodyB);
          if (dedupTwins && realA > realB)
            return;  // periodic dedup, as in solveVelocityKokkos
          Kokkos::atomic_add(&counts(realB), 1);
        }
        Kokkos::atomic_add(&counts(realA), 1);
      });
  space.fence();
}

/// Accumulate normal-restitution velocity deltas for `numManifolds` manifolds.
/// massSplit (the 'jacobi' diagnostic, §3.1): `velCounts` holds the global per-body counts n of
/// countVelocityJacobiKokkos and is only read; each manifold is solved against copies of mass
/// m / n (w~ = n_A w_A + n_B w_B, a wall side 0) and the TRUE-mass deltas are accumulated, to be
/// applied with factor 1. Off: the solve bumps `velCounts` and the caller count-averages the sum
/// -- the pre-framework count-averaged Jacobi, kept ONLY for the kernel unit test against its
/// serial reference; no production path calls it (the colour-saturation fallbacks were deleted
/// with complete colouring, docs/contact_solve_framework.md §4.2 / §12 S3).
inline void solveVelocityKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<const float* [3], CpMem> invInertia,
    Kokkos::View<const float* [4], CpMem> quat, Kokkos::View<const float* [3], CpMem> velPred,
    Kokkos::View<const float* [3], CpMem> angVelPred, Kokkos::View<const int*, CpMem> realIdx,
    float growthRate, float restitutionNormal, float restVelThreshold,
    Kokkos::View<float* [3], CpMem> deltaVel, Kokkos::View<float* [3], CpMem> deltaAngVel,
    Kokkos::View<int*, CpMem> velCounts, Kokkos::View<const int*, CpMem> onlyColor = {},
    int colorFilter = 0, Kokkos::View<const unsigned char*, CpMem> persistent = {},
    Kokkos::View<const float* [3], CpMem> posPred = {}, F3 gHat = {},
    Kokkos::View<const unsigned char*, CpMem> grounded = {}, bool massSplit = false,
    bool dedupTwins = true) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  const bool filt = onlyColor.extent(0) > 0;
  const bool usePersist = persistent.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::solve_velocity", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        if (filt && onlyColor(idx) != colorFilter)
          return;  // colour filter (kernel unit test only; no production caller)
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;

        const int idA = m.bodyA, idB = m.bodyB;
        const int realA = realIdx(idA);
        int realB = idB;
        if (idB >= 0) {
          realB = realIdx(idB);
          if (dedupTwins && realA > realB)
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

        // Separation vector for the approaching-sign gate + growth velocity. For a BOUNDARY
        // (idB<0), rBavg is the ABSOLUTE wall contact point (kept absolute for the position solve's
        // plane linearisation), NOT a body-relative lever — so rAavg - rBavg would depend on where
        // the contact sits in world space, flipping `alignment` (and thus the approaching test)
        // around a curved wall / a wall far from the origin and injecting energy (grains "jump" on
        // the way down a rotating drum). The grain's own contact lever rAavg is the meaningful
        // relative vector (dot(Nsum, rAavg) = radius > 0, a consistent convention). Body-body keeps
        // rAavg-rBavg.
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
        // (the floor drains one layer per sweep and the pile never cools -- the phantom-fall
        // state). Treating the LOWER body of a loaded contact as static drains the column's
        // momentum through the support chain into the ground: the upper body is corrected, the
        // lower keeps its (already supported) velocity. Near-horizontal pairs stay symmetric; new
        // contacts and g = 0 runs are untouched (momentum-conserving impacts with material
        // restitution).
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
          const float riseThr =
              4.0f * restVelThreshold;  // rise = -v.gHat (gHat points down-gravity)
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
        // Mass-split Jacobi: the same numerator against copies of mass m / n.
        const float wSolve =
            massSplit ? static_cast<float>(velCounts(realA)) * wA_n +
                            ((idB >= 0) ? static_cast<float>(velCounts(realB)) * wB_n : 0.0f)
                      : wTotal;

        // Resting-contact regularization (the standard PBD/XPBD restitution threshold): bounce only
        // when the physical approach speed |vn|/|Nsum| exceeds ~2 g dt (what one substep of free
        // fall gains). Below it the contact is RESTING — its vn is integration noise, and bouncing
        // it every substep across a dense pile's contact chains (impulses Jacobi-SUMMED per body,
        // no mass splitting) pumps energy without bound: a settled 180k glass-bead bed switched to
        // e=0.8 reached |v| ~ 3e6 cells/s within 15 substeps. With e=0 the impulse still cancels
        // the approach velocity — exactly the quasi-static dissipation a resting pile needs.
        if (Kokkos::fabs(vn) < restVelThreshold * lenN)
          restitution = 0.0f;

        const float lambda = (-restitution * vn - vn) / wSolve;

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
          if (!massSplit)
            Kokkos::atomic_add(&velCounts(realB), 1);
        }
        if (applyA && !massSplit)
          Kokkos::atomic_add(&velCounts(realA), 1);
      });
  space.fence();
}

/// Apply the accumulated velocity deltas AVERAGED by the per-body manifold count — the velocity-
/// solve twin of the position solve's constraint-count averaging (applyUpdatesKokkos). A body in a
/// dense pile receives one full-strength impulse per touching manifold; the raw Jacobi SUM
/// overshoots by ~the contact count and diverges hard for e ≳ 0.5 (a settled 180k glass-bead bed
/// switched to e=0.8 grew |v| by ~5x per substep). A lone binary collision (count 1) is unchanged.
/// averaged = false (the mass-split 'jacobi' diagnostic, §3.1): the deltas are already true-mass
/// impulses against copies of mass m / count, so the apply is a plain add (factor 1).
template <class V3, class Vi>
inline void applyVelocityDeltasAveragedKokkos(int n, V3 velPred, V3 angVelPred, V3 deltaVel,
                                              V3 deltaAngVel, Vi velCounts, bool averaged = true) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::apply_vel_avg", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        const int count = velCounts(i);
        if (count <= 0)
          return;
        // Over-relaxed average: omega=2 halves the convergence loss of plain 1/count averaging
        // (the crush-dissipation rate) while staying far below the raw-sum overshoot (omega=count)
        // that detonates a resting pile at e=0.8. count==1 (binary collision) stays exact.
        const float f = averaged ? Kokkos::fmin(1.0f, 2.0f / static_cast<float>(count)) : 1.0f;
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
// The Jacobi solve above sums every touching manifold's impulse onto a body, then relaxes the sum
// by the contact count to stay stable — a stable but UNDER-converged approximation that
// under-dissipates in dense multi-contact regions (effective restitution rises above the prescribed
// e). The colored Gauss–Seidel path removes that approximation: graph-colour the manifolds so no
// two sharing a real body share a colour, then sweep colour-by-colour applying each impulse IN
// PLACE (read the current velocity, apply, write) — a body sees the updates of every
// previously-solved contact in the same sweep. Within a colour the manifolds are an independent set
// (no shared body), so the in-place read-modify-write is race-free WITHOUT atomics or averaging,
// and the fixed-point is the true coupled multi-contact solution, so the dissipation is correct by
// construction. count==1 (a binary collision) is identical to the Jacobi path; the difference is
// confined to dense clusters.

/// Greedy graph-colour the manifolds: no two manifolds sharing a real body get the same colour.
/// Round-based max-index (Jones–Plassmann) arbitration, no adjacency lists: each round every still-
/// uncoloured manifold contends for its endpoint bodies via atomicMax(bodyWinner, idx); a manifold
/// that wins BOTH endpoints has no uncoloured conflict this round, so it commits the lowest colour
/// free at either endpoint (a per-body bitmask) — and, as the unique winner of those bodies,
/// updates the masks race-free. Inactive manifolds (empty, or the periodic-dedup duplicate
/// realA>realB) are tagged -2 and skipped by the solve. Runs once per step; the colouring is reused
/// across the velocity sweeps. Returns the number of colours used (0 if no active manifolds).
inline int colorManifoldsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
                                Kokkos::View<const int*, CpMem> realIdx, int numReal,
                                Kokkos::View<int*, CpMem> mColor,
                                Kokkos::View<long long*, CpMem> bodyWinner,
                                Kokkos::View<std::uint64_t*, CpMem> bodyMask, int& leftover,
                                Kokkos::View<const unsigned char*, CpMem> sleepMask = {},
                                SlotOverride ov = {}, bool dedupTwins = true, int vertexSpan = 0) {
  leftover = 0;
  CpExec space;
  if (numManifolds <= 0 || numReal <= 0)
    return 0;
  // Colouring vertices: realIdx(body), or a hub copy slot through `ov` (§4.4); vertexSpan covers
  // the copy slots (0 = numReal, the no-copy span).
  const int nV = vertexSpan > 0 ? vertexSpan : numReal;
  const bool sleepOn = sleepMask.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::color_init_bodies", Kokkos::RangePolicy<CpExec>(space, 0, nV),
      KOKKOS_LAMBDA(int i) { bodyMask(i) = 0; });
  // -2 inactive (skip forever), -1 uncoloured, >=0 committed colour.
  Kokkos::parallel_for(
      "peclet::dem::color_init_manifolds", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0 || (sleepOn && sleepMask(idx))) {
          mColor(idx) = -2;  // inactive or a frozen (both-asleep) island manifold
          return;
        }
        // Periodic twin dedup (§4.2 item 2): single-rank only; under MPI ownership already
        // excludes the non-owned twin (dedupTwins = !Hooks::distributed).
        if (dedupTwins && m.bodyB >= 0 && realIdx(m.bodyA) > realIdx(m.bodyB)) {
          mColor(idx) = -2;  // periodic dedup: the (realA<=realB) twin carries this contact
          return;
        }
        mColor(idx) = -1;
      });

  int remaining = 1, prevRemaining = -1;
  // Round cap (docs/contact_solve_framework.md §12 S7): the vertices coloured are MANIFOLDS, so
  // the bound is their count + 2. Every round commits at least the globally highest-key
  // uncoloured manifold (keys are unique: colorKey carries the index in its low word), so the
  // loop terminates within numManifolds rounds; O(log n) in practice.
  const int maxRounds = numManifolds + 2;
  for (int round = 0; round < maxRounds && remaining > 0; ++round) {
    Kokkos::parallel_for(
        "peclet::dem::color_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, nV),
        KOKKOS_LAMBDA(int i) { bodyWinner(i) = -1; });
    Kokkos::parallel_for(
        "peclet::dem::color_contend", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) {
          if (mColor(idx) != -1)
            return;
          const ManifoldC m = manifolds(idx);
          const long long key = colorKey(idx);
          Kokkos::atomic_max(&bodyWinner(ov.slotA(idx, realIdx(m.bodyA))), key);
          if (m.bodyB >= 0)
            Kokkos::atomic_max(&bodyWinner(ov.slotB(idx, realIdx(m.bodyB))), key);
        });
    int rem = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::color_commit", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx, int& acc) {
          if (mColor(idx) != -1)
            return;
          const ManifoldC m = manifolds(idx);
          const int ea = ov.slotA(idx, realIdx(m.bodyA));
          const int eb = (m.bodyB >= 0) ? ov.slotB(idx, realIdx(m.bodyB)) : -1;
          // Winner iff it holds BOTH its endpoints -> no uncoloured conflict, sole writer of ea/eb.
          const long long key = colorKey(idx);
          if (bodyWinner(ea) != key || (eb >= 0 && bodyWinner(eb) != key)) {
            acc += 1;
            return;
          }
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int c = 0;  // lowest free colour of 64; never forced (§4.2 item 1)
          while (c < kColorPalette && ((forbidden >> c) & 1))
            ++c;
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 3
          if (c > 62)
            c = 62;  // G13 mutant 3: the old forced colour 62
#endif
          if (c == kColorPalette) {
            mColor(idx) = kColorUncolourable;  // no free colour: neither coloured nor remaining
            return;
          }
          mColor(idx) = c;
          const std::uint64_t bit = std::uint64_t(1) << c;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    if (rem == prevRemaining)
      break;  // safety bound only: every round commits (or marks -3) the highest-key contender
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
        if (mColor(idx) == -1 || mColor(idx) == kColorUncolourable)
          acc += 1;
      },
      leftover);
  space.fence();
  return maxc + 1;
}

/// Incremental (warm-started) manifold colouring for the single-GPU PGS path. Carry each surviving
/// pair's colour from the previous substep — matched by pair key against the sorted (prevPairKeys,
/// prevColor) ledger committed alongside the warm impulses — seed the per-body colour masks from
/// those carried colours, then run the SAME Jones-Plassmann arbitration ONLY over the NEW
/// (colour == -1) manifolds. Result is bit-identical in FORM to colorManifoldsKokkos (a valid
/// body-disjoint colouring + numColors); only the assignment differs.
///
/// VALIDITY (why the carried colours need no re-arbitration): within a single-GPU run realIndices
/// are stable, so a surviving pair key maps to the same two real bodies. Last substep's colouring
/// was valid, so any two survivors sharing a real body carried DIFFERENT colours — the survivor
/// subset is still a valid colouring. A conflict edge can only appear via a NEW manifold, which is
/// arbitrated against the frozen (carried) masks. Hence seeding the masks non-atomically-safe with
/// atomic_or and running the rounds over the -1 set reproduces a valid colouring. `forceFull`
/// bypasses the carry entirely (fresh substep / colour-count-creep recompaction) — identical output
/// to colorManifoldsKokkos. Single-GPU only: under MPI migration a carried colour can cross into a
/// neighbourhood it never arbitrated against, so the distributed path keeps the full recolour.
inline int colorManifoldsIncrementalKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const int*, CpMem> realIdx, int numReal,
    Kokkos::View<const unsigned long long*, CpMem> prevKeys,
    Kokkos::View<const int*, CpMem> prevColor, int prevCount, Kokkos::View<int*, CpMem> mColor,
    Kokkos::View<long long*, CpMem> bodyWinner, Kokkos::View<std::uint64_t*, CpMem> bodyMask,
    int& leftover, bool forceFull, Kokkos::View<const unsigned char*, CpMem> sleepMask = {},
    bool dedupTwins = true) {
  leftover = 0;
  CpExec space;
  if (numManifolds <= 0 || numReal <= 0)
    return 0;
  Kokkos::parallel_for(
      "peclet::dem::icolor_init_bodies", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) { bodyMask(i) = 0; });
  const bool full = forceFull || prevCount <= 0;
  const bool sleepOn = sleepMask.extent(0) > 0;
  // Seed each manifold's colour: -2 inactive/dedup/both-asleep, else the carried colour (matched by
  // pair key) or -1 (new / full recolour).
  Kokkos::parallel_for(
      "peclet::dem::icolor_seed", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0 || (sleepOn && sleepMask(idx))) {
          mColor(idx) = -2;  // inactive or a frozen (both-asleep) island manifold
          return;
        }
        if (dedupTwins && m.bodyB >= 0 && realIdx(m.bodyA) > realIdx(m.bodyB)) {
          mColor(idx) = -2;  // periodic dedup twin
          return;
        }
        int c = -1;
        if (!full) {
          const unsigned long long k = pairKeyOf(m, realIdx);
          int lo = 0, hi = prevCount;
          while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (prevKeys(mid) < k)
              lo = mid + 1;
            else
              hi = mid;
          }
          if (lo < prevCount && prevKeys(lo) == k) {
            const int pc = prevColor(lo);
            if (pc >= 0)
              c = pc;  // carry (leftover -1 last step -> re-arbitrate as new)
          }
        }
        mColor(idx) = c;
      });
  // Seed the per-body masks from the carried colours (survivors are conflict-free on single-GPU).
  if (!full)
    Kokkos::parallel_for(
        "peclet::dem::icolor_seed_mask", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) {
          const int c = mColor(idx);
          if (c < 0)
            return;
          const ManifoldC m = manifolds(idx);
          const std::uint64_t bit = std::uint64_t(1) << c;
          Kokkos::atomic_or(&bodyMask(realIdx(m.bodyA)), bit);
          if (m.bodyB >= 0)
            Kokkos::atomic_or(&bodyMask(realIdx(m.bodyB)), bit);
        });
  // Jones-Plassmann arbitration over the uncoloured (-1) set only — identical body of work to
  // colorManifoldsKokkos, but the frozen carried colours restrict it to the few new manifolds.
  int remaining = 1, prevRemaining = -1;
  const int maxRounds = numManifolds + 2;  // §12 S7: bounded by the vertices coloured
  for (int round = 0; round < maxRounds && remaining > 0; ++round) {
    Kokkos::parallel_for(
        "peclet::dem::icolor_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
        KOKKOS_LAMBDA(int i) { bodyWinner(i) = -1; });
    Kokkos::parallel_for(
        "peclet::dem::icolor_contend", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
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
        "peclet::dem::icolor_commit", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx, int& acc) {
          if (mColor(idx) != -1)
            return;
          const ManifoldC m = manifolds(idx);
          const int ea = realIdx(m.bodyA);
          const int eb = (m.bodyB >= 0) ? realIdx(m.bodyB) : -1;
          const long long key = colorKey(idx);
          if (bodyWinner(ea) != key || (eb >= 0 && bodyWinner(eb) != key)) {
            acc += 1;
            return;
          }
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int c = 0;  // lowest free colour of 64; never forced (§4.2 item 1)
          while (c < kColorPalette && ((forbidden >> c) & 1))
            ++c;
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 3
          if (c > 62)
            c = 62;  // G13 mutant 3: the old forced colour 62
#endif
          if (c == kColorPalette) {
            mColor(idx) = kColorUncolourable;
            return;
          }
          mColor(idx) = c;
          const std::uint64_t bit = std::uint64_t(1) << c;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    if (rem == prevRemaining)
      break;  // safety bound only (see colorManifoldsKokkos)
    prevRemaining = rem;
    remaining = rem;
  }
  int maxc = -1;
  Kokkos::parallel_reduce(
      "peclet::dem::icolor_max", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx, int& mx) {
        if (mColor(idx) > mx)
          mx = mColor(idx);
      },
      Kokkos::Max<int>(maxc));
  Kokkos::parallel_reduce(
      "peclet::dem::icolor_leftover", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx, int& acc) {
        if (mColor(idx) == -1 || mColor(idx) == kColorUncolourable)
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
                             Kokkos::View<float*, CpMem> vn0, Kokkos::View<float* [3], CpMem> vt0,
                             bool dedupTwins = true) {
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
        if (dedupTwins && idB >= 0 && realA > realIdx(idB))
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
        // Pre-solve tangential surface velocity at the averaged contact point (physical units):
        // the reference for the tangential-restitution target -beta * vt0 (Walton impact law).
        {
          const float lenN = Kokkos::sqrt(dot3(Nsum, Nsum));
          F3 vt{0, 0, 0};
          if (lenN > 1e-9f) {
            const F3 nhat = scale3(Nsum, 1.0f / lenN);
            const int realB2 = (idB >= 0) ? realIdx(idB) : -1;
            const F3 wAv = ld3(angVelPred, realA);
            const F3 wBv = (realB2 >= 0) ? ld3(angVelPred, realB2) : F3{0, 0, 0};
            const F3 vrel = sub3(add3(vA, cross3v(wAv, rAavg)), add3(vB, cross3v(wBv, rBavg)));
            vt = sub3(vrel, scale3(nhat, dot3(vrel, nhat)));
          }
          vt0(idx, 0) = vt.x;
          vt0(idx, 1) = vt.y;
          vt0(idx, 2) = vt.z;
        }
      });
}

/// Event-level (Poisson) restitution bookkeeping, once per substep AFTER all velocity phases
/// (restitutionModel == 1 only). Per pair, two carried floats define the current impact EVENT:
///  * restVPeak — the event's peak physical approach speed (> 0 = event active). Set/refreshed
///    whenever the pre-solve approach is kinetic (v0til > vRest*lenN), decayed by 1/256 per
///    substep, cleared once it ages below the resting threshold: a buried/absorbed event's bank
///    EVAPORATES (energy went to heat) instead of popping the pile later. A resting pile never
///    sets vPeak, so it never banks — the dense-pile energy-bomb guard.
///  * restBank — the remaining OWED separation impulse: while the event is active the contact
///    banks e x its applied normal-impulse FLUX every substep (pTot - pR, the momentum the chain
///    actually transmitted; a per-substep kinetic gate would miss the co-moving compression
///    plateau where vn0 ~ 0), minus the separation already delivered (pR — so a clean one-substep
///    binary impact nets exactly 0), minus what the sweep's release channel injected (restRel).
inline void updateRestitutionBankKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<const float* [3], CpMem> invInertia,
    Kokkos::View<const float* [4], CpMem> quat, Kokkos::View<const float* [3], CpMem> velPred,
    Kokkos::View<const float* [3], CpMem> angVelPred, Kokkos::View<const int*, CpMem> realIdx,
    float growthRate, float restitutionNormal, float restVelThreshold,
    Kokkos::View<const float*, CpMem> vn0, Kokkos::View<const float*, CpMem> lambdaAcc,
    Kokkos::View<const float*, CpMem> restRel, Kokkos::View<float*, CpMem> restBank,
    Kokkos::View<float*, CpMem> restVPeak, bool dedupTwins = true) {
  using detail::genInvMass;
  using detail::ld3;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::rest_bank_update", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0)
          return;
        const int idA = m.bodyA, idB = m.bodyB;
        const int realA = realIdx(idA);
        const int realB = (idB >= 0) ? realIdx(idB) : idB;
        if (dedupTwins && idB >= 0 && realA > realB)
          return;  // periodic dedup: the canonical twin owns the bank
        const float invN = 1.0f / static_cast<float>(m.num_points);
        const F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
        const float lenN = Kokkos::sqrt(dot3(Nsum, Nsum));
        if (lenN < 1e-9f)
          return;
        const F3 TauA{m.torque_armA_sum.x, m.torque_armA_sum.y, m.torque_armA_sum.z};
        const F3 TauB{m.torque_armB_sum.x, m.torque_armB_sum.y, m.torque_armB_sum.z};
        const F3 vA = ld3(velPred, realA), wA = ld3(angVelPred, realA);
        F3 vB{0, 0, 0}, wB{0, 0, 0};
        if (idB >= 0) {
          vB = ld3(velPred, realB);
          wB = ld3(angVelPred, realB);
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
        const float alignment = dot3(Nsum, diffCenters);
        const float sgn = (alignment > 0.0f) ? 1.0f : -1.0f;
        // Symmetric effective inverse mass (the release/bank ledger is momentum bookkeeping at
        // the pair's own masses, independent of any stabilization sidedness this substep).
        const F3 invIA = ld3(invInertia, realA);
        const F3 invIB = (idB >= 0) ? ld3(invInertia, realB) : F3{0, 0, 0};
        const F4 qA = F4{quat(realA, 0), quat(realA, 1), quat(realA, 2), quat(realA, 3)};
        const F4 qB = (idB >= 0)
                          ? F4{quat(realB, 0), quat(realB, 1), quat(realB, 2), quat(realB, 3)}
                          : F4{0, 0, 0, 1};
        const float Nsq = dot3(Nsum, Nsum);
        const float wTotal = Nsq * invMass(realA) + genInvMass(TauA, invIA, qA) +
                             Nsq * ((idB >= 0) ? invMass(realB) : 0.0f) +
                             genInvMass(TauB, invIB, qB);
        if (wTotal <= 0.0f)
          return;
        float e = restitutionNormal;
        {
          const float ra = m.restitution_sum * invN;
          if (ra >= 0.0f)
            e = ra;
        }
        const float v0til = sgn * vn0(idx);
        const float vtilEnd = sgn * vn;
        float owed = restBank(idx);
        // Event state: refresh the peak on a kinetic approach, age it 1/256 per substep.
        float vPeak = restVPeak(idx) * (1.0f - 1.0f / 256.0f);
        if (v0til > restVelThreshold * lenN)
          vPeak = Kokkos::fmax(vPeak, v0til / lenN);
        if (vPeak > restVelThreshold) {
          // Active event: bank e x this substep's compression FLUX (full applied normal impulse
          // minus the reflection share — the momentum the chain actually transmitted; the
          // m_eff-scale approach-destruction measure under-banks a chain-loaded impact by ~3000x),
          // pay down the separation already delivered, and deduct the release channel's spend.
          // A clean one-substep binary impact nets 0: pTot = (1+e) m_eff v0, pR = e m_eff v0.
          const float pR = Kokkos::fmax(0.0f, -vtilEnd) / wTotal * lenN;
          const float pTot = Kokkos::fmax(lambdaAcc(idx), 0.0f) * lenN;
          const float pC = Kokkos::fmax(0.0f, pTot - pR);
          owed += e * pC - pR - Kokkos::fmax(restRel(idx), 0.0f) * lenN;
          restBank(idx) = Kokkos::fmax(owed, 0.0f);
          restVPeak(idx) = vPeak;
        } else {
          restBank(idx) = 0.0f;  // event aged out (or none): the residual budget evaporates
          restVPeak(idx) = 0.0f;
        }
      });
}

/// Orphan-account aging, once per substep over the OWNED bodies: both the balance and the carried
/// event peak decay 1/64 per substep (e-fold ~3 ms at dt = 5e-5 — long enough for the rebound
/// payout, which completes within ~2 ms of turnaround, short enough that a BURIED impactor's
/// slowly re-fed credit cannot keep fluidizing its crater: with the pair-state 1/256 decay the
/// 100k Dosta plateau crept 0.007 deeper), and once the peak ages below the resting threshold
/// the whole account evaporates — stranded credit on a settling body never pops the pile later.
inline void decayBodyOrphanKokkos(Kokkos::View<float*, CpMem> orphan,
                                  Kokkos::View<float*, CpMem> orphanVPeak, int numOwned,
                                  float restVelThreshold) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::rest_orphan_decay", Kokkos::RangePolicy<CpExec>(space, 0, numOwned),
      KOKKOS_LAMBDA(int i) {
        const float decayed = orphanVPeak(i) * (1.0f - 1.0f / 64.0f);
        if (decayed <= restVelThreshold || orphan(i) <= 0.0f) {
          orphan(i) = 0.0f;
          orphanVPeak(i) = 0.0f;
        } else {
          orphan(i) *= (1.0f - 1.0f / 64.0f);
          orphanVPeak(i) = decayed;
        }
      });
}

/// Orphan transfer: previous-ledger entries NOT matched by any current manifold (their pair died
/// this substep) credit their remaining owed budget to the endpoint BODIES, mass-weighted — the
/// heavier endpoint keeps the larger share (energy of a later release scales J^2/2m, so the light
/// grain is the dangerous store and the heavy impactor is both the safe one and the event's
/// physical carrier); boundary pairs (wall endpoint) credit everything to the particle. The
/// carried event peak joins by max. Key components are keyIdx identities: REAL slots on the
/// single-GPU path (direct index); global ids under MPI — resolved through the sorted
/// (gidSorted, slotSorted) map over owned + ghost slots. Under MPI the pair's bank lives only in
/// its owning rank's ledger (docs/mpi_momentum_conservation.md §4.6): a credit to a ghost endpoint
/// is delivered to that body's owner by the velocity reverse at the next sync, and an endpoint
/// that has left this rank's halo is not found (it gets the conservative half, and that half is
/// dropped).
inline void scatterOrphanBanksKokkos(
    Kokkos::View<const unsigned long long*, CpMem> prevKeys,
    Kokkos::View<const float*, CpMem> prevRestBank, Kokkos::View<const float*, CpMem> prevRestVPeak,
    Kokkos::View<const unsigned char*, CpMem> matched, int prevCount,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<float*, CpMem> orphan,
    Kokkos::View<float*, CpMem> orphanVPeak, Kokkos::View<const int*, CpMem> gidSorted = {},
    Kokkos::View<const int*, CpMem> slotSorted = {}, int ownedLimit = -1) {
  // ownedLimit >= 0 (the distributed step: numReal): credit a dead entry only on the rank that
  // owns its LOWER-gid endpoint -- with symmetric visibility (§5.1) exactly the pair's owner, so
  // an entry carried on both endpoints' ranks (migration pack) is credited once
  // (docs/contact_solve_framework.md §6.3). -1: no restriction (single rank, byte-identical).
  CpExec space;
  const int nMap = static_cast<int>(gidSorted.extent(0));
  const int nBody = static_cast<int>(orphan.extent(0));
  Kokkos::parallel_for(
      "peclet::dem::rest_orphan_scatter", Kokkos::RangePolicy<CpExec>(space, 0, prevCount),
      KOKKOS_LAMBDA(int e) {
        if (matched(e))
          return;
        const float owed = prevRestBank(e);
        if (owed <= 0.0f)
          return;
        const unsigned long long k = prevKeys(e);
        if (k == ~0ull)
          return;
        const unsigned hi = static_cast<unsigned>(k >> 32);
        const unsigned lo = static_cast<unsigned>(k & 0xFFFFFFFFu);
        auto resolve = [&](unsigned id) -> int {
          if (nMap == 0)  // single-GPU: identities ARE real slots
            return (static_cast<int>(id) < nBody) ? static_cast<int>(id) : -1;
          int a = 0, b = nMap;  // MPI: binary-search the sorted gid -> slot map
          while (a < b) {
            const int m = (a + b) >> 1;
            if (gidSorted(m) < static_cast<int>(id))
              a = m + 1;
            else
              b = m;
          }
          return (a < nMap && gidSorted(a) == static_cast<int>(id)) ? slotSorted(a) : -1;
        };
        const int sA = resolve(hi);
        if (ownedLimit >= 0 && !(sA >= 0 && sA < ownedLimit))
          return;  // not the pair's owner: a stale copy is never credited
        const int sB = (lo != 0xFFFFFFFFu) ? resolve(lo) : -1;
        const float vpk = prevRestVPeak(e);
        float shareA = 1.0f;  // boundary (wall) pair: everything to the particle
        if (lo != 0xFFFFFFFFu) {
          if (sB >= 0 && sA >= 0) {
            const float wA = invMass(sA), wB = invMass(sB);
            if (wA + wB > 0.0f)
              shareA = wB / (wA + wB);  // heavier endpoint (smaller invMass) keeps more
          } else {
            shareA = 0.5f;  // body-body with an unresolvable endpoint (MPI edge): conservative half
          }
        }
        if (sA >= 0 && shareA > 0.0f) {
          Kokkos::atomic_add(&orphan(sA), owed * shareA);
          Kokkos::atomic_max(&orphanVPeak(sA), vpk);
        }
        if (sB >= 0 && shareA < 1.0f) {
          Kokkos::atomic_add(&orphan(sB), owed * (1.0f - shareA));
          Kokkos::atomic_max(&orphanVPeak(sB), vpk);
        }
      });
}

/// Poisson-restitution diagnostics: (sum, max, count>0) over the committed owed-impulse store
/// (namespace scope: nvcc forbids KOKKOS_LAMBDA in member functions).
inline std::tuple<double, float, int> restBankStatsKokkos(Kokkos::View<const float*, CpMem> bank,
                                                          int n) {
  double s = 0.0;
  float mx = 0.0f;
  int cnt = 0;
  if (n > 0) {
    Kokkos::parallel_reduce(
        "peclet::dem::rest_bank_stats", Kokkos::RangePolicy<CpExec>(0, n),
        KOKKOS_LAMBDA(int i, double& ls, float& lm, int& lc) {
          const float v = bank(i);
          ls += v;
          if (v > lm)
            lm = v;
          if (v > 0.0f)
            ++lc;
        },
        s, Kokkos::Max<float>(mx), Kokkos::Sum<int>(cnt));
  }
  return {s, mx, cnt};
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
                                   Kokkos::View<const float*, CpMem> vn0, float approachThr,
                                   bool dedupTwins = true) {
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
        if (dedupTwins && realA > realB)
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
                                 Kokkos::View<float* [3], CpMem> warmT, bool dedupTwins = true) {
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
        if (dedupTwins && idB >= 0 && realA > realB)
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
}

/// One full colored PGS sweep. Per manifold: current approach vtil = s*vn, restitution target
/// -e*max(vtil0,0) (Newton; Moreau: -e*vtil0, see restTarget) (e via the resting threshold on
/// vn0), incremental impulse dp = (vtil-target)/w,
/// accumulator projection p := max(0, p+dp), apply the applied difference in place. maxApproach
/// records the largest applied velocity correction (physical units) for the adaptive stop.
///
/// The per-manifold solve body lives in PGSManifoldSweep so the colored sweep and the
/// level-ordered ("multilevel") bucket sweep share it verbatim: solveOne(idx) must only run
/// concurrently on manifolds that are body-disjoint within one launch (a colour class, or a
/// (level, colour) bucket -- a subset of a colour class).
struct PGSManifoldSweep {
  Kokkos::View<const ManifoldC*, CpMem> manifolds;
  Kokkos::View<const float*, CpMem> invMass;
  Kokkos::View<const float* [3], CpMem> invInertia;
  Kokkos::View<const float* [4], CpMem> quat;
  Kokkos::View<float* [3], CpMem> velPred;
  Kokkos::View<float* [3], CpMem> angVelPred;
  Kokkos::View<const int*, CpMem> realIdx;
  float growthRate;
  float restitutionNormal;
  float restVelThreshold;
  Kokkos::View<float, CpMem> maxApproach;
  // Quasi-static residual (corrections on contacts with |vn0| <= 4 restVelThreshold): the
  // multilevel pass's stop criterion. The colored caller aliases this to maxApproach (the
  // duplicate atomic_max is idempotent), so main-loop behaviour is unchanged.
  Kokkos::View<float, CpMem> maxApproachQS;
  Kokkos::View<float*, CpMem> lambdaAcc;
  Kokkos::View<const float*, CpMem> vn0;
  Kokkos::View<const unsigned char*, CpMem> sideFlag;
  Kokkos::View<float* [3], CpMem> lambdaT;
  float frictionDynamic;
  Kokkos::View<const float* [3], CpMem> vt0;
  float restitutionTangent;
  Kokkos::View<const float*, CpMem> posImpulse;
  // Event-level (Poisson) restitution release (restitutionModel == 1; both views empty
  // otherwise): restBank(idx) is the pair's remaining OWED separation impulse (physical units,
  // warm-carried by pair key) and restRel(idx) the per-substep release accumulator (lambda
  // units, zeroed each substep). Per-substep Newton restitution stays ALIVE alongside the
  // bank — its micro-reflections are genuine returned energy, and the accounting's pR term
  // deducts each one from the owed budget so the channels never double-count (measured: forcing
  // e = 0 on persistent contacts cost more rebound than the bank recovered). restBank is
  // writable: a releasing pair whose own budget rails can DRAW from its bodies' orphan accounts
  // (transferred into restBank at the moment of need, so the post-solve accounting sees one
  // consistent pair ledger). Same-body contacts never run concurrently (colouring), so the
  // in-place body-account read-modify-write is race-free without atomics.
  Kokkos::View<float*, CpMem> restBank;
  Kokkos::View<float*, CpMem> restRel;
  // Event peak approach speed (physical): caps the release separation-velocity target at
  // e x vPeak — the event-level rebound speed — so a large flux-banked budget against a light
  // partner becomes a SUSTAINED unloading push over many substeps (the Hertz-like collective
  // rebound) instead of an impulsive dump (a 190 m/s kick to a 1e-5 kg grain, measured absurd).
  Kokkos::View<const float*, CpMem> restVPeak;
  // Orphan accounts (see Particles::bodyOrphan): balance + carried event peak speed per REAL
  // body. The peak matters as much as the balance — a contact formed late under a decelerating
  // impactor only saw the residual approach, so its own e*vPeak target would cap the rebound at
  // e x the late-stage speed; the orphaned peak restores the full event's velocity scale.
  Kokkos::View<float*, CpMem> restOrphan;
  Kokkos::View<const float*, CpMem> restOrphanVPeak;
  // Hub-copy slot overrides + split relaxation (docs/contact_solve_framework.md §4.4, §4.5);
  // empty = realIdx, no relaxation. Every state access below uses realA / realB.
  SlotOverride ov{};
  // Restitution target law (docs/contact_physics_followups.md §4, WO-C1; a diagnostics A/B,
  // Particles::restitutionTarget): 0 = Newton (default) -- -e v0til on a contact APPROACHING
  // before the solve, 0 on a pre-separating one; 1 = Moreau -- -e v0til on EVERY closed contact
  // whose pre-solve |vn0| reaches the resting threshold (a pre-separating pair driven into
  // approach may approach up to e |v0til| before it resists). Moreau is energy-consistent for a
  // uniform e (§4.1). Never combined with the Poisson bank (the setter refuses, R-C4).
  int restTarget = 0;

  KOKKOS_FUNCTION void solveOne(int idx) const {
    using detail::genInvMass;
    using detail::ld3;
    {
      const ManifoldC m = manifolds(idx);
      const int idA = m.bodyA, idB = m.bodyB;
      const int realA = ov.slotA(idx, realIdx(idA));
      const int realB = (idB >= 0) ? ov.slotB(idx, realIdx(idB)) : idB;
      const float invMassA = invMass(realA);
      const float invMassB = (idB >= 0) ? invMass(realB) : 0.0f;
      const F3 invIA = ld3(invInertia, realA);
      const F3 invIB = (idB >= 0) ? ld3(invInertia, realB) : F3{0, 0, 0};
      const F4 qA = F4{quat(realA, 0), quat(realA, 1), quat(realA, 2), quat(realA, 3)};
      const F4 qB = (idB >= 0) ? F4{quat(realB, 0), quat(realB, 1), quat(realB, 2), quat(realB, 3)}
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
      const float eMat = restitution;  // raw material e (release cap), before the event gates
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
      // Poisson mode keeps per-substep Newton restitution ALIVE alongside the bank: the
      // micro-reflections it produces are genuine returned energy (measured: forcing e = 0 on
      // persistent contacts cost more rebound than the bank recovered), and the accounting's pR
      // term deducts every reflection from the owed budget, so the two channels never
      // double-count.
      float target;
      if (restTarget == 1)  // Moreau: restitution is already 0 below the threshold / one-sided
        target = (Kokkos::fabs(vn0(idx)) >= restVelThreshold * lenN) ? -restitution * v0til : 0.0f;
      else  // Newton (default)
        target = (v0til > 0.0f) ? -restitution * v0til : 0.0f;
      const float vtil = sgn * vn;
      float dp = (vtil - target) / wTotal;
      if (ov.relax(realA, realB))
        dp *= ov.omega;  // split relaxation before the clamp (§4.5; omega_vel = 1)
      const float pOld = lambdaAcc(idx);
      float pNew = pOld + dp;
      if (pNew < 0.0f)
        pNew = 0.0f;
      const float dApplied = pNew - pOld;
      if (dApplied != 0.0f) {
        lambdaAcc(idx) = pNew;
        Kokkos::atomic_max(&maxApproach(), Kokkos::fabs(dApplied) * wTotal / lenN);
        if (Kokkos::fabs(vn0(idx)) <= 4.0f * restVelThreshold * lenN)
          Kokkos::atomic_max(&maxApproachQS(), Kokkos::fabs(dApplied) * wTotal / lenN);
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

      // ---- Event-level (Poisson) restitution release ----
      // A pair with banked compression (restBank > 0) that is NOT in a kinetic approach pushes
      // toward the event's separation-velocity target -owed*w — what releasing the full remaining
      // budget delivers against the pair's effective mass, i.e. the event-level rebound speed.
      // The push runs through its OWN accumulator clamped to [0, owed] (the friction-cone-shaped
      // budget cap): a free pair leaves at the target speed, while a loaded chain re-absorbs the
      // attempt through lambda >= 0 on its other contacts with the total injected impulse bounded
      // by the budget — velocity-targeted, never impulsive, no unbounded force fight. One-sided
      // (side-flagged) contacts are held externally and never release. Spent budget is deducted
      // once per substep by updateRestitutionBankKokkos. Unloading detection: KINETICALLY
      // separating pre-solve approach (beyond the resting threshold) — jitter separations during
      // compression (position-solve pushback, chain oscillation) fire sub-threshold every substep
      // and would drain the bank as fast as it fills (measured: bank plateaued at ~1/6 of the
      // event flux with a v0til < 0 gate); the genuine rebound onset separates kinetically.
      if (restRel.extent(0) > 0 && sf == 0 && v0til < -restVelThreshold * lenN) {
        float owed = restBank(idx);
        float vPeak = restVPeak.extent(0) > 0 ? restVPeak(idx) : 0.0f;
        // Orphan availability at the endpoints: makes budget-less fresh pairs under an event
        // carrier eligible, and lifts the velocity target to the orphaned event peak.
        float orphA = 0.0f, orphB = 0.0f;
        if (restOrphan.extent(0) > 0) {
          orphA = restOrphan(realA);
          if (orphA > 0.0f)
            vPeak = Kokkos::fmax(vPeak, restOrphanVPeak(realA));
          if (idB >= 0) {
            orphB = restOrphan(realB);
            if (orphB > 0.0f)
              vPeak = Kokkos::fmax(vPeak, restOrphanVPeak(realB));
          }
        }
        if ((owed > 0.0f || orphA > 0.0f || orphB > 0.0f) && vPeak > 0.0f) {
          const F3 vA3 = ld3(velPred, realA), wA3 = ld3(angVelPred, realA);
          F3 vB3{0, 0, 0}, wB3{0, 0, 0};
          if (idB >= 0) {
            vB3 = ld3(velPred, realB);
            wB3 = ld3(angVelPred, realB);
          } else {
            vB3 = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
          }
          float vn3 = dot3(vA3, Nsum) + dot3(wA3, TauA) + dot3(vB3, F3{-Nsum.x, -Nsum.y, -Nsum.z}) +
                      dot3(wB3, TauB);
          vn3 += dot3(vGrowth, Nsum);
          const float vtil3 = sgn * vn3;
          // Symmetric release: both endpoints are pushed. (A one-sided variant that held a
          // grounded lower side was measured WORSE on the 25k Dosta impact — +0.60 vs +0.88
          // rebound — because the light partner's downward reaction is what re-compresses and
          // re-releases the layers below; it was retired with the env A/B in 1.0.0.)
          const bool relA = true, relB = (idB >= 0);
          const float wRel = wTotal;
          if (wRel > 0.0f) {
            // Separation-velocity target: the event-level rebound speed e x vPeak. The budget is
            // enforced by the accumulator clamp alone — folding it into the velocity target
            // (owed * wRel) prematurely stalls a one-sided release against a heavy impactor
            // (owed/m_ball ~ cm/s) with most of the budget unspent. Physical velocity -> vtil
            // units is x lenN.
            const float vTphys = eMat * vPeak;
            const float targetR = -vTphys * lenN;
            const float dpR = (vtil3 - targetR) / wRel;
            float cap = owed / lenN;  // physical budget in lambda units
            const float rOld = restRel(idx);
            float rNew = rOld + dpR;
            if (rNew < 0.0f)
              rNew = 0.0f;
            // Own budget railed with orphan balance at the endpoints: draw the shortfall from the
            // body accounts INTO the pair bank (heavier-first order is irrelevant; drained in
            // sequence). The drawn amount is spent by this very increment, so the post-solve
            // accounting's owed -= released cancels it exactly.
            if (rNew > cap && (orphA > 0.0f || orphB > 0.0f)) {
              float need = (rNew - cap) * lenN;
              float draw = 0.0f;
              if (orphA > 0.0f) {
                const float d = Kokkos::fmin(need, orphA);
                restOrphan(realA) = orphA - d;
                need -= d;
                draw += d;
              }
              if (need > 0.0f && orphB > 0.0f) {
                const float d = Kokkos::fmin(need, orphB);
                restOrphan(realB) = orphB - d;
                draw += d;
              }
              if (draw > 0.0f) {
                owed += draw;
                restBank(idx) = owed;
                cap = owed / lenN;
              }
            }
            if (rNew > cap)
              rNew = cap;
            const float dR = rNew - rOld;
            if (dR != 0.0f) {
              restRel(idx) = rNew;
              Kokkos::atomic_max(&maxApproach(), Kokkos::fabs(dR) * wRel / lenN);
              if (Kokkos::fabs(vn0(idx)) <= 4.0f * restVelThreshold * lenN)
                Kokkos::atomic_max(&maxApproachQS(), Kokkos::fabs(dR) * wRel / lenN);
              const float lambdaR = -sgn * dR;
              const F3 JlinR = scale3(Nsum, lambdaR);
              if (relA) {
                velPred(realA, 0) += JlinR.x * invMassA;
                velPred(realA, 1) += JlinR.y * invMassA;
                velPred(realA, 2) += JlinR.z * invMassA;
                const F3 Jl = invRotateVector(qA, scale3(TauA, lambdaR));
                const F3 dwl{Jl.x * invIA.x, Jl.y * invIA.y, Jl.z * invIA.z};
                const F3 dww = rotateVector(qA, dwl);
                angVelPred(realA, 0) += dww.x;
                angVelPred(realA, 1) += dww.y;
                angVelPred(realA, 2) += dww.z;
              }
              if (relB) {
                velPred(realB, 0) += -JlinR.x * invMassB;
                velPred(realB, 1) += -JlinR.y * invMassB;
                velPred(realB, 2) += -JlinR.z * invMassB;
                const F3 Jl = invRotateVector(qB, scale3(TauB, lambdaR));
                const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
                const F3 dww = rotateVector(qB, dwl);
                angVelPred(realB, 0) += dww.x;
                angVelPred(realB, 1) += dww.y;
                angVelPred(realB, 2) += dww.z;
              }
            }
          }
        }
      }

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
        const bool haveOld = (ltOld.x != 0.0f || ltOld.y != 0.0f || ltOld.z != 0.0f);
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
          const F3 vrel = sub3(add3(vA2, cross3v(wA2, rAavg)), add3(vB2, cross3v(wB2, rBavg)));
          const F3 vt = sub3(vrel, scale3(nhat, dot3(vrel, nhat)));
          // Walton tangential restitution: for a COLLIDING contact the target surface
          // velocity is -beta * vt0 (pre-solve tangential velocity); sustained contacts
          // (below the resting threshold) and one-sided stabilization contacts run beta = 0
          // -- the same event classification that gates normal restitution. The cone clamp
          // below turns this into the stick/slide transition of the (e, mu, beta) law.
          float beta = restitutionTangent;
          if (beta != 0.0f && (sf != 0 || Kokkos::fabs(vn0(idx)) < restVelThreshold * lenN))
            beta = 0.0f;
          F3 vtErr = vt;
          if (beta != 0.0f && vt0.extent(0) > 0) {
            F3 v0{vt0(idx, 0), vt0(idx, 1), vt0(idx, 2)};
            v0 = sub3(v0, scale3(nhat, dot3(v0, nhat)));  // current tangent plane
            vtErr = add3(vt, scale3(v0, beta));
          }
          const float vtLen = Kokkos::sqrt(dot3(vtErr, vtErr));
          F3 ltNew = ltOld;
          float wT = invMassA + invMassB;
          if (vtLen > 1e-9f) {
            const F3 that = scale3(vtErr, 1.0f / vtLen);
            float wAt = (sf == 2) ? 0.0f : invMassA + genInvMass(cross3v(rAavg, that), invIA, qA);
            float wBt = (sf == 1 || idB < 0)
                            ? 0.0f
                            : invMassB + genInvMass(cross3v(rBavg, that), invIB, qB);
            wT = wAt + wBt;
            if (wT > 1e-9f)
              ltNew = add3(ltOld, scale3(that, -vtLen / wT));
          }
          ltNew = sub3(ltNew, scale3(nhat, dot3(ltNew, nhat)));
          // Coulomb bound = mu * TOTAL normal load: velocity-impulse channel (lambdaAcc,
          // physical impulse = lambdaAcc * |Nsum|) + the position-projection channel carried
          // from last substep (already physical impulse units).
          float nTot = Kokkos::fmax(lambdaAcc(idx), 0.0f) * lenN;
          // The carry is a QUASI-STATIC corrector: for colliding contacts the one-substep lag
          // double-counts (crater contacts already carry a large velocity impulse) -- gate by
          // the same event classification as e and beta.
          if (posImpulse.extent(0) > 0 && Kokkos::fabs(vn0(idx)) < restVelThreshold * lenN)
            nTot += Kokkos::fmax(posImpulse(idx), 0.0f);
          const float bound = mu * nTot;
          const float ltLen = Kokkos::sqrt(dot3(ltNew, ltNew));
          if (ltLen > bound)
            ltNew = (bound > 0.0f) ? scale3(ltNew, bound / ltLen) : F3{0, 0, 0};
          const F3 dApp = sub3(ltNew, ltOld);
          if (dApp.x != 0.0f || dApp.y != 0.0f || dApp.z != 0.0f) {
            lambdaT(idx, 0) = ltNew.x;
            lambdaT(idx, 1) = ltNew.y;
            lambdaT(idx, 2) = ltNew.z;
            const float dLen = Kokkos::sqrt(dot3(dApp, dApp));
            if (wT > 1e-9f) {
              Kokkos::atomic_max(&maxApproach(), dLen * wT);
              if (Kokkos::fabs(vn0(idx)) <= 4.0f * restVelThreshold * lenN)
                Kokkos::atomic_max(&maxApproachQS(), dLen * wT);
            }
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
              const F3 Jl = invRotateVector(qB, cross3v(rBavg, scale3(dApp, -1.0f)));
              const F3 dwl{Jl.x * invIB.x, Jl.y * invIB.y, Jl.z * invIB.z};
              const F3 dww = rotateVector(qB, dwl);
              angVelPred(realB, 0) += dww.x;
              angVelPred(realB, 1) += dww.y;
              angVelPred(realB, 2) += dww.z;
            }
          }
        }
      }
    }
  }
};

/// Build the shared per-manifold sweep functor (the colored launch loop, the fused kernels and
/// the fused multilevel iteration loop all run this one body). maxApproachQS empty => aliased
/// to maxApproach (the duplicate atomic_max is idempotent).
inline PGSManifoldSweep makePGSManifoldSweep(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, Kokkos::View<const float*, CpMem> invMass,
    Kokkos::View<const float* [3], CpMem> invInertia, Kokkos::View<const float* [4], CpMem> quat,
    Kokkos::View<float* [3], CpMem> velPred, Kokkos::View<float* [3], CpMem> angVelPred,
    Kokkos::View<const int*, CpMem> realIdx, float growthRate, float restitutionNormal,
    float restVelThreshold, Kokkos::View<float, CpMem> maxApproach,
    Kokkos::View<float, CpMem> maxApproachQS, Kokkos::View<float*, CpMem> lambdaAcc,
    Kokkos::View<const float*, CpMem> vn0, Kokkos::View<const unsigned char*, CpMem> sideFlag,
    Kokkos::View<float* [3], CpMem> lambdaT, float frictionDynamic,
    Kokkos::View<const float* [3], CpMem> vt0, float restitutionTangent,
    Kokkos::View<const float*, CpMem> posImpulse, Kokkos::View<float*, CpMem> restBank,
    Kokkos::View<float*, CpMem> restRel, Kokkos::View<const float*, CpMem> restVPeak,
    Kokkos::View<float*, CpMem> restOrphan, Kokkos::View<const float*, CpMem> restOrphanVPeak,
    SlotOverride ov = {}, int restTarget = 0) {
  return PGSManifoldSweep{manifolds,
                          invMass,
                          invInertia,
                          quat,
                          velPred,
                          angVelPred,
                          realIdx,
                          growthRate,
                          restitutionNormal,
                          restVelThreshold,
                          maxApproach,
                          maxApproachQS.data() ? maxApproachQS : maxApproach,
                          lambdaAcc,
                          vn0,
                          sideFlag,
                          lambdaT,
                          frictionDynamic,
                          vt0,
                          restitutionTangent,
                          posImpulse,
                          restBank,
                          restRel,
                          restVPeak,
                          restOrphan,
                          restOrphanVPeak,
                          ov,
                          restTarget};
}

/// Returns true when the sweep (or, with `loop`, the whole iteration loop) was submitted;
/// false ONLY in loop mode when the fused path is unavailable — the caller then runs its own
/// host-side iteration loop (which calls back in per-iteration mode).
inline bool solveVelocityPGSKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const int*, CpMem> mColor, int numColors,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<const float* [3], CpMem> invInertia,
    Kokkos::View<const float* [4], CpMem> quat, Kokkos::View<float* [3], CpMem> velPred,
    Kokkos::View<float* [3], CpMem> angVelPred, Kokkos::View<const int*, CpMem> realIdx,
    float growthRate, float restitutionNormal, float restVelThreshold,
    Kokkos::View<float, CpMem> maxApproach, Kokkos::View<float*, CpMem> lambdaAcc,
    Kokkos::View<const float*, CpMem> vn0, Kokkos::View<const unsigned char*, CpMem> sideFlag,
    Kokkos::View<float* [3], CpMem> lambdaT, float frictionDynamic,
    Kokkos::View<const float* [3], CpMem> vt0 = {}, float restitutionTangent = 0.0f,
    Kokkos::View<const float*, CpMem> posImpulse = {},
    Kokkos::View<float, CpMem> maxApproachQS = {}, Kokkos::View<float*, CpMem> restBank = {},
    Kokkos::View<float*, CpMem> restRel = {}, Kokkos::View<const float*, CpMem> restVPeak = {},
    Kokkos::View<float*, CpMem> restOrphan = {},
    Kokkos::View<const float*, CpMem> restOrphanVPeak = {},
    Kokkos::View<const int*, CpMem> colorPerm = {}, const std::vector<int>* colorOffs = nullptr,
    const FusedSweepCtx* fused = nullptr, const FusedLoopSpec* loop = nullptr, SlotOverride ov = {},
    int restTarget = 0) {
  CpExec space;
  const PGSManifoldSweep f = makePGSManifoldSweep(
      manifolds, invMass, invInertia, quat, velPred, angVelPred, realIdx, growthRate,
      restitutionNormal, restVelThreshold, maxApproach, maxApproachQS, lambdaAcc, vn0, sideFlag,
      lambdaT, frictionDynamic, vt0, restitutionTangent, posImpulse, restBank, restRel, restVPeak,
      restOrphan, restOrphanVPeak, ov, restTarget);
  // Fused mode (CUDA): one persistent kernel iterates the colours device-side with a grid
  // barrier between them — same per-manifold math, same colour ordering, bit-identical to the
  // launch loop below (see solver_fused.hpp). Loop mode additionally iterates the whole
  // adaptive loop on-device against the residual the caller's stop would have read.
#ifdef KOKKOS_ENABLE_CUDA
  if (loop) {
    if (fused && fused->maxBucket > 0 && colorOffs)
      return demLaunchFusedSweepLoop(space, f, colorPerm, *fused, numColors, *loop,
                                     (maxApproachQS.data() ? maxApproachQS : maxApproach).data());
    return false;
  }
  if (fused && fused->maxBucket > 0 && colorOffs &&
      demLaunchFusedColorSweep(space, f, colorPerm, *fused, numColors))
    return true;
#else
  (void)fused;
  if (loop)
    return false;
#endif
  // Dense-bucket mode (colorOffs from buildColorBucketsKokkos): each colour launch covers only
  // its own manifolds instead of scanning all of them — bit-identical (colour classes are
  // body-disjoint). No fence: the caller's residual readback synchronizes, and an explicit fence
  // here would serialize host submission with GPU execution (the step is submission-bound).
  for (int color = 0; color < numColors; ++color) {
    if (colorOffs) {
      const int b = (*colorOffs)[color], e = (*colorOffs)[color + 1];
      if (b == e)
        continue;
      Kokkos::parallel_for(
          "peclet::dem::solve_velocity_pgs", Kokkos::RangePolicy<CpExec>(space, b, e),
          KOKKOS_LAMBDA(int i2) { f.solveOne(colorPerm(i2)); });
    } else {
      Kokkos::parallel_for(
          "peclet::dem::solve_velocity_pgs", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
          KOKKOS_LAMBDA(int idx) {
            if (mColor(idx) == color)
              f.solveOne(idx);
          });
    }
  }
  return true;
}

/// Bucket the active coloured manifolds by (support level, colour) for the level-ordered
/// ("multilevel") stabilization sweeps. A manifold's level is the smaller of its bodies'
/// height-from-floor BFS levels (wall/plane manifolds sit at their body's level, i.e. 0),
/// clamped to 1023; ungrounded manifolds land in the last buckets. key = level*64 + colour;
/// key-sorting a permutation groups each (level, colour) bucket contiguously, and the host
/// bucket list (begin, end into the permutation) drives the ordered sweeps. Inactive and
/// uncoloured manifolds (colour < 0: periodic dups, empty) are keyed out entirely. (Complete
/// colouring leaves no active manifold uncoloured: docs/contact_solve_framework.md §4.2.)
inline void buildLevelColorBucketsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                         int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                         Kokkos::View<const int*, CpMem> mColor,
                                         Kokkos::View<const int*, CpMem> heights,
                                         Kokkos::View<int*, CpMem> keys,
                                         Kokkos::View<int*, CpMem> perm,
                                         std::vector<std::pair<int, int>>& buckets) {
  buckets.clear();
  if (numManifolds <= 0)
    return;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::level_keys", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        perm(idx) = idx;
        const int c = mColor(idx);
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0 || c < 0) {
          keys(idx) = INT_MAX;
          return;
        }
        int h = heights(realIdx(m.bodyA));
        if (m.bodyB >= 0) {
          const int hB = heights(realIdx(m.bodyB));
          if (hB < h)
            h = hB;
        }
        if (h > 1023)
          h = 1023;
        keys(idx) = h * 64 + c;
      });
  const auto rng = Kokkos::pair<int, int>(0, numManifolds);
  auto kd = Kokkos::subview(keys, rng);
  auto pd = Kokkos::subview(perm, rng);
  Kokkos::Experimental::sort_by_key(space, kd, pd);
  auto hk = Kokkos::create_mirror_view(kd);
  Kokkos::deep_copy(space, hk, kd);
  for (int b = 0; b < numManifolds && hk(b) != INT_MAX;) {
    int e = b + 1;
    while (e < numManifolds && hk(e) == hk(b))
      ++e;
    buckets.emplace_back(b, e);
    b = e;
  }
}

/// Level-ordered symmetric sweep: launch one PGS kernel per (level, colour) bucket, ascending
/// (bottom-up: a chain's load drains towards the floor in ~one pass) or descending (top-down:
/// the return pass carries rebound waves back up). Same impulse math and lambda >= 0 projection
/// as the colored sweep (shared PGSManifoldSweep) -- fully symmetric, exact momentum
/// conservation; the ordering only accelerates transport. Launches are stream-ordered, so the
/// Gauss-Seidel dependency between buckets needs no per-bucket fence.
inline void solveVelocityPGSBucketsKokkos(const PGSManifoldSweep& f,
                                          Kokkos::View<const int*, CpMem> perm,
                                          const std::vector<std::pair<int, int>>& buckets,
                                          bool topDown) {
  CpExec space;
  const int nb = static_cast<int>(buckets.size());
  for (int i = 0; i < nb; ++i) {
    const auto [b, e] = buckets[topDown ? nb - 1 - i : i];
    Kokkos::parallel_for(
        "peclet::dem::solve_velocity_pgs_lvl", Kokkos::RangePolicy<CpExec>(space, b, e),
        KOKKOS_LAMBDA(int i2) { f.solveOne(perm(i2)); });
  }
  space.fence();
}

/// Rank-level X (docs/contact_solve_framework.md §1.4): the holder of a body whose velocity
/// activity mask has several bits (ranks) in sync interval t -- the colour t mod C if it is in the
/// mask, else the (t mod popcount)-th set bit (lowest = 0th). Every rank evaluates the same mask
/// and t, so all ranks agree on the holder without communication.
KOKKOS_INLINE_FUNCTION int xPopcount(unsigned long long m) {
  int n = 0;
  while (m != 0ull) {
    m &= m - 1ull;
    ++n;
  }
  return n;
}
KOKKOS_INLINE_FUNCTION int xHolder(unsigned long long mask, long long t, int numColors) {
  const int tc = static_cast<int>(t % numColors);
  if ((mask >> tc) & 1ull)
    return tc;
  int j = static_cast<int>(t % xPopcount(mask));
  for (int b = 0; b < 64; ++b)
    if ((mask >> b) & 1ull) {
      if (j == 0)
        return b;
      --j;
    }
  return -1;  // unreachable: popcount(mask) >= 2 at every call
}
/// Rank-level X evaluated inline in the one-shot sweep (only for the manifolds of the colour being
/// swept): the same test as computeXGateKokkos, without a separate pass over every manifold per
/// iteration. `mask` empty = off (every contact fires).
struct XGateSpec {
  Kokkos::View<const unsigned long long*, CpMem> mask;
  int color = 0, numColors = 1;
  long long t = 0;
  KOKKOS_INLINE_FUNCTION bool fires(int slotA, int slotB) const {
    if (mask.extent(0) == 0)
      return true;
    const unsigned long long mA = mask(slotA);
    if (xPopcount(mA) >= 2 && xHolder(mA, t, numColors) != color)
      return false;
    if (slotB >= 0) {
      const unsigned long long mB = mask(slotB);
      if (xPopcount(mB) >= 2 && xHolder(mB, t, numColors) != color)
        return false;
    }
    return true;
  }
};
/// The per-manifold fire gate of sync interval t: 1 iff this rank (colour `color`) holds every
/// endpoint whose mask names two or more ranks. Masks live at the canonical velocity slot
/// realIdx(body), so every image and hub copy of one body on this rank shares the decision.
inline void computeXGateKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int nm,
                               Kokkos::View<const int*, CpMem> realIdx,
                               Kokkos::View<const unsigned long long*, CpMem> velMask,
                               Kokkos::View<unsigned char*, CpMem> gate, int color, int numColors,
                               long long t) {
  Kokkos::parallel_for(
      "peclet::dem::x_gate", Kokkos::RangePolicy<CpExec>(0, nm), KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        bool fire = true;
        const unsigned long long mA = velMask(realIdx(m.bodyA));
        if (xPopcount(mA) >= 2 && xHolder(mA, t, numColors) != color)
          fire = false;
        if (m.bodyB >= 0) {
          const unsigned long long mB = velMask(realIdx(m.bodyB));
          if (xPopcount(mB) >= 2 && xHolder(mB, t, numColors) != color)
            fire = false;
        }
        gate(idx) = fire ? 1 : 0;
      });
}
/// Diagnostics (iterCounters): the number of gated-off manifolds of one sweep.
inline long long countUnfiredKokkos(Kokkos::View<const unsigned char*, CpMem> gate, int nm) {
  long long n = 0;
  Kokkos::parallel_reduce(
      "peclet::dem::x_unfired", Kokkos::RangePolicy<CpExec>(0, nm),
      KOKKOS_LAMBDA(int idx, long long& acc) { acc += gate(idx) == 0 ? 1 : 0; }, n);
  return n;
}

/// Colored Gauss–Seidel normal-restitution solve: sweep the `numColors` colour classes in order,
/// applying each manifold's impulse directly to velPred/angVelPred (in place). Same per-manifold
/// impulse math as solveVelocityKokkos (growth-velocity term, approach gate, resting-contact e=0
/// threshold, and the R·(invI_local·(Rᵀ J))·… world-space angular update) — only the write-back
/// differs (in-place RMW instead of atomic-accumulate + count-average). Race-free because a colour
/// is an independent set of manifolds. One outer call = one full sweep over all colours; the caller
/// loops it velocityIterations times.
inline void solveVelocityColoredGSKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const int*, CpMem> mColor, int numColors,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<const float* [3], CpMem> invInertia,
    Kokkos::View<const float* [4], CpMem> quat, Kokkos::View<float* [3], CpMem> velPred,
    Kokkos::View<float* [3], CpMem> angVelPred, Kokkos::View<const int*, CpMem> realIdx,
    float growthRate, float restitutionNormal, float restVelThreshold,
    Kokkos::View<float, CpMem> maxApproach,
    Kokkos::View<const unsigned char*, CpMem> persistent = {},
    Kokkos::View<const float* [3], CpMem> posPred = {}, F3 gHat = {},
    Kokkos::View<const unsigned char*, CpMem> grounded = {}, SlotOverride ov = {},
    XGateSpec xg = {}) {
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
          // hub-copy slot overrides (§4.4; empty = realIdx): every state access uses realA/B
          const int realA = ov.slotA(idx, realIdx(idA));
          const int realB = (idB >= 0) ? ov.slotB(idx, realIdx(idB)) : idB;

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

          // Record this approaching pair's physical approach speed for the adaptive stop: the
          // caller ends the velocity loop once no manifold approaches faster than the resting
          // threshold.
          Kokkos::atomic_max(&maxApproach(), Kokkos::fabs(vn) / lenN);
          // Rank-level X (docs/contact_solve_framework.md §1.4): a gated contact whose rank does
          // not hold one of its rank-split bodies in this sync interval records its approach (the
          // stop must not end while it still approaches) but writes nothing.
          if (!xg.fires(realIdx(idA), idB >= 0 ? realIdx(idB) : -1))
            return;

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
    // dependency). A per-colour fence would only stall the host. One fence after the sweep
    // suffices.
  }
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_VELOCITY_HPP
