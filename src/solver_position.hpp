/// @file
/// @brief dem — portable (Kokkos) XPBD position solve (pure overlap removal).
///
/// Kokkos port of solve_position_jacobi_kernel (solver_position.cu): one thread per contact
/// evaluates the linearized non-penetration constraint C(x) (using the delta-rotated lever arms /
/// normal), computes the XPBD position correction, and atomically scatters delta_pos / delta_quat
/// and bumps the per-body constraint count (the caller averages by it). Faithful copy of the CUDA
/// math over the SoA Views. Friction is a separate cluster (ported separately); the position solve
/// does overlap only.
#ifndef DEM_SOLVER_POSITION_HPP
#define DEM_SOLVER_POSITION_HPP

#include <Kokkos_Core.hpp>
#include <vector>

#include "contact_preprocessing.hpp"  // ContactC, CpExec/CpMem
#include "dem_portable.hpp"
#include "solver_fused.hpp"

namespace peclet::dem {

namespace detail {
KOKKOS_INLINE_FUNCTION float computeW(F3 r, F3 dir, float invM, F3 invI) {
  const F3 rn = cross3v(r, dir);
  return invM + rn.x * rn.x * invI.x + rn.y * rn.y * invI.y + rn.z * rn.z * invI.z;
}
}  // namespace detail

/// Mass-split Jacobi count pass (docs/contact_solve_framework.md §3.1, D3): per body slot, the
/// number of contacts the Jacobi position solve visits (every contact; a wall side counts
/// nothing). Accumulates into `counts`, handed over zeroed (the apply clears it). Under MPI the
/// caller then makes the counts global (syncContactCounts).
inline void countPositionJacobiKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                      int numContacts, Kokkos::View<int*, CpMem> counts) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::count_position_jacobi", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        const ContactC c = contacts(idx);
        Kokkos::atomic_add(&counts(c.bodyA), 1);
        if (c.bodyB >= 0)
          Kokkos::atomic_add(&counts(c.bodyB), 1);
      });
  space.fence();
}

/// Accumulate XPBD position corrections for `numContacts` contacts.
/// massSplit (the 'jacobi' diagnostic, §3.1): `constraintCounts` holds the global per-body counts
/// n of countPositionJacobiKokkos and is only read; each contact's correction is solved against
/// copies of mass m / n (w~ = n_A w_A + n_B w_B, a wall side 0) and the TRUE-mass deltas are
/// accumulated, to be applied with factor 1. Off: the solve bumps the counts and the caller
/// averages the sum by 1/count -- the pre-framework count-averaged Jacobi, kept ONLY for the kernel
/// unit test against its serial reference; no production path calls it (the colour-saturation
/// fallbacks were deleted with complete colouring, docs/contact_solve_framework.md §4.2 / §12 S3).
inline void solvePositionKokkos(
    Kokkos::View<const ContactC*, CpMem> contacts, int numContacts,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<const float* [3], CpMem> posPred,
    Kokkos::View<const float* [4], CpMem> quatPred,
    Kokkos::View<const float* [4], CpMem> quatStatic,
    Kokkos::View<const float* [3], CpMem> invInertia, Kokkos::View<float* [3], CpMem> deltaPos,
    Kokkos::View<float* [4], CpMem> deltaQuat, Kokkos::View<int*, CpMem> constraintCounts,
    Kokkos::View<float, CpMem> maxOverlap, Kokkos::View<const int*, CpMem> onlyColor = {},
    int colorFilter = 0, bool massSplit = false) {
  (void)invInertia;  // translation-only diagonal (below); kept in the signature for the callers
  (void)deltaQuat;   // never written: the rotation was never applied (applyUpdatesKokkos)
  CpExec space;
  const bool filt = onlyColor.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::solve_position", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        if (filt && onlyColor(idx) != colorFilter)
          return;  // colour filter (kernel unit test only; no production caller)
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
        if (C >= 0.0f)
          return;

        // The translational effective mass of the solve views: the correction is translation
        // only, so the diagonal carries no rotational term (docs/contact_physics_followups.md
        // §3.3, WO-B1).
        const float wA = invMassA;
        const float wB = invMassB;
        const float wTotal = wA + wB;
        if (wTotal < 1e-6f)
          return;
        // Mass-split Jacobi: the same numerator against copies of mass m / n.
        const float wSolve =
            massSplit ? static_cast<float>(constraintCounts(idA)) * wA +
                            ((idB >= 0) ? static_cast<float>(constraintCounts(idB)) * wB : 0.0f)
                      : wTotal;

        const float dLambda = -C / wSolve;

        // Translation-only correction (applyUpdatesKokkos commits deltaPos only).
        Kokkos::atomic_add(&deltaPos(idA, 0), n.x * dLambda * invMassA);
        Kokkos::atomic_add(&deltaPos(idA, 1), n.y * dLambda * invMassA);
        Kokkos::atomic_add(&deltaPos(idA, 2), n.z * dLambda * invMassA);
        if (idB >= 0) {
          Kokkos::atomic_add(&deltaPos(idB, 0), -n.x * dLambda * invMassB);
          Kokkos::atomic_add(&deltaPos(idB, 1), -n.y * dLambda * invMassB);
          Kokkos::atomic_add(&deltaPos(idB, 2), -n.z * dLambda * invMassB);
          if (!massSplit)
            Kokkos::atomic_add(&constraintCounts(idB), 1);
        }
        if (!massSplit)
          Kokkos::atomic_add(&constraintCounts(idA), 1);

        if (C < 0.0f)
          Kokkos::atomic_max(&maxOverlap(), -C);
      });
  space.fence();
}

// ============================ colored Gauss–Seidel position solve ============================
// The overlap solve above is the position-side twin of the Jacobi restitution solve: one thread per
// contact scatters an XPBD non-penetration correction, and the caller relaxes the per-body SUM by
// the contact count (applyUpdatesKokkos, 1/count). Same trade-off — stable but under-converged, so
// a body wedged by many neighbours keeps residual overlap. The colored path removes the averaging:
// graph- colour the CONTACT graph (vertices = body slots, edges = contacts; raw bodyA/bodyB, the
// same indices the solve writes) so no two contacts sharing a body get one colour, then sweep
// colour-by- colour applying each correction IN PLACE. Within a colour the contacts are an
// independent set, so the read-modify-write is race-free without atomics, and each sweep sees the
// previous colours' moves — a true sequential projection that resolves stacked contacts far better
// per iteration. Translation only, matching the Jacobi path (applyUpdatesKokkos applies deltaPos,
// not deltaQuat; the angular contact response lives in the velocity solve), and it never touches
// velocity — overlap removal stays decoupled from the velocity update.

/// A position unit is sleep-inactive iff all its contacts are sleep-masked (§4.3).
KOKKOS_INLINE_FUNCTION bool unitAsleep(const PosUnits& units,
                                       const Kokkos::View<const unsigned char*, CpMem>& sleepMask,
                                       int u) {
  for (int k = units.begin(u); k < units.end(u); ++k)
    if (!sleepMask(units.contact(k)))
      return false;
  return true;
}

/// Greedy graph-colour the contacts (raw bodyA/bodyB): no two contacts sharing a body get the same
/// colour. Round-based max-index arbitration, identical machinery to colorManifoldsKokkos but over
/// the per-contact graph with direct (non-realIdx) body slots; every contact is active. `numBodies`
/// is the body-slot count (numParticles, incl. periodic ghosts, which the position solve corrects
/// independently). Returns the number of colours used.
///
/// With `units` (docs/contact_solve_framework.md §4.3) the edges are position UNITS, not contacts:
/// numContacts is then the unit count and cColor is per unit; a unit's edge is its leader
/// contact's (bodyA, bodyB) with key colorKey(leader). A one-point unit is exactly today's edge.
inline int colorContactsKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int numContacts,
                               int numBodies, Kokkos::View<int*, CpMem> cColor,
                               Kokkos::View<long long*, CpMem> bodyWinner,
                               Kokkos::View<std::uint64_t*, CpMem> bodyMask, int& leftover,
                               Kokkos::View<const unsigned char*, CpMem> sleepMask = {},
                               PosUnits units = {}, SlotOverride ov = {}, int vertexSpan = 0) {
  leftover = 0;
  CpExec space;
  if (numContacts <= 0 || numBodies <= 0)
    return 0;
  // Colouring vertices: the raw contact body slot, or a hub copy slot through `ov` (§4.4);
  // vertexSpan covers the copy slots (0 = numBodies, the no-copy span).
  const int nV = vertexSpan > 0 ? vertexSpan : numBodies;
  const bool sleepOn = sleepMask.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::pcolor_init_bodies", Kokkos::RangePolicy<CpExec>(space, 0, nV),
      KOKKOS_LAMBDA(int i) { bodyMask(i) = 0; });
  Kokkos::parallel_for(
      "peclet::dem::pcolor_init_contacts", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        cColor(idx) = (sleepOn && unitAsleep(units, sleepMask, idx)) ? -2 : -1;
      });

  int remaining = 1, prevRemaining = -1;
  // Round cap (docs/contact_solve_framework.md §12 S7): the vertices coloured are the UNITS
  // (numContacts here), so the bound is their count + 2. Each round commits at least the
  // globally highest-key uncoloured unit (colorKey(leader) is unique: the leader index is its
  // low word), so the loop terminates within numContacts rounds; O(log n) in practice.
  const int maxRounds = numContacts + 2;
  for (int round = 0; round < maxRounds && remaining > 0; ++round) {
    Kokkos::parallel_for(
        "peclet::dem::pcolor_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, nV),
        KOKKOS_LAMBDA(int i) { bodyWinner(i) = -1; });
    Kokkos::parallel_for(
        "peclet::dem::pcolor_contend", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx) {
          if (cColor(idx) != -1)
            return;
          const int lead = units.leader(idx);  // a unit's edge: its leader's bodies and key
          const ContactC c = contacts(lead);
          const long long key = colorKey(lead);  // hashed priority (see solver_velocity.hpp)
          Kokkos::atomic_max(&bodyWinner(ov.slotA(idx, c.bodyA)), key);
          if (c.bodyB >= 0)
            Kokkos::atomic_max(&bodyWinner(ov.slotB(idx, c.bodyB)), key);
        });
    int rem = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::pcolor_commit", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx, int& acc) {
          if (cColor(idx) != -1)
            return;
          const int lead = units.leader(idx);
          const ContactC c = contacts(lead);
          const int ea = ov.slotA(idx, c.bodyA);
          const int eb = (c.bodyB >= 0) ? ov.slotB(idx, c.bodyB) : -1;  // <0: wall/boundary
          const long long key = colorKey(lead);
          if (bodyWinner(ea) != key || (eb >= 0 && bodyWinner(eb) != key)) {
            acc += 1;
            return;
          }
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int col = 0;  // lowest free colour of 64; never forced (§4.2 item 1)
          while (col < kColorPalette && ((forbidden >> col) & 1))
            ++col;
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 3
          if (col > 62)
            col = 62;  // G13 mutant 3: the old forced colour 62
#endif
          if (col == kColorPalette) {
            cColor(idx) = kColorUncolourable;  // no free colour: neither coloured nor remaining
            return;
          }
          cColor(idx) = col;
          const std::uint64_t bit = std::uint64_t(1) << col;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    // Stall break: a safety bound only -- every round commits (or marks -3) the highest-key
    // contender, so rem strictly decreases (§12 S7). Anything left -1 is a leftover.
    if (rem == prevRemaining)
      break;
    prevRemaining = rem;
    remaining = rem;
  }

  int maxc = -1;
  Kokkos::parallel_reduce(
      "peclet::dem::pcolor_max", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx, int& mx) {
        if (cColor(idx) > mx)
          mx = cColor(idx);
      },
      Kokkos::Max<int>(maxc));
  Kokkos::parallel_reduce(
      "peclet::dem::pcolor_leftover", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx, int& acc) {
        if (cColor(idx) == -1 || cColor(idx) == kColorUncolourable)
          acc += 1;
      },
      leftover);
  space.fence();
  return maxc + 1;
}

/// Incremental (warm-started) contact colouring for the single-GPU PGS position solve. Twin of
/// colorManifoldsIncrementalKokkos, but over the per-contact graph: key every contact by its
/// canonical pair, carry the previous substep's colour by that key (sorted ledger), seed the
/// per-body masks, and re-arbitrate only the NEW (-1) contacts. A pair can own MORE than one
/// contact (non-spherical multi-point patches) and periodic ghost slots churn, so the carried
/// colours are CONFLICT-CHECKED while the mask is built (a repeated colour on a shared body forces
/// a full recolour that substep) — the sphere / non-periodic bed hits the fast path, everything
/// else stays correct by falling back. `forceFull` bypasses the carry. keysOut holds this substep's
/// contact keys for the subsequent commit. Returns the number of colours used.
inline int colorContactsIncrementalKokkos(
    Kokkos::View<const ContactC*, CpMem> contacts, int numContacts, int numBodies,
    Kokkos::View<const unsigned long long*, CpMem> prevKeys,
    Kokkos::View<const int*, CpMem> prevColor, int prevCount, Kokkos::View<int*, CpMem> cColor,
    Kokkos::View<unsigned long long*, CpMem> keysOut, Kokkos::View<long long*, CpMem> bodyWinner,
    Kokkos::View<std::uint64_t*, CpMem> bodyMask, int& leftover, bool forceFull,
    Kokkos::View<const unsigned char*, CpMem> sleepMask = {}, PosUnits units = {}) {
  leftover = 0;
  CpExec space;
  if (numContacts <= 0 || numBodies <= 0)
    return 0;
  const bool full0 = forceFull || prevCount <= 0;
  const bool sleepOn = sleepMask.extent(0) > 0;
  // Key every contact and seed its colour from the carried ledger (or -1 on a full recolour, or
  // -2 for a frozen both-asleep contact — excluded from the sweeps like an inactive one).
  Kokkos::parallel_for(
      "peclet::dem::pcolor_i_seed", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx) {
        // Every contact of the unit records its own pair key for the per-contact commit; the unit
        // is seeded from its leader's key (identical to the per-contact seed for a one-point unit).
        for (int k2 = units.begin(idx); k2 < units.end(idx); ++k2) {
          const int ci = units.contact(k2);
          keysOut(ci) = pairKey(contacts(ci));
        }
        const unsigned long long k = keysOut(units.leader(idx));
        if (sleepOn && unitAsleep(units, sleepMask, idx)) {
          cColor(idx) = -2;
          return;
        }
        int col = -1;
        if (!full0) {
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
              col = pc;
          }
        }
        cColor(idx) = col;
      });
  Kokkos::parallel_for(
      "peclet::dem::pcolor_i_init_bodies", Kokkos::RangePolicy<CpExec>(space, 0, numBodies),
      KOKKOS_LAMBDA(int i) { bodyMask(i) = 0; });
  if (!full0) {
    // Seed the per-body masks from carried colours, self-healing any conflict WITHOUT a host sync
    // (a host readback here stalls the async submission pipeline and measured net-slower than the
    // launches it saves). atomic_fetch_or serialises the claim: a contact that finds its colour bit
    // already set at either endpoint DEMOTES itself to -1 and is re-arbitrated in the rounds below.
    // A demoted contact may leave a spurious set bit at its other endpoint, but a spurious bit only
    // over-constrains (forbids one colour there) — it never lets two same-colour contacts share a
    // body, so the colouring stays valid. The sphere / non-periodic bed has no conflicts (one
    // contact per pair, stable slots) and hits the pure carry path.
    Kokkos::parallel_for(
        "peclet::dem::pcolor_i_mask", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx) {
          const int col = cColor(idx);
          if (col < 0)
            return;
          const ContactC c = contacts(units.leader(idx));
          const std::uint64_t bit = std::uint64_t(1) << col;
          bool conflict = ((Kokkos::atomic_fetch_or(&bodyMask(c.bodyA), bit) >> col) & 1) != 0;
          if (c.bodyB >= 0)
            conflict |= ((Kokkos::atomic_fetch_or(&bodyMask(c.bodyB), bit) >> col) & 1) != 0;
          if (conflict)
            cColor(idx) = -1;  // re-arbitrate in the rounds
        });
  }
  // Jones-Plassmann rounds over the uncoloured (-1) contacts only (identical to
  // colorContactsKokkos).
  int remaining = 1, prevRemaining = -1;
  // Round cap (docs/contact_solve_framework.md §12 S7): the vertices coloured are the UNITS
  // (numContacts here), so the bound is their count + 2. Each round commits at least the
  // globally highest-key uncoloured unit (colorKey(leader) is unique: the leader index is its
  // low word), so the loop terminates within numContacts rounds; O(log n) in practice.
  const int maxRounds = numContacts + 2;
  for (int round = 0; round < maxRounds && remaining > 0; ++round) {
    Kokkos::parallel_for(
        "peclet::dem::pcolor_i_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, numBodies),
        KOKKOS_LAMBDA(int i) { bodyWinner(i) = -1; });
    Kokkos::parallel_for(
        "peclet::dem::pcolor_i_contend", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx) {
          if (cColor(idx) != -1)
            return;
          const int lead = units.leader(idx);  // a unit's edge: its leader's bodies and key
          const ContactC c = contacts(lead);
          const long long key = colorKey(lead);
          Kokkos::atomic_max(&bodyWinner(c.bodyA), key);
          if (c.bodyB >= 0)
            Kokkos::atomic_max(&bodyWinner(c.bodyB), key);
        });
    int rem = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::pcolor_i_commit", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
        KOKKOS_LAMBDA(int idx, int& acc) {
          if (cColor(idx) != -1)
            return;
          const int lead = units.leader(idx);
          const ContactC c = contacts(lead);
          const int ea = c.bodyA;
          const int eb = c.bodyB;
          const long long key = colorKey(lead);
          if (bodyWinner(ea) != key || (eb >= 0 && bodyWinner(eb) != key)) {
            acc += 1;
            return;
          }
          std::uint64_t forbidden = bodyMask(ea);
          if (eb >= 0)
            forbidden |= bodyMask(eb);
          int col = 0;  // lowest free colour of 64; never forced (§4.2 item 1)
          while (col < kColorPalette && ((forbidden >> col) & 1))
            ++col;
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 3
          if (col > 62)
            col = 62;  // G13 mutant 3: the old forced colour 62
#endif
          if (col == kColorPalette) {
            cColor(idx) = kColorUncolourable;
            return;
          }
          cColor(idx) = col;
          const std::uint64_t bit = std::uint64_t(1) << col;
          bodyMask(ea) |= bit;
          if (eb >= 0)
            bodyMask(eb) |= bit;
        },
        rem);
    space.fence();
    if (rem == prevRemaining)
      break;
    prevRemaining = rem;
    remaining = rem;
  }
  int maxc = -1;
  Kokkos::parallel_reduce(
      "peclet::dem::pcolor_i_max", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx, int& mx) {
        if (cColor(idx) > mx)
          mx = cColor(idx);
      },
      Kokkos::Max<int>(maxc));
  Kokkos::parallel_reduce(
      "peclet::dem::pcolor_i_leftover", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int idx, int& acc) {
        if (cColor(idx) == -1 || cColor(idx) == kColorUncolourable)
          acc += 1;
      },
      leftover);
  space.fence();
  return maxc + 1;
}

/// Copy each position unit's colour onto its contacts (contactColor(i) = unitColor(unit of i)).
inline void expandUnitColorsKokkos(const PosUnits& units, int numUnits,
                                   Kokkos::View<const int*, CpMem> unitColor,
                                   Kokkos::View<int*, CpMem> contactColor) {
  if (numUnits <= 0)
    return;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::pcolor_expand", Kokkos::RangePolicy<CpExec>(space, 0, numUnits),
      KOKKOS_LAMBDA(int u) {
        const int col = unitColor(u);
        for (int k = units.begin(u); k < units.end(u); ++k)
          contactColor(units.contact(k)) = col;
      });
}

/// Commit this substep's per-contact (key, colour) sorted by key, for next substep's warm gather.
inline void commitContactColorKokkos(Kokkos::View<const unsigned long long*, CpMem> keys,
                                     Kokkos::View<const int*, CpMem> color,
                                     Kokkos::View<unsigned long long*, CpMem> prevKeys,
                                     Kokkos::View<int*, CpMem> prevColor,
                                     Kokkos::View<int*, CpMem> perm, int numContacts) {
  if (numContacts <= 0)
    return;
  CpExec space;
  const int n = numContacts;
  const auto rng = Kokkos::pair<int, int>(0, n);
  auto kd = Kokkos::subview(prevKeys, rng);
  Kokkos::deep_copy(space, kd, Kokkos::subview(keys, rng));
  Kokkos::parallel_for(
      "peclet::dem::pcolor_commit_iota", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { perm(i) = i; });
  {
    auto pd = Kokkos::subview(perm, rng);
    Kokkos::Experimental::sort_by_key(space, kd, pd);
  }
  Kokkos::View<int*, CpMem> pc = prevColor;
  Kokkos::View<const int*, CpMem> c = color;
  Kokkos::parallel_for(
      "peclet::dem::pcolor_commit_gather", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { pc(i) = c(perm(i)); });
  space.fence();
}

/// The per-contact overlap-projection body lives in PositionContactSweep so the colored launch
/// loop and the fused colour sweep (solver_fused.hpp) share it verbatim: solveOne(idx) must only
/// run concurrently on contacts that are body-disjoint within one launch (a colour class).
struct PositionContactSweep {
  Kokkos::View<const ContactC*, CpMem> contacts;
  Kokkos::View<const float*, CpMem> invMass;
  Kokkos::View<float* [3], CpMem> posPred;
  Kokkos::View<const float* [4], CpMem> quatPred;
  Kokkos::View<const float* [4], CpMem> quatStatic;
  Kokkos::View<const float* [3], CpMem> invInertia;
  Kokkos::View<float, CpMem> maxOverlap;
  Kokkos::View<float*, CpMem> posLambdaAcc;

  PosUnits units;  // empty = one contact per work item (the identity)
  // Hub-copy slot overrides per UNIT (docs/contact_solve_framework.md §4.4); empty = the raw
  // contact bodies.
  SlotOverride ov{};
  // Accumulated projection (WO-12, §13.5; USER 2026-09-26): with a stop residual view and the
  // per-contact ledger posLambdaAcc, each contact's net push Lambda >= 0 is projected,
  // Lambda' = max(0, Lambda - omega C / w), and the CHANGE d = Lambda' - Lambda is applied -- it
  // may be negative (an overshoot is retracted), so over-relaxation omega in (0, 2) is legitimate
  // and the fixed point is the unique least-displacement solution. posResidual collects max |d| w
  // (the position change), the stop quantity; maxOverlap keeps "the largest violation seen". Empty
  // posResidual = the pre-WO-12 incremental projection at omega 1 (kernel unit tests only).
  Kokkos::View<float, CpMem> posResidual{};
  float omega = 1.0f;

  /// One work item: every contact of position unit u, sequentially in ascending contact index
  /// (a pair's points are a serial Gauss-Seidel sweep inside one item; §4.3). A one-point unit is
  /// today's per-contact arithmetic on today's contact.
  KOKKOS_FUNCTION void solveOne(int u) const {
    for (int k = units.begin(u); k < units.end(u); ++k)
      solveContact(units.contact(k), u);
  }

  KOKKOS_FUNCTION void solveContact(int idx, int u) const {
    const ContactC c = contacts(idx);
    int idA = c.bodyA, idB = c.bodyB;
    if (ov.a.extent(0) > 0) {
      // The unit's overrides are keyed by its leader's (bodyA, bodyB); a contact of the unit may
      // list the pair in either order, so map each end by body.
      const ContactC L = contacts(units.leader(u));
      idA = (c.bodyA == L.bodyA) ? ov.slotA(u, c.bodyA) : ov.slotB(u, c.bodyA);
      if (c.bodyB >= 0)
        idB = (c.bodyB == L.bodyA) ? ov.slotA(u, c.bodyB) : ov.slotB(u, c.bodyB);
    }
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
      n = F3{c.normal.x, c.normal.y, c.normal.z};
      const F3 pAsurf = add3(pA, rA);
      C = dot3(sub3(pAsurf, F3{c.rB.x, c.rB.y, c.rB.z}), n);
    } else {
      const F3 pAc = add3(pA, rA);
      const F3 pBc = add3(pB, rB);
      C = dot3(sub3(pAc, pBc), n);
    }
    const bool accumulated = posResidual.extent(0) > 0 && posLambdaAcc.extent(0) > 0;
    const float lam = accumulated ? posLambdaAcc(idx) : 0.0f;
    if (C >= 0.0f && lam <= 0.0f)
      return;  // separated and never pushed: nothing to project or retract

    // The translational effective mass of the solve views (invMass* = k invM under mass
    // splitting; a wall side 0): the correction below is translation only, so the diagonal has
    // no rotational term, and |dLambda| wTotal is the true relative position change
    // (docs/contact_physics_followups.md §3.3, WO-B1).
    const float wTotal = invMassA + invMassB;
    if (wTotal < 1e-6f)
      return;
    if (C < 0.0f)
      Kokkos::atomic_max(&maxOverlap(), -C);
    float dLambda;
    if (accumulated) {
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 7
      // G13 mutant 7: over-relaxed WITHOUT retraction (the pre-WO-12 defect: a permanent gap).
      const float lamNew = Kokkos::fmax(lam, lam - omega * C / wTotal);
#else
      const float lamNew = Kokkos::fmax(0.0f, lam - omega * C / wTotal);
#endif
      dLambda = lamNew - lam;
      if (dLambda == 0.0f)
        return;
      posLambdaAcc(idx) = lamNew;  // the unit's work item owns the contact: plain RMW
      Kokkos::atomic_max(&posResidual(), Kokkos::fabs(dLambda) * wTotal);
    } else {
      dLambda = -C / wTotal;
      if (posLambdaAcc.extent(0) > 0)
        Kokkos::atomic_add(&posLambdaAcc(idx), dLambda);
    }
    // Position-channel normal load bookkeeping: posLambdaAcc is the contact's NET position
    // impulse (the friction cone must see the TOTAL normal force; the caller converts it to
    // impulse units and carries it into the next substep's Coulomb bound -- else a jostled bed's
    // bound under-counts and stick leaks, measured as 99% sliding wall contacts in the drum).

    // Translation-only correction, in place (rotation discarded to match applyUpdatesKokkos).
    posPred(idA, 0) += n.x * dLambda * invMassA;
    posPred(idA, 1) += n.y * dLambda * invMassA;
    posPred(idA, 2) += n.z * dLambda * invMassA;
    if (idB >= 0) {
      posPred(idB, 0) += -n.x * dLambda * invMassB;
      posPred(idB, 1) += -n.y * dLambda * invMassB;
      posPred(idB, 2) += -n.z * dLambda * invMassB;
    }
  }
};

/// Colored Gauss–Seidel XPBD overlap solve: sweep the `numColors` colour classes in order, applying
/// each contact's non-penetration correction directly to posPred (in place, translation only). Same
/// per-contact math as solvePositionKokkos — only the write-back differs (in-place RMW instead of
/// atomic-accumulate + count-average). Race-free because a colour is an independent set of
/// contacts. One outer call = one full sweep over all colours; the caller loops it
/// positionIterations times. Dense-bucket mode (colorPerm/colorOffs from buildColorBucketsKokkos)
/// covers only each colour's own contacts; the fused mode collapses the whole sweep into one kernel
/// — both bit-identical.
inline bool solvePositionColoredGSKokkos(
    Kokkos::View<const ContactC*, CpMem> contacts, const PosUnits& units, int numContacts,
    Kokkos::View<const int*, CpMem> cColor, int numColors,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<float* [3], CpMem> posPred,
    Kokkos::View<const float* [4], CpMem> quatPred,
    Kokkos::View<const float* [4], CpMem> quatStatic,
    Kokkos::View<const float* [3], CpMem> invInertia, Kokkos::View<float, CpMem> maxOverlap,
    Kokkos::View<float*, CpMem> posLambdaAcc = {}, Kokkos::View<const int*, CpMem> colorPerm = {},
    const std::vector<int>* colorOffs = nullptr, const FusedSweepCtx* fused = nullptr,
    const FusedLoopSpec* loop = nullptr, SlotOverride ov = {},
    Kokkos::View<float, CpMem> posResidual = {}, float omega = 1.0f) {
  CpExec space;
  const PositionContactSweep f{contacts,   invMass,      posPred, quatPred, quatStatic,  invInertia,
                               maxOverlap, posLambdaAcc, units,   ov,       posResidual, omega};
#ifdef KOKKOS_ENABLE_CUDA
  if (loop) {
    // The device loop's stop residual: the position change under the accumulated projection
    // (WO-12), else the overlap.
    if (fused && fused->maxBucket > 0 && colorOffs)
      return demLaunchFusedSweepLoop(
          space, f, colorPerm, *fused, numColors, *loop,
          posResidual.extent(0) > 0 ? posResidual.data() : maxOverlap.data());
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
  for (int color = 0; color < numColors; ++color) {
    if (colorOffs) {
      const int b = (*colorOffs)[color], e = (*colorOffs)[color + 1];
      if (b == e)
        continue;
      Kokkos::parallel_for(
          "peclet::dem::solve_position_gs", Kokkos::RangePolicy<CpExec>(space, b, e),
          KOKKOS_LAMBDA(int i2) { f.solveOne(colorPerm(i2)); });
    } else {
      Kokkos::parallel_for(
          "peclet::dem::solve_position_gs", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
          KOKKOS_LAMBDA(int idx) {
            if (cColor(idx) == color)
              f.solveOne(idx);
          });
    }
    // Stream-ordered on the device, so colour c+1 already sees colour c's moves — no host fence per
    // colour (that would only stall the host). No trailing fence either: the caller's residual
    // readback synchronizes, and a fence here would break CUDA-graph capture of the sweep.
  }
  return true;
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_POSITION_HPP
