/// @file
/// @brief dem — multilevel (GraphMG-style) momentum-conserving contact stabilization.
///
/// The principled replacement for the one-sided grounded pass: a collapsing column cannot be
/// arrested by plain symmetric PGS (momentum moves ~one layer per sweep) and the one-sided pass
/// arrests it by DELETING momentum (measured cost: it kills a ballistic impactor's rebound). The
/// multilevel pass instead accelerates momentum TRANSPORT: greedy pairwise aggregation over the
/// quasi-static contact graph builds super-bodies (summed mass, momentum-weighted velocity), and
/// the fine manifolds crossing aggregate boundaries are re-solved with the AGGREGATE masses --
/// the coarse analogue of the held lower side is the genuinely huge inertia of the supported
/// chain, so a wall contact drains a whole column's momentum in one coarse impulse while every
/// impulse stays symmetric (exact momentum conservation; the floor/walls are the only sink).
///
/// Structure per trigger (residual above the resting threshold after the main sweeps):
///   build: L levels of matching (ballistic pairs |vn0| > qsThr and walls never merge; stop when
///          matching stalls or the group count is small), per-level group masses, and a per-level
///          graph coloring of the crossing manifolds (6 bits/level packed into one word per
///          manifold; the coloring guarantees group-disjointness within a launch).
///   cycle: per iteration, one fine colored-PGS smoothing sweep, then levels fine -> coarse:
///          restrict (V_g = sum m v / sum m -- momentum-conserving), a few colored coarse PGS
///          sweeps (translation-only, e = 0, lambda >= 0 on the SHARED fine accumulator so the
///          force-network ledger stays consistent for warm start + the friction cone's Coulomb
///          bound), prolongate (v_i += dV_g: uniform per aggregate = mass-proportional impulse).
///
/// The inequality constraint is respected on every level (Kornhuber-style truncation reduces to:
/// only compressed/quasi-static contacts aggregate or restrict; lambda >= 0 projection is the
/// same accumulator projection as the fine sweep). Friction stays fine-level only.
#ifndef DEM_SOLVER_MULTILEVEL_HPP
#define DEM_SOLVER_MULTILEVEL_HPP

#include <Kokkos_Core.hpp>
#include <utility>
#include <vector>

#include "contact_preprocessing.hpp"  // ManifoldC, colorKey, CpExec/CpMem
#include "solver_fused.hpp"           // demGridBarrier / FusedSweepCtx (fused coarse cycle)
#include "solver_velocity.hpp"        // PGSManifoldSweep

namespace peclet::dem {

/// 6-bit colour slots per level in the packed word; slot value 63 = not crossing / not eligible
/// at that level (skipped by the coarse sweeps).
inline constexpr int kMlMaxLevels = 10;
inline constexpr int kMlSlotSkip = 63;

/// Host-side description of one built hierarchy (offsets into the packed group pools).
struct ContactHierarchy {
  int numLevels = 0;           // coarse levels built (0 = aggregation found nothing)
  std::vector<int> groupOff;   // per level: offset of its group arrays in the group pools
  std::vector<int> numGroups;  // per level: group count
  std::vector<int> parentOff;  // per level: offset of its parent map (from level l-1 ids)
  std::vector<int> numColors;  // per level: colours used by the crossing-manifold coloring
};

/// Device scratch for the multilevel pass, sized once (see Particles::allocate).
struct MlScratch {
  Kokkos::View<long long*, CpMem> colorPacked;  // per manifold: 6-bit colour slot per level
  Kokkos::View<int*, CpMem> parent;             // packed parent maps (pool, 5*cap)
  Kokkos::View<float*, CpMem> invMassG;         // packed per-group inverse mass (pool, 4*cap)
  Kokkos::View<float* [3], CpMem> velG;         // packed per-group velocity (pool, 4*cap)
  Kokkos::View<float* [3], CpMem> velG0;        // restriction snapshot (pool, 4*cap)
  Kokkos::View<float*, CpMem> massG;            // packed per-group mass accumulator (pool, 4*cap)
  Kokkos::View<int*, CpMem> grp;                // composed REAL-body -> current-level group map
  Kokkos::View<int*, CpMem> mate;               // matching scratch (per group of the finer level)
};

namespace mldetail {
/// Effective mass with fixed bodies (invMass == 0) mapped to a huge-but-finite mass so the
/// momentum-weighted restriction stays finite and the group's inverse mass underflows to ~0.
KOKKOS_INLINE_FUNCTION float effMass(float invMass) {
  return 1.0f / Kokkos::fmax(invMass, 1e-30f);
}

/// Optional eligibility gates beyond the quasi-static approach test (a SELECTION, not a sink --
/// impulses stay symmetric; gating only decides which contacts join the coarse problem):
///   kGatePersistent -- contact must have existed last substep (a fresh contact is an event).
///   kGateCone       -- support-oriented only (|up| > 0.3|dx|, computeSideFlagsKokkos's cone).
///   kGateSlip       -- exclude sustained shear (|vt0| > qsThr): a discharging silo's bulk is
///                      normal-quasi-static in every direction, and cancelling aggregate-relative
///                      approach there acts as fake bulk viscosity (measured: large-orifice
///                      discharge 16.7 k/s vs 24.5 reference ungated) -- but it SLIPS, while a
///                      crushing bed does not.
inline constexpr int kGatePersistent = 1;
inline constexpr int kGateCone = 2;
inline constexpr int kGateSlip = 4;

/// Coarse-eligibility of a manifold: active + base-coloured (non-dup) + quasi-static approach
/// (|vn0| <= qsThr: a ballistic pair is never aggregated, so an impactor keeps its fine-level
/// momentum-conserving physics), plus the gates selected in gateMask.
KOKKOS_INLINE_FUNCTION bool eligible(const ManifoldC& m, int idx,
                                     Kokkos::View<const int*, CpMem> mColor,
                                     Kokkos::View<const float*, CpMem> vn0,
                                     Kokkos::View<const float* [3], CpMem> vt0,
                                     Kokkos::View<const unsigned char*, CpMem> persistent,
                                     Kokkos::View<const float* [3], CpMem> posPred, F3 gHat,
                                     float qsThr, int gateMask) {
  if (m.num_points <= 0 || mColor(idx) < 0)
    return false;
  if ((gateMask & kGatePersistent) && persistent(idx) == 0)
    return false;
  if (Kokkos::fabs(vn0(idx)) > qsThr)
    return false;  // ballistic pair: never aggregated (rebound stays fine + symmetric)
  if (gateMask & kGateSlip) {
    // Slip threshold = qsThr. Measured trade-off: a pour-settling bed's load-bearing contacts
    // slip at 2-8 g dt (a tighter floor excludes them and the bed crushes), while silo bulk
    // creep slips at <= 8 g dt (so some of it stays in and costs ~7% discharge vs the one-sided
    // default). The two distributions overlap in g dt units -- no absolute threshold separates
    // them; a slip-persistence (time-integrated) gate is the identified next refinement.
    const F3 vt{vt0(idx, 0), vt0(idx, 1), vt0(idx, 2)};
    if (dot3(vt, vt) > qsThr * qsThr)
      return false;
  }
  if (m.bodyB < 0)
    return true;
  if (gateMask & kGateCone) {
    const F3 dx = sub3(ldF3(posPred, m.bodyA), ldF3(posPred, m.bodyB));
    const float up = -(dx.x * gHat.x + dx.y * gHat.y + dx.z * gHat.z);
    return Kokkos::fabs(up) > 0.3f * Kokkos::sqrt(dot3(dx, dx));
  }
  return true;
}
}  // namespace mldetail

/// Build the aggregation hierarchy + per-level crossing-manifold colorings. Eligibility for both
/// matching and coarse solving: mldetail::eligible (active, non-dup, persistent, quasi-static,
/// support-oriented). Walls (bodyB < 0) never merge but their manifolds ARE coarse contacts
/// (the momentum sink). Returns the host hierarchy description.
inline ContactHierarchy buildContactHierarchyKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const int*, CpMem> realIdx, Kokkos::View<const int*, CpMem> mColor,
    Kokkos::View<const float*, CpMem> vn0, Kokkos::View<const float* [3], CpMem> vt0,
    Kokkos::View<const unsigned char*, CpMem> persistent,
    Kokkos::View<const float* [3], CpMem> posPred, F3 gHat,
    Kokkos::View<const float*, CpMem> invMass, float qsThr, int gateMask, int numReal, MlScratch& S,
    Kokkos::View<long long*, CpMem> winner, Kokkos::View<std::uint64_t*, CpMem> colorMask,
    bool excludeImmovable = false) {
  ContactHierarchy H;
  CpExec space;
  if (numManifolds <= 0 || numReal <= 0)
    return H;
  {  // level-0 composed map = identity; packed colours all "skip"
    auto grp = S.grp;
    Kokkos::parallel_for(
        "peclet::dem::ml_grp_init", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
        KOKKOS_LAMBDA(int i) { grp(i) = i; });
    auto cp = S.colorPacked;
    Kokkos::parallel_for(
        "peclet::dem::ml_packed_init", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) { cp(idx) = ~0ll; });  // every 6-bit slot = 63 (skip)
  }

  int ngPrev = numReal;  // group count of the finer level (level 0 = real bodies)
  int parentOff = 0, groupOff = 0;
  const int parentCap = static_cast<int>(S.parent.extent(0));
  const int groupCap = static_cast<int>(S.invMassG.extent(0));

  for (int lvl = 1; lvl <= kMlMaxLevels; ++lvl) {
    if (parentOff + ngPrev > parentCap)
      break;  // pool exhausted (matching stalled repeatedly) -- use what we have
    // ---- greedy random-priority matching on the current group graph ----
    Kokkos::parallel_for(
        "peclet::dem::ml_match_reset", Kokkos::RangePolicy<CpExec>(space, 0, ngPrev),
        KOKKOS_LAMBDA(int g) { winner(g) = -1; });
    {
      auto grp = S.grp;
      Kokkos::parallel_for(
          "peclet::dem::ml_match_contend", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
          KOKKOS_LAMBDA(int idx) {
            const ManifoldC m = manifolds(idx);
            if (m.bodyB < 0 || !mldetail::eligible(m, idx, mColor, vn0, vt0, persistent, posPred,
                                                   gHat, qsThr, gateMask))
              return;
            // Never aggregate an immovable (sleeping / pinned) body: a group carrying its huge
            // effMass wrecks the coarse-solve conditioning and pins the awake partner. (Gated so
            // the sleeping-off path is bit-identical.)
            if (excludeImmovable &&
                (invMass(realIdx(m.bodyA)) == 0.0f || invMass(realIdx(m.bodyB)) == 0.0f))
              return;
            const int gA = grp(realIdx(m.bodyA)), gB = grp(realIdx(m.bodyB));
            if (gA == gB)
              return;
            const long long key = colorKey(idx);
            Kokkos::atomic_max(&winner(gA), key);
            Kokkos::atomic_max(&winner(gB), key);
          });
      auto mate = S.mate;
      Kokkos::parallel_for(
          "peclet::dem::ml_mate_init", Kokkos::RangePolicy<CpExec>(space, 0, ngPrev),
          KOKKOS_LAMBDA(int g) { mate(g) = g; });
      Kokkos::parallel_for(
          "peclet::dem::ml_match_commit", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
          KOKKOS_LAMBDA(int idx) {
            const ManifoldC m = manifolds(idx);
            if (m.bodyB < 0 || !mldetail::eligible(m, idx, mColor, vn0, vt0, persistent, posPred,
                                                   gHat, qsThr, gateMask))
              return;
            if (excludeImmovable &&
                (invMass(realIdx(m.bodyA)) == 0.0f || invMass(realIdx(m.bodyB)) == 0.0f))
              return;
            const int gA = grp(realIdx(m.bodyA)), gB = grp(realIdx(m.bodyB));
            if (gA == gB)
              return;
            const long long key = colorKey(idx);
            if (winner(gA) == key && winner(gB) == key) {  // sole winner of both endpoints
              mate(gA) = gB;
              mate(gB) = gA;
            }
          });
    }
    // ---- compact new group ids: matched pair -> one id (leader = smaller), singleton keeps ----
    int ngNew = 0;
    {
      auto mate = S.mate;
      auto parent = S.parent;
      const int off = parentOff;
      Kokkos::parallel_scan(
          "peclet::dem::ml_compact", Kokkos::RangePolicy<CpExec>(space, 0, ngPrev),
          KOKKOS_LAMBDA(int g, int& run, const bool final) {
            const bool leader = (mate(g) >= g);
            if (leader)
              ++run;
            if (final && leader)
              parent(off + g) = run - 1;
          },
          ngNew);
      Kokkos::parallel_for(
          "peclet::dem::ml_parent_nonleader", Kokkos::RangePolicy<CpExec>(space, 0, ngPrev),
          KOKKOS_LAMBDA(int g) {
            if (mate(g) < g)
              parent(off + g) = parent(off + mate(g));
          });
    }
    if (ngNew >= ngPrev || ngNew > (9 * ngPrev) / 10)
      break;  // matching stalled: deeper levels would not shrink the problem
    if (groupOff + ngNew > groupCap)
      break;
    // ---- compose the body -> group map and record the level ----
    {
      auto grp = S.grp;
      auto parent = S.parent;
      const int off = parentOff;
      Kokkos::parallel_for(
          "peclet::dem::ml_grp_compose", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
          KOKKOS_LAMBDA(int i) { grp(i) = parent(off + grp(i)); });
    }
    // ---- per-group inverse mass ----
    {
      auto grp = S.grp;
      auto massG = S.massG;
      auto invMassG = S.invMassG;
      const int off = groupOff;
      Kokkos::parallel_for(
          "peclet::dem::ml_mass_reset", Kokkos::RangePolicy<CpExec>(space, 0, ngNew),
          KOKKOS_LAMBDA(int g) { massG(off + g) = 0.0f; });
      Kokkos::parallel_for(
          "peclet::dem::ml_mass_accum", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
          KOKKOS_LAMBDA(int i) {
            Kokkos::atomic_add(&massG(off + grp(i)), mldetail::effMass(invMass(i)));
          });
      Kokkos::parallel_for(
          "peclet::dem::ml_mass_invert", Kokkos::RangePolicy<CpExec>(space, 0, ngNew),
          KOKKOS_LAMBDA(int g) { invMassG(off + g) = 1.0f / massG(off + g); });
    }
    // ---- colour the crossing manifolds of this level on the group graph ----
    // Same round-based random-priority arbitration as colorManifoldsKokkos, endpoints = groups.
    // Committed colours land in this level's 6-bit slot of colorPacked (63 stays = skip).
    {
      auto grp = S.grp;
      auto cp = S.colorPacked;
      const int slotShift = 6 * (lvl - 1);
      Kokkos::parallel_for(
          "peclet::dem::ml_color_mask_reset", Kokkos::RangePolicy<CpExec>(space, 0, ngNew),
          KOKKOS_LAMBDA(int g) { colorMask(g) = 0; });
      // temp per-manifold state via mate-array-free trick: track "uncoloured" in the packed slot
      // itself (63 = pending here; eligibility recomputed per round).
      int remaining = 1, prevRemaining = -1, maxc = -1;
      for (int round = 0; round < 64 && remaining > 0; ++round) {
        Kokkos::parallel_for(
            "peclet::dem::ml_color_reset_winner", Kokkos::RangePolicy<CpExec>(space, 0, ngNew),
            KOKKOS_LAMBDA(int g) { winner(g) = -1; });
        Kokkos::parallel_for(
            "peclet::dem::ml_color_contend", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
            KOKKOS_LAMBDA(int idx) {
              const ManifoldC m = manifolds(idx);
              if (!mldetail::eligible(m, idx, mColor, vn0, vt0, persistent, posPred, gHat, qsThr,
                                      gateMask))
                return;
              if (((cp(idx) >> slotShift) & 63) != kMlSlotSkip)
                return;  // committed in an earlier round
              const int gA = grp(realIdx(m.bodyA));
              const int gB = (m.bodyB >= 0) ? grp(realIdx(m.bodyB)) : -1;
              if (gB == gA)
                return;  // internal to an aggregate at this level
              const long long key = colorKey(idx);
              Kokkos::atomic_max(&winner(gA), key);
              if (gB >= 0)
                Kokkos::atomic_max(&winner(gB), key);
            });
        int rem = 0;
        Kokkos::parallel_reduce(
            "peclet::dem::ml_color_commit", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
            KOKKOS_LAMBDA(int idx, int& acc) {
              const ManifoldC m = manifolds(idx);
              if (!mldetail::eligible(m, idx, mColor, vn0, vt0, persistent, posPred, gHat, qsThr,
                                      gateMask))
                return;
              if (((cp(idx) >> slotShift) & 63) != kMlSlotSkip)
                return;
              const int gA = grp(realIdx(m.bodyA));
              const int gB = (m.bodyB >= 0) ? grp(realIdx(m.bodyB)) : -1;
              if (gB == gA)
                return;
              const long long key = colorKey(idx);
              if (winner(gA) != key || (gB >= 0 && winner(gB) != key)) {
                acc += 1;
                return;
              }
              std::uint64_t forbidden = colorMask(gA);
              if (gB >= 0)
                forbidden |= colorMask(gB);
              int c = 0;
              while (c < kMlSlotSkip - 1 && (forbidden & (std::uint64_t(1) << c)))
                ++c;
              cp(idx) = (cp(idx) & ~(63ll << slotShift)) | (static_cast<long long>(c) << slotShift);
              const std::uint64_t bit = std::uint64_t(1) << c;
              colorMask(gA) |= bit;
              if (gB >= 0)
                colorMask(gB) |= bit;
            },
            rem);
        space.fence();
        if (rem == prevRemaining)
          break;  // mask saturation: leftovers keep slot 63 and are skipped at this level
        prevRemaining = rem;
        remaining = rem;
      }
      Kokkos::parallel_reduce(
          "peclet::dem::ml_color_max", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
          KOKKOS_LAMBDA(int idx, int& mx) {
            const int c = static_cast<int>((cp(idx) >> slotShift) & 63);
            if (c != kMlSlotSkip && c > mx)
              mx = c;
          },
          Kokkos::Max<int>(maxc));
      space.fence();
      H.numColors.push_back(maxc + 1);
    }
    H.groupOff.push_back(groupOff);
    H.numGroups.push_back(ngNew);
    H.parentOff.push_back(parentOff);
    H.numLevels = lvl;
    parentOff += ngPrev;
    groupOff += ngNew;
    ngPrev = ngNew;
    if (ngNew <= 32)
      break;  // coarse enough: a handful of super-bodies solves in one sweep
  }
  space.fence();
  return H;
}

/// One multilevel stabilization cycle over an already-built hierarchy: fine colored smoothing is
/// the caller's business; this runs the coarse leg (fine -> coarse), updating velPred in place.
/// Eligibility is baked into the packed per-level colours (slot 63 = skip), and the composed
/// body -> group map is rebuilt from identity each cycle (numReal-sized passes; cheap next to
/// the sweeps).
/// Dense per-(level, colour) buckets for the coarse cycle, built ONCE per hierarchy (the
/// structure is static for the whole substep, so every sweep of every stabilization iteration
/// reuses them instead of scanning ALL manifolds per colour — measured 40% of GPU time and the
/// bulk of the launch-submission bound at 25k). perm segment for level l lives at
/// [(l-1)*numManifolds, l*numManifolds). Colour slots >= numColors (kMlSlotSkip = uncoloured /
/// non-crossing) are excluded — exactly the set the scan-mode colour filter never matched.
inline void buildCoarseBucketsKokkos(const ContactHierarchy& H, MlScratch& S, int numManifolds,
                                     Kokkos::View<int*, CpMem> colorScratch,
                                     Kokkos::View<int*, CpMem> perm,
                                     Kokkos::View<int*, CpMem> cursor,
                                     std::vector<std::vector<int>>& offs) {
  offs.assign(static_cast<std::size_t>(H.numLevels), {});
  if (numManifolds <= 0)
    return;
  CpExec space;
  for (int lvl = 1; lvl <= H.numLevels; ++lvl) {
    const int slotShift = 6 * (lvl - 1);
    const int nCol = H.numColors[static_cast<std::size_t>(lvl) - 1];
    auto cp = S.colorPacked;
    auto cs = colorScratch;
    Kokkos::parallel_for(
        "peclet::dem::ml_bucket_colors", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) {
          const int c = static_cast<int>((cp(idx) >> slotShift) & 63);
          cs(idx) = (c < nCol) ? c : -1;
        });
    auto seg =
        Kokkos::subview(perm, Kokkos::pair<int, int>((lvl - 1) * numManifolds, lvl * numManifolds));
    buildColorBucketsKokkos(Kokkos::View<const int*, CpMem>(colorScratch), numManifolds, nCol, seg,
                            cursor, offs[static_cast<std::size_t>(lvl) - 1]);
  }
}

/// The per-manifold coarse-PGS body (translation-only, e = 0, shared fine accumulator) lives in
/// MlCoarseSweep so the per-colour launch loop and the fused whole-cycle kernel share it
/// verbatim. grp aliases the composed body -> group map of the CURRENT level (written by the
/// compose phase between sweeps); off is the level's group-pool offset.
struct MlCoarseSweep {
  Kokkos::View<const ManifoldC*, CpMem> manifolds;
  Kokkos::View<const int*, CpMem> realIdx;
  Kokkos::View<const int*, CpMem> grp;
  Kokkos::View<float* [3], CpMem> velG;
  Kokkos::View<const float*, CpMem> invMassG;
  Kokkos::View<float*, CpMem> lambdaAcc;
  Kokkos::View<float, CpMem> maxApproach;
  Kokkos::View<const float*, CpMem> restRel;

  KOKKOS_FUNCTION void solveOne(int idx, int off) const {
    // Poisson release in flight on this pair: the coarse e = 0 solve targets vtil = 0 on
    // the SHARED accumulator and would retract the just-injected separation velocity —
    // the releasing contact skips the coarse transport this substep.
    if (restRel.extent(0) > 0 && restRel(idx) > 0.0f)
      return;
    const ManifoldC m = manifolds(idx);
    const int gA = grp(realIdx(m.bodyA));
    const int gB = (m.bodyB >= 0) ? grp(realIdx(m.bodyB)) : -1;
    const float invN = 1.0f / static_cast<float>(m.num_points);
    const F3 Nsum{m.normal_sum.x, m.normal_sum.y, m.normal_sum.z};
    const float lenN = Kokkos::sqrt(dot3(Nsum, Nsum));
    if (lenN < 1e-9f)
      return;
    const F3 rAavg = scale3(F3{m.rA_sum.x, m.rA_sum.y, m.rA_sum.z}, invN);
    const F3 rBavg = scale3(F3{m.rB_sum.x, m.rB_sum.y, m.rB_sum.z}, invN);
    const F3 diffCenters = (gB < 0) ? rAavg : sub3(rAavg, rBavg);
    const F3 vA{velG(off + gA, 0), velG(off + gA, 1), velG(off + gA, 2)};
    F3 vB{0, 0, 0};
    if (gB >= 0)
      vB = F3{velG(off + gB, 0), velG(off + gB, 1), velG(off + gB, 2)};
    else
      vB = scale3(F3{m.wallVel_sum.x, m.wallVel_sum.y, m.wallVel_sum.z}, invN);
    const float vn = dot3(sub3(vA, vB), Nsum);
    const float alignment = dot3(Nsum, diffCenters);
    const float sgn = (alignment > 0.0f) ? 1.0f : -1.0f;
    const float invMA = invMassG(off + gA);
    const float invMB = (gB >= 0) ? invMassG(off + gB) : 0.0f;
    const float w = dot3(Nsum, Nsum) * (invMA + invMB);
    if (w <= 0.0f)
      return;
    const float vtil = sgn * vn;
    const float dp = vtil / w;  // e = 0: target 0 (pure inelastic support)
    const float pOld = lambdaAcc(idx);
    float pNew = pOld + dp;
    if (pNew < 0.0f)
      pNew = 0.0f;
    const float d = pNew - pOld;
    if (d == 0.0f)
      return;
    lambdaAcc(idx) = pNew;
    Kokkos::atomic_max(&maxApproach(), Kokkos::fabs(d) * w / lenN);
    const float lambda = -sgn * d;
    const F3 J = scale3(Nsum, lambda);
    velG(off + gA, 0) += J.x * invMA;
    velG(off + gA, 1) += J.y * invMA;
    velG(off + gA, 2) += J.z * invMA;
    if (gB >= 0) {
      velG(off + gB, 0) -= J.x * invMB;
      velG(off + gB, 1) -= J.y * invMB;
      velG(off + gB, 2) -= J.z * invMB;
    }
  }
};

/// Host-POD description of a built hierarchy for the fused coarse-cycle kernel (fixed-size
/// arrays: kernel argument by value).
struct MlFusedMeta {
  int numLevels = 0, numReal = 0, coarseSweeps = 0;
  int groupOff[kMlMaxLevels] = {};
  int numGroups[kMlMaxLevels] = {};
  int parentOff[kMlMaxLevels] = {};
  int numColors[kMlMaxLevels] = {};
  int permBase[kMlMaxLevels] = {};  // level's segment base in bkPerm
  int offsBase[kMlMaxLevels] = {};  // level's colour-offset base in the flat device offs
};

/// Fused-coarse-cycle context: flat per-level colour offsets on device + barrier + meta.
/// maxWork == 0 <=> inactive (per-colour launch path).
struct MlFusedCtx {
  Kokkos::View<const int*, CpMem> offsDev;
  Kokkos::View<unsigned*, CpMem> bar;
  MlFusedMeta meta;
  int maxWork = 0;
};

#ifdef KOKKOS_ENABLE_CUDA
/// The ENTIRE coarse cycle as one persistent kernel: per level, compose -> restrict -> coarse
/// colour sweeps -> prolongate, with a grid barrier between the phases (and between colours —
/// the Gauss–Seidel dependency). Work distribution is grid-stride; every phase's per-item math
/// is the launch path's, verbatim (shared MlCoarseSweep::solveOne; the restrict's atomic-add
/// ordering is nondeterministic in BOTH paths). This one launch replaces the ~2,000 tiny
/// launches per step the coarse cycle was measured to emit at 25k.
__device__ inline void demMlCoarseCycleDevice(
    const MlCoarseSweep& f, const Kokkos::View<const int*, CpMem>& parent,
    const Kokkos::View<const float*, CpMem>& invMass,
    const Kokkos::View<float* [3], CpMem>& velPred, const Kokkos::View<float* [3], CpMem>& velG0,
    const Kokkos::View<const float*, CpMem>& massG, const Kokkos::View<int*, CpMem>& grpW,
    const Kokkos::View<const int*, CpMem>& bkPerm, const Kokkos::View<const int*, CpMem>& offsDev,
    const MlFusedMeta& meta, unsigned* bar, unsigned& k, int tid, int stride) {
  for (int i = tid; i < meta.numReal; i += stride)  // reset the composed map to identity
    grpW(i) = i;
  demGridBarrier(bar, k++);
  for (int lvl = 1; lvl <= meta.numLevels; ++lvl) {
    const int off = meta.groupOff[lvl - 1], ng = meta.numGroups[lvl - 1];
    const int pOff = meta.parentOff[lvl - 1], nCol = meta.numColors[lvl - 1];
    for (int i = tid; i < meta.numReal; i += stride)  // compose to this level
      grpW(i) = parent(pOff + grpW(i));
    demGridBarrier(bar, k++);
    for (int g = tid; g < ng; g += stride)  // restrict: reset
      f.velG(off + g, 0) = f.velG(off + g, 1) = f.velG(off + g, 2) = 0.0f;
    demGridBarrier(bar, k++);
    for (int i = tid; i < meta.numReal; i += stride) {  // restrict: momentum accumulate
      const float m = mldetail::effMass(invMass(i));
      const int g = grpW(i);
      Kokkos::atomic_add(&f.velG(off + g, 0), m * velPred(i, 0));
      Kokkos::atomic_add(&f.velG(off + g, 1), m * velPred(i, 1));
      Kokkos::atomic_add(&f.velG(off + g, 2), m * velPred(i, 2));
    }
    demGridBarrier(bar, k++);
    for (int g = tid; g < ng; g += stride) {  // restrict: normalize + snapshot V0
      const float invM = 1.0f / massG(off + g);
      for (int c = 0; c < 3; ++c) {
        f.velG(off + g, c) *= invM;
        velG0(off + g, c) = f.velG(off + g, c);
      }
    }
    demGridBarrier(bar, k++);
    for (int s = 0; s < meta.coarseSweeps; ++s)
      for (int color = 0; color < nCol; ++color) {
        const int b = offsDev(meta.offsBase[lvl - 1] + color);
        const int e = offsDev(meta.offsBase[lvl - 1] + color + 1);
        if (b == e)
          continue;  // empty class: nothing written, no barrier (uniform: same offsets read)
        for (int i2 = b + tid; i2 < e; i2 += stride)
          f.solveOne(bkPerm(meta.permBase[lvl - 1] + i2), off);
        demGridBarrier(bar, k++);
      }
    for (int i = tid; i < meta.numReal; i += stride) {  // prolongate
      const int g = grpW(i);
      velPred(i, 0) += f.velG(off + g, 0) - velG0(off + g, 0);
      velPred(i, 1) += f.velG(off + g, 1) - velG0(off + g, 1);
      velPred(i, 2) += f.velG(off + g, 2) - velG0(off + g, 2);
    }
    demGridBarrier(bar, k++);
  }
}

__global__ void demFusedCoarseCycleK(
    MlCoarseSweep f, Kokkos::View<const int*, CpMem> parent,
    Kokkos::View<const float*, CpMem> invMass, Kokkos::View<float* [3], CpMem> velPred,
    Kokkos::View<float* [3], CpMem> velG0, Kokkos::View<const float*, CpMem> massG,
    Kokkos::View<int*, CpMem> grpW, Kokkos::View<const int*, CpMem> bkPerm,
    Kokkos::View<const int*, CpMem> offsDev, MlFusedMeta meta, unsigned* bar) {
  const int stride = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned k = 0;
  demMlCoarseCycleDevice(f, parent, invMass, velPred, velG0, massG, grpW, bkPerm, offsDev, meta,
                         bar, k, tid, stride);
}

/// The ENTIRE multilevel stabilization loop as one kernel: per iteration, zero the QS residual,
/// run the fine colored smoothing sweep, run the coarse cycle, then take the host loop's
/// adaptive stop on-device (same residual, same tolerance — the per-iteration readback and the
/// graph capture disappear). meta.coarseSweeps, the colour orderings and every per-item body
/// are the launch path's, verbatim.
__global__ void demFusedMlLoopK(
    PGSManifoldSweep fine, Kokkos::View<const int*, CpMem> vPerm,
    Kokkos::View<const int*, CpMem> vOffs, int vCols, MlCoarseSweep f,
    Kokkos::View<const int*, CpMem> parent, Kokkos::View<const float*, CpMem> invMass,
    Kokkos::View<float* [3], CpMem> velPred, Kokkos::View<float* [3], CpMem> velG0,
    Kokkos::View<const float*, CpMem> massG, Kokkos::View<int*, CpMem> grpW,
    Kokkos::View<const int*, CpMem> bkPerm, Kokkos::View<const int*, CpMem> offsDev,
    MlFusedMeta meta, int maxIters, float tol, float* resQS, unsigned* bar) {
  const int stride = gridDim.x * blockDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned k = 0;
  for (int it = 0; it < maxIters; ++it) {
    if (tid == 0)
      *resQS = 0.0f;
    demGridBarrier(bar, k++);
    for (int c = 0; c < vCols; ++c) {  // fine colored smoothing sweep
      const int b = vOffs(c), e = vOffs(c + 1);
      if (b == e)
        continue;
      for (int i = b + tid; i < e; i += stride)
        fine.solveOne(vPerm(i));
      demGridBarrier(bar, k++);
    }
    demMlCoarseCycleDevice(f, parent, invMass, velPred, velG0, massG, grpW, bkPerm, offsDev, meta,
                           bar, k, tid, stride);
    const float r = *reinterpret_cast<volatile float*>(resQS);
    if (r <= tol)
      break;
    demGridBarrier(bar, k++);  // every block has read resQS before the next zero
  }
}
#endif  // KOKKOS_ENABLE_CUDA

#ifdef KOKKOS_ENABLE_CUDA
/// Launch the whole multilevel stabilization loop as one kernel (see demFusedMlLoopK). The
/// final QS residual stays in maxApproachQS for the host's post-loop read. Returns false when
/// the fused path cannot run — the caller keeps its host-side iteration loop.
inline bool demLaunchFusedMlLoop(
    CpExec& space, const PGSManifoldSweep& fine, Kokkos::View<const int*, CpMem> vPerm,
    const FusedSweepCtx& velCtx, int vCols, Kokkos::View<const ManifoldC*, CpMem> manifolds,
    Kokkos::View<const int*, CpMem> realIdx, Kokkos::View<const float*, CpMem> invMass,
    Kokkos::View<float* [3], CpMem> velPred, Kokkos::View<float*, CpMem> lambdaAcc,
    Kokkos::View<float, CpMem> maxApproachQS, Kokkos::View<const float*, CpMem> restRel,
    MlScratch& S, Kokkos::View<const int*, CpMem> bkPerm, const MlFusedCtx& ml, int maxIters,
    float tol) {
  const int maxGrid =
      std::min(demFusedMaxGrid(demFusedMlLoopK), (static_cast<int>(ml.bar.extent(0)) - 1) / 8);
  if (maxGrid <= 0 || vCols <= 0 || velCtx.maxBucket <= 0 || ml.maxWork <= 0)
    return false;
  const MlCoarseSweep f{manifolds,
                        realIdx,
                        Kokkos::View<const int*, CpMem>(S.grp),
                        S.velG,
                        Kokkos::View<const float*, CpMem>(S.invMassG),
                        lambdaAcc,
                        maxApproachQS,
                        restRel};
  const int work = std::max(velCtx.maxBucket, ml.maxWork);
  const int want = std::min((work + kFusedBlock - 1) / kFusedBlock, std::max(1, demFusedGridCap()));
  const int grid = want < maxGrid ? want : maxGrid;
  cudaStream_t str = space.cuda_stream();
  cudaMemsetAsync(ml.bar.data(), 0, (static_cast<std::size_t>(grid) * 8 + 1) * sizeof(unsigned),
                  str);
  demFusedMlLoopK<<<grid, kFusedBlock, 0, str>>>(
      fine, vPerm, velCtx.offsDev, vCols, f, Kokkos::View<const int*, CpMem>(S.parent), invMass,
      velPred, S.velG0, Kokkos::View<const float*, CpMem>(S.massG), S.grp, bkPerm, ml.offsDev,
      ml.meta, maxIters, tol, maxApproachQS.data(), ml.bar.data());
  return true;
}
#endif  // KOKKOS_ENABLE_CUDA

/// Build the fused-coarse-cycle context from an already-built hierarchy + its dense buckets:
/// flatten the per-level colour offsets into the pooled device view (async upload) and record
/// the grid-sizing work bound. Inactive (maxWork = 0) when fused sweeps are off / no CUDA / the
/// pooled view is too small.
inline MlFusedCtx demMakeMlFusedCtx(CpExec& space, const ContactHierarchy& H,
                                    const std::vector<std::vector<int>>& bkOffs, int numManifolds,
                                    int numReal, int coarseSweeps,
                                    Kokkos::View<int*, CpMem> offsDev,
                                    Kokkos::View<unsigned*, CpMem> bar) {
  MlFusedCtx ctx;
#ifdef KOKKOS_ENABLE_CUDA
  if (H.numLevels <= 0 || H.numLevels > kMlMaxLevels || numReal <= 0)
    return ctx;
  std::vector<int> flat;
  ctx.meta.numLevels = H.numLevels;
  ctx.meta.numReal = numReal;
  ctx.meta.coarseSweeps = coarseSweeps;
  int maxWork = numReal;  // compose/restrict/prolongate phases (ng <= numReal always)
  for (int lvl = 1; lvl <= H.numLevels; ++lvl) {
    const auto& lo = bkOffs[static_cast<std::size_t>(lvl) - 1];
    if (lo.size() != static_cast<std::size_t>(H.numColors[lvl - 1]) + 1)
      return ctx;  // bucket build skipped this level (empty) — keep the launch path
    ctx.meta.groupOff[lvl - 1] = H.groupOff[lvl - 1];
    ctx.meta.numGroups[lvl - 1] = H.numGroups[lvl - 1];
    ctx.meta.parentOff[lvl - 1] = H.parentOff[lvl - 1];
    ctx.meta.numColors[lvl - 1] = H.numColors[lvl - 1];
    ctx.meta.permBase[lvl - 1] = (lvl - 1) * numManifolds;
    ctx.meta.offsBase[lvl - 1] = static_cast<int>(flat.size());
    for (std::size_t c = 0; c + 1 < lo.size(); ++c)
      maxWork = std::max(maxWork, lo[c + 1] - lo[c]);
    flat.insert(flat.end(), lo.begin(), lo.end());
  }
  if (flat.empty() || flat.size() > offsDev.extent(0))
    return ctx;
  const Kokkos::View<const int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h(flat.data(),
                                                                               flat.size());
  auto d = Kokkos::subview(offsDev, Kokkos::pair<std::size_t, std::size_t>(0, flat.size()));
  Kokkos::deep_copy(space, d, h);
  ctx.offsDev = Kokkos::View<const int*, CpMem>(offsDev);
  ctx.bar = bar;
  ctx.maxWork = maxWork;
#else
  (void)space;
  (void)H;
  (void)bkOffs;
  (void)numManifolds;
  (void)numReal;
  (void)coarseSweeps;
  (void)offsDev;
  (void)bar;
#endif
  return ctx;
}

inline void multilevelCoarseCycleKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const int*, CpMem> realIdx, Kokkos::View<const float*, CpMem> invMass,
    Kokkos::View<float* [3], CpMem> velPred, Kokkos::View<float*, CpMem> lambdaAcc,
    Kokkos::View<float, CpMem> maxApproach, int numReal, const ContactHierarchy& H, MlScratch& S,
    int coarseSweeps, Kokkos::View<const float*, CpMem> restRel = {},
    const std::vector<std::vector<int>>* bkOffs = nullptr,
    Kokkos::View<const int*, CpMem> bkPerm = {}, const MlFusedCtx* fused = nullptr) {
  CpExec space;
#ifdef KOKKOS_ENABLE_CUDA
  if (fused && fused->maxWork > 0) {
    const MlCoarseSweep f{manifolds,
                          realIdx,
                          Kokkos::View<const int*, CpMem>(S.grp),
                          S.velG,
                          Kokkos::View<const float*, CpMem>(S.invMassG),
                          lambdaAcc,
                          maxApproach,
                          restRel};
    const int maxGrid = std::min(demFusedMaxGrid(demFusedCoarseCycleK),
                                 (static_cast<int>(fused->bar.extent(0)) - 1) / 8);
    if (maxGrid > 0) {
      const int want = std::min((fused->maxWork + kFusedBlock - 1) / kFusedBlock,
                                std::max(1, demFusedGridCap()));
      const int grid = want < maxGrid ? want : maxGrid;
      cudaStream_t str = space.cuda_stream();
      cudaMemsetAsync(fused->bar.data(), 0,
                      (static_cast<std::size_t>(grid) * 8 + 1) * sizeof(unsigned), str);
      demFusedCoarseCycleK<<<grid, kFusedBlock, 0, str>>>(
          f, Kokkos::View<const int*, CpMem>(S.parent), invMass, velPred, S.velG0,
          Kokkos::View<const float*, CpMem>(S.massG), S.grp, bkPerm, fused->offsDev, fused->meta,
          fused->bar.data());
      return;
    }
  }
#else
  (void)fused;
#endif
  {  // reset the composed map to identity; each level applies its parent map on top
    auto grp = S.grp;
    Kokkos::parallel_for(
        "peclet::dem::ml_cycle_grp_init", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
        KOKKOS_LAMBDA(int i) { grp(i) = i; });
  }
  for (int lvl = 1; lvl <= H.numLevels; ++lvl) {
    const int off = H.groupOff[lvl - 1];
    const int ng = H.numGroups[lvl - 1];
    const int pOff = H.parentOff[lvl - 1];
    const int nCol = H.numColors[lvl - 1];
    const int slotShift = 6 * (lvl - 1);
    auto grp = S.grp;
    {  // compose to this level
      auto parent = S.parent;
      Kokkos::parallel_for(
          "peclet::dem::ml_cycle_compose", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
          KOKKOS_LAMBDA(int i) { grp(i) = parent(pOff + grp(i)); });
    }
    {  // restrict: V_g = sum(m v) / sum(m) (momentum-conserving), snapshot V0
      auto velG = S.velG;
      auto velG0 = S.velG0;
      auto massG = S.massG;
      Kokkos::parallel_for(
          "peclet::dem::ml_restrict_reset", Kokkos::RangePolicy<CpExec>(space, 0, ng),
          KOKKOS_LAMBDA(int g) { velG(off + g, 0) = velG(off + g, 1) = velG(off + g, 2) = 0.0f; });
      Kokkos::parallel_for(
          "peclet::dem::ml_restrict_accum", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
          KOKKOS_LAMBDA(int i) {
            const float m = mldetail::effMass(invMass(i));
            const int g = grp(i);
            Kokkos::atomic_add(&velG(off + g, 0), m * velPred(i, 0));
            Kokkos::atomic_add(&velG(off + g, 1), m * velPred(i, 1));
            Kokkos::atomic_add(&velG(off + g, 2), m * velPred(i, 2));
          });
      Kokkos::parallel_for(
          "peclet::dem::ml_restrict_norm", Kokkos::RangePolicy<CpExec>(space, 0, ng),
          KOKKOS_LAMBDA(int g) {
            const float invM = 1.0f / massG(off + g);
            for (int c = 0; c < 3; ++c) {
              velG(off + g, c) *= invM;
              velG0(off + g, c) = velG(off + g, c);
            }
          });
    }
    // coarse colored PGS sweeps: translation-only, e = 0, shared lambda accumulator (the
    // per-manifold body is MlCoarseSweep::solveOne — shared verbatim with the fused kernel)
    const MlCoarseSweep fSweep{manifolds,
                               realIdx,
                               Kokkos::View<const int*, CpMem>(S.grp),
                               S.velG,
                               Kokkos::View<const float*, CpMem>(S.invMassG),
                               lambdaAcc,
                               maxApproach,
                               restRel};
    for (int s = 0; s < coarseSweeps; ++s) {
      for (int color = 0; color < nCol; ++color) {
        auto cp = S.colorPacked;
        const bool dense = bkOffs != nullptr;
        int rb = 0, re = numManifolds;
        if (dense) {
          const auto& lo = (*bkOffs)[static_cast<std::size_t>(lvl) - 1];
          rb = (lvl - 1) * numManifolds + lo[static_cast<std::size_t>(color)];
          re = (lvl - 1) * numManifolds + lo[static_cast<std::size_t>(color) + 1];
          if (rb == re)
            continue;
        }
        Kokkos::parallel_for(
            "peclet::dem::ml_coarse_pgs", Kokkos::RangePolicy<CpExec>(space, rb, re),
            KOKKOS_LAMBDA(int i2) {
              const int idx = dense ? bkPerm(i2) : i2;
              if (!dense && static_cast<int>((cp(idx) >> slotShift) & 63) != color)
                return;
              fSweep.solveOne(idx, off);
            });
      }
    }
    {  // prolongate: every member takes its aggregate's velocity delta (mass-proportional impulse)
      auto velG = S.velG;
      auto velG0 = S.velG0;
      Kokkos::parallel_for(
          "peclet::dem::ml_prolongate", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
          KOKKOS_LAMBDA(int i) {
            const int g = grp(i);
            velPred(i, 0) += velG(off + g, 0) - velG0(off + g, 0);
            velPred(i, 1) += velG(off + g, 1) - velG0(off + g, 1);
            velPred(i, 2) += velG(off + g, 2) - velG0(off + g, 2);
          });
    }
  }
}

}  // namespace peclet::dem

#endif  // DEM_SOLVER_MULTILEVEL_HPP
