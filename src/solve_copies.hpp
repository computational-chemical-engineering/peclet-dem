/// @file
/// @brief dem — body copies of the contact solve (docs/contact_solve_framework.md §4.4, §4.5,
/// §WO-4 item 2): hub copies of a colouring vertex above kHubEdgeBudget edges, the fold groups of a
/// phase (hubs; demStep's periodic images in the position phase), the mass-split solve views, and
/// the seed / fold / re-seed kernels demSolveContacts runs around its sweeps. Free functions over
/// the Particles SoA; no ArborX, no MPI.
#ifndef DEM_SOLVE_COPIES_HPP
#define DEM_SOLVE_COPIES_HPP

#include <cstddef>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <string>

#include "contact_preprocessing.hpp"  // ManifoldC, ContactC, PosUnits, SlotOverride
#include "particles.hpp"              // Particles, PhaseCopies

namespace peclet::dem {

// ================================ contact-solve copies ================================
// docs/contact_solve_framework.md §4.4 (hub copies), §4.5 (solve views, relaxation) and §WO-4
// item 2 (demStep's periodic images in the position phase). A body updated through more than one
// slot in a phase is a fold GROUP: its slots (the base, hub copies, images) are solved with the
// mass-split inverse masses k x invMass, and after every sweep the group is folded -- the base
// takes sigma + (sum of the members' increments) / k and every member is re-seeded to it (plus its
// periodic shift). sigma is the phase seed, re-marked after every rank sync.

/// Hub edge budget (§4.4): a colouring vertex with more active edges is split into
/// ceil(d / kHubEdgeBudget) copies, so no copy carries more than 32 and the 64-colour greedy
/// provably succeeds (§1.6).
inline constexpr int kHubEdgeBudget = 32;
/// Split relaxation (§4.5, §13.1): the PGS normal step on an edge touching a mass-split slot is
/// scaled by kSplitOmegaVelocity before the clamp (1: no relaxation; the pre-designed lever of
/// R-F2, legitimate there because the PGS normal multiplier is accumulated and its clamp can
/// retract an overshoot). The overlap projection is the accumulated, retractable form since WO-12
/// (projected SOR on each contact's net push at P.positionOmega = 1.5, solver_position.hpp); the
/// non-accumulated POCS of §13.1 could not retract and was held at omega 1. At omega 1 each update
/// of that old form was an exact projection in the split metric, the local fold and the rank
/// reconciliation are exact projections onto the consensus subspaces, and cyclic projection onto
/// these half-spaces and subspaces converges to a feasible point (Fejer-monotone; which point
/// depends on the order, as for serial POCS).
inline constexpr float kSplitOmegaVelocity = 1.0f;

/// Grow-only (re)allocation of a copy-machinery view to at least n entries.
template <class V>
inline void growCopyView(V& v, std::size_t n, const char* label) {
  if (v.extent(0) < n)
    v = V(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), n + n / 2 + 64);
}

/// Velocity-phase colouring edges: the vertices realIdx(bodyA / bodyB) of each manifold (-1: a
/// wall end, or both ends of an inactive (-2) manifold). §4.4 step 1: active = colour != -2.
inline void velocityEdgeEndsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int nm,
                                   Kokkos::View<const int*, CpMem> realIdx,
                                   Kokkos::View<const int*, CpMem> color,
                                   Kokkos::View<int*, CpMem> eA, Kokkos::View<int*, CpMem> eB) {
  Kokkos::parallel_for(
      "peclet::dem::copies_vel_ends", Kokkos::RangePolicy<CpExec>(0, nm), KOKKOS_LAMBDA(int e) {
        const ManifoldC m = manifolds(e);
        const bool act = color(e) != -2;
        eA(e) = act ? realIdx(m.bodyA) : -1;
        eB(e) = (act && m.bodyB >= 0) ? realIdx(m.bodyB) : -1;
      });
}

/// Position-phase colouring edges: one per unit, the leader contact's raw (bodyA, bodyB);
/// active = the unit is not sleep-masked (colour != -2).
inline void positionEdgeEndsKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                   const PosUnits& units, int nu,
                                   Kokkos::View<const int*, CpMem> color,
                                   Kokkos::View<int*, CpMem> eA, Kokkos::View<int*, CpMem> eB) {
  Kokkos::parallel_for(
      "peclet::dem::copies_pos_ends", Kokkos::RangePolicy<CpExec>(0, nu), KOKKOS_LAMBDA(int u) {
        const ContactC c = contacts(units.leader(u));
        const bool act = color(u) != -2;
        eA(u) = act ? c.bodyA : -1;
        eB(u) = (act && c.bodyB >= 0) ? c.bodyB : -1;
      });
}

/// Per-vertex active degree over [0, nV) (§4.4 step 1). Wall edges have one endpoint.
inline void vertexDegreeKokkos(Kokkos::View<const int*, CpMem> eA,
                               Kokkos::View<const int*, CpMem> eB, int nE, int nV,
                               Kokkos::View<int*, CpMem> deg) {
  Kokkos::deep_copy(Kokkos::subview(deg, Kokkos::pair<int, int>(0, nV)), 0);
  Kokkos::parallel_for(
      "peclet::dem::copies_degree", Kokkos::RangePolicy<CpExec>(0, nE), KOKKOS_LAMBDA(int e) {
        if (eA(e) >= 0)
          Kokkos::atomic_add(&deg(eA(e)), 1);
        if (eB(e) >= 0)
          Kokkos::atomic_add(&deg(eB(e)), 1);
      });
  Kokkos::fence();
}

/// Build one phase's hub copies (§4.4 steps 2, 3 and 5): every vertex with d > kHubEdgeBudget
/// active edges gets s = ceil(d / 32) copies -- its own slot plus s - 1 slots appended at
/// [slotBase, slotBase + nCopies). Its incident edges, listed as (hub, edge index[, end]) and
/// sorted, go round-robin: the j-th to copy j mod s (copy 0 = the base slot); an edge between two
/// hubs is assigned independently at each end. Writes the per-edge overrides slotA / slotB (-1 =
/// none) and copyBase. Deterministic: the order is the sorted key, never thread order.
inline void buildHubCopiesKokkos(Kokkos::View<const int*, CpMem> eA,
                                 Kokkos::View<const int*, CpMem> eB, int nE, int nV, int slotBase,
                                 Kokkos::View<const int*, CpMem> deg, PhaseCopies& H) {
  CpExec space;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  H.nHubs = 0;
  H.nCopies = 0;
  H.slotBase = slotBase;
  H.maxHubDegree = 0;
  growCopyView(H.slotA, static_cast<std::size_t>(nE), "peclet::dem::copies_slotA");
  growCopyView(H.slotB, static_cast<std::size_t>(nE), "peclet::dem::copies_slotB");
  Kokkos::deep_copy(space, Kokkos::subview(H.slotA, Kokkos::pair<int, int>(0, nE)), -1);
  Kokkos::deep_copy(space, Kokkos::subview(H.slotB, Kokkos::pair<int, int>(0, nE)), -1);
  Kokkos::View<int*, CpMem> hubId(
      view_alloc(space, "peclet::dem::copies_hubid", WithoutInitializing), nV);
  int nHubs = 0;
  Kokkos::parallel_scan(
      "peclet::dem::copies_hub_scan", Kokkos::RangePolicy<CpExec>(space, 0, nV),
      KOKKOS_LAMBDA(int v, int& run, const bool final) {
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 3
        const bool hub = false;  // G13 mutant 3: no hub copies
#else
        const bool hub = deg(v) > kHubEdgeBudget;
#endif
        if (final)
          hubId(v) = hub ? run : -1;
        if (hub)
          ++run;
      },
      nHubs);
  space.fence();
  H.nHubs = nHubs;
  if (nHubs == 0)
    return;
  growCopyView(H.hubVertex, static_cast<std::size_t>(nHubs), "peclet::dem::copies_hubv");
  growCopyView(H.hubS, static_cast<std::size_t>(nHubs), "peclet::dem::copies_hubs");
  auto hv = H.hubVertex;
  auto hs = H.hubS;
  Kokkos::parallel_for(
      "peclet::dem::copies_hub_list", Kokkos::RangePolicy<CpExec>(space, 0, nV),
      KOKKOS_LAMBDA(int v) {
        const int h = hubId(v);
        if (h < 0)
          return;
        hv(h) = v;
        hs(h) = (deg(v) + kHubEdgeBudget - 1) / kHubEdgeBudget;
      });
  Kokkos::View<int*, CpMem> copyOff(
      view_alloc(space, "peclet::dem::copies_off", WithoutInitializing), nHubs);
  Kokkos::View<int*, CpMem> degOff(
      view_alloc(space, "peclet::dem::copies_doff", WithoutInitializing), nHubs);
  int nCopies = 0, nEntries = 0, maxDeg = 0;
  Kokkos::parallel_scan(
      "peclet::dem::copies_off_scan", Kokkos::RangePolicy<CpExec>(space, 0, nHubs),
      KOKKOS_LAMBDA(int h, int& run, const bool final) {
        if (final)
          copyOff(h) = run;
        run += hs(h) - 1;
      },
      nCopies);
  Kokkos::parallel_scan(
      "peclet::dem::copies_doff_scan", Kokkos::RangePolicy<CpExec>(space, 0, nHubs),
      KOKKOS_LAMBDA(int h, int& run, const bool final) {
        if (final)
          degOff(h) = run;
        run += deg(hv(h));
      },
      nEntries);
  Kokkos::parallel_reduce(
      "peclet::dem::copies_maxdeg", Kokkos::RangePolicy<CpExec>(space, 0, nHubs),
      KOKKOS_LAMBDA(int h, int& m) {
        if (deg(hv(h)) > m)
          m = deg(hv(h));
      },
      Kokkos::Max<int>(maxDeg));
  space.fence();
  H.nCopies = nCopies;
  H.maxHubDegree = maxDeg;
  // (hub, edge, end) keys: the hub in the high bits, then the edge index, then the end (0 = A,
  // 1 = B), so each hub's edges sort by edge index -- canonical (manifolds and units are sorted by
  // pair key).
  Kokkos::View<std::uint64_t*, CpMem> keys(
      view_alloc(space, "peclet::dem::copies_keys", WithoutInitializing), nEntries);
  Kokkos::View<int, CpMem> cursor("peclet::dem::copies_cursor");
  Kokkos::parallel_for(
      "peclet::dem::copies_entries", Kokkos::RangePolicy<CpExec>(space, 0, nE),
      KOKKOS_LAMBDA(int e) {
        const int ends[2] = {eA(e), eB(e)};
        for (int end = 0; end < 2; ++end) {
          const int v = ends[end];
          if (v < 0 || hubId(v) < 0)
            continue;
          const int p = Kokkos::atomic_fetch_add(&cursor(), 1);
          keys(p) = (static_cast<std::uint64_t>(hubId(v)) << 33) |
                    (static_cast<std::uint64_t>(static_cast<unsigned>(e)) << 1) |
                    static_cast<std::uint64_t>(end);
        }
      });
  Kokkos::sort(space, keys);
  growCopyView(H.copyBase, static_cast<std::size_t>(nCopies > 0 ? nCopies : 1),
               "peclet::dem::copies_base");
  auto sA = H.slotA;
  auto sB = H.slotB;
  auto cb = H.copyBase;
  Kokkos::parallel_for(
      "peclet::dem::copies_assign", Kokkos::RangePolicy<CpExec>(space, 0, nEntries),
      KOKKOS_LAMBDA(int p) {
        const std::uint64_t k = keys(p);
        const int h = static_cast<int>(k >> 33);
        const int e = static_cast<int>((k >> 1) & 0xFFFFFFFFull);
        const int end = static_cast<int>(k & 1ull);
        const int s = hs(h);
        const int c = (p - degOff(h)) % s;
        const int slot = (c == 0) ? hv(h) : slotBase + copyOff(h) + c - 1;
        if (end == 0)
          sA(e) = slot;
        else
          sB(e) = slot;
        if (c > 0)
          cb(copyOff(h) + c - 1) = hv(h);
      });
  space.fence();
}

/// Build the fold groups of a phase (§4.4 "local fold", §WO-4 item 2): every body with more than
/// one slot among [0, span) (the phase's vertex slots) and the hub copies. A slot's body is
/// realIdx(slot) (a copy's is its hub vertex's), so in demStep's position phase a body's periodic
/// images join its group; under MPI and in the velocity phase realIdx is the identity on the
/// vertices and only hubs form groups. Members are sorted (body, slot), so the body's own slot --
/// the smallest -- comes first. k = the members with at least one active edge (a copy always has
/// one). imageShift (empty = none) gives an image slot's periodic shift; a copy takes its hub
/// vertex's.
///
/// identityBodies: every vertex slot is its own body (a copy's body is its hub vertex), so only
/// hubs form groups. The velocity phase (vertices are realIdx slots; under MPI the §6.2 slot map
/// makes the other slots of a body aliases, which must not join its group) and the distributed
/// position phase (raw slots: every image slot is its own copy, reconciled at rank level, §13.3).
/// On the single-rank velocity phase realIdx is the identity on [0, span), so it is the same
/// grouping.
inline void buildCopyGroupsKokkos(Kokkos::View<const int*, CpMem> realIdx, int span, int numReal,
                                  Kokkos::View<const int*, CpMem> deg,
                                  Kokkos::View<const float* [3], CpMem> imageShift, PhaseCopies& H,
                                  bool identityBodies = false) {
  CpExec space;
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  const int nC = H.nCopies, base = H.slotBase, nQ = span + nC;
  H.nGroups = 0;
  if (nQ <= 0)
    return;
  auto cb = H.copyBase;
  Kokkos::View<int*, CpMem> cnt("peclet::dem::copies_cnt", span);
  Kokkos::parallel_for(
      "peclet::dem::copies_grp_cnt", Kokkos::RangePolicy<CpExec>(space, 0, nQ),
      KOKKOS_LAMBDA(int qi) {
        const int o = identityBodies ? ((qi < span) ? qi : cb(qi - span))
                                     : ((qi < span) ? realIdx(qi) : realIdx(cb(qi - span)));
        Kokkos::atomic_add(&cnt(o), 1);
      });
  Kokkos::View<std::uint64_t*, CpMem> keys(
      view_alloc(space, "peclet::dem::copies_gkeys", WithoutInitializing), nQ);
  int nMem = 0;
  Kokkos::parallel_reduce(
      "peclet::dem::copies_grp_keys", Kokkos::RangePolicy<CpExec>(space, 0, nQ),
      KOKKOS_LAMBDA(int qi, int& acc) {
        const int q = (qi < span) ? qi : base + (qi - span);
        const int o = identityBodies ? ((qi < span) ? qi : cb(qi - span))
                                     : ((qi < span) ? realIdx(qi) : realIdx(cb(qi - span)));
        if (cnt(o) >= 2) {
          keys(qi) = (static_cast<std::uint64_t>(static_cast<unsigned>(o)) << 32) |
                     static_cast<unsigned>(q);
          acc += 1;
        } else {
          keys(qi) = ~std::uint64_t(0);
        }
      },
      nMem);
  space.fence();
  if (nMem == 0)
    return;
  Kokkos::sort(space, keys);
  growCopyView(H.groupStart, static_cast<std::size_t>(nMem) + 1, "peclet::dem::copies_gstart");
  growCopyView(H.groupSlot, static_cast<std::size_t>(nMem), "peclet::dem::copies_gslot");
  growCopyView(H.groupShift, static_cast<std::size_t>(nMem), "peclet::dem::copies_gshift");
  growCopyView(H.groupActive, static_cast<std::size_t>(nMem), "peclet::dem::copies_gact");
  auto gs = H.groupStart;
  auto gq = H.groupSlot;
  auto gsh = H.groupShift;
  auto ga = H.groupActive;
  const bool haveShift = imageShift.extent(0) > 0;
  int nG = 0;
  Kokkos::parallel_scan(
      "peclet::dem::copies_grp_scan", Kokkos::RangePolicy<CpExec>(space, 0, nMem),
      KOKKOS_LAMBDA(int p, int& run, const bool final) {
        const bool isNew = (p == 0) || ((keys(p) >> 32) != (keys(p - 1) >> 32));
        if (final) {
          const int q = static_cast<int>(keys(p) & 0xFFFFFFFFull);
          gq(p) = q;
          const int b = (q < base) ? q : cb(q - base);  // an image's (or its copy's) image slot
          F3 sh{0.0f, 0.0f, 0.0f};
          if (haveShift && b >= numReal && b < span)
            sh = F3{imageShift(b, 0), imageShift(b, 1), imageShift(b, 2)};
          gsh(p, 0) = sh.x;
          gsh(p, 1) = sh.y;
          gsh(p, 2) = sh.z;
          ga(p) = (q >= base || deg(q) > 0) ? 1 : 0;  // a copy always has an edge
          if (isNew)
            gs(run) = p;
        }
        if (isNew)
          ++run;
      },
      nG);
  Kokkos::deep_copy(space, Kokkos::subview(gs, nG), nMem);
  growCopyView(H.groupK, static_cast<std::size_t>(nG), "peclet::dem::copies_gk");
  growCopyView(H.seedX, static_cast<std::size_t>(nG), "peclet::dem::copies_seedx");
  growCopyView(H.seedW, static_cast<std::size_t>(nG), "peclet::dem::copies_seedw");
  auto gk = H.groupK;
  Kokkos::parallel_for(
      "peclet::dem::copies_grp_k", Kokkos::RangePolicy<CpExec>(space, 0, nG), KOKKOS_LAMBDA(int g) {
        int k = 0;
        for (int p = gs(g); p < gs(g + 1); ++p)
          k += ga(p);
        gk(g) = k > 0 ? k : 1;
      });
  space.fence();
  H.nGroups = nG;
}

/// Solve views of a phase (§4.5): invMassSolve / invInertiaSolve = the phase's masses on every
/// slot in [0, nSlots), times k on every member of a group, and splitSlot = (k > 1). Built only
/// while the phase has groups.
inline void buildSolveViewsKokkos(Particles& P, const PhaseCopies& H, int nSlots,
                                  Kokkos::View<const float*, CpMem> invMass,
                                  Kokkos::View<const float* [3], CpMem> invInertia) {
  CpExec space;
  growCopyView(P.invMassSolve, static_cast<std::size_t>(nSlots), "peclet::dem::invMassSolve");
  growCopyView(P.invInertiaSolve, static_cast<std::size_t>(nSlots), "peclet::dem::invInertiaSolve");
  growCopyView(P.splitSlot, static_cast<std::size_t>(nSlots), "peclet::dem::splitSlot");
  auto ims = P.invMassSolve;
  auto iis = P.invInertiaSolve;
  auto spl = P.splitSlot;
  Kokkos::parallel_for(
      "peclet::dem::copies_solve_views", Kokkos::RangePolicy<CpExec>(space, 0, nSlots),
      KOKKOS_LAMBDA(int q) {
        const bool own = q < static_cast<int>(invMass.extent(0));
        ims(q) = own ? invMass(q) : 0.0f;
        for (int c = 0; c < 3; ++c)
          iis(q, c) = own ? invInertia(q, c) : 0.0f;
        spl(q) = 0;
      });
  auto gs = H.groupStart;
  auto gq = H.groupSlot;
  auto gk = H.groupK;
  Kokkos::parallel_for(
      "peclet::dem::copies_solve_split", Kokkos::RangePolicy<CpExec>(space, 0, H.nGroups),
      KOKKOS_LAMBDA(int g) {
        const int b = gq(gs(g));
        const int k = gk(g);
        const float kf = static_cast<float>(k);
        const float m = kf * invMass(b);
        const float i0 = kf * invInertia(b, 0), i1 = kf * invInertia(b, 1),
                    i2 = kf * invInertia(b, 2);
        for (int p = gs(g); p < gs(g + 1); ++p) {
          const int q = gq(p);
          ims(q) = m;
          iis(q, 0) = i0;
          iis(q, 1) = i1;
          iis(q, 2) = i2;
          spl(q) = k > 1 ? 1 : 0;
        }
      });
  space.fence();
}

/// Seed the copy slots' read-only state from their hub vertex (§4.4 step 6): orientation, local
/// inverse inertia and grounded level (velocity), predicted orientation (position).
inline void seedCopySlotStateKokkos(Particles& P, const PhaseCopies& H) {
  if (H.nCopies <= 0)
    return;
  auto cb = H.copyBase;
  const int base = H.slotBase;
  auto quat = P.quat;
  auto quatPred = P.quatPred;
  auto invI = P.invInertia;
  auto grounded = P.groundedLevel;
  Kokkos::parallel_for(
      "peclet::dem::copies_seed_state", Kokkos::RangePolicy<CpExec>(0, H.nCopies),
      KOKKOS_LAMBDA(int c) {
        const int q = base + c, b = cb(c);
        for (int j = 0; j < 4; ++j) {
          quat(q, j) = quat(b, j);
          quatPred(q, j) = quatPred(b, j);
        }
        for (int j = 0; j < 3; ++j)
          invI(q, j) = invI(b, j);
        grounded(q) = grounded(b);
      });
  Kokkos::fence();
}

/// Mark the phase seed sigma of every group and re-seed its members from the base (§4.4 step 6,
/// and after every rank sync): x(q) = sigma + shift(q); w likewise (velocity, no shift). With
/// `orphan` (Poisson, velocity) every member's account becomes the share B / k of the base's
/// balance B, and the peak the base's. shareFromBase (rank-level M, §13.3): the base already holds
/// the share B / k_global (set by the opening / rank sync), and every member copies it.
inline void markCopySeedsKokkos(const PhaseCopies& H, Kokkos::View<float* [3], CpMem> x,
                                Kokkos::View<float* [3], CpMem> w,
                                Kokkos::View<float*, CpMem> orphan,
                                Kokkos::View<float*, CpMem> orphanPeak,
                                bool shareFromBase = false) {
  if (H.nGroups <= 0)
    return;
  auto gs = H.groupStart;
  auto gq = H.groupSlot;
  auto gsh = H.groupShift;
  auto gk = H.groupK;
  auto sx = H.seedX;
  auto sw = H.seedW;
  const bool haveW = w.extent(0) > 0;
  const bool haveO = orphan.extent(0) > 0;
  Kokkos::parallel_for(
      "peclet::dem::copies_mark", Kokkos::RangePolicy<CpExec>(0, H.nGroups), KOKKOS_LAMBDA(int g) {
        const int b = gq(gs(g));
        for (int c = 0; c < 3; ++c) {
          sx(g, c) = x(b, c);
          if (haveW)
            sw(g, c) = w(b, c);
        }
        float share = 0.0f, pk = 0.0f;
        if (haveO) {
          share = shareFromBase ? orphan(b) : orphan(b) / static_cast<float>(gk(g));
          pk = orphanPeak(b);
        }
        for (int p = gs(g); p < gs(g + 1); ++p) {
          const int q = gq(p);
          if (q != b) {
            for (int c = 0; c < 3; ++c) {
              x(q, c) = sx(g, c) + gsh(p, c);
              if (haveW)
                w(q, c) = sw(g, c);
            }
          }
          if (haveO) {
            orphan(q) = share;
            orphanPeak(q) = pk;
          }
        }
      });
}

/// The local fold (§4.4), once per iteration after the sweep: per group
/// T = sum over the ACTIVE members of (x(q) - sigma - shift(q)) (the M-consensus of §1.3: an
/// inactive member carries no increment of its own, only the re-seeded consensus); the base takes
/// sigma + T / k and every member is re-seeded to it (plus its shift). w likewise. Orphan accounts
/// (Poisson): every member takes (sum of the active members' accounts) / k, the peak their max.
/// With `consensus` (§12 S14) the largest correction the fold applies to an active member's x,
/// |T / k - (x(q) - sigma - shift(q))| (increment form: exactly 0 once the sweep stops moving the
/// copies), is atomic-maxed into it; w, orphan and the result are unaffected.
inline void foldCopiesKokkos(CpExec space, const PhaseCopies& H, Kokkos::View<float* [3], CpMem> x,
                             Kokkos::View<float* [3], CpMem> w, Kokkos::View<float*, CpMem> orphan,
                             Kokkos::View<float*, CpMem> orphanPeak,
                             Kokkos::View<float, CpMem> consensus = {}) {
  if (H.nGroups <= 0)
    return;
  auto gs = H.groupStart;
  auto gq = H.groupSlot;
  auto gsh = H.groupShift;
  auto gk = H.groupK;
  auto ga = H.groupActive;
  auto sx = H.seedX;
  auto sw = H.seedW;
  const bool haveW = w.extent(0) > 0;
  const bool haveO = orphan.extent(0) > 0;
  const bool haveC = consensus.data() != nullptr;
  Kokkos::parallel_for(
      "peclet::dem::copies_fold", Kokkos::RangePolicy<CpExec>(space, 0, H.nGroups),
      KOKKOS_LAMBDA(int g) {
        const float kf = static_cast<float>(gk(g));
        float tx[3] = {0.0f, 0.0f, 0.0f}, tw[3] = {0.0f, 0.0f, 0.0f};
        float oSum = 0.0f, oPk = 0.0f;
        for (int p = gs(g); p < gs(g + 1); ++p) {
          if (!ga(p))
            continue;
          const int q = gq(p);
          for (int c = 0; c < 3; ++c) {
            tx[c] += x(q, c) - (sx(g, c) + gsh(p, c));
            if (haveW)
              tw[c] += w(q, c) - sw(g, c);
          }
          if (haveO) {
            oSum += orphan(q);
            oPk = Kokkos::fmax(oPk, orphanPeak(q));
          }
        }
        float nx[3], nw[3];
        for (int c = 0; c < 3; ++c) {
#if defined(PECLET_DEM_TEST_MUTANT) && PECLET_DEM_TEST_MUTANT == 2
          nx[c] = sx(g, c) + tx[c];  // G13 mutant 2: raw sum of split-mass increments (no 1/k)
          nw[c] = haveW ? sw(g, c) + tw[c] : 0.0f;
#else
          nx[c] = sx(g, c) + tx[c] / kf;
          nw[c] = haveW ? sw(g, c) + tw[c] / kf : 0.0f;
#endif
        }
        const float share = oSum / kf;
        if (haveC) {
          float cmax = 0.0f;
          for (int p = gs(g); p < gs(g + 1); ++p) {
            if (!ga(p))
              continue;
            const int q = gq(p);
            float s2 = 0.0f;
            for (int c = 0; c < 3; ++c) {
              const float e = tx[c] / kf - (x(q, c) - (sx(g, c) + gsh(p, c)));
              s2 += e * e;
            }
            cmax = Kokkos::fmax(cmax, Kokkos::sqrt(s2));
          }
          if (cmax > 0.0f)
            Kokkos::atomic_max(&consensus(), cmax);
        }
        for (int p = gs(g); p < gs(g + 1); ++p) {
          const int q = gq(p);
          for (int c = 0; c < 3; ++c) {
            x(q, c) = nx[c] + gsh(p, c);
            if (haveW)
              w(q, c) = nw[c];
          }
          if (haveO) {
            orphan(q) = share;
            orphanPeak(q) = oPk;
          }
        }
      });
}

/// Re-seed every member from its base, sigma unchanged (the multilevel pass, after the coarse
/// cycle has moved the bases; §4.4 "copies re-seeded again after the coarse cycle").
inline void reseedCopiesKokkos(CpExec space, const PhaseCopies& H,
                               Kokkos::View<float* [3], CpMem> x,
                               Kokkos::View<float* [3], CpMem> w) {
  if (H.nGroups <= 0)
    return;
  auto gs = H.groupStart;
  auto gq = H.groupSlot;
  auto gsh = H.groupShift;
  Kokkos::parallel_for(
      "peclet::dem::copies_reseed", Kokkos::RangePolicy<CpExec>(space, 0, H.nGroups),
      KOKKOS_LAMBDA(int g) {
        const int b = gq(gs(g));
        for (int p = gs(g) + 1; p < gs(g + 1); ++p) {
          const int q = gq(p);
          for (int c = 0; c < 3; ++c) {
            x(q, c) = x(b, c) + gsh(p, c);
            w(q, c) = w(b, c);
          }
        }
      });
}

/// Orphan accounts back from share form (before a rank sync and at the end of the velocity
/// phase): the base holds the body's balance, the sum of its members' shares.
inline void unfoldOrphanKokkos(const PhaseCopies& H, Kokkos::View<float*, CpMem> orphan) {
  if (H.nGroups <= 0 || orphan.extent(0) == 0)
    return;
  auto gs = H.groupStart;
  auto gq = H.groupSlot;
  auto ga = H.groupActive;
  Kokkos::parallel_for(
      "peclet::dem::copies_unfold", Kokkos::RangePolicy<CpExec>(0, H.nGroups),
      KOKKOS_LAMBDA(int g) {
        float s = 0.0f;
        for (int p = gs(g); p < gs(g + 1); ++p)
          if (ga(p))
            s += orphan(gq(p));
        orphan(gq(gs(g))) = s;
      });
  Kokkos::fence();
}

/// Rank-level activity pass (docs/contact_solve_framework.md §13.3 C4; the distributed step
/// only), velocity phase: hit(slot) = 1 for every slot that is an end -- through the hub-copy
/// overrides, else realIdx -- of an owned manifold of colour >= 0 in the final colouring. A
/// same-value store, so the order of the writers does not matter.
inline void activityHitVelocityKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds, int nm,
                                      Kokkos::View<const int*, CpMem> realIdx,
                                      Kokkos::View<const int*, CpMem> color, SlotOverride ov,
                                      Kokkos::View<unsigned char*, CpMem> hit, int nSlots) {
  CpExec space;
  Kokkos::deep_copy(space, Kokkos::subview(hit, Kokkos::pair<int, int>(0, nSlots)),
                    static_cast<unsigned char>(0));
  Kokkos::parallel_for(
      "peclet::dem::activity_vel", Kokkos::RangePolicy<CpExec>(space, 0, nm), KOKKOS_LAMBDA(int e) {
        if (color(e) < 0)
          return;
        const ManifoldC m = manifolds(e);
        Kokkos::atomic_store(&hit(ov.slotA(e, realIdx(m.bodyA))), static_cast<unsigned char>(1));
        if (m.bodyB >= 0)
          Kokkos::atomic_store(&hit(ov.slotB(e, realIdx(m.bodyB))), static_cast<unsigned char>(1));
      });
  space.fence();
}
/// The same over the position phase's units (raw slots of the leader contact, through the
/// overrides).
inline void activityHitPositionKokkos(Kokkos::View<const ContactC*, CpMem> contacts,
                                      const PosUnits& units, int nu,
                                      Kokkos::View<const int*, CpMem> color, SlotOverride ov,
                                      Kokkos::View<unsigned char*, CpMem> hit, int nSlots) {
  CpExec space;
  Kokkos::deep_copy(space, Kokkos::subview(hit, Kokkos::pair<int, int>(0, nSlots)),
                    static_cast<unsigned char>(0));
  Kokkos::parallel_for(
      "peclet::dem::activity_pos", Kokkos::RangePolicy<CpExec>(space, 0, nu), KOKKOS_LAMBDA(int u) {
        if (color(u) < 0)
          return;
        const ContactC c = contacts(units.leader(u));
        Kokkos::atomic_store(&hit(ov.slotA(u, c.bodyA)), static_cast<unsigned char>(1));
        if (c.bodyB >= 0)
          Kokkos::atomic_store(&hit(ov.slotB(u, c.bodyB)), static_cast<unsigned char>(1));
      });
  space.fence();
}
/// a(q) from the pass (§13.3): hit(q) for a slot outside any group; at a hub base the number of
/// its group's members (base + copies) that were hit, which also refills groupActive and groupK
/// (floored at 1, WO-4's local fold divisor). Copy slots get 0 (they are never packed; their
/// solve mass takes the base's k). a is written on [0, nSlots + H.nCopies).
inline void finishActivityKokkos(Kokkos::View<const unsigned char*, CpMem> hit, int nSlots,
                                 PhaseCopies& H, Kokkos::View<int*, CpMem> a) {
  CpExec space;
  const int nAll = nSlots + (H.nGroups > 0 ? H.nCopies : 0);
  Kokkos::parallel_for(
      "peclet::dem::activity_a", Kokkos::RangePolicy<CpExec>(space, 0, nAll),
      KOKKOS_LAMBDA(int q) { a(q) = q < nSlots ? static_cast<int>(hit(q)) : 0; });
  if (H.nGroups > 0) {
    auto gs = H.groupStart;
    auto gq = H.groupSlot;
    auto ga = H.groupActive;
    auto gk = H.groupK;
    Kokkos::parallel_for(
        "peclet::dem::activity_groups", Kokkos::RangePolicy<CpExec>(space, 0, H.nGroups),
        KOKKOS_LAMBDA(int g) {
          int n = 0;
          for (int p = gs(g); p < gs(g + 1); ++p) {
            const unsigned char h = hit(gq(p));
            ga(p) = h;
            n += h;
          }
          gk(g) = n > 0 ? n : 1;
          a(gq(gs(g))) = n;
        });
  }
  space.fence();
}

/// Solve views under rank-level M (§13.3, §4.5): invMassSolve / invInertiaSolve = k x the body's
/// masses on every slot in [0, nSlots) (nSlots = slotBase + the phase's copies), with k the
/// slot's global active copy count kRank (owner-computed, forwarded); a hub copy takes its base's
/// k. splitSlot = (k > 1). k <= 1 keeps the true masses bit for bit.
inline void buildSolveViewsRankKKokkos(Particles& P, const PhaseCopies& H, int nSlots,
                                       Kokkos::View<const float*, CpMem> invMass,
                                       Kokkos::View<const float* [3], CpMem> invInertia,
                                       Kokkos::View<const int*, CpMem> kRank) {
  CpExec space;
  growCopyView(P.invMassSolve, static_cast<std::size_t>(nSlots), "peclet::dem::invMassSolve");
  growCopyView(P.invInertiaSolve, static_cast<std::size_t>(nSlots), "peclet::dem::invInertiaSolve");
  growCopyView(P.splitSlot, static_cast<std::size_t>(nSlots), "peclet::dem::splitSlot");
  auto ims = P.invMassSolve;
  auto iis = P.invInertiaSolve;
  auto spl = P.splitSlot;
  auto cb = H.copyBase;
  const int base = H.slotBase;
  const bool haveCopies = H.nCopies > 0;
  Kokkos::parallel_for(
      "peclet::dem::rankk_solve_views", Kokkos::RangePolicy<CpExec>(space, 0, nSlots),
      KOKKOS_LAMBDA(int q) {
        const int b = (haveCopies && q >= base) ? cb(q - base) : q;
        const int k = kRank(b);
        if (k > 1) {
          const float kf = static_cast<float>(k);
          ims(q) = kf * invMass(b);
          for (int c = 0; c < 3; ++c)
            iis(q, c) = kf * invInertia(b, c);
          spl(q) = 1;
        } else {
          ims(q) = invMass(b);
          for (int c = 0; c < 3; ++c)
            iis(q, c) = invInertia(b, c);
          spl(q) = 0;
        }
      });
  space.fence();
}

/// The multilevel coarse vertex's inverse mass (§13.2): invMass(q) (k(q) / max(1, a(q))), the
/// ratio computed first so that k = a gives invMass bit for bit.
inline void buildInvMassCoarseKokkos(Kokkos::View<const float*, CpMem> invMass,
                                     Kokkos::View<const int*, CpMem> k,
                                     Kokkos::View<const int*, CpMem> a, int n,
                                     Kokkos::View<float*, CpMem> out) {
  Kokkos::parallel_for(
      "peclet::dem::inv_mass_coarse", Kokkos::RangePolicy<CpExec>(0, n), KOKKOS_LAMBDA(int q) {
        const int aq = a(q) > 1 ? a(q) : 1;
        const float ratio = static_cast<float>(k(q)) / static_cast<float>(aq);
        out(q) = invMass(q) * ratio;
      });
  Kokkos::fence();
}

/// Debug-build check (§13.2, §13.5 WO-5 item 4): the vertices with a = 0 that sit in a level-1
/// multilevel group of >= 2 members (must be none: an inactive copy never joins a coarse edge).
inline int countInactiveAggregatedKokkos(Kokkos::View<const int*, CpMem> parent, int off, int nV,
                                         int nGroups1, Kokkos::View<const int*, CpMem> a) {
  if (nV <= 0 || nGroups1 <= 0)
    return 0;
  Kokkos::View<int*, CpMem> cnt("peclet::dem::ml_inact_cnt", nGroups1);
  Kokkos::parallel_for(
      "peclet::dem::ml_inact_hist", Kokkos::RangePolicy<CpExec>(0, nV),
      KOKKOS_LAMBDA(int v) { Kokkos::atomic_add(&cnt(parent(off + v)), 1); });
  int n = 0;
  Kokkos::parallel_reduce(
      "peclet::dem::ml_inact_count", Kokkos::RangePolicy<CpExec>(0, nV),
      KOKKOS_LAMBDA(int v, int& acc) {
        if (a(v) == 0 && cnt(parent(off + v)) >= 2)
          acc += 1;
      },
      n);
  return n;
}

/// Count the bodies whose group has k > 1 (split_stats).
inline int countSplitBodiesKokkos(const PhaseCopies& H) {
  if (H.nGroups <= 0)
    return 0;
  auto gk = H.groupK;
  int n = 0;
  Kokkos::parallel_reduce(
      "peclet::dem::copies_count_split", Kokkos::RangePolicy<CpExec>(0, H.nGroups),
      KOKKOS_LAMBDA(int g, int& acc) { acc += gk(g) > 1 ? 1 : 0; }, n);
  return n;
}

/// split_stats.mlHubAggregated (§13.2's positive control): the vertices v in [0, nV) with
/// split(v) != 0 (k > 1) whose level-1 multilevel group -- parent(off + v), the first level's
/// parent map -- has at least two members. Diagnostic only (reads the hierarchy, writes nothing
/// the solve reads).
inline int countSplitAggregatedKokkos(Kokkos::View<const int*, CpMem> parent, int off, int nV,
                                      int nGroups1,
                                      Kokkos::View<const unsigned char*, CpMem> split) {
  if (nV <= 0 || nGroups1 <= 0 || split.extent(0) < static_cast<std::size_t>(nV))
    return 0;
  Kokkos::View<int*, CpMem> cnt("peclet::dem::ml_agg_cnt", nGroups1);
  Kokkos::parallel_for(
      "peclet::dem::ml_agg_hist", Kokkos::RangePolicy<CpExec>(0, nV),
      KOKKOS_LAMBDA(int v) { Kokkos::atomic_add(&cnt(parent(off + v)), 1); });
  int n = 0;
  Kokkos::parallel_reduce(
      "peclet::dem::ml_agg_count", Kokkos::RangePolicy<CpExec>(0, nV),
      KOKKOS_LAMBDA(int v, int& acc) {
        if (split(v) != 0 && cnt(parent(off + v)) >= 2)
          acc += 1;
      },
      n);
  return n;
}

/// Light hubs (split_stats, R-P3): velocity hubs whose mass is below 10x the mean mass of their
/// partners over the hub's active edges.
inline int countLightHubsKokkos(const PhaseCopies& H, Kokkos::View<const int*, CpMem> eA,
                                Kokkos::View<const int*, CpMem> eB, int nE, int nV,
                                Kokkos::View<const float*, CpMem> invMass) {
  if (H.nHubs <= 0)
    return 0;
  const int nH = H.nHubs;
  auto hv = H.hubVertex;
  Kokkos::View<int*, CpMem> hubOf("peclet::dem::lh_hubof", nV);
  Kokkos::deep_copy(hubOf, -1);
  Kokkos::parallel_for(
      "peclet::dem::lh_map", Kokkos::RangePolicy<CpExec>(0, nH),
      KOKKOS_LAMBDA(int h) { hubOf(hv(h)) = h; });
  Kokkos::View<double*, CpMem> msum("peclet::dem::lh_msum", nH);
  Kokkos::View<int*, CpMem> mcnt("peclet::dem::lh_mcnt", nH);
  Kokkos::parallel_for(
      "peclet::dem::lh_accum", Kokkos::RangePolicy<CpExec>(0, nE), KOKKOS_LAMBDA(int e) {
        const int a = eA(e), b = eB(e);
        if (a < 0 || b < 0)
          return;
        const int ends[2][2] = {{a, b}, {b, a}};
        for (int j = 0; j < 2; ++j) {
          const int h = hubOf(ends[j][0]);
          const float im = invMass(ends[j][1]);
          if (h >= 0 && im > 0.0f) {
            Kokkos::atomic_add(&msum(h), 1.0 / static_cast<double>(im));
            Kokkos::atomic_add(&mcnt(h), 1);
          }
        }
      });
  int n = 0;
  Kokkos::parallel_reduce(
      "peclet::dem::lh_count", Kokkos::RangePolicy<CpExec>(0, nH),
      KOKKOS_LAMBDA(int h, int& acc) {
        const float im = invMass(hv(h));
        if (mcnt(h) == 0 || im <= 0.0f)
          return;
        const double mHub = 1.0 / static_cast<double>(im);
        if (mHub < 10.0 * msum(h) / static_cast<double>(mcnt(h)))
          acc += 1;
      },
      n);
  return n;
}

}  // namespace peclet::dem

#endif  // DEM_SOLVE_COPIES_HPP
