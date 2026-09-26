/// @file
/// @brief dem — portable (Kokkos) contact->manifold reduction, replacing the thrust-based
/// reduce_contacts_to_manifolds() in contact_preprocessing.cu.
///
/// Same pipeline: key each contact by its canonical pair, group by key, and within a group sum the
/// aligned normals / torque arms / lever arms and count the points. thrust::sort_by_key +
/// reduce_by_key become Kokkos::Experimental::sort_by_key + a scan-based segmented reduction
/// (atomic accumulation), so no thrust/cub. The per-contact math (ContactToManifold /
/// TransformAndFilter) is reused verbatim as KOKKOS_INLINE_FUNCTION and is shared with the host
/// reference in the test.
///
/// One intentional change vs the thrust version: a manifold's (bodyA,bodyB) is decoded determinist
/// -ically from the pair key (canonical min/max, or (idA,-1) for boundary), rather than taken from
/// the first contact of an unstably-sorted run. The summed quantities are commutative, so they are
/// unaffected. Decoupled from ParticleSystemData (portable POD mirrors) for standalone validation.
#ifndef DEM_CONTACT_PREPROCESSING_HPP
#define DEM_CONTACT_PREPROCESSING_HPP

#include <cstddef>
#include <cstdint>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <utility>

#include "dem_portable.hpp"  // F4, cross3

namespace peclet::dem {

using CpExec = Kokkos::DefaultExecutionSpace;
using CpMem = CpExec::memory_space;

/// Portable mirror of ParticleSystem.cuh ContactConstraint (the fields this reduction touches).
struct ContactC {
  int bodyA;
  int bodyB;   // < 0 => boundary/static: -1 - wallIndex (wallContactId)
  F4 normal;   // .xyz = world normal
  F4 rA;       // lever arm on A
  F4 rB;       // lever arm on B
  float dist;  // signed penetration (>0 => inactive)
  float friction_lambda_n;
  float weight;
  // --- moving / per-material boundary (idB<0) extension; sentinels for body-body & static planes
  // so the solvers fall back to the global material and a zero wall velocity (unchanged behaviour).
  // ---
  F4 boundaryVel{0.0f, 0.0f, 0.0f, 0.0f};  // wall surface velocity at the contact point (xyz)
  float boundaryRestitution{-1.0f};        // per-wall normal restitution; < 0 => use the global one
  float boundaryFriction{-1.0f};           // per-wall Coulomb friction;   < 0 => use the global one
};

/// Wall identity of a boundary contact (docs/contact_solve_framework.md §12 S6): a contact with a
/// wall carries bodyB = -1 - wallIndex, where wallIndex runs over the analytic planes first
/// ([0, numPlanes)) and then the SDF walls (numPlanes + w). Every "is a wall" test is bodyB < 0;
/// the index only separates the position units of one body against different walls. The manifold
/// reduction key (pairKey) ignores it: all walls of a body still form ONE velocity manifold.
KOKKOS_INLINE_FUNCTION constexpr int wallContactId(int wallIndex) {
  return -1 - wallIndex;
}
KOKKOS_INLINE_FUNCTION constexpr int wallIndexOf(int bodyB) {
  return -1 - bodyB;
}

/// Portable mirror of ManifoldConstraint.
struct ManifoldC {
  int bodyA;
  int bodyB;
  F4 normal_sum;
  F4 torque_armA_sum;
  F4 torque_armB_sum;
  F4 rA_sum;
  F4 rB_sum;
  int num_points;
  // Σ over the manifold's active contacts of the boundary (idB<0) extension; the velocity solve
  // averages by num_points. wallVel_sum -> the wall's surface velocity seen by this contact patch;
  // restitution_sum -> per-wall restitution (a < 0 average keeps the global material).
  F4 wallVel_sum{0.0f, 0.0f, 0.0f, 0.0f};
  float restitution_sum{0.0f};
  float friction_sum{0.0f};  // per-contact mu (pair table / wall); a < 0 average = global material
};

/// The one 64-bit pair identity every gid-keyed ledger uses (the XPBD persistent-contact and
/// warm-start ledgers, the Hertz–Mindlin history): `(min(a, b) << 32) | max(a, b)`, symmetric in
/// (a, b) so a pair is one key whichever body is A. `a`/`b` are whatever identity is stable for
/// the run — the real index single-rank, the global id under MPI.
KOKKOS_INLINE_FUNCTION unsigned long long pairKeyFromGids(unsigned a, unsigned b) {
  const unsigned hi = a < b ? a : b, lo = a < b ? b : a;
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}

/// lower_bound over a ledger's sorted keys: the first index whose key is >= `key` (`count` when
/// none). A hit is `i < count && keys(i) == key`; the index doubles as the ledger slot of the
/// carried payload.
template <class KeysView>
KOKKOS_INLINE_FUNCTION int lowerBoundKey(const KeysView& keys, int count, unsigned long long key) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (keys(mid) < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

/// Persistent-contact detection for the gravity-gated restitution rule. Key = pairKeyFromGids over
/// (real body A, real body B); boundary manifolds (bodyB = -1: planes + SDF walls, merged per
/// particle) use 0xFFFFFFFF for B. Flags each manifold whose pair already existed in the PREVIOUS
/// substep (prevKeys sorted, device binary search). Real indices are stable within a single-GPU
/// run, so the key identifies the physical pair across substeps.
KOKKOS_INLINE_FUNCTION unsigned long long pairKeyOf(const ManifoldC& m,
                                                    Kokkos::View<const int*, CpMem> realIdx) {
  const unsigned a = static_cast<unsigned>(realIdx(m.bodyA));
  const unsigned b = (m.bodyB >= 0) ? static_cast<unsigned>(realIdx(m.bodyB)) : 0xFFFFFFFFu;
  return pairKeyFromGids(a, b);
}

/// `keyIdx` maps a body slot to the identity the pair key is built from: the REAL index map on the
/// single-GPU path (stable within a run), or the GLOBAL particle id under MPI (stable across ranks,
/// halo rebuilds and ownership migration — local slots are neither).
inline void markPersistentManifoldsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                          int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                          Kokkos::View<const int*, CpMem> keyIdx,
                                          Kokkos::View<const unsigned long long*, CpMem> prevKeys,
                                          int prevCount,
                                          Kokkos::View<unsigned long long*, CpMem> outKeys,
                                          Kokkos::View<unsigned char*, CpMem> outFlags) {
  (void)realIdx;
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::mark_persistent", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        if (m.num_points <= 0) {
          outKeys(idx) = ~0ull;  // hi = 0xFFFFFFFF: can never match a live manifold's key
          outFlags(idx) = 0;
          return;
        }
        const unsigned long long k = pairKeyOf(m, keyIdx);
        outKeys(idx) = k;
        const int lo = lowerBoundKey(prevKeys, prevCount, k);
        outFlags(idx) = (lo < prevCount && prevKeys(lo) == k) ? 1 : 0;
      });
}

/// Warm-start gather for the PGS velocity solve: per manifold, write its pair key and look up the
/// previous substep's converged push impulse (0 for a new contact). Periodic-ghost duplicate
/// manifolds (realA > realB twin) get key ~0 and warm 0 -- the canonical twin carries the impulse.
/// `keyIdx`: see markPersistentManifoldsKokkos (realIdx keeps the periodic-dedup role; keyIdx
/// builds the cross-substep pair identity).
inline void gatherWarmLambdaKokkos(
    Kokkos::View<const ManifoldC*, CpMem> manifolds, int numManifolds,
    Kokkos::View<const int*, CpMem> realIdx, Kokkos::View<const int*, CpMem> keyIdx,
    Kokkos::View<const unsigned long long*, CpMem> prevKeys,
    Kokkos::View<const float*, CpMem> prevLambda, Kokkos::View<const float* [3], CpMem> prevLambdaT,
    Kokkos::View<const float*, CpMem> prevPosImpulse,
    Kokkos::View<const float*, CpMem> prevRestBank, Kokkos::View<const float*, CpMem> prevRestVPeak,
    int prevCount, Kokkos::View<unsigned long long*, CpMem> outKeys,
    Kokkos::View<float*, CpMem> outWarm, Kokkos::View<float* [3], CpMem> outWarmT,
    Kokkos::View<float*, CpMem> outPosImpulse, Kokkos::View<float*, CpMem> outRestBank,
    Kokkos::View<float*, CpMem> outRestVPeak, Kokkos::View<unsigned char*, CpMem> outMatched = {},
    bool dedupTwins = true) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::gather_warm", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        bool dup = false;
        if (dedupTwins && m.num_points > 0 && m.bodyB >= 0 && realIdx(m.bodyA) > realIdx(m.bodyB))
          dup = true;  // periodic dedup (single rank; §4.2 item 2)
        if (m.num_points <= 0 || dup) {
          outKeys(idx) = ~0ull;
          outWarm(idx) = 0.0f;
          outWarmT(idx, 0) = outWarmT(idx, 1) = outWarmT(idx, 2) = 0.0f;
          outPosImpulse(idx) = 0.0f;
          outRestBank(idx) = 0.0f;
          outRestVPeak(idx) = 0.0f;
          return;
        }
        const unsigned long long k = pairKeyOf(m, keyIdx);
        outKeys(idx) = k;
        const int lo = lowerBoundKey(prevKeys, prevCount, k);
        const bool hit = (lo < prevCount && prevKeys(lo) == k);
        if (hit && outMatched.extent(0) > 0)
          outMatched(lo) = 1;  // prev entry survives; unmatched entries orphan their bank
        outWarm(idx) = hit ? prevLambda(lo) : 0.0f;
        outWarmT(idx, 0) = hit ? prevLambdaT(lo, 0) : 0.0f;
        outWarmT(idx, 1) = hit ? prevLambdaT(lo, 1) : 0.0f;
        outWarmT(idx, 2) = hit ? prevLambdaT(lo, 2) : 0.0f;
        outPosImpulse(idx) = hit ? prevPosImpulse(lo) : 0.0f;
        outRestBank(idx) = hit ? prevRestBank(lo) : 0.0f;
        outRestVPeak(idx) = hit ? prevRestVPeak(lo) : 0.0f;
      });
}

/// Save this substep's keys + converged impulses (normal AND tangential) and key-sort them for
/// next substep's gather. A permutation sort carries both value arrays through one key sort.
inline void commitPairKeysLambdaKokkos(
    Kokkos::View<const unsigned long long*, CpMem> keys, Kokkos::View<const float*, CpMem> lambda,
    Kokkos::View<const float* [3], CpMem> lambdaT, Kokkos::View<const float*, CpMem> restBank,
    Kokkos::View<const float*, CpMem> restVPeak, Kokkos::View<unsigned long long*, CpMem> prevKeys,
    Kokkos::View<float*, CpMem> prevLambda, Kokkos::View<float* [3], CpMem> prevLambdaT,
    Kokkos::View<float*, CpMem> prevRestBank, Kokkos::View<float*, CpMem> prevRestVPeak,
    Kokkos::View<int*, CpMem> perm,  // pooled scratch, >= n
    int numManifolds,
    // Optional: carry the per-manifold colour by pair key too
    // (single-GPU incremental colouring; empty views = skip).
    Kokkos::View<const int*, CpMem> color = {}, Kokkos::View<int*, CpMem> prevColor = {}) {
  if (numManifolds <= 0)
    return;
  CpExec space;
  const int n = numManifolds;
  const auto rng = Kokkos::pair<int, int>(0, n);
  auto kd = Kokkos::subview(prevKeys, rng);
  Kokkos::deep_copy(space, kd, Kokkos::subview(keys, rng));
  Kokkos::parallel_for(
      "peclet::dem::commit_iota", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { perm(i) = i; });
  {
    auto pd = Kokkos::subview(perm, rng);
    Kokkos::Experimental::sort_by_key(space, kd, pd);
  }
  Kokkos::View<float*, CpMem> pl = prevLambda;
  Kokkos::View<float* [3], CpMem> plt = prevLambdaT;
  Kokkos::View<float*, CpMem> prb = prevRestBank;
  Kokkos::View<float*, CpMem> prv = prevRestVPeak;
  const bool carryColor = color.extent(0) > 0 && prevColor.extent(0) > 0;
  Kokkos::View<const int*, CpMem> col = color;
  Kokkos::View<int*, CpMem> pcol = prevColor;
  Kokkos::parallel_for(
      "peclet::dem::commit_gather", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        const int j = perm(i);
        pl(i) = lambda(j);
        plt(i, 0) = lambdaT(j, 0);
        plt(i, 1) = lambdaT(j, 1);
        plt(i, 2) = lambdaT(j, 2);
        prb(i) = restBank(j);
        prv(i) = restVPeak(j);
        if (carryColor)
          pcol(i) = col(j);
      });
  space.fence();
}

/// Copy this substep's keys into prevKeys and sort them for next substep's binary search.
inline void commitPairKeysKokkos(Kokkos::View<const unsigned long long*, CpMem> keys,
                                 Kokkos::View<unsigned long long*, CpMem> prevKeys,
                                 int numManifolds) {
  if (numManifolds <= 0)
    return;
  CpExec space;
  auto src = Kokkos::subview(keys, Kokkos::pair<int, int>(0, numManifolds));
  auto dst = Kokkos::subview(prevKeys, Kokkos::pair<int, int>(0, numManifolds));
  Kokkos::deep_copy(space, dst, src);
  Kokkos::sort(space, dst);
  space.fence();
}

/// Guendelman support levels, warm-started: decay every body's level by `decay`, re-seed 255 at
/// wall/plane contacts, then `sweeps` monotone propagation passes lower -> upper (255 -> 254 ->
/// ...) through the manifold graph. Warm start makes a few sweeps per substep track slowly-moving
/// support fronts; the decay retires groundedness ~32 substeps after lift-off. Geometry-only (no
/// persistence / velocity condition): grounded means "has a contact path down to the floor".
inline void updateGroundedLevelsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                       int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                       Kokkos::View<const float* [3], CpMem> posPred, F3 gHat,
                                       Kokkos::View<unsigned char*, CpMem> grounded, int numReal,
                                       int sweeps, int decay) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::grounded_decay", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const int g = static_cast<int>(grounded(i)) - decay;
        grounded(i) = static_cast<unsigned char>(g > 0 ? g : 0);
      });
  for (int s = 0; s < sweeps; ++s) {
    Kokkos::parallel_for(
        "peclet::dem::grounded_sweep", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx) {
          const ManifoldC m = manifolds(idx);
          if (m.num_points <= 0)
            return;
          const int realA = realIdx(m.bodyA);
          if (m.bodyB < 0) {
            Kokkos::atomic_max(&grounded(realA), static_cast<unsigned char>(255));
            return;
          }
          const int realB = realIdx(m.bodyB);
          const F3 dx = sub3(ldF3(posPred, m.bodyA), ldF3(posPred, m.bodyB));
          const float up = -(dx.x * gHat.x + dx.y * gHat.y + dx.z * gHat.z);  // >0: A above B
          const float thr = 0.3f * Kokkos::sqrt(dot3(dx, dx));
          if (up > thr) {  // B supports A
            const int lvl = static_cast<int>(grounded(realB)) - 1;
            if (lvl > 0)
              Kokkos::atomic_max(&grounded(realA), static_cast<unsigned char>(lvl));
          } else if (up < -thr) {  // A supports B
            const int lvl = static_cast<int>(grounded(realA)) - 1;
            if (lvl > 0)
              Kokkos::atomic_max(&grounded(realB), static_cast<unsigned char>(lvl));
          }
        });
  }
}

/// Height-from-floor BFS levels for the level-ordered ("multilevel") stabilization pass: 0 at a
/// wall/plane contact, else 1 + min over supporting contacts, kLevelInf with no contact path to
/// the floor. Recomputed FRESH each time the pass triggers -- no warm start and no decay: a
/// stale-low height would mis-order the sweeps, and the pass runs rarely enough that the exact
/// BFS (early exit once a sweep changes nothing) is affordable. The support-orientation test
/// (up vs 0.3|dx|) matches updateGroundedLevelsKokkos; ghost-slot positions keep the pair
/// geometry periodic-aware.
inline constexpr int kLevelInf = 1 << 28;
inline void computeHeightLevelsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                      int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                      Kokkos::View<const float* [3], CpMem> posPred, F3 gHat,
                                      Kokkos::View<int*, CpMem> heights, int numReal) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::height_init", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) { heights(i) = kLevelInf; });
  const int maxSweeps = 1024;  // >= deepest supported column; early exit ends real runs sooner
  for (int s = 0; s < maxSweeps; ++s) {
    int changed = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::height_sweep", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
        KOKKOS_LAMBDA(int idx, int& acc) {
          const ManifoldC m = manifolds(idx);
          if (m.num_points <= 0)
            return;
          const int realA = realIdx(m.bodyA);
          if (m.bodyB < 0) {
            if (Kokkos::atomic_fetch_min(&heights(realA), 0) > 0)
              acc += 1;
            return;
          }
          const int realB = realIdx(m.bodyB);
          const F3 dx = sub3(ldF3(posPred, m.bodyA), ldF3(posPred, m.bodyB));
          const float up = -(dx.x * gHat.x + dx.y * gHat.y + dx.z * gHat.z);  // >0: A above B
          const float thr = 0.3f * Kokkos::sqrt(dot3(dx, dx));
          if (up > thr) {  // B supports A
            const int cand = heights(realB);
            if (cand < kLevelInf && Kokkos::atomic_fetch_min(&heights(realA), cand + 1) > cand + 1)
              acc += 1;
          } else if (up < -thr) {  // A supports B
            const int cand = heights(realA);
            if (cand < kLevelInf && Kokkos::atomic_fetch_min(&heights(realB), cand + 1) > cand + 1)
              acc += 1;
          }
        },
        changed);
    if (changed == 0)
      break;
  }
  space.fence();
}

/// Dense colour buckets (numColors <= 64): perm[offs[c] .. offs[c+1]) lists the item indices of
/// colour c. Histogram + host prefix + scatter — two small kernels and one 64-int readback,
/// amortized over every sweep that would otherwise scan ALL n items per colour per iteration
/// (measured 25k multilevel: the per-colour full scans were 40% of GPU time and the step is
/// host-submission-bound). Within a colour the scatter order is arbitrary — colour classes are
/// body-disjoint, so the sweep result is bit-identical. Items with colour < 0 (inactive, dups,
/// mask-saturation leftovers) are excluded, exactly like the scan-mode colour filter.
inline void buildColorBucketsKokkos(Kokkos::View<const int*, CpMem> colorOf, int n, int numColors,
                                    Kokkos::View<int*, CpMem> perm,
                                    Kokkos::View<int*, CpMem> cursor,  // scratch, >= 64 ints
                                    std::vector<int>& offs) {
  offs.assign(static_cast<std::size_t>(numColors) + 1, 0);
  if (n <= 0 || numColors <= 0)
    return;
  CpExec space;
  auto cnt = Kokkos::subview(cursor, Kokkos::pair<int, int>(0, numColors));
  Kokkos::deep_copy(space, cnt, 0);
  Kokkos::parallel_for(
      "peclet::dem::bucket_hist", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        const int c = colorOf(i);
        if (c >= 0)
          Kokkos::atomic_add(&cursor(c), 1);
      });
  auto hCnt = Kokkos::create_mirror_view(cnt);
  Kokkos::deep_copy(space, hCnt, cnt);
  space.fence();
  for (int c = 0; c < numColors; ++c)
    offs[static_cast<std::size_t>(c) + 1] = offs[static_cast<std::size_t>(c)] + hCnt(c);
  for (int c = 0; c < numColors; ++c)
    hCnt(c) = offs[static_cast<std::size_t>(c)];
  Kokkos::deep_copy(space, cnt, hCnt);
  Kokkos::parallel_for(
      "peclet::dem::bucket_scatter", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        const int c = colorOf(i);
        if (c >= 0)
          perm(Kokkos::atomic_fetch_add(&cursor(c), 1)) = i;
      });
}

/// splitmix32 finalizer: a well-mixed pseudo-random priority per edge index. Random priorities make
/// the Jones-Plassmann arbitration finish in O(log n) rounds w.h.p.; RAW indices are adversarial
/// for lattice-ordered dense packs (monotone index chains -> one win per round -> O(chain) rounds).
KOKKOS_INLINE_FUNCTION long long colorKey(int idx) {
  unsigned int z = static_cast<unsigned>(idx) + 0x9e3779b9u;
  z = (z ^ (z >> 16)) * 0x21f0aaadu;
  z = (z ^ (z >> 15)) * 0x735a2d97u;
  z ^= (z >> 15);
  // priority in the high word, unique index in the low word (unique key per edge; >= 0)
  return (static_cast<long long>(z & 0x7fffffffu) << 32) | static_cast<unsigned>(idx);
}

/// Canonical pair key: (min<<32)|max, or (idA<<32)|0xFFFFFFFF for a boundary (idB<0) contact.
KOKKOS_INLINE_FUNCTION std::uint64_t pairKey(const ContactC& c) {
  const int idA = c.bodyA, idB = c.bodyB;
  if (idB < 0)
    return (static_cast<std::uint64_t>(static_cast<unsigned>(idA)) << 32) | 0xFFFFFFFFu;
  const unsigned u = (idA < idB) ? idA : idB;
  const unsigned v = (idA < idB) ? idB : idA;
  return (static_cast<std::uint64_t>(u) << 32) | v;
}

/// Distributed step: bit 63 of a manifold-reduction key marks a contact this rank sees but does
/// not own (docs/mpi_momentum_conservation.md §2.2), so the sort puts the owned manifolds first.
/// Slots are < 2^31, so no single-rank key ever has it set.
inline constexpr std::uint64_t kNonOwnedPairBit = 1ull << 63;

/// Colouring palette (docs/contact_solve_framework.md §4.2 item 1): the greedy takes the lowest
/// colour free at both endpoints among bits 0..63 of the uint64 mask and never forces one. An edge
/// with no free colour is marked kColorUncolourable; it is neither coloured nor remaining, so the
/// arbitration terminates, and it counts as `leftover` (after hub copies: an invariant violation).
inline constexpr int kColorPalette = 64;
inline constexpr int kColorUncolourable = -3;

/// Per-edge slot overrides of a phase's hub copies (docs/contact_solve_framework.md §4.4 item 5)
/// plus the per-slot mass-split flag and relaxation of §4.5. Empty views = today's indices, no
/// relaxation (the identity). `a(e)` / `b(e)` are the slots of edge e's A / B end (-1 = the
/// default index: realIdx(bodyX) in the velocity phase, the raw contact body in the position
/// phase). Every STATE access of a phase's sweeps goes through slotA / slotB; keys, labels and side
/// flags keep the body.
struct SlotOverride {
  Kokkos::View<const int*, CpMem> a, b;
  Kokkos::View<const unsigned char*, CpMem> split;  // per slot: 1 = its body has k > 1
  float omega = 1.0f;                               // the phase's split relaxation (§4.5)
  KOKKOS_INLINE_FUNCTION int slotA(int e, int def) const {
    if (a.extent(0) > 0) {
      const int s = a(e);
      if (s >= 0)
        return s;
    }
    return def;
  }
  KOKKOS_INLINE_FUNCTION int slotB(int e, int def) const {
    if (b.extent(0) > 0) {
      const int s = b(e);
      if (s >= 0)
        return s;
    }
    return def;
  }
  /// True iff the edge touches a mass-split slot (its step is scaled by omega before the clamp).
  KOKKOS_INLINE_FUNCTION bool relax(int sA, int sB) const {
    return split.extent(0) > 0 && (split(sA) != 0 || (sB >= 0 && split(sB) != 0));
  }
};

/// Decode the canonical (bodyA, bodyB) from a pair key (bodyB = -1 for boundary). Bit 63
/// (kNonOwnedPairBit) is masked: the identity on every key below 2^63.
KOKKOS_INLINE_FUNCTION void decodeKey(std::uint64_t key, int& bodyA, int& bodyB) {
  const unsigned v = static_cast<unsigned>(key & 0xFFFFFFFFu);
  bodyA = static_cast<int>((key >> 32) & 0x7FFFFFFFu);
  bodyB = (v == 0xFFFFFFFFu) ? -1 : static_cast<int>(v);
}

/// Per-contact transform to a single-point manifold, aligned to the canonical pair. Inactive
/// contacts (dist > 0) contribute a zero manifold with num_points = 0. Mirrors ContactToManifold +
/// TransformAndFilter from contact_preprocessing.cu.
KOKKOS_INLINE_FUNCTION ManifoldC transformContact(const ContactC& c) {
  ManifoldC m{};
  const int idA = c.bodyA, idB = c.bodyB;
  if (c.dist > 0.0f) {
    m.num_points = 0;
    return m;  // inactive (sums already zero)
  }
  const bool flip = (idB >= 0 && idB < idA);
  m.num_points = 1;

  const F4 n_vec = c.normal;
  F4 n_aligned = flip ? n_vec : F4{-n_vec.x, -n_vec.y, -n_vec.z, 0.0f};

  const F4 shift{c.normal.x * c.dist * 0.5f, c.normal.y * c.dist * 0.5f, c.normal.z * c.dist * 0.5f,
                 0.0f};
  const F4 rA_mid{c.rA.x - shift.x, c.rA.y - shift.y, c.rA.z - shift.z, 0.0f};
  const F4 rB_mid{c.rB.x + shift.x, c.rB.y + shift.y, c.rB.z + shift.z, 0.0f};

  const F4 r_can = flip ? rB_mid : rA_mid;
  const F4 r_other = flip ? rA_mid : rB_mid;

  const F4 tau_can = cross3(r_can, n_aligned);
  const F4 tau_other = cross3(r_other, F4{-n_aligned.x, -n_aligned.y, -n_aligned.z, 0.0f});

  m.normal_sum = F4{n_aligned.x, n_aligned.y, n_aligned.z, 0.0f};
  m.torque_armA_sum = F4{tau_can.x, tau_can.y, tau_can.z, 0.0f};
  m.torque_armB_sum = F4{tau_other.x, tau_other.y, tau_other.z, 0.0f};
  m.rA_sum = F4{r_can.x, r_can.y, r_can.z, 0.0f};
  m.rB_sum = F4{r_other.x, r_other.y, r_other.z, 0.0f};
  // Boundary (idB<0) moving-wall extension: carry the wall velocity + per-wall restitution through
  // to the (count-averaged) velocity solve. Zero / -1 sentinel for body-body & static planes.
  m.wallVel_sum = F4{c.boundaryVel.x, c.boundaryVel.y, c.boundaryVel.z, 0.0f};
  m.restitution_sum = c.boundaryRestitution;
  m.friction_sum = c.boundaryFriction;
  return m;
}

/// Stable partition of contacts [0, n) by `owns(contact)`: the owned ones first, [0, ncOwned),
/// then the rest, each part in its old relative order (distributed step,
/// docs/mpi_momentum_conservation.md §2.2). One exclusive scan + one scatter into a scratch copy;
/// with every contact owned (np = 1 on a closed domain) the contacts are unchanged bit for bit.
/// Returns ncOwned.
template <class Owns>
inline int partitionContactsKokkos(Kokkos::View<ContactC*, CpMem> contacts, int n,
                                   const Owns& owns) {
  if (n == 0)
    return 0;
  CpExec space;
  Kokkos::View<int*, CpMem> before(
      Kokkos::view_alloc(space, "peclet::dem::cp::partBefore", Kokkos::WithoutInitializing), n);
  Kokkos::View<ContactC*, CpMem> ct = contacts;
  int ncOwned = 0;
  Kokkos::parallel_scan(
      "peclet::dem::cp::partScan", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i, int& run, const bool final) {
        const int f = owns(ct(i)) ? 1 : 0;
        if (final)
          before(i) = run;  // exclusive
        run += f;
      },
      ncOwned);
  Kokkos::View<ContactC*, CpMem> part(
      Kokkos::view_alloc(space, "peclet::dem::cp::part", Kokkos::WithoutInitializing), n);
  const int no = ncOwned;
  Kokkos::parallel_for(
      "peclet::dem::cp::partScatter", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        const bool f = owns(ct(i));
        part(f ? before(i) : no + (i - before(i))) = ct(i);
      });
  Kokkos::deep_copy(space,
                    Kokkos::subview(contacts, std::pair<std::size_t, std::size_t>(
                                                  0, static_cast<std::size_t>(n))),
                    part);
  space.fence();
  return ncOwned;
}

/// Reduce `n` contacts to manifolds (one per unique canonical pair). outManifolds must hold at
/// least the number of unique pairs; returns that count (also written to outCount).
///
/// Distributed step (docs/mpi_momentum_conservation.md §2.2): with `numOwnedContacts` >= 0 the
/// contacts at index >= numOwnedContacts (not owned by this rank, after partitionContactsKokkos)
/// get kNonOwnedPairBit in their key, so the sort puts the owned manifolds first, [0, nmOwned) in
/// their old relative order, and the non-owned ones after them; `*numOwnedManifolds` receives
/// nmOwned. The defaults reproduce the single-rank reduction exactly.
inline int reduceContactsToManifoldsKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int n,
                                           Kokkos::View<ManifoldC*, CpMem> outManifolds,
                                           Kokkos::View<int, CpMem> outCount,
                                           Kokkos::View<int*, CpMem> contactSlot = {},
                                           int numOwnedContacts = -1,
                                           int* numOwnedManifolds = nullptr) {
  CpExec space;
  if (n == 0) {
    Kokkos::deep_copy(space, outCount, 0);
    if (numOwnedManifolds)
      *numOwnedManifolds = 0;
    return 0;
  }

  // 1. Key every contact and seed an identity permutation.
  Kokkos::View<std::uint64_t*, CpMem> keys(
      Kokkos::view_alloc(space, "peclet::dem::cp::keys", Kokkos::WithoutInitializing), n);
  Kokkos::View<int*, CpMem> perm(
      Kokkos::view_alloc(space, "peclet::dem::cp::perm", Kokkos::WithoutInitializing), n);
  const int nOwnedC = numOwnedContacts;
  Kokkos::parallel_for(
      "peclet::dem::cp::key", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        keys(i) = pairKey(contacts(i)) |
                  ((nOwnedC >= 0 && i >= nOwnedC) ? kNonOwnedPairBit : std::uint64_t{0});
        perm(i) = i;
      });

  // 2. Sort the permutation by key (groups equal pairs contiguously).
  Kokkos::Experimental::sort_by_key(space, keys, perm);

  // 3. Segment id per sorted position: inclusive scan of "key changed" minus 1.
  Kokkos::View<int*, CpMem> segId(
      Kokkos::view_alloc(space, "peclet::dem::cp::segId", Kokkos::WithoutInitializing), n);
  int numSeg = 0;
  Kokkos::parallel_scan(
      "peclet::dem::cp::segscan", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int p, int& run, const bool final) {
        const bool isNew = (p == 0) || (keys(p) != keys(p - 1));
        if (isNew)
          ++run;
        if (final)
          segId(p) = run - 1;  // 0-based segment index
      },
      numSeg);
  if (numOwnedManifolds) {  // owned segment leaders (their keys carry no kNonOwnedPairBit)
    Kokkos::View<std::uint64_t*, CpMem> kk = keys;
    int nOwnedM = 0;
    Kokkos::parallel_reduce(
        "peclet::dem::cp::ownedSegs", Kokkos::RangePolicy<CpExec>(space, 0, n),
        KOKKOS_LAMBDA(int p, int& cnt) {
          const bool leader = (p == 0) || (kk(p) != kk(p - 1));
          if (leader && (kk(p) & kNonOwnedPairBit) == 0)
            ++cnt;
        },
        nOwnedM);
    *numOwnedManifolds = nOwnedM;
  }

  // 4. Initialise one manifold per segment (canonical ids from the key, sums zero).
  Kokkos::View<std::uint64_t*, CpMem> k = keys;
  Kokkos::View<int*, CpMem> sid = segId;
  Kokkos::View<ManifoldC*, CpMem> out = outManifolds;
  Kokkos::parallel_for(
      "peclet::dem::cp::init", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int p) {
        const bool leader = (p == 0) || (k(p) != k(p - 1));
        if (leader) {
          ManifoldC m{};
          decodeKey(k(p), m.bodyA, m.bodyB);
          out(sid(p)) = m;
        }
      });

  // 5. Accumulate each contact's transformed manifold into its segment (atomic; order-independent).
  Kokkos::View<const ContactC*, CpMem> ct = contacts;
  Kokkos::View<int*, CpMem> pm = perm;
  Kokkos::parallel_for(
      "peclet::dem::cp::accum", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int p) {
        const ManifoldC m = transformContact(ct(pm(p)));
        const int s = sid(p);
        Kokkos::atomic_add(&out(s).num_points, m.num_points);
        Kokkos::atomic_add(&out(s).normal_sum.x, m.normal_sum.x);
        Kokkos::atomic_add(&out(s).normal_sum.y, m.normal_sum.y);
        Kokkos::atomic_add(&out(s).normal_sum.z, m.normal_sum.z);
        Kokkos::atomic_add(&out(s).torque_armA_sum.x, m.torque_armA_sum.x);
        Kokkos::atomic_add(&out(s).torque_armA_sum.y, m.torque_armA_sum.y);
        Kokkos::atomic_add(&out(s).torque_armA_sum.z, m.torque_armA_sum.z);
        Kokkos::atomic_add(&out(s).torque_armB_sum.x, m.torque_armB_sum.x);
        Kokkos::atomic_add(&out(s).torque_armB_sum.y, m.torque_armB_sum.y);
        Kokkos::atomic_add(&out(s).torque_armB_sum.z, m.torque_armB_sum.z);
        Kokkos::atomic_add(&out(s).rA_sum.x, m.rA_sum.x);
        Kokkos::atomic_add(&out(s).rA_sum.y, m.rA_sum.y);
        Kokkos::atomic_add(&out(s).rA_sum.z, m.rA_sum.z);
        Kokkos::atomic_add(&out(s).rB_sum.x, m.rB_sum.x);
        Kokkos::atomic_add(&out(s).rB_sum.y, m.rB_sum.y);
        Kokkos::atomic_add(&out(s).rB_sum.z, m.rB_sum.z);
        Kokkos::atomic_add(&out(s).wallVel_sum.x, m.wallVel_sum.x);
        Kokkos::atomic_add(&out(s).wallVel_sum.y, m.wallVel_sum.y);
        Kokkos::atomic_add(&out(s).wallVel_sum.z, m.wallVel_sum.z);
        Kokkos::atomic_add(&out(s).restitution_sum, m.restitution_sum);
        Kokkos::atomic_add(&out(s).friction_sum, m.friction_sum);
      });
  // Optional contact -> manifold slot map (PGS friction bound reads lambdaAcc through it).
  if (contactSlot.extent(0) >= (size_t)n) {
    Kokkos::View<int*, CpMem> cs = contactSlot;
    Kokkos::parallel_for(
        "peclet::dem::cp::slotmap", Kokkos::RangePolicy<CpExec>(space, 0, n),
        KOKKOS_LAMBDA(int p) { cs(pm(p)) = sid(p); });
  }
  space.fence();

  Kokkos::deep_copy(space, outCount, numSeg);
  return numSeg;
}

/// Position-unit key (docs/contact_solve_framework.md §4.3, §12 S6): the canonical pair key for a
/// body-body contact, and (bodyA << 32) | (unsigned)bodyB for a wall contact, which separates the
/// walls of one body (bodyB = -1 - wallIndex, so the low word is 0xFFFFFFFF - wallIndex >= 2^31,
/// never a body slot). Wall 0 keeps pairKey's value; pairKey itself is unchanged.
KOKKOS_INLINE_FUNCTION std::uint64_t unitKey(const ContactC& c) {
  if (c.bodyB < 0)
    return (static_cast<std::uint64_t>(static_cast<unsigned>(c.bodyA)) << 32) |
           static_cast<unsigned>(c.bodyB);
  return pairKey(c);
}

/// The position units of a contact list as a CSR (§4.3): unit u owns the contacts
/// list(start(u)) .. list(start(u + 1) - 1), sorted ascending by contact index, so list(start(u))
/// is its leader (smallest contact index). Units are numbered by ascending leader. Empty views =
/// the identity (every contact its own unit), which the colourings and the sweep treat exactly as
/// the per-contact code they replace.
struct PosUnits {
  Kokkos::View<const int*, CpMem> start;  // numUnits + 1
  Kokkos::View<const int*, CpMem> list;   // numContacts
  KOKKOS_INLINE_FUNCTION bool on() const { return start.extent(0) > 0; }
  KOKKOS_INLINE_FUNCTION int begin(int u) const { return on() ? start(u) : u; }
  KOKKOS_INLINE_FUNCTION int end(int u) const { return on() ? start(u + 1) : u + 1; }
  KOKKOS_INLINE_FUNCTION int contact(int k) const { return on() ? list(k) : k; }
  KOKKOS_INLINE_FUNCTION int leader(int u) const { return on() ? list(start(u)) : u; }
};

/// Build the position units of contacts [0, n) (docs/contact_solve_framework.md §4.3, §12 S6): a
/// unit is the set of contacts of one body pair, or of one body and one wall. Two key sorts: by
/// unitKey to group (leader = the segment's smallest contact index), then by (leader, contact
/// index) for the canonical CSR. Returns the unit count; unitStart gets numUnits + 1 entries.
inline int buildPositionUnitsKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int n,
                                    Kokkos::View<int*, CpMem> unitStart,
                                    Kokkos::View<int*, CpMem> unitContacts) {
  CpExec space;
  if (n <= 0) {
    Kokkos::deep_copy(space, Kokkos::subview(unitStart, 0), 0);
    space.fence();
    return 0;
  }
  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  Kokkos::View<std::uint64_t*, CpMem> keys(
      view_alloc(space, "peclet::dem::cp::ukeys", WithoutInitializing), n);
  Kokkos::View<int*, CpMem> perm(view_alloc(space, "peclet::dem::cp::uperm", WithoutInitializing),
                                 n);
  Kokkos::parallel_for(
      "peclet::dem::cp::ukey", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        keys(i) = unitKey(contacts(i));
        perm(i) = i;
      });
  Kokkos::Experimental::sort_by_key(space, keys, perm);
  Kokkos::View<int*, CpMem> segId(view_alloc(space, "peclet::dem::cp::useg", WithoutInitializing),
                                  n);
  int numSeg = 0;
  Kokkos::parallel_scan(
      "peclet::dem::cp::usegscan", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int p, int& run, const bool final) {
        if (p == 0 || keys(p) != keys(p - 1))
          ++run;
        if (final)
          segId(p) = run - 1;
      },
      numSeg);
  if (numSeg == n) {
    // Every unit is one contact (spheres): the leader is the contact itself and the canonical
    // order is the contact order, so the CSR is the identity -- written directly, without the
    // second sort. The sweeps then take the empty PosUnits (no indirection; bitwise identical).
    Kokkos::View<int*, CpMem> us = unitStart;
    Kokkos::View<int*, CpMem> uc = unitContacts;
    Kokkos::parallel_for(
        "peclet::dem::cp::uident", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int p) {
          us(p) = p;
          uc(p) = p;
        });
    Kokkos::deep_copy(space, Kokkos::subview(unitStart, n), n);
    space.fence();
    return n;
  }
  Kokkos::View<int*, CpMem> segLead(
      view_alloc(space, "peclet::dem::cp::ulead", WithoutInitializing), numSeg);
  Kokkos::deep_copy(space, segLead, n);
  Kokkos::parallel_for(
      "peclet::dem::cp::uleadmin", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int p) { Kokkos::atomic_min(&segLead(segId(p)), perm(p)); });
  // Second key: (leader << 32) | contact index -- unique, so the order does not depend on the
  // sort's stability: units by ascending leader, contacts ascending inside each unit. perm(p)
  // already holds the key's low word (the contact index), so it is the value the sort carries.
  Kokkos::parallel_for(
      "peclet::dem::cp::ukey2", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int p) {
        const int i = perm(p);
        keys(p) = (static_cast<std::uint64_t>(static_cast<unsigned>(segLead(segId(p)))) << 32) |
                  static_cast<unsigned>(i);
      });
  Kokkos::Experimental::sort_by_key(space, keys, perm);
  Kokkos::View<int*, CpMem> us = unitStart;
  Kokkos::View<int*, CpMem> uc = unitContacts;
  int nu = 0;
  Kokkos::parallel_scan(
      "peclet::dem::cp::ustart", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int p, int& run, const bool final) {
        const bool isNew = (p == 0) || ((keys(p) >> 32) != (keys(p - 1) >> 32));
        if (final) {
          uc(p) = perm(p);
          if (isNew)
            us(run) = p;
        }
        if (isNew)
          ++run;
      },
      nu);
  Kokkos::deep_copy(space, Kokkos::subview(unitStart, nu), n);
  space.fence();
  return nu;
}

/// PGS friction bound: overwrite each contact's friction_lambda_n with its manifold's converged
/// PGS push impulse (lambdaAcc, shared equally over the manifold's contact points). The legacy
/// accumulateNormalImpulse bound derives from approach velocities, which the PGS warm start has
/// already cancelled -- without this the Coulomb bound is ~0 and friction is inert.
inline void frictionBoundFromLambdaKokkos(Kokkos::View<ContactC*, CpMem> contacts, int numContacts,
                                          Kokkos::View<const int*, CpMem> contactSlot,
                                          Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                          Kokkos::View<const float*, CpMem> lambdaAcc) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::cp::pgs_friction_bound", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int i) {
        const int s = contactSlot(i);
        const ManifoldC m = manifolds(s);
        const int np = (m.num_points > 0) ? m.num_points : 1;
        contacts(i).friction_lambda_n = lambdaAcc(s) / static_cast<float>(np);
      });
  space.fence();
}

/// After the position solve: convert the per-contact positional lambdas into an impulse-
/// equivalent per manifold (lambda_pos / dt has force units; x dt back to impulse => just
/// lambda_pos * m_eff... the positional lambda already carries 1/w mass weighting, so the
/// impulse equivalent over the substep is lambda_pos / dt * dt = lambda_pos / (w*...) -- we
/// store lambda_pos/dt * dt = lambda_pos scaled by 1/dt to velocity-impulse units) and write
/// it into the sorted prev-store so next substep's warm gather can top up the Coulomb bound.
inline void commitPosImpulseKokkos(
    Kokkos::View<const float*, CpMem> posLambdaContact, int numContacts,
    Kokkos::View<const int*, CpMem> contactSlot, Kokkos::View<const ManifoldC*, CpMem> manifolds,
    int numManifolds, Kokkos::View<const unsigned long long*, CpMem> keys,
    Kokkos::View<const unsigned long long*, CpMem> prevKeysSorted, int prevCount, float dt,
    Kokkos::View<float*, CpMem> scratchManifold, Kokkos::View<float*, CpMem> prevPosImpulse) {
  CpExec space;
  auto sm = Kokkos::subview(scratchManifold, Kokkos::pair<int, int>(0, numManifolds));
  Kokkos::deep_copy(space, sm, 0.0f);
  Kokkos::parallel_for(
      "peclet::dem::pos_load_reduce", Kokkos::RangePolicy<CpExec>(space, 0, numContacts),
      KOKKOS_LAMBDA(int i) {
        const float l = posLambdaContact(i);
        if (l != 0.0f)
          Kokkos::atomic_add(&scratchManifold(contactSlot(i)), l / dt);
      });
  Kokkos::parallel_for(
      "peclet::dem::pos_load_scatter", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const unsigned long long k = keys(idx);
        if (k == ~0ull)
          return;
        int lo = 0, hi = prevCount;
        while (lo < hi) {
          const int m = (lo + hi) >> 1;
          if (prevKeysSorted(m) < k)
            lo = m + 1;
          else
            hi = m;
        }
        if (lo < prevCount && prevKeysSorted(lo) == k)
          prevPosImpulse(lo) = scratchManifold(idx);
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_CONTACT_PREPROCESSING_HPP
