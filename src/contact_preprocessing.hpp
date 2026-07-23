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

#include <cstdint>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "dem_portable.hpp"  // F4, cross3

namespace peclet::dem {

using CpExec = Kokkos::DefaultExecutionSpace;
using CpMem = CpExec::memory_space;

/// Portable mirror of ParticleSystem.cuh ContactConstraint (the fields this reduction touches).
struct ContactC {
  int bodyA;
  int bodyB;   // < 0 => boundary/static
  F4 normal;   // .xyz = world normal
  F4 rA;       // lever arm on A
  F4 rB;       // lever arm on B
  float dist;  // signed penetration (>0 => inactive)
  float friction_lambda_n;
  float weight;
  // --- moving / per-material boundary (idB<0) extension; sentinels for body-body & static planes so
  // the solvers fall back to the global material and a zero wall velocity (unchanged behaviour). ---
  F4 boundaryVel{0.0f, 0.0f, 0.0f, 0.0f};  // wall surface velocity at the contact point (xyz)
  float boundaryRestitution{-1.0f};        // per-wall normal restitution; < 0 => use the global one
  float boundaryFriction{-1.0f};           // per-wall Coulomb friction;   < 0 => use the global one
};

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

/// Persistent-contact detection for the gravity-gated restitution rule. Key = (min real body,
/// max real body) packed in 64 bits; boundary manifolds (bodyB = -1: planes + SDF walls, merged per
/// particle) use lo = 0xFFFFFFFF. Flags each manifold whose pair already existed in the PREVIOUS
/// substep (prevKeys sorted, device binary search). Real indices are stable within a single-GPU
/// run, so the key identifies the physical pair across substeps.
KOKKOS_INLINE_FUNCTION unsigned long long pairKeyOf(const ManifoldC& m,
                                                    Kokkos::View<const int*, CpMem> realIdx) {
  const unsigned a = static_cast<unsigned>(realIdx(m.bodyA));
  const unsigned b = (m.bodyB >= 0) ? static_cast<unsigned>(realIdx(m.bodyB)) : 0xFFFFFFFFu;
  const unsigned hi = a < b ? a : b, lo = a < b ? b : a;
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}

inline void markPersistentManifoldsKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                          int numManifolds,
                                          Kokkos::View<const int*, CpMem> realIdx,
                                          Kokkos::View<const unsigned long long*, CpMem> prevKeys,
                                          int prevCount,
                                          Kokkos::View<unsigned long long*, CpMem> outKeys,
                                          Kokkos::View<unsigned char*, CpMem> outFlags) {
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
        const unsigned long long k = pairKeyOf(m, realIdx);
        outKeys(idx) = k;
        int lo = 0, hi = prevCount;
        while (lo < hi) {  // lower_bound on the sorted previous-substep keys
          const int mid = (lo + hi) >> 1;
          if (prevKeys(mid) < k)
            lo = mid + 1;
          else
            hi = mid;
        }
        outFlags(idx) = (lo < prevCount && prevKeys(lo) == k) ? 1 : 0;
      });
  space.fence();
}

/// Warm-start gather for the PGS velocity solve: per manifold, write its pair key and look up the
/// previous substep's converged push impulse (0 for a new contact). Periodic-ghost duplicate
/// manifolds (realA > realB twin) get key ~0 and warm 0 -- the canonical twin carries the impulse.
inline void gatherWarmLambdaKokkos(Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                   int numManifolds, Kokkos::View<const int*, CpMem> realIdx,
                                   Kokkos::View<const unsigned long long*, CpMem> prevKeys,
                                   Kokkos::View<const float*, CpMem> prevLambda,
                                   Kokkos::View<const float* [3], CpMem> prevLambdaT,
                                   Kokkos::View<const float*, CpMem> prevPosImpulse, int prevCount,
                                   Kokkos::View<unsigned long long*, CpMem> outKeys,
                                   Kokkos::View<float*, CpMem> outWarm,
                                   Kokkos::View<float* [3], CpMem> outWarmT,
                                   Kokkos::View<float*, CpMem> outPosImpulse) {
  CpExec space;
  Kokkos::parallel_for(
      "peclet::dem::gather_warm", Kokkos::RangePolicy<CpExec>(space, 0, numManifolds),
      KOKKOS_LAMBDA(int idx) {
        const ManifoldC m = manifolds(idx);
        bool dup = false;
        if (m.num_points > 0 && m.bodyB >= 0 && realIdx(m.bodyA) > realIdx(m.bodyB))
          dup = true;
        if (m.num_points <= 0 || dup) {
          outKeys(idx) = ~0ull;
          outWarm(idx) = 0.0f;
          outWarmT(idx, 0) = outWarmT(idx, 1) = outWarmT(idx, 2) = 0.0f;
          outPosImpulse(idx) = 0.0f;
          return;
        }
        const unsigned long long k = pairKeyOf(m, realIdx);
        outKeys(idx) = k;
        int lo = 0, hi = prevCount;
        while (lo < hi) {
          const int mid = (lo + hi) >> 1;
          if (prevKeys(mid) < k)
            lo = mid + 1;
          else
            hi = mid;
        }
        const bool hit = (lo < prevCount && prevKeys(lo) == k);
        outWarm(idx) = hit ? prevLambda(lo) : 0.0f;
        outWarmT(idx, 0) = hit ? prevLambdaT(lo, 0) : 0.0f;
        outWarmT(idx, 1) = hit ? prevLambdaT(lo, 1) : 0.0f;
        outWarmT(idx, 2) = hit ? prevLambdaT(lo, 2) : 0.0f;
        outPosImpulse(idx) = hit ? prevPosImpulse(lo) : 0.0f;
      });
  space.fence();
}

/// Save this substep's keys + converged impulses (normal AND tangential) and key-sort them for
/// next substep's gather. A permutation sort carries both value arrays through one key sort.
inline void commitPairKeysLambdaKokkos(Kokkos::View<const unsigned long long*, CpMem> keys,
                                       Kokkos::View<const float*, CpMem> lambda,
                                       Kokkos::View<const float* [3], CpMem> lambdaT,
                                       Kokkos::View<unsigned long long*, CpMem> prevKeys,
                                       Kokkos::View<float*, CpMem> prevLambda,
                                       Kokkos::View<float* [3], CpMem> prevLambdaT,
                                       int numManifolds) {
  if (numManifolds <= 0)
    return;
  CpExec space;
  const int n = numManifolds;
  const auto rng = Kokkos::pair<int, int>(0, n);
  auto kd = Kokkos::subview(prevKeys, rng);
  Kokkos::deep_copy(space, kd, Kokkos::subview(keys, rng));
  Kokkos::View<int*, CpMem> perm(
      Kokkos::view_alloc(space, "peclet::dem::commit_perm", Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "peclet::dem::commit_iota", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) { perm(i) = i; });
  Kokkos::Experimental::sort_by_key(space, kd, perm);
  Kokkos::View<float*, CpMem> pl = prevLambda;
  Kokkos::View<float* [3], CpMem> plt = prevLambdaT;
  Kokkos::parallel_for(
      "peclet::dem::commit_gather", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        const int j = perm(i);
        pl(i) = lambda(j);
        plt(i, 0) = lambdaT(j, 0);
        plt(i, 1) = lambdaT(j, 1);
        plt(i, 2) = lambdaT(j, 2);
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
/// wall/plane contacts, then `sweeps` monotone propagation passes lower -> upper (255 -> 254 -> ...)
/// through the manifold graph. Warm start makes a few sweeps per substep track slowly-moving
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
  space.fence();
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

/// splitmix32 finalizer: a well-mixed pseudo-random priority per edge index. Random priorities make
/// the Jones-Plassmann arbitration finish in O(log n) rounds w.h.p.; RAW indices are adversarial for
/// lattice-ordered dense packs (monotone index chains -> one win per round -> O(chain) rounds).
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

/// Decode the canonical (bodyA, bodyB) from a pair key (bodyB = -1 for boundary).
KOKKOS_INLINE_FUNCTION void decodeKey(std::uint64_t key, int& bodyA, int& bodyB) {
  const unsigned v = static_cast<unsigned>(key & 0xFFFFFFFFu);
  bodyA = static_cast<int>(key >> 32);
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

/// Reduce `n` contacts to manifolds (one per unique canonical pair). outManifolds must hold at
/// least the number of unique pairs; returns that count (also written to outCount).
inline int reduceContactsToManifoldsKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int n,
                                           Kokkos::View<ManifoldC*, CpMem> outManifolds,
                                           Kokkos::View<int, CpMem> outCount,
                                           Kokkos::View<int*, CpMem> contactSlot = {}) {
  CpExec space;
  if (n == 0) {
    Kokkos::deep_copy(space, outCount, 0);
    return 0;
  }

  // 1. Key every contact and seed an identity permutation.
  Kokkos::View<std::uint64_t*, CpMem> keys(
      Kokkos::view_alloc(space, "peclet::dem::cp::keys", Kokkos::WithoutInitializing), n);
  Kokkos::View<int*, CpMem> perm(
      Kokkos::view_alloc(space, "peclet::dem::cp::perm", Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "peclet::dem::cp::key", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        keys(i) = pairKey(contacts(i));
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
inline void commitPosImpulseKokkos(Kokkos::View<const float*, CpMem> posLambdaContact,
                                   int numContacts, Kokkos::View<const int*, CpMem> contactSlot,
                                   Kokkos::View<const ManifoldC*, CpMem> manifolds,
                                   int numManifolds,
                                   Kokkos::View<const unsigned long long*, CpMem> keys,
                                   Kokkos::View<const unsigned long long*, CpMem> prevKeysSorted,
                                   int prevCount, float dt,
                                   Kokkos::View<float*, CpMem> scratchManifold,
                                   Kokkos::View<float*, CpMem> prevPosImpulse) {
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
