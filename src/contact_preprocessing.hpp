/// @file
/// @brief dem — portable (Kokkos) contact->manifold reduction, replacing the thrust-based
/// reduce_contacts_to_manifolds() in contact_preprocessing.cu.
///
/// Same pipeline: key each contact by its canonical pair, group by key, and within a group sum the
/// aligned normals / torque arms / lever arms and count the points. thrust::sort_by_key +
/// reduce_by_key become Kokkos::Experimental::sort_by_key + a scan-based segmented reduction (atomic
/// accumulation), so no thrust/cub. The per-contact math (ContactToManifold / TransformAndFilter) is
/// reused verbatim as KOKKOS_INLINE_FUNCTION and is shared with the host reference in the test.
///
/// One intentional change vs the thrust version: a manifold's (bodyA,bodyB) is decoded determinist
/// -ically from the pair key (canonical min/max, or (idA,-1) for boundary), rather than taken from
/// the first contact of an unstably-sorted run. The summed quantities are commutative, so they are
/// unaffected. Decoupled from ParticleSystemData (portable POD mirrors) for standalone validation.
#ifndef DEM_CONTACT_PREPROCESSING_HPP
#define DEM_CONTACT_PREPROCESSING_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <cstdint>

#include "dem_portable.hpp"  // F4, cross3

namespace dem {

using CpExec = Kokkos::DefaultExecutionSpace;
using CpMem = CpExec::memory_space;

/// Portable mirror of ParticleSystem.cuh ContactConstraint (the fields this reduction touches).
struct ContactC {
  int bodyA;
  int bodyB;  // < 0 => boundary/static
  F4 normal;  // .xyz = world normal
  F4 rA;      // lever arm on A
  F4 rB;      // lever arm on B
  float dist;  // signed penetration (>0 => inactive)
  float friction_lambda_n;
  float weight;
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
};

/// Canonical pair key: (min<<32)|max, or (idA<<32)|0xFFFFFFFF for a boundary (idB<0) contact.
KOKKOS_INLINE_FUNCTION std::uint64_t pairKey(const ContactC& c) {
  const int idA = c.bodyA, idB = c.bodyB;
  if (idB < 0) return (static_cast<std::uint64_t>(static_cast<unsigned>(idA)) << 32) | 0xFFFFFFFFu;
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
  return m;
}

/// Reduce `n` contacts to manifolds (one per unique canonical pair). outManifolds must hold at least
/// the number of unique pairs; returns that count (also written to outCount).
inline int reduceContactsToManifoldsKokkos(Kokkos::View<const ContactC*, CpMem> contacts, int n,
                                           Kokkos::View<ManifoldC*, CpMem> outManifolds,
                                           Kokkos::View<int, CpMem> outCount) {
  CpExec space;
  if (n == 0) {
    Kokkos::deep_copy(space, outCount, 0);
    return 0;
  }

  // 1. Key every contact and seed an identity permutation.
  Kokkos::View<std::uint64_t*, CpMem> keys(Kokkos::view_alloc(space, "dem::cp::keys", Kokkos::WithoutInitializing), n);
  Kokkos::View<int*, CpMem> perm(Kokkos::view_alloc(space, "dem::cp::perm", Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "dem::cp::key", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) {
        keys(i) = pairKey(contacts(i));
        perm(i) = i;
      });

  // 2. Sort the permutation by key (groups equal pairs contiguously).
  Kokkos::Experimental::sort_by_key(space, keys, perm);

  // 3. Segment id per sorted position: inclusive scan of "key changed" minus 1.
  Kokkos::View<int*, CpMem> segId(Kokkos::view_alloc(space, "dem::cp::segId", Kokkos::WithoutInitializing), n);
  int numSeg = 0;
  Kokkos::parallel_scan(
      "dem::cp::segscan", Kokkos::RangePolicy<CpExec>(space, 0, n),
      KOKKOS_LAMBDA(int p, int& run, const bool final) {
        const bool isNew = (p == 0) || (keys(p) != keys(p - 1));
        if (isNew) ++run;
        if (final) segId(p) = run - 1;  // 0-based segment index
      },
      numSeg);

  // 4. Initialise one manifold per segment (canonical ids from the key, sums zero).
  Kokkos::View<std::uint64_t*, CpMem> k = keys;
  Kokkos::View<int*, CpMem> sid = segId;
  Kokkos::View<ManifoldC*, CpMem> out = outManifolds;
  Kokkos::parallel_for(
      "dem::cp::init", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int p) {
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
      "dem::cp::accum", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int p) {
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
      });
  space.fence();

  Kokkos::deep_copy(space, outCount, numSeg);
  return numSeg;
}

}  // namespace dem

#endif  // DEM_CONTACT_PREPROCESSING_HPP
