#include "ParticleSystem.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

// -----------------------------------------------------------------------------
// Math Helpers
// -----------------------------------------------------------------------------
__host__ __device__ inline float3 cross_product_device(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

// -----------------------------------------------------------------------------
// Functors
// -----------------------------------------------------------------------------

// Extract PairID from Contact
struct ContactToPairID {
  __host__ __device__ unsigned long long
  operator()(const ContactConstraint &c) const {
    int idA = c.bodyA;
    int idB = c.bodyB;
    if (idB < 0) {
      // Boundary/Static: idB = -1. Treat (idA, -1) as unique pair.
      // Key = (idA << 32) | (unsigned int)(-1) ?
      // Or just ensure it doesn't conflict.
      // Let's use standard pair logic but handle -1.
      // if idB < 0, it's effectively "idA" interacting with world.
      // Canonical: idB is always "larger" if we treat -1 as MAX_INT? No.
      // Let's treat -1 as Non-Canonical always.
      // So Canonical = idA.
      return ((unsigned long long)idA << 32) | 0xFFFFFFFF;
    }
    // Canonical Pair ID: (min << 32) | max
    unsigned int u = (idA < idB) ? idA : idB;
    unsigned int v = (idA < idB) ? idB : idA;
    return ((unsigned long long)u << 32) | v;
  }
};

// Transform: ContactConstraint -> ManifoldConstraint
struct ContactToManifold {
  __host__ __device__ ManifoldConstraint
  operator()(const ContactConstraint &c) const {
    ManifoldConstraint m;

    // Canonical Normals: Always point derived from Body A to Body B (or
    // similar) Here we assume normal points A -> B. We want to accumulate
    // contributions for the pair (A, B). If pair is (B, A), we must flip normal
    // and torques.

    // Key Logic sorts by pair (min_id, max_id).
    // If c.bodyA < c.bodyB: It matches the key. (Canonical)
    // If c.bodyA > c.bodyB: It is flipped relative to key order.

    // Note: The Key Generation handled (A, B) order.
    // But the ContactConstraint 'c' still has original A and B and Normal.
    // If we are aggregating, we must align everything to the Canonical Pair
    // (idA < idB).

    int idA = c.bodyA;
    int idB = c.bodyB; // Could be world (-1)

    // Determine Canonical IDs
    int can_A = idA;
    int can_B = idB;
    bool flip = false;

    if (idB >= 0 && idB < idA) {
      can_A = idB;
      can_B = idA;
      flip = true;
    }

    m.bodyA = can_A; // Matches key
    m.bodyB = can_B;
    m.num_points = 1;

    // Normal: Points A->B.
    // If flip (we are looking at B->A contact), the normal in 'c' points A->B
    // (original). So relative to Canonical (B, A) [i.e. NewA=B, NewB=A], The
    // force on NewA (B) is -ForceOnA. Normal in 'c' is usually Normal on B ? Or
    // Normal from A to B? Convention: Normal points FROM A TO B. Force on B is
    // +Normal. Force on A is -Normal. We want Normal_Sum to be the Force
    // Direction on Canonical Body A (can_A). Case 1: (A, B), can_A=A. Force on
    // A is -Normal. Case 2: (B, A), can_A=B. Force on B is +Normal. So: If
    // !flip: N_aligned = -c.normal If flip:  N_aligned = +c.normal

    float3 n_vec = make_float3(c.normal.x, c.normal.y, c.normal.z);
    float3 n_aligned;
    if (!flip) {
      n_aligned = make_float3(-n_vec.x, -n_vec.y, -n_vec.z);
    } else {
      n_aligned = n_vec;
    }

    // Torques:
    // Torque = r x Force.
    // Force on can_A is N_aligned.
    // rA is vector from A's center to contact pivot.
    // c.rA is rA (for bodyA). c.rB is rB (for bodyB).
    // If !flip: can_A = A. Force = N_aligned. TorqueA = rA x N_aligned.
    //           can_B = B. Force = -N_aligned. TorqueB = rB x -N_aligned.
    // If flip:  can_A = B. Force = N_aligned (which is +Normal).
    //           TorqueA (on B) = rB x N_aligned.
    //           can_B = A. Force = -N_aligned. TorqueB (on A) = rA x
    //           -N_aligned.

    float3 rA = make_float3(c.rA.x, c.rA.y, c.rA.z);
    float3 rB = make_float3(c.rB.x, c.rB.y, c.rB.z);

    float3 r_can, r_other;
    // Compute Midpoint Lever Arms (Centering) to ensure Unique Torque Point
    // rA_unified = rA - 0.5 * n * dist
    // rB_unified = rB + 0.5 * n * dist
    // (Assuming n points A->B. In narrowphase, n points B->A? No, usually A->B.
    // Let's check: narrowphase uses SDF on B. n = grad(dist). dist > 0 if
    // outside. n points OUT of B. So n points B->A. dist is signed distance
    // from B surface. If n points B->A: P_A = PosA + rA. P_B = PosB + rB. P_B =
    // P_A - dist * n. (P_A is query on A, P_B is surface on B). Vector A->B is
    // -dist * n. Midpoint = P_A - 0.5 * dist * n. rA_mid = rA - 0.5 * dist * n.
    // rB_mid = rB + 0.5 * dist * n.

    // Normal in constraint 'c':
    // narrowphase: c.normal = n_world. (n_world = rotate(n_local)).
    // n_local was grad(sdf).

    float3 shift_vec =
        make_float3(c.normal.x * c.dist * 0.5f, c.normal.y * c.dist * 0.5f,
                    c.normal.z * c.dist * 0.5f);

    float3 rA_mid =
        make_float3(rA.x - shift_vec.x, rA.y - shift_vec.y, rA.z - shift_vec.z);
    float3 rB_mid =
        make_float3(rB.x + shift_vec.x, rB.y + shift_vec.y, rB.z + shift_vec.z);

    if (!flip) {
      r_can = rA_mid;
      r_other = rB_mid;
    } else {
      r_can = rB_mid;
      r_other = rA_mid;
    }

    float3 tau_can = cross_product_device(r_can, n_aligned);
    float3 tau_other = cross_product_device(
        r_other, make_float3(-n_aligned.x, -n_aligned.y, -n_aligned.z));

    m.normal_sum = make_float4(n_aligned.x, n_aligned.y, n_aligned.z, 0.0f);
    m.torque_armA_sum = make_float4(tau_can.x, tau_can.y, tau_can.z, 0.0f);
    m.torque_armB_sum =
        make_float4(tau_other.x, tau_other.y, tau_other.z, 0.0f);
    m.rA_sum = make_float4(r_can.x, r_can.y, r_can.z, 0.0f);
    m.rB_sum = make_float4(r_other.x, r_other.y, r_other.z, 0.0f);

    return m;
  }
};

// Reducer: Sum Manifolds
struct SumManifold {
  __host__ __device__ ManifoldConstraint
  operator()(const ManifoldConstraint &a, const ManifoldConstraint &b) const {
    ManifoldConstraint res;
    res.bodyA = a.bodyA; // Should be same
    res.bodyB = a.bodyB;
    res.num_points = a.num_points + b.num_points;

    res.normal_sum = make_float4(a.normal_sum.x + b.normal_sum.x,
                                 a.normal_sum.y + b.normal_sum.y,
                                 a.normal_sum.z + b.normal_sum.z, 0.0f);

    res.torque_armA_sum =
        make_float4(a.torque_armA_sum.x + b.torque_armA_sum.x,
                    a.torque_armA_sum.y + b.torque_armA_sum.y,
                    a.torque_armA_sum.z + b.torque_armA_sum.z, 0.0f);

    res.torque_armB_sum =
        make_float4(a.torque_armB_sum.x + b.torque_armB_sum.x,
                    a.torque_armB_sum.y + b.torque_armB_sum.y,
                    a.torque_armB_sum.z + b.torque_armB_sum.z, 0.0f);

    res.rA_sum = make_float4(a.rA_sum.x + b.rA_sum.x, a.rA_sum.y + b.rA_sum.y,
                             a.rA_sum.z + b.rA_sum.z, 0.0f);
    res.rB_sum = make_float4(a.rB_sum.x + b.rB_sum.x, a.rB_sum.y + b.rB_sum.y,
                             a.rB_sum.z + b.rB_sum.z, 0.0f);
    return res;
  }
};

// TransformAndFilter: Handles inactive contacts by returning zero-manifold
struct TransformAndFilter {
  __host__ __device__ ManifoldConstraint
  operator()(const ContactConstraint &c) const {
    if (c.dist > 0.0f) {
      // Inactive
      ManifoldConstraint m;
      m.num_points = 0;
      m.normal_sum = make_float4(0, 0, 0, 0);
      m.torque_armA_sum = make_float4(0, 0, 0, 0);
      m.torque_armB_sum = make_float4(0, 0, 0, 0);
      m.rA_sum = make_float4(0, 0, 0, 0);
      m.rB_sum = make_float4(0, 0, 0, 0);
      m.bodyA = c.bodyA;
      m.bodyB = c.bodyB;
      return m;
    }
    ContactToManifold converter;
    return converter(c);
  }
};

// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------

void reduce_contacts_to_manifolds(ParticleSystemData &ps) {
  int num_contacts;
  cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int),
             cudaMemcpyDeviceToHost);

  if (num_contacts == 0) {
    cudaMemset(ps.d_manifold_count, 0, sizeof(int));
    return;
  }

  // Wrap Pointers
  thrust::device_ptr<ContactConstraint> d_contacts_ptr(ps.d_contacts);
  thrust::device_ptr<ManifoldConstraint> d_manifolds_ptr(ps.d_manifolds);

  // 1. Sort by PairID
  thrust::device_vector<unsigned long long> keys(num_contacts);

  // Create Keys
  thrust::transform(d_contacts_ptr, d_contacts_ptr + num_contacts, keys.begin(),
                    ContactToPairID());

  // Sort Keys and Data
  thrust::sort_by_key(keys.begin(), keys.end(), d_contacts_ptr);

  // 2. Reduce to Manifolds
  // Usage: reduce_by_key(keys_in, vals_in (transformed), keys_out, vals_out)

  // Create Transform Iterator
  // Use global TransformAndFilter
  auto man_iter =
      thrust::make_transform_iterator(d_contacts_ptr, TransformAndFilter());

  // Keys out
  thrust::device_vector<unsigned long long> keys_out(num_contacts);

  auto end_pair = thrust::reduce_by_key(
      keys.begin(), keys.end(), man_iter, keys_out.begin(), d_manifolds_ptr,
      thrust::equal_to<unsigned long long>(), SumManifold());

  int num_manifolds = end_pair.second - d_manifolds_ptr;

  // Copy count
  cudaMemcpy(ps.d_manifold_count, &num_manifolds, sizeof(int),
             cudaMemcpyHostToDevice);
}
