#include "ParticleSystem.cuh"
#include "memory_utils.cuh" // For CUDA_CHECK
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void detect_contacts_sphere_kernel(ParticleSystemData ps,
                                              float global_scale,
                                              float safety_margin) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_potential = *ps.d_potential_count;
  if (idx == 0) {
    // printf("Narrowphase: Potential Collisions = %d\n", num_potential);
  }
  if (idx >= num_potential)
    return;

  int2 pair = ps.d_potential_collisions[idx];
  int idA = pair.x;
  int idB = pair.y;

  // Load Predicted Positions
  float4 pA_w = ps.d_pos_pred[idA];
  float4 pB_w = ps.d_pos_pred[idB];

  float3 pA = make_float3(pA_w.x, pA_w.y, pA_w.z);
  float3 pB = make_float3(pB_w.x, pB_w.y, pB_w.z);

  // Radii
  float rA = 1.0f * ps.d_scale[idA] * global_scale;
  float rB = 1.0f * ps.d_scale[idB] * global_scale;

  // Distance Check
  float3 diff = make_float3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
  float dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
  float dist = sqrtf(dist_sq);

  float overlap = dist - (rA + rB);

  // SPECULATIVE CONTACT LOGIC
  // Generate constraint if overlap < safety_margin (i.e. if penetrating OR
  // close enough) overlap is negative if penetrating. If overlap = 0.01 and
  // margin = 0.05 -> Generate. If overlap = 0.06 -> Skip.

  if (overlap < safety_margin) {
    // Generate Contact
    int contact_idx = atomicAdd(ps.d_contact_count, 1);
    if (contact_idx >= ps.max_contacts) {
      atomicSub(ps.d_contact_count, 1); // Revert
      return;                           // Overflow
    }

    // Normal B -> A ?
    // Plan says: normal (B -> A).
    // Use standard: Normal points FROM B TO A (separating direction).
    // If B is at (1,0), A is at (0,0). Normal should be (-1, 0).
    // diff is pB - pA = (1,0).
    // So normal = -diff / dist.
    // OR: Normal points FROM A TO B?
    // Let's stick to: Normal is direction to move A to resolve.
    // If A is overlapping B, A should move away from B.
    // Vector B->A.

    float3 normal;
    if (dist > 1e-9f) {
      normal = make_float3(-diff.x / dist, -diff.y / dist, -diff.z / dist);
    } else {
      // Coincident? Pick arbitrary.
      normal = make_float3(0, 1, 0);
    }

    // Contact Points & Lever Arms
    // For Sphere: contact point is on surface along normal.
    // rA = vector from Center A to Contact Point.
    // Found interaction
    // of overlap". Let's use midpoint
    // of the overlap region? rA = -normal * rA? (Vector from Center to Surface)
    // If Normal points B->A. Surface of A in direction of B is -Normal. So rA =
    // -normal * rA. Wait, Normal B->A means it points "out" of B towards A. So
    // contact on A is at pA - normal * rA.

    float3 rA_vec = make_float3(normal.x * -rA, normal.y * -rA, normal.z * -rA);
    float3 rB_vec = make_float3(normal.x * rB, normal.y * rB, normal.z * rB);

    // Wait, check signs.
    // A is at 0. B is at 10. Normal B->A is (-1, 0).
    // Contact on A surface towards B is at +rA.
    // normal is -1. -normal * rA = +1 * rA. Correct.
    // Contact on B surface towards A is at -rB.
    // normal is -1. +normal * rB = -1 * rB. Correct.

    ContactConstraint c;
    c.bodyA = idA;
    c.bodyB = idB;
    c.normal = make_float4(normal.x, normal.y, normal.z, 0.0f);
    c.rA = make_float4(rA_vec.x, rA_vec.y, rA_vec.z, 0.0f);
    c.rB = make_float4(rB_vec.x, rB_vec.y, rB_vec.z, 0.0f);
    c.dist = overlap;           // Signed distance
    c.friction_lambda_n = 0.0f; // Initialize

    ps.d_contacts[contact_idx] = c;
  }
}

__global__ void detect_ground_kernel(ParticleSystemData ps, float global_scale,
                                     float safety_margin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_real)
    return;

  float4 p_w = ps.d_pos_pred[idx];
  float s = ps.d_scale[idx] * global_scale;
  float radius = 1.0f * s;
  float ground_y = ps.domain_min.y;

  // Dist = (y - rad) - ground
  float dist = (p_w.y - radius) - ground_y;

  if (dist < safety_margin) {
    int contact_idx = atomicAdd(ps.d_contact_count, 1);
    if (contact_idx >= ps.max_contacts) {
      atomicSub(ps.d_contact_count, 1);
      return;
    }

    float3 normal = make_float3(0, 1, 0); // Wall Normal (Ground -> Up)
    // Correct? Standard "B -> A". If Ground is B. Ground points UP to A.
    // Yes.

    float3 rA_vec = make_float3(0, -radius, 0); // Center to bottom
    float3 rB_vec = make_float3(
        0, 0,
        0); // Irrelevant for static plane typically, but use contact point?
    // Contact point on Wall = (p.x, ground_y, p.z).
    // rB is relative to Center of B. Static body has no center?
    // Let's use 0 and handle mass=inf in solver.

    ContactConstraint c;
    c.bodyA = idx;
    c.bodyB = -1; // Ground
    c.normal = make_float4(normal.x, normal.y, normal.z, 0.0f);
    c.rA = make_float4(rA_vec.x, rA_vec.y, rA_vec.z, 0.0f);
    c.rB = make_float4(0, 0, 0, 0);
    c.dist = dist;
    c.friction_lambda_n = 0.0f;
    ps.d_contacts[contact_idx] = c;
  }
}

void launch_narrowphase(ParticleSystemData ps, float global_scale) {
  // 1. Reset contact count
  // Done in integration predict_and_clear? No, that was for accumulators.
  // Contact count needs reset before this or at start of frame.
  // Plan said "Reset: Set *d_contact_count = 0" in Kernel A
  // (predict_and_clear). So we don't need to reset here if
  // launch_integrate_predict was called.

  // Check potential collisions count
  int num_potential;
  CUDA_CHECK(cudaMemcpy(&num_potential, ps.d_potential_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (num_potential == 0)
    return;
  int threads = 256;
  int blocks;

  float safety_margin = 0.1f * global_scale; // Consistent with Broadphase

  if (num_potential > 0) {
    blocks = (num_potential + threads - 1) / threads;
    detect_contacts_sphere_kernel<<<blocks, threads>>>(ps, global_scale,
                                                       safety_margin);
    CUDA_CHECK(cudaGetLastError());
  }

  // Ground Check
  // Can simple call ground kernel with same threading (check per particle)
  // Or just append after?
  // Let's iterate all particles again?
  // blocks/threads same as ps.num_real
  int num_real_particles = ps.num_real; // Use a new variable name for clarity
  if (num_real_particles > 0) {
    blocks = (num_real_particles + threads - 1) / threads;
    detect_ground_kernel<<<blocks, threads>>>(ps, global_scale, safety_margin);
    CUDA_CHECK(cudaGetLastError());
  }
}
