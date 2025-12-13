#include "ParticleSystem.cuh"
#include "math_utils.cuh"
#include "memory_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float3 cross_product(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ float dot_product(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void solve_velocity_jacobi_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_contacts = *ps.d_contact_count; // Atomic count from narrowphase

  if (idx >= num_contacts)
    return;

  ContactConstraint c = ps.d_contacts[idx];

  // FILTER: Ignore safety margin contacts for Velocity Solve
  if (c.dist > 0.0f)
    return;

  int idA = c.bodyA;
  int idB = c.bodyB;

  // Load Masses/Inertia
  float invMassA = ps.d_pos[idA].w;
  float invMassB = 0.0f;
  if (idB >= 0)
    invMassB = ps.d_pos[idB].w;

  // Load Current Predicted Velocities
  // In Jacobi, we read the PREDICTED state (which contains specific gravity
  // update and previous iterations?) If we iterate, we should read "current
  // guess". d_vel_pred is the current guess.

  float4 vA_w = ps.d_vel_pred[idA];
  float4 wA_w = ps.d_ang_vel_pred[idA];

  float3 vA = make_float3(vA_w.x, vA_w.y, vA_w.z);
  float3 wA = make_float3(wA_w.x, wA_w.y, wA_w.z);
  float3 vB = make_float3(0, 0, 0);
  float3 wB = make_float3(0, 0, 0);
  if (idB >= 0) {
    float4 vB_w_local = ps.d_vel_pred[idB];
    float4 wB_w_local = ps.d_ang_vel_pred[idB];
    vB = make_float3(vB_w_local.x, vB_w_local.y, vB_w_local.z);
    wB = make_float3(wB_w_local.x, wB_w_local.y, wB_w_local.z);
  }

  // Contact Geometry
  float3 n = make_float3(c.normal.x, c.normal.y, c.normal.z);
  float3 rA = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 rB = make_float3(c.rB.x, c.rB.y, c.rB.z);

  // Relative Velocity at Contact Point
  // v_rel = (vA + wA x rA) - (vB + wB x rB)
  float3 vPA = vA + cross_product(wA, rA);
  float3 vPB = vB + cross_product(wB, rB);
  float3 v_rel = vPA - vPB;

  float vn = dot_product(v_rel, n);

  // Activity Check
  // If dist > 0 (Speculative) AND vn > 0 (Separating) -> Inactive
  // Activity Check: Only apply if approaching (vn < 0)
  // Strict non-penetration: If separating (vn >= 0), do nothing.
  // Inequality Constraint C(x) >= 0.
  if (vn >= 0.0f) {
    return;
  }

  if (idx < 5) {
    printf("Contact %d: vn=%f, n=(%f, %f, %f), idA=%d, idB=%d\n", idx, vn, n.x,
           n.y, n.z, idA, idB);
  }

  // --- MIN-SCALING WEIGHTING ---
  int countA = ps.d_constraint_counts[idA];
  int countB = (idB >= 0) ? ps.d_constraint_counts[idB] : 1;
  // If B is wall, countB ?? Wall is singular constraint? Or infinite?
  // User Plan: Scale = 1 / min(Na, Nb).
  // If B is Wall (idB < 0), treat as Count=1 (Stiff interaction) -> min(Na, 1)
  // = 1. This means Wall interaction has full weight 1.0. Correct.

  // Avoid division by zero (should not happen if constraint exists)
  if (countA == 0)
    countA = 1;
  if (countB == 0)
    countB = 1;

  int min_count = (countA < countB) ? countA : countB;

  // float weight = 1.0f / (float)min_count;
  float weight = 1.0f; // DEBUG: Force unweighted

  // Calculate Effective Mass (lambda = J / Meff)
  // ... (No Change to Mass Calc) ...
  // Calculate Effective Mass (lambda = J / Meff)
  // 1 / Meff = sum( invMass + (r x n)^T I^-1 (r x n) )
  float3 invIA_vec =
      make_float3(ps.d_inv_inertia[idA].x, ps.d_inv_inertia[idA].y,
                  ps.d_inv_inertia[idA].z);
  float3 invIB_vec = make_float3(0, 0, 0);
  if (idB >= 0)
    invIB_vec = make_float3(ps.d_inv_inertia[idB].x, ps.d_inv_inertia[idB].y,
                            ps.d_inv_inertia[idB].z);

  auto compute_generalized_inv_mass = [&](float3 r, float3 dir, float invM,
                                          float3 invI) {
    float3 rn = cross_product(r, dir);
    return invM + rn.x * rn.x * invI.x + rn.y * rn.y * invI.y +
           rn.z * rn.z * invI.z;
  };

  float wA_n = compute_generalized_inv_mass(rA, n, invMassA, invIA_vec);
  float wB_n = compute_generalized_inv_mass(rB, n, invMassB, invIB_vec);
  float w_total_n = wA_n + wB_n;

  if (w_total_n < 1e-6f)
    return; // Infinite mass or error

  // ----------------------------
  // Normal Impulse (Restitution)
  // ----------------------------
  // ... (Restitution Logic Removed for Simplicity / Robustness, usually e=0 for
  // stacking) ... Or keep simple restitution logic? Let's use simple target:
  float vn_target = 0.0f;
  // If we want restitution, add it here. For stability now, 0.0 is safest.

  float dV = vn_target - vn; // dV is positive (vn < 0)

  // Impulse J = dV / w_total
  float lambda_n = dV / w_total_n;

  // Clamping not needed if vn < 0 constraint checked above.
  // (vn_target >= vn usually)

  // SCALE IMPULSE BY WEIGHT
  float3 impulse_n =
      make_float3(n.x * lambda_n * weight, n.y * lambda_n * weight,
                  n.z * lambda_n * weight);

  // ----------------------------
  // Friction (Tangential)
  // ----------------------------
  float3 vt_vec = v_rel - make_float3(n.x * vn, n.y * vn, n.z * vn);
  float vt_len = sqrtf(dot_product(vt_vec, vt_vec));

  float3 impulse_t = make_float3(0, 0, 0);

  if (vt_len > 1e-6f) {
    float3 t =
        make_float3(vt_vec.x / vt_len, vt_vec.y / vt_len, vt_vec.z / vt_len);

    float wA_t = compute_generalized_inv_mass(rA, t, invMassA, invIA_vec);
    float wB_t = compute_generalized_inv_mass(rB, t, invMassB, invIB_vec);
    float w_total_t = wA_t + wB_t;

    // Target vt = 0
    float lambda_t = -vt_len / w_total_t;

    // Coulomb Clamping: |lambda_t| <= mu * lambda_n
    float max_friction = ps.friction_dynamic * lambda_n;
    if (lambda_t < -max_friction)
      lambda_t = -max_friction;
    if (lambda_t > max_friction)
      lambda_t = max_friction;

    // SCALE IMPULSE BY WEIGHT
    impulse_t = make_float3(t.x * lambda_t * weight, t.y * lambda_t * weight,
                            t.z * lambda_t * weight);
  }

  // ----------------------------
  // Apply Impulses (Atomic)
  // ----------------------------
  float3 total_J = impulse_n + impulse_t;

  // Body A: +J
  float3 dvA = make_float3(total_J.x * invMassA, total_J.y * invMassA,
                           total_J.z * invMassA);
  float3 rxJ_A = cross_product(rA, total_J);
  float3 dwA = make_float3(rxJ_A.x * invIA_vec.x, rxJ_A.y * invIA_vec.y,
                           rxJ_A.z * invIA_vec.z);

  atomicAddVector(ps.d_delta_vel + idA, make_float4(dvA.x, dvA.y, dvA.z, 0));
  atomicAddVector(ps.d_delta_ang_vel + idA,
                  make_float4(dwA.x, dwA.y, dwA.z, 0));

  // Body B: -J
  if (idB >= 0) {
    atomicAddVector(ps.d_delta_vel + idB,
                    make_float4(-total_J.x * invMassB, -total_J.y * invMassB,
                                -total_J.z * invMassB, 0.0f));
    float3 rxJ_B = cross_product(rB, total_J);
    float3 dwB = make_float3(-rxJ_B.x * invIB_vec.x, -rxJ_B.y * invIB_vec.y,
                             -rxJ_B.z * invIB_vec.z);
    atomicAddVector(ps.d_delta_ang_vel + idB,
                    make_float4(dwB.x, dwB.y, dwB.z, 0));
  }
  // Note: We DO NOT increment constraint counts here anymore,
  // as we use the pre-pass counts for weighting logic.
}

void launch_velocity_solve(ParticleSystemData ps) {
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (num_contacts == 0)
    return;

  int threads = 256;
  int blocks = (num_contacts + threads - 1) / threads;

  // We can iterate here if we want multiple Jacobi passes.
  // For Phase 1, start with 1 or 2 iterations?
  // Plan doesn't strictly specify iterations within the kernel, usually invoked
  // multiple times in simulation loop. Just launch logic once.
  solve_velocity_jacobi_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}
