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
  if (vn >= 0.0f) {
    return;
  }

  // Calculate Effective Mass (lambda = J / Meff)
  // 1 / Meff = sum( invMass + (r x n)^T I^-1 (r x n) )
  // For spheres, I^-1 is scalar (diagonal.
  // Let's assume sphere inertia for now or use d_inv_inertia.
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
    // J * I^-1 * J^T. I^-1 is diagonal.
    // (rn.x * Ix, rn.y * Iy, rn.z * Iz) . rn
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
  // Target Velocity
  // We need v_old for restitution?
  // v_old is in d_vel_old (if we stored it).
  // Or we just approximate v_target = 0 (perfectly inelastic) + bias?

  // Implementation Plan said: Target v_n = -e * (v_old . n)
  // Where v_old comes from d_vel_old logic (integration.cu saved d_vel_old
  // which is v_initial_step). But we need relative velocity old. Do we have
  // w_old? integration.cu does NOT save d_ang_vel_old. Let's assume for now
  // restitution applies to LINEAR part or ignore angular restitution? Better:
  // Standard PBD often ignores angular restitution or uses current. Let's use
  // d_vel_old for linear, current ang vel for rotational approximation? Or just
  // solve for v_n = 0 if doubtful (restitution=0 for packing is fine!).

  // Let's compute actual restitution target if possible.
  // Use ps.d_vel[idA] (Read Only state is "old" state before update).
  // YES! ps.d_vel matches start of substep.
  float4 vA_old_w = ps.d_vel[idA];
  float4 wA_old_w = ps.d_ang_vel[idA];

  float3 vA_old = make_float3(vA_old_w.x, vA_old_w.y, vA_old_w.z);
  float3 wA_old = make_float3(wA_old_w.x, wA_old_w.y, wA_old_w.z);
  float3 vB_old = make_float3(0, 0, 0);
  float3 wB_old = make_float3(0, 0, 0);
  if (idB >= 0) {
    float4 vB_old_w_local = ps.d_vel[idB];
    float4 wB_old_w_local = ps.d_ang_vel[idB];
    vB_old = make_float3(vB_old_w_local.x, vB_old_w_local.y, vB_old_w_local.z);
    wB_old = make_float3(wB_old_w_local.x, wB_old_w_local.y, wB_old_w_local.z);
  }

  float3 vPA_old = vA_old + cross_product(wA_old, rA);
  float3 vPB_old = vB_old + cross_product(wB_old, rB);
  float vn_old = dot_product(vPA_old - vPB_old, n);

  float vn_target = -ps.restitution_normal * vn_old;
  if (vn_target < 0.0f)
    vn_target =
        0.0f; // Don't suck particles together (only bounce if separating?)
  // Restitution creates separation velocity.
  // If vn_old < 0 (approaching), target is positive (separating).

  // Threshold? If collision speed is low, set e=0 to prevent jitter.
  if (fabsf(vn_old) < 0.1f)
    vn_target = 0.0f; // Stable stacking threshold

  float dV = vn_target - vn;

  // If dV < 0 (we are already separating faster than target), do we correct?
  // For inequality constraint (contact), we only push.
  // If vn > vn_target (and vn_target >= 0), it means we are separating fast.
  // But we must NOT pull.
  // The total impulse J must be > 0 (repulsive).
  // In Jacobi, we usually compute full delta, but we need to Clamp accumulated
  // J. But we don't store J per constraint in this buffer (only dist,
  // friction_lambda). Simple Jacobi (Stateless): Clamp Lambda > 0.

  float lambda_n = dV / w_total_n;

  // Apply only repulsive Impulse
  // Check if total lambda > 0? No stored lambda in stateless pass effectively
  // assumes prev lambda=0. This is "One Shot" Jacobi if we don't store lambda.
  // If we want iteratively better, we should store lambda.
  // ps.d_contacts[idx].friction_lambda_n could be used for Normal Lambda too?
  // Let's assume stateless for Phase 1 (simpler).
  // If lambda_n < 0 -> set to 0. (Don't pull).

  if (lambda_n < 0.0f)
    lambda_n = 0.0f;

  float3 impulse_n =
      make_float3(n.x * lambda_n, n.y * lambda_n, n.z * lambda_n);

  // ----------------------------
  // Friction (Tangential)
  // ----------------------------
  // Tangent velocity
  float3 vt_vec = v_rel - make_float3(n.x * vn, n.y * vn, n.z * vn);
  float vt_len = sqrtf(dot_product(vt_vec, vt_vec));

  float3 impulse_t = make_float3(0, 0, 0);

  if (vt_len > 1e-6f) {
    float3 t =
        make_float3(vt_vec.x / vt_len, vt_vec.y / vt_len, vt_vec.z / vt_len);

    float wA_t = compute_generalized_inv_mass(rA, t, invMassA, invIA_vec);
    float wB_t = compute_generalized_inv_mass(rB, t, invMassB, invIB_vec);
    float w_total_t = wA_t + wB_t;

    // Target vt = 0 (Static Friction)
    // delta_vt = 0 - vt_len.

    float lambda_t = -vt_len / w_total_t;

    // Coulomb Clamping: |lambda_t| <= mu * lambda_n
    float max_friction = ps.friction_dynamic * lambda_n;
    if (lambda_t < -max_friction)
      lambda_t = -max_friction;
    if (lambda_t > max_friction)
      lambda_t = max_friction;

    impulse_t = make_float3(t.x * lambda_t, t.y * lambda_t, t.z * lambda_t);
  }

  // ----------------------------
  // Apply Impulses (Atomic)
  // ----------------------------
  float3 total_J = impulse_n + impulse_t;

  // Body A: +J
  // Delta V = J * invM
  // Delta W = I^-1 * (r x J)
  float3 dvA = make_float3(total_J.x * invMassA, total_J.y * invMassA,
                           total_J.z * invMassA);
  float3 rxJ_A = cross_product(rA, total_J);
  float3 dwA = make_float3(rxJ_A.x * invIA_vec.x, rxJ_A.y * invIA_vec.y,
                           rxJ_A.z * invIA_vec.z);

  atomicAddVector(ps.d_delta_vel + idA, make_float4(dvA.x, dvA.y, dvA.z, 0));
  atomicAddVector(ps.d_delta_ang_vel + idA,
                  make_float4(dwA.x, dwA.y, dwA.z, 0));
  atomicAdd(ps.d_constraint_counts + idA, 1);

  // Body B: -J (Only if not static)
  if (idB >= 0) {
    float3 dvB = make_float3(-total_J.x * invMassB, -total_J.y * invMassB,
                             -total_J.z * invMassB);
    float3 rxJ_B =
        cross_product(rB, make_float3(-total_J.x, -total_J.y, -total_J.z));
    float3 dwB = make_float3(rxJ_B.x * invIB_vec.x, rxJ_B.y * invIB_vec.y,
                             rxJ_B.z * invIB_vec.z);

    atomicAddVector(ps.d_delta_vel + idB, make_float4(dvB.x, dvB.y, dvB.z, 0));
    atomicAddVector(ps.d_delta_ang_vel + idB,
                    make_float4(dwB.x, dwB.y, dwB.z, 0));
    atomicAdd(ps.d_constraint_counts + idB, 1);
  }
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
