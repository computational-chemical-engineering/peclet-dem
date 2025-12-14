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

// Physics-Based Impulse with Restitution, Friction, and Expansion Velocity
// ------------------------------------------------------------------
__global__ void solve_velocity_jacobi_kernel(ParticleSystemData ps, float nu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_contacts = *ps.d_contact_count;

  if (idx >= num_contacts)
    return;
  ContactConstraint c = ps.d_contacts[idx];

  // STRICT MOMENTUM CONSERVATION (Current Overlap Filter)
  if (c.dist_current > 0.0f)
    return;

  int idA = c.bodyA;
  int idB = c.bodyB;

  // Load Masses/Inertia
  float invMassA = ps.d_pos[idA].w;
  float invMassB = 0.0f;
  if (idB >= 0)
    invMassB = ps.d_pos[idB].w;

  // 1. Get Velocities (Linear + Angular)
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

  // Load Positions for correct lever arms
  float4 posA_w = ps.d_pos[idA];
  float3 posA = make_float3(posA_w.x, posA_w.y, posA_w.z);
  float3 posB = make_float3(0, 0, 0); // Will be overwritten or unused

  // Contact Geometry
  float3 n = make_float3(c.normal.x, c.normal.y, c.normal.z);
  float3 rA = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 rB = make_float3(c.rB.x, c.rB.y, c.rB.z);

  // Correction for Angular Momentum Conservation:
  // Ensure impulses are applied at a COMMON point (average of surface points).
  // P_A = posA + rA
  // P_B = posB + rB
  // P_common = 0.5 * (P_A + P_B)
  // rA_common = P_common - posA
  // rB_common = P_common - posB

  if (idB >= 0) {
    float4 posB_w = ps.d_pos[idB];
    posB = make_float3(posB_w.x, posB_w.y, posB_w.z);

    float3 PA = make_float3(posA.x + rA.x, posA.y + rA.y, posA.z + rA.z);
    float3 PB = make_float3(posB.x + rB.x, posB.y + rB.y, posB.z + rB.z);
    float3 P_common = make_float3(0.5f * (PA.x + PB.x), 0.5f * (PA.y + PB.y),
                                  0.5f * (PA.z + PB.z));

    rA = make_float3(P_common.x - posA.x, P_common.y - posA.y,
                     P_common.z - posA.z);
    rB = make_float3(P_common.x - posB.x, P_common.y - posB.y,
                     P_common.z - posB.z);
  } else {
    // Static collision (B is NULL/World)
    // Usually rB is relative to origin(0,0,0) or contact point?
    // In static case, rB in constraint is usually '0' relative to something or
    // contact point itself? For now, let's assume P_B is "World Point" which
    // matches P_A visually? Actually if B is static wall, P_B is fixed.
    // Conservation of L for system [A + Earth] isn't tracked.
    // So mainly important for A-B dynamic.
  }

  // Relative Velocity at Contact Point
  float3 vPA = vA + cross_product(wA, rA);
  float3 vPB = vB + cross_product(wB, rB);

  // Expansion Velocity: v_surf = v + w x r + nu * r
  float3 vExpA = make_float3(rA.x * nu, rA.y * nu, rA.z * nu);
  float3 vExpB = make_float3(rB.x * nu, rB.y * nu, rB.z * nu);
  vPA = make_float3(vPA.x + vExpA.x, vPA.y + vExpA.y, vPA.z + vExpA.z);
  vPB = make_float3(vPB.x + vExpB.x, vPB.y + vExpB.y, vPB.z + vExpB.z);

  float3 v_rel = make_float3(vPA.x - vPB.x, vPA.y - vPB.y, vPA.z - vPB.z);
  float vn = dot_product(v_rel, n);

  // Activity Check: Only apply if approaching (vn < 0)
  if (vn >= 0.0f) {
    return;
  }

  // Rigorous Weighting (Pre-computed by Pair-Sort)
  float weight = c.weight;

  // Calculate Effective Mass components
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

  if (w_total_n < 1e-9f)
    return;

  // ----------------------------
  // Normal Impulse (Restitution)
  // ----------------------------
  // Target: v_new = -e * v_old
  float vn_target = -ps.restitution_normal * vn;

  float dV = vn_target - vn;
  float lambda_n = dV / w_total_n;
  // J_n should be positive (pushing apart). dV is positive (since vn < 0).

  // Filter/Clamp? Normal Force is repulsive.
  if (lambda_n < 0.0f)
    lambda_n = 0.0f; // Should not happen given vn < 0 check.

  // Scale by Weight
  float3 impulse_n =
      make_float3(n.x * lambda_n * weight, n.y * lambda_n * weight,
                  n.z * lambda_n * weight);

  // ----------------------------
  // Tangential (Friction + Restitution)
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

    // USER DEFINITION FOR TANGENTIAL RESTITUTION:
    // e_t = 1.0  -> No Friction/Change (v' = v)
    // e_t = 0.0  -> Fully Inelastic/Stick (v' = 0)
    // e_t = -1.0 -> Reflection (v' = -v)
    //
    // Target Delta v:
    // v' = e_t * v
    // dv = v' - v = (e_t - 1) * v

    float dv_t_mag = (ps.restitution_tangent - 1.0f) *
                     vt_len; // Negative value for e_t < 1.0

    float lambda_t_raw = dv_t_mag / w_total_t;

    // Friction Limit
    // Use mu based on velocity heuristic?
    float mu = (vt_len < 1e-4f)
                   ? ps.friction_dynamic
                   : ps.friction_dynamic; // TODO: ps.friction_static

    float max_friction = mu * lambda_n;

    // Clamp magnitude
    if (lambda_t_raw < -max_friction)
      lambda_t_raw = -max_friction;
    if (lambda_t_raw > max_friction)
      lambda_t_raw = max_friction; // Should not happen?

    // Apply Weight
    impulse_t =
        make_float3(t.x * lambda_t_raw * weight, t.y * lambda_t_raw * weight,
                    t.z * lambda_t_raw * weight);
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

void launch_velocity_solve(ParticleSystemData ps, float nu) {
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
  solve_velocity_jacobi_kernel<<<blocks, threads>>>(ps, nu);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}
