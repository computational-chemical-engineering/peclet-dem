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

// Manifold Velocity Solver
// ------------------------------------------------------------------
__global__ void solve_velocity_jacobi_kernel(ParticleSystemData ps, float nu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_manifolds = *ps.d_manifold_count;

  if (idx >= num_manifolds)
    return;

  ManifoldConstraint m = ps.d_manifolds[idx];

  // Activity Check: If num_points is 0 (filtered out), skip
  if (m.num_points <= 0)
    return;

  int idA = m.bodyA;
  int idB = m.bodyB;

  // Deduplication for Periodic Boundaries
  // (Real A, Ghost B) and (Real B, Ghost A) represent the same interaction.
  // We process only if real_ID(A) <= real_ID(B).

  int realA = ps.d_real_indices[idA];
  int realB = idB; // Default for non-periodic/boundary
  if (idB >= 0) {
    realB = ps.d_real_indices[idB];
    if (realA > realB)
      return;
  }

  // Load Masses/Inertia
  float invMassA = ps.d_pos[realA].w;
  float invMassB = 0.0f;
  if (idB >= 0) // Check existence
    invMassB = ps.d_pos[realB].w;

  float3 invIA =
      make_float3(ps.d_inv_inertia[realA].x, ps.d_inv_inertia[realA].y,
                  ps.d_inv_inertia[realA].z);
  float3 invIB = make_float3(0, 0, 0);
  if (idB >= 0)
    invIB = make_float3(ps.d_inv_inertia[realB].x, ps.d_inv_inertia[realB].y,
                        ps.d_inv_inertia[realB].z);

  // Load Orientation
  // Use REAL orientation for consistency (should be identical, but safe)
  float4 qA = ps.d_quat[realA];
  float4 qB = make_float4(0, 0, 0, 1);
  if (idB >= 0)
    qB = ps.d_quat[realB];

  // 1. Get Velocities (Linear + Angular)
  // CRITICAL: Read from REAL index to get latest iterative updates
  float4 vA_w = ps.d_vel_pred[realA];
  float4 wA_w = ps.d_ang_vel_pred[realA];

  float3 vA = make_float3(vA_w.x, vA_w.y, vA_w.z);
  float3 wA = make_float3(wA_w.x, wA_w.y, wA_w.z);
  float3 vB = make_float3(0, 0, 0);
  float3 wB = make_float3(0, 0, 0);
  if (idB >= 0) {
    float4 vB_w = ps.d_vel_pred[realB];
    float4 wB_w = ps.d_ang_vel_pred[realB];
    vB = make_float3(vB_w.x, vB_w.y, vB_w.z);
    wB = make_float3(wB_w.x, wB_w.y, wB_w.z);
  }

  // Load Geometric Sums
  float3 N_sum = make_float3(m.normal_sum.x, m.normal_sum.y, m.normal_sum.z);
  float3 TauA_sum = make_float3(m.torque_armA_sum.x, m.torque_armA_sum.y,
                                m.torque_armA_sum.z);
  float3 TauB_sum = make_float3(m.torque_armB_sum.x, m.torque_armB_sum.y,
                                m.torque_armB_sum.z);

  // Reconstruct Aggregate Relative Velocity
  float termA = dot_product(vA, N_sum) + dot_product(wA, TauA_sum);
  float termB = dot_product(vB, make_float3(-N_sum.x, -N_sum.y, -N_sum.z)) +
                dot_product(wB, TauB_sum);
  float vn_agg = termA + termB;

  // Check Approaching
  // vn_agg = (vA - vB) . N
  // Approaching if vn_agg > 0 (e.g. A-> +5, B-> -5, N=+1 => 10)
  // Separating if vn_agg < 0
  if (vn_agg < 0) {
    return; // Separating
  }

  // Compute Effective Mass W
  // Helper for Quad term: v^T I^-1 v
  // I^-1_world = R I^-1_local R^T
  // v^T R I^-1_local R^T v = (R^T v)^T I^-1_local (R^T v)
  auto compute_generalized_inv_mass = [&](float3 tau, float3 invI_local,
                                          float4 q) {
    float3 tau_local = inv_rotate_vector(q, tau);
    return tau_local.x * tau_local.x * invI_local.x +
           tau_local.y * tau_local.y * invI_local.y +
           tau_local.z * tau_local.z * invI_local.z;
  };

  // Recalculate w_n with standard Tau Sums
  float wA_n = dot_product(N_sum, N_sum) * invMassA +
               compute_generalized_inv_mass(TauA_sum, invIA, qA);
  float wB_n = dot_product(N_sum, N_sum) * invMassB +
               compute_generalized_inv_mass(TauB_sum, invIB, qB);
  float w_total = wA_n + wB_n;

  if (w_total <= 0)
    return;

  // Target: V_new = -e * V_old
  float vn_target = -ps.restitution_normal * vn_agg;
  float dV = vn_target - vn_agg;
  float lambda = dV / w_total;

  // Apply Normal Impulse
  float3 J_lin_A =
      make_float3(N_sum.x * lambda, N_sum.y * lambda, N_sum.z * lambda);
  float3 J_ang_A = make_float3(TauA_sum.x * lambda, TauA_sum.y * lambda,
                               TauA_sum.z * lambda);

  float3 J_lin_B = make_float3(-J_lin_A.x, -J_lin_A.y, -J_lin_A.z);
  float3 J_ang_B = make_float3(TauB_sum.x * lambda, TauB_sum.y * lambda,
                               TauB_sum.z * lambda);

  // Atomic Application
  // Redirect to Real ID for Consistency (Momentum Conservation)
  int acc_idA = ps.d_real_indices[idA];

  atomicAddVector(ps.d_delta_vel + acc_idA,
                  make_float4(J_lin_A.x * invMassA, J_lin_A.y * invMassA,
                              J_lin_A.z * invMassA, 0));

  // Angular Update: dw = I_world^-1 * J_ang
  // dw = R * (I_local^-1 * (R^T * J_ang))
  auto apply_ang_impulse = [&](int id, float3 J_ang, float3 invI_local,
                               float4 q) {
    float3 J_local = inv_rotate_vector(q, J_ang);
    float3 dw_local =
        make_float3(J_local.x * invI_local.x, J_local.y * invI_local.y,
                    J_local.z * invI_local.z);
    float3 dw_world = rotate_vector(q, dw_local);
    atomicAddVector(ps.d_delta_ang_vel + id,
                    make_float4(dw_world.x, dw_world.y, dw_world.z, 0));
  };

  apply_ang_impulse(acc_idA, J_ang_A, invIA, qA);

  if (idB >= 0) {
    int acc_idB = ps.d_real_indices[idB];
    atomicAddVector(ps.d_delta_vel + acc_idB,
                    make_float4(J_lin_B.x * invMassB, J_lin_B.y * invMassB,
                                J_lin_B.z * invMassB, 0));
    apply_ang_impulse(acc_idB, J_ang_B, invIB, qB);
  }

  // -------------------------------------------------------------------------
  // Friction (Tangential) - Approximate
  // -------------------------------------------------------------------------
  // Construct Representative Point and Tangent Basis
  // Average rA, rB.
  float N_count = (float)m.num_points;
  float inv_N = 1.0f / N_count;
  float3 rA_avg =
      make_float3(m.rA_sum.x * inv_N, m.rA_sum.y * inv_N, m.rA_sum.z * inv_N);
  float3 rB_avg =
      make_float3(m.rB_sum.x * inv_N, m.rB_sum.y * inv_N, m.rB_sum.z * inv_N);

  // Normalized Normal Direction
  float len_N = sqrtf(dot_product(N_sum, N_sum));
  if (len_N < 1e-9f)
    return;
  float3 n = make_float3(N_sum.x / len_N, N_sum.y / len_N, N_sum.z / len_N);

  // Relative Velocity at Average Point
  // v_rel_point = (vA + wA x rA_avg) - (vB + wB x rB_avg)
  float3 vPA = vA + cross_product(wA, rA_avg);
  float3 vPB = vB + cross_product(wB, rB_avg);
  float3 v_rel_point = make_float3(vPA.x - vPB.x, vPA.y - vPB.y, vPA.z - vPB.z);

  // Tangent Velocity
  float vn_point = dot_product(v_rel_point, n);
  float3 vt_vec = make_float3(v_rel_point.x - vn_point * n.x,
                              v_rel_point.y - vn_point * n.y,
                              v_rel_point.z - vn_point * n.z);
  float vt_len = sqrtf(dot_product(vt_vec, vt_vec));

  if (vt_len > 1e-5f) {
    float3 t =
        make_float3(vt_vec.x / vt_len, vt_vec.y / vt_len, vt_vec.z / vt_len);

    // Generalized Mass for Tangent
    // J_t acting at Average Point along t
    // Force A: t. Torque A: rA_avg x t.
    // Force B: -t. Torque B: rB_avg x -t.
    float3 tauA_t = cross_product(rA_avg, t);
    float3 tauB_t = cross_product(rB_avg, make_float3(-t.x, -t.y, -t.z));

    float wA_t = invMassA + compute_generalized_inv_mass(tauA_t, invIA, qA);
    float wB_t = invMassB + compute_generalized_inv_mass(tauB_t, invIB, qB);
    float w_total_t = wA_t + wB_t;

    // Target
    float dv_t_mag =
        (ps.restitution_tangent - 1.0f) * vt_len; // -V_old if inelastic
    float lambda_t = dv_t_mag / w_total_t;

    // Friction Clamp (Coulomb)
    float max_f = ps.friction_dynamic * lambda;
    // Note: lambda > 0 usually.
    if (lambda_t < -max_f)
      lambda_t = -max_f;
    if (lambda_t > max_f)
      lambda_t = max_f;

    // Apply
    float3 J_lin_t_A =
        make_float3(t.x * lambda_t, t.y * lambda_t, t.z * lambda_t);
    float3 J_ang_t_A = make_float3(tauA_t.x * lambda_t, tauA_t.y * lambda_t,
                                   tauA_t.z * lambda_t);

    float3 J_lin_t_B = make_float3(-J_lin_t_A.x, -J_lin_t_A.y, -J_lin_t_A.z);
    float3 J_ang_t_B = make_float3(tauB_t.x * lambda_t, tauB_t.y * lambda_t,
                                   tauB_t.z * lambda_t);

    // Map from possible Ghost ID to Real ID for Accumulation
    int acc_idA = ps.d_real_indices[idA];

    atomicAddVector(ps.d_delta_vel + acc_idA,
                    make_float4(J_lin_t_A.x * invMassA, J_lin_t_A.y * invMassA,
                                J_lin_t_A.z * invMassA, 0));

    apply_ang_impulse(
        acc_idA, J_ang_t_A, invIA,
        qA); // Using original qA/invIA is correct, but accumulate to acc_idA

    if (idB >= 0) {
      int acc_idB = ps.d_real_indices[idB];
      atomicAddVector(ps.d_delta_vel + acc_idB,
                      make_float4(J_lin_t_B.x * invMassB,
                                  J_lin_t_B.y * invMassB,
                                  J_lin_t_B.z * invMassB, 0));
      apply_ang_impulse(acc_idB, J_ang_t_B, invIB, qB);
    }
  }
}

void launch_velocity_solve(ParticleSystemData ps, float nu) {
  int num_manifolds;
  CUDA_CHECK(cudaMemcpy(&num_manifolds, ps.d_manifold_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (num_manifolds == 0)
    return;

  int threads = 256;
  int blocks = (num_manifolds + threads - 1) / threads;

  solve_velocity_jacobi_kernel<<<blocks, threads>>>(ps, nu);
  CUDA_CHECK(cudaGetLastError());
}
