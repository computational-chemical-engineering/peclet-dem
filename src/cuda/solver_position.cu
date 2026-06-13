#include "ParticleSystem.cuh"
#include "math_utils.cuh"
#include "memory_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ float3 cross_product_p(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ float dot_product_p(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void solve_position_jacobi_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_contacts = *ps.d_contact_count;

  if (idx >= num_contacts)
    return;

  // Load Contact
  ContactConstraint c = ps.d_contacts[idx];

  int idA = c.bodyA;
  int idB = c.bodyB;

  float invMassA = ps.d_pos[idA].w; // w is inv_mass
  float invMassB = 0.0f;
  if (idB >= 0)
    invMassB = ps.d_pos[idB].w;

  // Load Predicted Positions/Orientations
  float4 pA_w = ps.d_pos_pred[idA];
  float4 qA = ps.d_quat_pred[idA]; // Restore qA
  float4 pB_w = make_float4(0, 0, 0, 0);
  float4 qB = make_float4(0, 0, 0, 1);
  if (idB >= 0) {
    pB_w = ps.d_pos_pred[idB];
    qB = ps.d_quat_pred[idB];
  } else {
    // Evaluate ground point?
    // We know rB from constraint. pB = ???
    // For ground, pB is usually irrelevant if we just use normal/dist?
    // But we compute overlap = |pA - pB| - ...
    // We need an anchor point for B.
    // Let anchor be pA_contact - diff?
    // If dist > 0, we can use pA - normal * dist?
    // Actually, if idB < 0, pB_w is effectively "Point on Wall".
    // Let's assume prediction doesn't move Wall.
    // Use constraint.rB to find original pB?
    // Actually, narrowphase computed dist.
    // If we want new dist, we need pA and Wall Geometry.
    // If idB = -1, imply "Simple Ground Plane at y=min.y"?
    // Or generic?
    // Let's assume idB < 0 means "use stored Constraint rB/normal and
    // don't re-eval C(x) from positions if we can't"? But we MUST
    // re-eval C(x) to perform projection properly. If Wall, C(x) = (pA.y
    // - radius) - GroundY. We can check if idB == -1 (Ground) and
    // compute C(x) explicitly. Or store "Wall" type. For Phase 1,
    // special case idB == -1 as Ground at domain_min.y.
  }

  float3 pA = make_float3(pA_w.x, pA_w.y, pA_w.z);
  float3 pB = make_float3(pB_w.x, pB_w.y, pB_w.z);

  // 1. Recover Vectors and Optimize Rotation (Delta Quaternion)
  // Logic: dq = q_pred * inv(q_static). r_curr = rotate(dq, r_init).

  float4 qA_static = ps.d_quat[idA];
  float4 qA_delta = quat_mult(qA, quat_inverse(qA_static)); // qA is d_quat_pred

  float3 rA_vec_init = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 rA_curr = rotate_vector(qA_delta, rA_vec_init);

  float3 rB_curr = make_float3(c.rB.x, c.rB.y, c.rB.z);
  float3 n_curr =
      make_float3(c.normal.x, c.normal.y, c.normal.z); // Default n_static

  if (idB >= 0) {
    float4 qB_static = ps.d_quat[idB];
    float4 qB_delta =
        quat_mult(qB, quat_inverse(qB_static)); // qB is d_quat_pred

    rB_curr = rotate_vector(qB_delta, rB_curr);
    n_curr = rotate_vector(qB_delta, n_curr);
  }

  // 3. Update variables
  float3 rA_vec = rA_curr;
  float3 rB_vec = rB_curr;
  float3 n = n_curr;

  // Special Ground Handling for idB < 0
  float C = 0.0f;
  // float3 n = make_float3(0, 1, 0); // Removed local var declaration to use
  // updated 'n'

  if (idB < 0) {
    // n is already set to c.normal (static) above because idB < 0 block skipped
    // rotation So n is correct for static wall.
    // Generic Static Wall/Plane Logic
    // Uses data stored in ContactConstraint:
    // c.normal: Normal pointing B->A
    // c.rB: Point on the wall/plane (Anchor)
    // C(x) = dot(pA - pB, n) - radiusA

    n = make_float3(c.normal.x, c.normal.y, c.normal.z);

    // Anchor Point on Wall
    float3 pB_anchor = make_float3(c.rB.x, c.rB.y, c.rB.z);

    // Surface Point on A
    float3 pA_surf =
        make_float3(pA.x + rA_vec.x, pA.y + rA_vec.y, pA.z + rA_vec.z);

    // Vector from Anchor to Surface A
    float3 diff = make_float3(pA_surf.x - pB_anchor.x, pA_surf.y - pB_anchor.y,
                              pA_surf.z - pB_anchor.z);

    C = dot_product_p(diff, n);

    if (C >= 0.0f)
      return;
  } else {
    // Generic Linearized Constraint (Valid for Spheres, Cylinders, SDFs)
    // C(x) = dot( (pA + rA) - (pB + rB), n )
    // We use the stored contact normal 'n' and lever arms 'rA','rB' from
    // Narrowphase.

    n = make_float3(c.normal.x, c.normal.y, c.normal.z);

    float3 pA_contact =
        make_float3(pA.x + rA_vec.x, pA.y + rA_vec.y, pA.z + rA_vec.z);
    float3 pB_contact =
        make_float3(pB.x + rB_vec.x, pB.y + rB_vec.y, pB.z + rB_vec.z);

    float3 separation_vec =
        make_float3(pA_contact.x - pB_contact.x, pA_contact.y - pB_contact.y,
                    pA_contact.z - pB_contact.z);

    C = dot_product_p(separation_vec, n);

    if (C >= 0.0f)
      return; // Separated or touching
  }

  // Generalized Inverse Mass
  // w = invM + (r x n) I^-1 (r x n)
  // Use the actual current lever arms (which include rotation)

  float3 cur_rA = rA_vec;
  float3 cur_rB = rB_vec;

  float3 invIA = make_float3(ps.d_inv_inertia[idA].x, ps.d_inv_inertia[idA].y,
                             ps.d_inv_inertia[idA].z);
  float3 invIB = make_float3(0, 0, 0);
  if (idB >= 0)
    invIB = make_float3(ps.d_inv_inertia[idB].x, ps.d_inv_inertia[idB].y,
                        ps.d_inv_inertia[idB].z);

  auto compute_w = [&](float3 r, float3 dir, float invM, float3 invI) {
    float3 rn = cross_product_p(r, dir);
    return invM + rn.x * rn.x * invI.x + rn.y * rn.y * invI.y +
           rn.z * rn.z * invI.z;
  };

  float wA = compute_w(cur_rA, n, invMassA, invIA);
  float wB = compute_w(cur_rB, n, invMassB, invIB);
  float w_total = wA + wB;

  if (w_total < 1e-6f)
    return;

  // XPBD: Delta Lambda = -C / (w_total + alpha_tilde)
  // Compliance alpha = 0 for rigid.
  // alpha_tilde = alpha / dt^2.
  float alpha_tilde = 0.0f;

  float dLambda = -C / (w_total + alpha_tilde);

  // Apply Delta Position
  float3 dpA = make_float3(n.x * dLambda * invMassA, n.y * dLambda * invMassA,
                           n.z * dLambda * invMassA);
  float3 dpB = make_float3(-n.x * dLambda * invMassB, -n.y * dLambda * invMassB,
                           -n.z * dLambda * invMassB);

  atomicAddVector(ps.d_delta_pos + idA, make_float4(dpA.x, dpA.y, dpA.z, 0));

  // Orientation Update
  // dTheta = I^-1 * (r x (dLambda * n))
  // dTheta = dLambda * I^-1 * (r x n)
  float3 rnA = cross_product_p(cur_rA, n);
  float3 dThetaA =
      make_float3(rnA.x * invIA.x * dLambda, rnA.y * invIA.y * dLambda,
                  rnA.z * invIA.z * dLambda);

  // Convert dTheta to dQuat
  // dq = 0.5 * (0, dTheta) * q
  // This is accumulation. d_delta_quat will be averaged.
  // q_new ~= q + 0.5 * dt * w * q. Here dTheta acts as "displacement
  // angle". d_quat_correction = 0.5 * dTheta * q. Wait, d_delta_quat is
  // added to q_pred. We compute dq value.
  float4 dqA;
  dqA.x = 0.5f * (dThetaA.x * qA.w + dThetaA.y * qA.z - dThetaA.z * qA.y);
  dqA.y = 0.5f * (dThetaA.y * qA.w + dThetaA.z * qA.x - dThetaA.x * qA.z);
  dqA.z = 0.5f * (dThetaA.z * qA.w + dThetaA.x * qA.y - dThetaA.y * qA.x);
  dqA.w = 0.5f * (-dThetaA.x * qA.x - dThetaA.y * qA.y - dThetaA.z * qA.z);

  atomicAddVector(ps.d_delta_quat + idA, dqA);

  // Body B
  if (idB >= 0) {
    float3 rnB_val = cross_product_p(cur_rB, n);
    float3 dThetaB = make_float3(-rnB_val.x * invIB.x * dLambda,
                                 -rnB_val.y * invIB.y * dLambda,
                                 -rnB_val.z * invIB.z * dLambda);

    float4 dqB;
    dqB.x = 0.5f * (dThetaB.x * qB.w + dThetaB.y * qB.z - dThetaB.z * qB.y);
    dqB.y = 0.5f * (dThetaB.y * qB.w + dThetaB.z * qB.x - dThetaB.x * qB.z);
    dqB.z = 0.5f * (dThetaB.z * qB.w + dThetaB.x * qB.y - dThetaB.y * qB.x);
    dqB.w = 0.5f * (-dThetaB.x * qB.x - dThetaB.y * qB.y - dThetaB.z * qB.z);

    atomicAddVector(ps.d_delta_quat + idB, dqB);
    atomicAddVector(ps.d_delta_pos + idB, make_float4(dpB.x, dpB.y, dpB.z, 0));
    atomicAdd(ps.d_constraint_counts + idB, 1);
  }

  // Count (only if active?)
  // Yes, if we added deltas, we increment.
  atomicAdd(ps.d_constraint_counts + idA, 1);

  // (Tangential friction is NOT done here: by design the position solve does pure overlap removal and all
  // dissipation -- including friction -- lives in the velocity solve. See launch_contact_friction.)

  // Track Max Overlap (Penetration = -C)
  // C is typically negative for overlap. So Penetration = -C.
  // We want to track the MAXIMUM penetration.
  if (C < 0.0f) {
    float penetration = -C;
    atomicMaxFloat(ps.d_max_overlap, penetration);
  }
}

void launch_position_solve(ParticleSystemData ps) {
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (num_contacts == 0)
    return;

  int threads = 256;
  int blocks = (num_contacts + threads - 1) / threads;

  solve_position_jacobi_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

// ===================================================================================================
// The single dissipative friction mechanism: per-contact Coulomb friction in the VELOCITY solve.
// Runs on the post-gravity-predict velocity, BEFORE the manifold normal solve, so the Coulomb cone is
// bounded by the VELOCITY-LEVEL normal impulse  lambda_n = max(0, approach)/w_n  -- the clean, correctly
// scaled normal force. For a resting contact the per-step approach is the external (gravity + growth) load,
// so lambda_n is the load impulse and friction gives stick (mu>=tan th) / slip a=g(sin th - mu cos th); for
// an impact lambda_n is the collisional impulse (so the same code is collisional + static friction).
// Strictly velocity-only -- no read of / coupling to the position solve (design principle: all dissipation
// lives in the velocity solve, the position solve does pure overlap removal). Gated on friction>0.
//
// Coincident plane (idB<0) contacts (a flat face on a plane) are Jacobi-averaged by their per-body count and
// bounded by the body's aggregate plane load, so a face brakes as one clean force, not N summed impulses
// (which would diverge). Particle-particle (idB>=0) is per-contact and momentum-conserving: equal/opposite
// on realA/realB, deduplicated across periodic ghosts (realA>realB skipped) exactly like the manifold solver.
// ===================================================================================================
__device__ inline float fric_w_dir(float3 r, float3 dir, float invM, float3 invI) {
  float3 rn = cross_product_p(r, dir);
  return invM + rn.x * rn.x * invI.x + rn.y * rn.y * invI.y + rn.z * rn.z * invI.z;
}

// Run inside the velocity-solve iteration loop: ACCUMULATE each contact's normal impulse lambda_n over the
// iterations. Each iteration the approach is the residual the manifold normal solve has not yet cancelled,
// so the running sum telescopes to the contact's FULL normal load -- crucially, the load propagated through a
// FORCE CHAIN (a deep contact bearing the column above keeps seeing residual approach as the load
// propagates over iterations). That converged per-contact impulse is the "full resting normal impulse" the
// velocity solve carries, and the only thing that bounds friction. A periodic-ghost duplicate (realA>realB)
// is skipped so the canonical copy owns the load.
__global__ void accumulate_normal_impulse_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= *ps.d_contact_count)
    return;
  ContactConstraint c = ps.d_contacts[idx];
  int idA = c.bodyA, idB = c.bodyB;
  if (idB < 0)
    return; // plane/wall contacts: single-layer rigid faces, bounded by the one-shot load (compute_plane_load)
  int realA = ps.d_real_indices[idA];
  int realB = ps.d_real_indices[idB];
  if (realA > realB)
    return;
  float invMassA = ps.d_pos[realA].w;
  float invMassB = ps.d_pos[realB].w;
  float3 invIA = make_float3(ps.d_inv_inertia[realA].x, ps.d_inv_inertia[realA].y, ps.d_inv_inertia[realA].z);
  float3 invIB = make_float3(ps.d_inv_inertia[realB].x, ps.d_inv_inertia[realB].y, ps.d_inv_inertia[realB].z);
  float3 rA = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 rB = make_float3(c.rB.x, c.rB.y, c.rB.z);
  float3 n = make_float3(c.normal.x, c.normal.y, c.normal.z);

  float3 vA = make_float3(ps.d_vel_pred[realA].x, ps.d_vel_pred[realA].y, ps.d_vel_pred[realA].z);
  float3 wA = make_float3(ps.d_ang_vel_pred[realA].x, ps.d_ang_vel_pred[realA].y, ps.d_ang_vel_pred[realA].z);
  float3 vB = make_float3(ps.d_vel_pred[realB].x, ps.d_vel_pred[realB].y, ps.d_vel_pred[realB].z);
  float3 wB = make_float3(ps.d_ang_vel_pred[realB].x, ps.d_ang_vel_pred[realB].y, ps.d_ang_vel_pred[realB].z);
  float3 vAc = make_float3(vA.x + cross_product_p(wA, rA).x, vA.y + cross_product_p(wA, rA).y,
                           vA.z + cross_product_p(wA, rA).z);
  float3 vBc = make_float3(vB.x + cross_product_p(wB, rB).x, vB.y + cross_product_p(wB, rB).y,
                           vB.z + cross_product_p(wB, rB).z);
  // growth: growing particles push their contacts together (same load the normal solve sees)
  float3 vrel = make_float3(vAc.x - vBc.x + ps.growth_rate * (rA.x - rB.x),
                            vAc.y - vBc.y + ps.growth_rate * (rA.y - rB.y),
                            vAc.z - vBc.z + ps.growth_rate * (rA.z - rB.z));
  float approach = -dot_product_p(vrel, n); // approaching if > 0
  if (approach <= 0.0f)
    return;
  float w_n = fric_w_dir(rA, n, invMassA, invIA) + fric_w_dir(rB, n, invMassB, invIB);
  if (w_n > 1e-6f)
    ps.d_contacts[idx].friction_lambda_n += approach / w_n; // accumulate the chain load over iterations
}

// Run ONCE before the velocity loop: plane (idB<0) contacts are single-layer rigid faces, so their normal
// load is the ONE-SHOT external (gravity) load -- approach*m (linear; a flat face carries the weight as a net
// force, not a torque) -- NOT a chain accumulation (a flat face's coplanar contacts converge poorly in the
// manifold solve, so accumulating would re-count the residual and over-hold). The body load is the MAX over
// the coplanar contacts; the solve averages the applied impulse by the count so the summed friction ==
// mu*(body load). Also marks each plane contact active (c.friction_lambda_n>0) so the solve processes it.
__global__ void compute_plane_load_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= *ps.d_contact_count)
    return;
  ContactConstraint c = ps.d_contacts[idx];
  if (c.bodyB >= 0)
    return; // plane/wall contacts only
  int idA = c.bodyA;
  float invMassA = ps.d_pos[idA].w;
  if (invMassA <= 0.0f)
    return;
  float3 invIA = make_float3(ps.d_inv_inertia[idA].x, ps.d_inv_inertia[idA].y, ps.d_inv_inertia[idA].z);
  float3 rA = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 n = make_float3(c.normal.x, c.normal.y, c.normal.z);
  float3 vA = make_float3(ps.d_vel_pred[idA].x, ps.d_vel_pred[idA].y, ps.d_vel_pred[idA].z);
  float3 wA = make_float3(ps.d_ang_vel_pred[idA].x, ps.d_ang_vel_pred[idA].y, ps.d_ang_vel_pred[idA].z);
  float3 vAc = make_float3(vA.x + cross_product_p(wA, rA).x, vA.y + cross_product_p(wA, rA).y,
                           vA.z + cross_product_p(wA, rA).z);
  float approach = -dot_product_p(vAc, n); // approaching the plane if > 0
  if (approach <= 0.0f)
    return;
  float w_n = fric_w_dir(rA, n, invMassA, invIA);
  ps.d_contacts[idx].friction_lambda_n = (w_n > 1e-6f) ? approach / w_n : 0.0f; // active flag for the solve
  atomicMaxFloat(&ps.d_plane_friction[idA].x, approach / invMassA);             // body linear load (approach*m)
  atomicAdd(&ps.d_plane_friction[idA].y, 1.0f);
}

// Pass 2: drive the tangential relative velocity toward zero, Coulomb-clamped by mu*lambda_n (an impulse).
__global__ void solve_contact_friction_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= *ps.d_contact_count)
    return;
  ContactConstraint c = ps.d_contacts[idx];
  float lambda_n = c.friction_lambda_n;
  if (lambda_n <= 0.0f)
    return; // no approach -> no normal load -> no friction (and skips deduped duplicates)

  int idA = c.bodyA, idB = c.bodyB;
  int realA = ps.d_real_indices[idA];
  int realB = (idB >= 0) ? ps.d_real_indices[idB] : -1;
  float invMassA = ps.d_pos[realA].w;
  float invMassB = (idB >= 0) ? ps.d_pos[realB].w : 0.0f;
  float3 invIA = make_float3(ps.d_inv_inertia[realA].x, ps.d_inv_inertia[realA].y, ps.d_inv_inertia[realA].z);
  float3 invIB = make_float3(0, 0, 0);
  if (idB >= 0)
    invIB = make_float3(ps.d_inv_inertia[realB].x, ps.d_inv_inertia[realB].y, ps.d_inv_inertia[realB].z);
  float3 rA = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 rB = make_float3(c.rB.x, c.rB.y, c.rB.z);
  float3 n = make_float3(c.normal.x, c.normal.y, c.normal.z);

  float3 vA = make_float3(ps.d_vel_pred[realA].x, ps.d_vel_pred[realA].y, ps.d_vel_pred[realA].z);
  float3 wA = make_float3(ps.d_ang_vel_pred[realA].x, ps.d_ang_vel_pred[realA].y, ps.d_ang_vel_pred[realA].z);
  float3 vB = make_float3(0, 0, 0), wB = make_float3(0, 0, 0);
  if (idB >= 0) {
    vB = make_float3(ps.d_vel_pred[realB].x, ps.d_vel_pred[realB].y, ps.d_vel_pred[realB].z);
    wB = make_float3(ps.d_ang_vel_pred[realB].x, ps.d_ang_vel_pred[realB].y, ps.d_ang_vel_pred[realB].z);
  }
  float3 vAc = make_float3(vA.x + cross_product_p(wA, rA).x, vA.y + cross_product_p(wA, rA).y,
                           vA.z + cross_product_p(wA, rA).z);
  float3 vBc = make_float3(vB.x + cross_product_p(wB, rB).x, vB.y + cross_product_p(wB, rB).y,
                           vB.z + cross_product_p(wB, rB).z);
  float3 vrel = make_float3(vAc.x - vBc.x, vAc.y - vBc.y, vAc.z - vBc.z);
  float vn = dot_product_p(vrel, n);
  float3 vt = make_float3(vrel.x - vn * n.x, vrel.y - vn * n.y, vrel.z - vn * n.z);
  float vt_len = sqrtf(dot_product_p(vt, vt));
  if (vt_len < 1e-7f)
    return;
  float3 t = make_float3(vt.x / vt_len, vt.y / vt_len, vt.z / vt_len);

  float3 rnA = cross_product_p(rA, t), rnB = cross_product_p(rB, t);
  float w_t = invMassA + invMassB + rnA.x * rnA.x * invIA.x + rnA.y * rnA.y * invIA.y +
              rnA.z * rnA.z * invIA.z + rnB.x * rnB.x * invIB.x + rnB.y * rnB.y * invIB.y +
              rnB.z * rnB.z * invIB.z;
  if (w_t < 1e-6f)
    return;

  // Coulomb cone bounded by the velocity-level normal impulse (an impulse already -> no /dt). For a flat face
  // on a plane, bound by the body's aggregate plane load and average by the coplanar count (else N parallel
  // impulses Jacobi-sum to an N-fold overshoot). Single plane / particle-particle contacts: bound = lambda_n,
  // inv_n = 1.
  float bound = lambda_n;
  float inv_n = 1.0f;
  if (idB < 0) {
    float2 pf = ps.d_plane_friction[idA];
    bound = pf.x;
    if (pf.y > 1.5f)
      inv_n = 1.0f / pf.y;
  }
  float lt = -vt_len / w_t;        // drive v_t -> 0 (lt <= 0); |lt| <= vt_len/w_t => only damps (stable)
  float maxf = ps.friction_dynamic * bound;
  if (lt < -maxf)
    lt = -maxf; // dynamic slip if over the cone, else static stick
  lt *= inv_n;
  atomicAddVector(ps.d_delta_vel + realA,
                  make_float4(t.x * lt * invMassA, t.y * lt * invMassA, t.z * lt * invMassA, 0));
  atomicAddVector(ps.d_delta_ang_vel + realA,
                  make_float4(rnA.x * invIA.x * lt, rnA.y * invIA.y * lt, rnA.z * invIA.z * lt, 0));
  if (idB >= 0) {
    atomicAddVector(ps.d_delta_vel + realB,
                    make_float4(-t.x * lt * invMassB, -t.y * lt * invMassB, -t.z * lt * invMassB, 0));
    atomicAddVector(ps.d_delta_ang_vel + realB,
                    make_float4(-rnB.x * invIB.x * lt, -rnB.y * invIB.y * lt, -rnB.z * invIB.z * lt, 0));
  }
}

// Run ONCE before the velocity loop: compute the one-shot plane (idB<0) loads from the post-gravity approach.
void launch_plane_load(ParticleSystemData ps) {
  int threads = 256;
  CUDA_CHECK(cudaMemset(ps.d_plane_friction, 0, ps.num_particles * sizeof(float2)));
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (num_contacts > 0) {
    int cblocks = (num_contacts + threads - 1) / threads;
    compute_plane_load_kernel<<<cblocks, threads>>>(ps);
    CUDA_CHECK(cudaGetLastError());
  }
}

// Called every velocity-solve iteration (alongside the manifold normal solve) to accumulate each body-body
// contact's normal load (the force-chain load). c.friction_lambda_n is zeroed by the narrowphase.
void launch_accumulate_normal_impulse(ParticleSystemData ps) {
  int threads = 256;
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (num_contacts > 0) {
    int cblocks = (num_contacts + threads - 1) / threads;
    accumulate_normal_impulse_kernel<<<cblocks, threads>>>(ps);
    CUDA_CHECK(cudaGetLastError());
  }
}

// Run ONCE after the velocity loop: friction bounded by the per-contact normal load (body-body: the
// accumulated chain load; plane: the one-shot body load in d_plane_friction). Accumulates the friction
// impulse into d_delta_vel / d_delta_ang_vel; the caller applies it with launch_apply_velocity_deltas.
void launch_contact_friction(ParticleSystemData ps) {
  int threads = 256;
  CUDA_CHECK(cudaMemset(ps.d_delta_vel, 0, ps.num_particles * sizeof(float4)));
  CUDA_CHECK(cudaMemset(ps.d_delta_ang_vel, 0, ps.num_particles * sizeof(float4)));
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (num_contacts > 0) {
    int cblocks = (num_contacts + threads - 1) / threads;
    solve_contact_friction_kernel<<<cblocks, threads>>>(ps);
    CUDA_CHECK(cudaGetLastError());
  }
}
