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

  // Accumulate the normal Lagrange multiplier (>0 for an overlap) over the position iterations -> the
  // contact's total normal correction. Fix C reads it as the resting normal force that bounds friction in
  // the velocity feedback (mu * friction_lambda_n / dt). Gated on friction>0 (frictionless: no write).
  if (ps.friction_dynamic > 0.0f && dLambda > 0.0f)
    atomicAdd(&ps.d_contacts[idx].friction_lambda_n, dLambda);

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

  // ---- XPBD positional (static/dynamic) friction (Mueller et al. 2020) ----
  // A Coulomb-limited tangential POSITION correction that removes the tangential relative DISPLACEMENT of
  // the contact points since the contact formed (anchor = start-of-step state, ps.d_pos + the stored anchor
  // lever arms c.rA/c.rB). It is bounded by mu*|dLambda| -- the normal contact force this iteration -- so it
  // acts on PERSISTENT/resting contacts (where the velocity-solver friction is inert because its normal
  // impulse ~ 0). Gated on friction>0, so the frictionless path is byte-identical. Applied with the same
  // generalized-mass weighting and sign convention as the normal constraint (B negates linear + angular),
  // and averaged by the same per-contact constraint count (we do not increment it again).
  if (ps.friction_dynamic > 0.0f) {
    float3 pA0 = make_float3(ps.d_pos[idA].x, ps.d_pos[idA].y, ps.d_pos[idA].z);
    float3 dpf = make_float3((pA.x + cur_rA.x) - (pA0.x + c.rA.x),
                             (pA.y + cur_rA.y) - (pA0.y + c.rA.y),
                             (pA.z + cur_rA.z) - (pA0.z + c.rA.z));
    if (idB >= 0) {
      float3 pB0 = make_float3(ps.d_pos[idB].x, ps.d_pos[idB].y, ps.d_pos[idB].z);
      dpf.x -= (pB.x + cur_rB.x) - (pB0.x + c.rB.x);
      dpf.y -= (pB.y + cur_rB.y) - (pB0.y + c.rB.y);
      dpf.z -= (pB.z + cur_rB.z) - (pB0.z + c.rB.z);
    }
    float dpn = dot_product_p(dpf, n);
    float3 dpt = make_float3(dpf.x - dpn * n.x, dpf.y - dpn * n.y, dpf.z - dpn * n.z);
    float dpt_len = sqrtf(dot_product_p(dpt, dpt));
    if (dpt_len > 1e-8f) {
      float3 t = make_float3(dpt.x / dpt_len, dpt.y / dpt_len, dpt.z / dpt_len);
      float w_t = compute_w(cur_rA, t, invMassA, invIA) + compute_w(cur_rB, t, invMassB, invIB);
      if (w_t > 1e-6f) {
        float dLt = -dpt_len / w_t;                       // remove the tangential slip (dLt <= 0)
        float maxf = ps.friction_dynamic * fabsf(dLambda); // Coulomb: |f| <= mu * normal force
        if (dLt < -maxf)
          dLt = -maxf;                                    // slip (dynamic) if over the cone, else stick
        float3 dpA_t = make_float3(t.x * dLt * invMassA, t.y * dLt * invMassA, t.z * dLt * invMassA);
        atomicAddVector(ps.d_delta_pos + idA, make_float4(dpA_t.x, dpA_t.y, dpA_t.z, 0));
        float3 rnA_t = cross_product_p(cur_rA, t);
        float3 thA = make_float3(rnA_t.x * invIA.x * dLt, rnA_t.y * invIA.y * dLt, rnA_t.z * invIA.z * dLt);
        float4 dqA_t = make_float4(0.5f * (thA.x * qA.w + thA.y * qA.z - thA.z * qA.y),
                                   0.5f * (thA.y * qA.w + thA.z * qA.x - thA.x * qA.z),
                                   0.5f * (thA.z * qA.w + thA.x * qA.y - thA.y * qA.x),
                                   0.5f * (-thA.x * qA.x - thA.y * qA.y - thA.z * qA.z));
        atomicAddVector(ps.d_delta_quat + idA, dqA_t);
        if (idB >= 0) {
          float3 dpB_t = make_float3(-t.x * dLt * invMassB, -t.y * dLt * invMassB, -t.z * dLt * invMassB);
          atomicAddVector(ps.d_delta_pos + idB, make_float4(dpB_t.x, dpB_t.y, dpB_t.z, 0));
          float3 rnB_t = cross_product_p(cur_rB, t);
          float3 thB = make_float3(-rnB_t.x * invIB.x * dLt, -rnB_t.y * invIB.y * dLt,
                                   -rnB_t.z * invIB.z * dLt);
          float4 dqB_t = make_float4(0.5f * (thB.x * qB.w + thB.y * qB.z - thB.z * qB.y),
                                     0.5f * (thB.y * qB.w + thB.z * qB.x - thB.x * qB.z),
                                     0.5f * (thB.z * qB.w + thB.x * qB.y - thB.y * qB.x),
                                     0.5f * (-thB.x * qB.x - thB.y * qB.y - thB.z * qB.z));
          atomicAddVector(ps.d_delta_quat + idB, dqB_t);
        }
      }
    }
  }

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
// Fix C: friction VELOCITY feedback -- static friction for PERSISTENT contacts.
// A per-contact velocity impulse that removes the tangential relative velocity, Coulomb-bounded by
// mu * (normal force), where the normal force = friction_lambda_n/dt (the accumulated normal position
// correction of this step's position solve, ~ the resting contact load). PROVABLY STABLE at any dt: the
// impulse only DAMPS v_t (it targets v_t -> 0 and is clamped to |v_t|/w_t), so it can never amplify -- there
// is NO Delta-x/dt coupling (the unstable normal post-stabilization is deliberately NOT coupled to velocity).
// Tangential only -> the velocity-solver normal/tangential RESTITUTION is untouched. Run ONCE after the
// position loop. Gated on friction>0 (frictionless path unaffected).
// ===================================================================================================
__global__ void solve_friction_velocity_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= *ps.d_contact_count)
    return;
  ContactConstraint c = ps.d_contacts[idx];
  float fn = c.friction_lambda_n; // accumulated normal correction (>= 0)
  if (fn <= 0.0f)
    return; // no resting normal force -> no static friction here

  // Apply directly to bodyA/bodyB (like the position solve): bodyA is the real owner, bodyB may be a
  // periodic ghost (its slot is discarded; the ghost's real owner gets friction from its own contact). No
  // realA/realB remap and NO manifold-style dedup -- that dedup is for the manifold velocity solver and
  // would wrongly skip raw contacts.
  int idA = c.bodyA, idB = c.bodyB;

  float invMassA = ps.d_pos[idA].w;
  float invMassB = (idB >= 0) ? ps.d_pos[idB].w : 0.0f;
  float3 invIA = make_float3(ps.d_inv_inertia[idA].x, ps.d_inv_inertia[idA].y, ps.d_inv_inertia[idA].z);
  float3 invIB = make_float3(0, 0, 0);
  if (idB >= 0)
    invIB = make_float3(ps.d_inv_inertia[idB].x, ps.d_inv_inertia[idB].y, ps.d_inv_inertia[idB].z);
  float4 qA = ps.d_quat_pred[idA];
  float4 qB = make_float4(0, 0, 0, 1);
  if (idB >= 0)
    qB = ps.d_quat_pred[idB];

  // current (rotated) lever arms + normal, matching the position solve
  float4 qA_delta = quat_mult(qA, quat_inverse(ps.d_quat[idA]));
  float3 rA = rotate_vector(qA_delta, make_float3(c.rA.x, c.rA.y, c.rA.z));
  float3 rB = make_float3(c.rB.x, c.rB.y, c.rB.z);
  float3 n = make_float3(c.normal.x, c.normal.y, c.normal.z);
  if (idB >= 0) {
    float4 qB_delta = quat_mult(qB, quat_inverse(ps.d_quat[idB]));
    rB = rotate_vector(qB_delta, rB);
    n = rotate_vector(qB_delta, n);
  }

  // tangential relative velocity at the contact point
  float3 vA = make_float3(ps.d_vel_pred[idA].x, ps.d_vel_pred[idA].y, ps.d_vel_pred[idA].z);
  float3 wA = make_float3(ps.d_ang_vel_pred[idA].x, ps.d_ang_vel_pred[idA].y, ps.d_ang_vel_pred[idA].z);
  float3 vB = make_float3(0, 0, 0), wB = make_float3(0, 0, 0);
  if (idB >= 0) {
    vB = make_float3(ps.d_vel_pred[idB].x, ps.d_vel_pred[idB].y, ps.d_vel_pred[idB].z);
    wB = make_float3(ps.d_ang_vel_pred[idB].x, ps.d_ang_vel_pred[idB].y, ps.d_ang_vel_pred[idB].z);
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

  // tangential generalized inverse mass
  float3 rnA = cross_product_p(rA, t), rnB = cross_product_p(rB, t);
  float w_t = invMassA + invMassB + rnA.x * rnA.x * invIA.x + rnA.y * rnA.y * invIA.y +
              rnA.z * rnA.z * invIA.z + rnB.x * rnB.x * invIB.x + rnB.y * rnB.y * invIB.y +
              rnB.z * rnB.z * invIB.z;
  if (w_t < 1e-6f)
    return;

  // impulse to drive v_t -> 0, clamped to the Coulomb cone mu*(normal force = fn/dt). lt <= 0 opposes vt;
  // |lt| <= vt_len/w_t means we can only remove v_t (never reverse it) -> stable at any dt.
  float lt = -vt_len / w_t;
  float maxf = ps.friction_dynamic * (fn / ps.dt);
  if (lt < -maxf)
    lt = -maxf; // dynamic slip if over the cone, else static stick
  atomicAddVector(ps.d_delta_vel + idA,
                  make_float4(t.x * lt * invMassA, t.y * lt * invMassA, t.z * lt * invMassA, 0));
  atomicAddVector(ps.d_delta_ang_vel + idA,
                  make_float4(rnA.x * invIA.x * lt, rnA.y * invIA.y * lt, rnA.z * invIA.z * lt, 0));
  if (idB >= 0) {
    atomicAddVector(ps.d_delta_vel + idB,
                    make_float4(-t.x * lt * invMassB, -t.y * lt * invMassB, -t.z * lt * invMassB, 0));
    atomicAddVector(ps.d_delta_ang_vel + idB,
                    make_float4(-rnB.x * invIB.x * lt, -rnB.y * invIB.y * lt, -rnB.z * invIB.z * lt, 0));
  }
}

__global__ void apply_friction_velocity_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;
  float4 dv = ps.d_delta_vel[idx], dw = ps.d_delta_ang_vel[idx];
  float4 v = ps.d_vel[idx], w = ps.d_ang_vel[idx];
  ps.d_vel[idx] = make_float4(v.x + dv.x, v.y + dv.y, v.z + dv.z, v.w);
  ps.d_vel_pred[idx] = ps.d_vel[idx];
  ps.d_ang_vel[idx] = make_float4(w.x + dw.x, w.y + dw.y, w.z + dw.z, w.w);
  ps.d_ang_vel_pred[idx] = ps.d_ang_vel[idx];
  ps.d_delta_vel[idx] = make_float4(0, 0, 0, 0);
  ps.d_delta_ang_vel[idx] = make_float4(0, 0, 0, 0);
}

void launch_friction_velocity(ParticleSystemData ps) {
  int threads = 256;
  int pblocks = (ps.num_particles + threads - 1) / threads;
  CUDA_CHECK(cudaMemset(ps.d_delta_vel, 0, ps.num_particles * sizeof(float4)));
  CUDA_CHECK(cudaMemset(ps.d_delta_ang_vel, 0, ps.num_particles * sizeof(float4)));
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int), cudaMemcpyDeviceToHost));
  if (num_contacts > 0) {
    int cblocks = (num_contacts + threads - 1) / threads;
    solve_friction_velocity_kernel<<<cblocks, threads>>>(ps);
    CUDA_CHECK(cudaGetLastError());
  }
  apply_friction_velocity_kernel<<<pblocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}
