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
  // printf("Solver Pos: Contact %d: A=%d B=%d Dist=%f\n", idx, c.bodyA,
  // c.bodyB, c.dist);

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

  // Original contact data was computed at start of narrowphase (using
  // pos_pred then). If we iterate position solve, pos_pred changes. We
  // should re-evaluate violation C(x).

  // Re-compute Lever Arms?
  // rA and rB in world space change if object rotates/moves.
  // However, for small steps, using original rA/rB is often sufficient
  // (and stable). Updating rA/rB strictly requires re-transforming local
  // points. But we stored World rA/rB in constraint. We don't have Local
  // rA/rB readily available in ContactConstraint struct? Narrowphase
  // calculated them. If we want accurate XPBD, we need current rA.
  // Approximation: Use stored rA/rB but offset by current position
  // shift? No, rotation matters. For Phase 1 (Spheres), rA is just
  // Radius * Normal? Normal also changes? For spheres, contact point is
  // always on line connecting centers. So we can re-compute everything
  // from pA, pB.

  float3 diff = make_float3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
  float dist_sq = dot_product_p(diff, diff);
  float dist = sqrtf(dist_sq);

  // Check overlap
  // We need Radii.
  // ps.d_scale is uniform.
  // We assume spheres for Phase 1 logic inside this kernel?
  // Ideally solver is generic.
  // If generic, we must trust constraint data or have means to update
  // it. Generic XPBD usually transforms local contact points to world
  // using current q. We didn't store local points in ContactConstraint.
  // We stored rA (vector from center). Let's assume rA rotates with
  // body. But for Sphere, rA should point to other body! If we treat it
  // as generic, we rotate rA_initial by (q_current * q_initial_inv). Too
  // complex for Phase 1. Re-evaluating Sphere collision is cheap. BUT
  // Plan says "Position Solver ... Input: ConstraintBuffer (Reuse!)".
  // And "Re-evaluate Overlap: C(x) = dist - current_separation". Wait,
  // `dist` in constraint is the *initial* overlap. `current_separation`
  // ... If we move particles apart, `current_dist` increases. New
  // Overlap = Initial Overlap + (Change in separation). Project delta_x
  // to cancel New Overlap.

  // For Spheres:
  // C(x) = |pA - pB| - (rA + rB).
  // If C(x) >= 0 -> Separated.

  // Use Pre-calculated radii from constraint?
  // We don't have radii in constraint. We have `c.dist` which was `|pA0
  // - pB0|
  // - (rA+rB)`. So `rA + rB = |pA0 - pB0| - c.dist`. Let `RestLen = |pA0
  // - pB0|
  // - c.dist`. Current C(x) = |pA - pB| - RestLen. Actually, `c.dist` is
  // negative for overlap. Example: Dist = 0.9, Radii sum = 1.0. c.dist =
  // -0.1. RestLen = 1.0. We can recover RestLen from initial positions
  // (which we don't have here easily unless we store p_old or use
  // constraint rA/rB). Vector rA was "Center to Contact". |rA| ~
  // RadiusA. |rB| ~ RadiusB. So RestLen ~ |rA| + |rB|.

  float3 rA_vec = make_float3(c.rA.x, c.rA.y, c.rA.z);
  float3 rB_vec = make_float3(c.rB.x, c.rB.y, c.rB.z);
  // Special Ground Handling for idB < 0
  float C = 0.0f;
  float3 n = make_float3(0, 1, 0);

  if (idB < 0) {
    // Ground Logic
    // C(x) = (pA.y - radiusA) - GroundY.
    float ground_y = ps.domain_min.y;
    float radiusA_val =
        sqrtf(dot_product_p(rA_vec, rA_vec)); // Approx radius from rA stored
    // pA_w is current predicted position
    C = (pA.y - radiusA_val) - ground_y;
    n = make_float3(0, 1, 0);

    if (C >= 0.0f)
      return;
  } else {
    // Particle-Particle Logic
    float rest_len = sqrtf(dot_product_p(rA_vec, rA_vec)) +
                     sqrtf(dot_product_p(rB_vec, rB_vec));
    C = dist - rest_len;

    if (C >= 0.0f)
      return; // Separated

    if (dist > 1e-9f) {
      n = make_float3(-diff.x / dist, -diff.y / dist,
                      -diff.z / dist); // B->A
    } else {
      n = make_float3(c.normal.x, c.normal.y, c.normal.z);
    }
  }

  // Generalized Inverse Mass
  // w = invM + (r x n) I^-1 (r x n)
  // Update rA, rB based on new normal?
  // For sphere: rA = -n * radiusA.
  // rA = n * (-radiusA).
  // radiusA = |c.rA|.
  // Simple: r vectors are aligned with normal for spheres (no torque
  // from normal force really). But for stability with friction (if we
  // added it here), or if we had off-center... Let's use computed rA/rB
  // based on current Normal.

  float radiusA = sqrtf(dot_product_p(rA_vec, rA_vec));
  float radiusB = sqrtf(dot_product_p(rB_vec, rB_vec));

  float3 cur_rA = make_float3(n.x * -radiusA, n.y * -radiusA, n.z * -radiusA);
  float3 cur_rB = make_float3(n.x * radiusB, n.y * radiusB, n.z * radiusB);

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
    float3 n_neg = make_float3(-n.x, -n.y, -n.z);
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
