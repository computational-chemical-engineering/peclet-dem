#include "ParticleSystem.cuh"
#include "memory_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ------------------------------------------------------------------
// Growth Mode Kernel
// ------------------------------------------------------------------
__global__ void update_growth_scales_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Scale = Target * Factor
  if (ps.growth_factor > 0.0f && ps.d_target_scales != nullptr) {
    ps.d_scale[idx] = ps.d_target_scales[idx] * ps.growth_factor;
  }
}

// Generate Ghosts for Periodic Boundaries (Legacy/Preserved)
// ------------------------------------------------------------------
__global__ void generate_ghosts_kernel(ParticleSystemData ps, float margin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_real)
    return;

  float4 p =
      ps.d_pos_pred[idx]; // Use Predicted position for Ghost Gen?
                          // Usually we ghost based on current state at start of
                          // frame. Let's use d_pos (ReadOnly State)
  p = ps.d_pos[idx];

  float3 pos = make_float3(p.x, p.y, p.z);
  float3 min_b = ps.domain_min;
  float3 max_b = ps.domain_max;
  float3 size = ps.domain_size;

  bool px = ps.periodic_x;
  bool py = ps.periodic_y;
  bool pz = ps.periodic_z;

  int shift_x[2] = {0, 0};
  int shift_y[2] = {0, 0};
  int shift_z[2] = {0, 0};
  int nx = 1, ny = 1, nz = 1;

  if (px) {
    if (pos.x < min_b.x + margin)
      shift_x[nx++] = 1; // Shift +L
    else if (pos.x > max_b.x - margin)
      shift_x[nx++] = -1; // Shift -L
  }
  if (py) {
    if (pos.y < min_b.y + margin)
      shift_y[ny++] = 1;
    else if (pos.y > max_b.y - margin)
      shift_y[ny++] = -1;
  }
  if (pz) {
    if (pos.z < min_b.z + margin)
      shift_z[nz++] = 1;
    else if (pos.z > max_b.z - margin)
      shift_z[nz++] = -1;
  }

  if (nx == 1 && ny == 1 && nz == 1)
    return;

  for (int ix = 0; ix < nx; ix++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int iz = 0; iz < nz; iz++) {
        if (shift_x[ix] == 0 && shift_y[iy] == 0 && shift_z[iz] == 0)
          continue;

        float3 offset = make_float3(shift_x[ix] * size.x, shift_y[iy] * size.y,
                                    shift_z[iz] * size.z);

        int ghost_idx = atomicAdd(ps.d_top_ghost, 1);
        if (ghost_idx >= ps.capacity) {
          atomicSub(ps.d_top_ghost, 1);
          continue;
        }

        // Copy State to Ghost
        float4 p_ghost = p;
        p_ghost.x += offset.x;
        p_ghost.y += offset.y;
        p_ghost.z += offset.z;

        // Initialize Ghost Predictors
        ps.d_pos[ghost_idx] = p_ghost;
        ps.d_pos_pred[ghost_idx] = p_ghost; // Initial guess
        ps.d_quat[ghost_idx] = ps.d_quat[idx];
        ps.d_quat_pred[ghost_idx] = ps.d_quat[idx];
        ps.d_vel[ghost_idx] = ps.d_vel[idx];
        ps.d_vel_pred[ghost_idx] = ps.d_vel[idx];
        ps.d_ang_vel[ghost_idx] = ps.d_ang_vel[idx];
        ps.d_ang_vel_pred[ghost_idx] = ps.d_ang_vel[idx];
        ps.d_scale[ghost_idx] = ps.d_scale[idx];
        ps.d_shape_ids[ghost_idx] = ps.d_shape_ids[idx];

        // Critical: Map Ghost Index back to Real Index for Momentum
        // Conservation
        ps.d_real_indices[ghost_idx] = idx;
      }
    }
  }
}

// ------------------------------------------------------------------
// 1. Predict and Clear
// ------------------------------------------------------------------

// --- Math Helpers ---
__device__ inline float3 cross_product(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ inline float3 rotate_vector(float3 v, float4 q) {
  // v' = v + 2*r x (r x v + w*v)
  float3 r = make_float3(q.x, q.y, q.z);
  float w = q.w;
  float3 cross1 = cross_product(r, v);
  // Wait, standard formula: q v q*
  // t = 2 * cross(q.xyz, v)
  // v' = v + q.w * t + cross(q.xyz, t)
  // Let's use the optimized one:
  float3 t = make_float3(2.0f * cross1.x, 2.0f * cross1.y, 2.0f * cross1.z);
  float3 cross2 = cross_product(r, t);
  return make_float3(v.x + w * t.x + cross2.x, v.y + w * t.y + cross2.y,
                     v.z + w * t.z + cross2.z);
}

__device__ inline float3 rotate_vector_inverse(float3 v, float4 q) {
  // Conjugate q: (-x, -y, -z, w)
  float4 q_inv = make_float4(-q.x, -q.y, -q.z, q.w);
  return rotate_vector(v, q_inv);
}

// ------------------------------------------------------------------
// 1a. Predict Velocity (Gravity + Gyroscopic Term)
// ------------------------------------------------------------------
__global__ void predict_velocity_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float4 vel_w = ps.d_vel[idx];
  float inv_mass = ps.d_pos[idx].w;

  // 1. Predict Linear Velocity (Gravity)
  float3 v_curr = make_float3(vel_w.x, vel_w.y, vel_w.z);
  float3 v_pred = v_curr;

  if (inv_mass > 0.0f) {
    v_pred.x += ps.gravity.x * ps.dt;
    v_pred.y += ps.gravity.y * ps.dt;
    v_pred.z += ps.gravity.z * ps.dt;
  }
  // Store predicted velocity (Gravity Only)
  ps.d_vel_pred[idx] = make_float4(v_pred.x, v_pred.y, v_pred.z, vel_w.w);

  // 1.5 Predict Angular Velocity (Euler Equations: Gyroscopic Precession)
  float4 ang_vel_w = ps.d_ang_vel[idx];
  float3 w_world = make_float3(ang_vel_w.x, ang_vel_w.y, ang_vel_w.z);
  float3 w_pred = w_world;

  // Only apply for dynamic bodies
  if (inv_mass > 0.0f) {
    // Load Inertia
    float4 inv_I_data = ps.d_inv_inertia[idx];
    float3 inv_I = make_float3(inv_I_data.x, inv_I_data.y, inv_I_data.z);

    // If object has finite inertia (not fixed rotation)
    if (inv_I.x > 0.0f || inv_I.y > 0.0f || inv_I.z > 0.0f) {
      float4 q = ps.d_quat[idx];

      // World -> Body
      float3 w_body = rotate_vector_inverse(w_world, q);

      // Compute derived I_body (Handle infinite/zero components safely)
      // Avoid division by zero if inv_I component is 0 (infinite I).
      // If inv_I is 0, alpha component will be 0 anyway.
      // But we need I for L = I * w.
      // If inv_I is 0, assume I is "very large" effectively, term dominates?
      // Actually if inv_I is 0, that axis is locked? Or Infinite?
      // In standard RB, we set inv_I=0 for locked axes.
      // We can approximate I = 1/inv_I. If inv_I < eps, I=?
      // If inv_I_x is 0, then alpha_x = 0 * tau_x = 0. So w_x is constant.
      // But we need L_x = I_x * w_x to compute tau_y/z.
      // If I_x is infinite, L_x is infinite. Tau is infinite.
      // This breaks 3D formulation.
      // Assumption: Fully dynamic bodies have non-zero inv_I on all axes.
      // If 1D/2D constraint, handle gracefully.

      float3 I_body = make_float3((inv_I.x > 1e-9f) ? 1.0f / inv_I.x : 0.0f,
                                  (inv_I.y > 1e-9f) ? 1.0f / inv_I.y : 0.0f,
                                  (inv_I.z > 1e-9f) ? 1.0f / inv_I.z : 0.0f);

      // Euler Equation: I dw/dt + w x (I w) = 0
      // dw/dt = - I_inv * (w x (I w))

      float3 L_body = make_float3(I_body.x * w_body.x, I_body.y * w_body.y,
                                  I_body.z * w_body.z);
      float3 w_cross_L = cross_product(w_body, L_body);

      float3 alpha_body =
          make_float3(-inv_I.x * w_cross_L.x, -inv_I.y * w_cross_L.y,
                      -inv_I.z * w_cross_L.z);

      // Semi-Implicit / Explicit Euler update
      w_body.x += alpha_body.x * ps.dt;
      w_body.y += alpha_body.y * ps.dt;
      w_body.z += alpha_body.z * ps.dt;

      // Body -> World
      w_pred = rotate_vector(w_body, q);
    }
  }
  // Store w_pred (Required for Solver!)
  ps.d_ang_vel_pred[idx] =
      make_float4(w_pred.x, w_pred.y, w_pred.z, ang_vel_w.w);

  // 2. Predict Position (Speculative for Collision)
  // XPBD uses x_pred = x_n + v_pred * dt for collision detection
  float3 x_curr =
      make_float3(ps.d_pos[idx].x, ps.d_pos[idx].y, ps.d_pos[idx].z);
  float3 x_pred = x_curr;

  if (inv_mass > 0.0f) {
    x_pred.x += v_pred.x * ps.dt;
    x_pred.y += v_pred.y * ps.dt;
    x_pred.z += v_pred.z * ps.dt;
  }
  ps.d_pos_pred[idx] =
      make_float4(x_pred.x, x_pred.y, x_pred.z, ps.d_pos[idx].w);

  // 3. Predict Quaternion (First Guess = Current)
  ps.d_quat_pred[idx] = ps.d_quat[idx];
  // Note: Angular prediction could go here, but q_n is usually sufficient for
  // Broadphase/Narrowphase or we can add simple omega integration if needed.

  // Clear Velocity Deltas (for Velocity Solve)
  ps.d_delta_vel[idx] = make_float4(0, 0, 0, 0);
  ps.d_delta_ang_vel[idx] = make_float4(0, 0, 0, 0);

  // Clear Position Deltas (for later Position Solve)
  ps.d_delta_pos[idx] = make_float4(0, 0, 0, 0);
  ps.d_delta_quat[idx] = make_float4(0, 0, 0, 0);
  ps.d_constraint_counts[idx] = 0;
}

// ------------------------------------------------------------------
// 1b. Apply Velocity Solve & Predict Position (Re-Integration)
// ------------------------------------------------------------------
__global__ void
apply_velocity_and_predict_position_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // 1. Apply accumulated velocity impulses (from Phase A: Velocity Solve)
  float4 v_pred_w = ps.d_vel_pred[idx]; // Currently holds v_n + g*dt
  float4 dv = ps.d_delta_vel[idx];

  // Summing up impulses (no averaging needed if atomicAdd was purely additive
  // impulse) Logic: v_solved = v_pred + delta_v
  float3 v_final =
      make_float3(v_pred_w.x + dv.x, v_pred_w.y + dv.y, v_pred_w.z + dv.z);

  // Update d_vel_pred with the "Solved" velocity
  ps.d_vel_pred[idx] = make_float4(v_final.x, v_final.y, v_final.z, v_pred_w.w);

  // PERSIST VELOCITY for next frame! (Key Change)
  // We want v_{n+1} = v_{solved}
  ps.d_vel[idx] = ps.d_vel_pred[idx];

  // 2. Predict Position
  float inv_mass = ps.d_pos[idx].w;
  float4 pos_curr = ps.d_pos[idx];
  float3 x_pred = make_float3(pos_curr.x, pos_curr.y, pos_curr.z);

  if (inv_mass > 0.0f) {
    x_pred.x += v_final.x * ps.dt;
    x_pred.y += v_final.y * ps.dt;
    x_pred.z += v_final.z * ps.dt;
  }
  ps.d_pos_pred[idx] = make_float4(x_pred.x, x_pred.y, x_pred.z, pos_curr.w);

  // 3. Predict Rotation (Simple Euler for now)
  // 3. Predict Rotation (Simple Euler for now)
  float4 q_curr = ps.d_quat[idx];

  // Use Solved Angular Velocity (accumulated in d_ang_vel_pred)
  float4 w_solved = ps.d_ang_vel_pred[idx];
  float3 omega = make_float3(w_solved.x, w_solved.y, w_solved.z);

  // Update State
  ps.d_ang_vel[idx] = make_float4(omega.x, omega.y, omega.z, w_solved.w);
  // ps.d_ang_vel_pred[idx] is already set.

  float4 q_pred = q_curr;
  if (inv_mass > 0.0f) {
    // Integrate q
    float dq_x = 0.5f * ps.dt *
                 (omega.x * q_curr.w + omega.y * q_curr.z - omega.z * q_curr.y);
    float dq_y = 0.5f * ps.dt *
                 (omega.y * q_curr.w + omega.z * q_curr.x - omega.x * q_curr.z);
    float dq_z = 0.5f * ps.dt *
                 (omega.z * q_curr.w + omega.x * q_curr.y - omega.y * q_curr.x);
    float dq_w =
        0.5f * ps.dt *
        (-omega.x * q_curr.x - omega.y * q_curr.y - omega.z * q_curr.z);
    q_pred.x += dq_x;
    q_pred.y += dq_y;
    q_pred.z += dq_z;
    q_pred.w += dq_w;

    // Normalize
    float len = sqrtf(q_pred.x * q_pred.x + q_pred.y * q_pred.y +
                      q_pred.z * q_pred.z + q_pred.w * q_pred.w);
    if (len > 1e-9f) {
      float inv = 1.0f / len;
      q_pred.x *= inv;
      q_pred.y *= inv;
      q_pred.z *= inv;
      q_pred.w *= inv;
    }
  }
  ps.d_quat_pred[idx] = q_pred;

  // Clear Position Deltas (for later Position Solve)
  // (We cleared them in step 1, but let's be safe or just clear
  // d_constraint_counts)
  // Actually, we must NOT clear d_delta_pos here if we want to run Phase B.
  // But Phase B starts with predict_position? No, apply_velocity... predicts.
}

// ------------------------------------------------------------------
// 1c. Apply Velocity Deltas (Iterative Solver Update)
// ------------------------------------------------------------------
__global__ void apply_velocity_deltas_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Apply Impulse
  float4 v_pred_w = ps.d_vel_pred[idx];
  float4 dv = ps.d_delta_vel[idx];

  // Summation (Standard Impulse Solver)
  ps.d_vel_pred[idx] = make_float4(v_pred_w.x + dv.x, v_pred_w.y + dv.y,
                                   v_pred_w.z + dv.z, v_pred_w.w);

  // Angular
  // No, Solver reads d_ang_vel_pred.
  // We should update d_ang_vel_pred.
  float4 w_pred = ps.d_ang_vel_pred[idx];
  float4 dw = ps.d_delta_ang_vel[idx];

  ps.d_ang_vel_pred[idx] =
      make_float4(w_pred.x + dw.x, w_pred.y + dw.y, w_pred.z + dw.z, w_pred.w);

  // Clear Buffers for next iteration
  ps.d_delta_vel[idx] = make_float4(0, 0, 0, 0);
  ps.d_delta_ang_vel[idx] = make_float4(0, 0, 0, 0);
  // ERROR FIX: Do NOT clear constraint counts here! They are needed for all
  // iterations. ps.d_constraint_counts[idx] = 0;
}

// ------------------------------------------------------------------
// 1d. Compute Contact Counts (Pre-Pass for Min-Scaling)
// ------------------------------------------------------------------
__global__ void compute_contact_counts_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_contacts = *ps.d_contact_count;

  if (idx >= num_contacts)
    return;

  // if (idx == 0) printf("ComputeCounts: num_contacts=%d\n", num_contacts);

  ContactConstraint c = ps.d_contacts[idx];

  if (c.dist > 0.0f)
    return;

  int idA = c.bodyA;
  int idB = c.bodyB;

  atomicAdd(ps.d_constraint_counts + idA, 1);
  if (idB >= 0) {
    atomicAdd(ps.d_constraint_counts + idB, 1);
  }
}

// ------------------------------------------------------------------
// 2. Apply Updates (Jacobi Average)
// ------------------------------------------------------------------
__global__ void apply_updates_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  int count = ps.d_constraint_counts[idx];
  if (count > 0) {
    float factor = 1.0f / (float)count;

    // Position
    float4 dp = ps.d_delta_pos[idx];
    float4 p = ps.d_pos_pred[idx];
    p.x += dp.x * factor;
    p.y += dp.y * factor;
    p.z += dp.z * factor;
    ps.d_pos_pred[idx] = p;

    // Velocity
    float4 dv = ps.d_delta_vel[idx];
    float4 v = ps.d_vel_pred[idx];
    v.x += dv.x * factor;
    v.y += dv.y * factor;
    v.z += dv.z * factor;
    ps.d_vel_pred[idx] = v;

    // Quaternion - normalize?
    // For now simplify: Rotation updates might be complex.
    // If we used d_delta_quat logic...
    // Let's assume No Rotation Correction for Phase 1 Sphere Packing (No
    // friction torque yet)

    // Clear for next iteration if re-used?
    // Solvers usually run once per substep, or iterating?
    // Projected Jacobi typically iterates.
    // If iterating, we need to clear deltas here.
    ps.d_delta_pos[idx] = make_float4(0, 0, 0, 0);
    ps.d_delta_vel[idx] = make_float4(0, 0, 0, 0);
    ps.d_constraint_counts[idx] = 0;
  }
}

// ------------------------------------------------------------------
// 3. Final Commit
// ------------------------------------------------------------------
__global__ void final_commit_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Periodic Wrap!
  float4 p = ps.d_pos_pred[idx];
  float3 pos = make_float3(p.x, p.y, p.z);
  float3 size = ps.domain_size;
  float3 min_b = ps.domain_min;
  float3 max_b = ps.domain_max;

  // Periodic Wrap!
  if (ps.periodic_x) {
    if (pos.x < min_b.x)
      pos.x += size.x;
    else if (pos.x >= max_b.x)
      pos.x -= size.x;
  }
  if (ps.periodic_y) {
    if (pos.y < min_b.y)
      pos.y += size.y;
    else if (pos.y >= max_b.y)
      pos.y -= size.y;
  }
  if (ps.periodic_z) {
    if (pos.z < min_b.z)
      pos.z += size.z;
    else if (pos.z >= max_b.z)
      pos.z -= size.z;
  }

  ps.d_pos[idx] = make_float4(pos.x, pos.y, pos.z, p.w);

  // UNCONDITIONAL v update from phase A (already stored in d_vel in
  // reintegrate) We do NOT update d_vel here from x_pred! But we need to
  // update d_vel_pred = d_vel for next step consistency? d_vel is already
  // updated in apply_velocity_and_predict_position_kernel.

  ps.d_quat[idx] = ps.d_quat_pred[idx];
  // Ang vel also updated in reintegrate
  // ps.d_ang_vel[idx] = ps.d_ang_vel_pred[idx]; // Done previously
}

// ------------------------------------------------------------------
// Wrappers
// ------------------------------------------------------------------

void launch_generate_ghosts(ParticleSystemData ps, float margin) {
  int threads = 256;
  int blocks = (ps.num_real + threads - 1) / threads;
  generate_ghosts_kernel<<<blocks, threads>>>(ps, margin);
  CUDA_CHECK(cudaGetLastError());
}

void launch_integrate_predict_velocity(ParticleSystemData ps, float dt,
                                       float3 gravity) {
  ps.dt = dt;
  ps.gravity = make_float4(gravity.x, gravity.y, gravity.z, 0);

  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;

  CUDA_CHECK(cudaMemset(ps.d_contact_count, 0, sizeof(int)));
  // Clear delta_vel explicitly? No, kernel does it.

  predict_velocity_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_velocity_and_predict_position(ParticleSystemData ps,
                                                float dt) {
  ps.dt = dt;
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;
  apply_velocity_and_predict_position_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

void launch_update_growth_scales(ParticleSystemData &ps) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;
  update_growth_scales_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

void launch_apply_velocity_deltas(ParticleSystemData ps) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;
  apply_velocity_deltas_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

void launch_compute_contact_counts(ParticleSystemData ps) {
  int num_contacts;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  // printf("HOST: num_contacts = %d\n", num_contacts);

  if (num_contacts == 0)
    return;

  int threads = 256;
  int blocks = (num_contacts + threads - 1) / threads;
  compute_contact_counts_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
}

void launch_apply_updates(ParticleSystemData ps) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;
  apply_updates_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

void launch_final_commit(ParticleSystemData ps, float dt) {
  ps.dt = dt;
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;
  final_commit_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}

void launch_integration(ParticleSystemData &ps) {
  int threads = 256;
  int blocks = (ps.num_particles + threads - 1) / threads;
  apply_velocity_and_predict_position_kernel<<<blocks, threads>>>(ps);
  CUDA_CHECK(cudaGetLastError());
}
