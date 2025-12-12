#include "ParticleSystem.cuh"
#include "memory_utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ------------------------------------------------------------------
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
      }
    }
  }
}

// ------------------------------------------------------------------
// 1. Predict and Clear
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// 1a. Predict Velocity (Gravity Only)
// ------------------------------------------------------------------
__global__ void predict_velocity_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  float4 vel_w = ps.d_vel[idx];
  float inv_mass = ps.d_pos[idx].w;

  // 1. Predict Velocity (Gravity)
  float3 v_curr = make_float3(vel_w.x, vel_w.y, vel_w.z);
  float3 v_pred = v_curr;

  if (inv_mass > 0.0f) {
    v_pred.x += ps.gravity.x * ps.dt;
    v_pred.y += ps.gravity.y * ps.dt;
    v_pred.z += ps.gravity.z * ps.dt;
  }
  // Store predicted velocity (Gravity Only)
  ps.d_vel_pred[idx] = make_float4(v_pred.x, v_pred.y, v_pred.z, vel_w.w);

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
  float4 q_curr = ps.d_quat[idx];
  float4 w_curr = ps.d_ang_vel[idx]; // Should include torque updates?
  // We didn't apply torque from velocity solve yet!
  float4 dw = ps.d_delta_ang_vel[idx];

  float3 omega = make_float3(w_curr.x + dw.x, w_curr.y + dw.y, w_curr.z + dw.z);
  // Update Angular Velocity state
  ps.d_ang_vel[idx] = make_float4(omega.x, omega.y, omega.z, w_curr.w);
  ps.d_ang_vel_pred[idx] = ps.d_ang_vel[idx];

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

  // Clear Deltas for Position Solve
  // (We cleared them in step 1, but let's be safe or just clear
  // d_constraint_counts) Actually, we need d_delta_pos cleared. (Done in step
  // 1)
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
  if (pos.z < min_b.z)
    pos.z += size.z;
  else if (pos.z >= max_b.z)
    pos.z -= size.z;

  ps.d_pos[idx] = make_float4(pos.x, pos.y, pos.z, p.w);

  // UNCONDITIONAL v update from phase A (already stored in d_vel in
  // reintegrate) We do NOT update d_vel here from x_pred! But we need to update
  // d_vel_pred = d_vel for next step consistency? d_vel is already updated in
  // apply_velocity_and_predict_position_kernel.

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
