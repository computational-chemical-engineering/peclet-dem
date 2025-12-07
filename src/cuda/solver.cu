#include "ParticleSystem.cuh"
#include "shapes/sdf_cylinder.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <vector_types.h>

// Helper math
__device__ inline float length(float3 v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__device__ inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 rotate_vector(float3 v, float4 q) {
  float ix = q.w * v.x + q.y * v.z - q.z * v.y;
  float iy = q.w * v.y + q.z * v.x - q.x * v.z;
  float iz = q.w * v.z + q.x * v.y - q.y * v.x;
  float iw = -q.x * v.x - q.y * v.y - q.z * v.z;

  float3 res;
  res.x = ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y;
  res.y = iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z;
  res.z = iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x;
  return res;
}

__device__ inline float3 inv_rotate_vector(float3 v, float4 q) {
  float4 inv_q = make_float4(-q.x, -q.y, -q.z, q.w);
  return rotate_vector(v, inv_q);
}

__global__ void solve_constraints_kernel(ParticleSystemData ps, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  if (idx < ps.num_real) {
    float ground_y = ps.domain_min.y;
    float4 p = ps.d_pos_star[idx];
    float s = ps.d_scale[idx];
    float r_bound = 1.5f * s;

    if (p.y - r_bound < ground_y) {
      float C = p.y - r_bound - ground_y;
      if (C < 0) {
        float3 grad = make_float3(0, 1, 0);
        float lambda = -C;
        atomicAdd(&ps.d_delta_pos[idx].x, lambda * grad.x);
        atomicAdd(&ps.d_delta_pos[idx].y, lambda * grad.y);
        atomicAdd(&ps.d_delta_pos[idx].z, lambda * grad.z);
        atomicAdd(&ps.d_constraint_counts[idx], 1);
      }
    }
  }
}

__global__ void apply_deltas_kernel(ParticleSystemData ps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Apply to both Real and Ghosts? Usually only Real matter for next step.
  // But Ghosts might need to be resolved for visualization?
  // Let's apply to all for now.

  float4 delta = ps.d_delta_pos[idx];
  int count = ps.d_constraint_counts[idx];

  if (count > 0) {
    // Averaging? Or just adding?
    // Simple XPBD adds. But with atomic accumulation from multiple constraints,
    // simple addition might be unstable if parameters aren't tuned for it
    // (stiffness). For rigid contact, dividing by number of constraints
    // (averaging) is often cleaner for parallel Jacobi-style. Let's try
    // averaging: delta / count (if count > 1?) Actually standard PBD: delta_p =
    // sum(delta_p_i). But we used stiffness=1 (lambda = -C). Let's just Add.

    // But wait on averaging:
    // If we use parallel Jacobi, we sum all corrections and apply.
    // Often checking count is good to normalize if many constraints fight.
    // Let's do simple addition first.

    float3 pos = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                             ps.d_pos_star[idx].z);
    ps.d_pos_star[idx] = make_float4(pos.x + delta.x, pos.y + delta.y,
                                     pos.z + delta.z, ps.d_pos_star[idx].w);

    // Reset for next substep
    ps.d_delta_pos[idx] = make_float4(0, 0, 0, 0);
    ps.d_constraint_counts[idx] = 0;
  }
}

// Check points of A against SDF of B
__device__ void check_collisions_one_way(ParticleSystemData &ps, int i, int j,
                                         float4 pi, float4 qi, float4 pj,
                                         float4 qj, ShapeData &shape_i,
                                         ShapeData &shape_j, float wi, float wj,
                                         float si, float sj) {
  float3 pos_i = make_float3(pi.x, pi.y, pi.z);
  float3 pos_j = make_float3(pj.x, pj.y, pj.z);

  // Transform from A local to B local:
  // P_b = R_b^T * ( (R_a * P_a + T_a) - T_b )

  // Pre-calculate relative transform could be faster, but per-point is fine.

  for (int k = 0; k < shape_i.num_points; ++k) {
    float4 local_pt_4 = shape_i.d_fine_points[k];
    // Scale Point: local * si
    float3 local_pt =
        make_float3(local_pt_4.x, local_pt_4.y, local_pt_4.z) * si;

    // To World
    float3 world_pt = rotate_vector(local_pt, qi);
    world_pt.x += pos_i.x;
    world_pt.y += pos_i.y;
    world_pt.z += pos_i.z;

    // To B Local
    float3 rel_pos = make_float3(world_pt.x - pos_j.x, world_pt.y - pos_j.y,
                                 world_pt.z - pos_j.z);
    float3 pt_in_b = inv_rotate_vector(rel_pos, qj);

    float dist = 1e9f;
    float3 local_grad = make_float3(0, 1, 0);

    // Eval SDF based on Shape Type of B
    if (shape_j.type == 0) {
      // SPHERE
      // params.x = radius
      float radius = shape_j.params.x * sj;
      float r = length(pt_in_b);
      dist = r - radius;
      // Gradient is just direction from center
      if (r > 1e-9f) {
        local_grad = make_float3(pt_in_b.x / r, pt_in_b.y / r, pt_in_b.z / r);
      }
    } else if (shape_j.type == 1) {
      // HOLLOW CYLINDER
      float base_radius = shape_j.params.x;
      float base_height =
          shape_j.params.y; // stored as just height? or half height?
      // Let's assume stored as passed: radius, height, thickness
      // But sdf_hollow_cylinder usually expects half-height?
      // Let's check sim.cpp: params are 0.5, 2.0.
      // We'll pass them directly scaled by sj.

      // Recalculate dist and grad using Finite Diff for Cylinder
      float br = base_radius * sj;
      float bh = base_height * sj;
      float bt = shape_j.params.z * sj;

      dist = sdf_hollow_cylinder(pt_in_b, br, bh, bt);

      // Gradient via FD
      float eps = 1e-3f;
      float d_dx = sdf_hollow_cylinder(
          make_float3(pt_in_b.x + eps, pt_in_b.y, pt_in_b.z), br, bh, bt);
      float d_dy = sdf_hollow_cylinder(
          make_float3(pt_in_b.x, pt_in_b.y + eps, pt_in_b.z), br, bh, bt);
      float d_dz = sdf_hollow_cylinder(
          make_float3(pt_in_b.x, pt_in_b.y, pt_in_b.z + eps), br, bh, bt);

      local_grad = make_float3(d_dx - dist, d_dy - dist, d_dz - dist);
      float len = length(local_grad);
      if (len > 1e-9f) {
        local_grad = local_grad * (1.0f / len);
      }
    }

    float thickness = 0.02f * ((si + sj) * 0.5f); // Scale contact margin too?
    if (dist < thickness) {
      // Collision!

      // Rotate gradient to world
      float3 world_grad =
          rotate_vector(local_grad, qj); // Normal of B surface in world

      // Constraint C = dist
      // lambda = -C / (w_i + w_j) ... simplified (ignoring angular for now)
      // Angular needed for torque!
      // For Phase 2 scaffold, linear push only is acceptable for "Basic
      // non-spherical". Adding angular is standard XPBD but more code. Let's
      // stick to linear push first to verify contact.

      float C = dist - thickness; // Penetration
      if (C >= 0)
        continue;

      float w_sum = wi + wj; // + rotational parts
      float lambda = -C / w_sum;

      float3 delta = make_float3(lambda * world_grad.x, lambda * world_grad.y,
                                 lambda * world_grad.z);

      // Apply
      atomicAdd(&ps.d_delta_pos[i].x, delta.x * wi);
      atomicAdd(&ps.d_delta_pos[i].y, delta.y * wi);
      atomicAdd(&ps.d_delta_pos[i].z, delta.z * wi);
      atomicAdd(&ps.d_constraint_counts[i], 1);

      atomicAdd(&ps.d_delta_pos[j].x, -delta.x * wj);
      atomicAdd(&ps.d_delta_pos[j].y, -delta.y * wj);
      atomicAdd(&ps.d_delta_pos[j].z, -delta.z * wj);
      atomicAdd(&ps.d_constraint_counts[j], 1);
    }
  }
}

__global__ void solve_contacts_kernel(ParticleSystemData ps, float dt,
                                      int num_contacts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_contacts)
    return;

  int2 pair = ps.d_potential_collisions[idx];
  int i = pair.x;
  int j = pair.y;

  // Early out if self or invalid
  if (i == j)
    return;

  float4 pi = ps.d_pos_star[i];
  float4 pj = ps.d_pos_star[j];
  float4 qi = ps.d_quat_star[i];
  float4 qj = ps.d_quat_star[j];
  float si = ps.d_scale[i];
  float sj = ps.d_scale[j];

  // Check weights (Ghost handling)
  float wi = pi.w;
  float wj = pj.w;
  bool i_is_ghost = (i >= ps.num_real);
  bool j_is_ghost = (j >= ps.num_real);
  if (i_is_ghost)
    wi = 0.0f;
  if (j_is_ghost)
    wj = 0.0f;
  if (wi + wj == 0.0f)
    return;

  ShapeData shape_i = ps.d_shapes[ps.d_shape_ids[i]];
  ShapeData shape_j = ps.d_shapes[ps.d_shape_ids[j]];

  // Bidirectional Check
  // A points vs B SDF
  check_collisions_one_way(ps, i, j, pi, qi, pj, qj, shape_i, shape_j, wi, wj,
                           si, sj);
  // B points vs A SDF
  check_collisions_one_way(ps, j, i, pj, qj, pi, qi, shape_j, shape_i, wj, wi,
                           sj, si);
}

void launch_solver(ParticleSystemData &ps, float dt) {
  int threads = 64; // Fewer threads per block due to loop register pressure
  int blocks_part = (ps.num_particles + threads - 1) / threads;
  solve_constraints_kernel<<<blocks_part, threads>>>(ps, dt);

  int num_contacts = 0;
  cudaMemcpy(&num_contacts, ps.d_potential_count, sizeof(int),
             cudaMemcpyDeviceToHost);

  if (num_contacts > 0) {
    int blocks_pair = (num_contacts + threads - 1) / threads;
    solve_contacts_kernel<<<blocks_pair, threads>>>(ps, dt, num_contacts);
  }

  // Apply Deltas
  apply_deltas_kernel<<<blocks_part, threads>>>(ps);

  cudaDeviceSynchronize();
}
