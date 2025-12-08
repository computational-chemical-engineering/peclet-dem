#include "ParticleSystem.cuh"
#include "memory_utils.cuh"
#include "shapes/sdf_cylinder.cuh"
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

__global__ void solve_ground_kernel(ParticleSystemData ps, float dt,
                                    float global_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_particles)
    return;

  // Constraint: y > ground
  // Simple plane at y=ground_level?
  // Let's assume domain_min.y is ground.

  // Only Real Particles? (Ghosts shouldn't collide with ground? Or yes?)
  // Typically periodic BC implies no ground unless it's a box.
  // But if we want packing with gravity, we act on floor.
  // Ghosts are for side boundaries. Top/Bottom might be open or wall.
  // For packing we assume floor at y=0 or min.y.

  if (idx < ps.num_real) {
    float ground_y = ps.domain_min.y;
    float4 p = ps.d_pos_star[idx];
    float s = ps.d_scale[idx] * global_scale;
    float r_bound =
        1.0f * s; // Radius for ground check. Careful: Cylinder might be taller.
                  // 1.5 * s was previous safe bound.
                  // For sphere 0.5 * s.
                  // Safe approximate: 1.0*s covers sphere and cylinder mostly.

    // Explicit shape check would be better but expensive for simple ground.
    // Let's stick to simple sphere approx for ground for now or bounding
    // sphere. For cylinder (r=0.5, h=2.0) -> max extent from center is
    // sqrt(0.5^2 + 1.0^2) ~ 1.12
    r_bound = 1.2f * s;

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

    if (count > 0) {
      // Jacobi-style Averaging for stability
      float factor = 1.0f / (float)count;
      // Optional: Over-relaxation factor (e.g. 1.2 ??? No, stick to 1.0 for
      // stability first)

      float3 pos = make_float3(ps.d_pos_star[idx].x, ps.d_pos_star[idx].y,
                               ps.d_pos_star[idx].z);

      ps.d_pos_star[idx] =
          make_float4(pos.x + delta.x * factor, pos.y + delta.y * factor,
                      pos.z + delta.z * factor, ps.d_pos_star[idx].w);

      // Reset for next substep
      ps.d_delta_pos[idx] = make_float4(0, 0, 0, 0);
      ps.d_constraint_counts[idx] = 0;
    }
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

  // Analytic Sphere-Sphere
  if (shape_i.type == 0 && shape_j.type == 0) {
    // Analytic Sphere-Sphere (Symmetric)
    // We removed 'if (i > j) return;' to ensure we capture the collision
    // regardless of order. GS will handle double-solving via immediate updates
    // (second solve sees C~0).

    float3 rel_pos =
        make_float3(pos_i.x - pos_j.x, pos_i.y - pos_j.y, pos_i.z - pos_j.z);
    float dist_sq = dot(rel_pos, rel_pos);
    float ri = shape_i.params.x * si;
    float rj = shape_j.params.x * sj;
    float sum_radii = ri + rj;
    float thickness = 0.02f * sum_radii * 0.5f; // Margin

    // Early exit if separated (check squared to avoid sqrt)
    if (dist_sq > (sum_radii + thickness) * (sum_radii + thickness))
      return;

    float dist = sqrtf(dist_sq);
    float C = dist - sum_radii - thickness;

    if (C < 0) {
      float3 world_grad = make_float3(0, 1, 0); // Default up
      if (dist > 1e-9f) {
        world_grad = rel_pos * (1.0f / dist); // Direction from J to I
      }

      float w_sum = wi + wj;
      float lambda = -C / (w_sum + 1e-6f); // Avoid div/0 if infinite mass

      if ((i == 0 && j == 1) || (i == 1 && j == 0)) {
        printf("DEBUG: Physics i=%d j=%d dist=%f C=%f lambda=%f wi=%f wj=%f\n",
               i, j, dist, C, lambda, wi, wj);
      }

      float3 delta = make_float3(lambda * world_grad.x, lambda * world_grad.y,
                                 lambda * world_grad.z);

      // Apply to I
      atomicAdd(&ps.d_delta_pos[i].x, delta.x * wi);
      atomicAdd(&ps.d_delta_pos[i].y, delta.y * wi);
      atomicAdd(&ps.d_delta_pos[i].z, delta.z * wi);
      atomicAdd(&ps.d_constraint_counts[i], 1);

      // Apply to J (Symmetric)
      // Note: check_collisions_one_way is called "one_way" but for Spheres we
      // do two-way to support the GS kernel which locks both.
      atomicAdd(&ps.d_delta_pos[j].x, -delta.x * wj);
      atomicAdd(&ps.d_delta_pos[j].y, -delta.y * wj);
      atomicAdd(&ps.d_delta_pos[j].z, -delta.z * wj);
      atomicAdd(&ps.d_constraint_counts[j], 1);
    }
    return;
    return;
  }

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

      float alpha =
          1e-4f; // Small compliance for stability (Hard contacts approx)
      // Ideally alpha = 1 / (k * dt2)

      float w_sum = wi + wj;
      // XPBD: lambda = (-C - alpha * lambda_prev) / (w_sum + alpha)
      // Assuming lambda_prev = 0 for one-iteration or stateless solver

      float lambda = -C / (w_sum + alpha);

      if (i < 10 || j < 10) {
        printf(
            "DEBUG: Solve i=%d j=%d C=%f wi=%f wj=%f lambda=%f si=%f sj=%f\n",
            i, j, C, wi, wj, lambda, si, sj);
      }

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
                                      int num_contacts, float global_scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_contacts)
    return;

  int2 pair = ps.d_potential_collisions[idx];
  int i = pair.x;
  int j = pair.y;

  // XPBD / PBD
  float4 pi = ps.d_pos_star[i];
  float4 qi = ps.d_quat_star[i];
  float4 pj = ps.d_pos_star[j];
  float4 qj = ps.d_quat_star[j];

  if (idx < 5) {
    printf("DEBUG: Pos i=(%f %f %f %f) j=(%f %f %f %f)\n", pi.x, pi.y, pi.z,
           pi.w, pj.x, pj.y, pj.z, pj.w);
  }

  float si = ps.d_scale[i] * global_scale;
  float sj = ps.d_scale[j] * global_scale;

  // Retrieve inverse masses
  float wi = ps.d_pos[i].w;
  float wj = ps.d_pos[j].w;

  // We need shape types.
  // Currently we only have 1 global shape list?
  // Or distinct shapes per particle?
  // ps.d_shapes is array of ShapeData.
  // But we don't have per-particle shape ID yet.
  // We initialized ps.d_shapes[0] as the ONE shape.
  // Let's assume all particles are Shape 0 for now (Mono-shape simulation).
  // Or we need d_shape_ids array.
  // For Phase 2/3 we implicitly assumed Shape 0.

  // Retrieve shapes (Default 0 for now)
  ShapeData shape_i = ps.d_shapes[0];
  ShapeData shape_j = ps.d_shapes[0];

  check_collisions_one_way(ps, i, j, pi, qi, pj, qj, shape_i, shape_j, wi, wj,
                           si, sj);
  // Note: check_collisions adds reaction to i and j.

  // Apply immediate update
  float4 delta_i = ps.d_delta_pos[i];
  float4 delta_j = ps.d_delta_pos[j];

  ps.d_pos_star[i] =
      make_float4(pi.x + delta_i.x, pi.y + delta_i.y, pi.z + delta_i.z, pi.w);
  ps.d_pos_star[j] =
      make_float4(pj.x + delta_j.x, pj.y + delta_j.y, pj.z + delta_j.z, pj.w);

  // Clear deltas for next
  ps.d_delta_pos[i] = make_float4(0, 0, 0, 0); // Optional cleanup
  ps.d_delta_pos[j] = make_float4(0, 0, 0, 0);

  // Unlock
  atomicExch(&ps.d_locks[j], 0);
  atomicExch(&ps.d_locks[i], 0);
}

__global__ void solve_contacts_gs_kernel(ParticleSystemData ps, float dt,
                                         int num_contacts, float global_scale,
                                         int offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_contacts)
    return;

  // Randomized Coloring: Shift index to vary lock priority
  // int idx = (tid + offset) % num_contacts;
  int idx = tid; // Debug: No offset

  int2 pair = ps.d_potential_collisions[idx];
  int i = pair.x;
  int j = pair.y;

  // Lock Ordering to prevent Deadlock (AB vs BA)
  int first = (i < j) ? i : j;
  int second = (i < j) ? j : i;

  // Try Lock First
  if (atomicCAS(&ps.d_locks[first], 0, 1) != 0)
    return;

  // Try Lock Second
  if (atomicCAS(&ps.d_locks[second], 0, 1) != 0) {
    atomicExch(&ps.d_locks[first], 0); // Release first
    return;
  }

  // Locked! Clear accumulators
  ps.d_delta_pos[i] = make_float4(0, 0, 0, 0);
  ps.d_delta_pos[j] = make_float4(0, 0, 0, 0);

  // Load Data
  float4 pi = ps.d_pos_star[i];
  float4 pj = ps.d_pos_star[j];

  // Shapes
  int shape_id_i = ps.d_shape_ids[i];
  int shape_id_j = ps.d_shape_ids[j];
  ShapeData shape_i = ps.d_shapes[shape_id_i];
  ShapeData shape_j = ps.d_shapes[shape_id_j];

  // Mass
  float wi = ps.d_pos[i].w;
  float wj = ps.d_pos[j].w;

  // Scale
  float si = ps.d_scale[i] * global_scale;
  float sj = ps.d_scale[j] * global_scale;

  // Radii
  // Assuming Sphere (Params.x = Radius)
  float ri = shape_i.params.x * si;
  float rj = shape_j.params.x * sj;
  float sum_radii = ri + rj;

  // Dist
  float3 p_i = make_float3(pi.x, pi.y, pi.z);
  float3 p_j = make_float3(pj.x, pj.y, pj.z);
  float3 rel = make_float3(p_i.x - p_j.x, p_i.y - p_j.y, p_i.z - p_j.z);
  float dist_sq = rel.x * rel.x + rel.y * rel.y + rel.z * rel.z;
  float dist = sqrtf(dist_sq);

  if (dist < sum_radii && dist > 1e-6f) {
    float C = dist - sum_radii;
    float alpha = 1e-4f; // Hardcoded compliance
    float w_sum = wi + wj;
    float lambda = -C / (w_sum + alpha);

    float3 grad = make_float3(rel.x / dist, rel.y / dist, rel.z / dist);
    float3 delta =
        make_float3(lambda * grad.x, lambda * grad.y, lambda * grad.z);

    // Apply to accumulators
    ps.d_delta_pos[i].x += delta.x * wi;
    ps.d_delta_pos[i].y += delta.y * wi;
    ps.d_delta_pos[i].z += delta.z * wi;

    ps.d_delta_pos[j].x -= delta.x * wj;
    ps.d_delta_pos[j].y -= delta.y * wj;
    ps.d_delta_pos[j].z -= delta.z * wj;
    // Update d_pos_star
    float4 delta_i = ps.d_delta_pos[i];
    float4 delta_j = ps.d_delta_pos[j];

    ps.d_pos_star[i] =
        make_float4(pi.x + delta_i.x, pi.y + delta_i.y, pi.z + delta_i.z, pi.w);
    ps.d_pos_star[j] =
        make_float4(pj.x + delta_j.x, pj.y + delta_j.y, pj.z + delta_j.z, pj.w);

    if (idx < 5) {
      printf("DEBUG: Delta i=(%f %f %f)\n", delta_i.x, delta_i.y, delta_i.z);
      printf("DEBUG: PosStar i=(%f %f %f)\n", ps.d_pos_star[i].x,
             ps.d_pos_star[i].y, ps.d_pos_star[i].z);
    }
  }

  // Unlock
  atomicExch(&ps.d_locks[j], 0);
  atomicExch(&ps.d_locks[i], 0);
}

void launch_solver(ParticleSystemData &ps, float dt, float global_scale,
                   int offset) {
  int threads = 256;
  int blocks_part = (ps.num_particles + threads - 1) / threads;

  // Ground Plane
  solve_ground_kernel<<<blocks_part, threads>>>(ps, dt, global_scale);

  int num_contacts = 0;
  CUDA_CHECK(cudaMemcpy(&num_contacts, ps.d_potential_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  if (num_contacts > 0 && num_contacts < 100) {
    if (num_contacts > 0 && num_contacts < 100)
      printf("DEBUG: Solver small case num_contacts = %d\n", num_contacts);
  }

  // Apply Deltas? No, GS applies immediately.
  // But Ground Plane? Ground plane uses atomicAdd to d_delta_pos in
  // solve_ground_kernel. If we want mixed mode, we should apply ground deltas
  // SEPARATELY. Currently solve_ground_kernel adds to d_delta_pos. We should
  // call apply_deltas_kernel ONLY for ground if we use GS for contacts? Or make
  // ground kernel GS too? Let's keep ground as Jacobi (apply_deltas) and
  // Contacts as GS. Ground runs first. atomicAdds to delta. Then
  // apply_deltas_kernel runs? Wait, if I run apply_deltas_kernel AFTER GS, it
  // will apply ground deltas. BUT GS kernel wipes d_delta_pos[i] when it locks
  // i! Danger: If Ground adds delta, and GS kernel runs, it overwrites delta
  // with 0! Fix: Apply Ground Deltas BEFORE Contacts.

  // New Order:
  // 1. Ground Kernel -> d_delta
  // 2. Apply Deltas (Ground) -> Clear d_delta
  // 3. GS Contacts -> Update Pos Direct

  apply_deltas_kernel<<<blocks_part, threads>>>(ps); // Apply Ground results
  CUDA_CHECK(cudaDeviceSynchronize());

  if (num_contacts > 0) {
    int blocks_pair = (num_contacts + threads - 1) / threads;
    // GS Solver: Immediate updates.
    // We might need multiple passes per substep to cover skipped locks?
    // Or just rely on the outer 'solver_substeps' loop (e.g. 20 substeps).
    // With Random Order (implied by atomic race), it converges well.

    solve_contacts_gs_kernel<<<blocks_pair, threads>>>(ps, dt, num_contacts,
                                                       global_scale, offset);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaDeviceSynchronize();
}
