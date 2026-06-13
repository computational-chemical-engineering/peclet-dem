#include "ParticleSystem.cuh"
#include "memory_utils.cuh" // For CUDA_CHECK
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "math_utils.cuh"
#include "shapes/sdf_analytic.cuh"

// Helper to evaluate SDF for any shape type
__device__ inline float sdf_eval(float3 p, ShapeDescriptor desc) {
  if (desc.type == SHAPE_ANALYTIC_SPHERE) {
    return dem::sdf_sphere(p, desc.params);
  } else if (desc.type == SHAPE_ANALYTIC_HOLLOW_CYLINDER) {
    return dem::sdf_hollow_cylinder(p, desc.params);
  } else if (desc.type == SHAPE_ANALYTIC_BOX) {
    return dem::sdf_box(p, desc.params);
  } else if (desc.type == SHAPE_GRID_SDF) {
    // Map to UVW
    // float3 uvw = (p - desc.aabb_min) / (desc.aabb_max - desc.aabb_min);
    // return tex3D<float>(desc.sdf_texture, uvw.x, uvw.y, uvw.z);
    return 1e9f; // Placeholder until Textures are set up
  }
  return 1e9f;
}

__global__ void detect_contacts_kernel(ParticleSystemData ps,
                                       float global_scale,
                                       float safety_margin) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_potential = *ps.d_potential_count;
  if (idx >= num_potential)
    return;

  int2 pair = ps.d_potential_collisions[idx];
  int idA = pair.x;
  int idB = pair.y;

  // 1. Load Shape Info
  int shape_id_A = ps.d_shape_ids[idA];
  int shape_id_B = ps.d_shape_ids[idB];
  ShapeDescriptor desc_A = ps.d_shapes[shape_id_A];
  ShapeDescriptor desc_B = ps.d_shapes[shape_id_B];

  // 2. Load World Transforms (Predicted)
  float4 pA_w = ps.d_pos_pred[idA];
  float4 qA_w = ps.d_quat_pred[idA];
  float4 pB_w = ps.d_pos_pred[idB];
  float4 qB_w = ps.d_quat_pred[idB];

  float3 posA = make_float3(pA_w.x, pA_w.y, pA_w.z);
  float3 posB = make_float3(pB_w.x, pB_w.y, pB_w.z);
  float scaleA = ps.d_scale[idA];
  // float scaleB = ps.d_scale[idB]; // Assume B params are pre-scaled or
  // handled in SDF

  // 3. Loop over Point Shell of Body A
  // If Body A is a simple Sphere (legacy), it might not have points set up yet.
  // Fallback: If num_points == 0, use center point?
  // Or assume Sphere Generator puts 1 point at (0,0,0) if it is a sphere.
  // For now, let's assume we iterate points.

  int countA = desc_A.num_points;
  float4 *shellA = desc_A.d_fine_points;

  // Optimization: If A is a sphere and has no points, treat as single point at
  // 0,0,0 with radius But SDF check logic implies point probe. Sphere-Sphere
  // logic used radius. General SDF logic: Distance(p) < margin. If A is a
  // sphere, we check Distance(center) < margin + radius? Yes. That is
  // equivalent to dist(surface) < margin. So if countA == 0 (Legacy Sphere), we
  // use 1 point (0,0,0) but margin increases by Radius A.

  // Actually, let's Stick to the Point Shell plan.
  // Even for Spheres, we should report "Surface Points" if we want robust
  // collision vs SDF. BUT sphere-sphere is exact. For Sphere vs Cylinder,
  // Sphere Center at distance 'd' means surface at 'd-r'. So Effective Margin =
  // safety_margin + radius_A.
  bool is_sphere_A = (desc_A.type == SHAPE_ANALYTIC_SPHERE);
  int iter_count = (countA > 0) ? countA : 1;

  for (int k = 0; k < iter_count; ++k) {
    float3 p_local_A = make_float3(0, 0, 0);
    float point_radius = 0.0f;

    if (countA > 0) {
      float4 p4 = shellA[k];
      p_local_A = make_float3(p4.x, p4.y, p4.z);
      // If points have encoded radius? p4.w?
    } else if (is_sphere_A) {
      // Use Center, effective margin += Radius
      // Radius = params.x * scale * global_scale
      float r_semantic = desc_A.params.x;
      point_radius = r_semantic * scaleA * global_scale;
    }

    // Transform A-Local -> World
    float3 p_world = posA + rotate_vector(qA_w, p_local_A * scaleA);

    // Transform World -> B-Local
    float3 p_rel = p_world - posB;
    float3 p_local_B = inv_rotate_vector(qB_w, p_rel);

    // Evaluate SDF with Handling for Scale B
    // Canonical Space: p_canonical = p_local / scaleB
    // dist_world = dist_canonical * scaleB
    float scaleB = ps.d_scale[idB];
    float3 p_canonical_B = p_local_B / scaleB;

    // Dist in Canonical Space (e.g. against Unit Sphere)
    float dist_canonical = sdf_eval(p_canonical_B, desc_B);

    // Scale back to World
    float dist = dist_canonical * scaleB;

    // Check Collision
    // Overlap if dist < (margin + point_radius)
    // i.e. dist - point_radius < margin
    float effective_dist = dist - point_radius;

    if (effective_dist < safety_margin) {
      // TRACK MAX OVERLAP
      if (effective_dist < 0) {
        atomicMaxFloat(ps.d_max_overlap, -effective_dist);
      }

      // Collision!
      int contact_idx = atomicAdd(ps.d_contact_count, 1);
      if (contact_idx >= ps.max_contacts) {
        atomicSub(ps.d_contact_count, 1);
        continue; // Overflow
      }

      // Compute Gradient (Normal) in B-Local via Central Diff
      // We compute gradient of Canonical SDF, which gives Normal in Local space
      // directly (Direction doesn't change with uniform scale)
      float eps = 1e-4f;
      float3 n_local;
      n_local.x = sdf_eval(make_float3(p_canonical_B.x + eps, p_canonical_B.y,
                                       p_canonical_B.z),
                           desc_B) -
                  sdf_eval(make_float3(p_canonical_B.x - eps, p_canonical_B.y,
                                       p_canonical_B.z),
                           desc_B);
      n_local.y = sdf_eval(make_float3(p_canonical_B.x, p_canonical_B.y + eps,
                                       p_canonical_B.z),
                           desc_B) -
                  sdf_eval(make_float3(p_canonical_B.x, p_canonical_B.y - eps,
                                       p_canonical_B.z),
                           desc_B);
      n_local.z = sdf_eval(make_float3(p_canonical_B.x, p_canonical_B.y,
                                       p_canonical_B.z + eps),
                           desc_B) -
                  sdf_eval(make_float3(p_canonical_B.x, p_canonical_B.y,
                                       p_canonical_B.z - eps),
                           desc_B);

      float len = length(n_local);
      if (len > 1e-9f) {
        n_local = n_local / len;
      } else {
        n_local = make_float3(0, 1, 0); // Fallback
      }

      // Transform Normal to World: B-Local -> World
      float3 n_world = rotate_vector(qB_w, n_local);

      // Revert to Surface Points
      // Logic: Sphere uses point_radius to find surface. SDF returns surface
      // point directly.
      float3 p_surf_A = p_world - n_world * point_radius;

      float3 rA_vec = p_surf_A - posA;
      // p_surf_B is (p_surf_A - n * dist)
      float3 rB_vec = (p_surf_A - n_world * effective_dist) - posB;

      ContactConstraint c;
      c.bodyA = idA;
      c.bodyB = idB;
      c.normal = make_float4(n_world.x, n_world.y, n_world.z, 0.0f);
      c.rA = make_float4(rA_vec.x, rA_vec.y, rA_vec.z, 0.0f);
      c.rB = make_float4(rB_vec.x, rB_vec.y, rB_vec.z, 0.0f);
      c.dist = effective_dist;

      c.friction_lambda_n = 0.0f;
      ps.d_contacts[contact_idx] = c;
    }
  }
}

__global__ void detect_boundary_kernel(ParticleSystemData ps,
                                       float global_scale,
                                       float safety_margin) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ps.num_real)
    return;

  float4 p_w = ps.d_pos_pred[idx];
  float s = ps.d_scale[idx] * global_scale;

  // Load Shape to get Radius
  int shape_id = ps.d_shape_ids[idx];
  ShapeDescriptor desc = ps.d_shapes[shape_id];

  float base_radius = desc.params.x;
  // Fallback/Safety: If params.x is 0 (uninitialized?), default to 1.0?
  // But initialize() sets it. If it's SDF grid (type 0), params might be
  // bounding box? For now, consistent with simulation.cpp packing, .x is
  // radius.
  if (base_radius == 0.0f)
    base_radius = 1.0f; // Minimal safety

  float radius = base_radius * s;

  // Helper lambda: emit a plane contact for a surface point whose lever arm
  // from the body centre is rA (world frame), at signed distance `dist` from the
  // plane. rA = -normal*radius for an analytic sphere; rA = R(q)*(p_shell*s) for
  // a point-shell shape (box/cylinder), so each corner/face point is its own
  // contact and the position solver/friction see the true contact geometry.
  auto add_plane_contact = [&](float3 normal, float dist, float3 plane_point,
                               float3 rA) {
    if (dist < safety_margin) {
      if (dist < 0) {
        atomicMaxFloat(ps.d_max_overlap, -dist);
      }
      int contact_idx = atomicAdd(ps.d_contact_count, 1);
      if (contact_idx >= ps.max_contacts) {
        atomicSub(ps.d_contact_count, 1);
        return;
      }

      ContactConstraint c;
      c.bodyA = idx;
      c.bodyB = -1; // Wall/Plane
      c.normal = make_float4(normal.x, normal.y, normal.z, 0.0f);
      c.rA = make_float4(rA.x, rA.y, rA.z, 0.0f);
      // Store World Space Contact Point on Plane in rB (Anchor)
      c.rB = make_float4(plane_point.x, plane_point.y, plane_point.z, 0.0f);
      c.dist = dist;
      c.friction_lambda_n = 0.0f;
      ps.d_contacts[contact_idx] = c;
    }
  };

  int num_points = desc.num_points;
  float4 *shell = desc.d_fine_points;
  float3 posA = make_float3(p_w.x, p_w.y, p_w.z);

  // Iterate over Explicit Planes
  for (int i = 0; i < ps.num_planes; ++i) {
    Plane p = ps.d_planes[i];

    if (num_points > 0) {
      // Point-shell shape (box, cylinder): test each surface point against the
      // plane in world space; the lever arm rotates with the body.
      float4 qA = ps.d_quat_pred[idx];
      for (int k = 0; k < num_points; ++k) {
        float4 pl = shell[k];
        float3 rA = rotate_vector(
            qA, make_float3(pl.x * s, pl.y * s, pl.z * s));
        float3 pwk = make_float3(posA.x + rA.x, posA.y + rA.y, posA.z + rA.z);
        float3 diff = make_float3(pwk.x - p.point.x, pwk.y - p.point.y,
                                  pwk.z - p.point.z);
        float dist = diff.x * p.normal.x + diff.y * p.normal.y +
                     diff.z * p.normal.z;
        add_plane_contact(p.normal, dist, p.point, rA);
      }
    } else {
      // Analytic sphere: centre distance minus radius.
      float3 diff =
          make_float3(p_w.x - p.point.x, p_w.y - p.point.y, p_w.z - p.point.z);
      float signed_dist_center =
          diff.x * p.normal.x + diff.y * p.normal.y + diff.z * p.normal.z;
      float dist = signed_dist_center - radius;
      float3 rA = make_float3(-p.normal.x * radius, -p.normal.y * radius,
                              -p.normal.z * radius);
      add_plane_contact(p.normal, dist, p.point, rA);
    }
  }

  // NOTE: Automatic Box Walls are REMOVED per user request.
  // Periodicity only affects Ghost Generation in Broadphase.
}

void launch_narrowphase(ParticleSystemData ps, float global_scale) {
  // Check potential collisions count
  int num_potential;
  CUDA_CHECK(cudaMemcpy(&num_potential, ps.d_potential_count, sizeof(int),
                        cudaMemcpyDeviceToHost));

  int threads = 256;
  int blocks;

  float safety_margin = 0.1f * global_scale;

  if (num_potential > 0) {
    blocks = (num_potential + threads - 1) / threads;
    detect_contacts_kernel<<<blocks, threads>>>(ps, global_scale,
                                                safety_margin);
    CUDA_CHECK(cudaGetLastError());
  }

  // Explicit Plane Checks
  // Always run if planes exist
  if (ps.num_planes > 0 && ps.num_real > 0) {
    blocks = (ps.num_real + threads - 1) / threads;
    detect_boundary_kernel<<<blocks, threads>>>(ps, global_scale,
                                                safety_margin);
    CUDA_CHECK(cudaGetLastError());
  }
}
