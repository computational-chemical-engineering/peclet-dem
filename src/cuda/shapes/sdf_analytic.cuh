#pragma once
#include "../math_utils.cuh"
#include <cuda_runtime.h>

namespace dem {

// -----------------------------------------------------------------------------
// Hollow Cylinder (Raschig Ring) SDF
// -----------------------------------------------------------------------------
// params.x = outer_radius (R)
// params.y = height (h)
// params.z = thickness (t)
__host__ __device__ inline float sdf_hollow_cylinder(float3 p, float4 params) {
  float r_outer = params.x;
  float h = params.y;
  float thick = params.z;

  // 1. Map to 2D Radial Plane
  // r = distance from central axis (Y-axis assumption)
  float r = sqrtf(p.x * p.x + p.z * p.z);
  float y = p.y;

  // 2. Center the "Wall Profile"
  // The wall is a rectangle of width 'thick' centered at 'r_mid'
  float r_mid = r_outer - (thick * 0.5f);

  // 3. 2D Box SDF Logic
  // Vector d = distance from the center of the wall segment
  // We treat the wall cross-section as a box of size (thick, h)
  float2 d;
  d.x = fabsf(r - r_mid) - (thick * 0.5f);
  d.y = fabsf(y) - (h * 0.5f);

  // 4. Combine Exterior and Interior distances
  // length(max(d, 0)) -> Distance to outside of box
  // min(max(d.x, d.y), 0) -> Distance to inside of box (negative)
  float outside_dist = length(make_float2(fmaxf(d.x, 0.0f), fmaxf(d.y, 0.0f)));
  float inside_dist = fminf(fmaxf(d.x, d.y), 0.0f);

  return outside_dist + inside_dist;
}

// -----------------------------------------------------------------------------
// Analytic Sphere SDF (Example/Reference)
// -----------------------------------------------------------------------------
// params.x = radius
__host__ __device__ inline float sdf_sphere(float3 p, float4 params) {
  return length(p) - params.x;
}

// -----------------------------------------------------------------------------
// Analytic Box / Cuboid SDF (exact)
// -----------------------------------------------------------------------------
// params.x,y,z = half-extents (hx, hy, hz); a cube has hx==hy==hz.
__host__ __device__ inline float sdf_box(float3 p, float4 params) {
  float dx = fabsf(p.x) - params.x;
  float dy = fabsf(p.y) - params.y;
  float dz = fabsf(p.z) - params.z;
  float ox = fmaxf(dx, 0.0f), oy = fmaxf(dy, 0.0f), oz = fmaxf(dz, 0.0f);
  float outside = sqrtf(ox * ox + oy * oy + oz * oz);
  float inside = fminf(fmaxf(dx, fmaxf(dy, dz)), 0.0f);
  return outside + inside;
}

} // namespace dem
