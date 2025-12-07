#pragma once
#include <cmath>
#include <cuda_runtime.h>

// SDF for a Hollow Cylinder
// Aligned along Y-axis by default
// p: Query position (in local frame of particle)
// radius: Outer radius
// height: Total height
// thickness: Wall thickness
__device__ inline float sdf_hollow_cylinder(float3 p, float radius,
                                            float height, float thickness) {
  // 1. SDF of infinite cylinder + planes = Finite Cylinder (Outer)
  float2 d_xz =
      make_float2(hypotf(p.x, p.z) - radius, abs(p.y) - height * 0.5f);

  // External distance (max(d, 0)) + Internal (min(max(d), 0))
  // hypotf works for Euclidean length of 2 args
  float dist_outer = min(max(d_xz.x, d_xz.y), 0.0f) +
                     hypotf(max(d_xz.x, 0.0f), max(d_xz.y, 0.0f));

  // Distance to the "Ring" in 2D cross section (XZ plane)
  float r_xz = hypotf(p.x, p.z);
  float d_annulus = abs(r_xz - (radius - thickness * 0.5f)) - thickness * 0.5f;

  float2 d_y_annulus = make_float2(d_annulus, abs(p.y) - height * 0.5f);

  float dist_final = min(max(d_y_annulus.x, d_y_annulus.y), 0.0f) +
                     hypotf(max(d_y_annulus.x, 0.0f), max(d_y_annulus.y, 0.0f));

  return dist_final;
}
