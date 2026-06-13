#pragma once
#include <vector>
#include <vector_types.h>

struct CylinderParams {
  float radius;
  float height;
  float thickness;
};

std::vector<float4> generate_cylinder_points(CylinderParams params,
                                             float spacing);

struct BoxParams {
  float hx; // half-extents
  float hy;
  float hz;
};

// Surface point shell of an axis-aligned box (all 6 faces sampled on a grid that
// includes the edges/corners), used as the collision source against an SDF/plane.
std::vector<float4> generate_box_points(BoxParams params, float spacing);
