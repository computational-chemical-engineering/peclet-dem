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
