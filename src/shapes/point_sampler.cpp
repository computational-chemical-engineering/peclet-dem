#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "point_sampler.h"
#include <vector_types.h>

// Generates points on the surface of a hollow cylinder
// Used for the "Point Shell" in narrowphase
std::vector<float4> generate_cylinder_points(CylinderParams params,
                                             float spacing) {
  std::vector<float4> points;

  float r_outer = params.radius;
  float r_inner = params.radius - params.thickness;
  float h = params.height;

  // 1. Outer Surface
  float circumference = 2.0f * M_PI * r_outer;
  int n_angular = ceil(circumference / spacing);
  int n_vertical = ceil(h / spacing);

  for (int i = 0; i < n_angular; ++i) {
    float theta = 2.0f * M_PI * i / n_angular;
    for (int j = 0; j <= n_vertical; ++j) {
      float y = -0.5f * h + h * j / n_vertical;
      float x = r_outer * cos(theta);
      float z = r_outer * sin(theta);
      points.push_back(
          {x, y, z, 0.0f}); // w=0 (can be signed distance or dummy)
    }
  }

  // 2. Inner Surface (if thick)
  if (params.thickness > 0.0f) {
    circumference = 2.0f * M_PI * r_inner;
    n_angular = ceil(circumference / spacing);
    for (int i = 0; i < n_angular; ++i) {
      float theta = 2.0f * M_PI * i / n_angular;
      for (int j = 0; j <= n_vertical; ++j) {
        float y = -0.5f * h + h * j / n_vertical;
        float x = r_inner * cos(theta);
        float z = r_inner * sin(theta);
        points.push_back({x, y, z, 0.0f});
      }
    }
  }

  // 3. Caps (Annulus)
  // Top and Bottom
  float dr = spacing;
  int n_radial = ceil(params.thickness / dr);
  for (int i = 0; i <= n_radial; ++i) {
    float r = r_inner + (r_outer - r_inner) * i / n_radial;
    circumference = 2.0f * M_PI * r;
    n_angular = ceil(circumference / spacing);
    for (int j = 0; j < n_angular; ++j) {
      float theta = 2.0f * M_PI * j / n_angular;
      float x = r * cos(theta);
      float z = r * sin(theta);

      // Top
      points.push_back({x, 0.5f * h, z, 0.0f});
      // Bottom
      points.push_back({x, -0.5f * h, z, 0.0f});
    }
  }

  return points;
}

// Surface points of an axis-aligned box: each of the 6 faces sampled on a grid
// whose endpoints land on the box edges, so the 8 corners are present (the
// corners carry the load when the box rests/topples on a plane). Edge/corner
// points are duplicated across adjacent faces, which is harmless.
std::vector<float4> generate_box_points(BoxParams p, float spacing) {
  std::vector<float4> points;
  auto nseg = [&](float half) {
    int n = (int)std::ceil(2.0f * half / spacing);
    return n < 1 ? 1 : n;
  };
  int nx = nseg(p.hx), ny = nseg(p.hy), nz = nseg(p.hz);
  auto lin = [](float half, int i, int n) {
    return -half + 2.0f * half * (float)i / (float)n;
  };

  // +/- x faces (vary y,z)
  for (int j = 0; j <= ny; ++j)
    for (int k = 0; k <= nz; ++k) {
      float y = lin(p.hy, j, ny), z = lin(p.hz, k, nz);
      points.push_back({p.hx, y, z, 0.0f});
      points.push_back({-p.hx, y, z, 0.0f});
    }
  // +/- y faces (vary x,z)
  for (int i = 0; i <= nx; ++i)
    for (int k = 0; k <= nz; ++k) {
      float x = lin(p.hx, i, nx), z = lin(p.hz, k, nz);
      points.push_back({x, p.hy, z, 0.0f});
      points.push_back({x, -p.hy, z, 0.0f});
    }
  // +/- z faces (vary x,y)
  for (int i = 0; i <= nx; ++i)
    for (int j = 0; j <= ny; ++j) {
      float x = lin(p.hx, i, nx), y = lin(p.hy, j, ny);
      points.push_back({x, y, p.hz, 0.0f});
      points.push_back({x, y, -p.hz, 0.0f});
    }
  return points;
}
