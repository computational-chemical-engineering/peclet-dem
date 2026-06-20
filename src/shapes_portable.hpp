/// @file
/// @brief dem — portable (host) surface-shell point generators for the analytic shapes.
///
/// Faithful copy of shapes/point_sampler.cpp (generate_cylinder_points / generate_box_points), but
/// emitting dem::F3 instead of CUDA float4 so the Kokkos module can build the point shell without the
/// CUDA vector_types.h dependency. The point math (spacing, ceil counts, cos/sin sampling, cap annulus,
/// box faces incl. edges/corners) is byte-for-byte the same so the generated shells match the CUDA path.
#ifndef DEM_SHAPES_PORTABLE_HPP
#define DEM_SHAPES_PORTABLE_HPP

#include <cmath>
#include <vector>

#include "dem_portable.hpp"  // dem::F3

namespace dem {

// Surface point shell of a hollow cylinder (outer wall + inner wall if thick + top/bottom annulus
// caps). Faithful copy of generate_cylinder_points.
inline std::vector<F3> genCylinderShell(float radius, float height, float thickness, float spacing) {
  std::vector<F3> points;

  float r_outer = radius;
  float r_inner = radius - thickness;
  float h = height;

  // 1. Outer surface
  float circumference = 2.0f * M_PI * r_outer;
  int n_angular = std::ceil(circumference / spacing);
  int n_vertical = std::ceil(h / spacing);

  for (int i = 0; i < n_angular; ++i) {
    float theta = 2.0f * M_PI * i / n_angular;
    for (int j = 0; j <= n_vertical; ++j) {
      float y = -0.5f * h + h * j / n_vertical;
      float x = r_outer * std::cos(theta);
      float z = r_outer * std::sin(theta);
      points.push_back(F3{x, y, z});
    }
  }

  // 2. Inner surface (if thick)
  if (thickness > 0.0f) {
    circumference = 2.0f * M_PI * r_inner;
    n_angular = std::ceil(circumference / spacing);
    for (int i = 0; i < n_angular; ++i) {
      float theta = 2.0f * M_PI * i / n_angular;
      for (int j = 0; j <= n_vertical; ++j) {
        float y = -0.5f * h + h * j / n_vertical;
        float x = r_inner * std::cos(theta);
        float z = r_inner * std::sin(theta);
        points.push_back(F3{x, y, z});
      }
    }
  }

  // 3. Caps (annulus) — top and bottom
  float dr = spacing;
  int n_radial = std::ceil(thickness / dr);
  for (int i = 0; i <= n_radial; ++i) {
    float r = r_inner + (r_outer - r_inner) * i / n_radial;
    circumference = 2.0f * M_PI * r;
    n_angular = std::ceil(circumference / spacing);
    for (int j = 0; j < n_angular; ++j) {
      float theta = 2.0f * M_PI * j / n_angular;
      float x = r * std::cos(theta);
      float z = r * std::sin(theta);
      points.push_back(F3{x, 0.5f * h, z});   // top
      points.push_back(F3{x, -0.5f * h, z});  // bottom
    }
  }

  return points;
}

// Surface points of an axis-aligned box: each of the 6 faces sampled on a grid whose endpoints land on
// the box edges (so the 8 corners are present). Faithful copy of generate_box_points.
inline std::vector<F3> genBoxShell(float hx, float hy, float hz, float spacing) {
  std::vector<F3> points;
  auto nseg = [&](float half) {
    int n = (int)std::ceil(2.0f * half / spacing);
    return n < 1 ? 1 : n;
  };
  int nx = nseg(hx), ny = nseg(hy), nz = nseg(hz);
  auto lin = [](float half, int i, int n) { return -half + 2.0f * half * (float)i / (float)n; };

  // +/- x faces (vary y,z)
  for (int j = 0; j <= ny; ++j)
    for (int k = 0; k <= nz; ++k) {
      float y = lin(hy, j, ny), z = lin(hz, k, nz);
      points.push_back(F3{hx, y, z});
      points.push_back(F3{-hx, y, z});
    }
  // +/- y faces (vary x,z)
  for (int i = 0; i <= nx; ++i)
    for (int k = 0; k <= nz; ++k) {
      float x = lin(hx, i, nx), z = lin(hz, k, nz);
      points.push_back(F3{x, hy, z});
      points.push_back(F3{x, -hy, z});
    }
  // +/- z faces (vary x,y)
  for (int i = 0; i <= nx; ++i)
    for (int j = 0; j <= ny; ++j) {
      float x = lin(hx, i, nx), y = lin(hy, j, ny);
      points.push_back(F3{x, y, hz});
      points.push_back(F3{x, y, -hz});
    }
  return points;
}

}  // namespace dem

#endif  // DEM_SHAPES_PORTABLE_HPP
