/// @file
/// @brief Input validation of the `peclet.dem` Python binding (src/dem_bindings.cpp only).
///
/// The C++ API (sim.hpp) is internal and trusts its caller; the public setters refuse a NaN/Inf or
/// a physically inadmissible value at the binding, as ValueError (std::invalid_argument), before it
/// reaches the solver. A valid value takes exactly the path it took before. The message names the
/// setter, the parameter, the admissible range with its dimension (dem works in the caller's
/// consistent units) and the offending value. Binding-only: no solver header includes this.
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "narrowphase.hpp"  // kMaxMaterials

namespace peclet::dem::binding_checks {

/// A triple: any sequence of three floats (tuple, list, (3,) array).
using F3Arg = std::array<float, 3>;

inline std::string num(double v) {
  char b[32];
  std::snprintf(b, sizeof b, "%g", v);
  return b;
}
inline void require(bool ok, const char* who, const std::string& name, const std::string& rule,
                    double v) {
  if (!ok)
    throw std::invalid_argument(std::string(who) + ": " + name + " must be " + rule + ", got " +
                                num(v));
}
inline void finite(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v), who, name, std::string("finite (") + dim + ")", v);
}
inline void positive(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v) && v > 0.0, who, name, std::string("finite and > 0 (") + dim + ")", v);
}
inline void nonneg(const char* who, const std::string& name, double v, const char* dim) {
  require(std::isfinite(v) && v >= 0.0, who, name, std::string("finite and >= 0 (") + dim + ")", v);
}
/// A restitution coefficient: [0, 1], dimensionless.
inline void restitution_coef(const char* who, const char* name, double v) {
  require(std::isfinite(v) && v >= 0.0 && v <= 1.0, who, name,
          "finite and in [0, 1] (dimensionless)", v);
}
/// An isotropic solid's Poisson ratio: (-1, 0.5], dimensionless.
inline void poisson_ratio(const char* who, const char* name, double v) {
  require(std::isfinite(v) && v > -1.0 && v <= 0.5, who, name,
          "finite and in (-1, 0.5] (dimensionless)", v);
}
/// Per-component forms for a triple (or a quaternion): the parameter is named `name[i]`.
inline std::string component(const char* name, std::size_t i) {
  return std::string(name) + "[" + std::to_string(i) + "]";
}
template <class T, std::size_t N>
inline void finite_each(const char* who, const char* name, const std::array<T, N>& a,
                        const char* dim) {
  for (std::size_t i = 0; i < N; ++i)
    finite(who, component(name, i), a[i], dim);
}
template <class T, std::size_t N>
inline void positive_each(const char* who, const char* name, const std::array<T, N>& a,
                          const char* dim) {
  for (std::size_t i = 0; i < N; ++i)
    positive(who, component(name, i), a[i], dim);
}
template <class T, std::size_t N>
inline void nonneg_each(const char* who, const char* name, const std::array<T, N>& a,
                        const char* dim) {
  for (std::size_t i = 0; i < N; ++i)
    nonneg(who, component(name, i), a[i], dim);
}
/// A count (capacity, iterations, samples per axis): an integer >= `lo`.
inline void count(const char* who, const std::string& name, long v, long lo) {
  require(v >= lo, who, name, "an integer >= " + std::to_string(lo), static_cast<double>(v));
}
/// A reconstruction grid's samples per axis (get_sdf_grid / export_sdf): each >= 1. A negative
/// count used to abort the interpreter.
inline void sample_counts(const char* who, int rx, int ry, int rz) {
  count(who, "resolution[0]", rx, 1);
  count(who, "resolution[1]", ry, 1);
  count(who, "resolution[2]", rz, 1);
}
/// The shared arguments of set_sdf_shape / add_sdf_shape.
inline void sdf_shape_args(const char* who, const F3Arg& origin, const F3Arg& spacing,
                           const F3Arg& inv_inertia, float bounding_radius) {
  finite_each(who, "origin", origin, "length");
  positive_each(who, "spacing", spacing, "length");
  nonneg_each(who, "inv_inertia", inv_inertia, "1/length^2");
  positive(who, "bounding_radius", bounding_radius, "length");
}
/// set_domain(min, max): finite corners, max > min on every axis.
inline void domain_corners(const F3Arg& mn, const F3Arg& mx) {
  finite_each("set_domain", "min", mn, "length");
  finite_each("set_domain", "max", mx, "length");
  for (std::size_t i = 0; i < 3; ++i)
    require(mx[i] > mn[i], "set_domain", component("max", i),
            "> min[" + std::to_string(i) + "] = " + num(mn[i]) + " (length)", mx[i]);
}
/// add_plane: a finite point and a finite, non-zero normal.
inline void plane(const F3Arg& p, const F3Arg& n) {
  finite_each("add_plane", "point", p, "length");
  finite_each("add_plane", "normal", n, "dimensionless");
  require(n[0] != 0.0f || n[1] != 0.0f || n[2] != 0.0f, "add_plane", "normal", "a non-zero vector",
          0.0);
}
/// init_mpi's global box: finite origin, positive extent, at least one ORB cell per axis.
inline void mpi_box(const std::tuple<double, double, double>& origin,
                    const std::tuple<double, double, double>& extent,
                    const std::tuple<long, long, long>& cells) {
  const auto [ox, oy, oz] = origin;
  const auto [ex, ey, ez] = extent;
  const auto [cx, cy, cz] = cells;
  finite_each("init_mpi", "origin", std::array<double, 3>{ox, oy, oz}, "length");
  positive_each("init_mpi", "extent", std::array<double, 3>{ex, ey, ez}, "length");
  const std::array<long, 3> c{cx, cy, cz};
  for (std::size_t i = 0; i < 3; ++i)
    count("init_mpi", component("cells", i), c[i], 1);
}
/// Material ids index the 8-entry pair tables (narrowphase.hpp kMaxMaterials).
inline constexpr int kMaxMaterialId = peclet::dem::kMaxMaterials - 1;
inline void material_ids(const std::vector<int>& ids) {
  for (std::size_t i = 0; i < ids.size(); ++i)
    require(ids[i] >= 0 && ids[i] <= kMaxMaterialId, "set_material_ids", component("ids", i),
            "an integer in [0, " + std::to_string(kMaxMaterialId) + "]", ids[i]);
}

}  // namespace peclet::dem::binding_checks
