/// @file
/// @brief dem — portable POD types + math + analytic SDFs shared by the Kokkos kernel ports.
///
/// Replaces the CUDA-only float3/float4 vector types and the __host__/__device__ helpers in
/// math_utils.cuh / shapes/sdf_analytic.cuh with backend-agnostic equivalents
/// (KOKKOS_INLINE_FUNCTION, usable on host and any Kokkos backend). The math is a faithful copy of
/// the CUDA versions so ports reproduce the existing behaviour exactly.
#ifndef DEM_PORTABLE_HPP
#define DEM_PORTABLE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

// Kokkos_Core.hpp above defines KOKKOS_VERSION, so core's PECLET_HD resolves to
// KOKKOS_INLINE_FUNCTION and the leaf formulas are device-callable on every backend.
#include "peclet/core/geom/primitives.hpp"

namespace peclet::dem {

// Portable mirrors of CUDA float3/float4 (POD, trivially copyable).
struct F3 {
  float x, y, z;
};
struct F4 {
  float x, y, z, w;
};

// --- vector ops ---
KOKKOS_INLINE_FUNCTION F3 add3(F3 a, F3 b) {
  return F3{a.x + b.x, a.y + b.y, a.z + b.z};
}
KOKKOS_INLINE_FUNCTION F3 sub3(F3 a, F3 b) {
  return F3{a.x - b.x, a.y - b.y, a.z - b.z};
}
KOKKOS_INLINE_FUNCTION F3 scale3(F3 a, float s) {
  return F3{a.x * s, a.y * s, a.z * s};
}
KOKKOS_INLINE_FUNCTION float dot3(F3 a, F3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
KOKKOS_INLINE_FUNCTION float len3(F3 v) {
  return Kokkos::sqrt(dot3(v, v));
}
KOKKOS_INLINE_FUNCTION F3 cross3v(F3 a, F3 b) {
  return F3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
KOKKOS_INLINE_FUNCTION F4 cross3(F4 a, F4 b) {
  return F4{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f};
}

// Load an F3 / F4 from a multidimensional Kokkos View row (templated => no per-header duplication).
template <class V>
KOKKOS_INLINE_FUNCTION F3 ldF3(const V& v, int i) {
  return F3{v(i, 0), v(i, 1), v(i, 2)};
}
template <class V>
KOKKOS_INLINE_FUNCTION F4 ldF4(const V& v, int i) {
  return F4{v(i, 0), v(i, 1), v(i, 2), v(i, 3)};
}

// --- quaternion rotate (q = {x,y,z,w}) — copy of math_utils.cuh rotate_vector/inv_rotate_vector
// ---
KOKKOS_INLINE_FUNCTION F3 rotateVector(F4 q, F3 v) {
  const F3 qv{q.x, q.y, q.z};
  const F3 t = scale3(cross3v(qv, v), 2.0f);
  return add3(add3(v, scale3(t, q.w)), cross3v(qv, t));
}
KOKKOS_INLINE_FUNCTION F3 invRotateVector(F4 q, F3 v) {
  return rotateVector(F4{-q.x, -q.y, -q.z, q.w}, v);
}
KOKKOS_INLINE_FUNCTION F4 quatInverse(F4 q) {
  return F4{-q.x, -q.y, -q.z, q.w};
}
KOKKOS_INLINE_FUNCTION F4 quatMult(F4 a, F4 b) {
  return F4{
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

// --- analytic SDFs (canonical/unit space) ---
//
// RELOCATED (suite/docs/ANALYTIC_SDF_GEOMETRY.md, Layer 0 rung 3): the formula bodies now live in
// peclet::core::geom::prim (core/include/peclet/core/geom/primitives.hpp), which transcribed them
// from THIS file operation-for-operation. The wrappers below keep dem's exact signatures and
// parameter packing, so every call site is untouched and the results are bit-identical -- verified
// by a raw-bit capture over 76000 evaluations on host-openmp and CUDA, before vs after.
//
// The enum values are also core's ShapeKind values (core's kGrid/kSphere/kHollowCylinder/kBox
// = 0/1/2/3 were chosen to match), so a shape id means the same thing on both sides.
enum ShapeKind { SHAPE_GRID_SDF = 0, SPHERE = 1, HOLLOW_CYLINDER = 2, BOX = 3 };

/// F3 -> core's Real-templated point type. Layout-identical; this is a type change, not a copy.
KOKKOS_INLINE_FUNCTION peclet::core::Vec3<float> toCoreVec(F3 p) {
  return peclet::core::Vec3<float>{p.x, p.y, p.z};
}

KOKKOS_INLINE_FUNCTION float sdfSphere(F3 p, F4 params) {
  return peclet::core::geom::prim::Sphere<float>{params.x}.eval(toCoreVec(p));
}

// params = (rOuter, height, thickness); tube about the y axis. core's HollowCylinder is the
// distance-exact form -- NOT core's HollowCylinderShell, which is the max-of-halfspaces variant
// that core/voro use and which is a different function (see the design note's TRAP section).
KOKKOS_INLINE_FUNCTION float sdfHollowCylinder(F3 p, F4 params) {
  return peclet::core::geom::prim::HollowCylinder<float>{params.x, params.y, params.z}.eval(
      toCoreVec(p));
}

KOKKOS_INLINE_FUNCTION float sdfBox(F3 p, F4 params) {
  return peclet::core::geom::prim::Box<float>{params.x, params.y, params.z}.eval(toCoreVec(p));
}

KOKKOS_INLINE_FUNCTION float sdfEval(F3 p, int type, F4 params) {
  if (type == SPHERE)
    return sdfSphere(p, params);
  if (type == HOLLOW_CYLINDER)
    return sdfHollowCylinder(p, params);
  if (type == BOX)
    return sdfBox(p, params);
  return 1e9f;  // grid SDF (texture) not yet ported
}

}  // namespace peclet::dem

#endif  // DEM_PORTABLE_HPP
