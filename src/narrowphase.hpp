/// @file
/// @brief dem — portable (Kokkos) narrow-phase: SDF point-shell collision + boundary planes.
///
/// Kokkos port of detect_contacts_kernel / detect_boundary_kernel (narrowphase.cu) over the
/// particle SoA expressed as Kokkos Views (positions/quaternions as View<float*[3]>/[4],
/// scales/shape-ids as View<float*>/<int*> — backend-default layout, so coalesced on GPU and
/// cache-friendly on CPU). The per-point math is a faithful copy of the CUDA kernels (same
/// scale/global_scale handling, same central-difference normal, same contact geometry) so results
/// match. Analytic shapes only for now; grid-SDF (texture) returns +inf (peclet::dem::sdfEval), as
/// in the CUDA placeholder.
#ifndef DEM_NARROWPHASE_HPP
#define DEM_NARROWPHASE_HPP

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ContactC, CpExec/CpMem
#include "dem_portable.hpp"

namespace peclet::dem {

/// Portable mirror of ShapeDescriptor (analytic fields + a flat-array point shell).
///
/// For an imported grid SDF (type == SHAPE_GRID_SDF) the analytic `params` is unused except
/// params.x, which carries the canonical bounding radius (used by the broad-phase splat / VTI
/// export). The signed-distance samples live in the shared Particles::sdfGrid View at
/// [gridOffset, gridOffset + nx*ny*nz), x-fastest (idx = x + y*nx + z*nx*ny), located at the regular
/// lattice nodes q = gridOrigin + (x,y,z) / gridInvSpacing in canonical (unrotated, unit-scale)
/// particle space. Analytic shapes leave these zero.
struct ShapeDesc {
  int type;         // peclet::dem::ShapeKind
  F4 params;        // analytic parameters (see sdf_analytic); grid: params.x = bounding radius
  int shellOffset;  // start index into the flat shell-points View
  int numPoints;    // shell size; 0 => analytic single-probe (sphere center)
  // --- grid-SDF fields (type == SHAPE_GRID_SDF); zero for analytic shapes ---
  int gridOffset = 0;               // start index into the shared sdfGrid View
  int nx = 0, ny = 0, nz = 0;       // lattice dimensions
  F3 gridOrigin{0, 0, 0};           // canonical coord of node (0,0,0)
  F3 gridInvSpacing{0, 0, 0};       // 1 / node spacing, per axis
};

struct PlaneP {
  F3 point;
  F3 normal;
};

// Convenience aliases for the SoA particle views the narrow-phase reads.
using PosView = Kokkos::View<const float* [3], CpMem>;
using QuatView = Kokkos::View<const float* [4], CpMem>;
using ScalarF = Kokkos::View<const float*, CpMem>;
using ScalarI = Kokkos::View<const int*, CpMem>;
using ShellView = Kokkos::View<const float* [3], CpMem>;
using GridView = Kokkos::View<const float*, CpMem>;

KOKKOS_INLINE_FUNCTION F3 loadF3(PosView v, int i) {
  return F3{v(i, 0), v(i, 1), v(i, 2)};
}
KOKKOS_INLINE_FUNCTION F4 loadF4(QuatView v, int i) {
  return F4{v(i, 0), v(i, 1), v(i, 2), v(i, 3)};
}

/// Trilinearly sample an imported grid SDF at canonical point `p`. The query is clamped into the
/// sample lattice and the Euclidean distance from `p` to that clamped point is added back, so a
/// probe outside the stored box gets a monotonically growing (positive) distance instead of a flat
/// clamped value — the standard "clamp + residual" extension that keeps the far field well-signed
/// and the central-difference normal sane near the grid boundary. Interior samples are exact
/// trilinear. Requires nx,ny,nz >= 2 (the shape builder guarantees this).
KOKKOS_INLINE_FUNCTION float sampleGridSdf(F3 p, const ShapeDesc& d, GridView grid) {
  const float fx = (p.x - d.gridOrigin.x) * d.gridInvSpacing.x;
  const float fy = (p.y - d.gridOrigin.y) * d.gridInvSpacing.y;
  const float fz = (p.z - d.gridOrigin.z) * d.gridInvSpacing.z;
  const float cx = Kokkos::fmin(Kokkos::fmax(fx, 0.0f), (float)(d.nx - 1));
  const float cy = Kokkos::fmin(Kokkos::fmax(fy, 0.0f), (float)(d.ny - 1));
  const float cz = Kokkos::fmin(Kokkos::fmax(fz, 0.0f), (float)(d.nz - 1));
  const int ix = (int)cx, iy = (int)cy, iz = (int)cz;
  const int ix1 = ix < d.nx - 1 ? ix + 1 : ix;
  const int iy1 = iy < d.ny - 1 ? iy + 1 : iy;
  const int iz1 = iz < d.nz - 1 ? iz + 1 : iz;
  const float tx = cx - ix, ty = cy - iy, tz = cz - iz;
  const long nxny = (long)d.nx * d.ny;
  const int off = d.gridOffset;
  auto at = [&](int x, int y, int z) { return grid(off + (long)z * nxny + (long)y * d.nx + x); };
  const float c00 = at(ix, iy, iz) * (1 - tx) + at(ix1, iy, iz) * tx;
  const float c10 = at(ix, iy1, iz) * (1 - tx) + at(ix1, iy1, iz) * tx;
  const float c01 = at(ix, iy, iz1) * (1 - tx) + at(ix1, iy, iz1) * tx;
  const float c11 = at(ix, iy1, iz1) * (1 - tx) + at(ix1, iy1, iz1) * tx;
  const float c0 = c00 * (1 - ty) + c10 * ty;
  const float c1 = c01 * (1 - ty) + c11 * ty;
  const float val = c0 * (1 - tz) + c1 * tz;
  // residual distance from p to the clamped lattice point (voxel units -> canonical units).
  const float rx = (d.gridInvSpacing.x > 0.0f) ? (fx - cx) / d.gridInvSpacing.x : 0.0f;
  const float ry = (d.gridInvSpacing.y > 0.0f) ? (fy - cy) / d.gridInvSpacing.y : 0.0f;
  const float rz = (d.gridInvSpacing.z > 0.0f) ? (fz - cz) / d.gridInvSpacing.z : 0.0f;
  return val + Kokkos::sqrt(rx * rx + ry * ry + rz * rz);
}

/// Canonical-space SDF of a shape: analytic dispatch, or a trilinear grid sample for an imported
/// grid SDF. `grid` may be an empty View when no grid shape is present.
KOKKOS_INLINE_FUNCTION float sdfEvalShape(F3 p, const ShapeDesc& d, GridView grid) {
  if (d.type == SHAPE_GRID_SDF)
    return sampleGridSdf(p, d, grid);
  return sdfEval(p, d.type, d.params);
}

/// Pair point-shell vs SDF contacts. pairs[numPairs][2] are (idA,idB) from the broad-phase; emits
/// ContactC into outContacts guarded by atomic outCount (clamped to outContacts.extent(0)).
inline void detectContactsKokkos(Kokkos::View<const int* [2], CpMem> pairs, int numPairs,
                                 PosView pos, QuatView quat, ScalarF scale, ScalarI shapeId,
                                 Kokkos::View<const ShapeDesc*, CpMem> shapes, ShellView shell,
                                 float globalScale, float margin,
                                 Kokkos::View<ContactC*, CpMem> outContacts,
                                 Kokkos::View<int, CpMem> outCount,
                                 Kokkos::View<float, CpMem> maxOverlap, GridView sdfGrid = GridView{}) {
  CpExec space;
  const int maxContacts = static_cast<int>(outContacts.extent(0));
  Kokkos::parallel_for(
      "peclet::dem::np::contacts", Kokkos::RangePolicy<CpExec>(space, 0, numPairs),
      KOKKOS_LAMBDA(int idx) {
        const int idA = pairs(idx, 0), idB = pairs(idx, 1);
        const ShapeDesc dA = shapes(shapeId(idA));
        const ShapeDesc dB = shapes(shapeId(idB));
        const F3 posA = loadF3(pos, idA), posB = loadF3(pos, idB);
        const F4 qA = loadF4(quat, idA), qB = loadF4(quat, idB);
        const float scaleA = scale(idA), scaleB = scale(idB);

        const int countA = dA.numPoints;
        const bool sphereA = (dA.type == SPHERE);
        const int iter = (countA > 0) ? countA : 1;

        for (int k = 0; k < iter; ++k) {
          F3 pLocalA{0, 0, 0};
          float pointRadius = 0.0f;
          if (countA > 0) {
            const int s = dA.shellOffset + k;
            pLocalA = F3{shell(s, 0), shell(s, 1), shell(s, 2)};
          } else if (sphereA) {
            pointRadius = dA.params.x * scaleA * globalScale;
          }

          const F3 pWorld = add3(posA, rotateVector(qA, scale3(pLocalA, scaleA)));
          const F3 pLocalB = invRotateVector(qB, sub3(pWorld, posB));
          const F3 pCanB = scale3(pLocalB, 1.0f / scaleB);

          const float dist = sdfEvalShape(pCanB, dB, sdfGrid) * scaleB;
          const float effDist = dist - pointRadius;
          if (effDist >= margin)
            continue;

          if (effDist < 0.0f)
            Kokkos::atomic_max(&maxOverlap(), -effDist);
          const int slot = Kokkos::atomic_fetch_add(&outCount(), 1);
          if (slot >= maxContacts) {
            Kokkos::atomic_add(&outCount(), -1);
            continue;
          }

          const float eps = 1e-4f;
          F3 nLoc{sdfEvalShape(F3{pCanB.x + eps, pCanB.y, pCanB.z}, dB, sdfGrid) -
                      sdfEvalShape(F3{pCanB.x - eps, pCanB.y, pCanB.z}, dB, sdfGrid),
                  sdfEvalShape(F3{pCanB.x, pCanB.y + eps, pCanB.z}, dB, sdfGrid) -
                      sdfEvalShape(F3{pCanB.x, pCanB.y - eps, pCanB.z}, dB, sdfGrid),
                  sdfEvalShape(F3{pCanB.x, pCanB.y, pCanB.z + eps}, dB, sdfGrid) -
                      sdfEvalShape(F3{pCanB.x, pCanB.y, pCanB.z - eps}, dB, sdfGrid)};
          const float len = len3(nLoc);
          nLoc = (len > 1e-9f) ? scale3(nLoc, 1.0f / len) : F3{0, 1, 0};

          const F3 nWorld = rotateVector(qB, nLoc);
          const F3 pSurfA = sub3(pWorld, scale3(nWorld, pointRadius));
          const F3 rA = sub3(pSurfA, posA);
          const F3 rB = sub3(sub3(pSurfA, scale3(nWorld, effDist)), posB);

          ContactC c{};
          c.bodyA = idA;
          c.bodyB = idB;
          c.normal = F4{nWorld.x, nWorld.y, nWorld.z, 0.0f};
          c.rA = F4{rA.x, rA.y, rA.z, 0.0f};
          c.rB = F4{rB.x, rB.y, rB.z, 0.0f};
          c.dist = effDist;
          c.friction_lambda_n = 0.0f;
          c.weight = 0.0f;
          outContacts(slot) = c;
        }
      });
  space.fence();
}

/// Per-real-particle contacts against explicit planes (point-shell shapes test each surface point;
/// analytic spheres use centre-minus-radius). bodyB = -1; plane anchor stored in rB.
inline void detectBoundaryKokkos(int numReal, int numPlanes, PosView pos, QuatView quat,
                                 ScalarF scale, ScalarI shapeId,
                                 Kokkos::View<const ShapeDesc*, CpMem> shapes, ShellView shell,
                                 Kokkos::View<const PlaneP*, CpMem> planes, float globalScale,
                                 float margin, Kokkos::View<ContactC*, CpMem> outContacts,
                                 Kokkos::View<int, CpMem> outCount,
                                 Kokkos::View<float, CpMem> maxOverlap) {
  CpExec space;
  const int maxContacts = static_cast<int>(outContacts.extent(0));
  Kokkos::parallel_for(
      "peclet::dem::np::boundary", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const F3 posA = loadF3(pos, i);
        const float s = scale(i) * globalScale;
        const ShapeDesc d = shapes(shapeId(i));
        float baseR = d.params.x;
        if (baseR == 0.0f)
          baseR = 1.0f;
        const float radius = baseR * s;
        const int numPts = d.numPoints;
        const F4 qA = loadF4(quat, i);

        for (int pi = 0; pi < numPlanes; ++pi) {
          const PlaneP pl = planes(pi);
          // emit helper, inlined (no lambda capture of mutable counters across backends).
          if (numPts > 0) {
            for (int k = 0; k < numPts; ++k) {
              const int si = d.shellOffset + k;
              const F3 rA =
                  rotateVector(qA, scale3(F3{shell(si, 0), shell(si, 1), shell(si, 2)}, s));
              const F3 pwk = add3(posA, rA);
              const float dist = dot3(sub3(pwk, pl.point), pl.normal);
              if (dist >= margin)
                continue;
              if (dist < 0.0f)
                Kokkos::atomic_max(&maxOverlap(), -dist);
              const int slot = Kokkos::atomic_fetch_add(&outCount(), 1);
              if (slot >= maxContacts) {
                Kokkos::atomic_add(&outCount(), -1);
                continue;
              }
              ContactC c{};
              c.bodyA = i;
              c.bodyB = -1;
              c.normal = F4{pl.normal.x, pl.normal.y, pl.normal.z, 0.0f};
              c.rA = F4{rA.x, rA.y, rA.z, 0.0f};
              c.rB = F4{pl.point.x, pl.point.y, pl.point.z, 0.0f};
              c.dist = dist;
              c.friction_lambda_n = 0.0f;
              c.weight = 0.0f;
              outContacts(slot) = c;
            }
          } else {
            const float dist = dot3(sub3(posA, pl.point), pl.normal) - radius;
            if (dist >= margin)
              continue;
            if (dist < 0.0f)
              Kokkos::atomic_max(&maxOverlap(), -dist);
            const int slot = Kokkos::atomic_fetch_add(&outCount(), 1);
            if (slot >= maxContacts) {
              Kokkos::atomic_add(&outCount(), -1);
              continue;
            }
            const F3 rA = scale3(pl.normal, -radius);
            ContactC c{};
            c.bodyA = i;
            c.bodyB = -1;
            c.normal = F4{pl.normal.x, pl.normal.y, pl.normal.z, 0.0f};
            c.rA = F4{rA.x, rA.y, rA.z, 0.0f};
            c.rB = F4{pl.point.x, pl.point.y, pl.point.z, 0.0f};
            c.dist = dist;
            c.friction_lambda_n = 0.0f;
            c.weight = 0.0f;
            outContacts(slot) = c;
          }
        }
      });
  space.fence();
}

}  // namespace peclet::dem

#endif  // DEM_NARROWPHASE_HPP
