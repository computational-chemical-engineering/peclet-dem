// packing-gpu — portable (Kokkos) narrow-phase: SDF point-shell collision + boundary planes.
//
// Kokkos port of detect_contacts_kernel / detect_boundary_kernel (narrowphase.cu) over the particle
// SoA expressed as Kokkos Views (positions/quaternions as View<float*[3]>/[4], scales/shape-ids as
// View<float*>/<int*> — backend-default layout, so coalesced on GPU and cache-friendly on CPU). The
// per-point math is a faithful copy of the CUDA kernels (same scale/global_scale handling, same
// central-difference normal, same contact geometry) so results match. Analytic shapes only for now;
// grid-SDF (texture) returns +inf (dem::sdfEval), as in the CUDA placeholder.
#ifndef DEM_NARROWPHASE_HPP
#define DEM_NARROWPHASE_HPP

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ContactC, CpExec/CpMem
#include "dem_portable.hpp"

namespace dem {

/// Portable mirror of ShapeDescriptor (analytic fields + a flat-array point shell).
struct ShapeDesc {
  int type;        // dem::ShapeKind
  F4 params;       // analytic parameters (see sdf_analytic)
  int shellOffset; // start index into the flat shell-points View
  int numPoints;   // shell size; 0 => analytic single-probe (sphere center)
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

KOKKOS_INLINE_FUNCTION F3 loadF3(PosView v, int i) { return F3{v(i, 0), v(i, 1), v(i, 2)}; }
KOKKOS_INLINE_FUNCTION F4 loadF4(QuatView v, int i) { return F4{v(i, 0), v(i, 1), v(i, 2), v(i, 3)}; }

/// Pair point-shell vs SDF contacts. pairs[numPairs][2] are (idA,idB) from the broad-phase; emits
/// ContactC into outContacts guarded by atomic outCount (clamped to outContacts.extent(0)).
inline void detectContactsKokkos(Kokkos::View<const int* [2], CpMem> pairs, int numPairs, PosView pos,
                                 QuatView quat, ScalarF scale, ScalarI shapeId,
                                 Kokkos::View<const ShapeDesc*, CpMem> shapes, ShellView shell,
                                 float globalScale, float margin,
                                 Kokkos::View<ContactC*, CpMem> outContacts,
                                 Kokkos::View<int, CpMem> outCount, Kokkos::View<float, CpMem> maxOverlap) {
  CpExec space;
  const int maxContacts = static_cast<int>(outContacts.extent(0));
  Kokkos::parallel_for(
      "dem::np::contacts", Kokkos::RangePolicy<CpExec>(space, 0, numPairs), KOKKOS_LAMBDA(int idx) {
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

          const float dist = sdfEval(pCanB, dB.type, dB.params) * scaleB;
          const float effDist = dist - pointRadius;
          if (effDist >= margin) continue;

          if (effDist < 0.0f) Kokkos::atomic_max(&maxOverlap(), -effDist);
          const int slot = Kokkos::atomic_fetch_add(&outCount(), 1);
          if (slot >= maxContacts) {
            Kokkos::atomic_add(&outCount(), -1);
            continue;
          }

          const float eps = 1e-4f;
          F3 nLoc{sdfEval(F3{pCanB.x + eps, pCanB.y, pCanB.z}, dB.type, dB.params) -
                      sdfEval(F3{pCanB.x - eps, pCanB.y, pCanB.z}, dB.type, dB.params),
                  sdfEval(F3{pCanB.x, pCanB.y + eps, pCanB.z}, dB.type, dB.params) -
                      sdfEval(F3{pCanB.x, pCanB.y - eps, pCanB.z}, dB.type, dB.params),
                  sdfEval(F3{pCanB.x, pCanB.y, pCanB.z + eps}, dB.type, dB.params) -
                      sdfEval(F3{pCanB.x, pCanB.y, pCanB.z - eps}, dB.type, dB.params)};
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
                                 Kokkos::View<int, CpMem> outCount, Kokkos::View<float, CpMem> maxOverlap) {
  CpExec space;
  const int maxContacts = static_cast<int>(outContacts.extent(0));
  Kokkos::parallel_for(
      "dem::np::boundary", Kokkos::RangePolicy<CpExec>(space, 0, numReal), KOKKOS_LAMBDA(int i) {
        const F3 posA = loadF3(pos, i);
        const float s = scale(i) * globalScale;
        const ShapeDesc d = shapes(shapeId(i));
        float baseR = d.params.x;
        if (baseR == 0.0f) baseR = 1.0f;
        const float radius = baseR * s;
        const int numPts = d.numPoints;
        const F4 qA = loadF4(quat, i);

        for (int pi = 0; pi < numPlanes; ++pi) {
          const PlaneP pl = planes(pi);
          // emit helper, inlined (no lambda capture of mutable counters across backends).
          if (numPts > 0) {
            for (int k = 0; k < numPts; ++k) {
              const int si = d.shellOffset + k;
              const F3 rA = rotateVector(qA, scale3(F3{shell(si, 0), shell(si, 1), shell(si, 2)}, s));
              const F3 pwk = add3(posA, rA);
              const float dist = dot3(sub3(pwk, pl.point), pl.normal);
              if (dist >= margin) continue;
              if (dist < 0.0f) Kokkos::atomic_max(&maxOverlap(), -dist);
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
            if (dist >= margin) continue;
            if (dist < 0.0f) Kokkos::atomic_max(&maxOverlap(), -dist);
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

}  // namespace dem

#endif  // DEM_NARROWPHASE_HPP
