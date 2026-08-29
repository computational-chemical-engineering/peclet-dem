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
#include "peclet/core/geom/scene.hpp"

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
  // --- grid-SDF descriptor (type == SHAPE_GRID_SDF); default/zero for analytic shapes ---
  // Carries core's descriptor DIRECTLY rather than loose fields the sampler marshals per call.
  // That is not cosmetic: building a GridDesc inside the sampler shifted nvcc's FMA contraction in
  // the trilinear chain and moved 185/4000 CUDA results by ~1 ULP (Layer 0 rung 3). Passing a
  // prebuilt descriptor through is bit-identical to the pre-port code on every backend.
  peclet::core::geom::GridDesc<float> grid;  // extension = kObject (a body)
  // --- composed-analytic descriptor (type == SHAPE_SCENE); null/-1 for everything else ---
  // A raw pointer into the Simulation's pooled device node View (same pattern as WallSdf.nodes:
  // the owning View lives on the Simulation so ShapeDesc stays POD and device-copyable). Child
  // indices are ABSOLUTE into the pool (rebased at add_scene_shape, like addAnalyticWall).
  const peclet::core::geom::ShapeNode<float>* nodes = nullptr;
  int nodeCount = 0;
  int shapeRoot = -1;
};

struct PlaneP {
  F3 point;
  F3 normal;
};

/// Static, world-space SDF container/geometry the particles collide against (a drum barrel, a hopper,
/// a vibrating tray, ...). Unlike a particle grid SDF (canonical body space, `sdfEvalShape`), this
/// field is sampled directly in WORLD coordinates, and its zero level set is the container wall.
///
/// SIGN: positive in the void where the grains live, negative inside the solid wall — so a grain
/// surface point has SDF > 0 when clear and SDF < 0 when it has poked into the wall (the penetration
/// depth), and the outward gradient points from the wall back into the void (the push-out normal).
///
/// The geometry never moves, but it carries a rigid-body SURFACE VELOCITY field
///   v(x) = linVel + angVel × (x − center)
/// evaluated at the contact point, so a grain touching the wall feels the wall's motion (a rotating
/// drum drags grains up its rising side; a translating `linVel` set sinusoidally each step is a
/// vibrating wall) even though the field itself is static. Plus a binary (particle–wall) material.
struct WallSdf {
  // world-space grid SDF samples live in Particles::wallGrid at [offset, offset + nx*ny*nz),
  // x-fastest (idx = x + y*nx + z*nx*ny), at nodes q = origin + (x,y,z)/invSpacing.
  // extension = kContainer -- see the sign discussion on sampleWallSdf below.
  peclet::core::geom::GridDesc<float> grid;
  // ANALYTIC alternative (Layer 1): when shapeRoot >= 0 the wall is a core shape tree evaluated
  // exactly, not a sampled field -- a stirrer becomes a CSG expression instead of a voxel grid,
  // with no resolution limit and no per-rank replicated sample pool. Raw pointer (like
  // core::geom::SceneView) so WallSdf stays POD and device-copyable with no signature churn at the
  // call sites. `sign` is -1 for the usual case of a CONTAINER built from an object tree: core's
  // leaves are negative-inside-solid, while a wall must read positive in the void where the grains
  // live.
  const peclet::core::geom::ShapeNode<float>* nodes = nullptr;
  int nodeCount = 0;
  int shapeRoot = -1;
  float sign = 1.0f;
  // rigid-body surface velocity field v(x) = linVel + angVel × (x − center) (set from the host).
  F3 linVel{0, 0, 0};
  F3 angVel{0, 0, 0};
  F3 center{0, 0, 0};
  // binary particle–wall material (independent of the body-body material).
  float restitution = 0.0f;
  float friction = 0.0f;
  // Optional material id: >= 0 and a pair table set -> particle-wall (e, mu) comes from the
  // pair table row (materialId(particle), materialId) instead of the binary values above.
  int materialId = -1;
};

/// Pair-material lookup: flat [K*K*2] table, entry ((a*K + b)*2) = restitution, +1 = friction.
constexpr int kMaxMaterials = 8;
using PairTableView = Kokkos::View<const float*, CpMem>;
using MatIdView = Kokkos::View<const unsigned char*, CpMem>;

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
/// sample lattice and the Euclidean distance from `p` to that clamped point is ADDED back (the
/// kObject policy), so a probe outside the stored box gets a monotonically growing (positive)
/// distance instead of a flat clamped value — the standard "clamp + residual" extension that keeps
/// the far field well-signed and the central-difference normal sane near the grid boundary.
/// Interior samples are exact trilinear. Requires nx,ny,nz >= 2 (the shape builder guarantees it).
///
/// RELOCATED (Layer 0 rung 3): the body is now peclet::core::geom::sampleGrid, which carries this
/// and the wall variant below as ONE routine parameterised by GridExtension — the two used to be
/// near-duplicate functions differing only in the residual's sign, which is exactly the kind of
/// silent divergence that cost 71k grains once (see sampleWallSdf).
KOKKOS_INLINE_FUNCTION float sampleGridSdf(F3 p, const ShapeDesc& d, GridView grid) {
  return peclet::core::geom::sampleGrid(toCoreVec(p), d.grid, grid);
}

/// Trilinearly sample a static world-space wall SDF at world point `p`. Positive in the void,
/// negative in the wall. The off-grid extension SUBTRACTS the clamp residual (the kContainer
/// policy) — the OPPOSITE of sampleGridSdf's object convention — because a wall SDF is a
/// CONTAINER: the void is bounded and everything beyond the stored box is wall-side, so a probe
/// outside the box must read an ever more negative (deeper-in-the-wall) distance. Adding the
/// residual here (the old behaviour) made a grain pushed past the box boundary — e.g. squeezed
/// through the floor plane, whose zero level sits exactly on the grid's lower face — read "clear"
/// and free-fall out of the simulation forever: a 180k glass-bead pile lost 71k grains through the
/// distributor during settling. That is why the sign is now a NAMED POLICY on one shared routine
/// (Layer 0 rung 3) instead of a one-character difference between two copied functions.
KOKKOS_INLINE_FUNCTION float sampleWallSdf(F3 p, const WallSdf& w, GridView grid) {
  if (w.shapeRoot >= 0)  // analytic wall: exact, no lattice
    return w.sign * peclet::core::geom::evalTree<float>(
                        peclet::core::geom::TablePtr<peclet::core::geom::ShapeNode<float>>{w.nodes},
                        w.nodeCount, w.shapeRoot, toCoreVec(p),
                        peclet::core::geom::TablePtr<peclet::core::geom::GridDesc<float>>{nullptr},
                        peclet::core::geom::PoolPtr<float>{nullptr});
  return peclet::core::geom::sampleGrid(toCoreVec(p), w.grid, grid);
}

/// Canonical-space SDF of a shape: analytic dispatch, or a trilinear grid sample for an imported
/// grid SDF. `grid` may be an empty View when no grid shape is present.
KOKKOS_INLINE_FUNCTION float sdfEvalShape(F3 p, const ShapeDesc& d, GridView grid) {
  if (d.type == SHAPE_GRID_SDF)
    return sampleGridSdf(p, d, grid);
  if (d.type == SHAPE_SCENE)  // composed analytic tree, exact in canonical body space
    return peclet::core::geom::evalTree<float>(
        peclet::core::geom::TablePtr<peclet::core::geom::ShapeNode<float>>{d.nodes}, d.nodeCount,
        d.shapeRoot, toCoreVec(p), peclet::core::geom::TablePtr<peclet::core::geom::GridDesc<float>>{nullptr},
        peclet::core::geom::PoolPtr<float>{nullptr});
  return sdfEval(p, d.type, d.params);
}

/// Outward gradient of a shape's SDF at canonical point `p`, NOT normalised (callers normalise,
/// matching the previous finite-difference code).
///
/// EXACT closed form for the analytic kinds (Layer 1). The old path spent SIX extra sdfEvalShape
/// calls per shell point on a central difference with a fixed eps = 1e-4 in canonical units, which
/// is both slower and less accurate: 1e-4 is a real length in canonical space, so a shape feature
/// or a per-particle scale far from 1 degrades the normal. A grid SDF has no closed form and keeps
/// the central difference.
KOKKOS_INLINE_FUNCTION F3 sdfGradShape(F3 p, const ShapeDesc& d, GridView grid) {
  namespace prim = peclet::core::geom::prim;
  const peclet::core::Vec3<float> q = toCoreVec(p);
  peclet::core::Vec3<float> g;
  switch (d.type) {
    case SPHERE:
      g = prim::Sphere<float>{d.params.x}.grad(q);
      break;
    case BOX:
      g = prim::Box<float>{d.params.x, d.params.y, d.params.z}.grad(q);
      break;
    case HOLLOW_CYLINDER:
      g = prim::HollowCylinder<float>{d.params.x, d.params.y, d.params.z}.grad(q);
      break;
    default: {  // SHAPE_GRID_SDF / SHAPE_SCENE: central difference (a tree has no closed-form
                // gradient at the runtime level yet; the fixed eps caveat from Layer 1 applies)
      const float eps = 1e-4f;
      return F3{sdfEvalShape(F3{p.x + eps, p.y, p.z}, d, grid) -
                    sdfEvalShape(F3{p.x - eps, p.y, p.z}, d, grid),
                sdfEvalShape(F3{p.x, p.y + eps, p.z}, d, grid) -
                    sdfEvalShape(F3{p.x, p.y - eps, p.z}, d, grid),
                sdfEvalShape(F3{p.x, p.y, p.z + eps}, d, grid) -
                    sdfEvalShape(F3{p.x, p.y, p.z - eps}, d, grid)};
    }
  }
  return F3{g.x, g.y, g.z};
}

/// Pair point-shell vs SDF contacts. pairs[numPairs][2] are (idA,idB) from the broad-phase; emits
/// ContactC into outContacts guarded by atomic outCount (clamped to outContacts.extent(0)).
inline void detectContactsKokkos(Kokkos::View<const int* [2], CpMem> pairs, int numPairs,
                                 PosView pos, QuatView quat, ScalarF scale, ScalarI shapeId,
                                 Kokkos::View<const ShapeDesc*, CpMem> shapes, ShellView shell,
                                 float globalScale, float margin,
                                 Kokkos::View<ContactC*, CpMem> outContacts,
                                 Kokkos::View<int, CpMem> outCount,
                                 Kokkos::View<float, CpMem> maxOverlap, GridView sdfGrid = GridView{},
                                 MatIdView matId = MatIdView{}, PairTableView pairTable = PairTableView{}) {
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
        // Canonical->world size = per-particle scale * global scale. globalScale must appear in the
        // shell placement, the canonical remap of B, and the distance rescale exactly as it does in
        // the sphere probe radius below — else a non-unit global_scale makes A's and B's radii
        // disagree (grains a real diameter apart read a huge penetration and the solver explodes).
        // The plane/wall boundary kernels already fold globalScale into their `s = scale*globalScale`.
        const float effScaleA = scale(idA) * globalScale, effScaleB = scale(idB) * globalScale;

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
            pointRadius = dA.params.x * effScaleA;
          }

          const F3 pWorld = add3(posA, rotateVector(qA, scale3(pLocalA, effScaleA)));
          const F3 pLocalB = invRotateVector(qB, sub3(pWorld, posB));
          const F3 pCanB = scale3(pLocalB, 1.0f / effScaleB);

          const float dist = sdfEvalShape(pCanB, dB, sdfGrid) * effScaleB;
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

          F3 nLoc = sdfGradShape(pCanB, dB, sdfGrid);
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
          if (pairTable.extent(0) > 0) {  // per-pair body-body material
            const int t = (int(matId(idA)) * kMaxMaterials + int(matId(idB))) * 2;
            c.boundaryRestitution = pairTable(t);
            c.boundaryFriction = pairTable(t + 1);
          }
          outContacts(slot) = c;
        }
      });
  space.fence();
}

/// Per-real-particle contacts against a static world-space wall SDF set (a drum barrel, hopper,
/// vibrating tray, ...). Mirrors detectBoundaryKokkos but the boundary is the wall's zero level set:
/// each surface point (or the sphere centre) is tested against sampleWallSdf, the outward gradient is
/// the contact normal, and every emitted contact carries the wall's rigid-body surface velocity at
/// the contact point plus the wall's binary (particle–wall) restitution/friction — so the moving-wall
/// terms flow through the manifold velocity solve and the per-contact friction sweep. bodyB = -1
/// (a boundary, like a plane); the wall surface point is stored in rB for the position solve's
/// plane-linearised non-penetration constraint.
inline void detectWallSdfKokkos(int numReal, int numWalls, PosView pos, QuatView quat, ScalarF scale,
                                ScalarI shapeId, Kokkos::View<const ShapeDesc*, CpMem> shapes,
                                ShellView shell, Kokkos::View<const WallSdf*, CpMem> walls,
                                GridView wallGrid, float globalScale, float margin,
                                Kokkos::View<ContactC*, CpMem> outContacts,
                                Kokkos::View<int, CpMem> outCount,
                                Kokkos::View<float, CpMem> maxOverlap,
                                MatIdView matId = MatIdView{}, PairTableView pairTable = PairTableView{}) {
  CpExec space;
  const int maxContacts = static_cast<int>(outContacts.extent(0));
  Kokkos::parallel_for(
      "peclet::dem::np::wallsdf", Kokkos::RangePolicy<CpExec>(space, 0, numReal),
      KOKKOS_LAMBDA(int i) {
        const F3 posA = loadF3(pos, i);
        const float s = scale(i) * globalScale;
        const ShapeDesc d = shapes(shapeId(i));
        float baseR = d.params.x;
        if (baseR == 0.0f)
          baseR = 1.0f;
        const float radius = baseR * s;  // analytic sphere radius (numPts == 0)
        const int numPts = d.numPoints;
        const F4 qA = loadF4(quat, i);
        const float eps = 1e-4f;

        for (int wi = 0; wi < numWalls; ++wi) {
          const WallSdf w = walls(wi);
          const int iter = (numPts > 0) ? numPts : 1;
          for (int k = 0; k < iter; ++k) {
            // World surface probe: a shell point (rotated + scaled) or, for a sphere, the centre.
            F3 rA{0, 0, 0};
            if (numPts > 0) {
              const int si = d.shellOffset + k;
              rA = rotateVector(qA, scale3(F3{shell(si, 0), shell(si, 1), shell(si, 2)}, s));
            }
            const F3 pw = add3(posA, rA);
            const float sdf = sampleWallSdf(pw, w, wallGrid);
            // Outward SDF gradient (central difference) = push-out normal (wall -> void).
            F3 n{sampleWallSdf(F3{pw.x + eps, pw.y, pw.z}, w, wallGrid) -
                     sampleWallSdf(F3{pw.x - eps, pw.y, pw.z}, w, wallGrid),
                 sampleWallSdf(F3{pw.x, pw.y + eps, pw.z}, w, wallGrid) -
                     sampleWallSdf(F3{pw.x, pw.y - eps, pw.z}, w, wallGrid),
                 sampleWallSdf(F3{pw.x, pw.y, pw.z + eps}, w, wallGrid) -
                     sampleWallSdf(F3{pw.x, pw.y, pw.z - eps}, w, wallGrid)};
            const float ln = len3(n);
            n = (ln > 1e-9f) ? scale3(n, 1.0f / ln) : F3{0, 1, 0};

            // For a sphere the nearest surface point is one radius toward the wall (−normal); its
            // signed gap to the wall is sdf − radius.
            const float dist = (numPts > 0) ? sdf : sdf - radius;
            if (dist >= margin)
              continue;
            const F3 rAeff = (numPts > 0) ? rA : scale3(n, -radius);
            const F3 pSurfA = add3(posA, rAeff);           // particle surface point
            const F3 pWall = sub3(pSurfA, scale3(n, dist));  // point on the wall along the normal

            if (dist < 0.0f)
              Kokkos::atomic_max(&maxOverlap(), -dist);
            const int slot = Kokkos::atomic_fetch_add(&outCount(), 1);
            if (slot >= maxContacts) {
              Kokkos::atomic_add(&outCount(), -1);
              continue;
            }
            // Rigid-body wall surface velocity at the contact point: linVel + angVel × (r − center).
            const F3 r = sub3(pWall, w.center);
            const F3 vWall = add3(w.linVel, cross3v(w.angVel, r));

            ContactC c{};
            c.bodyA = i;
            c.bodyB = -1;
            c.normal = F4{n.x, n.y, n.z, 0.0f};
            c.rA = F4{rAeff.x, rAeff.y, rAeff.z, 0.0f};
            c.rB = F4{pWall.x, pWall.y, pWall.z, 0.0f};
            c.dist = dist;
            c.friction_lambda_n = 0.0f;
            c.weight = 0.0f;
            c.boundaryVel = F4{vWall.x, vWall.y, vWall.z, 0.0f};
            if (w.materialId >= 0 && pairTable.extent(0) > 0) {
              const int t = (int(matId(i)) * kMaxMaterials + w.materialId) * 2;
              c.boundaryRestitution = pairTable(t);
              c.boundaryFriction = pairTable(t + 1);
            } else {
              c.boundaryRestitution = w.restitution;
              c.boundaryFriction = w.friction;
            }
            outContacts(slot) = c;
          }
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
