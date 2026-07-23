/// @file
/// @brief dem — portable (Kokkos) particle SoA container: the storage the dem flip pivots on.
///
/// Replaces ParticleSystemData's float4* arrays with Kokkos SoA Views (backend-default layout:
/// coalesced on GPU, cache-friendly on CPU; packed .w scalars split into their own arrays). Holds
/// the per-particle state, the predicted/delta buffers, the collision/contact/manifold buffers, the
/// atomic counters (rank-0 Views), and the static shape/plane data — everything the ported kernels
/// (broadphase_arborx / narrowphase / contact_preprocessing / solver_* / integration / periodicity)
/// operate on. allocate() sizes them for a given capacity.
#ifndef DEM_PARTICLES_HPP
#define DEM_PARTICLES_HPP

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "contact_preprocessing.hpp"  // ContactC, ManifoldC
#include "integration.hpp"            // Domain, V3/V4/Vf/Vi
#include "narrowphase.hpp"            // ShapeDesc, PlaneP

namespace peclet::dem {

struct Particles {
  // --- per-particle state (size = capacity) ---
  V3 pos;
  Vf invMass;
  V4 quat;
  V3 vel;
  V3 angVel;
  V3 invInertia;
  Vf scale;
  Vf targetScale;
  Vi shapeId;
  V3 posPred;
  V4 quatPred;
  V3 velPred;
  V3 angVelPred;
  V3 deltaPos;
  V4 deltaQuat;
  V3 deltaVel;
  V3 deltaAngVel;
  Vi constraintCounts;
  Vi realIndices;
  Kokkos::View<float* [2], CpMem> planeFriction;
  Vf rad;      // effective broadphase radius scratch (scale * globalScale)
  V3 extForce;  // per-particle external FORCE (e.g. fluid drag); F=ma => dv = extForce*invMass*dt

  // --- collision/contact/manifold buffers ---
  Kokkos::View<int* [2], CpMem> pairs;        // broadphase candidates (maxPairs)
  Kokkos::View<ContactC*, CpMem> contacts;    // narrowphase output (maxContacts)
  Kokkos::View<ManifoldC*, CpMem> manifolds;  // reduction output (maxContacts)
  // Graph-colouring scratch for the single-GPU colored Gauss–Seidel velocity solve: per-manifold
  // colour (maxContacts; -2 inactive, -1 uncoloured, >=0 colour), plus per-body arbitration winner
  // and committed-colour bitmask (both indexed by REAL body index, sized capacity).
  Kokkos::View<int*, CpMem> manifoldColor;      // per-manifold colour (velocity solve)
  // Persistent-contact restitution (gravity-gated): pair keys of this/last substep's manifolds +
  // the per-manifold "existed last substep" flag. A persistent contact is LOADED, not a fresh
  // impact — it gets e = 0 (the impulse still cancels the approach: pure inelastic support), which
  // is how the velocity solve carries a pile's static weight through impulse chains. Restitution
  // stays reserved for newly formed contacts (genuine impacts). |g| = 0 leaves all of this idle.
  Kokkos::View<unsigned long long*, CpMem> pairKeys;
  Kokkos::View<unsigned long long*, CpMem> prevPairKeys;
  Kokkos::View<unsigned char*, CpMem> manifoldPersistent;
  int prevPairCount = 0;
  // Grounded level per REAL body (Guendelman support levels, warm-started + decayed): 255 at a
  // wall/plane contact, propagated lower -> upper through the contact graph a few sweeps per
  // substep. One-sided (shock-propagation) impulses require the LOWER body grounded > 0, so a
  // gas-borne emulsion or lifted slug (no path to the floor) keeps momentum-conserving impulses
  // and its weight stays on the gas -- only genuinely supported chains drain into the ground.
  Kokkos::View<unsigned char*, CpMem> groundedLevel;
  // Warm-started PGS (projected Gauss-Seidel) velocity solve: per-manifold accumulated push
  // impulse (this substep), the previous substep's converged impulses (sorted alongside
  // prevPairKeys for the warm-start gather), and the pre-solve approach velocity (restitution
  // bias). At convergence lambdaAcc IS the contact force network (x dt).
  Kokkos::View<float*, CpMem> lambdaAcc;
  Kokkos::View<float*, CpMem> prevLambda;
  Kokkos::View<float*, CpMem> vn0;
  Kokkos::View<float* [3], CpMem> vt0;  // pre-solve tangential surface velocity (beta reference)
  // Friction-cone PGS: per-manifold accumulated tangential impulse (world frame, kept in the
  // contact tangent plane by projection) and the previous substep's converged values (sorted
  // alongside prevPairKeys for the warm-start gather). |lambdaT| <= mu * lambdaAcc at all times.
  Kokkos::View<float* [3], CpMem> lambdaT;
  Kokkos::View<float* [3], CpMem> prevLambdaT;
  // Position-channel normal load (see solvePositionColoredGSKokkos): per-contact positional
  // lambda this substep, its per-manifold impulse-equivalent, the previous substep's value
  // (warm-gathered) that tops up the friction cone's Coulomb bound, and the sorted store.
  Kokkos::View<float*, CpMem> hertzSnPair;   // per cached pair: last step's patch stiffness sum
  Kokkos::View<float*, CpMem> hertzSnWall;   // per (particle, wall): same, for wall patches
  Kokkos::View<float*, CpMem> posLambdaContact;
  Kokkos::View<float*, CpMem> posImpulse;      // gathered: LAST substep's position-channel load
  Kokkos::View<float*, CpMem> prevPosImpulse;  // sorted alongside prevPairKeys
  // Side flags for the STABILIZATION pass (0 = symmetric): zeroed for the main momentum-
  // conserving sweeps, filled from persistence+grounding only if statics fail to converge.
  Kokkos::View<unsigned char*, CpMem> sideFlags;
  // Level-ordered ("ordered") stabilization: per-REAL-body height-from-floor BFS level
  // (recomputed fresh each time the pass triggers) + the per-manifold (level, colour) bucket
  // key and permutation the ordered sweeps iterate through.
  Kokkos::View<int*, CpMem> heightLevel;
  Kokkos::View<int*, CpMem> levelKey;
  Kokkos::View<int*, CpMem> levelPerm;
  // Multilevel (GraphMG) stabilization scratch (solver_multilevel.hpp): packed per-manifold
  // 6-bit colour per level, pooled parent maps / group mass / group velocity arrays (levels
  // shrink >= 10% each, so the pools hold every level), the composed body -> group map, and
  // the matching scratch.
  Kokkos::View<long long*, CpMem> mlColorPacked;
  Kokkos::View<int*, CpMem> mlParent;
  Kokkos::View<float*, CpMem> mlInvMassG;
  Kokkos::View<float* [3], CpMem> mlVelG;
  Kokkos::View<float* [3], CpMem> mlVelG0;
  Kokkos::View<float*, CpMem> mlMassG;
  Kokkos::View<int*, CpMem> mlGrp;
  Kokkos::View<int*, CpMem> mlMate;
  // --- Hertz-Mindlin soft-sphere engine (solver_hertz.hpp): cached Verlet pair list state ---
  Kokkos::View<float* [3], CpMem> hertzXi;        // per cached pair: Mindlin shear history
  Kokkos::View<unsigned long long*, CpMem> hertzKeys;      // keys of the cached pairs
  Kokkos::View<unsigned long long*, CpMem> hertzPrevKeys;  // sorted keys of the PREVIOUS list
  Kokkos::View<float* [3], CpMem> hertzPrevXi;             // xi aligned with hertzPrevKeys
  Kokkos::View<float* [3], CpMem> hertzXiWall;    // per (particle, wall) shear history
  Kokkos::View<int*, CpMem> hertzWallCand;   // near-wall candidate slots (i * 8 + wallIdx)
  Kokkos::View<int, CpMem> hertzWallCandCount;
  int hertzNumWallCand = 0;
  // Effective Hertz contact-curvature radius for non-spherical shapes, as a fraction of the
  // bounding radius (true curvature is undefined at faces/edges; spheres use their real radius).
  float hertzContactRadiusFrac = 0.5f;
  Kokkos::View<float* [3], CpMem> hertzRefPos;    // positions at the last pair build
  Kokkos::View<float, CpMem> hertzDispMax;        // max |pos-ref|^2 since the build
  Kokkos::View<float*, CpMem> hertzE, hertzNu;    // per-material Young / Poisson
  int hertzNumPairs = -1;                          // -1: no valid cached list
  int hertzPrevCount = 0;
  static constexpr int kHertzMaxWalls = 4;
  // Stabilization mode of the staged velocity solve (Phase B; see sim.hpp): 0 = off (pure
  // symmetric PGS), 1 = one-sided grounded pass (default), 2 = multilevel (level-ordered
  // symmetric sweeps -- momentum is transported, never deleted), 3 = escalate (extra symmetric
  // sweeps; diagnostic/fallback).
  int stabilizationMode = 1;
  // Per-particle material id + flat pair-material table [kMaxMaterials^2 * 2] of (restitution,
  // friction) rows; zero-length pairMaterials = feature off (global material everywhere).
  Kokkos::View<unsigned char*, CpMem> materialId;
  Kokkos::View<float*, CpMem> pairMaterials;
  Kokkos::View<int*, CpMem> contactSlot;  // contact -> manifold slot (PGS friction bound)
  Kokkos::View<int*, CpMem> contactColor;       // per-contact colour (position solve)
  // Per-body round-winner key for the colouring arbitration. 64-bit: hashed-random priority in the
  // high word (splitmix32 of the edge index), the unique edge index in the low word. Random
  // priorities give O(log n) arbitration rounds w.h.p.; RAW indices are adversarial for poured
  // lattice beds (monotone index chains -> O(chain) rounds -> minutes per step at 1M grains).
  Kokkos::View<long long*, CpMem> bodyWinner;
  Kokkos::View<std::uint64_t*, CpMem> bodyColorMask;  // per-body committed-colour bitmask

  // --- atomic counters / scalars (rank-0 Views) ---
  Kokkos::View<int, CpMem> pairCount, contactCount, manifoldCount, topGhost;
  Kokkos::View<float, CpMem> maxOverlap;
  // Max physical approach speed among approaching manifolds in the last velocity sweep — drives the
  // colored-GS velocity loop's adaptive stop (converged once no pair approaches above the resting
  // threshold). maxOverlap plays the same role for the position loop.
  Kokkos::View<float, CpMem> maxApproach;
  // Quasi-static share of maxApproach (corrections on contacts with |vn0| <= 4 vRest): the
  // multilevel stabilization loop's stop criterion -- flowing scenes keep ballistic churn out
  // of it, so the pass ends after ~one cycle instead of burning its full budget as an
  // over-convergence brake on discharge.
  Kokkos::View<float, CpMem> maxApproachQS;

  // --- static geometry ---
  Kokkos::View<ShapeDesc*, CpMem> shapes;
  Kokkos::View<float* [3], CpMem> shell;
  Kokkos::View<PlaneP*, CpMem> planes;
  Kokkos::View<float*, CpMem> sdfGrid;  // concatenated grid-SDF samples (imported shapes)
  // static world-space SDF walls (drum barrel, hopper, vibrating tray) + their concatenated samples.
  Kokkos::View<WallSdf*, CpMem> walls;
  Kokkos::View<float*, CpMem> wallGrid;

  // --- sizes & params (host) ---
  int capacity = 0, numReal = 0, numParticles = 0;
  int maxPairs = 0, maxContacts = 0, numPlanes = 0, numWalls = 0;
  // max wall friction (host) — gates the friction path so a frictional wall works even with a
  // frictionless body-body material (global frictionDynamic == 0).
  float wallFrictionMax = 0.0f;
  Domain domain{};
  F3 gravity{0, 0, 0};
  float dt = 1e-3f, globalScale = 1.0f, growthRate = 0.0f, growthFactor = -1.0f;
  float baseRadius = 1.0f;  // shape canonical radius; effective radius = scale * globalScale * baseRadius
  float thermostatTau = 0.0f, thermostatTemp = 0.0f,
        thermostatKB = 1.0f;  // Berendsen (tau>0 enables)
  float frictionDynamic = 0.0f, restitutionNormal = 0.0f, skin = 0.1f;
  // Walton tangential restitution beta (0 = tangentially dead stick, the pre-beta behaviour;
  // > 0 reverses the pre-collision tangential surface velocity of COLLIDING contacts).
  float restitutionTangent = 0.0f;
  int positionIterations = 10, velocityIterations = 0;
  // Single-GPU collision solves: true = colored Gauss–Seidel for BOTH the restitution (velocity) and
  // the overlap (position) solve — correct coupled multi-contact impulses + non-penetration, default;
  // false = count-averaged Jacobi (the legacy robust path, still used by step_mpi).
  bool velocityUseGS = true;

  // nPlanes is the plane-array CAPACITY; numPlanes (the live count) stays 0 until planes are added.
  void allocate(int cap, int maxPairs_, int maxContacts_, int nShapes, int nShell, int nPlanes) {
    capacity = cap;
    maxPairs = maxPairs_;
    maxContacts = maxContacts_;
    numPlanes = 0;
    pos = V3("pos", cap);
    invMass = Vf("invMass", cap);
    quat = V4("quat", cap);
    vel = V3("vel", cap);
    angVel = V3("angVel", cap);
    invInertia = V3("invInertia", cap);
    scale = Vf("scale", cap);
    targetScale = Vf("targetScale", cap);
    shapeId = Vi("shapeId", cap);
    posPred = V3("posPred", cap);
    quatPred = V4("quatPred", cap);
    velPred = V3("velPred", cap);
    angVelPred = V3("angVelPred", cap);
    deltaPos = V3("deltaPos", cap);
    deltaQuat = V4("deltaQuat", cap);
    deltaVel = V3("deltaVel", cap);
    deltaAngVel = V3("deltaAngVel", cap);
    constraintCounts = Vi("constraintCounts", cap);
    realIndices = Vi("realIndices", cap);
    planeFriction = Kokkos::View<float* [2], CpMem>("planeFriction", cap);
    rad = Vf("rad", cap);
    extForce = V3("extForce", cap);  // zero-initialised => no external force by default
    pairs = Kokkos::View<int* [2], CpMem>("pairs", maxPairs);
    contacts = Kokkos::View<ContactC*, CpMem>("contacts", maxContacts);
    manifolds = Kokkos::View<ManifoldC*, CpMem>("manifolds", maxContacts);
    manifoldColor = Kokkos::View<int*, CpMem>("manifoldColor", maxContacts);
    pairKeys = Kokkos::View<unsigned long long*, CpMem>("pairKeys", maxContacts);
    prevPairKeys = Kokkos::View<unsigned long long*, CpMem>("prevPairKeys", maxContacts);
    manifoldPersistent = Kokkos::View<unsigned char*, CpMem>("manifoldPersistent", maxContacts);
    prevPairCount = 0;
    contactColor = Kokkos::View<int*, CpMem>("contactColor", maxContacts);
    bodyWinner = Kokkos::View<long long*, CpMem>("bodyWinner", cap);
    bodyColorMask = Kokkos::View<std::uint64_t*, CpMem>("bodyColorMask", cap);
    groundedLevel = Kokkos::View<unsigned char*, CpMem>("groundedLevel", cap);
    materialId = Kokkos::View<unsigned char*, CpMem>("materialId", cap);
    lambdaAcc = Kokkos::View<float*, CpMem>("lambdaAcc", maxContacts);
    lambdaT = Kokkos::View<float* [3], CpMem>("lambdaT", maxContacts);
    posLambdaContact = Kokkos::View<float*, CpMem>("posLambdaContact", maxContacts);
    posImpulse = Kokkos::View<float*, CpMem>("posImpulse", maxContacts);
    prevPosImpulse = Kokkos::View<float*, CpMem>("prevPosImpulse", maxContacts);
    sideFlags = Kokkos::View<unsigned char*, CpMem>("sideFlags", maxContacts);
    heightLevel = Kokkos::View<int*, CpMem>("heightLevel", cap);
    levelKey = Kokkos::View<int*, CpMem>("levelKey", maxContacts);
    levelPerm = Kokkos::View<int*, CpMem>("levelPerm", maxContacts);
    mlColorPacked = Kokkos::View<long long*, CpMem>("mlColorPacked", maxContacts);
    mlParent = Kokkos::View<int*, CpMem>("mlParent", 5 * cap);
    mlInvMassG = Kokkos::View<float*, CpMem>("mlInvMassG", 4 * cap);
    mlVelG = Kokkos::View<float* [3], CpMem>("mlVelG", 4 * cap);
    mlVelG0 = Kokkos::View<float* [3], CpMem>("mlVelG0", 4 * cap);
    mlMassG = Kokkos::View<float*, CpMem>("mlMassG", 4 * cap);
    mlGrp = Kokkos::View<int*, CpMem>("mlGrp", cap);
    mlMate = Kokkos::View<int*, CpMem>("mlMate", cap);
    hertzXiWall = Kokkos::View<float* [3], CpMem>("hertzXiWall", cap * kHertzMaxWalls);
    hertzWallCand = Kokkos::View<int*, CpMem>("hertzWallCand", cap * 2);
    hertzSnWall = Kokkos::View<float*, CpMem>("hertzSnWall", cap * kHertzMaxWalls);
    hertzWallCandCount = Kokkos::View<int, CpMem>("hertzWallCandCount");
    hertzRefPos = Kokkos::View<float* [3], CpMem>("hertzRefPos", cap);
    hertzDispMax = Kokkos::View<float, CpMem>("hertzDispMax");
    hertzE = Kokkos::View<float*, CpMem>("hertzE", 8);
    hertzNu = Kokkos::View<float*, CpMem>("hertzNu", 8);
    {
      auto he = Kokkos::create_mirror_view(hertzE);
      auto hn = Kokkos::create_mirror_view(hertzNu);
      for (int i = 0; i < 8; ++i) {
        he(i) = 1.0e9f;
        hn(i) = 0.25f;
      }
      Kokkos::deep_copy(hertzE, he);
      Kokkos::deep_copy(hertzNu, hn);
    }
    prevLambdaT = Kokkos::View<float* [3], CpMem>("prevLambdaT", maxContacts);
    contactSlot = Kokkos::View<int*, CpMem>("contactSlot", maxContacts);
    prevLambda = Kokkos::View<float*, CpMem>("prevLambda", maxContacts);
    vn0 = Kokkos::View<float*, CpMem>("vn0", maxContacts);
    vt0 = Kokkos::View<float* [3], CpMem>("vt0", maxContacts);
    pairCount = Kokkos::View<int, CpMem>("pairCount");
    contactCount = Kokkos::View<int, CpMem>("contactCount");
    manifoldCount = Kokkos::View<int, CpMem>("manifoldCount");
    topGhost = Kokkos::View<int, CpMem>("topGhost");
    maxOverlap = Kokkos::View<float, CpMem>("maxOverlap");
    maxApproach = Kokkos::View<float, CpMem>("maxApproach");
    maxApproachQS = Kokkos::View<float, CpMem>("maxApproachQS");
    shapes = Kokkos::View<ShapeDesc*, CpMem>("shapes", nShapes > 0 ? nShapes : 1);
    shell = Kokkos::View<float* [3], CpMem>("shell", nShell > 0 ? nShell : 1);
    planes = Kokkos::View<PlaneP*, CpMem>("planes", nPlanes > 0 ? nPlanes : 1);
    sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", 1);    // resized by setSdfShape
    walls = Kokkos::View<WallSdf*, CpMem>("walls", 1);      // resized by addSdfWall
    wallGrid = Kokkos::View<float*, CpMem>("wallGrid", 1);  // concatenated wall samples
    numWalls = 0;
  }

  // Grow the per-particle SoA to hold at least `newCap` particles (real + periodic-ghost headroom),
  // preserving the existing [0,numReal) state (Kokkos::resize copies the overlapping subextent). The
  // single-GPU step sizes this from the domain before generating ghosts; without the headroom every
  // ghost slot overflows `capacity` and cross-boundary contacts silently vanish. A no-op when the
  // SoA is already large enough (e.g. the MPI path, whose caller pre-sizes capacity for the
  // worst-case ghost band). The collision buffers (pairs/contacts/manifolds) keep their
  // construction-time sizing — each real particle still issues one broad-phase query.
  void ensureCapacity(int newCap) {
    if (newCap <= capacity)
      return;
    Kokkos::resize(pos, newCap);
    Kokkos::resize(invMass, newCap);
    Kokkos::resize(quat, newCap);
    Kokkos::resize(vel, newCap);
    Kokkos::resize(angVel, newCap);
    Kokkos::resize(invInertia, newCap);
    Kokkos::resize(scale, newCap);
    Kokkos::resize(targetScale, newCap);
    Kokkos::resize(shapeId, newCap);
    Kokkos::resize(posPred, newCap);
    Kokkos::resize(quatPred, newCap);
    Kokkos::resize(velPred, newCap);
    Kokkos::resize(angVelPred, newCap);
    Kokkos::resize(deltaPos, newCap);
    Kokkos::resize(deltaQuat, newCap);
    Kokkos::resize(deltaVel, newCap);
    Kokkos::resize(deltaAngVel, newCap);
    Kokkos::resize(constraintCounts, newCap);
    Kokkos::resize(realIndices, newCap);
    Kokkos::resize(bodyWinner, newCap);
    Kokkos::resize(bodyColorMask, newCap);
    Kokkos::resize(groundedLevel, newCap);
    Kokkos::resize(heightLevel, newCap);
    Kokkos::resize(planeFriction, newCap);
    Kokkos::resize(rad, newCap);
    Kokkos::resize(extForce, newCap);
    capacity = newCap;
  }

  // Const views for the read-only kernel inputs.
  Kokkos::View<const float* [3], CpMem> cpos() const { return pos; }
  Kokkos::View<const float*, CpMem> crad() const { return rad; }
};

}  // namespace peclet::dem

#endif  // DEM_PARTICLES_HPP
