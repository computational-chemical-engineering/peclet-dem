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
  Vf rad;  // effective broadphase radius scratch (scale * globalScale)

  // --- collision/contact/manifold buffers ---
  Kokkos::View<int* [2], CpMem> pairs;        // broadphase candidates (maxPairs)
  Kokkos::View<ContactC*, CpMem> contacts;    // narrowphase output (maxContacts)
  Kokkos::View<ManifoldC*, CpMem> manifolds;  // reduction output (maxContacts)

  // --- atomic counters / scalars (rank-0 Views) ---
  Kokkos::View<int, CpMem> pairCount, contactCount, manifoldCount, topGhost;
  Kokkos::View<float, CpMem> maxOverlap;

  // --- static geometry ---
  Kokkos::View<ShapeDesc*, CpMem> shapes;
  Kokkos::View<float* [3], CpMem> shell;
  Kokkos::View<PlaneP*, CpMem> planes;
  Kokkos::View<float*, CpMem> sdfGrid;  // concatenated grid-SDF samples (imported shapes)

  // --- sizes & params (host) ---
  int capacity = 0, numReal = 0, numParticles = 0;
  int maxPairs = 0, maxContacts = 0, numPlanes = 0;
  Domain domain{};
  F3 gravity{0, 0, 0};
  float dt = 1e-3f, globalScale = 1.0f, growthRate = 0.0f, growthFactor = -1.0f;
  float thermostatTau = 0.0f, thermostatTemp = 0.0f,
        thermostatKB = 1.0f;  // Berendsen (tau>0 enables)
  float frictionDynamic = 0.0f, restitutionNormal = 0.0f, skin = 0.1f;
  int positionIterations = 10, velocityIterations = 0;

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
    pairs = Kokkos::View<int* [2], CpMem>("pairs", maxPairs);
    contacts = Kokkos::View<ContactC*, CpMem>("contacts", maxContacts);
    manifolds = Kokkos::View<ManifoldC*, CpMem>("manifolds", maxContacts);
    pairCount = Kokkos::View<int, CpMem>("pairCount");
    contactCount = Kokkos::View<int, CpMem>("contactCount");
    manifoldCount = Kokkos::View<int, CpMem>("manifoldCount");
    topGhost = Kokkos::View<int, CpMem>("topGhost");
    maxOverlap = Kokkos::View<float, CpMem>("maxOverlap");
    shapes = Kokkos::View<ShapeDesc*, CpMem>("shapes", nShapes > 0 ? nShapes : 1);
    shell = Kokkos::View<float* [3], CpMem>("shell", nShell > 0 ? nShell : 1);
    planes = Kokkos::View<PlaneP*, CpMem>("planes", nPlanes > 0 ? nPlanes : 1);
    sdfGrid = Kokkos::View<float*, CpMem>("sdfGrid", 1);  // resized by setSdfShape
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
    Kokkos::resize(planeFriction, newCap);
    Kokkos::resize(rad, newCap);
    capacity = newCap;
  }

  // Const views for the read-only kernel inputs.
  Kokkos::View<const float* [3], CpMem> cpos() const { return pos; }
  Kokkos::View<const float*, CpMem> crad() const { return rad; }
};

}  // namespace peclet::dem

#endif  // DEM_PARTICLES_HPP
