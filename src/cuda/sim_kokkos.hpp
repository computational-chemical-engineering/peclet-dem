// packing-gpu — portable (Kokkos) Simulation facade: the demgpu flip's host-facing driver.
//
// Owns a dem::Particles SoA and runs the full XPBD DEM step by composing the ported kernels in the
// simulation.cpp step() order. Exposes a small std::vector-based API (binding-agnostic) so a pybind
// module can drive it from Python (set/get arrays, step). Sphere shapes + analytic planes for now.
#ifndef DEM_SIM_KOKKOS_HPP
#define DEM_SIM_KOKKOS_HPP

#include <Kokkos_Core.hpp>

#include <vector>

#include "broadphase_arborx.hpp"
#include "contact_preprocessing_kokkos.hpp"
#include "integration_kokkos.hpp"
#include "narrowphase_kokkos.hpp"
#include "particles_kokkos.hpp"
#include "periodicity_kokkos.hpp"
#include "solver_friction_kokkos.hpp"
#include "solver_position_kokkos.hpp"
#include "solver_velocity_kokkos.hpp"

namespace dem {

inline int readInt(Kokkos::View<int, CpMem> v) { int h; Kokkos::deep_copy(h, v); return h; }

/// One full XPBD DEM substep over the particle SoA (mirrors simulation.cpp Simulation::step()).
inline void demStep(Particles& P) {
  CpExec space;
  const float margin = 0.1f * P.globalScale;

  if (P.growthFactor > 0.0f) updateGrowthScalesKokkos(P.numReal, P.scale, P.scale, P.growthFactor);

  predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                        P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                        P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt);

  { auto ri = P.realIndices;
    Kokkos::parallel_for("self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal),
                         KOKKOS_LAMBDA(int i) { ri(i) = i; }); }
  Kokkos::deep_copy(space, P.topGhost, P.numReal);
  generateGhostsKokkos(P.numReal, P.capacity, P.domain, P.skin, P.pos, P.invMass, P.posPred, P.vel,
                       P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale, P.shapeId,
                       P.realIndices, P.topGhost);
  P.numParticles = readInt(P.topGhost);

  { auto sc = P.scale; auto rad = P.rad; float gs = P.globalScale;
    Kokkos::parallel_for("rad", Kokkos::RangePolicy<CpExec>(space, 0, P.numParticles),
                         KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs; }); }
  findCollisionsArborX(P.cpos(), P.crad(), P.numParticles, P.numReal, margin, P.pairs, P.pairCount);
  const int np = readInt(P.pairCount);

  Kokkos::deep_copy(space, P.contactCount, 0);
  Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
  detectContactsKokkos(P.pairs, np, P.pos, P.quat, P.scale, P.shapeId, P.shapes, P.shell,
                       P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  detectBoundaryKokkos(P.numReal, P.numPlanes, P.pos, P.quat, P.scale, P.shapeId, P.shapes, P.shell,
                       P.planes, P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
  const int nc = readInt(P.contactCount);

  reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount);
  const int nm = readInt(P.manifoldCount);

  const bool friction = (P.frictionDynamic > 0.0f);
  if (friction)
    computePlaneLoadKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred, P.planeFriction);

  for (int it = 0; it < P.velocityIterations; ++it) {
    if (friction)
      accumulateNormalImpulseKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                                    P.realIndices, P.growthRate);
    solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                        P.realIndices, P.growthRate, P.restitutionNormal, P.deltaVel, P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
  }
  if (friction) {
    countFrictionContactsKokkos(P.contacts, nc, P.realIndices, P.planeFriction);
    solveContactFrictionKokkos(P.contacts, nc, P.invMass, P.invInertia, P.velPred, P.angVelPred,
                               P.realIndices, P.planeFriction, P.frictionDynamic, P.deltaVel, P.deltaAngVel);
    applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);
  }

  applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat, P.velPred,
                                        P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);

  for (int it = 0; it < P.positionIterations; ++it) {
    solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                        P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
    applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel, P.constraintCounts);
  }

  finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);
}

/// Host-facing facade with std::vector setters/getters (binding-agnostic).
class KokkosSim {
 public:
  explicit KokkosSim(int capacity) {
    P_.allocate(capacity, capacity * 64, capacity * 16, /*shapes*/ 1, /*shell*/ 1, /*planes*/ 8);
    // default sphere shape (radius 1) + identity-ish defaults
    setSphereShape(1.0f);
  }

  void setSphereShape(float radius) {
    auto h = Kokkos::create_mirror_view(P_.shapes);
    h(0) = ShapeDesc{SPHERE, F4{radius, 0, 0, 0}, 0, 0};
    Kokkos::deep_copy(P_.shapes, h);
  }
  void setDomain(float lx, float ly, float lz, bool px, bool py, bool pz) {
    P_.domain = Domain{F3{0, 0, 0}, F3{lx, ly, lz}, F3{lx, ly, lz}, px, py, pz};
    P_.skin = 0.1f * P_.globalScale;
  }
  void setGravity(float gx, float gy, float gz) { P_.gravity = F3{gx, gy, gz}; }
  void setSolverIterations(int pos, int vel) { P_.positionIterations = pos; P_.velocityIterations = vel; }
  void setGlobalScale(float s) { P_.globalScale = s; P_.skin = 0.1f * s; }
  void setDt(float dt) { P_.dt = dt; }
  void setMaterialParams(float restitution, float friction) {
    P_.restitutionNormal = restitution; P_.frictionDynamic = friction;
  }
  void addPlane(float px, float py, float pz, float nx, float ny, float nz) {
    auto h = Kokkos::create_mirror_view(P_.planes);
    Kokkos::deep_copy(h, P_.planes);
    if (P_.numPlanes < static_cast<int>(P_.planes.extent(0)))
      h(P_.numPlanes++) = PlaneP{F3{px, py, pz}, F3{nx, ny, nz}};
    Kokkos::deep_copy(P_.planes, h);
  }

  // positions: flat [n*3]; (re)sets the real-particle count and default state.
  void setPositions(const std::vector<float>& xyz) {
    const int n = static_cast<int>(xyz.size() / 3);
    P_.numReal = n; P_.numParticles = n;
    auto pos = Kokkos::create_mirror_view(P_.pos);
    auto q = Kokkos::create_mirror_view(P_.quat);
    auto im = Kokkos::create_mirror_view(P_.invMass);
    auto sc = Kokkos::create_mirror_view(P_.scale);
    auto ii = Kokkos::create_mirror_view(P_.invInertia);
    auto sid = Kokkos::create_mirror_view(P_.shapeId);
    auto vel = Kokkos::create_mirror_view(P_.vel);
    auto av = Kokkos::create_mirror_view(P_.angVel);
    for (int i = 0; i < n; ++i) {
      pos(i, 0) = xyz[3*i]; pos(i, 1) = xyz[3*i+1]; pos(i, 2) = xyz[3*i+2];
      q(i, 0) = 0; q(i, 1) = 0; q(i, 2) = 0; q(i, 3) = 1;
      im(i) = 1.0f; sc(i) = 1.0f; sid(i) = 0;
      ii(i, 0) = ii(i, 1) = ii(i, 2) = 2.5f;
      vel(i, 0) = vel(i, 1) = vel(i, 2) = 0; av(i, 0) = av(i, 1) = av(i, 2) = 0;
    }
    Kokkos::deep_copy(P_.pos, pos); Kokkos::deep_copy(P_.quat, q); Kokkos::deep_copy(P_.invMass, im);
    Kokkos::deep_copy(P_.scale, sc); Kokkos::deep_copy(P_.invInertia, ii); Kokkos::deep_copy(P_.shapeId, sid);
    Kokkos::deep_copy(P_.vel, vel); Kokkos::deep_copy(P_.angVel, av);
  }
  void setScalesUniform(float s) {
    auto sc = Kokkos::create_mirror_view(P_.scale);
    for (int i = 0; i < P_.numReal; ++i) sc(i) = s;
    Kokkos::deep_copy(P_.scale, sc);
  }

  std::vector<float> getPositions() const {
    auto pos = Kokkos::create_mirror_view(P_.pos);
    Kokkos::deep_copy(pos, P_.pos);
    std::vector<float> out(static_cast<size_t>(P_.numReal) * 3);
    for (int i = 0; i < P_.numReal; ++i) { out[3*i]=pos(i,0); out[3*i+1]=pos(i,1); out[3*i+2]=pos(i,2); }
    return out;
  }

  void step(int nsteps) { for (int s = 0; s < nsteps; ++s) demStep(P_); }

  int numParticles() const { return P_.numReal; }
  int numContacts() { return readInt(P_.contactCount); }
  float maxOverlap() { float h; Kokkos::deep_copy(h, P_.maxOverlap); return h; }

 private:
  Particles P_;
};

}  // namespace dem

#endif  // DEM_SIM_KOKKOS_HPP
