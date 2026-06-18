// End-to-end smoke test of the assembled Kokkos+ArborX DEM pipeline (the demgpu flip target).
//
// Composes EVERY ported unit on the dem::Particles SoA container in the real simulation.cpp step()
// order — predict -> ghosts -> broad-phase(ArborX) -> narrow-phase -> contact->manifold -> velocity
// solve -> re-integrate -> position solve -> final commit — and runs several steps on a small
// periodic sphere packing under gravity with a ground plane. This proves the ported headers compose
// (no cross-header conflicts) and the pipeline runs end-to-end on the active backend; it asserts the
// state stays finite and bounded (gross-error guard). Bit-exact parity with the CUDA solver is the
// job of verify_packing_hollow_cylinders.py once wired into demgpu.so.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

#include "broadphase_arborx.hpp"
#include "contact_preprocessing_kokkos.hpp"
#include "integration_kokkos.hpp"
#include "narrowphase_kokkos.hpp"
#include "particles_kokkos.hpp"
#include "periodicity_kokkos.hpp"
#include "solver_position_kokkos.hpp"
#include "solver_velocity_kokkos.hpp"

using namespace dem;

static int readi(Kokkos::View<int, CpMem> v) { int h; Kokkos::deep_copy(h, v); return h; }

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int G = 4, numReal = G * G * G;  // 64 spheres
    const int capacity = numReal * 8;
    const float L = 8.0f, spacing = 1.9f, gscale = 1.0f, margin = 0.1f * gscale;
    CpExec space;

    Particles P;
    P.allocate(capacity, numReal * 64, capacity * 16, 1, 1, 1);
    P.numReal = numReal; P.numParticles = numReal; P.globalScale = gscale;
    P.dt = 0.005f; P.gravity = F3{0, 0, -9.8f}; P.skin = margin;
    P.domain = Domain{F3{0, 0, 0}, F3{L, L, L}, F3{L, L, L}, true, true, false};  // z non-periodic

    // Init: jittered grid of unit spheres, identity orientation, at rest.
    {
      auto pos = Kokkos::create_mirror_view(P.pos);
      auto q = Kokkos::create_mirror_view(P.quat);
      auto im = Kokkos::create_mirror_view(P.invMass);
      auto sc = Kokkos::create_mirror_view(P.scale);
      auto ii = Kokkos::create_mirror_view(P.invInertia);
      auto sid = Kokkos::create_mirror_view(P.shapeId);
      auto vel = Kokkos::create_mirror_view(P.vel);
      auto av = Kokkos::create_mirror_view(P.angVel);
      for (int i = 0; i < numReal; ++i) {
        int ix = i % G, iy = (i / G) % G, iz = i / (G * G);
        pos(i, 0) = 0.5f + ix * spacing; pos(i, 1) = 0.5f + iy * spacing; pos(i, 2) = 1.0f + iz * spacing;
        q(i, 0) = 0; q(i, 1) = 0; q(i, 2) = 0; q(i, 3) = 1;
        im(i) = 1.0f; sc(i) = 1.0f; sid(i) = 0; ii(i, 0) = ii(i, 1) = ii(i, 2) = 2.5f;
        vel(i, 0) = vel(i, 1) = vel(i, 2) = 0; av(i, 0) = av(i, 1) = av(i, 2) = 0;
      }
      Kokkos::deep_copy(P.pos, pos); Kokkos::deep_copy(P.quat, q); Kokkos::deep_copy(P.invMass, im);
      Kokkos::deep_copy(P.scale, sc); Kokkos::deep_copy(P.invInertia, ii); Kokkos::deep_copy(P.shapeId, sid);
      Kokkos::deep_copy(P.vel, vel); Kokkos::deep_copy(P.angVel, av);
    }
    { auto h = Kokkos::create_mirror_view(P.shapes); h(0) = ShapeDesc{SPHERE, F4{1, 0, 0, 0}, 0, 0}; Kokkos::deep_copy(P.shapes, h); }
    { auto h = Kokkos::create_mirror_view(P.planes); h(0) = PlaneP{F3{0, 0, 0}, F3{0, 0, 1}}; Kokkos::deep_copy(P.planes, h); }

    auto fillRad = [&](int n) {
      auto pos = P.pos; auto sc = P.scale; auto rad = P.rad; float gs = P.globalScale;
      Kokkos::parallel_for("rad", Kokkos::RangePolicy<CpExec>(space, 0, n), KOKKOS_LAMBDA(int i) { rad(i) = sc(i) * gs; });
    };

    const int nsteps = 5;
    for (int s = 0; s < nsteps; ++s) {
      // 1. predict velocity (gravity + gyroscopic), speculative position, clear deltas.
      predictVelocityKokkos(P.numReal, P.pos, P.invMass, P.vel, P.quat, P.angVel, P.invInertia,
                            P.posPred, P.quatPred, P.velPred, P.angVelPred, P.deltaPos, P.deltaQuat,
                            P.deltaVel, P.deltaAngVel, P.constraintCounts, P.gravity, P.dt);
      // 2. periodic ghosts from predicted state.
      Kokkos::deep_copy(space, P.topGhost, P.numReal);
      generateGhostsKokkos(P.numReal, P.capacity, P.domain, P.skin, P.pos, P.invMass, P.posPred,
                           P.vel, P.velPred, P.quat, P.quatPred, P.angVel, P.angVelPred, P.scale,
                           P.shapeId, P.realIndices, P.topGhost);
      P.numParticles = readi(P.topGhost);
      // also map real particles to themselves (so velocity/friction real-index lookups are valid).
      { auto ri = P.realIndices; Kokkos::parallel_for("self", Kokkos::RangePolicy<CpExec>(space, 0, P.numReal), KOKKOS_LAMBDA(int i){ ri(i)=i; }); }

      // 3. broad-phase (ArborX) over all particles, queried from real ones.
      fillRad(P.numParticles);
      findCollisionsArborX(P.cpos(), P.crad(), P.numParticles, P.numReal, margin, P.pairs, P.pairCount);
      const int np = readi(P.pairCount);

      // 4. narrow-phase: pair contacts + boundary plane contacts.
      Kokkos::deep_copy(space, P.contactCount, 0);
      Kokkos::deep_copy(space, P.maxOverlap, 0.0f);
      detectContactsKokkos(P.pairs, np, P.pos, P.quat, P.scale, P.shapeId, P.shapes, P.shell,
                           P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
      detectBoundaryKokkos(P.numReal, P.numPlanes, P.pos, P.quat, P.scale, P.shapeId, P.shapes,
                           P.shell, P.planes, P.globalScale, margin, P.contacts, P.contactCount, P.maxOverlap);
      const int nc = readi(P.contactCount);

      // 5. reduce to manifolds.
      reduceContactsToManifoldsKokkos(P.contacts, nc, P.manifolds, P.manifoldCount);
      const int nm = readi(P.manifoldCount);

      // 6. velocity solve (one iteration) + apply.
      solveVelocityKokkos(P.manifolds, nm, P.invMass, P.invInertia, P.quat, P.velPred, P.angVelPred,
                          P.realIndices, P.growthRate, P.restitutionNormal, P.deltaVel, P.deltaAngVel);
      applyVelocityDeltasKokkos(P.numParticles, P.velPred, P.angVelPred, P.deltaVel, P.deltaAngVel);

      // 7. re-integrate (persist v, predict x, integrate q).
      applyVelocityAndPredictPositionKokkos(P.numParticles, P.pos, P.invMass, P.vel, P.quat,
                                            P.velPred, P.angVelPred, P.posPred, P.quatPred, P.angVel, P.dt);

      // 8. position solve (one iteration): project + Jacobi count-average.
      solvePositionKokkos(P.contacts, nc, P.invMass, P.posPred, P.quatPred, P.quat, P.invInertia,
                          P.deltaPos, P.deltaQuat, P.constraintCounts, P.maxOverlap);
      applyUpdatesKokkos(P.numParticles, P.posPred, P.velPred, P.deltaPos, P.deltaVel, P.constraintCounts);

      // 9. final commit + periodic wrap.
      finalCommitKokkos(P.numReal, P.pos, P.invMass, P.posPred, P.quat, P.quatPred, P.domain);
    }

    // --- finiteness / boundedness guard ---
    auto pos = Kokkos::create_mirror_view(P.pos);
    Kokkos::deep_copy(pos, P.pos);
    int bad = 0;
    for (int i = 0; i < numReal; ++i)
      for (int d = 0; d < 3; ++d) {
        float v = pos(i, d);
        if (!std::isfinite(v) || std::fabs(v) > 1e3f) ++bad;
      }
    if (bad) { std::fprintf(stderr, "FAIL: %d non-finite/diverged coords after %d steps\n", bad, nsteps); status = 1; }
    else std::printf("[pipeline] PASS: %d steps of %d spheres ran end-to-end, state finite (exec: %s)\n",
                     nsteps, numReal, CpExec::name());
  }
  Kokkos::finalize();
  return status;
}
