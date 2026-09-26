// G-B0 (docs/contact_physics_followups.md §6, WO-B0 / F1): an analytic hollow cylinder or box must
// register its CIRCUMSCRIBED radius as the broad-phase / band radius. With the geometric radius
// (the pre-WO-B0 registration) the broad-phase box of a tube was R + margin, so its end and corner
// contacts were never detected: coaxial tubes overlapping by 0.3 gave 0 contacts and did not move.
//
// The full single-rank pipeline (Simulation::step: ArborX broad phase, narrow phase, the overlap
// projection), which is why this lives in tests/arborx rather than the Kokkos-only kernel suite.
//
// Shallow scenes (penetration 0.03 < t/2 = 0.09): the pair must be detected (contacts > 0), and
// after one step (20 position iterations, velocity solve off, g = 0) the committed overlap
// (Simulation::computeOverlaps, the re-measured state) must be <= 1e-3:
//   coax     two coaxial tubes (D 1, H 1.5, wall 0.18) at centre distance 1.47
//   tjunc    B's axis perpendicular to A's; B's end cap presses 0.03 into A's outer wall
//   cubes    two cubes of half-extent R = 1: A with one body diagonal pointing at B's centre, B
//            axis-aligned, A's corner 0.03 R inside B's face
// Detection only (tunnelled states, docs/contact_physics_followups.md §3.2: no overlap bound):
//   coaxial tubes at centre distance 1.4 and 1.2 give contacts > 0 (0 before WO-B0).
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "sim.hpp"

using peclet::dem::F3;
using peclet::dem::Simulation;

namespace {

constexpr float kTubeR = 0.5f, kTubeH = 1.5f, kTubeWall = 0.18f;

struct Scene {
  std::string name;
  int shape;          // peclet::dem::ShapeKind
  float r, h, t;      // initializeShape arguments
  float x[2][3];      // centres
  float q[2][4];      // orientations (x, y, z, w)
  bool boundOverlap;  // gate the committed overlap (shallow scenes only)
};

// (x, y, z, w) quaternion rotating the unit vector u onto the unit vector v (u != -v).
void quatFromTo(const double u[3], const double v[3], float out[4]) {
  const double c[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                       u[0] * v[1] - u[1] * v[0]};
  const double w = 1.0 + u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const double n = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2] + w * w);
  out[0] = static_cast<float>(c[0] / n);
  out[1] = static_cast<float>(c[1] / n);
  out[2] = static_cast<float>(c[2] / n);
  out[3] = static_cast<float>(w / n);
}

int runScene(const Scene& s) {
  Simulation sim(64);
  sim.setDomain(16.0f, 16.0f, 16.0f, false, false, false);
  sim.setDomainMinMax(F3{-8.0f, -8.0f, -8.0f}, F3{8.0f, 8.0f, 8.0f});
  sim.setGlobalScale(1.0f);
  sim.initializeShape(s.shape, s.r, s.h, s.t);
  sim.setDt(1e-2f);
  sim.setGravity(0.0f, 0.0f, 0.0f);
  sim.setSolverIterations(20, 0);  // velocity solve off (the dem default)
  sim.setPositions({s.x[0][0], s.x[0][1], s.x[0][2], s.x[1][0], s.x[1][1], s.x[1][2]});
  sim.setQuaternions(
      {s.q[0][0], s.q[0][1], s.q[0][2], s.q[0][3], s.q[1][0], s.q[1][1], s.q[1][2], s.q[1][3]});
  sim.step(1);
  const int nc = sim.numContacts();
  const float ov = sim.computeOverlaps();
  const std::vector<float> p = sim.getPositions();
  double moved = 0.0;
  for (int i = 0; i < 2; ++i)
    for (int d = 0; d < 3; ++d)
      moved = std::fmax(moved, std::fabs(p[3 * i + d] - s.x[i][d]));
  const bool ok = nc > 0 && (!s.boundOverlap || ov <= 1e-3f);
  std::printf("[shape_detection] %-9s contacts=%d committed_overlap=%.3e max|dx|=%.3e  %s\n",
              s.name.c_str(), nc, ov, moved, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int fail = 0;
  {
    const float id[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<Scene> scenes;
    auto tubes = [&](const std::string& name, const float xa[3], const float qa[4],
                     const float xb[3], const float qb[4], bool bound) {
      Scene s{name, peclet::dem::HOLLOW_CYLINDER, kTubeR, kTubeH, kTubeWall, {}, {}, bound};
      for (int d = 0; d < 3; ++d) {
        s.x[0][d] = xa[d];
        s.x[1][d] = xb[d];
      }
      for (int d = 0; d < 4; ++d) {
        s.q[0][d] = qa[d];
        s.q[1][d] = qb[d];
      }
      scenes.push_back(s);
    };
    // Coaxial along the body axis (y): end faces overlap by H - distance.
    const float o[3] = {0.0f, 0.0f, 0.0f};
    const float c147[3] = {0.0f, 1.47f, 0.0f}, c14[3] = {0.0f, 1.4f, 0.0f},
                c12[3] = {0.0f, 1.2f, 0.0f};
    tubes("coax1.47", o, id, c147, id, true);
    // T-junction: A's axis y at the origin (outer wall at x = R); B's axis x, its end face at
    // x = R - 0.03, so the end annulus presses 0.03 into A's outer wall around z = 0.
    {
      const double ey[3] = {0, 1, 0}, ex[3] = {1, 0, 0};
      float qb[4];
      quatFromTo(ey, ex, qb);
      const float xb[3] = {kTubeR - 0.03f + 0.5f * kTubeH, 0.0f, 0.0f};
      tubes("tjunc", o, id, xb, qb, true);
    }
    // Two cubes of half-extent 1: B axis-aligned at the origin (face at x = 1); A's body diagonal
    // (1, 1, 1)/sqrt(3) rotated onto -x, its corner 0.03 inside B's face.
    {
      const double R = 1.0, diag[3] = {1 / std::sqrt(3.0), 1 / std::sqrt(3.0), 1 / std::sqrt(3.0)},
                   mx[3] = {-1, 0, 0};
      Scene s{"cubes", peclet::dem::BOX, static_cast<float>(R), 0.0f, 0.0f, {}, {}, true};
      s.x[0][0] = static_cast<float>(R - 0.03 * R + std::sqrt(3.0) * R);
      s.x[0][1] = s.x[0][2] = 0.0f;
      quatFromTo(diag, mx, s.q[0]);
      for (int d = 0; d < 3; ++d)
        s.x[1][d] = 0.0f;
      for (int d = 0; d < 4; ++d)
        s.q[1][d] = id[d];
      scenes.push_back(s);
    }
    tubes("coax1.4", o, id, c14, id, false);
    tubes("coax1.2", o, id, c12, id, false);
    for (const Scene& s : scenes)
      fail |= runScene(s);
  }
  Kokkos::finalize();
  if (fail)
    std::fprintf(stderr, "FAIL: shape detection (G-B0)\n");
  return fail;
}
