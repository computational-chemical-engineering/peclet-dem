// Correctness of the Kokkos narrow-phase (detectContactsKokkos / detectBoundaryKokkos) against a
// host replication of the identical per-point math.
//
// Scene: a jittered 3D grid of spheres spaced so that only face-neighbours overlap decisively (no
// borderline contacts), with random orientations, plus a ground plane. We run the device kernels
// and a host loop over the same dem:: math, then require the two contact sets to match (sorted,
// within tolerance). Runs on whatever backend Kokkos was built for.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <vector>

#include "narrowphase.hpp"

using namespace peclet::dem;

struct CC {
  int a, b;
  float n[3], rA[3], rB[3], dist;
};
static CC toCC(const ContactC& c) {
  return CC{c.bodyA,
            c.bodyB,
            {c.normal.x, c.normal.y, c.normal.z},
            {c.rA.x, c.rA.y, c.rA.z},
            {c.rB.x, c.rB.y, c.rB.z},
            c.dist};
}
static bool less_cc(const CC& x, const CC& y) {
  if (x.a != y.a)
    return x.a < y.a;
  return x.b < y.b;
}
static bool close(float a, float b, float tol) {
  return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}
// dist is computed before the gradient, so it matches tightly. The normal comes from a central
// difference of the SDF (catastrophic cancellation of two sqrt's), so host vs device differ by
// ~1e-3 in direction — expected, not a bug; rA/rB inherit it. Use a looser tol for those.
static bool eq_cc(const CC& x, const CC& y) {
  if (x.a != y.a || x.b != y.b || !close(x.dist, y.dist, 1e-4f))
    return false;
  for (int d = 0; d < 3; ++d)
    if (!close(x.n[d], y.n[d], 5e-3f) || !close(x.rA[d], y.rA[d], 5e-3f) ||
        !close(x.rB[d], y.rB[d], 5e-3f))
      return false;
  return true;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int G = 6;             // 6x6x6 grid
    const int N = G * G * G;     // 216 spheres
    const float globalScale = 0.7f;  // non-unit: exercises the canonical->world globalScale folding
    // all lengths scale with globalScale so the config is the unit-scale one in world units: spacing
    // < world diameter (2*globalScale) => face neighbours overlap; lift = one world radius onto z=0.
    const float spacing = 1.8f * globalScale;
    const float margin = 0.1f * globalScale;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> jit(-0.02f, 0.02f);
    std::normal_distribution<float> nq(0.f, 1.f);

    std::vector<float> px(N), py(N), pz(N), qx(N), qy(N), qz(N), qw(N), sc(N);
    std::vector<int> sid(N);
    for (int i = 0; i < N; ++i) {
      int ix = i % G, iy = (i / G) % G, iz = i / (G * G);
      px[i] = ix * spacing + jit(rng);
      py[i] = iy * spacing + jit(rng);
      pz[i] = iz * spacing + globalScale + jit(rng);  // lift one world radius onto the z=0 plane
      float a = nq(rng), b = nq(rng), c = nq(rng), d = nq(rng);
      float nrm = std::sqrt(a * a + b * b + c * c + d * d) + 1e-12f;
      qx[i] = a / nrm;
      qy[i] = b / nrm;
      qz[i] = c / nrm;
      qw[i] = d / nrm;
      sc[i] = 1.0f;
      sid[i] = i & 1;  // two sphere shapes (radius 1.0 / 1.02)
    }

    std::vector<ShapeDesc> shapes = {ShapeDesc{SPHERE, F4{1.00f, 0, 0, 0}, 0, 0},
                                     ShapeDesc{SPHERE, F4{1.02f, 0, 0, 0}, 0, 0}};
    std::vector<PlaneP> planes = {PlaneP{F3{0, 0, 0}, F3{0, 0, 1}}};

    // All i<j pairs (small N).
    std::vector<int> pa, pb;
    for (int i = 0; i < N; ++i)
      for (int j = i + 1; j < N; ++j) {
        pa.push_back(i);
        pb.push_back(j);
      }
    const int numPairs = static_cast<int>(pa.size());

    // --- upload SoA ---
    Kokkos::View<float* [3], CpMem> pos("pos", N);
    Kokkos::View<float* [4], CpMem> quat("quat", N);
    Kokkos::View<float*, CpMem> scale("scale", N);
    Kokkos::View<int*, CpMem> shapeId("sid", N);
    {
      auto hp = Kokkos::create_mirror_view(pos);
      auto hq = Kokkos::create_mirror_view(quat);
      auto hs = Kokkos::create_mirror_view(scale);
      auto hi = Kokkos::create_mirror_view(shapeId);
      for (int i = 0; i < N; ++i) {
        hp(i, 0) = px[i];
        hp(i, 1) = py[i];
        hp(i, 2) = pz[i];
        hq(i, 0) = qx[i];
        hq(i, 1) = qy[i];
        hq(i, 2) = qz[i];
        hq(i, 3) = qw[i];
        hs(i) = sc[i];
        hi(i) = sid[i];
      }
      Kokkos::deep_copy(pos, hp);
      Kokkos::deep_copy(quat, hq);
      Kokkos::deep_copy(scale, hs);
      Kokkos::deep_copy(shapeId, hi);
    }
    Kokkos::View<ShapeDesc*, CpMem> dShapes("shapes", shapes.size());
    {
      auto h = Kokkos::create_mirror_view(dShapes);
      for (size_t i = 0; i < shapes.size(); ++i)
        h(i) = shapes[i];
      Kokkos::deep_copy(dShapes, h);
    }
    Kokkos::View<PlaneP*, CpMem> dPlanes("planes", planes.size());
    {
      auto h = Kokkos::create_mirror_view(dPlanes);
      for (size_t i = 0; i < planes.size(); ++i)
        h(i) = planes[i];
      Kokkos::deep_copy(dPlanes, h);
    }
    Kokkos::View<float* [3], CpMem> shell("shell", 1);  // unused (spheres)
    Kokkos::View<int* [2], CpMem> pairs("pairs", numPairs);
    {
      auto h = Kokkos::create_mirror_view(pairs);
      for (int i = 0; i < numPairs; ++i) {
        h(i, 0) = pa[i];
        h(i, 1) = pb[i];
      }
      Kokkos::deep_copy(pairs, h);
    }

    const int maxContacts = N * 32;
    Kokkos::View<ContactC*, CpMem> dContacts("contacts", maxContacts);
    Kokkos::View<int, CpMem> dCount("count");
    Kokkos::View<float, CpMem> dMaxOv("maxov");

    detectContactsKokkos(pairs, numPairs, pos, quat, scale, shapeId, dShapes, shell, globalScale,
                         margin, dContacts, dCount, dMaxOv);
    detectBoundaryKokkos(N, static_cast<int>(planes.size()), pos, quat, scale, shapeId, dShapes,
                         shell, dPlanes, globalScale, margin, dContacts, dCount, dMaxOv);

    int nc = 0;
    Kokkos::deep_copy(nc, dCount);
    std::vector<CC> got;
    {
      auto h = Kokkos::create_mirror_view(dContacts);
      Kokkos::deep_copy(h, dContacts);
      for (int i = 0; i < nc && i < maxContacts; ++i)
        got.push_back(toCC(h(i)));
    }

    // --- host reference: identical math ---
    std::vector<CC> ref;
    auto loadF3h = [&](int i) { return F3{px[i], py[i], pz[i]}; };
    auto loadF4h = [&](int i) { return F4{qx[i], qy[i], qz[i], qw[i]}; };
    for (int idx = 0; idx < numPairs; ++idx) {
      int idA = pa[idx], idB = pb[idx];
      ShapeDesc dA = shapes[sid[idA]], dB = shapes[sid[idB]];
      F3 posA = loadF3h(idA), posB = loadF3h(idB);
      F4 qA = loadF4h(idA), qB = loadF4h(idB);
      float effScaleA = sc[idA] * globalScale, effScaleB = sc[idB] * globalScale;
      int iter = (dA.numPoints > 0) ? dA.numPoints : 1;
      for (int k = 0; k < iter; ++k) {
        F3 pLocalA{0, 0, 0};
        float pr = 0.0f;
        if (dA.numPoints > 0) {
          /* shell unused here */
        } else if (dA.type == SPHERE) {
          pr = dA.params.x * effScaleA;
        }
        F3 pWorld = add3(posA, rotateVector(qA, scale3(pLocalA, effScaleA)));
        F3 pCanB = scale3(invRotateVector(qB, sub3(pWorld, posB)), 1.0f / effScaleB);
        float dist = sdfEval(pCanB, dB.type, dB.params) * effScaleB;
        float effDist = dist - pr;
        if (effDist >= margin)
          continue;
        float eps = 1e-4f;
        F3 nLoc{sdfEval(F3{pCanB.x + eps, pCanB.y, pCanB.z}, dB.type, dB.params) -
                    sdfEval(F3{pCanB.x - eps, pCanB.y, pCanB.z}, dB.type, dB.params),
                sdfEval(F3{pCanB.x, pCanB.y + eps, pCanB.z}, dB.type, dB.params) -
                    sdfEval(F3{pCanB.x, pCanB.y - eps, pCanB.z}, dB.type, dB.params),
                sdfEval(F3{pCanB.x, pCanB.y, pCanB.z + eps}, dB.type, dB.params) -
                    sdfEval(F3{pCanB.x, pCanB.y, pCanB.z - eps}, dB.type, dB.params)};
        float len = len3(nLoc);
        nLoc = (len > 1e-9f) ? scale3(nLoc, 1.0f / len) : F3{0, 1, 0};
        F3 nWorld = rotateVector(qB, nLoc);
        F3 pSurfA = sub3(pWorld, scale3(nWorld, pr));
        F3 rA = sub3(pSurfA, posA);
        F3 rB = sub3(sub3(pSurfA, scale3(nWorld, effDist)), posB);
        ref.push_back(CC{idA,
                         idB,
                         {nWorld.x, nWorld.y, nWorld.z},
                         {rA.x, rA.y, rA.z},
                         {rB.x, rB.y, rB.z},
                         effDist});
      }
    }
    for (int i = 0; i < N; ++i) {
      ShapeDesc d = shapes[sid[i]];
      F3 posA = loadF3h(i);
      float s = sc[i] * globalScale;
      float baseR = d.params.x;
      if (baseR == 0.0f)
        baseR = 1.0f;
      float radius = baseR * s;
      for (const auto& pl : planes) {
        float dist = dot3(sub3(posA, pl.point), pl.normal) - radius;  // sphere branch
        if (dist >= margin)
          continue;
        F3 rA = scale3(pl.normal, -radius);
        ref.push_back(CC{i,
                         -1,
                         {pl.normal.x, pl.normal.y, pl.normal.z},
                         {rA.x, rA.y, rA.z},
                         {pl.point.x, pl.point.y, pl.point.z},
                         dist});
      }
    }

    std::sort(got.begin(), got.end(), less_cc);
    std::sort(ref.begin(), ref.end(), less_cc);

    if (got.size() != ref.size()) {
      std::fprintf(stderr, "FAIL: contact count device %zu != host %zu\n", got.size(), ref.size());
      status = 1;
    } else {
      int bad = 0;
      for (size_t i = 0; i < got.size(); ++i)
        if (!eq_cc(got[i], ref[i]))
          ++bad;
      if (bad) {
        std::fprintf(stderr, "FAIL: %d/%zu contacts differ\n", bad, got.size());
        status = 1;
      } else {
        std::printf("[narrowphase] PASS: %zu contacts match host reference (exec: %s)\n",
                    got.size(), CpExec::name());
      }
    }
  }
  Kokkos::finalize();
  return status;
}
