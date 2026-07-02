// Correctness of the ArborX broad-phase (dem::findCollisionsArborX) against a brute-force O(N^2)
// AABB-overlap oracle — the unambiguous ground truth the cuBQL broad-phase also satisfies.
//
// Generate random particles with varied radii (and a block of "ghost" particles that issue no
// queries but can be hit), run the ArborX broad-phase, and require the emitted pair set {(i<j)} to
// equal the set of all i<j (i real) whose padded AABBs overlap. Run on whatever backend Kokkos was
// built for (CUDA locally; OpenMP for CI).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "broadphase_arborx.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int numParticles = 4000;
    const int numReal = 3600;  // last 400 are ghosts (queried against, but issue no query)
    const float margin = 0.05f;
    const float boxL = 20.0f;

    // Host particle data: positions in [0,boxL)^3, radii in [0.3, 0.7].
    std::vector<float> hx(numParticles), hy(numParticles), hz(numParticles), hr(numParticles);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> upos(0.f, boxL);
    std::uniform_real_distribution<float> urad(0.3f, 0.7f);
    for (int i = 0; i < numParticles; ++i) {
      hx[i] = upos(rng);
      hy[i] = upos(rng);
      hz[i] = upos(rng);
      hr[i] = urad(rng);
    }

    // Upload to device Views.
    Kokkos::View<float* [3], dem::BpMem> pos("pos", numParticles);
    Kokkos::View<float*, dem::BpMem> rad("rad", numParticles);
    {
      auto hpos = Kokkos::create_mirror_view(pos);
      auto hrad = Kokkos::create_mirror_view(rad);
      for (int i = 0; i < numParticles; ++i) {
        hpos(i, 0) = hx[i];
        hpos(i, 1) = hy[i];
        hpos(i, 2) = hz[i];
        hrad(i) = hr[i];
      }
      Kokkos::deep_copy(pos, hpos);
      Kokkos::deep_copy(rad, hrad);
    }

    const int maxPairs = numParticles * 64;
    Kokkos::View<int* [2], dem::BpMem> outPairs("pairs", maxPairs);
    Kokkos::View<int, dem::BpMem> outCount("count");

    const int n =
        dem::findCollisionsArborX(pos, rad, numParticles, numReal, margin, outPairs, outCount);

    if (n > maxPairs) {
      std::fprintf(stderr, "FAIL: pair buffer overflow (%d > %d)\n", n, maxPairs);
      status = 1;
    }

    // Collect ArborX pairs into a set.
    std::set<std::pair<int, int>> got;
    {
      auto hp = Kokkos::create_mirror_view(outPairs);
      Kokkos::deep_copy(hp, outPairs);
      for (int k = 0; k < n && k < maxPairs; ++k)
        got.insert({hp(k, 0), hp(k, 1)});
    }

    // Brute-force oracle: i real, i<j, AABBs overlap on every axis (half = r+margin).
    std::set<std::pair<int, int>> expect;
    for (int i = 0; i < numReal; ++i) {
      const float bi = hr[i] + margin;
      for (int j = i + 1; j < numParticles; ++j) {
        const float bj = hr[j] + margin;
        const float s = bi + bj;
        if (std::fabs(hx[i] - hx[j]) <= s && std::fabs(hy[i] - hy[j]) <= s &&
            std::fabs(hz[i] - hz[j]) <= s) {
          expect.insert({i, j});
        }
      }
    }

    if (got != expect) {
      std::size_t missing = 0, extra = 0;
      for (const auto& p : expect)
        if (!got.count(p))
          ++missing;
      for (const auto& p : got)
        if (!expect.count(p))
          ++extra;
      std::fprintf(stderr,
                   "FAIL: pair sets differ — got %zu, expected %zu (missing %zu, extra %zu)\n",
                   got.size(), expect.size(), missing, extra);
      status = 1;
    } else {
      std::printf("[arborx_broadphase] PASS: %zu candidate pairs match brute force (exec: %s)\n",
                  expect.size(), dem::BpExec::name());
    }
  }
  Kokkos::finalize();
  return status;
}
