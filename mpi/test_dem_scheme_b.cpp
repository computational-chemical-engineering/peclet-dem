// Scheme B (Newton-on: compute each pair once, reverse the ghost force to owners, integrate owned) validated against a serial reference, using transport-core's
// ParticleHaloTopology.reverse.  Each pair is computed exactly ONCE (owned-ghost pairs are computed only on
// the rank whose particle has the lower id), the reaction force is placed on the ghost, and
// reverse(ghostForce, ownedForce, +=) accumulates those contributions back onto the owners. The
// resulting total force on every owned particle must equal the serial all-pairs force, so the
// trajectories match (to round-off). Schemes A (replicate) and C (local ghost integration) are in test_dem_step.cpp / test_dem_scheme_c.cpp.
//
// (Scheme A — frozen ghosts, replicated pair computation — is validated in test_dem_step.cpp.)
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_halo.hpp"
#include "tpx/halo/particle_migrator.hpp"

using namespace tpx;
using tpx::decomp::BlockDecomposer;
using tpx::halo::DomainMap;
using tpx::halo::ParticleHaloTopology;
using tpx::halo::ParticleMigrator;

static constexpr int kSteps = 25;
static const double Ld[3] = {10.0, 8.0, 6.0};
static const double dmin[3] = {0, 0, 0};
static constexpr double kR = 0.4, kRcut = 2.0 * kR, kStiff = 0.2, kDt = 0.02, kDamp = 0.99;

struct F3 {
  double v[3] = {0, 0, 0};
  F3& operator+=(const F3& o) {
    v[0] += o.v[0];
    v[1] += o.v[1];
    v[2] += o.v[2];
    return *this;
  }
};
struct Pay {
  double vel[3];
  std::int64_t id;
};

static double frac(std::uint64_t x, int s) {
  x ^= (std::uint64_t)s * 2654435761u;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (double)(x & 0xFFFFFF) / (double)0x1000000;
}
static double mi(double d, double L) { return d - L * std::round(d / L); }

// repulsion on particle at a from particle at b (zero if beyond contact). min-image.
static F3 pair_force(const Vec<3>& a, const Vec<3>& b) {
  double s[3] = {mi(a[0] - b[0], Ld[0]), mi(a[1] - b[1], Ld[1]), mi(a[2] - b[2], Ld[2])};
  double d = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  F3 f;
  if (d < kRcut && d > 1e-12) {
    double m = kStiff * (kRcut - d) / d;
    f.v[0] = m * s[0];
    f.v[1] = m * s[1];
    f.v[2] = m * s[2];
  }
  return f;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const std::int64_t N = 400;

  std::vector<Vec<3>> p0(N), v0(N);
  for (std::int64_t i = 0; i < N; ++i)
    for (int a = 0; a < 3; ++a) {
      p0[i][a] = dmin[a] + frac(i, a) * Ld[a];
      v0[i][a] = (frac(i, 10 + a) - 0.5) * 0.2;
    }

  // --- serial reference (replicated, all-pairs) ---
  std::vector<Vec<3>> sp = p0, sv = v0;
  for (int step = 0; step < kSteps; ++step) {
    std::vector<F3> F(N);
    for (std::int64_t i = 0; i < N; ++i)
      for (std::int64_t j = 0; j < N; ++j)
        if (j != i) {
          F3 f = pair_force(sp[i], sp[j]);
          F[i] += f;
        }
    for (std::int64_t i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a) {
        sv[i][a] = (sv[i][a] + F[i].v[a] * kDt) * kDamp;
        sp[i][a] += sv[i][a] * kDt;
        sp[i][a] = dmin[a] + std::fmod(std::fmod(sp[i][a] - dmin[a], Ld[a]) + Ld[a], Ld[a]);
      }
  }

  // --- distributed scheme C ---
  IVec<3> gsize{40, 32, 24};
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize);
  DomainMap<3> map;
  for (int a = 0; a < 3; ++a) {
    map.origin[a] = dmin[a];
    map.cellSize[a] = Ld[a] / gsize[a];
    map.periodic[a] = true;
  }
  ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  ParticleHaloTopology<3> halo;
  halo.init(mig);

  const std::size_t stride = sizeof(Pay);
  std::vector<Vec<3>> pos;
  std::vector<char> payload;
  for (std::int64_t id = rank; id < N; id += size) {
    pos.push_back(p0[id]);
    Pay pay{};
    for (int a = 0; a < 3; ++a) pay.vel[a] = v0[id][a];
    pay.id = id;
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &pay, stride);
  }

  for (int step = 0; step < kSteps; ++step) {
    mig.migrate(pos, payload, stride);
    std::size_t n = pos.size();
    std::vector<std::int64_t> id(n);
    std::vector<Vec<3>> vel(n);
    for (std::size_t i = 0; i < n; ++i) {
      Pay p;
      std::memcpy(&p, &payload[i * stride], stride);
      id[i] = p.id;
      vel[i] = {p.vel[0], p.vel[1], p.vel[2]};
    }

    halo.build(pos, kRcut);
    std::size_t G = halo.numGhost();
    const auto& gpos = halo.ghostPositions();
    std::vector<double> ownIdD(n), ghIdD(G);
    for (std::size_t i = 0; i < n; ++i) ownIdD[i] = (double)id[i];
    halo.forward(ownIdD.data(), ghIdD.data());

    std::vector<F3> Fown(n), Fgh(G);
    // owned-owned: each unordered pair once
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        F3 f = pair_force(pos[i], pos[j]);
        Fown[i] += f;
        F3 mf{{-f.v[0], -f.v[1], -f.v[2]}};
        Fown[j] += mf;
      }
    // owned-ghost: compute once, on the rank whose id is lower; reaction goes to the ghost
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t g = 0; g < G; ++g) {
        if (id[i] >= (std::int64_t)std::llround(ghIdD[g])) continue;
        F3 f = pair_force(pos[i], gpos[g]);
        Fown[i] += f;
        F3 mf{{-f.v[0], -f.v[1], -f.v[2]}};
        Fgh[g] += mf;
      }
    // accumulate ghost reactions back onto their owners
    halo.reverse(Fgh.data(), Fown.data());

    for (std::size_t i = 0; i < n; ++i)
      for (int a = 0; a < 3; ++a) {
        vel[i][a] = (vel[i][a] + Fown[i].v[a] * kDt) * kDamp;
        pos[i][a] += vel[i][a] * kDt;
        pos[i][a] = dmin[a] + std::fmod(std::fmod(pos[i][a] - dmin[a], Ld[a]) + Ld[a], Ld[a]);
      }
    for (std::size_t i = 0; i < n; ++i) {
      Pay p;
      p.id = id[i];
      for (int a = 0; a < 3; ++a) p.vel[a] = vel[i][a];
      std::memcpy(&payload[i * stride], &p, stride);
    }
  }

  int fail = 0;
  for (std::size_t i = 0; i < pos.size(); ++i) {
    Pay p;
    std::memcpy(&p, &payload[i * stride], stride);
    for (int a = 0; a < 3; ++a)
      if (std::fabs(pos[i][a] - sp[p.id][a]) > 1e-9 || std::fabs(p.vel[a] - sv[p.id][a]) > 1e-9)
        ++fail;
  }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      std::printf("OK (np=%d): scheme B (compute-once + reverse-force, integrate owned) matches serial\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d state mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
