// Scheme C (the user's): compute each pair once, reverse-accumulate forces to owners, then FORWARD
// the total force to ghosts and integrate ghosts LOCALLY — so ghost state is re-derived by each
// rank rather than imported. Validated with transport-core's ParticleHaloTopology (reverse +
// forward).
//
// Two checks, np=1,2,4:
//   (1) owned trajectories match a serial all-pairs reference (forces are correct), and
//   (2) the locally-integrated ghost copies stay consistent with their owners
//       (forwardPositions(current owned) == the locally-integrated ghost positions).
// To isolate the local-integration property the halo is built ONCE with a skin and positions are
// not wrapped (periodicity is handled by the min-image force); particles drift << skin over the
// run. Scheme A is in test_dem_step.cpp, scheme B in test_dem_scheme_b.cpp.
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_halo_topology.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

using namespace tpx;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::DomainMap;
using peclet::core::halo::ParticleHaloTopology;
using peclet::core::halo::ParticleMigrator;

static constexpr int kSteps = 15;
static const double Ld[3] = {10.0, 8.0, 6.0};
static const double dmin[3] = {0, 0, 0};
static constexpr double kR = 0.4, kRcut = 2.0 * kR, kSkin = 0.5 * kRcut;
static constexpr double kStiff = 0.2, kDt = 0.02, kDamp = 0.99;

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
static double mi(double d, double L) {
  return d - L * std::round(d / L);
}
static F3 pf(const Vec<3>& a, const Vec<3>& b) {
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

  // serial reference: all-pairs, NO position wrap (min-image force).
  std::vector<Vec<3>> sp = p0, sv = v0;
  for (int step = 0; step < kSteps; ++step) {
    std::vector<F3> F(N);
    for (std::int64_t i = 0; i < N; ++i)
      for (std::int64_t j = 0; j < N; ++j)
        if (j != i)
          F[i] += pf(sp[i], sp[j]);
    for (std::int64_t i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a) {
        sv[i][a] = (sv[i][a] + F[i].v[a] * kDt) * kDamp;
        sp[i][a] += sv[i][a] * kDt;
      }
  }

  // distributed scheme C
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
    Pay p{};
    for (int a = 0; a < 3; ++a)
      p.vel[a] = v0[id][a];
    p.id = id;
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &p, stride);
  }
  mig.migrate(pos, payload, stride);  // establish ownership once
  std::size_t n = pos.size();
  std::vector<std::int64_t> id(n);
  std::vector<Vec<3>> vel(n);
  for (std::size_t i = 0; i < n; ++i) {
    Pay p;
    std::memcpy(&p, &payload[i * stride], stride);
    id[i] = p.id;
    vel[i] = {p.vel[0], p.vel[1], p.vel[2]};
  }

  // build ONCE with a skin; maintain ghost state locally thereafter
  halo.build(pos, kRcut + kSkin);
  std::size_t G = halo.numGhost();
  std::vector<Vec<3>> gpos = halo.ghostPositions();
  std::vector<Vec<3>> gvel(G);
  std::vector<double> ownIdD(n), ghIdD(G);
  for (std::size_t i = 0; i < n; ++i)
    ownIdD[i] = (double)id[i];
  halo.forward(ownIdD.data(), ghIdD.data());
  halo.forward(vel.data(), gvel.data());  // initial ghost velocities

  for (int step = 0; step < kSteps; ++step) {
    std::vector<F3> Fown(n), Fgh(G);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        F3 f = pf(pos[i], pos[j]);
        Fown[i] += f;
        Fown[j] += F3{{-f.v[0], -f.v[1], -f.v[2]}};
      }
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t g = 0; g < G; ++g) {
        if (id[i] >= (std::int64_t)std::llround(ghIdD[g]))
          continue;
        F3 f = pf(pos[i], gpos[g]);
        Fown[i] += f;
        Fgh[g] += F3{{-f.v[0], -f.v[1], -f.v[2]}};
      }
    halo.reverse(Fgh.data(), Fown.data());  // owners get total force
    std::vector<F3> FghTot(G);
    halo.forward(Fown.data(), FghTot.data());  // ghosts get THEIR total force

    for (std::size_t i = 0; i < n; ++i)
      for (int a = 0; a < 3; ++a) {
        vel[i][a] = (vel[i][a] + Fown[i].v[a] * kDt) * kDamp;
        pos[i][a] += vel[i][a] * kDt;
      }
    for (std::size_t g = 0; g < G; ++g)
      for (int a = 0; a < 3; ++a) {  // integrate ghosts LOCALLY (no state import)
        gvel[g][a] = (gvel[g][a] + FghTot[g].v[a] * kDt) * kDamp;
        gpos[g][a] += gvel[g][a] * kDt;
      }
  }

  int fail = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (int a = 0; a < 3; ++a)
      if (std::fabs(pos[i][a] - sp[id[i]][a]) > 1e-9 || std::fabs(vel[i][a] - sv[id[i]][a]) > 1e-9)
        ++fail;
  // ghost consistency: forwarding current owned positions reproduces the locally-integrated ghosts
  std::vector<Vec<3>> fresh(G);
  halo.forwardPositions(pos.data(), fresh.data());
  for (std::size_t g = 0; g < G; ++g)
    for (int a = 0; a < 3; ++a)
      if (std::fabs(fresh[g][a] - gpos[g][a]) > 1e-9)
        ++fail;

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total == 0)
      std::printf(
          "OK (np=%d): scheme C (local ghost integration via forwarded forces) matches serial"
          " AND ghosts stay consistent\n",
          size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
