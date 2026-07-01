// Track B, step 1: the distributed per-step Lagrangian loop, validated against a serial reference.
//
// This is the exact loop the real Simulation::step integration will follow, exercised with a simple
// soft-sphere repulsion force in place of packing's XPBD/cuBQL (so it builds with just MPI +
// transport-core). Per step, each rank:
//     migrate (reassign ownership) -> gatherGhosts(rcut) -> forces on OWNED from owned+ghost
//     within the interaction radius -> integrate owned.
// Every rank also runs the full N-body system serially (replicated, deterministic) as the reference;
// the distributed owned-particle states must match it for every particle, np=1,2,4.
//
// Bit-exactness across decompositions: forces are summed in a canonical order (sorted by neighbour
// id) and separations use the minimum image, so the same neighbour set yields identical arithmetic
// regardless of which rank owns which particle.
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"

using namespace tpx;
using peclet::core::decomp::BlockDecomposer;
using peclet::core::halo::DomainMap;
using peclet::core::halo::ParticleMigrator;

// Simulation constants.
static constexpr int kSteps = 25;
static const double L[3] = {10.0, 8.0, 6.0};      // periodic box size
static const double dmin[3] = {0.0, 0.0, 0.0};
static constexpr double kRadius = 0.4;            // particle radius (contact at 2r)
static constexpr double kRcut = 2.0 * kRadius;    // interaction radius
static constexpr double kStiff = 0.2;             // contact stiffness
static constexpr double kDt = 0.02;
static constexpr double kDamp = 0.99;             // velocity damping (keeps it bounded)

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
static double minImage(double d, double Li) { return d - Li * std::round(d / Li); }

// Soft-sphere repulsion on particle at xi from neighbours (id, position), summed in id order.
static std::array<double, 3> force(const Vec<3>& xi, std::int64_t selfId,
                                   std::vector<std::pair<std::int64_t, Vec<3>>>& nbr) {
  std::sort(nbr.begin(), nbr.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::array<double, 3> F{0, 0, 0};
  for (auto& [jid, xj] : nbr) {
    if (jid == selfId) continue;
    double s[3] = {minImage(xi[0] - xj[0], L[0]), minImage(xi[1] - xj[1], L[1]),
                   minImage(xi[2] - xj[2], L[2])};
    double d = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    if (d >= kRcut || d < 1e-12) continue;
    double mag = kStiff * (kRcut - d) / d;  // push apart
    F[0] += mag * s[0];
    F[1] += mag * s[1];
    F[2] += mag * s[2];
  }
  return F;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::int64_t N = 400;

  // Deterministic initial state (id 0..N-1).
  std::vector<Vec<3>> p0(N);
  std::vector<Vec<3>> v0(N);
  for (std::int64_t i = 0; i < N; ++i) {
    for (int a = 0; a < 3; ++a) {
      p0[i][a] = dmin[a] + frac(i, a) * L[a];
      v0[i][a] = (frac(i, 10 + a) - 0.5) * 0.2;
    }
  }

  // --- Serial reference (replicated on every rank, deterministic) ---
  std::vector<Vec<3>> sp = p0, sv = v0;
  for (int step = 0; step < kSteps; ++step) {
    std::vector<Vec<3>> F(N);
    for (std::int64_t i = 0; i < N; ++i) {
      std::vector<std::pair<std::int64_t, Vec<3>>> nbr;
      nbr.reserve(N);
      for (std::int64_t j = 0; j < N; ++j) nbr.push_back({j, sp[j]});
      F[i] = force(sp[i], i, nbr);
    }
    for (std::int64_t i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a) {
        sv[i][a] = (sv[i][a] + F[i][a] * kDt) * kDamp;
        sp[i][a] += sv[i][a] * kDt;
        sp[i][a] = dmin[a] + std::fmod(std::fmod(sp[i][a] - dmin[a], L[a]) + L[a], L[a]);
      }
  }

  // --- Distributed ---
  IVec<3> gsize{40, 32, 24};
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize);
  DomainMap<3> map;
  for (int a = 0; a < 3; ++a) {
    map.origin[a] = dmin[a];
    map.cellSize[a] = L[a] / gsize[a];
    map.periodic[a] = true;
  }
  ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);

  const std::size_t stride = sizeof(Pay);
  std::vector<Vec<3>> pos;
  std::vector<char> payload;
  for (std::int64_t id = rank; id < N; id += size) {  // scatter by id%size; migrate fixes ownership
    pos.push_back(p0[id]);
    Pay pay{};
    pay.vel[0] = v0[id][0]; pay.vel[1] = v0[id][1]; pay.vel[2] = v0[id][2];
    pay.id = id;
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &pay, stride);
  }

  for (int step = 0; step < kSteps; ++step) {
    mig.migrate(pos, payload, stride);
    std::vector<Vec<3>> gpos;
    std::vector<char> gpay;
    mig.gatherGhosts(pos, payload, stride, kRcut, gpos, gpay);

    std::size_t n = pos.size();
    std::vector<Vec<3>> F(n);
    for (std::size_t i = 0; i < n; ++i) {
      Pay pi;
      std::memcpy(&pi, &payload[i * stride], stride);
      std::vector<std::pair<std::int64_t, Vec<3>>> nbr;
      nbr.reserve(n + gpos.size());
      for (std::size_t j = 0; j < n; ++j) {
        Pay pj;
        std::memcpy(&pj, &payload[j * stride], stride);
        nbr.push_back({pj.id, pos[j]});
      }
      for (std::size_t j = 0; j < gpos.size(); ++j) {
        Pay pj;
        std::memcpy(&pj, &gpay[j * stride], stride);
        nbr.push_back({pj.id, gpos[j]});
      }
      F[i] = force(pos[i], pi.id, nbr);
    }
    for (std::size_t i = 0; i < n; ++i) {
      Pay pi;
      std::memcpy(&pi, &payload[i * stride], stride);
      for (int a = 0; a < 3; ++a) {
        pi.vel[a] = (pi.vel[a] + F[i][a] * kDt) * kDamp;
        pos[i][a] += pi.vel[a] * kDt;
        pos[i][a] = dmin[a] + std::fmod(std::fmod(pos[i][a] - dmin[a], L[a]) + L[a], L[a]);
      }
      std::memcpy(&payload[i * stride], &pi, stride);
    }
  }

  // Compare each owned particle to the serial reference (by id).
  int fail = 0;
  for (std::size_t i = 0; i < pos.size(); ++i) {
    Pay pi;
    std::memcpy(&pi, &payload[i * stride], stride);
    for (int a = 0; a < 3; ++a) {
      if (std::fabs(pos[i][a] - sp[pi.id][a]) > 1e-9) ++fail;
      if (std::fabs(pi.vel[a] - sv[pi.id][a]) > 1e-9) ++fail;
    }
  }
  std::int64_t lcount = (std::int64_t)pos.size(), gcount = 0;
  MPI_Reduce(&lcount, &gcount, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("# DEM loop: %lld particles, %d steps (count check %lld==%lld)\n", (long long)N,
                kSteps, (long long)gcount, (long long)N);
    if (total == 0)
      std::printf("OK (np=%d): distributed migrate+ghost+force+integrate matches serial\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d state mismatches\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
