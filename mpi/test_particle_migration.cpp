// First MPI step for packing-gpu (mirrors cfd-gpu's test_mac_halo): validate that transport-core's
// ParticleMigrator migrates packing-style particles across rank-owned blocks of the domain.
//
// packing-gpu stores particles as SoA float4 arrays (d_pos[.xyz,.w=inv_mass], d_vel, d_quat,
// d_ang_vel, d_inv_inertia, d_scale, d_shape_ids) in a periodic box [domain_min, domain_min+size).
// Here each particle's full per-particle record is the migrator's opaque payload; its position drives
// ownership. We decompose the domain (ORB over a cell grid covering the box), then over several
// random-walk + migrate steps require, globally and every step: particle count conserved, every
// particle on its owning rank, and the id multiset exactly preserved (sum + xor reductions).
//
// Host-side: migration is infrequent and host-staged (CUDA-aware MPI is unavailable on this box), so
// a real integration would download particles, migrate, and upload — the same pattern validated here.
#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_migrator.hpp"

using namespace tpx;
using tpx::decomp::BlockDecomposer;
using tpx::halo::DomainMap;
using tpx::halo::ParticleMigrator;

// packing-gpu per-particle record (the migrator payload). Mirrors the SoA float4 state arrays.
struct PackingParticle {
  float pos[4];          // .xyz position, .w inv_mass
  float vel[4];          // .xyz velocity, .w type/phase
  float quat[4];         // orientation
  float ang_vel[4];      // angular velocity
  float inv_inertia[4];  // diagonal inertia + pad
  float scale;
  int shape_id;
  std::int64_t id;  // unique id for conservation checks
};

static double frac(std::uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return static_cast<double>(x & 0xFFFFFF) / static_cast<double>(0x1000000);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // packing-style domain: a periodic box. Decompose a cell grid covering it.
  const double dmin[3] = {0.0, 0.0, 0.0};
  const double dsize[3] = {10.0, 8.0, 6.0};
  IVec<3> gsize{40, 32, 24};  // decomposition resolution
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), gsize);

  DomainMap<3> map;
  for (int i = 0; i < 3; ++i) {
    map.origin[i] = dmin[i];
    map.cellSize[i] = dsize[i] / static_cast<double>(gsize[i]);
    map.periodic[i] = true;
  }
  ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);

  // Generate N particles deterministically, scattered across ranks by id%size.
  const std::int64_t N = 5000;
  const std::size_t stride = sizeof(PackingParticle);
  std::vector<Vec<3>> pos;
  std::vector<char> payload;
  for (std::int64_t id = rank; id < N; id += size) {
    Vec<3> x{dmin[0] + frac((std::uint64_t)id * 3 + 0) * dsize[0],
             dmin[1] + frac((std::uint64_t)id * 3 + 1) * dsize[1],
             dmin[2] + frac((std::uint64_t)id * 3 + 2) * dsize[2]};
    pos.push_back(x);
    PackingParticle p{};
    p.pos[0] = (float)x[0]; p.pos[1] = (float)x[1]; p.pos[2] = (float)x[2]; p.pos[3] = 1.0f;
    p.quat[3] = 1.0f;
    p.scale = 1.0f;
    p.shape_id = (int)(id % 4);
    p.id = id;
    std::size_t off = payload.size();
    payload.resize(off + stride);
    std::memcpy(&payload[off], &p, stride);
  }

  std::int64_t expectSum = N * (N - 1) / 2, expectXor = 0;
  for (std::int64_t i = 0; i < N; ++i) expectXor ^= i;

  int fail = 0;
  for (int step = 0; step < 6; ++step) {
    mig.migrate(pos, payload, stride);

    std::int64_t lcount = (std::int64_t)pos.size(), lsum = 0, lxor = 0;
    for (std::size_t k = 0; k < pos.size(); ++k) {
      if (mig.ownerOf(pos[k]) != rank) ++fail;
      PackingParticle p;
      std::memcpy(&p, &payload[k * stride], stride);
      lsum += p.id;
      lxor ^= p.id;
    }
    std::int64_t gcount = 0, gsum = 0, gxor = 0;
    MPI_Allreduce(&lcount, &gcount, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lsum, &gsum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&lxor, &gxor, 1, MPI_INT64_T, MPI_BXOR, MPI_COMM_WORLD);
    if (gcount != N || gsum != expectSum || gxor != expectXor) {
      if (rank == 0)
        std::fprintf(stderr, "  step %d FAILED: count=%lld sum=%lld xor=%lld\n", step,
                     (long long)gcount, (long long)gsum, (long long)gxor);
      ++fail;
    }

    // random walk for the next step
    for (std::size_t k = 0; k < pos.size(); ++k) {
      PackingParticle p;
      std::memcpy(&p, &payload[k * stride], stride);
      for (int i = 0; i < 3; ++i)
        pos[k][i] += (frac((std::uint64_t)p.id * 7 + i + step * 131) - 0.5) * 0.25 * dsize[i];
    }
  }

  // Ghost particles: gather copies within one interaction radius of the block boundary, so a real
  // integration would run cuBQL broadphase locally over owned + ghost particles per rank.
  const double rcut = 0.5;
  std::vector<Vec<3>> gpos;
  std::vector<char> gpay;
  std::size_t nghost = mig.gatherGhosts(pos, payload, stride, rcut, gpos, gpay);
  // Invariant: every received ghost image lies within rcut of this rank's block.
  Vec<3> img;
  for (std::size_t k = 0; k < gpos.size(); ++k)
    if (!mig.withinRcutOfBlock(gpos[k], rank, rcut, img)) ++fail;
  long long lg = (long long)nghost, gghost = 0;
  MPI_Reduce(&lg, &gghost, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    std::printf("# %lld ghost particles gathered (rcut=%.2f)\n", gghost, rcut);
    if (total == 0)
      std::printf(
          "OK (np=%d): %lld packing particles migrate+ghost correctly over 6 steps\n", size,
          (long long)N);
    else
      std::fprintf(stderr, "FAILED (np=%d): %d\n", size, total);
  }
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
