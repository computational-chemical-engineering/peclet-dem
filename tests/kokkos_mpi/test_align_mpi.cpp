// dem — the ALIGNED weighted ORB of migrate_to_weights(w, align) is flow's partition, cell for
// cell.
//
// A coupled CFD-DEM run co-locates by construction: flow's rebalance_by_weights(w) and dem's
// migrate_to_weights(w, align) each build the partition from the same replicated weight field, and
// nothing but the split positions is shared. flow chooses the partition with core's
// chooseAlignedWeighted(np, G, w) (the largest alignment 2^a within the 1.05 weight-imbalance
// budget, amr/docs/amr_mg_core_boundary.md §11.4) and returns 2^a; dem must rebuild exactly that
// partition from (w, 2^a). A different partition would silently put particles outside the fluid
// block their drag is deposited in. We check, on every rank:
//   (1) flow's choice: dem's decomposition after migrate_to_weights(w, 1 << a) equals
//       chooseAlignedWeighted(np, G, w).dec -- same block origins and sizes, and the same owner for
//       every cell of the grid -- for several weight fields, at least one of them aligned (a > 0);
//   (2) every forced alignment 1, 2, 4, ... that the grid admits: dem's decomposition equals core's
//       init(np, G, w, {align, ...}) directly (align 1: the plain weighted constructor);
//   (3) ownership follows: every owned particle lies in this rank's block of that partition and the
//       global count is conserved;
//   (4) validation: align not a power of two, not dividing the grid, leaving fewer aligned boxes
//       than ranks, or weights of the wrong length -> std::invalid_argument (Python ValueError).
// np = 1, 2, 4, 8. Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "sim.hpp"

using peclet::core::IVec;
using peclet::core::Real;
using peclet::core::decomp::BlockDecomposer;
using peclet::dem::Simulation;

static constexpr long GX = 32, GY = 32, GZ = 16;  // ORB cell grid (2^5, 2^5, 2^4)
static constexpr double H = 1.0;                  // cell size: the domain is GX x GY x GZ
static constexpr double R = 0.3;
static constexpr int N = 3000;

// Deterministic per-(index,axis) hash -> [0,1).
static double h01(int g, int c) {
  unsigned h = (unsigned)g * 2654435761u + (unsigned)c * 40503u + 1u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return (double)(h & 0xffff) / 65535.0;
}

static std::size_t lin(long i, long j, long k) {
  return (std::size_t)(i + GX * (j + GY * k));
}

// A settled heap: 1 + 12 below a surface that falls linearly in x and y (moves every split), 1
// above -- the shape CfdDem.rebalance(gamma) builds (1 + gamma * count).
static std::vector<Real> heapWeights(double bed, double tilt) {
  std::vector<Real> w((std::size_t)(GX * GY * GZ), 1.0);
  for (long k = 0; k < GZ; ++k)
    for (long j = 0; j < GY; ++j)
      for (long i = 0; i < GX; ++i) {
        const double x = (i + 0.5) / GX, y = (j + 0.5) / GY, z = (k + 0.5) / GZ;
        const double zs = bed * (1.0 + tilt * (0.5 - x) + tilt * (0.5 - y));
        if (z < zs)
          w[lin(i, j, k)] = 13.0;
      }
  return w;
}

// Rough per-cell counts: 1 + a hashed integer 0..7 in a corner-weighted region.
static std::vector<Real> patchyWeights() {
  std::vector<Real> w((std::size_t)(GX * GY * GZ), 1.0);
  for (long k = 0; k < GZ; ++k)
    for (long j = 0; j < GY; ++j)
      for (long i = 0; i < GX; ++i)
        if (i + j < GX)
          w[lin(i, j, k)] += std::floor(8.0 * h01((int)lin(i, j, k), 3));
  return w;
}

// Same partition: block origins and sizes, and the owner of every grid cell (the tree walk the
// particle migration uses). Returns the number of differences.
static long compareDecomp(const BlockDecomposer<3>& a, const BlockDecomposer<3>& b) {
  long bad = 0;
  if (a.numBlocks() != b.numBlocks() || a.globalSize() != b.globalSize())
    return 1;
  for (std::size_t n = 0; n < a.numBlocks(); ++n)
    if (a.origins()[n] != b.origins()[n] || a.sizes()[n] != b.sizes()[n])
      ++bad;
  for (long k = 0; k < GZ; ++k)
    for (long j = 0; j < GY; ++j)
      for (long i = 0; i < GX; ++i)
        if (a.ownerOf(IVec<3>{i, j, k}) != b.ownerOf(IVec<3>{i, j, k}))
          ++bad;
  return bad;
}

// Owned particles lie in this rank's block of `dec` (cell of the position, as the migrator bins
// it).
static long outsideBlock(Simulation& sim, const BlockDecomposer<3>& dec, int rank) {
  const std::vector<float> pos = sim.getPositions();
  const IVec<3> o = dec.origins()[(std::size_t)rank], s = dec.sizes()[(std::size_t)rank];
  long bad = 0;
  for (std::size_t p = 0; p < pos.size() / 3; ++p)
    for (int c = 0; c < 3; ++c) {
      const long cell = (long)std::floor((double)pos[3 * p + c] / H);
      if (cell < o[c] || cell >= o[c] + s[c])
        ++bad;
    }
  return bad;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const IVec<3> G{GX, GY, GZ};

    // Global particle set, all of it handed to rank 0; the first migration distributes it.
    std::vector<float> gpos((std::size_t)N * 3);
    for (int g = 0; g < N; ++g) {
      gpos[3 * g] = (float)(R + (GX * H - 2 * R) * h01(g, 0));
      gpos[3 * g + 1] = (float)(R + (GY * H - 2 * R) * h01(g, 1));
      gpos[3 * g + 2] = (float)(R + (GZ * H - 2 * R) * h01(g, 2));
    }
    Simulation sim(2 * N);
    sim.setDomain(GX * H, GY * H, GZ * H, false, false, false);
    sim.setSphereShape(R);
    sim.setDt(1e-3f);
    sim.setPositions(rank == 0 ? gpos : std::vector<float>{});
    sim.initMpi({0.0, 0.0, 0.0}, {GX * H, GY * H, GZ * H}, {GX, GY, GZ}, {false, false, false},
                MPI_COMM_WORLD);

    auto check = [&](const char* what, const BlockDecomposer<3>& ref) {
      const long dd = compareDecomp(sim.decomposer(), ref);
      const long ob = outsideBlock(sim, ref, rank);
      long lc = sim.numParticles(), gc = 0;
      MPI_Allreduce(&lc, &gc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (dd != 0 || ob != 0 || gc != N) {
        fail = 1;
        std::fprintf(stderr,
                     "  rank %d %s: %ld partition differences, %ld outside the block, "
                     "count %ld\n",
                     rank, what, dd, ob, gc);
      }
    };

    const std::vector<std::vector<Real>> fields = {heapWeights(0.4, 0.0), heapWeights(0.4, 0.5),
                                                   heapWeights(0.3, 0.3), patchyWeights()};
    int alignedCases = 0;
    for (std::size_t f = 0; f < fields.size(); ++f) {
      const auto& w = fields[f];
      // (1) flow's choice (flow_ibm_mpi.hpp rebalanceByWeights: the same call, the same defaults).
      const auto c = peclet::core::decomp::chooseAlignedWeighted((std::size_t)size, G, w);
      sim.migrateToWeights(w, 1 << c.a);
      char what[96];
      std::snprintf(what, sizeof what, "field %zu, flow's choice a = %d", f, c.a);
      check(what, c.dec);
      if (c.a > 0)
        ++alignedCases;
      if (rank == 0)
        std::printf("  np=%d field %zu: a = %d, weight imbalance %.4f\n", size, f, c.a,
                    (double)c.imbalance);
      // (2) every alignment the grid admits, against core directly.
      for (long align = 1; GZ % align == 0 && (GX / align) * (GY / align) * (GZ / align) >= size;
           align *= 2) {
        BlockDecomposer<3> ref;
        if (align == 1)
          ref = BlockDecomposer<3>((std::size_t)size, G, w);
        else
          ref.init((std::size_t)size, G, w, IVec<3>{align, align, align});
        sim.migrateToWeights(w, (int)align);
        std::snprintf(what, sizeof what, "field %zu, align %ld", f, align);
        check(what, ref);
      }
    }
    // A heap moves every split at np > 1, and one of these fields must actually exercise the
    // alignment -- otherwise (1) would compare the plain weighted ORB with itself.
    if (size > 1 && alignedCases == 0) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  no field chose a > 0: the aligned path is untested\n");
    }

    // (4) validation (pure, identical on every rank: no rank enters a collective alone).
    auto expectThrow = [&](const char* what, const std::vector<Real>& w, int align) {
      bool threw = false;
      try {
        sim.migrateToWeights(w, align);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      if (!threw) {
        fail = 1;
        if (rank == 0)
          std::fprintf(stderr, "  %s: no std::invalid_argument\n", what);
      }
    };
    expectThrow("align 3", fields[0], 3);
    expectThrow("align 0", fields[0], 0);
    expectThrow("align 32 (does not divide GZ = 16)", fields[0], 32);
    expectThrow("short weights", std::vector<Real>((std::size_t)(GX * GY * GZ - 1), 1.0), 1);
    if (size > 4)
      expectThrow("align 16 (2 x 2 x 1 aligned boxes, fewer than np)", fields[0], 16);
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): migrate_to_weights(w, align) builds flow's aligned partition\n",
                  size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
