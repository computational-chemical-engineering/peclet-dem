// dem — ParticleHalo::migrateTo onto a shared, externally-supplied decomposition (the co-rebalance
// counterpart of flow's Solver::redistribute). Particles owned under an equal-cell ORB are migrated
// onto a WEIGHTED ORB; the global count must be conserved and every particle must end up owned by
// the rank the new decomposition assigns it. Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "mpi_halo.hpp"
#include "particles.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::dem::ParticleHalo;
using peclet::dem::Particles;

static constexpr int GX = 16;  // ORB cell grid + physical domain [0,GX)^3

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, rank = 0, size = 1;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BlockDecomposer<3> D1((std::size_t)size, IVec<3>{GX, GX, GX});  // equal cell
    std::vector<peclet::core::Real> w((std::size_t)GX * GX * GX, 1.0);
    for (int z = 0; z < GX; ++z)
      for (int y = 0; y < GX; ++y)
        for (int x = 0; x < GX; ++x)
          if (x < GX / 2)
            w[(std::size_t)x + (std::size_t)y * GX + (std::size_t)z * GX * GX] = 6.0;
    BlockDecomposer<3> D2((std::size_t)size, IVec<3>{GX, GX, GX}, w);  // weighted

    // global particles on a 8^3 lattice; this rank keeps those D1 owns (cellSize == 1, origin 0).
    const int NG = 8;
    std::vector<std::array<float, 3>> mine;
    for (int i = 0; i < NG; ++i)
      for (int j = 0; j < NG; ++j)
        for (int k = 0; k < NG; ++k) {
          std::array<float, 3> p{(i + 0.5f) * GX / NG, (j + 0.5f) * GX / NG, (k + 0.5f) * GX / NG};
          IVec<3> cell{(long)p[0], (long)p[1], (long)p[2]};
          if (D1.ownerOf(cell) == rank)
            mine.push_back(p);
        }
    const int no = (int)mine.size();

    Particles P;
    P.allocate(NG * NG * NG, 64, 64, 1, 1, 8);  // cap = global count (safe upper bound)
    auto hp = Kokkos::create_mirror_view(P.pos);
    for (int i = 0; i < no; ++i)
      for (int c = 0; c < 3; ++c)
        hp(i, c) = mine[(std::size_t)i][c];
    Kokkos::deep_copy(P.pos, hp);
    P.numReal = no;
    P.numParticles = no;

    ParticleHalo halo;
    halo.initMpi(D1, {0.0, 0.0, 0.0}, {(double)GX, (double)GX, (double)GX}, {true, true, true},
                 MPI_COMM_WORLD);

    int before = 0;
    MPI_Allreduce(&no, &before, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int newN = halo.migrateTo(P, D2);
    int after = 0;
    MPI_Allreduce(&newN, &after, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // every owned particle must now be D2-owned by this rank.
    auto hp2 = Kokkos::create_mirror_view(P.pos);
    Kokkos::deep_copy(hp2, P.pos);
    int wrongOwner = 0;
    for (int i = 0; i < newN; ++i) {
      IVec<3> cell{(long)hp2(i, 0), (long)hp2(i, 1), (long)hp2(i, 2)};
      if (D2.ownerOf(cell) != rank)
        ++wrongOwner;
    }
    int wrongTotal = 0;
    MPI_Allreduce(&wrongOwner, &wrongTotal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (before != after || wrongTotal != 0)
      fail = 1;
    if (rank == 0)
      std::printf("  migrateTo np=%d: count %d->%d (conserved=%s) wrong-owner=%d\n", size, before,
                  after, before == after ? "yes" : "NO", wrongTotal);
  }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(total == 0 ? "OK (np=%d): migrateTo onto a shared weighted decomposition\n"
                           : "FAILED (np=%d)\n",
                size);
  Kokkos::finalize();
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
