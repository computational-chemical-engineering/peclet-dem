// dem — the owner-exclusive contact scheme's building blocks (docs/mpi_momentum_conservation.md):
// the ghost -> owner reconciliation of ParticleHalo (G6) on a directly constructed Particles +
// ParticleHalo, as in test_migrate_mpi.cpp.
//
// Modes (argv[1]):
//   reverse_closed    closed box: cross-rank ghosts only (none at np = 1)
//   reverse_periodic  periodic on every axis: periodic self-ghosts appear at np = 1 (and on every
//                     undecomposed axis), cross-rank wraps at np >= 2
//
// reverse_*: gather the ghosts of a jittered lattice, mark the baselines, add to every ghost slot
// the integer vector (gid+1, 2(gid+1), 3(gid+1)) in velPred and in posPred, reconcile
// (syncVelocities / syncPositions), and require
//   * each owned row gained exactly c(gid) (gid+1, 2(gid+1), 3(gid+1)), c = the number of ghost
//     copies of that gid over all ranks (integers and dyadic positions: every sum is exact and
//     order-independent);
//   * every ghost equals its owner afterwards (velocities verbatim; positions up to the periodic
//     shift, a multiple of the box length per axis);
//   * a second reconciliation with no writes changes no owned row, bit for bit.
// Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <random>
#include <string>
#include <vector>

#include "mpi_halo.hpp"
#include "particles.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"

using peclet::core::IVec;
using peclet::core::decomp::BlockDecomposer;
using peclet::dem::ParticleHalo;
using peclet::dem::Particles;

static constexpr int GX = 16;  // ORB cell grid + physical domain [0, GX)^3
static constexpr int NG = 8;   // lattice points per axis (spacing 2)
static constexpr double kBand = 2.5;

// Jittered lattice on dyadic values (multiples of 1/64), so position sums stay exact.
static std::vector<std::array<float, 3>> lattice() {
  std::mt19937 rng(424242u);
  std::uniform_int_distribution<int> jit(-16, 16);
  std::vector<std::array<float, 3>> p;
  for (int k = 0; k < NG; ++k)
    for (int j = 0; j < NG; ++j)
      for (int i = 0; i < NG; ++i) {
        const int idx[3] = {i, j, k};
        std::array<float, 3> q;
        for (int d = 0; d < 3; ++d)
          q[d] = (2.0f * idx[d] + 1.0f) + static_cast<float>(jit(rng)) / 64.0f;
        p.push_back(q);
      }
  return p;
}

struct Local {
  std::vector<float> v, x;  // [3 * numParticles]
  std::vector<int> gid;
};
static Local download(const Particles& P, int n) {
  Local L;
  auto hv = Kokkos::create_mirror_view(P.velPred);
  auto hx = Kokkos::create_mirror_view(P.posPred);
  auto hg = Kokkos::create_mirror_view(P.gid);
  Kokkos::deep_copy(hv, P.velPred);
  Kokkos::deep_copy(hx, P.posPred);
  Kokkos::deep_copy(hg, P.gid);
  L.v.resize(3 * n);
  L.x.resize(3 * n);
  L.gid.resize(n);
  for (int i = 0; i < n; ++i) {
    L.gid[i] = hg(i);
    for (int d = 0; d < 3; ++d) {
      L.v[3 * i + d] = hv(i, d);
      L.x[3 * i + d] = hx(i, d);
    }
  }
  return L;
}

static int runReverse(bool periodic, int rank, int size) {
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  const auto all = lattice();
  const int nGlobal = static_cast<int>(all.size());
  std::vector<int> mine;
  for (int g = 0; g < nGlobal; ++g) {
    IVec<3> cell{(long)all[g][0], (long)all[g][1], (long)all[g][2]};
    if (dec.ownerOf(cell) == rank)
      mine.push_back(g);
  }
  const int no = static_cast<int>(mine.size());

  Particles P;
  P.allocate(8 * nGlobal, 64, 64, 1, 1, 8);
  {
    auto hp = Kokkos::create_mirror_view(P.pos);
    auto hg = Kokkos::create_mirror_view(P.gid);
    for (int i = 0; i < no; ++i) {
      for (int d = 0; d < 3; ++d)
        hp(i, d) = all[mine[i]][d];
      hg(i) = mine[i];
    }
    Kokkos::deep_copy(P.pos, hp);
    Kokkos::deep_copy(P.posPred, hp);
    Kokkos::deep_copy(P.gid, hg);
  }
  P.numReal = no;
  P.numParticles = no;

  ParticleHalo halo;
  halo.initMpi(dec, {0.0, 0.0, 0.0}, {(double)GX, (double)GX, (double)GX},
               {periodic, periodic, periodic}, MPI_COMM_WORLD);
  halo.gather(P, kBand);
  const int ng = halo.numGhost(), n = no + ng;

  // c(gid): ghost copies over all ranks.
  const Local L0 = download(P, n);
  std::vector<int> cLoc(nGlobal, 0), c(nGlobal, 0);
  for (int s = no; s < n; ++s)
    ++cLoc[L0.gid[s]];
  MPI_Allreduce(cLoc.data(), c.data(), nGlobal, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Baselines, then the integer writes into every ghost slot.
  halo.markVelocityBaseline(P);
  halo.publishPositions(P, true);
  {
    auto hv = Kokkos::create_mirror_view(P.velPred);
    auto hx = Kokkos::create_mirror_view(P.posPred);
    Kokkos::deep_copy(hv, P.velPred);
    Kokkos::deep_copy(hx, P.posPred);
    for (int s = no; s < n; ++s)
      for (int d = 0; d < 3; ++d) {
        const float inc = static_cast<float>((d + 1) * (L0.gid[s] + 1));
        hv(s, d) += inc;
        hx(s, d) += inc;
      }
    Kokkos::deep_copy(P.velPred, hv);
    Kokkos::deep_copy(P.posPred, hx);
  }
  const Local Lb = download(P, n);  // owned rows as before the reconciliation
  halo.syncVelocities(P, true);
  halo.syncPositions(P, true);
  const Local L1 = download(P, n);

  int bad = 0;
  // Owned rows: exactly c(gid) times the vector (velocities started at 0; one float add for x).
  for (int i = 0; i < no; ++i)
    for (int d = 0; d < 3; ++d) {
      const float s = static_cast<float>(c[L1.gid[i]] * (d + 1) * (L1.gid[i] + 1));
      if (L1.v[3 * i + d] != Lb.v[3 * i + d] + s || L1.x[3 * i + d] != Lb.x[3 * i + d] + s)
        ++bad;
    }
  // Ghost == owner: the owners' values by gid (each gid is owned once; the sum is a copy).
  std::vector<float> ownV(3 * nGlobal, 0.0f), ownX(3 * nGlobal, 0.0f), gV(3 * nGlobal),
      gX(3 * nGlobal);
  for (int i = 0; i < no; ++i)
    for (int d = 0; d < 3; ++d) {
      ownV[3 * L1.gid[i] + d] = L1.v[3 * i + d];
      ownX[3 * L1.gid[i] + d] = L1.x[3 * i + d];
    }
  MPI_Allreduce(ownV.data(), gV.data(), 3 * nGlobal, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(ownX.data(), gX.data(), 3 * nGlobal, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  int badGhost = 0;
  for (int s = no; s < n; ++s)
    for (int d = 0; d < 3; ++d) {
      const int g = L1.gid[s];
      if (L1.v[3 * s + d] != gV[3 * g + d])
        ++badGhost;
      const float dx = L1.x[3 * s + d] - gX[3 * g + d];  // 0 or +-GX (the periodic shift)
      if (!(dx == 0.0f || (periodic && (dx == GX || dx == -GX))))
        ++badGhost;
    }
  // A second reconciliation with no writes: owned rows unchanged bit for bit.
  halo.syncVelocities(P, true);
  halo.syncPositions(P, true);
  const Local L2 = download(P, n);
  int badSecond = 0;
  for (int i = 0; i < 3 * no; ++i)
    if (std::memcmp(&L2.v[i], &L1.v[i], 4) != 0 || std::memcmp(&L2.x[i], &L1.x[i], 4) != 0)
      ++badSecond;

  int loc[4] = {bad, badGhost, badSecond, ng}, tot[4];
  MPI_Allreduce(loc, tot, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  int copies = 0;
  for (int g = 0; g < nGlobal; ++g)
    copies += c[g];
  if (rank == 0)
    std::printf(
        "  reverse_%s np=%d: ghosts=%d copies=%d owned-mismatch=%d ghost!=owner=%d "
        "second-sync-changed=%d\n",
        periodic ? "periodic" : "closed", size, tot[3], copies, tot[0], tot[1], tot[2]);
  // Periodic runs must actually exercise ghosts (self-ghosts at np = 1).
  const bool exercised = !periodic || copies > 0;
  return (tot[0] == 0 && tot[1] == 0 && tot[2] == 0 && exercised) ? 0 : 1;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, rank = 0, size = 1;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::string mode = (argc > 1) ? argv[1] : "reverse_closed";
    if (mode == "reverse_closed")
      fail = runReverse(false, rank, size);
    else if (mode == "reverse_periodic")
      fail = runReverse(true, rank, size);
    else {
      if (rank == 0)
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
      fail = 1;
    }
  }
  int total = 0;
  MPI_Allreduce(&fail, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    std::printf(total == 0 ? "OK (np=%d)\n" : "FAILED (np=%d)\n", size);
  Kokkos::finalize();
  MPI_Finalize();
  return total == 0 ? 0 : 1;
}
