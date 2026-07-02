// dem — distributed load re-balancing test (KokkosSim::rebalance / enable_mpi_step
// rebalance_every).
//
// A skewed particle cloud (density concentrated toward x=0) is distributed over the ORB blocks, so
// the equal-cell decomposition leaves the low-x ranks overloaded. Each particle is given a DISTINCT
// committed state derived deterministically from its position (velocity, angular velocity,
// orientation, scale, inverse inertia). A rebalance re-decomposes by particle count and migrates
// ownership. We check:
//   (1) every migrated particle's full committed state is preserved bit-for-bit — after the
//   migration a
//       particle at position p still carries exactly f(p) in every field (the SoA pack/unpack
//       round-trips correctly across ranks);
//   (2) the global particle count is conserved;
//   (3) the per-rank count imbalance (max/mean) drops vs the equal-cell decomposition;
//   (4) the distributed step still runs and stays finite after migration, both via an explicit
//       rebalance() and via the wired enable_mpi_step(rebalance_every=...) path.
// np = 1,2,4. Build with -DDEM_MPI.
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <vector>

#include "sim.hpp"
#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_migrator.hpp"

using dem::KokkosSim;
using tpx::IVec;

static constexpr double R = 1.0, GS = 1.0;
static constexpr int GX = 16;      // ORB cell grid per axis
static constexpr double L = 30.0;  // box edge (sparse: mild overlaps -> stable relaxation)
static constexpr double RCUT = 3.0;
static constexpr int N = 2000;

// Deterministic per-(index,axis) hash -> [0,1).
static double h01(int g, int c) {
  unsigned h = (unsigned)g * 2654435761u + (unsigned)c * 40503u + 1u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return (double)(h & 0xffff) / 65535.0;
}

// Global positions: x concentrated toward 0 (u^2 warp) so the density is strongly non-uniform; y,z
// uniform. Clamped inside the closed box.
static std::vector<float> skewedPositions() {
  std::vector<float> p((std::size_t)N * 3);
  const double lo = R + 0.05, hi = L - R - 0.05;
  for (int g = 0; g < N; ++g) {
    double ux = h01(g, 0), uy = h01(g, 1), uz = h01(g, 2);
    double x = lo + (hi - lo) * ux * ux;  // dense near x=lo
    double y = lo + (hi - lo) * uy;
    double z = lo + (hi - lo) * uz;
    p[3 * g] = (float)x;
    p[3 * g + 1] = (float)y;
    p[3 * g + 2] = (float)z;
  }
  return p;
}

// Distinct committed state as deterministic float functions of position (so a particle is
// identifiable by its position after migration, and every field is independently checkable).
static std::array<float, 3> fVel(float x, float y, float z) {
  return {x * 0.013f, y * 0.017f, z * 0.019f};
}
static std::array<float, 3> fAng(float x, float y, float z) {
  return {z * 0.011f, x * 0.007f, y * 0.005f};
}
static std::array<float, 3> fInvI(float x, float y, float z) {
  return {1.0f + 0.001f * x, 1.0f + 0.001f * y, 1.0f + 0.001f * z};
}
static float fScale(float x, float y, float z) {
  return 1.0f + 0.0001f * (x + y + z);
}
static std::array<float, 4> fQuat(float x, float y, float z) {
  float q[4] = {1.0f, 0.01f * x, 0.01f * y, 0.01f * z};
  float n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

static void configure(KokkosSim& sim) {
  sim.setDomain(L, L, L, false, false, false);
  sim.setGlobalScale(GS);
  sim.setSphereShape(R);
  sim.setDt(1e-2f);
  sim.setGravity(0, 0, 0);
  sim.setSolverIterations(10, 0);
  sim.setMaterialParams(0.0f, 0.0f, 0.0f);
}

// Set each owned particle's committed state from f(position).
static void setStateFromPos(KokkosSim& sim, const std::vector<float>& pos) {
  const int n = (int)(pos.size() / 3);
  std::vector<float> vel(n * 3), ang(n * 3), invI(n * 3), quat(n * 4), scale(n);
  for (int i = 0; i < n; ++i) {
    float x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    auto v = fVel(x, y, z);
    auto a = fAng(x, y, z);
    auto ii = fInvI(x, y, z);
    auto q = fQuat(x, y, z);
    for (int c = 0; c < 3; ++c) {
      vel[3 * i + c] = v[c];
      ang[3 * i + c] = a[c];
      invI[3 * i + c] = ii[c];
    }
    for (int c = 0; c < 4; ++c)
      quat[4 * i + c] = q[c];
    scale[i] = fScale(x, y, z);
  }
  sim.setVelocities(vel);
  sim.setAngularVelocities(ang);
  sim.setInvInertia(invI);
  sim.setQuaternions(quat);
  sim.setScales(scale);
}

// Verify this rank's owned particles all carry f(position) bit-for-bit. Returns mismatch count.
static int verifyState(KokkosSim& sim) {
  const std::vector<float> pos = sim.getPositions(), vel = sim.getVelocities(),
                           ang = sim.getAngularVelocities(), invI = sim.getInvInertia(),
                           quat = sim.getQuaternions(), scale = sim.getScales();
  const int n = (int)(pos.size() / 3);
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    float x = pos[3 * i], y = pos[3 * i + 1], z = pos[3 * i + 2];
    auto v = fVel(x, y, z);
    auto a = fAng(x, y, z);
    auto ii = fInvI(x, y, z);
    auto q = fQuat(x, y, z);
    for (int c = 0; c < 3; ++c) {
      if (vel[3 * i + c] != v[c])
        ++bad;
      if (ang[3 * i + c] != a[c])
        ++bad;
      if (invI[3 * i + c] != ii[c])
        ++bad;
    }
    for (int c = 0; c < 4; ++c)
      if (quat[4 * i + c] != q[c])
        ++bad;
    if (scale[i] != fScale(x, y, z))
      ++bad;
  }
  return bad;
}

static double imbalance(int localCount, MPI_Comm comm) {
  long lc = localCount, mx = 0, sum = 0;
  MPI_Allreduce(&lc, &mx, 1, MPI_LONG, MPI_MAX, comm);
  MPI_Allreduce(&lc, &sum, 1, MPI_LONG, MPI_SUM, comm);
  int size = 1;
  MPI_Comm_size(comm, &size);
  return (double)mx / ((double)sum / (double)size);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::vector<float> gpos = skewedPositions();
    const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
    const std::tuple<long, long, long> gsize{GX, GX, GX};
    const std::tuple<bool, bool, bool> per{false, false, false};

    // Which global particles this rank owns under the equal-cell ORB.
    tpx::decomp::BlockDecomposer<3> dec((std::size_t)size, IVec<3>{GX, GX, GX});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = 0;
      map.cellSize[i] = L / GX;
      map.periodic[i] = false;
    }
    tpx::halo::ParticleMigrator<3> mig;
    mig.init(dec, rank, map, MPI_COMM_WORLD);
    std::vector<float> ownedPos;
    for (int g = 0; g < N; ++g) {
      tpx::Vec<3> x{gpos[3 * g], gpos[3 * g + 1], gpos[3 * g + 2]};
      if (mig.ownerOf(x) == rank) {
        ownedPos.push_back(gpos[3 * g]);
        ownedPos.push_back(gpos[3 * g + 1]);
        ownedPos.push_back(gpos[3 * g + 2]);
      }
    }
    const int nOwned = (int)(ownedPos.size() / 3);
    const int cap = 4 * N;  // generous: peak per-rank owned + ghost headroom

    KokkosSim sim(cap);
    configure(sim);
    sim.setPositions(ownedPos);
    setStateFromPos(sim, ownedPos);
    sim.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
    sim.enableMpiStep(RCUT, 1, /*forward_rotation=*/false);

    const double imb0 = imbalance(nOwned, MPI_COMM_WORLD);

    // (1)+(2)+(3): explicit rebalance preserves every field + count, and improves the balance.
    const int newN = sim.rebalance();
    if (verifyState(sim) != 0) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  state corrupted by migration\n");
    }
    long lN = newN, gN = 0;
    MPI_Allreduce(&lN, &gN, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (gN != N) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  count not conserved: %ld != %d\n", gN, N);
    }
    const double imb1 = imbalance(newN, MPI_COMM_WORLD);
    if (size > 1 && !(imb0 > 1.2 && imb1 < imb0 && imb1 < 1.6)) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  imbalance not improved: %.3f -> %.3f\n", imb0, imb1);
    }

    // (4a): the step runs and stays finite after an explicit rebalance.
    sim.stepMpi(4);
    float ov = sim.maxOverlap();
    if (!std::isfinite(ov)) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  non-finite overlap after rebalance\n");
    }

    if (rank == 0)
      std::printf("  np=%d: imbalance %.3f -> %.3f, count %ld, overlap %.3e\n", size, imb0, imb1,
                  gN, ov);
  }

  // (4b): the wired auto-rebalance path (rebalance_every>0) runs cleanly from a fresh sim.
  {
    const std::vector<float> gpos = skewedPositions();
    const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
    const std::tuple<long, long, long> gsize{GX, GX, GX};
    const std::tuple<bool, bool, bool> per{false, false, false};
    tpx::decomp::BlockDecomposer<3> dec((std::size_t)size, IVec<3>{GX, GX, GX});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = 0;
      map.cellSize[i] = L / GX;
      map.periodic[i] = false;
    }
    tpx::halo::ParticleMigrator<3> mig;
    mig.init(dec, rank, map, MPI_COMM_WORLD);
    std::vector<float> ownedPos;
    for (int g = 0; g < N; ++g) {
      tpx::Vec<3> x{gpos[3 * g], gpos[3 * g + 1], gpos[3 * g + 2]};
      if (mig.ownerOf(x) == rank) {
        ownedPos.push_back(gpos[3 * g]);
        ownedPos.push_back(gpos[3 * g + 1]);
        ownedPos.push_back(gpos[3 * g + 2]);
      }
    }
    KokkosSim sim(4 * N);
    configure(sim);
    sim.setPositions(ownedPos);
    sim.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
    sim.enableMpiStep(RCUT, 1, false, /*rebalance_every=*/3);
    sim.stepMpi(6);
    if (!std::isfinite(sim.maxOverlap())) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  auto-rebalance path non-finite\n");
    }
    long lc = sim.numParticles(), gc = 0;
    MPI_Allreduce(&lc, &gc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (gc != N) {
      fail = 1;
      if (rank == 0)
        std::fprintf(stderr, "  auto-rebalance lost particles: %ld\n", gc);
    }
  }

  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): rebalance migrates state intact + improves balance\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
