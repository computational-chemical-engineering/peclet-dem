// packing-gpu — the distributed (MPI) Kokkos demStep capstone test (the last suite-migration piece).
//
// Runs the REAL KokkosSim XPBD step two ways and checks they agree:
//   * distributed  : each rank owns the ORB block of a global particle set, calls initMpi over
//                     MPI_COMM_WORLD + stepMpi (gather ghosts -> broad/narrow/solve with per-iteration
//                     owner->ghost forwards). Cross-block body-body contacts are reconstructed by the
//                     transport-core particle halo (ParticleHalo<3> + DeviceParticleHaloKokkos<3>).
//   * single-rank  : the SAME stepMpi code on a size-1 communicator (MPI_COMM_SELF) over ALL particles
//                     -> one block spanning the whole (NON-periodic) domain == the native complete run.
//
// The scene is a closed (non-periodic) box of overlapping spheres relaxed by the position solve, so the
// single-block reference needs no ghosts (it owns everything) and the distributed run must rebuild the
// split-plane neighbours via ghosts. Validation is WEAKER than the cfd MPI test: the XPBD solver
// accumulates per-particle deltas via atomics, so the decomposition reorders the sums and the result is
// NOT bit-exact even at np=1 -- but it must stay physically equivalent (positions agree to a small,
// slowly growing tolerance over a few steps; the overlap is reduced the same way). Per-process: the
// reference is computed once on rank 0 and broadcast (CUDA demgpu + Kokkos demgpu_kokkos can't co-import,
// and the single-rank Kokkos run IS the reference). Build with -DDEMGPU_KOKKOS_MPI.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "sim_kokkos.hpp"

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_migrator.hpp"

using dem::KokkosSim;
using tpx::IVec;

// Scene: G^3 spheres on a lattice with spacing < diameter (=> initial overlap), centred in a closed
// non-periodic box. globalScale = radius (scale=1), so the body-body contact distance is ~2*globalScale;
// RCUT generously covers it so every cross-block contact is ghosted symmetrically on both owners.
static constexpr int G = 4;             // grid per axis (G^3 particles)
static constexpr double SPACING = 1.8;  // < diameter 2*R => overlap 0.2
static constexpr double R = 1.0, GS = 1.0, L = 14.0;
static constexpr int STEPS = 8, GX = 16;
static constexpr double RCUT = 3.0;     // >= contact diameter (2*GS) + margin
static constexpr int VEL_ITERS = 4, POS_ITERS = 20;

// Deterministic per-index jitter (a cheap hash -> [-J,J]) breaks the lattice symmetry so the contact
// pattern is uneven: if the distributed decomposition reordered the atomic delta accumulation away
// from the single-block reference, the positions would diverge. (They don't -- the gather/forward
// reconstruct each owner's full neighbourhood exactly.)
static double jitter(int g, int c) {
  unsigned h = static_cast<unsigned>(g) * 2654435761u + static_cast<unsigned>(c) * 40503u + 1u;
  h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
  return (static_cast<double>(h & 0xffff) / 65535.0 - 0.5) * 2.0 * (0.12 * SPACING);
}

static std::vector<float> globalPositions(int& n) {
  std::vector<float> p;
  const double span = (G - 1) * SPACING;
  const double off = 0.5 * (L - span);
  for (int iz = 0; iz < G; ++iz)
    for (int iy = 0; iy < G; ++iy)
      for (int ix = 0; ix < G; ++ix) {
        const int g = static_cast<int>(p.size() / 3);
        p.push_back(static_cast<float>(off + ix * SPACING + jitter(g, 0)));
        p.push_back(static_cast<float>(off + iy * SPACING + jitter(g, 1)));
        p.push_back(static_cast<float>(off + iz * SPACING + jitter(g, 2)));
      }
  n = G * G * G;
  return p;
}

static void configure(KokkosSim& s) {
  s.setDomain(L, L, L, /*px=*/false, /*py=*/false, /*pz=*/false);  // closed box; no periodic wrap
  s.setGlobalScale(GS);
  s.setSphereShape(R);
  s.setDt(1e-2f);
  s.setGravity(0, 0, 0);                 // pure overlap relaxation (no body force)
  s.setSolverIterations(POS_ITERS, VEL_ITERS);
  s.setMaterialParams(0.0f, 0.0f, 0.0f); // no restitution / friction (frictionless XPBD)
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int n = 0;
    const std::vector<float> gpos = globalPositions(n);
    const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
    const std::tuple<long, long, long> gsize{GX, GX, GX};
    const std::tuple<bool, bool, bool> periodic{false, false, false};

    // --- ownership: which global particles this rank owns (ORB block of the gsize cell grid) ---
    tpx::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) { map.origin[i] = 0; map.cellSize[i] = L / GX; map.periodic[i] = false; }
    tpx::halo::ParticleMigrator<3> mig;
    mig.init(dec, rank, map, MPI_COMM_WORLD);
    std::vector<float> ownedPos;
    std::vector<int> ownedGid;
    for (int g = 0; g < n; ++g) {
      tpx::Vec<3> x{gpos[3 * g], gpos[3 * g + 1], gpos[3 * g + 2]};
      if (mig.ownerOf(x) == rank) {
        ownedPos.push_back(gpos[3 * g]); ownedPos.push_back(gpos[3 * g + 1]); ownedPos.push_back(gpos[3 * g + 2]);
        ownedGid.push_back(g);
      }
    }
    const int nOwned = static_cast<int>(ownedGid.size());

    // --- distributed solve (MPI_COMM_WORLD) ---
    KokkosSim dist(2 * n + 64);  // capacity: owned + ghost headroom
    configure(dist);
    dist.setPositions(ownedPos);
    dist.initMpi(origin, dsize, gsize, periodic, MPI_COMM_WORLD);
    dist.enableMpiStep(RCUT, /*sync_every=*/1, /*forward_rotation=*/false);  // spheres
    dist.stepMpi(STEPS);
    const std::vector<float> distPos = dist.getPositions();
    const float distOv = dist.maxOverlap();
    int totGhost = 0, myGhost = dist.numGhost();
    MPI_Allreduce(&myGhost, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // --- single-rank reference (rank 0, MPI_COMM_SELF, ALL particles, the SAME stepMpi code) ---
    std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
    float refOv = 0.0f;
    if (rank == 0) {
      KokkosSim ref(2 * n + 64);
      configure(ref);
      ref.setPositions(gpos);
      ref.initMpi(origin, dsize, gsize, periodic, MPI_COMM_SELF);  // size-1: one block, no ghosts
      ref.enableMpiStep(RCUT, 1, false);
      ref.stepMpi(STEPS);
      refPos = ref.getPositions();
      refOv = ref.maxOverlap();
    }
    MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&refOv, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- compare this rank's owned particles (by global id) against the reference trajectory ---
    double localMax = 0.0;
    for (int i = 0; i < nOwned; ++i) {
      const int g = ownedGid[i];
      for (int c = 0; c < 3; ++c)
        localMax = std::max(localMax, static_cast<double>(std::fabs(distPos[3 * i + c] - refPos[3 * g + c])));
    }
    double posErr = 0.0;
    MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double ovErr = std::fabs(static_cast<double>(distOv) - static_cast<double>(refOv));

    // np=1: distributed(WORLD,size 1) and reference(SELF) run the IDENTICAL single block (no ghosts),
    // so the gap is pure float atomic-add reordering between two launch sequences (~1e-4). np>1 adds
    // the genuine decomposition/ghost reordering on top -- still physically equivalent (~1e-4..1e-3).
    const double posTol = (size == 1) ? 1e-3 : 1e-2;
    if (rank == 0)
      std::printf("  np=%d  particles=%d  ghosts(total)=%d  posErr=%.3e (tol %.0e)  overlap dist=%.4e ref=%.4e d=%.2e\n",
                  size, n, totGhost, posErr, posTol, distOv, refOv, ovErr);
    if (!(posErr < posTol) || !(ovErr < 5e-3)) fail = 1;
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d): distributed Kokkos demStep == single-rank (physical equivalence)\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
