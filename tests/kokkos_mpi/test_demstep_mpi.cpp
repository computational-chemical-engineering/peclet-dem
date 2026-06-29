// packing-gpu — the distributed (MPI) Kokkos demStep capstone test (the last suite-migration piece).
//
// Runs the REAL KokkosSim XPBD step two ways and checks they agree:
//   * distributed  : each rank owns the ORB block of a global particle set, calls initMpi over
//                     MPI_COMM_WORLD + stepMpi (gather ghosts -> broad/narrow/solve with per-iteration
//                     owner->ghost forwards). Cross-block body-body contacts are reconstructed by the
//                     transport-core particle halo (ParticleHaloTopology<3> + ParticleHalo<3>).
//   * single-rank  : the SAME stepMpi code on a size-1 communicator (MPI_COMM_SELF) over ALL particles
//                     -> one block spanning the whole domain == the native complete run.
//
// Two scenes (argv[1] = "closed" | "periodic"):
//   closed   : a non-periodic box of overlapping spheres relaxed by the position solve. The single
//              block owns everything (no ghosts needed); the distributed run rebuilds the split-plane
//              neighbours via ghosts.
//   periodic : a FULLY periodic tiled box, so neighbours wrap across every face. This exercises the
//              LOCAL periodic self-ghosts (ParticleHaloTopology build(..., includePeriodicSelf=true)) on every
//              UNDECOMPOSED periodic axis -- all three at np=1, and the z of the np=4 2x2x1 ORB layout
//              (each rank is its own periodic image on a "x1" axis, so those wrap contacts exist only
//              as self-ghosts). A periodic box needs a thick ghost boundary layer, so capacity is sized
//              generously (gather throws on overflow rather than corrupting the SoA).
//
// Validation is WEAKER than the cfd MPI test: the XPBD solver accumulates per-particle deltas via
// atomics, so the decomposition reorders the sums and the result is NOT bit-exact even at np=1. closed
// (spacing < diameter, over-constrained) has an essentially unique packing -> tight position parity;
// periodic (spacing > diameter) has cage slack -> a manifold of zero-overlap configs, so the physical
// check is that the overlap relaxes identically and positions stay bounded (no divergence). Per-process:
// the reference is computed once on rank 0 and broadcast. Build with -DDEM_MPI.
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "sim.hpp"

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_migrator.hpp"

using dem::KokkosSim;
using tpx::IVec;

// globalScale = radius (scale=1), so the body-body contact distance is ~2*globalScale; RCUT generously
// covers it so every cross-block / periodic-wrap contact is ghosted symmetrically on both owners.
static constexpr double R = 1.0, GS = 1.0;
static constexpr int STEPS = 8, GX = 16;
static constexpr double RCUT = 3.0;  // >= contact diameter (2*GS) + margin
static constexpr int VEL_ITERS = 4, POS_ITERS = 20;

struct Scene {
  bool per[3];  // per-axis periodicity
  int G;        // particles per axis
  double L;     // box edge
  double spacing;
};

// Deterministic per-index jitter (a cheap hash -> [-amp,amp]) breaks the lattice symmetry so the contact
// pattern is uneven: if the distributed decomposition reordered the atomic delta accumulation away from
// the single-block reference, the positions would diverge.
static double jitter(int g, int c, double amp) {
  unsigned h = static_cast<unsigned>(g) * 2654435761u + static_cast<unsigned>(c) * 40503u + 1u;
  h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
  return (static_cast<double>(h & 0xffff) / 65535.0 - 0.5) * 2.0 * amp;
}

// A periodic axis TILES the box (L = G*spacing) so neighbours wrap across that face; a non-periodic axis
// centres the lattice with a margin. spacing > diameter keeps the base lattice overlap-free (resolvable);
// jitter then makes some neighbours (incl. across the periodic faces) overlap, which the solver relaxes.
static std::vector<float> scenePositions(const Scene& s, int& n) {
  std::vector<float> p;
  const double span = (s.G - 1) * s.spacing;
  const double amp = 0.12 * s.spacing;
  const double base[3] = {s.per[0] ? 0.5 * s.spacing : 0.5 * (s.L - span),
                          s.per[1] ? 0.5 * s.spacing : 0.5 * (s.L - span),
                          s.per[2] ? 0.5 * s.spacing : 0.5 * (s.L - span)};
  for (int iz = 0; iz < s.G; ++iz)
    for (int iy = 0; iy < s.G; ++iy)
      for (int ix = 0; ix < s.G; ++ix) {
        const int g = static_cast<int>(p.size() / 3);
        double q[3] = {base[0] + ix * s.spacing + jitter(g, 0, amp),
                       base[1] + iy * s.spacing + jitter(g, 1, amp),
                       base[2] + iz * s.spacing + jitter(g, 2, amp)};
        for (int d = 0; d < 3; ++d)
          if (s.per[d]) { q[d] = std::fmod(q[d], s.L); if (q[d] < 0) q[d] += s.L; }
        p.push_back((float)q[0]); p.push_back((float)q[1]); p.push_back((float)q[2]);
      }
  n = s.G * s.G * s.G;
  return p;
}

static void configure(KokkosSim& sim, const Scene& s) {
  sim.setDomain(s.L, s.L, s.L, s.per[0], s.per[1], s.per[2]);
  sim.setGlobalScale(GS);
  sim.setSphereShape(R);
  sim.setDt(1e-2f);
  sim.setGravity(0, 0, 0);
  sim.setSolverIterations(POS_ITERS, VEL_ITERS);
  sim.setMaterialParams(0.0f, 0.0f, 0.0f);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool periodic = (argc > 1 && std::strcmp(argv[1], "periodic") == 0);
    // periodic = FULLY periodic tiled box: exercises self-ghosts on every undecomposed periodic axis
    // (all three at np=1; z of the np=4 2x2x1 layout; etc.). A periodic box needs a thick ghost
    // boundary layer, so the KokkosSim capacity below is sized generously (gather throws on overflow).
    const Scene scene = periodic ? Scene{{true, true, true}, 5, 11.0, 2.2}
                                 : Scene{{false, false, false}, 4, 14.0, 1.8};
    const int cap = 8 * scene.G * scene.G * scene.G + 64;  // ghost headroom (fully-periodic = thick layer)
    const double L = scene.L;

    int n = 0;
    const std::vector<float> gpos = scenePositions(scene, n);
    const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
    const std::tuple<long, long, long> gsize{GX, GX, GX};
    const std::tuple<bool, bool, bool> per{scene.per[0], scene.per[1], scene.per[2]};

    // --- ownership: which global particles this rank owns (ORB block of the gsize cell grid) ---
    tpx::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) { map.origin[i] = 0; map.cellSize[i] = L / GX; map.periodic[i] = scene.per[i]; }
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
    KokkosSim dist(cap);  // capacity: owned + ghost headroom
    configure(dist, scene);
    dist.setPositions(ownedPos);
    dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
    // Verlet-skin ghost reuse (D2): build the halo topology with a band of RCUT+GS and reuse it across
    // substeps until a particle moves > GS. The result must still match the rebuild-every-substep
    // reference (which keeps skin=0 below) — reuse only changes WHEN the topology is rebuilt.
    dist.enableMpiStep(RCUT, /*sync_every=*/1, /*forward_rotation=*/false, /*rebalance_every=*/0,
                       /*verlet_skin=*/GS);  // spheres
    dist.stepMpi(STEPS);
    const long distRebuilds = dist.mpiRebuilds(), distGathers = dist.mpiGathers();
    const std::vector<float> distPos = dist.getPositions();
    const float distOv = dist.maxOverlap();
    int totGhost = 0, myGhost = dist.numGhost();
    MPI_Allreduce(&myGhost, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // --- single-rank reference (rank 0, MPI_COMM_SELF, ALL particles, the SAME stepMpi code) ---
    std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
    float refOv = 0.0f;
    if (rank == 0) {
      KokkosSim ref(cap);
      configure(ref, scene);
      ref.setPositions(gpos);
      ref.initMpi(origin, dsize, gsize, per, MPI_COMM_SELF);  // size-1 block; periodic via self-ghosts
      ref.enableMpiStep(RCUT, 1, false);
      ref.stepMpi(STEPS);
      refPos = ref.getPositions();
      refOv = ref.maxOverlap();
    }
    MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&refOv, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- compare this rank's owned particles (by global id) against the reference trajectory ---
    // A force-balanced, fully-relaxed packing with no walls/gravity has a rigid-translation gauge
    // freedom (zero-energy mode); a periodic axis makes it an exactly free collective slide. The
    // distributed and reference solves, differing only in atomic-accumulation order, can drift along
    // it. Remove that gauge (subtract the mean displacement per axis) before measuring divergence --
    // the residual is the physically meaningful, gauge-invariant error.
    double lsum[3] = {0, 0, 0};
    auto disp = [&](int i, int c) {
      double d = static_cast<double>(distPos[3 * i + c]) - static_cast<double>(refPos[3 * ownedGid[i] + c]);
      if (scene.per[c]) d -= L * std::round(d / L);  // minimum image (a particle may wrap a face)
      return d;
    };
    for (int i = 0; i < nOwned; ++i) for (int c = 0; c < 3; ++c) lsum[c] += disp(i, c);
    double gsum[3] = {0, 0, 0};
    MPI_Allreduce(lsum, gsum, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double mean[3] = {gsum[0] / n, gsum[1] / n, gsum[2] / n};
    double localMax = 0.0;
    for (int i = 0; i < nOwned; ++i)
      for (int c = 0; c < 3; ++c) localMax = std::max(localMax, std::fabs(disp(i, c) - mean[c]));
    double posErr = 0.0;
    MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double ovErr = std::fabs(static_cast<double>(distOv) - static_cast<double>(refOv));

    // Pass criteria differ by scene because they are constrained differently:
    //  * closed (spacing < diameter, over-constrained): the relaxed packing is essentially unique, so
    //    positions agree tightly. np=1 is the IDENTICAL single block (gap = pure float atomic-add
    //    reorder, ~1e-4); np>1 adds decomposition/ghost reorder (~1e-4..1e-3).
    //  * periodic (spacing > diameter, under-constrained): cage slack (~0.2) admits a manifold of
    //    zero-overlap configs, so dist and ref settle into DIFFERENT ones within the slack -- position
    //    parity is not meaningful. The physical check is that the overlap relaxes identically (proves
    //    the z-wrap contacts are handled by the self-ghosts; a missing-contact bug leaves residual
    //    overlap ~0.2, and the np=1-fully-periodic instability blows it to ~2). posErr only guards
    //    against divergence (the broken case gives ~5, far beyond the cage slack).
    const double posTol = periodic ? 0.30 : (size == 1 ? 1e-3 : 1e-2);
    const bool relaxed = !periodic || (distOv < 1e-3 && refOv < 1e-3);
    if (rank == 0)
      std::printf("  [%-8s] np=%d  particles=%d  ghosts(total)=%d  posErr=%.3e (tol %.0e)  overlap dist=%.4e ref=%.4e d=%.2e\n",
                  periodic ? "periodic" : "closed", size, n, totGhost, posErr, posTol, distOv, refOv, ovErr);
    if (!(posErr < posTol) || !(ovErr < 5e-3) || !relaxed) fail = 1;
    // D2: with a Verlet skin the topology must be reused on at least one substep (fewer rebuilds than
    // gathers) — otherwise the reuse path is never exercised. STEPS=8 gathers, displacement per substep
    // ≪ GS in a relaxing pack, so most substeps reuse.
    if (rank == 0)
      std::printf("  [%-8s] np=%d  halo rebuilds=%ld / gathers=%ld (Verlet-skin reuse)\n",
                  periodic ? "periodic" : "closed", size, distRebuilds, distGathers);
    if (distRebuilds >= distGathers) fail = 1;
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
