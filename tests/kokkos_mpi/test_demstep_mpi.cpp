// dem — distributed (MPI) demStep validation.
//
// Two families of scenes (argv[1]):
//
//   jacobi_closed / jacobi_periodic — the EXACT-REDUNDANT baseline: velocityUseGS=false routes the
//     distributed step through the legacy count-averaged Jacobi solves, whose ghost-pair handling
//     is exactly redundant (each rank computes the full serial delta of its owned bodies).
//     Reference = the SAME stepMpi code on MPI_COMM_SELF (one block spanning the whole domain);
//     gravity/materials off, tight tolerances (the historical demstep_mpi test).
//
//   modern / modern_rebal — the MODERN solver stack under MPI: gravity + per-pair materials +
//     friction ON, so the distributed step runs the full sequence (rank-local colored Gauss-
//     Seidel, warm-started PGS with gid-keyed persistent contacts, grounded statics /
//     stabilization, friction cone, colored-GS position solve, Allreduce'd adaptive stops).
//     Reference = the REAL single-GPU demStep (Simulation::step) over the global particle set,
//     computed on rank 0 and broadcast. The distributed scheme is processor-block Gauss-Seidel:
//     same fixed point, NOT bit-exact (rank-local sweep order + syncEvery lag), so the check is
//     tolerance-based on the settled positions. modern_rebal additionally re-decomposes +
//     migrates ownership every few steps (rebalance_every), exercising the persistent-ledger /
//     grounded-level carry in MigratePack mid-run.
//
// np = 1, 2, 4. Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <string>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "sim.hpp"

using peclet::core::IVec;
using peclet::dem::Simulation;

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

// Deterministic per-index jitter (a cheap hash -> [-amp,amp]) breaks the lattice symmetry.
static double jitter(int g, int c, double amp) {
  unsigned h = static_cast<unsigned>(g) * 2654435761u + static_cast<unsigned>(c) * 40503u + 1u;
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return (static_cast<double>(h & 0xffff) / 65535.0 - 0.5) * 2.0 * amp;
}

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
          if (s.per[d]) {
            q[d] = std::fmod(q[d], s.L);
            if (q[d] < 0)
              q[d] += s.L;
          }
        p.push_back((float)q[0]);
        p.push_back((float)q[1]);
        p.push_back((float)q[2]);
      }
  n = s.G * s.G * s.G;
  return p;
}

// ---- ownership: which global particles this rank owns (ORB block of the gsize cell grid) ----
static void ownedOf(const std::vector<float>& gpos, int n, const double boxSize[3], bool per[3],
                    int rank, int size, std::vector<float>& ownedPos, std::vector<int>& ownedGid) {
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  peclet::core::halo::DomainMap<3> map;
  for (int i = 0; i < 3; ++i) {
    map.origin[i] = 0;
    map.cellSize[i] = boxSize[i] / GX;
    map.periodic[i] = per[i];
  }
  peclet::core::halo::ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  for (int g = 0; g < n; ++g) {
    peclet::core::Vec<3> x{gpos[3 * g], gpos[3 * g + 1], gpos[3 * g + 2]};
    if (mig.ownerOf(x) == rank) {
      ownedPos.push_back(gpos[3 * g]);
      ownedPos.push_back(gpos[3 * g + 1]);
      ownedPos.push_back(gpos[3 * g + 2]);
      ownedGid.push_back(g);
    }
  }
}

// ================================ jacobi_* (exact-redundant baseline) ========================
static int runJacobi(bool periodic, int rank, int size) {
  const Scene scene = periodic ? Scene{{true, true, true}, 5, 11.0, 2.2}
                               : Scene{{false, false, false}, 4, 14.0, 1.8};
  const int cap = 8 * scene.G * scene.G * scene.G + 64;
  const double L = scene.L;

  int n = 0;
  const std::vector<float> gpos = scenePositions(scene, n);
  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{scene.per[0], scene.per[1], scene.per[2]};

  auto configure = [&](Simulation& sim) {
    sim.setDomain(L, L, L, scene.per[0], scene.per[1], scene.per[2]);
    sim.setGlobalScale(GS);
    sim.setSphereShape(R);
    sim.setDt(1e-2f);
    sim.setGravity(0, 0, 0);
    sim.setSolverIterations(POS_ITERS, VEL_ITERS);
    sim.setMaterialParams(0.0f, 0.0f, 0.0f);
    sim.setVelocityUseGS(false);  // legacy count-averaged Jacobi = the exact-redundant scheme
  };

  std::vector<float> ownedPos;
  std::vector<int> ownedGid;
  bool perArr[3] = {scene.per[0], scene.per[1], scene.per[2]};
  const double boxArr[3] = {L, L, L};
  ownedOf(gpos, n, boxArr, perArr, rank, size, ownedPos, ownedGid);
  const int nOwned = static_cast<int>(ownedGid.size());

  // --- distributed solve (MPI_COMM_WORLD), Verlet-skin ghost reuse on (D2) ---
  Simulation dist(cap);
  configure(dist);
  dist.setPositions(ownedPos);
  dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  dist.enableMpiStep(RCUT, /*sync_every=*/1, /*forward_rotation=*/false, /*rebalance_every=*/0,
                     /*verlet_skin=*/GS);
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
    Simulation ref(cap);
    configure(ref);
    ref.setPositions(gpos);
    ref.initMpi(origin, dsize, gsize, per, MPI_COMM_SELF);
    ref.enableMpiStep(RCUT, 1, false);
    ref.stepMpi(STEPS);
    refPos = ref.getPositions();
    refOv = ref.maxOverlap();
  }
  MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&refOv, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Compare owned particles by global id, modulo the rigid-translation gauge (see the historical
  // test): subtract the mean displacement per axis; minimum image on periodic axes.
  double lsum[3] = {0, 0, 0};
  auto disp = [&](int i, int c) {
    double d =
        static_cast<double>(distPos[3 * i + c]) - static_cast<double>(refPos[3 * ownedGid[i] + c]);
    if (scene.per[c])
      d -= L * std::round(d / L);
    return d;
  };
  for (int i = 0; i < nOwned; ++i)
    for (int c = 0; c < 3; ++c)
      lsum[c] += disp(i, c);
  double gsum[3] = {0, 0, 0};
  MPI_Allreduce(lsum, gsum, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double mean[3] = {gsum[0] / n, gsum[1] / n, gsum[2] / n};
  double localMax = 0.0;
  for (int i = 0; i < nOwned; ++i)
    for (int c = 0; c < 3; ++c)
      localMax = std::max(localMax, std::fabs(disp(i, c) - mean[c]));
  double posErr = 0.0;
  MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  const double ovErr = std::fabs(static_cast<double>(distOv) - static_cast<double>(refOv));

  // closed (over-constrained): essentially unique packing -> tight parity. periodic
  // (under-constrained): cage slack -> overlap parity is the physical check, posErr only guards
  // divergence.
  const double posTol = periodic ? 0.30 : (size == 1 ? 1e-3 : 1e-2);
  const bool relaxed = !periodic || (distOv < 1e-3 && refOv < 1e-3);
  int fail = 0;
  if (rank == 0)
    std::printf(
        "  [jacobi_%-8s] np=%d particles=%d ghosts(total)=%d posErr=%.3e (tol %.0e) overlap "
        "dist=%.4e ref=%.4e d=%.2e\n",
        periodic ? "periodic" : "closed", size, n, totGhost, posErr, posTol, distOv, refOv, ovErr);
  if (!(posErr < posTol) || !(ovErr < 5e-3) || !relaxed)
    fail = 1;
  if (rank == 0)
    std::printf("  [jacobi_%-8s] np=%d halo rebuilds=%ld / gathers=%ld (Verlet-skin reuse)\n",
                periodic ? "periodic" : "closed", size, distRebuilds, distGathers);
  if (distRebuilds >= distGathers)
    fail = 1;
  return fail;
}

// ==================================== modern (full stack) ====================================
static int runModern(bool rebal, int rank, int size) {
  // Gravity settle onto a floor plane, periodic x/y (self-ghost coverage on undecomposed axes,
  // cross-rank wrap on decomposed ones), two materials with distinct pair rows, friction on.
  const double L = 12.0, H = 24.0;
  const float dt = 5e-3f;
  const int NSTEP = 40;
  const int GXY = 6, GZ = 8;  // 6x6 tiling (spacing 2 tiles L=12), 8 layers
  const int n = GXY * GXY * GZ;
  const double spacing = 2.0;

  std::vector<float> gpos;
  std::vector<int> gmat;
  for (int iz = 0; iz < GZ; ++iz)
    for (int iy = 0; iy < GXY; ++iy)
      for (int ix = 0; ix < GXY; ++ix) {
        const int g = static_cast<int>(gpos.size() / 3);
        double x = 1.0 + ix * spacing + jitter(g, 0, 0.15);
        double y = 1.0 + iy * spacing + jitter(g, 1, 0.15);
        double z = 2.2 + iz * 2.05 + jitter(g, 2, 0.1);
        x = std::fmod(x + L, L);
        y = std::fmod(y + L, L);
        gpos.push_back((float)x);
        gpos.push_back((float)y);
        gpos.push_back((float)z);
        gmat.push_back(g % 2);
      }

  auto configure = [&](Simulation& sim) {
    sim.setDomain(L, L, H, true, true, false);
    sim.setGlobalScale(1.0f);
    sim.setSphereShape(1.0f);
    sim.setDt(dt);
    sim.setGravity(0, 0, -10.0f);
    sim.setSolverIterations(10, 8);
    sim.setMaterialParams(0.3f, 0.0f, 0.4f);  // global (e, beta, mu)
    sim.setPairMaterial(0, 0, 0.30f, 0.40f);
    sim.setPairMaterial(0, 1, 0.10f, 0.20f);
    sim.setPairMaterial(1, 1, 0.40f, 0.50f);
    sim.addPlane(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f);  // floor at z=1
  };

  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, H};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{true, true, false};
  bool perArr[3] = {true, true, false};

  std::vector<float> ownedPos;
  std::vector<int> ownedGid;
  const double boxArr[3] = {L, L, H};
  ownedOf(gpos, n, boxArr, perArr, rank, size, ownedPos, ownedGid);
  const int nOwned = static_cast<int>(ownedGid.size());
  const int cap = 8 * n + 64;

  // --- distributed run (the full modern stack through demSolveContacts + MpiSolveHooks) ---
  Simulation dist(cap);
  configure(dist);
  dist.setPositions(ownedPos);
  {
    std::vector<int> mats(nOwned);
    for (int i = 0; i < nOwned; ++i)
      mats[i] = gmat[(std::size_t)ownedGid[i]];
    dist.setMaterialIds(mats);
  }
  dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  dist.enableMpiStep(RCUT, /*sync_every=*/1, /*forward_rotation=*/true,
                     /*rebalance_every=*/rebal ? 10 : 0);
  dist.stepMpi(NSTEP);
  const std::vector<float> distPos = dist.getPositions();
  // NOTE: after a rebalance the owned set changed — re-derive the gid list from the ledger?
  // Ownership moved WITH the particles, so map positions back by nearest reference particle is
  // fragile; instead modern_rebal only checks global invariants + the aggregate profile below.
  int totGhost = 0, myGhost = dist.numGhost();
  MPI_Allreduce(&myGhost, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  long lc = dist.numParticles(), gc = 0;
  MPI_Allreduce(&lc, &gc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  // --- reference: the REAL single-GPU demStep over the global set (rank 0, broadcast) ---
  std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
  if (rank == 0) {
    Simulation ref(cap);
    configure(ref);
    ref.setPositions(gpos);
    ref.setMaterialIds(gmat);
    for (int s = 0; s < NSTEP; ++s)
      ref.step(dt);
    refPos = ref.getPositions();
  }
  MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int fail = 0;
  double posErr = 0.0;
  if (!rebal) {
    // Per-particle comparison by global id (ownership fixed => local order still matches
    // ownedGid). Tolerance-based: same fixed point, different sweep order.
    double localMax = 0.0;
    for (int i = 0; i < nOwned; ++i)
      for (int c = 0; c < 3; ++c) {
        double d = static_cast<double>(distPos[3 * i + c]) -
                   static_cast<double>(refPos[3 * ownedGid[i] + c]);
        if (c < 2)
          d -= L * std::round(d / L);  // periodic minimum image (x, y)
        localMax = std::max(localMax, std::fabs(d));
      }
    MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }
  // Aggregate settle profile (both configs): center-of-mass height + max height vs reference —
  // gauge-free, survives ownership migration.
  double lzsum = 0.0, lzmax = 0.0;
  for (int i = 0; i < (int)(distPos.size() / 3); ++i) {
    lzsum += distPos[3 * i + 2];
    lzmax = std::max(lzmax, (double)distPos[3 * i + 2]);
  }
  double gzsum = 0.0, gzmax = 0.0;
  MPI_Allreduce(&lzsum, &gzsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lzmax, &gzmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double rzsum = 0.0, rzmax = 0.0;
  for (int g = 0; g < n; ++g) {
    rzsum += refPos[3 * g + 2];
    rzmax = std::max(rzmax, (double)refPos[3 * g + 2]);
  }
  const double comErr = std::fabs(gzsum / n - rzsum / n);
  const double topErr = std::fabs(gzmax - rzmax);

  const double posTol = 0.30;  // fraction of a diameter (r=1): transient sweep-order divergence
  const double comTol = 0.05, topTol = 0.5;
  if (rank == 0)
    std::printf(
        "  [modern%-6s] np=%d particles=%ld/%d ghosts=%d posErr=%.3e (tol %.2f) comErr=%.3e "
        "topErr=%.3e\n",
        rebal ? "_rebal" : "", size, gc, n, totGhost, posErr, posTol, comErr, topErr);
  if (gc != n)
    fail = 1;
  if (!rebal && !(posErr < posTol))
    fail = 1;
  if (!(comErr < comTol) || !(topErr < topTol))
    fail = 1;
  for (float v : distPos)
    if (!std::isfinite(v))
      fail = 1;
  return fail;
}

// ============================ hertz (force-based engine, MPI) ================================
static int runHertz(bool rebal, int rank, int size) {
  // Explicit Hertz-Mindlin settle onto an SDF floor: closed box, two materials with distinct pair
  // rows, soft E so dt = 1e-4 is Rayleigh-stable. Reference = the REAL single-rank step_hertz.
  const double L = 12.0, H = 24.0;
  const float dt = 1e-4f;
  const int CALLS = 5, SUB = 100;
  const int GXY = 6, GZL = 4;
  const int n = GXY * GXY * GZL;

  // Overlapping lattice (spacing 1.15 < diameter 1.2) with the bottom layer penetrating the SDF
  // floor: pair AND wall contacts are live from step one (a spaced lattice would only validate
  // free flight over this horizon), and the lateral overlaps put active pairs across every rank
  // boundary.
  std::vector<float> gpos;
  std::vector<int> gmat;
  for (int iz = 0; iz < GZL; ++iz)
    for (int iy = 0; iy < GXY; ++iy)
      for (int ix = 0; ix < GXY; ++ix) {
        const int g = static_cast<int>(gpos.size() / 3);
        gpos.push_back((float)(1.0 + ix * 1.15 + jitter(g, 0, 0.06)));
        gpos.push_back((float)(1.0 + iy * 1.15 + jitter(g, 1, 0.06)));
        gpos.push_back((float)(1.55 + iz * 1.15 + jitter(g, 2, 0.03)));
        gmat.push_back(g % 2);
      }

  auto configure = [&](Simulation& sim) {
    sim.setDomain(L, L, H, false, false, false);
    sim.setGlobalScale(1.0f);
    sim.setSphereShape(0.6f);
    sim.setDt(dt);
    sim.setGravity(0, 0, -10.0f);
    sim.setMaterialParams(0.5f, 0.0f, 0.4f);
    sim.setHertzMaterial(0, 1.0e5f, 0.25f);
    sim.setHertzMaterial(1, 1.0e5f, 0.25f);
    sim.setPairMaterial(0, 0, 0.50f, 0.40f);
    sim.setPairMaterial(0, 1, 0.30f, 0.20f);
    sim.setPairMaterial(1, 1, 0.60f, 0.50f);
    // SDF floor: solid below z = 1 (negative inside), coarse trilinear grid over the box.
    const int gn = 9;
    std::vector<float> grid((std::size_t)gn * gn * gn);
    for (int k = 0; k < gn; ++k)
      for (int j = 0; j < gn; ++j)
        for (int i = 0; i < gn; ++i) {
          const float z = -1.0f + (float)k * (float)(L + 2.0) / (gn - 1);
          grid[(std::size_t)i + (std::size_t)j * gn + (std::size_t)k * gn * gn] = z - 1.0f;
        }
    const float sp = (float)(L + 2.0) / (gn - 1);
    sim.addSdfWall(grid, gn, gn, gn, peclet::dem::F3{-1.0f, -1.0f, -1.0f},
                   peclet::dem::F3{sp, sp, sp}, 0.4f, 0.35f);
  };

  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, H};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{false, false, false};
  bool perArr[3] = {false, false, false};
  const double boxArr[3] = {L, L, H};

  std::vector<float> ownedPos;
  std::vector<int> ownedGid;
  ownedOf(gpos, n, boxArr, perArr, rank, size, ownedPos, ownedGid);
  const int nOwned = static_cast<int>(ownedGid.size());
  const int cap = 4 * n + 64;

  Simulation dist(cap);
  configure(dist);
  dist.setPositions(ownedPos);
  {
    std::vector<int> mats(nOwned);
    for (int i = 0; i < nOwned; ++i)
      mats[i] = gmat[(std::size_t)ownedGid[i]];
    dist.setMaterialIds(mats);
  }
  dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  dist.enableMpiStep(/*rcut (unused by the force path)*/ 2.0, 1, true,
                     /*rebalance_every=*/rebal ? 2 : 0);
  for (int c = 0; c < CALLS; ++c)
    dist.stepHertzMpi(dt, SUB, 0.3f);
  const std::vector<float> distPos = dist.getPositions();
  long lc = dist.numParticles(), gc = 0;
  MPI_Allreduce(&lc, &gc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  int totGhost = 0, myGhost = dist.numGhost();
  MPI_Allreduce(&myGhost, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  // Reference: the REAL single-GPU force engine over the global set (rank 0, broadcast).
  std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
  if (rank == 0) {
    Simulation ref(cap);
    configure(ref);
    ref.setPositions(gpos);
    ref.setMaterialIds(gmat);
    for (int c = 0; c < CALLS; ++c)
      ref.stepHertz(dt, SUB, 0.3f);
    refPos = ref.getPositions();
  }
  MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int fail = 0;
  double posErr = 0.0;
  if (!rebal) {  // ownership fixed => local order matches ownedGid
    double localMax = 0.0;
    for (int i = 0; i < nOwned; ++i)
      for (int c = 0; c < 3; ++c)
        localMax = std::max(
            localMax, std::fabs((double)distPos[3 * i + c] - (double)refPos[3 * ownedGid[i] + c]));
    MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }
  double lzsum = 0.0, lzmax = 0.0;
  for (int i = 0; i < (int)(distPos.size() / 3); ++i) {
    lzsum += distPos[3 * i + 2];
    lzmax = std::max(lzmax, (double)distPos[3 * i + 2]);
  }
  double gzsum = 0.0, gzmax = 0.0;
  MPI_Allreduce(&lzsum, &gzsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lzmax, &gzmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double rzsum = 0.0, rzmax = 0.0;
  for (int g = 0; g < n; ++g) {
    rzsum += refPos[3 * g + 2];
    rzmax = std::max(rzmax, (double)refPos[3 * g + 2]);
  }
  const double comErr = std::fabs(gzsum / n - rzsum / n);
  const double topErr = std::fabs(gzmax - rzmax);

  // Explicit force dynamics: rank-local atomic-order roundoff only, but a stiff contact ODE
  // amplifies it — tolerance sized to a small fraction of the radius (0.5).
  const double posTol = 0.05, comTol = 0.01, topTol = 0.1;
  if (rank == 0)
    std::printf(
        "  [hertz%-6s] np=%d particles=%ld/%d ghosts=%d posErr=%.3e (tol %.2f) comErr=%.3e "
        "topErr=%.3e\n",
        rebal ? "_rebal" : "", size, gc, n, totGhost, posErr, posTol, comErr, topErr);
  if (gc != n)
    fail = 1;
  if (!rebal && !(posErr < posTol))
    fail = 1;
  if (!(comErr < comTol) || !(topErr < topTol))
    fail = 1;
  for (float v : distPos)
    if (!std::isfinite(v))
      fail = 1;
  return fail;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int fail = 0, size = 1, rank = 0;
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::string mode = (argc > 1) ? argv[1] : "jacobi_closed";
    if (mode == "jacobi_closed")
      fail = runJacobi(false, rank, size);
    else if (mode == "jacobi_periodic")
      fail = runJacobi(true, rank, size);
    else if (mode == "modern")
      fail = runModern(false, rank, size);
    else if (mode == "modern_rebal")
      fail = runModern(true, rank, size);
    else if (mode == "hertz")
      fail = runHertz(false, rank, size);
    else if (mode == "hertz_rebal")
      fail = runHertz(true, rank, size);
    else {
      if (rank == 0)
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
      fail = 1;
    }
  }
  int totalFail = 0;
  MPI_Allreduce(&fail, &totalFail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (totalFail == 0)
      std::printf("OK (np=%d)\n", size);
    else
      std::fprintf(stderr, "FAILED (np=%d)\n", size);
  }
  Kokkos::finalize();
  MPI_Finalize();
  return totalFail == 0 ? 0 : 1;
}
