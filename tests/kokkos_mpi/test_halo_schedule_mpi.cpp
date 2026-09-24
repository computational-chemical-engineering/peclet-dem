// dem — the distributed step's collective schedule: every rank must make the same sequence of
// point-to-point and collective calls, whatever its local particle layout. Each mode builds a scene
// in which a rank-LOCAL condition differs across ranks and would, if it steered an exchange,
// desynchronise the ranks — the failure is a DEADLOCK, so ctest's TIMEOUT is what turns a
// regression red.
//
//   one_sided_xpbd / one_sided_hertz — an ASYMMETRIC halo: particles sit just inside the upper
//     side of every ORB split plane and far from the lower side, so an upper-side rank owes ghosts
//     to its lower neighbour while receiving none itself (numGhost == 0, sends > 0). The XPBD step
//     (demStepMpi: gather + per-iteration syncs) and the force step (demStepHertzMpi: gather +
//     per-step state refresh) must still send. This is the layout that deadlocked the coupling's
//     test_mpi_moving_suspension at np=4 (a particle row exactly on the y = 16 block face).
//
//   skin_divergent — Verlet-skin ghost reuse with only SOME ranks' particles moving past the
//     skin: the topology rebuild (an NBX collective) must be taken by all ranks or none.
//
//   skin_migrate_reorder — Verlet-skin ghost reuse across an ownership migration that leaves every
//     rank's owned COUNT unchanged but REORDERS its slots: two particles swap owners across the
//     x = 8 face and each lands in the slot the other left, within the skin of that slot's
//     build-time position. The cached topology is slot-indexed over the pre-migration owned set;
//     a migration drops it, so the step after the migration rebuilds (before 2026-09-24 it was
//     reused here -- soundly, as it happens: every slot stayed within the skin -- see
//     ParticleHalo::unpackState).
//
// Every mode also checks the physics against the same step run on MPI_COMM_SELF over the global
// particle set (rank 0, broadcast): overlapping pairs sit wholly inside one block and ghosts never
// touch owned bodies, so the distributed result is the single-rank one.
//
// np = 1, 2, 4. Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "peclet/core/common/types.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "sim.hpp"

using peclet::core::IVec;
using peclet::dem::Simulation;

static constexpr double L = 16.0;  // closed cube; the ORB splits it at 8 (np=2: x; np=4: x, y)
static constexpr int GX = 16;      // ORB cell grid
static constexpr float RAD = 0.5f;

// Two overlapping spheres (overlap 0.1) at every site of a 3x3x2 lattice whose x and y planes are
// {2.0, 8.4, 13.0}: 8.4 is 0.4 above the split plane x|y = 8 (inside every ghost band used below),
// 2.0 and 13.0 are >= 5 away from it, so ghosts flow only from the upper to the lower side. The
// z planes (3.0, 12.0) keep clear of a z split. Returns the global positions (3 per particle).
static std::vector<float> scene(int& n) {
  const float xy[3] = {2.0f, 8.4f, 13.0f};
  const float zs[2] = {3.0f, 12.0f};
  std::vector<float> p;
  for (float z : zs)
    for (float y : xy)
      for (float x : xy)
        for (int k = 0; k < 2; ++k) {
          p.push_back(x);
          p.push_back(y);
          p.push_back(z + 0.9f * static_cast<float>(k));
        }
  n = static_cast<int>(p.size() / 3);
  return p;
}

// Global ids this rank owns under the equal-cell ORB (the decomposition initMpi builds).
static void ownedOf(const std::vector<float>& gpos, int n, int rank, int size,
                    std::vector<float>& ownedPos, std::vector<int>& ownedGid) {
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  peclet::core::halo::DomainMap<3> map;
  for (int i = 0; i < 3; ++i) {
    map.origin[i] = 0;
    map.cellSize[i] = L / GX;
    map.periodic[i] = false;
  }
  peclet::core::halo::ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  for (int g = 0; g < n; ++g) {
    peclet::core::Vec<3> x{gpos[3 * g], gpos[3 * g + 1], gpos[3 * g + 2]};
    if (mig.ownerOf(x) == rank) {
      for (int c = 0; c < 3; ++c)
        ownedPos.push_back(gpos[3 * g + c]);
      ownedGid.push_back(g);
    }
  }
}

enum class Mode { OneSidedXpbd, OneSidedHertz, SkinDivergent };

// skin_migrate_reorder (see the file comment). Cross-face overlapping pairs at y = 4 and y = 12
// (ghosts both ways at every split), plus the swap pair p = (7.8, 4, 10) moving +x and
// q = (8.2, 4, 11.2) moving -x at 40 (0.4 per step; they pass 1.2 apart in z, never touching).
// p and q are the LAST two global ids, hence the last owned slot on their ranks: migration keeps
// the staying particles in order and appends the arrivals, so after step 1 q takes p's slot and p
// takes q's, each 1.2 < skin 1.5 from that slot's build-time position.
static int runSkinMigrate(int rank, int size) {
  const float dt = 1e-2f, SKIN = 1.5f;
  const double RCUT = 1.5;
  const int STEPS = 4;
  std::vector<float> gpos, gvel;
  auto add = [&](float x, float y, float z, float vx) {
    gpos.insert(gpos.end(), {x, y, z});
    gvel.insert(gvel.end(), {vx, 0.0f, 0.0f});
  };
  for (float y : {4.0f, 12.0f})
    for (float z : {4.0f, 13.0f}) {
      add(7.6f, y, z, 0.0f);  // overlapping pair (0.8 apart) straddling x = 8
      add(8.4f, y, z, 0.0f);
    }
  add(7.8f, 4.0f, 10.0f, 40.0f);   // p
  add(8.2f, 4.0f, 11.2f, -40.0f);  // q
  const int n = static_cast<int>(gpos.size() / 3);

  auto configure = [&](Simulation& sim) {
    sim.setDomain(L, L, L, false, false, false);
    sim.setGlobalScale(1.0f);
    sim.setSphereShape(RAD);
    sim.setDt(dt);
    sim.setGravity(0, 0, 0);
    sim.setSolverIterations(20, 4);
    sim.setMaterialParams(0.5f, 0.0f, 0.0f);
  };
  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{false, false, false};
  const std::vector<peclet::core::Real> uniform(static_cast<std::size_t>(GX) * GX * GX, 1.0);

  std::vector<float> ownedPos;
  std::vector<int> ownedGid;
  ownedOf(gpos, n, rank, size, ownedPos, ownedGid);
  std::vector<float> ownedVel;
  for (int g : ownedGid)
    for (int c = 0; c < 3; ++c)
      ownedVel.push_back(gvel[3 * g + c]);
  const int cap = 8 * n + 64;

  Simulation dist(cap);
  configure(dist);
  dist.setPositions(ownedPos);
  dist.setVelocities(ownedVel);
  dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  dist.enableMpiStep(RCUT, 1, true, /*rebalance_every=*/0, /*verlet_skin=*/SKIN);
  int reordered = 0, countChanged = 0;
  long rebuildsAfterMigrateStep = -1;
  for (int s = 0; s < STEPS; ++s) {
    dist.stepMpi(1);
    const std::vector<float> before = dist.getPositions();
    dist.migrateToWeights(uniform);  // the SAME equal-cell ORB: a pure ownership move
    const std::vector<float> after = dist.getPositions();
    if (s == 0) {
      countChanged = before.size() != after.size() ? 1 : 0;
      for (std::size_t i = 0; !countChanged && i < after.size(); ++i)
        if (after[i] != before[i])
          reordered = 1;
      const long r0 = dist.mpiRebuilds();
      dist.stepMpi(1);  // the step that re-gathers right after the reordering migration
      ++s;
      rebuildsAfterMigrateStep = dist.mpiRebuilds() - r0;
    }
  }
  const std::vector<float> distPos = dist.getPositions();

  std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
  if (rank == 0) {
    Simulation ref(cap);
    configure(ref);
    ref.setPositions(gpos);
    ref.setVelocities(gvel);
    ref.initMpi(origin, dsize, gsize, per, MPI_COMM_SELF);
    ref.enableMpiStep(RCUT, 1, true, 0, SKIN);
    for (int s = 0; s < STEPS; ++s) {
      ref.stepMpi(1);
      ref.migrateToWeights(uniform);
    }
    refPos = ref.getPositions();
  }
  MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Ownership moved, so match each owned particle to its nearest reference particle (the scene's
  // particles stay >= 0.8 apart; a lost cross-rank contact moves one by ~0.1).
  double localMax = 0.0;
  const int nOwned = static_cast<int>(distPos.size() / 3);
  for (int i = 0; i < nOwned; ++i) {
    double best = 1e30;
    for (int g = 0; g < n; ++g) {
      double d2 = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double d = static_cast<double>(distPos[3 * i + c]) - refPos[3 * g + c];
        d2 += d * d;
      }
      best = std::min(best, d2);
    }
    localMax = std::max(localMax, std::sqrt(best));
  }
  double posErr = 0.0;
  MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  long lc = nOwned, gc = 0;
  MPI_Allreduce(&lc, &gc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  int anyReorder = 0, anyCountChange = 0;
  MPI_Allreduce(&reordered, &anyReorder, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&countChanged, &anyCountChange, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const double posTol = 1e-5;
  if (rank == 0)
    std::printf(
        "  [skin_migrate_reorder] np=%d particles=%ld/%d reordered=%d count-changed=%d "
        "rebuilds in the post-migration step=%ld posErr=%.3e (tol %.0e)\n",
        size, gc, n, anyReorder, anyCountChange, rebuildsAfterMigrateStep, posErr, posTol);
  int fail = 0;
  if (gc != n || !(posErr < posTol))
    fail = 1;
  // At np >= 2 the scene must do what it is for: a same-count migration that reorders slots.
  if (size > 1 && (!anyReorder || anyCountChange))
    fail = 1;
  // A migration drops the cached topology, so the step after it rebuilds (every rank: the skin
  // decision is global).
  if (rebuildsAfterMigrateStep != 1)
    fail = 1;
  for (float v : distPos)
    if (!std::isfinite(v))
      fail = 1;
  return fail;
}

static int run(Mode mode, int rank, int size) {
  const char* name = mode == Mode::OneSidedXpbd    ? "one_sided_xpbd"
                     : mode == Mode::OneSidedHertz ? "one_sided_hertz"
                                                   : "skin_divergent";
  const bool hertz = mode == Mode::OneSidedHertz;
  const bool skin = mode == Mode::SkinDivergent;
  const float dt = hertz ? 1e-4f : 1e-2f;
  const int STEPS = 8, SUB = 50;
  const double RCUT = 1.5;         // XPBD ghost band (>= contact diameter 1.0 + margin)
  const float VERLET_SKIN = 0.3f;  // skin_divergent only
  const float HERTZ_SKIN = 0.3f;   // force-path pair-list skin fraction (band 2.3 * RAD)

  int n = 0;
  const std::vector<float> gpos = scene(n);
  // skin_divergent: the x = 13 column (only upper-x ranks own it) slides along +y at 10 per unit
  // time -- 0.1 per step, past the 0.3 skin after the third step -- while everything else rests.
  std::vector<float> gvel(static_cast<std::size_t>(n) * 3, 0.0f);
  if (skin)
    for (int g = 0; g < n; ++g)
      if (gpos[3 * g] > 12.0f)
        gvel[3 * g + 1] = 10.0f;

  auto configure = [&](Simulation& sim) {
    sim.setDomain(L, L, L, false, false, false);
    sim.setGlobalScale(1.0f);
    sim.setSphereShape(RAD);
    sim.setDt(dt);
    sim.setGravity(0, 0, 0);
    sim.setSolverIterations(20, 4);
    sim.setMaterialParams(0.5f, 0.0f, 0.0f);
    if (hertz)
      sim.setHertzMaterial(0, 1.0e5f, 0.25f);
  };
  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{false, false, false};

  std::vector<float> ownedPos;
  std::vector<int> ownedGid;
  ownedOf(gpos, n, rank, size, ownedPos, ownedGid);
  const int nOwned = static_cast<int>(ownedGid.size());
  std::vector<float> ownedVel;
  for (int g : ownedGid)
    for (int c = 0; c < 3; ++c)
      ownedVel.push_back(gvel[3 * g + c]);
  const int cap = 4 * n + 64;

  // --- distributed (MPI_COMM_WORLD) ---
  Simulation dist(cap);
  configure(dist);
  dist.setPositions(ownedPos);
  dist.setVelocities(ownedVel);
  dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  dist.enableMpiStep(RCUT, /*sync_every=*/1, /*forward_rotation=*/true, /*rebalance_every=*/0,
                     /*verlet_skin=*/skin ? VERLET_SKIN : 0.0);
  // The first step's ghost count is the layout the test is about (the force path's later calls
  // re-gather on the same static layout; the skin mode's column stays inside its blocks).
  if (hertz)
    dist.stepHertzMpi(SUB, HERTZ_SKIN);
  else
    dist.stepMpi(1);
  int myGhost = dist.numGhost(), minGhost = 0, totGhost = 0;
  MPI_Allreduce(&myGhost, &minGhost, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&myGhost, &totGhost, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int s = 1; s < STEPS; ++s) {
    if (hertz)
      dist.stepHertzMpi(SUB, HERTZ_SKIN);
    else
      dist.stepMpi(1);
  }
  const std::vector<float> distPos = dist.getPositions();
  long myRebuilds = dist.mpiRebuilds(), minRebuilds = 0, maxRebuilds = 0;
  MPI_Allreduce(&myRebuilds, &minRebuilds, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&myRebuilds, &maxRebuilds, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  const long gathers = dist.mpiGathers();

  // --- single-rank reference (rank 0, MPI_COMM_SELF, all particles, the same step code) ---
  std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
  if (rank == 0) {
    Simulation ref(cap);
    configure(ref);
    ref.setPositions(gpos);
    ref.setVelocities(gvel);
    ref.initMpi(origin, dsize, gsize, per, MPI_COMM_SELF);
    ref.enableMpiStep(RCUT, 1, true, 0, skin ? VERLET_SKIN : 0.0);
    for (int s = 0; s < STEPS; ++s) {
      if (hertz)
        ref.stepHertzMpi(SUB, HERTZ_SKIN);
      else
        ref.stepMpi(1);
    }
    refPos = ref.getPositions();
  }
  MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Ownership is fixed (no rebalance) and the particles stay inside their blocks, so the local
  // order is ownedGid's.
  double localMax = 0.0;
  for (int i = 0; i < nOwned; ++i)
    for (int c = 0; c < 3; ++c)
      localMax = std::max(localMax, std::fabs(static_cast<double>(distPos[3 * i + c]) -
                                              static_cast<double>(refPos[3 * ownedGid[i] + c])));
  double posErr = 0.0;
  MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  // The pairs separate by ~0.1 over the run; the scheme only reorders independent pairs.
  const double posTol = 1e-5;

  int fail = 0;
  if (rank == 0)
    std::printf(
        "  [%-15s] np=%d particles=%d ghosts(total)=%d min-ghosts/rank=%d posErr=%.3e "
        "(tol %.0e) halo rebuilds=%ld..%ld / gathers=%ld\n",
        name, size, n, totGhost, minGhost, posErr, posTol, minRebuilds, maxRebuilds, gathers);
  if (!(posErr < posTol))
    fail = 1;
  // The scene must actually exercise the one-sided layout: some rank holds no ghosts while the
  // others do (at np=1 there is no cross-rank halo; closed box, so no self-ghosts either).
  if (size > 1 && !(minGhost == 0 && totGhost > 0))
    fail = 1;
  if (skin) {
    // One schedule for all ranks, and the skin path truly exercised: some gathers reused the
    // topology, and the moving column forced at least one rebuild after the first.
    if (minRebuilds != maxRebuilds || !(maxRebuilds > 1 && maxRebuilds < gathers))
      fail = 1;
  }
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
    const std::string mode = (argc > 1) ? argv[1] : "one_sided_xpbd";
    if (mode == "one_sided_xpbd")
      fail = run(Mode::OneSidedXpbd, rank, size);
    else if (mode == "one_sided_hertz")
      fail = run(Mode::OneSidedHertz, rank, size);
    else if (mode == "skin_divergent")
      fail = run(Mode::SkinDivergent, rank, size);
    else if (mode == "skin_migrate_reorder")
      fail = runSkinMigrate(rank, size);
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
