// dem — the distributed XPBD step's ghost band must cover every contact the narrow phase can make.
//
// A contact across a block face is resolved correctly only if BOTH partners are present on BOTH
// owners (each owner computes its body's full delta from the owned + ghost set). The narrow phase
// reports a pair when the gap is below the broadphase margin, i.e. at centre distance
// < r_i + r_j + margin, so a particle up to 2 R_max + margin from the face can be a partner of a
// body on the other side. Each mode builds a closed-box scene in which one cross-face pair sits
// just inside that range and runs the distributed step against the same step on MPI_COMM_SELF
// over the global particle set (no ghosts there, so no contact can be missed). A missed contact
// resolves the pair on one side only and moves a body by ~1e-2; the tolerance is 1e-4.
//
//   band_change      — the band grows between calls (enable_mpi_step again, a larger Verlet skin):
//                      the cached ghost lists, built with the old band, must not be reused.
//   default_band     — default rcut, polydisperse: a large particle on one rank, a small one
//                      1-2 small radii from the face on the other. The band is the global reach.
//   margin           — polydisperse, a large body on one rank only: the narrow-phase margin must
//                      be the same on every rank (a rank-local 0.1 R_max drops the pair on the
//                      rank whose own grains are small).
//   explicit_rcut    — an explicit rcut = 2r (what coupling passes) is below the reach 2r + margin:
//                      the band must still cover the margin.
//   coax_tubes       — two coaxial hollow cylinders (D 1, H 1.5, wall 0.18, axis x) straddling the
//                      x = 8 face at centre distance 1.47 (end faces 0.03 deep): every rank that
//                      owns a tube must see the pair (contacts > 0) -- the broad phase and band are
//                      sized by the tube's circumscribed radius (docs/contact_physics_followups.md
//                      F1, WO-B0, G-B0); with the geometric radius no rank saw it.
//
// np = 1, 2, 4, 8 (the x = 8 split carries every pair). Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
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

static constexpr double L = 16.0;  // closed cube; every ORB layout used splits x at 8
static constexpr int GX = 16;
static constexpr float RAD = 0.5f;  // base radius (scale 1)

struct Body {
  float x, y, z, vx, scale;
  float q[4] = {0.0f, 0.0f, 0.0f, 1.0f};  // orientation (x, y, z, w); loaded for tube scenes only
};

// Owned bodies of this rank under the equal-cell ORB (the decomposition initMpi builds).
static std::vector<int> ownedOf(const std::vector<Body>& b, int rank, int size) {
  peclet::core::decomp::BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  peclet::core::halo::DomainMap<3> map;
  for (int i = 0; i < 3; ++i) {
    map.origin[i] = 0;
    map.cellSize[i] = L / GX;
    map.periodic[i] = false;
  }
  peclet::core::halo::ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  std::vector<int> gids;
  for (int g = 0; g < static_cast<int>(b.size()); ++g)
    if (mig.ownerOf(peclet::core::Vec<3>{b[g].x, b[g].y, b[g].z}) == rank)
      gids.push_back(g);
  return gids;
}

static void load(Simulation& sim, const std::vector<Body>& b, const std::vector<int>& gids,
                 bool tubes) {
  std::vector<float> p, v, s, q;
  for (int g : gids) {
    p.insert(p.end(), {b[g].x, b[g].y, b[g].z});
    v.insert(v.end(), {b[g].vx, 0.0f, 0.0f});
    s.push_back(b[g].scale);
    q.insert(q.end(), {b[g].q[0], b[g].q[1], b[g].q[2], b[g].q[3]});
  }
  sim.setPositions(p);
  sim.setVelocities(v);
  sim.setScales(s);
  if (tubes)
    sim.setQuaternions(q);
}

// Run `drive` (enable_mpi_step + steps) distributed and on MPI_COMM_SELF; compare by global id.
// tubes: the bodies are hollow cylinders (D 1, H 1.5, wall 0.18) instead of spheres of radius RAD,
// and every rank that owns a body must have seen a contact in the distributed run.
static int runCase(const char* name, const std::vector<Body>& bodies,
                   const std::function<void(Simulation&)>& drive, int rank, int size,
                   bool tubes = false) {
  const int n = static_cast<int>(bodies.size());
  const int cap = 16 * n + 64;
  auto configure = [&](Simulation& sim) {
    sim.setDomain(L, L, L, false, false, false);
    sim.setGlobalScale(1.0f);
    if (tubes)
      sim.initializeShape(peclet::dem::HOLLOW_CYLINDER, 0.5f, 1.5f, 0.18f);
    else
      sim.setSphereShape(RAD);
    sim.setDt(1e-2f);
    sim.setGravity(0, 0, 0);
    sim.setSolverIterations(20, 4);
    sim.setMaterialParams(0.0f, 0.0f, 0.0f);
  };
  const std::tuple<double, double, double> origin{0, 0, 0}, dsize{L, L, L};
  const std::tuple<long, long, long> gsize{GX, GX, GX};
  const std::tuple<bool, bool, bool> per{false, false, false};

  const std::vector<int> gids = ownedOf(bodies, rank, size);
  Simulation dist(cap);
  configure(dist);
  load(dist, bodies, gids, tubes);
  dist.initMpi(origin, dsize, gsize, per, MPI_COMM_WORLD);
  drive(dist);
  const std::vector<float> distPos = dist.getPositions();
  // A rank that owns a body of the (single) pair must see it: its visible contacts are non-empty.
  const int blind = (tubes && !gids.empty() && dist.numContacts() == 0) ? 1 : 0;
  int blindRanks = 0;
  MPI_Allreduce(&blind, &blindRanks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  const long rebuilds = dist.mpiRebuilds();

  std::vector<float> refPos(static_cast<std::size_t>(n) * 3, 0.0f);
  if (rank == 0) {
    std::vector<int> all(n);
    for (int g = 0; g < n; ++g)
      all[g] = g;
    Simulation ref(cap);
    configure(ref);
    load(ref, bodies, all, tubes);
    ref.initMpi(origin, dsize, gsize, per, MPI_COMM_SELF);
    drive(ref);
    refPos = ref.getPositions();
  }
  MPI_Bcast(refPos.data(), n * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Ownership is fixed (no migration), so the local order is gids'.
  double localMax = 0.0;
  for (std::size_t i = 0; i < gids.size(); ++i)
    for (int c = 0; c < 3; ++c)
      localMax = std::max(localMax, std::fabs(static_cast<double>(distPos[3 * i + c]) -
                                              static_cast<double>(refPos[3 * gids[i] + c])));
  double posErr = 0.0;
  MPI_Allreduce(&localMax, &posErr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  long maxRebuilds = 0;
  MPI_Allreduce(&rebuilds, &maxRebuilds, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  const double posTol = 1e-4;
  // Centre-of-mass x shift over the run (every body has unit mass, no body moves initially
  // except band_change's q): a conserving solve leaves it where the initial velocities carry it.
  double lcom = 0.0, gcom = 0.0, com0 = 0.0, comRef = 0.0;
  for (std::size_t i = 0; i < gids.size(); ++i)
    lcom += distPos[3 * i];
  MPI_Allreduce(&lcom, &gcom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int g = 0; g < n; ++g) {
    com0 += bodies[g].x / n;
    comRef += refPos[3 * g] / n;
  }
  if (rank == 0)
    std::printf(
        "  [%-14s] np=%d particles=%d posErr=%.3e (tol %.0e) halo rebuilds=%ld "
        "comShift dist=%.3e ref=%.3e\n",
        name, size, n, posErr, posTol, maxRebuilds, gcom / n - com0, comRef - com0);
  if (tubes && rank == 0)
    std::printf("  [%-14s] owner ranks that saw no contact: %d\n", name, blindRanks);
  int fail = (!(posErr < posTol) || blindRanks > 0) ? 1 : 0;
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
    const std::string mode = (argc > 1) ? argv[1] : "band_change";
    const float y = 4.0f, z = 4.0f;
    if (mode == "band_change") {
      // r = 0.5 everywhere (reach 2r + margin = 1.05). p rests 0.1 left of the face; q starts 1.2
      // right of it moving -x at 5 (0.05 per step). The first call builds the ghost lists with
      // band 1.1 + skin 0.05 = 1.15, which leaves q out (no contact yet: gap 0.3). The second
      // enable_mpi_step raises the skin to 0.5 (band 1.6); q then closes to contact within 0.45
      // of its build position -- under the new skin, so only the band change can force the
      // rebuild that brings q to p's owner.
      // One resting filler at the centre of every octant keeps every rank non-empty up to np=8
      // (an empty rank always votes for a rebuild, which would mask the stale band).
      std::vector<Body> b{{7.9f, y, z, 0.0f, 1.0f}, {9.2f, y, z, -5.0f, 1.0f}};
      for (float fx : {4.0f, 12.0f})
        for (float fy : {4.0f, 12.0f})
          for (float fz : {4.0f, 12.0f})
            b.push_back({fx, fy, fz, 0.0f, 1.0f});
      fail = runCase(
          "band_change", b,
          [](Simulation& s) {
            s.enableMpiStep(1.1, 1, true, 0, /*verlet_skin=*/0.05);
            s.stepMpi(1);
            s.enableMpiStep(1.1, 1, true, 0, /*verlet_skin=*/0.5);
            s.stepMpi(8);
          },
          rank, size);
    } else if (mode == "default_band") {
      // p: scale 2 (r = 1.0), 0.3 left of the face. q: r = 0.5, 1.0 right of it (two of its own
      // radii), overlapping p by 0.2. A band of one rank-LOCAL radius (q's owner: 0.5) never sends
      // q to p's owner.
      const std::vector<Body> b{{7.7f, y, z, 0.0f, 2.0f}, {9.0f, y, z, 0.0f, 1.0f}};
      fail = runCase(
          "default_band", b,
          [](Simulation& s) {
            s.enableMpiStep(/*rcut: default*/ 0.0, 1, true);
            s.stepMpi(4);
          },
          rank, size);
    } else if (mode == "margin") {
      // p: r = 1.0, 0.3 left of the face (its owner's margin 0.1 R_max = 0.1). q: r = 0.5, gap
      // 0.07 to p; s: r = 0.5, overlapping q by 0.3 on the far side. The position solve pushes q
      // into p. q's owner (grains of r = 0.5 only) had margin 0.05 < 0.07: it dropped the p-q
      // pair that p's owner kept.
      const std::vector<Body> b{
          {7.7f, y, z, 0.0f, 2.0f}, {9.27f, y, z, 0.0f, 1.0f}, {9.97f, y, z, 0.0f, 1.0f}};
      fail = runCase(
          "margin", b,
          [](Simulation& s) {
            s.enableMpiStep(0.0, 1, true);
            s.stepMpi(4);
          },
          rank, size);
    } else if (mode == "explicit_rcut") {
      // r = 0.5 everywhere, rcut = 2r = 1.0 (the coupling's choice), reach 1.05. p 0.02 left of
      // the face; q gap 0.03 to p (1.01 right of the face, outside a band of 1.0); s overlaps q by
      // 0.3 on the far side and pushes it into p.
      const std::vector<Body> b{
          {7.98f, y, z, 0.0f, 1.0f}, {9.01f, y, z, 0.0f, 1.0f}, {9.71f, y, z, 0.0f, 1.0f}};
      fail = runCase(
          "explicit_rcut", b,
          [](Simulation& s) {
            s.enableMpiStep(1.0, 1, true);
            s.stepMpi(4);
          },
          rank, size);
    } else if (mode == "coax_tubes") {
      // Axis x (body y rotated by -90 degrees about z), centres 0.735 either side of x = 8: the
      // end faces overlap by 1.5 - 1.47 = 0.03, a contact only the circumscribed radius (0.901)
      // brings into the broad phase (the geometric 0.5 + margin never reached the partner).
      const float h = std::sqrt(0.5f);
      Body a{8.0f - 0.735f, y, z, 0.0f, 1.0f}, b{8.0f + 0.735f, y, z, 0.0f, 1.0f};
      for (Body* t : {&a, &b}) {
        t->q[2] = -h;
        t->q[3] = h;
      }
      fail = runCase(
          "coax_tubes", {a, b},
          [](Simulation& s) {
            s.enableMpiStep(0.0, 1, true);
            s.stepMpi(1);
          },
          rank, size, /*tubes=*/true);
    } else {
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
