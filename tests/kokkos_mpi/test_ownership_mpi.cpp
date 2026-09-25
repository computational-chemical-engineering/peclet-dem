// dem — the owner-exclusive contact scheme's building blocks (docs/mpi_momentum_conservation.md):
// the ghost -> owner reconciliation of ParticleHalo (G6) on a directly constructed Particles +
// ParticleHalo, as in test_migrate_mpi.cpp, and the contact-ownership rule (G5).
//
// Modes (argv[1]):
//   reverse_closed    closed box: cross-rank ghosts only (none at np = 1)
//   reverse_periodic  periodic on every axis: periodic self-ghosts appear at np = 1 (and on every
//                     undecomposed axis), cross-rank wraps at np >= 2
//   exactly_once_closed    a dense polydisperse lattice in a closed box
//   exactly_once_periodic  the same lattice, periodic on every axis (self-image twins at np = 1)
//   exactly_once_drift     closed; after the distribution (no migration) every particle within
//                          1 R of an interior face of its owner's block is displaced 1.0 R across
//                          the nearest such face into the neighbour's block, skipping any with an
//                          already-displaced particle within reach (greedy in gid order). Pairs
//                          then exist that only ONE owner sees (§1.2); a lower-gid-only rule drops
//                          them. np = 1 has no interior face, so there it equals closed.
//
// exactly_once_*: each rank gathers its ghosts, runs the broad + narrow phase of the distributed
// step and partitionContactsKokkos with ParticleHalo::contactOwnership; every rank's owned ACTIVE
// (dist <= 0) contacts, keyed by pairKeyFromGids, are gathered to rank 0 and compared with the
// same narrow phase on the whole set on MPI_COMM_SELF: no key may be owned twice, and the union
// of the owned keys must equal the serial active set.
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

#include <algorithm>
#include <map>

#include "mpi_halo.hpp"
#include "particles.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "sim.hpp"

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

// ---- exactly-once ownership (G5) ----
static constexpr float kRad = 0.5f;  // base radius; scales 1 +- 0.1 (the lattice spacing is 1)

// Simulation with its Particles exposed (the test drives the distributed step's phases by hand).
struct ProbeSim : peclet::dem::Simulation {
  using peclet::dem::Simulation::Simulation;
  Particles& parts() { return P_; }
};

struct Scene {
  std::vector<std::array<float, 3>> x;
  std::vector<float> scale;
  std::vector<int> owner;  // ORB owner at the ORIGINAL position (the partition is fixed)
};
static Scene makeScene(int size, bool drift) {
  std::mt19937 rng(777u);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  Scene sc;
  // drift: strong jitter, so a displaced particle finds partners beyond the band of its owner's
  // block (the one-sided pairs of §1.2), not only its lattice neighbours. closed / periodic: weak
  // jitter. (With the strong jitter in the PERIODIC box, np 2 and 4 lose 2-3 corner-wrap pairs
  // that NO rank reports -- a pair wrapping across a decomposed AND an undecomposed periodic axis
  // at once needs a second image of the partner on the same destination rank, and core's halo
  // keeps one. That gap predates the ownership rule and is outside it; see AFTER.md.)
  const float jitter = drift ? 0.3f : 0.04f;
  for (int k = 0; k < GX; ++k)
    for (int j = 0; j < GX; ++j)
      for (int i = 0; i < GX; ++i) {
        const int idx[3] = {i, j, k};
        std::array<float, 3> q;
        for (int d = 0; d < 3; ++d)
          q[d] = static_cast<float>(idx[d]) + 0.5f + jitter * uni(rng);
        sc.x.push_back(q);
        sc.scale.push_back(1.0f + 0.1f * uni(rng));
        sc.owner.push_back(
            static_cast<int>(dec.ownerOf(IVec<3>{(long)q[0], (long)q[1], (long)q[2]})));
      }
  if (!drift)
    return sc;
  const double reach = 2.1 * 1.1 * kRad;  // xpbdContactReach of the largest grain
  std::vector<std::array<float, 3>> moved;
  for (std::size_t g = 0; g < sc.x.size(); ++g) {
    const auto b = dec.block(static_cast<std::size_t>(sc.owner[g]));
    int bestAxis = -1;
    float bestDist = kRad, bestSign = 0.0f;
    for (int d = 0; d < 3; ++d) {
      const float lo = static_cast<float>(b.origin[d]), hi = static_cast<float>(b.origin[d] + b.size[d]);
      if (b.origin[d] > 0 && sc.x[g][d] - lo < bestDist) {  // interior low face
        bestDist = sc.x[g][d] - lo;
        bestAxis = d;
        bestSign = -1.0f;
      }
      if (b.origin[d] + b.size[d] < GX && hi - sc.x[g][d] < bestDist) {  // interior high face
        bestDist = hi - sc.x[g][d];
        bestAxis = d;
        bestSign = 1.0f;
      }
    }
    if (bestAxis < 0)
      continue;
    std::array<float, 3> q = sc.x[g];
    q[bestAxis] += bestSign * kRad;
    bool clash = false;
    for (const auto& m : moved) {
      const double dx = q[0] - m[0], dy = q[1] - m[1], dz = q[2] - m[2];
      if (dx * dx + dy * dy + dz * dz < reach * reach) {
        clash = true;
        break;
      }
    }
    if (clash)
      continue;
    sc.x[g] = q;
    moved.push_back(q);
  }
  return sc;
}

// Run gather + broad/narrow phase + partition on `comm` for the bodies `mine`; return the owned
// active keys and (for the diagnostic) every visible active key.
static void ownedKeys(const Scene& sc, const std::vector<int>& mine, bool periodic, MPI_Comm comm,
                      std::vector<unsigned long long>& owned,
                      std::vector<unsigned long long>& visible) {
  const int no = static_cast<int>(mine.size());
  ProbeSim sim(8 * static_cast<int>(sc.x.size()) + 64);
  sim.setDomain(static_cast<float>(GX), static_cast<float>(GX), static_cast<float>(GX), periodic,
                periodic, periodic);
  sim.setGlobalScale(1.0f);
  sim.setSphereShape(kRad);
  std::vector<float> xyz, sc3;
  for (int g : mine) {
    xyz.insert(xyz.end(), {sc.x[g][0], sc.x[g][1], sc.x[g][2]});
    sc3.push_back(sc.scale[g]);
  }
  sim.setPositions(xyz);
  sim.setScales(sc3);
  Particles& P = sim.parts();
  {
    auto hg = Kokkos::create_mirror_view(P.gid);
    for (int i = 0; i < no; ++i)
      hg(i) = mine[i];
    Kokkos::deep_copy(P.gid, hg);
  }
  Kokkos::deep_copy(P.posPred, P.pos);
  Kokkos::deep_copy(P.quatPred, P.quat);
  ParticleHalo halo;
  halo.initMpi({0.0, 0.0, 0.0}, {(double)GX, (double)GX, (double)GX}, {GX, GX, GX},
               {periodic, periodic, periodic}, comm);
  const float rMax = peclet::dem::globalMaxRadius(P, comm);
  const float margin = 0.1f * rMax;
  P.numParticles = P.numReal;
  halo.gather(P, peclet::dem::xpbdContactReach(rMax));
  peclet::dem::fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numParticles);
  const int np = peclet::dem::findCollisionsGrow(P, margin);
  const int nc = peclet::dem::narrowPhaseGrow(P, np, margin);
  const int ncOwned = peclet::dem::partitionContactsKokkos(P.contacts, nc, halo.contactOwnership(P));
  auto hc = Kokkos::create_mirror_view(P.contacts);
  auto hg = Kokkos::create_mirror_view(P.gid);
  Kokkos::deep_copy(hc, P.contacts);
  Kokkos::deep_copy(hg, P.gid);
  for (int i = 0; i < nc; ++i) {
    const auto& c = hc(i);
    if (c.dist > 0.0f)
      continue;
    const unsigned a = static_cast<unsigned>(hg(c.bodyA));
    const unsigned b = c.bodyB >= 0 ? static_cast<unsigned>(hg(c.bodyB)) : 0xFFFFFFFFu;
    if (a == b)
      continue;  // a body against its own periodic image (never owned)
    const unsigned long long k = peclet::dem::pairKeyFromGids(a, b);
    visible.push_back(k);
    if (i < ncOwned)
      owned.push_back(k);
  }
  P.numParticles = P.numReal;
}

static std::vector<unsigned long long> gatherKeys(const std::vector<unsigned long long>& mine,
                                                  int rank, int size) {
  const int n = static_cast<int>(mine.size());
  std::vector<int> cnt(size), off(size, 0);
  MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int tot = 0;
  if (rank == 0)
    for (int r = 0; r < size; ++r) {
      off[r] = tot;
      tot += cnt[r];
    }
  std::vector<unsigned long long> all(rank == 0 ? tot : 0);
  MPI_Gatherv(mine.data(), n, MPI_UNSIGNED_LONG_LONG, all.data(), cnt.data(), off.data(),
              MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  return all;
}

static int runExactlyOnce(const std::string& which, int rank, int size) {
  const bool periodic = which == "periodic", drift = which == "drift";
  const Scene sc = makeScene(size, drift);
  std::vector<int> mine, every;
  for (int g = 0; g < static_cast<int>(sc.x.size()); ++g) {
    every.push_back(g);
    if (sc.owner[g] == rank)
      mine.push_back(g);
  }
  std::vector<unsigned long long> owned, visible;
  ownedKeys(sc, mine, periodic, MPI_COMM_WORLD, owned, visible);
  // Keys owned by this rank, one per pair (a sphere pair is one contact point).
  const auto allOwned = gatherKeys(owned, rank, size);
  // Visible keys per rank, deduplicated rank-locally (self-image twins are one physical pair).
  std::sort(visible.begin(), visible.end());
  visible.erase(std::unique(visible.begin(), visible.end()), visible.end());
  const auto allVisible = gatherKeys(visible, rank, size);
  int fail = 0;
  if (rank == 0) {
    std::vector<unsigned long long> serialOwned, serialVisible;
    ownedKeys(sc, every, periodic, MPI_COMM_SELF, serialOwned, serialVisible);
    std::sort(serialVisible.begin(), serialVisible.end());
    serialVisible.erase(std::unique(serialVisible.begin(), serialVisible.end()),
                        serialVisible.end());
    std::vector<unsigned long long> o = allOwned;
    std::sort(o.begin(), o.end());
    int dup = 0;
    for (std::size_t i = 1; i < o.size(); ++i)
      if (o[i] == o[i - 1])
        ++dup;
    std::vector<unsigned long long> u = o;
    u.erase(std::unique(u.begin(), u.end()), u.end());
    const bool same = (u == serialVisible);
    std::vector<unsigned long long> vis = allVisible;
    std::sort(vis.begin(), vis.end());
    int missing = 0, missingUnseen = 0, extra = 0;
    for (auto k : serialVisible)
      if (!std::binary_search(u.begin(), u.end(), k)) {
        ++missing;
        if (!std::binary_search(vis.begin(), vis.end(), k))
          ++missingUnseen;  // no rank sees it at all: a halo gap, not an ownership decision
      }
    for (auto k : u)
      if (!std::binary_search(serialVisible.begin(), serialVisible.end(), k))
        ++extra;
    // Diagnostic: cross-rank pairs that only ONE rank sees (the drifted case of §1.2).
    std::map<unsigned long long, int> seen;
    for (auto k : allVisible)
      ++seen[k];
    int oneSided = 0;
    for (const auto& [k, n] : seen) {
      const unsigned a = static_cast<unsigned>(k >> 32), b = static_cast<unsigned>(k & 0xFFFFFFFFu);
      if (b != 0xFFFFFFFFu && sc.owner[a] != sc.owner[b] && n == 1)
        ++oneSided;
    }
    std::printf(
        "  exactly_once_%s np=%d: serial active pairs=%zu owned=%zu duplicates=%d missing=%d "
        "(seen by no rank: %d) extra=%d one-sided-visible=%d\n",
        which.c_str(), size, serialVisible.size(), o.size(), dup, missing, missingUnseen, extra,
        oneSided);
    if (dup != 0 || !same || serialVisible.empty() || (drift && size > 1 && oneSided == 0))
      fail = 1;
  }
  MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return fail;
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
    else if (mode == "exactly_once_closed")
      fail = runExactlyOnce("closed", rank, size);
    else if (mode == "exactly_once_periodic")
      fail = runExactlyOnce("periodic", rank, size);
    else if (mode == "exactly_once_drift")
      fail = runExactlyOnce("drift", rank, size);
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
