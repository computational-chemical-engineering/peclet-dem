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
// oracle_{closed,shear,periodic}: the report-only dynamic visibility oracle (see runOracle).
// Build with -DPECLET_DEM_MPI.
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <map>
#include <random>
#include <string>
#include <vector>

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
static Scene makeScene(int size, bool drift, float jitterOverride = -1.0f) {
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
  const float jitter = jitterOverride >= 0.0f ? jitterOverride : (drift ? 0.3f : 0.04f);
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
      const float lo = static_cast<float>(b.origin[d]),
                  hi = static_cast<float>(b.origin[d] + b.size[d]);
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

// One contact of the narrow phase, by global ids: the partner's periodic image relative to bodyA
// (integer box lengths; filled only when every partner's owner row is local, i.e. on
// MPI_COMM_SELF -- the oracle's serial reference), dist, and whether this rank owns it.
struct CRec {
  unsigned a, b;  // gid of bodyA, gid of bodyB (0xFFFFFFFF: a wall)
  int img[3];     // image of bodyB's slot minus image of bodyA's slot
  float dist;
  bool owned;
};

// Run gather + broad/narrow phase + partition on `comm` for the bodies `mine` (gids = mine) and
// return every contact the narrow phase reports on this rank.
static void detectContacts(const Scene& sc, const std::vector<int>& mine, bool periodic,
                           MPI_Comm comm, std::vector<CRec>& out) {
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
  // The production visibility path (docs/contact_solve_framework.md §5.1): the drift vote (a
  // particle S beyond its owner's block moves every particle to its current block) and the band
  // reach + S + d, exactly as demStepMpi runs them.
  const peclet::dem::MpiDriftVote vote = peclet::dem::mpiDriftVote(P, halo);
  Kokkos::deep_copy(P.posPred, P.pos);  // a migration carried pos / quat, not the predicted copies
  Kokkos::deep_copy(P.quatPred, P.quat);
  const float margin = vote.margin;
  P.numParticles = P.numReal;
  halo.gather(P, peclet::dem::mpiXpbdBand(P, halo, 0.0, vote));
  const int nOwned = P.numReal;  // after a drift migration, not mine.size()
  peclet::dem::fillWorldRadiiKokkos(P.scale, P.rad, P.globalScale, P.baseRadius, P.numParticles);
  const int np = peclet::dem::findCollisionsGrow(P, margin);
  const int nc = peclet::dem::narrowPhaseGrow(P, np, margin);
  const int ncOwned =
      peclet::dem::partitionContactsKokkos(P.contacts, nc, halo.contactOwnership(P));
  auto hc = Kokkos::create_mirror_view(P.contacts);
  auto hg = Kokkos::create_mirror_view(P.gid);
  auto hx = Kokkos::create_mirror_view(P.posPred);
  Kokkos::deep_copy(hc, P.contacts);
  Kokkos::deep_copy(hg, P.gid);
  Kokkos::deep_copy(hx, P.posPred);
  // Owner row of each gid present locally (for the image of a slot: slot - owner row, in box
  // lengths; a ghost copy is its owner's position plus the periodic shift).
  std::map<int, int> row;
  for (int i = 0; i < nOwned; ++i)
    row[hg(i)] = i;
  auto image = [&](int slot, int d) {
    if (slot < nOwned)
      return 0;
    const auto it = row.find(hg(slot));
    if (it == row.end())
      return 0;
    return static_cast<int>(std::lround((hx(slot, d) - hx(it->second, d)) / double(GX)));
  };
  for (int i = 0; i < nc; ++i) {
    const auto& c = hc(i);
    CRec r;
    r.a = static_cast<unsigned>(hg(c.bodyA));
    r.b = c.bodyB >= 0 ? static_cast<unsigned>(hg(c.bodyB)) : 0xFFFFFFFFu;
    for (int d = 0; d < 3; ++d)
      r.img[d] = c.bodyB >= 0 ? image(c.bodyB, d) - image(c.bodyA, d) : 0;
    r.dist = c.dist;
    r.owned = i < ncOwned;
    out.push_back(r);
  }
  P.numParticles = P.numReal;
}

// The owned active keys and (for the diagnostic) every visible active key of detectContacts.
static void ownedKeys(const Scene& sc, const std::vector<int>& mine, bool periodic, MPI_Comm comm,
                      std::vector<unsigned long long>& owned,
                      std::vector<unsigned long long>& visible) {
  std::vector<CRec> cs;
  detectContacts(sc, mine, periodic, comm, cs);
  for (const CRec& c : cs) {
    if (c.dist > 0.0f)
      continue;
    if (c.a == c.b)
      continue;  // a body against its own periodic image (never owned)
    const unsigned long long k = peclet::dem::pairKeyFromGids(c.a, c.b);
    visible.push_back(k);
    if (c.owned)
      owned.push_back(k);
  }
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

// ---- REPORT-ONLY: pairs that NO rank sees (docs/contact_evidence/FOLLOWUPS.md, defect 4) ----
// Each mode builds a scene with a fixed owner per particle, runs the distributed step's contact
// detection (gather at the XPBD band + broad/narrow phase, ownedKeys) on every rank and the same
// detection on the whole set on MPI_COMM_SELF, and counts the serial active pairs that are not
// VISIBLE on any rank (so no rank can solve them, whatever the ownership rule). It never fails
// today; when the halo supplies every pair, kMissedGate = true makes a missed pair a failure.
static constexpr bool kMissedGate = true;  // WO-9: every periodic image is sent
// WO-7 (docs/contact_solve_framework.md §5.1): the drift vote + migrateToBlocks + band reach + S +
// d make every drifted pair visible, so the drift probes are gates. The periodic probe stays
// report-only until WO-9 (all periodic images).
static constexpr bool kMissedDriftGate = true;

// Serial active pairs, and the union of the pairs visible on any rank (on rank 0).
static void visibleVsSerial(const Scene& sc, bool periodic, int rank, int size,
                            std::vector<unsigned long long>& serial,
                            std::vector<unsigned long long>& missed) {
  std::vector<int> mine, every;
  for (int g = 0; g < static_cast<int>(sc.x.size()); ++g) {
    every.push_back(g);
    if (sc.owner[g] == rank)
      mine.push_back(g);
  }
  std::vector<unsigned long long> owned, visible;
  ownedKeys(sc, mine, periodic, MPI_COMM_WORLD, owned, visible);
  std::sort(visible.begin(), visible.end());
  visible.erase(std::unique(visible.begin(), visible.end()), visible.end());
  std::vector<unsigned long long> all = gatherKeys(visible, rank, size);
  serial.clear();
  missed.clear();
  if (rank != 0)
    return;
  std::vector<unsigned long long> so, sv;
  ownedKeys(sc, every, periodic, MPI_COMM_SELF, so, sv);
  std::sort(sv.begin(), sv.end());
  sv.erase(std::unique(sv.begin(), sv.end()), sv.end());
  serial = sv;
  std::sort(all.begin(), all.end());
  for (auto k : sv)
    if (!std::binary_search(all.begin(), all.end(), k))
      missed.push_back(k);
}

// Distance from x to block r (closed: no images).
static double gapToBlock(const BlockDecomposer<3>& dec, int r, const std::array<float, 3>& x) {
  const auto b = dec.block(static_cast<std::size_t>(r));
  double d2 = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double lo = b.origin[d], hi = b.origin[d] + b.size[d];
    const double gap = x[d] < lo ? lo - x[d] : (x[d] > hi ? x[d] - hi : 0.0);
    d2 += gap * gap;
  }
  return std::sqrt(d2);
}

static void printBlocks(const BlockDecomposer<3>& dec, int size) {
  for (int r = 0; r < size; ++r) {
    const auto b = dec.block(static_cast<std::size_t>(r));
    std::printf("  block %d: [%ld,%ld) x [%ld,%ld) x [%ld,%ld)\n", r, (long)b.origin[0],
                (long)(b.origin[0] + b.size[0]), (long)b.origin[1], (long)(b.origin[1] + b.size[1]),
                (long)b.origin[2], (long)(b.origin[2] + b.size[2]));
  }
}

// (a) drift, minimal geometry: two touching unit grains A (owner r) and B (owner s) across the
// shared face of two face-adjacent blocks (axis a, at coordinate F), both moved a distance y past
// the common upper end of the two blocks on a second axis b (as if advected with
// rebalance_every = 0). r receives B only while B is within `band` of block r, s receives A only
// while A is within `band` of block s; with the centres F -+ d/2 both distances are
// sqrt((d/2)^2 + y^2), so the pair is lost once y > sqrt(band^2 - d^2/4).
static int runMissedDriftPair(int rank, int size) {
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  int r = -1, s = -1, a = -1, bax = -1;
  for (int i = 0; i < size && r < 0; ++i)
    for (int j = 0; j < size && r < 0; ++j) {
      if (i == j)
        continue;
      const auto bi = dec.block(i), bj = dec.block(j);
      for (int ax = 0; ax < 3 && r < 0; ++ax) {
        if (bi.origin[ax] + bi.size[ax] != bj.origin[ax])
          continue;
        bool same = true;
        for (int o = 0; o < 3; ++o)
          if (o != ax && (bi.origin[o] != bj.origin[o] || bi.size[o] != bj.size[o]))
            same = false;
        if (!same)
          continue;
        for (int o = 0; o < 3 && r < 0; ++o)
          if (o != ax && bi.origin[o] + bi.size[o] < GX) {
            r = i;
            s = j;
            a = ax;
            bax = o;
          }
      }
    }
  if (rank == 0) {
    std::printf("  missed_drift_pair np=%d layout:\n", size);
    printBlocks(dec, size);
  }
  if (r < 0) {
    if (rank == 0)
      std::printf(
          "  missed_drift_pair np=%d: no two face-adjacent blocks end below the domain on a second "
          "axis -- a drifted pair is always within the band of one owner here (closed box)\n",
          size);
    return 0;
  }
  const auto br = dec.block(r);
  const double F = br.origin[a] + br.size[a];
  const double top = br.origin[bax] + br.size[bax];
  int third = 3 - a - bax;
  const double mid = br.origin[third] + 0.5 * br.size[third];
  const double dsep = 2.0 * kRad - 0.02;  // touching, overlap 0.02
  const double band = peclet::dem::xpbdContactReach(kRad);
  const double yStar = std::sqrt(band * band - 0.25 * dsep * dsep);
  if (rank == 0)
    std::printf(
        "  missed_drift_pair np=%d: A owner %d, B owner %d, face axis %d at %.1f, drift axis %d "
        "past "
        "%.1f; R = %.2f, band = %.4f, |AB| = %.3f, predicted loss for y > %.4f (= %.3f R)\n",
        size, r, s, a, F, bax, top, kRad, band, dsep, yStar, yStar / kRad);
  int lostTot = 0;
  for (double yR : {0.0, 0.5, 1.0, 1.5, 1.8, 1.85, 1.86, 1.9, 2.0, 3.0}) {
    Scene sc;
    std::array<float, 3> pa{}, pb{};
    pa[a] = static_cast<float>(F - 0.5 * dsep);
    pb[a] = static_cast<float>(F + 0.5 * dsep);
    pa[bax] = pb[bax] = static_cast<float>(top + yR * kRad);
    pa[third] = pb[third] = static_cast<float>(mid);
    sc.x = {pa, pb};
    sc.scale = {1.0f, 1.0f};
    sc.owner = {r, s};
    std::vector<unsigned long long> serial, missed;
    visibleVsSerial(sc, false, rank, size, serial, missed);
    if (rank == 0) {
      std::printf(
          "  MISSED mode=drift_pair np=%d y/R=%.2f serial=%zu missed=%zu dist(A,block s)=%.4f "
          "dist(B,block r)=%.4f\n",
          size, yR, serial.size(), missed.size(), gapToBlock(dec, s, pa), gapToBlock(dec, r, pb));
      lostTot += static_cast<int>(missed.size());
    }
  }
  MPI_Bcast(&lostTot, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (kMissedDriftGate && lostTot > 0) ? 1 : 0;
}

// (a) drift, lattice: the weak-jitter closed lattice of exactly_once_closed, owners at the
// original positions, then every particle translated by D along x (a uniform advection with
// rebalance_every = 0; particles leaving [0, GX) are dropped). Counts the pairs no rank sees.
static int runMissedDriftLattice(int rank, int size) {
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  const Scene base = makeScene(size, false);
  int lostTot = 0;
  for (double DR : {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0}) {
    Scene sc;
    for (std::size_t g = 0; g < base.x.size(); ++g) {
      std::array<float, 3> q = base.x[g];
      q[0] += static_cast<float>(DR * kRad);
      if (q[0] >= GX)
        continue;
      sc.x.push_back(q);
      sc.scale.push_back(base.scale[g]);
      sc.owner.push_back(base.owner[g]);
    }
    std::vector<unsigned long long> serial, missed;
    visibleVsSerial(sc, false, rank, size, serial, missed);
    if (rank == 0) {
      std::printf("  MISSED mode=drift_lattice np=%d D/R=%.1f N=%zu serial=%zu missed=%zu\n", size,
                  DR, sc.x.size(), serial.size(), missed.size());
      if (!missed.empty()) {
        const unsigned ga = static_cast<unsigned>(missed[0] >> 32),
                       gb = static_cast<unsigned>(missed[0] & 0xFFFFFFFFu);
        const auto &xa = sc.x[ga], &xb = sc.x[gb];
        std::printf(
            "    e.g. gid %u (owner %d) at (%.3f, %.3f, %.3f) and gid %u (owner %d) at (%.3f, "
            "%.3f, %.3f): dist(a, block %d) = %.3f, dist(b, block %d) = %.3f, band %.3f\n",
            ga, sc.owner[ga], xa[0], xa[1], xa[2], gb, sc.owner[gb], xb[0], xb[1], xb[2],
            sc.owner[gb], gapToBlock(dec, sc.owner[gb], xa), sc.owner[ga],
            gapToBlock(dec, sc.owner[ga], xb), peclet::dem::xpbdContactReach(1.1f * kRad));
      }
      lostTot += static_cast<int>(missed.size());
    }
  }
  MPI_Bcast(&lostTot, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (kMissedDriftGate && lostTot > 0) ? 1 : 0;
}

// (b) periodic, strong jitter: the lattice with jitter 0.3 (as exactly_once_drift, no
// displacement) in a box periodic on every axis. For each missed pair prints the owners, the
// minimum-image separation, which axes it wraps and which are decomposed, and, for each owner's
// block, the image of the partner the halo sends (the single nearest one per axis,
// withinRcutOfBlock) versus the image the pair needs, and whether the needed image is inside the
// band (imagesWithinRcutOfBlock): a needed image inside the band but not sent is an image-
// enumeration gap, one outside the band a band-width gap.
static int runMissedPeriodic(int rank, int size) {
  BlockDecomposer<3> dec(static_cast<std::size_t>(size), IVec<3>{GX, GX, GX});
  const Scene sc = makeScene(size, false, 0.3f);  // the strong jitter, no displacement
  peclet::core::halo::DomainMap<3> map;
  for (int d = 0; d < 3; ++d) {
    map.origin[d] = 0.0;
    map.cellSize[d] = 1.0;
    map.periodic[d] = true;
  }
  peclet::core::halo::ParticleMigrator<3> mig;
  mig.init(dec, rank, map, MPI_COMM_WORLD);
  std::vector<unsigned long long> serial, missed;
  visibleVsSerial(sc, true, rank, size, serial, missed);
  if (rank == 0) {
    std::printf("  missed_periodic np=%d layout:\n", size);
    printBlocks(dec, size);
    std::printf("  MISSED mode=periodic np=%d serial=%zu missed=%zu\n", size, serial.size(),
                missed.size());
    const double band = peclet::dem::xpbdContactReach(1.1f * kRad);  // global R_max = 1.1 R
    for (auto k : missed) {
      const unsigned ga = static_cast<unsigned>(k >> 32),
                     gb = static_cast<unsigned>(k & 0xFFFFFFFFu);
      const int ra = sc.owner[ga], rb = sc.owner[gb];
      const auto &xa = sc.x[ga], &xb = sc.x[gb];
      // minimum-image separation b - a and the wrap per axis
      double sep[3], d2 = 0.0;
      int wrap[3];
      for (int d = 0; d < 3; ++d) {
        double s = xb[d] - xa[d];
        wrap[d] = 0;
        if (s > 0.5 * GX) {
          s -= GX;
          wrap[d] = -1;
        } else if (s < -0.5 * GX) {
          s += GX;
          wrap[d] = 1;
        }
        sep[d] = s;
        d2 += s * s;
      }
      std::printf(
          "    pair gid %u (owner %d) (%.3f, %.3f, %.3f) -- gid %u (owner %d) (%.3f, %.3f, "
          "%.3f): |sep| = %.3f, wrap of b per axis (%+d, %+d, %+d) x GX\n",
          ga, ra, xa[0], xa[1], xa[2], gb, rb, xb[0], xb[1], xb[2], std::sqrt(d2), wrap[0], wrap[1],
          wrap[2]);
      // What each owner receives of the other, versus what the pair needs.
      for (int side = 0; side < 2; ++side) {
        const int dst = side == 0 ? ra : rb;   // receiving rank
        const auto& xs = side == 0 ? xb : xa;  // the partner it must receive
        int need[3];
        for (int d = 0; d < 3; ++d)
          need[d] = side == 0 ? wrap[d] : -wrap[d];
        peclet::core::Vec<3> v{xs[0], xs[1], xs[2]}, img;
        const bool sent = mig.withinRcutOfBlock(v, dst, band, img);
        int got[3];
        for (int d = 0; d < 3; ++d)
          got[d] = static_cast<int>(std::lround((img[d] - v[d]) / GX));
        std::vector<peclet::core::Vec<3>> shifts;
        mig.imagesWithinRcutOfBlock(v, dst, band, /*allowIdentity=*/true, shifts);
        bool needInBand = false;
        for (const auto& sh : shifts) {
          bool eq = true;
          for (int d = 0; d < 3; ++d)
            if (std::lround(sh[d] / GX) != need[d])
              eq = false;
          needInBand = needInBand || eq;
        }
        std::printf(
            "      rank %d receives gid %u: %s, image (%+d, %+d, %+d); needs (%+d, %+d, %+d), "
            "which is %s the band (%zu images qualify)%s\n",
            dst, side == 0 ? gb : ga, sent ? "sent" : "NOT sent", got[0], got[1], got[2], need[0],
            need[1], need[2], needInBand ? "INSIDE" : "outside", shifts.size(),
            dst == (side == 0 ? rb : ra) ? " [same rank: a periodic self-ghost]" : "");
      }
    }
  }
  int lost = static_cast<int>(missed.size());
  MPI_Bcast(&lost, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return (kMissedGate && lost > 0) ? 1 : 0;
}

// ---- REPORT-ONLY: the dynamic visibility oracle (docs/contact_solve_framework.md WO-0 item 3;
// the gate G5 once the framework lands: missing = dup = extra = 0 at every step) ----
// oracle_{closed,shear,periodic}: the lattice of makeScene (radii 1 +- 0.1 R), owners at the
// original positions, advanced by step_mpi for 50 steps with rebalance_every = 0 (dt 1e-2, g = 0,
// e = 0.5, frictionless, pos/vel iterations 20/8, the API's unit masses). closed: weak jitter,
// at rest; shear: the same plus v_x = 0.2 (z - GX/2) in units of R per step, i.e. 0.2 (z - GX/2)
// / dt; periodic: every axis periodic, strong jitter 0.3, at rest. After every step each rank's
// owned contacts of the substep (Simulation::debugCaptureContacts: global ids, each slot's
// periodic image, dist) and its owned predicted positions + radii are gathered to rank 0, which
// runs the same narrow phase on MPI_COMM_SELF over the gathered predicted state (periodic
// self-ghosts included) and compares pairs keyed by (min gid, max gid, image of the max-gid body
// relative to the min-gid body):
//   required = serial pairs with dist < margin - 1e-4 R_max (margin = 0.1 R_max, the step's)
//   missing  = required pairs owned by no rank; dup = pairs owned more than once;
//   extra    = owned pairs that are not a serial pair with dist < margin + 1e-4 R_max.
// One ORACLE line per run: the maximum of each over the steps and the first step it was > 0.
// Options: --dump=<path> (final committed state, records {int32 gid; float32 pos[3], vel[3]}
// sorted by lattice index), --capture=0 (run without the capture and without the comparison: the
// capture's inertness check).
static constexpr bool kOracleGate = false;

using OKey = std::array<int, 5>;  // min gid, max gid, image (3)
static bool canonicalKey(int a, int b, const int img[3], OKey& k) {
  if (a < 0 || b < 0 || a == b)
    return false;  // a wall, or a body against its own image
  if (a < b)
    k = {a, b, img[0], img[1], img[2]};
  else
    k = {b, a, -img[0], -img[1], -img[2]};
  return true;
}

template <typename T>
static std::vector<T> gatherTo0(const std::vector<T>& mine, MPI_Datatype type, int rank, int size) {
  const int n = static_cast<int>(mine.size());
  std::vector<int> cnt(size), off(size, 0);
  MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  int tot = 0;
  if (rank == 0)
    for (int r = 0; r < size; ++r) {
      off[r] = tot;
      tot += cnt[r];
    }
  std::vector<T> all(rank == 0 ? tot : 0);
  MPI_Gatherv(mine.data(), n, type, all.data(), cnt.data(), off.data(), type, 0, MPI_COMM_WORLD);
  return all;
}

static int runOracle(const std::string& which, int rank, int size, const std::string& dump,
                     bool capture) {
  const bool periodic = which == "periodic", shear = which == "shear";
  const Scene sc = makeScene(size, false, periodic ? 0.3f : -1.0f);
  const float dt = 1e-2f;
  const int steps = 50;
  std::vector<int> mine;
  for (int g = 0; g < static_cast<int>(sc.x.size()); ++g)
    if (sc.owner[g] == rank)
      mine.push_back(g);
  const int no = static_cast<int>(mine.size());
  ProbeSim sim(8 * static_cast<int>(sc.x.size()) + 64);
  sim.setDomain(static_cast<float>(GX), static_cast<float>(GX), static_cast<float>(GX), periodic,
                periodic, periodic);
  sim.setGlobalScale(1.0f);
  sim.setSphereShape(kRad);
  sim.setDt(dt);
  sim.setGravity(0.0f, 0.0f, 0.0f);
  sim.setSolverIterations(20, 8);
  sim.setMaterialParams(0.5f, 0.0f, 0.0f);
  {
    std::vector<float> xyz, sc3, v;
    for (int g : mine) {
      xyz.insert(xyz.end(), {sc.x[g][0], sc.x[g][1], sc.x[g][2]});
      sc3.push_back(sc.scale[g]);
      const float vx = shear ? 0.2f * (sc.x[g][2] - 0.5f * GX) / dt : 0.0f;
      v.insert(v.end(), {vx, 0.0f, 0.0f});
    }
    sim.setPositions(xyz);
    sim.setScales(sc3);
    sim.setVelocities(v);
  }
  sim.initMpi({0.0, 0.0, 0.0}, {(double)GX, (double)GX, (double)GX}, {GX, GX, GX},
              {periodic, periodic, periodic}, MPI_COMM_WORLD);
  sim.enableMpiStep(0.0, 1, true, /*rebalance_every=*/0);
  sim.debugCaptureContacts(capture);

  int mx[3] = {0, 0, 0}, first[3] = {0, 0, 0}, reqMax = 0, ownedMax = 0;
  std::array<int, 5> firstMissKey{-1, -1, 0, 0, 0};
  for (int s = 1; s <= steps; ++s) {
    sim.stepMpi(1);
    if (!capture)
      continue;
    const peclet::dem::DebugContactCapture& cap = sim.debugCapturedContacts();
    std::vector<int> ci;
    std::vector<float> cf, bf;
    std::vector<int> bi;
    for (std::size_t i = 0; i < cap.gidA.size(); ++i) {
      ci.insert(ci.end(), {cap.gidA[i], cap.gidB[i]});
      for (int d = 0; d < 3; ++d)
        ci.push_back(cap.imageB[3 * i + d] - cap.imageA[3 * i + d]);
      cf.push_back(cap.dist[i]);
    }
    for (std::size_t i = 0; i < cap.gid.size(); ++i) {
      bi.push_back(cap.gid[i]);
      bf.insert(bf.end(),
                {cap.posPred[3 * i], cap.posPred[3 * i + 1], cap.posPred[3 * i + 2], cap.rad[i]});
    }
    const std::vector<int> aci = gatherTo0(ci, MPI_INT, rank, size);
    const std::vector<float> acf = gatherTo0(cf, MPI_FLOAT, rank, size);
    const std::vector<int> abi = gatherTo0(bi, MPI_INT, rank, size);
    const std::vector<float> abf = gatherTo0(bf, MPI_FLOAT, rank, size);
    if (rank != 0)
      continue;
    // The serial reference over the gathered predicted state (index = global id).
    const int nb = static_cast<int>(abi.size());
    Scene ser;
    ser.x.resize(nb);
    ser.scale.resize(nb);
    std::vector<int> all(nb);
    float rMax = 0.0f;
    for (int k = 0; k < nb; ++k) {
      const int g = abi[k];
      ser.x[g] = {abf[4 * k], abf[4 * k + 1], abf[4 * k + 2]};
      ser.scale[g] = abf[4 * k + 3] / kRad;  // world radius = scale * 1 * kRad (a power of two)
      rMax = std::max(rMax, abf[4 * k + 3]);
    }
    for (int g = 0; g < nb; ++g)
      all[g] = g;
    std::vector<CRec> cs;
    detectContacts(ser, all, periodic, MPI_COMM_SELF, cs);
    const float margin = 0.1f * rMax;
    const double tol = 1e-4 * rMax;
    std::map<OKey, float> serial;  // key -> smallest dist over its contacts / twins
    for (const CRec& c : cs) {
      OKey k;
      const int b = c.b == 0xFFFFFFFFu ? -1 : static_cast<int>(c.b);
      if (!canonicalKey(static_cast<int>(c.a), b, c.img, k))
        continue;
      const auto it = serial.find(k);
      if (it == serial.end() || c.dist < it->second)
        serial[k] = c.dist;
    }
    std::map<OKey, int> owned;
    for (std::size_t i = 0; i < acf.size(); ++i) {
      OKey k;
      if (canonicalKey(aci[5 * i], aci[5 * i + 1], &aci[5 * i + 2], k))
        ++owned[k];
    }
    int miss = 0, dup = 0, extra = 0, req = 0;
    for (const auto& [k, d] : serial)
      if (d < margin - tol) {
        ++req;
        if (!owned.count(k)) {
          if (miss == 0 && first[0] == 0) {
            firstMissKey = k;
          }
          ++miss;
        }
      }
    for (const auto& [k, n] : owned) {
      if (n > 1)
        ++dup;
      const auto it = serial.find(k);
      if (it == serial.end() || !(it->second < margin + tol))
        ++extra;
    }
    const int cur[3] = {miss, dup, extra};
    for (int q = 0; q < 3; ++q) {
      mx[q] = std::max(mx[q], cur[q]);
      if (cur[q] > 0 && first[q] == 0)
        first[q] = s;
    }
    reqMax = std::max(reqMax, req);
    ownedMax = std::max(ownedMax, static_cast<int>(owned.size()));
  }
  const int thr = Kokkos::DefaultHostExecutionSpace().concurrency();
  if (rank == 0 && capture) {
    std::printf(
        "ORACLE mode=oracle_%s np=%d thr=%d steps=%d required=%d owned=%d missing=%d dup=%d "
        "extra=%d firstMiss=%d firstDup=%d firstExtra=%d\n",
        which.c_str(), size, thr, steps, reqMax, ownedMax, mx[0], mx[1], mx[2], first[0], first[1],
        first[2]);
    if (first[0] > 0)
      std::printf("  first missing pair (step %d): gid %d -- gid %d, image (%+d, %+d, %+d)\n",
                  first[0], firstMissKey[0], firstMissKey[1], firstMissKey[2], firstMissKey[3],
                  firstMissKey[4]);
  }
  if (!dump.empty()) {  // final committed state, sorted by lattice index (fixed ownership)
    const std::vector<float> x = sim.getPositions(), v = sim.getVelocities();
    std::vector<float> rec;
    for (int i = 0; i < no; ++i) {
      float g;
      const std::int32_t gi = mine[i];
      std::memcpy(&g, &gi, 4);
      rec.insert(rec.end(),
                 {g, x[3 * i], x[3 * i + 1], x[3 * i + 2], v[3 * i], v[3 * i + 1], v[3 * i + 2]});
    }
    const std::vector<float> allr = gatherTo0(rec, MPI_FLOAT, rank, size);
    if (rank == 0) {
      const int nr = static_cast<int>(allr.size() / 7);
      std::vector<int> order(nr);
      for (int k = 0; k < nr; ++k)
        order[k] = k;
      auto gidOf = [&](int k) {
        std::int32_t g;
        std::memcpy(&g, &allr[7 * static_cast<std::size_t>(k)], 4);
        return g;
      };
      std::sort(order.begin(), order.end(), [&](int a, int b) { return gidOf(a) < gidOf(b); });
      if (std::FILE* f = std::fopen(dump.c_str(), "wb")) {
        for (int k : order)
          std::fwrite(&allr[7 * static_cast<std::size_t>(k)], 4, 7, f);
        std::fclose(f);
      }
    }
  }
  // WO-7: closed and sheared scenes are gates (every pair within reach solved exactly once);
  // the periodic scene waits for WO-9 (all periodic images).
  const bool gateOn = true;  // WO-9: closed, sheared and periodic are all gates
  // The gate is missing = dup = 0 (docs/contact_solve_framework.md WO-7). `extra` (a pair solved
  // at a gap within 1e-4 R_max beyond the margin on a thread-order / round-off edge) is reported:
  // a speculative contact acts only on approach, so it is not a visibility failure.
  int bad = (gateOn && (mx[0] > 0 || mx[1] > 0 || (kOracleGate && mx[2] > 0))) ? 1 : 0;
  MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return bad;
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
    else if (mode == "missed_drift_pair")
      fail = runMissedDriftPair(rank, size);
    else if (mode == "missed_drift_lattice")
      fail = runMissedDriftLattice(rank, size);
    else if (mode == "missed_periodic")
      fail = runMissedPeriodic(rank, size);
    else if (mode == "oracle_closed" || mode == "oracle_shear" || mode == "oracle_periodic") {
      std::string dump;
      bool capture = true;
      for (int a = 2; a < argc; ++a) {
        if (std::strncmp(argv[a], "--dump=", 7) == 0)
          dump = argv[a] + 7;
        else if (std::strcmp(argv[a], "--capture=0") == 0)
          capture = false;
      }
      fail = runOracle(mode.substr(7), rank, size, dump, capture);
    } else {
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
