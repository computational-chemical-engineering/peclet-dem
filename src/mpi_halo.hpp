/// @file
/// @brief dem — portable (Kokkos) owner<->ghost particle halo for the distributed XPBD step.
///
/// Kokkos counterpart of src/mpi/mpi_halo.h (MpiParticleHalo): a thin wrapper over transport-core's
/// peclet::core::halo::ParticleHaloTopology<3> (host topology, periodic image shift) +
/// ParticleHalo<3> (on-device gather/scatter + host-staged MPI). Rebuilt each substep from the
/// owned positions; it gathers ghost copies of the owners' FULL state into the Particles SoA ghost
/// slots and refreshes them owner->ghost during the velocity/position solves -- the EXACT
/// distributed scheme (ghosts carry REAL mass; every owned particle sees all its neighbours so it
/// computes its full serial XPBD delta locally; ghost deltas are discarded via a self-mapped
/// realIndices).
///
/// Faithful Kokkos port of Simulation::mpi_gather_ghosts / mpi_forward_positions / mpi_forward4.
/// Gated behind PECLET_DEM_MPI (mirrors cfd's CFD_MPI): the default dem module never includes this,
/// so it stays byte-identical when the macro is off.
#ifndef PECLET_DEM_MPI_HALO_HPP
#define PECLET_DEM_MPI_HALO_HPP
#ifdef PECLET_DEM_MPI

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "dem_portable.hpp"  // F3, F4
#include "particles.hpp"     // Particles, V3/V4/Vf/Vi, CpExec/CpMem
#include "peclet/core/common/types.hpp"
#include "peclet/core/common/view.hpp"
#include "peclet/core/decomp/block_decomposer.hpp"
#include "peclet/core/halo/particle_halo.hpp"
#include "peclet/core/halo/particle_halo_topology.hpp"
#include "peclet/core/halo/particle_migrator.hpp"
#include "peclet/core/halo/particle_rebalance.hpp"

namespace peclet::dem {

// All non-position per-particle state in one record, so the substep gather is a single MPI exchange
// (latency dominates the host-staged path). Mirrors MpiParticleHalo::GatherPack; positions are
// forwarded separately (they need the periodic image shift). invMass is its own field here (the
// CUDA SoA carries it in pos.w; the Kokkos SoA keeps it in a separate array). POD =>
// MPI_BYTE-copyable.
struct MpiGatherPack {
  F3 vel, velPred, angVel, angVelPred, invInertia;
  F4 quat, quatPred;
  float scale, invMass;
  int shape;
  // Modern-solver state a ghost must mirror from its owner: the global id (persistent-pair keys
  // are gid-based — local slots are not stable identities across ranks / halo rebuilds), the
  // material id (the narrowphase reads matId by RAW slot index), and the owner's warm grounded
  // level (Guendelman support levels; the rank-local propagation sweeps continue from it, so a
  // support chain crossing a rank boundary stays grounded).
  int gid;
  unsigned char material, grounded;
  // Poisson-restitution orphan account (owner-authoritative; the ghost copy's rank-local
  // drawdowns are overwritten by the next mirror — the redundant ghost-pair solve computes
  // the identical drawdown on the owner).
  float orphan, orphanVPeak;
};

// One persistent-contact ledger entry carried through an ownership migration: the gid-based pair
// key plus the previous substep's converged normal / tangential impulses and position-channel
// load (the warm start + Coulomb-bound carry). POD => MPI_BYTE-copyable.
struct WarmPairEntry {
  unsigned long long key;
  float lambda;
  float lambdaT[3];
  float posImpulse;
  float restBank;   // event-level (Poisson) restitution: remaining owed separation impulse
  float restVPeak;  // ... and the event's peak approach speed (> 0 = event active)
};
// Per-particle cap on carried pairs (sphere kissing number 12 + wall; lowest-weight entries are
// dropped beyond it — a dropped entry only costs the receiving rank a cold warm-start there).
inline constexpr int kWarmCarryMax = 14;

// One force-engine (Hertz–Mindlin) per-pair history entry carried through an ownership migration:
// the gid-based pair key + the Mindlin shear spring xi. The lagged patch stiffness (snPair) is NOT
// carried — it resets at every pair-list rebuild anyway, and a migration forces one.
struct HertzPairEntry {
  unsigned long long key;
  float xi[3];
};

// The committed per-particle state that defines a particle across steps — everything except its
// position (which drives ownership and travels as the migrator's coordinate) and the predicted /
// delta / ghost scratch the step rebuilds. This is the payload moved when a particle changes owner
// during a load re-balance. POD => MPI_BYTE-copyable. Carries the particle's slice of the
// persistent-contact ledger (each pair rides on BOTH endpoints; the unpack dedupes by key) plus
// its grounded level, so a rebalance does not cold-restart the statics force network.
struct MigratePack {
  F4 quat;
  F3 vel, angVel, invInertia;
  float invMass, scale, targetScale;
  int shapeId;
  float planeFric0, planeFric1;
  int gid;
  unsigned char materialId, groundedLevel, numWarm, numHertz;
  float orphan, orphanVPeak;  // Poisson orphan account rides with its body across ownership
  WarmPairEntry warm[kWarmCarryMax];
  // Force-engine (Hertz–Mindlin) history: the particle's slice of the cached pair list's Mindlin
  // springs plus its per-(particle, wall) shear history + lagged wall patch stiffness.
  HertzPairEntry hertz[kWarmCarryMax];
  float hertzXiWall[Particles::kHertzMaxWalls][3];
  float hertzSnWall[Particles::kHertzMaxWalls];
};

// --- free-function pack/unpack kernels (namespace scope: nvcc forbids KOKKOS_LAMBDA in member fns)
// ---

inline void haloPackF3(V3 field, peclet::core::View<F3> owned, int n) {
  Kokkos::parallel_for(
      "peclet::dem::halo::packF3", Kokkos::RangePolicy<CpExec>(0, n),
      KOKKOS_LAMBDA(int i) { owned(i) = F3{field(i, 0), field(i, 1), field(i, 2)}; });
}
inline void haloPackF4(V4 field, peclet::core::View<F4> owned, int n) {
  Kokkos::parallel_for(
      "peclet::dem::halo::packF4", Kokkos::RangePolicy<CpExec>(0, n),
      KOKKOS_LAMBDA(int i) { owned(i) = F4{field(i, 0), field(i, 1), field(i, 2), field(i, 3)}; });
}
// ghost[g] -> field(no+g,:), optionally adding the per-ghost periodic image shift (positions only).
inline void haloUnpackF3(V3 field, peclet::core::View<F3> ghost, peclet::core::View<F3> shift,
                         int no, int ng, bool doShift) {
  Kokkos::parallel_for(
      "peclet::dem::halo::unpackF3", Kokkos::RangePolicy<CpExec>(0, ng), KOKKOS_LAMBDA(int g) {
        F3 v = ghost(g);
        if (doShift) {
          v.x += shift(g).x;
          v.y += shift(g).y;
          v.z += shift(g).z;
        }
        field(no + g, 0) = v.x;
        field(no + g, 1) = v.y;
        field(no + g, 2) = v.z;
      });
}
inline void haloUnpackF4(V4 field, peclet::core::View<F4> ghost, int no, int ng) {
  Kokkos::parallel_for(
      "peclet::dem::halo::unpackF4", Kokkos::RangePolicy<CpExec>(0, ng), KOKKOS_LAMBDA(int g) {
        field(no + g, 0) = ghost(g).x;
        field(no + g, 1) = ghost(g).y;
        field(no + g, 2) = ghost(g).z;
        field(no + g, 3) = ghost(g).w;
      });
}

inline void haloPackGather(V3 vel, V3 velPred, V3 angVel, V3 angVelPred, V3 invInertia, V4 quat,
                           V4 quatPred, Vf scale, Vf invMass, Vi shapeId, Vi gid,
                           Kokkos::View<unsigned char*, CpMem> materialId,
                           Kokkos::View<unsigned char*, CpMem> grounded, Vf orphan, Vf orphanVPeak,
                           peclet::core::View<MpiGatherPack> owned, int n) {
  Kokkos::parallel_for(
      "peclet::dem::halo::packGather", Kokkos::RangePolicy<CpExec>(0, n), KOKKOS_LAMBDA(int i) {
        MpiGatherPack g;
        g.vel = F3{vel(i, 0), vel(i, 1), vel(i, 2)};
        g.velPred = F3{velPred(i, 0), velPred(i, 1), velPred(i, 2)};
        g.angVel = F3{angVel(i, 0), angVel(i, 1), angVel(i, 2)};
        g.angVelPred = F3{angVelPred(i, 0), angVelPred(i, 1), angVelPred(i, 2)};
        g.invInertia = F3{invInertia(i, 0), invInertia(i, 1), invInertia(i, 2)};
        g.quat = F4{quat(i, 0), quat(i, 1), quat(i, 2), quat(i, 3)};
        g.quatPred = F4{quatPred(i, 0), quatPred(i, 1), quatPred(i, 2), quatPred(i, 3)};
        g.scale = scale(i);
        g.invMass = invMass(i);
        g.shape = shapeId(i);
        g.gid = gid(i);
        g.material = materialId(i);
        g.grounded = grounded(i);
        g.orphan = orphan(i);
        g.orphanVPeak = orphanVPeak(i);
        owned(i) = g;
      });
}
// Unpack the gathered owner state into the ghost slots [no, no+ng) and self-map realIndices (the
// owner is remote, so velocity/position deltas landing on the ghost slot are discarded next
// forward).
inline void haloUnpackGather(V3 vel, V3 velPred, V3 angVel, V3 angVelPred, V3 invInertia, V4 quat,
                             V4 quatPred, Vf scale, Vf invMass, Vi shapeId, Vi realIndices, Vi gid,
                             Kokkos::View<unsigned char*, CpMem> materialId,
                             Kokkos::View<unsigned char*, CpMem> grounded, Vf orphan,
                             Vf orphanVPeak, peclet::core::View<MpiGatherPack> ghost, int no,
                             int ng) {
  Kokkos::parallel_for(
      "peclet::dem::halo::unpackGather", Kokkos::RangePolicy<CpExec>(0, ng), KOKKOS_LAMBDA(int g) {
        const MpiGatherPack p = ghost(g);
        const int s = no + g;
        vel(s, 0) = p.vel.x;
        vel(s, 1) = p.vel.y;
        vel(s, 2) = p.vel.z;
        velPred(s, 0) = p.velPred.x;
        velPred(s, 1) = p.velPred.y;
        velPred(s, 2) = p.velPred.z;
        angVel(s, 0) = p.angVel.x;
        angVel(s, 1) = p.angVel.y;
        angVel(s, 2) = p.angVel.z;
        angVelPred(s, 0) = p.angVelPred.x;
        angVelPred(s, 1) = p.angVelPred.y;
        angVelPred(s, 2) = p.angVelPred.z;
        invInertia(s, 0) = p.invInertia.x;
        invInertia(s, 1) = p.invInertia.y;
        invInertia(s, 2) = p.invInertia.z;
        quat(s, 0) = p.quat.x;
        quat(s, 1) = p.quat.y;
        quat(s, 2) = p.quat.z;
        quat(s, 3) = p.quat.w;
        quatPred(s, 0) = p.quatPred.x;
        quatPred(s, 1) = p.quatPred.y;
        quatPred(s, 2) = p.quatPred.z;
        quatPred(s, 3) = p.quatPred.w;
        scale(s) = p.scale;
        invMass(s) = p.invMass;
        shapeId(s) = p.shape;
        realIndices(s) = s;
        gid(s) = p.gid;
        materialId(s) = p.material;
        grounded(s) = p.grounded;
        orphan(s) = p.orphan;
        orphanVPeak(s) = p.orphanVPeak;
      });
}

/// Owner<->ghost halo driver for the distributed Kokkos demStep. Set up once (initMpi), then each
/// substep: gather() (rebuild + populate ghost slots) and per-iteration forward/forwardPositions.
class ParticleHalo {
 public:
  // Block decomposition over the GLOBAL domain (the per-block solver stays non-periodic; the halo
  // supplies the periodic wrap). gsize is the ORB cell grid. Mirrors MpiParticleHalo::init.
  void initMpi(std::array<double, 3> origin, std::array<double, 3> size, std::array<long, 3> gsize,
               std::array<bool, 3> periodic, MPI_Comm comm) {
    comm_ = comm;
    int sz = 1;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &sz);
    dec_.init(static_cast<std::size_t>(sz), peclet::core::IVec<3>{gsize[0], gsize[1], gsize[2]});
    peclet::core::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gsize[i]);
      map.periodic[i] = periodic[i];
    }
    map_ = map;
    mig_.init(dec_, rank_, map, comm_);
    halo_.init(mig_);
    inited_ = true;
  }
  // Shared-decomposition overload: adopt an EXTERNALLY-built ORB (so dem shares one BlockDecomposer
  // with the flow solver in a coupled run, and migrateTo() can move onto a re-decomposed
  // partition). `size`/`origin` map the ORB cell grid (= dec.globalSize()) to the physical domain.
  void initMpi(const peclet::core::decomp::BlockDecomposer<3>& dec, std::array<double, 3> origin,
               std::array<double, 3> size, std::array<bool, 3> periodic, MPI_Comm comm) {
    comm_ = comm;
    MPI_Comm_rank(comm_, &rank_);
    dec_ = dec;
    const auto& gs = dec.globalSize();
    peclet::core::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gs[i]);
      map.periodic[i] = periodic[i];
    }
    map_ = map;
    mig_.init(dec_, rank_, map, comm_);
    halo_.init(mig_);
    inited_ = true;
  }
  const peclet::core::decomp::BlockDecomposer<3>& decomposer() const { return dec_; }

  bool inited() const { return inited_; }
  int rank() const { return rank_; }
  int numGhost() const { return numGhost_; }

  /// Verlet-skin ghost reuse (D2): build the owner↔ghost topology with a band of `rcut + skin` and
  /// REUSE it across gather() calls until an owned particle has moved more than `skin` since the
  /// last build — only then download positions + rebuild the host topology + re-init the device
  /// halo. The per-substep forward exchange (the actual ghost-state move) always runs. `skin == 0`
  /// (the default) rebuilds every call — bit-identical to the original behaviour. A larger skin
  /// trades a thicker ghost band (more forwarded ghosts, more SoA capacity) for fewer host rebuilds
  /// + position D2Hs.
  void setVerletSkin(float skin) { verletSkin_ = skin < 0.0f ? 0.0f : skin; }
  float verletSkin() const { return verletSkin_; }
  /// Drop the cached owner↔ghost topology so the NEXT gather() rebuilds it unconditionally — the
  /// force-based step drives its own rebuild cadence (one gather per Verlet pair-list rebuild,
  /// possibly with a different band than the last build), independent of the skin heuristic.
  void invalidateTopology() { haveTopo_ = false; }
  /// Number of topology rebuilds vs total gather() calls since construction (for
  /// benchmarking/tests).
  long numRebuilds() const { return nRebuild_; }
  long numGathers() const { return nGather_; }

  // Rebuild the owner<->ghost correspondence over the current owned positions and populate the
  // ghost slots [numReal, numReal+numGhost) with the owners' full forwarded state. Sets
  // P.numParticles. Returns numReal+numGhost. Faithful port of Simulation::mpi_gather_ghosts.
  int gather(Particles& P, double rcut) {
    const int no = P.numReal;
    ++nGather_;

    // Decide whether to rebuild the owner↔ghost topology or reuse the cached one (Verlet skin, D2).
    // A rebuild is forced when reuse is off (skin==0), on the first gather, when the owned count
    // changed (a migration happened ⇒ topology invalid), or when an owned particle has displaced ≥
    // skin since the last build (a particle could have entered the rcut band without being in the
    // rcut+skin list).
    bool rebuild = (verletSkin_ <= 0.0f) || !haveTopo_ || (no != lastNumReal_);
    if (!rebuild && maxOwnedDisplacement(P.pos, no) >= verletSkin_)
      rebuild = true;
    numReal_ = no;

    if (rebuild) {
      ++nRebuild_;
      const double band = rcut + static_cast<double>(verletSkin_);
      // (1) download owned positions, (re)build the host halo topology, capture it on device.
      auto hpos = Kokkos::create_mirror_view(P.pos);
      Kokkos::deep_copy(hpos, P.pos);
      std::vector<peclet::core::Vec<3>> pv(static_cast<std::size_t>(no));
      for (int i = 0; i < no; ++i)
        pv[i] = peclet::core::Vec<3>{hpos(i, 0), hpos(i, 1), hpos(i, 2)};
      // includePeriodicSelf: a rank that owns a full (undecomposed) periodic axis -- a "x1" ORB
      // axis (e.g. z of a 2x2x1 layout) or np=1 -- is its own periodic image on that axis, so the
      // periodic neighbours are local self-ghosts the cross-rank exchange never makes. This
      // supplies them.
      halo_.build(pv, band, /*includePeriodicSelf=*/true);
      dev_.init(halo_);
      const int ng = static_cast<int>(halo_.numGhost());
      // The halo topology (forward / device self-gather) writes ALL ng ghost slots [no, no+ng); the
      // Particles SoA must have room for them. Silently truncating ng here would leave the halo
      // writing past the truncated count -> out-of-bounds SoA writes (memory corruption, not a
      // clean drop). So require adequate capacity and fail loudly instead. Size Simulation capacity
      // for the worst-case ghost band (a fully periodic box at rcut+skin needs a thick boundary
      // layer of ghosts).
      if (no + ng > P.capacity)
        throw std::runtime_error(
            "ParticleHalo::gather: ghost overflow -- need capacity >= " + std::to_string(no + ng) +
            " (numReal=" + std::to_string(no) + " + numGhost=" + std::to_string(ng) + "), have " +
            std::to_string(P.capacity) + "; increase the Simulation capacity.");
      numGhost_ = ng;
      allocBuffers(no, ng);
      uploadShift();
      // Snapshot the owned positions at build time — the reference for the displacement check.
      if (verletSkin_ > 0.0f) {
        if (refPos_.extent(0) < static_cast<std::size_t>(no))
          refPos_ = V3("peclet::dem::halo::refPos", static_cast<std::size_t>(P.capacity));
        Kokkos::deep_copy(Kokkos::subview(refPos_, std::pair<int, int>(0, no), Kokkos::ALL),
                          Kokkos::subview(P.pos, std::pair<int, int>(0, no), Kokkos::ALL));
      }
      haveTopo_ = true;
      lastNumReal_ = no;
    }

    const int ng = numGhost_;
    P.numParticles = no + ng;
    // self-map realIndices for the reals (owner deltas land on themselves); done every step like
    // demStep.
    selfMapReals(P.realIndices, no);
    if (ng == 0)
      return no;

    // (2) positions (committed + predicted) with the periodic image shift. d_pos_pred was already
    // advanced by predict_velocity, so it is forwarded too (NOT copied from pos) -- see CUDA
    // comment.
    forwardPositions(P.pos);
    forwardPositions(P.posPred);

    // (3) all other state packed into one record -> single exchange -> ghost slots + self-mapped
    // idx.
    haloPackGather(P.vel, P.velPred, P.angVel, P.angVelPred, P.invInertia, P.quat, P.quatPred,
                   P.scale, P.invMass, P.shapeId, P.gid, P.materialId, P.groundedLevel,
                   P.bodyOrphan, P.bodyOrphanVPeak, ownedPack_, no);
    dev_.forward(ownedPack_, ghostPack_);
    haloUnpackGather(P.vel, P.velPred, P.angVel, P.angVelPred, P.invInertia, P.quat, P.quatPred,
                     P.scale, P.invMass, P.shapeId, P.realIndices, P.gid, P.materialId,
                     P.groundedLevel, P.bodyOrphan, P.bodyOrphanVPeak, ghostPack_, no, ng);
    return no + ng;
  }

  MPI_Comm comm() const { return comm_; }

  // Dynamic load re-balance: re-decompose the ORB by per-block particle COUNT (weighted ORB) and
  // migrate each owned particle, with its committed state, to its new owner. A pure redistribution
  // of the same global particle set (count conserved, per-particle state preserved) — only
  // ownership and this rank's owned slice change; the physics result is unchanged. Must be called
  // at a step boundary (committed pos/quat/vel/angVel valid; predicted/delta/ghost scratch are
  // rebuilt next gather()). Returns this rank's new owned count. No-op-safe at np=1.
  // Weighted re-decompose by particle count (the equal-cell ORB imbalances as particles cluster)
  // and migrate owners. dec_ updated in place; mig_ points to it. Returns the new owned count.
  int rebalance(Particles& P) {
    std::vector<peclet::core::Vec<3>> pos;
    std::vector<char> payload;
    packState(P, pos, payload);
    const std::size_t newN = peclet::core::halo::rebalanceByParticleCount(
        dec_, mig_, pos, payload, sizeof(MigratePack), comm_);
    if ((int)newN > P.capacity)
      throw std::runtime_error("ParticleHalo::rebalance: owned overflow -- rank received " +
                               std::to_string(newN) + " particles, capacity " +
                               std::to_string(P.capacity));
    unpackState(P, pos, payload, newN);
    return (int)newN;
  }
  // Migrate particles onto an EXTERNALLY-supplied decomposition (dynamic co-rebalancing: the same
  // BlockDecomposer the flow solver redistributes onto). Pure ownership move — counts/state
  // conserved. Must be called at a step boundary. No-op-safe at np=1.
  int migrateTo(Particles& P, const peclet::core::decomp::BlockDecomposer<3>& newDec) {
    std::vector<peclet::core::Vec<3>> pos;
    std::vector<char> payload;
    packState(P, pos, payload);
    dec_ =
        newDec;  // in place: mig_ still points at dec_; mig_.migrate() sends to dec_.ownerOf(...)
    const std::size_t newN = mig_.migrate(pos, payload, sizeof(MigratePack));
    if ((int)newN > P.capacity)
      throw std::runtime_error("ParticleHalo::migrateTo: owned overflow -- rank received " +
                               std::to_string(newN) + " particles, capacity " +
                               std::to_string(P.capacity));
    unpackState(P, pos, payload, newN);
    return (int)newN;
  }
  // Migrate onto the weighted ORB of per-cell weights `w` (global x-fastest, matching the ORB
  // grid). The Lagrangian half of the co-rebalance: dem builds the SAME deterministic partition
  // flow does from the same weight field, so no BlockDecomposer object crosses the language
  // boundary.
  int migrateToWeights(Particles& P, const std::vector<peclet::core::Real>& w) {
    int size = 1;
    MPI_Comm_size(comm_, &size);
    peclet::core::decomp::BlockDecomposer<3> newDec((std::size_t)size, dec_.globalSize(), w);
    return migrateTo(P, newDec);
  }

 private:
  // Download the committed state and pack it (position drives ownership; the rest is the payload).
  // Each particle also carries its slice of the persistent-contact ledger (see MigratePack).
  void packState(Particles& P, std::vector<peclet::core::Vec<3>>& pos, std::vector<char>& payload) {
    const int no = P.numReal;
    auto h_pos = Kokkos::create_mirror_view(P.pos);
    auto h_quat = Kokkos::create_mirror_view(P.quat);
    auto h_vel = Kokkos::create_mirror_view(P.vel);
    auto h_angVel = Kokkos::create_mirror_view(P.angVel);
    auto h_invI = Kokkos::create_mirror_view(P.invInertia);
    auto h_invM = Kokkos::create_mirror_view(P.invMass);
    auto h_scale = Kokkos::create_mirror_view(P.scale);
    auto h_tScale = Kokkos::create_mirror_view(P.targetScale);
    auto h_shape = Kokkos::create_mirror_view(P.shapeId);
    auto h_pf = Kokkos::create_mirror_view(P.planeFriction);
    auto h_gid = Kokkos::create_mirror_view(P.gid);
    auto h_mat = Kokkos::create_mirror_view(P.materialId);
    auto h_grd = Kokkos::create_mirror_view(P.groundedLevel);
    Kokkos::deep_copy(h_pos, P.pos);
    Kokkos::deep_copy(h_quat, P.quat);
    Kokkos::deep_copy(h_vel, P.vel);
    Kokkos::deep_copy(h_angVel, P.angVel);
    Kokkos::deep_copy(h_invI, P.invInertia);
    Kokkos::deep_copy(h_invM, P.invMass);
    Kokkos::deep_copy(h_scale, P.scale);
    Kokkos::deep_copy(h_tScale, P.targetScale);
    Kokkos::deep_copy(h_shape, P.shapeId);
    Kokkos::deep_copy(h_pf, P.planeFriction);
    Kokkos::deep_copy(h_gid, P.gid);
    Kokkos::deep_copy(h_mat, P.materialId);
    Kokkos::deep_copy(h_grd, P.groundedLevel);
    auto h_orp = Kokkos::create_mirror_view(P.bodyOrphan);
    auto h_ovp = Kokkos::create_mirror_view(P.bodyOrphanVPeak);
    Kokkos::deep_copy(h_orp, P.bodyOrphan);
    Kokkos::deep_copy(h_ovp, P.bodyOrphanVPeak);
    pos.assign((std::size_t)no, peclet::core::Vec<3>{});
    payload.assign((std::size_t)no * sizeof(MigratePack), 0);
    std::vector<MigratePack> packs((std::size_t)no);
    std::unordered_map<unsigned, int> gidToLocal;
    gidToLocal.reserve((std::size_t)no * 2);
    for (int i = 0; i < no; ++i) {
      pos[(std::size_t)i] = peclet::core::Vec<3>{h_pos(i, 0), h_pos(i, 1), h_pos(i, 2)};
      MigratePack& m = packs[(std::size_t)i];
      m.quat = F4{h_quat(i, 0), h_quat(i, 1), h_quat(i, 2), h_quat(i, 3)};
      m.vel = F3{h_vel(i, 0), h_vel(i, 1), h_vel(i, 2)};
      m.angVel = F3{h_angVel(i, 0), h_angVel(i, 1), h_angVel(i, 2)};
      m.invInertia = F3{h_invI(i, 0), h_invI(i, 1), h_invI(i, 2)};
      m.invMass = h_invM(i);
      m.scale = h_scale(i);
      m.targetScale = h_tScale(i);
      m.shapeId = h_shape(i);
      m.planeFric0 = h_pf(i, 0);
      m.planeFric1 = h_pf(i, 1);
      m.gid = h_gid(i);
      m.materialId = h_mat(i);
      m.groundedLevel = h_grd(i);
      m.orphan = h_orp(i);
      m.orphanVPeak = h_ovp(i);
      m.numWarm = 0;
      gidToLocal.emplace(static_cast<unsigned>(h_gid(i)), i);
    }
    // Distribute the previous-substep converged ledger onto its endpoint particles: each pair
    // rides on BOTH locally-owned endpoints (the redundant ghost-pair pattern means either owner
    // may need it; the unpack dedupes by key). Beyond kWarmCarryMax the lowest-|impulse| entry is
    // evicted — a dropped pair merely warm-starts cold on the receiving rank.
    if (P.prevPairCount > 0) {
      const int pc = P.prevPairCount;
      // Full-view mirrors (not row-range subviews): a row range of a LayoutLeft float*[3] view is
      // non-contiguous, and a device->host deep_copy of it has no copy mechanism on CUDA.
      auto h_k = Kokkos::create_mirror_view(P.prevPairKeys);
      auto h_l = Kokkos::create_mirror_view(P.prevLambda);
      auto h_lt = Kokkos::create_mirror_view(P.prevLambdaT);
      auto h_pi = Kokkos::create_mirror_view(P.prevPosImpulse);
      auto h_rb = Kokkos::create_mirror_view(P.prevRestBank);
      auto h_rv = Kokkos::create_mirror_view(P.prevRestVPeak);
      Kokkos::deep_copy(h_k, P.prevPairKeys);
      Kokkos::deep_copy(h_l, P.prevLambda);
      Kokkos::deep_copy(h_lt, P.prevLambdaT);
      Kokkos::deep_copy(h_pi, P.prevPosImpulse);
      Kokkos::deep_copy(h_rb, P.prevRestBank);
      Kokkos::deep_copy(h_rv, P.prevRestVPeak);
      auto attach = [&](int i, const WarmPairEntry& e, float w) {
        MigratePack& m = packs[(std::size_t)i];
        if (m.numWarm < kWarmCarryMax) {
          m.warm[m.numWarm++] = e;
          return;
        }
        int worst = 0;
        float worstW = 1e30f;
        for (int s = 0; s < kWarmCarryMax; ++s) {
          const float ws = std::fabs(m.warm[s].lambda) + std::fabs(m.warm[s].posImpulse);
          if (ws < worstW) {
            worstW = ws;
            worst = s;
          }
        }
        if (w > worstW)
          m.warm[worst] = e;
      };
      for (int e = 0; e < pc; ++e) {
        const unsigned long long k = h_k(e);
        if (k == ~0ull)
          continue;
        WarmPairEntry we;
        we.key = k;
        we.lambda = h_l(e);
        we.lambdaT[0] = h_lt(e, 0);
        we.lambdaT[1] = h_lt(e, 1);
        we.lambdaT[2] = h_lt(e, 2);
        we.posImpulse = h_pi(e);
        we.restBank = h_rb(e);
        we.restVPeak = h_rv(e);
        if (we.lambda == 0.0f && we.posImpulse == 0.0f && we.lambdaT[0] == 0.0f &&
            we.lambdaT[1] == 0.0f && we.lambdaT[2] == 0.0f && we.restBank == 0.0f)
          continue;  // dead entry: carrying it only evicts live ones
        const float w = std::fabs(we.lambda) + std::fabs(we.posImpulse) + std::fabs(we.restBank);
        const unsigned hi = static_cast<unsigned>(k >> 32);
        const unsigned lo = static_cast<unsigned>(k & 0xFFFFFFFFu);
        if (auto it = gidToLocal.find(hi); it != gidToLocal.end())
          attach(it->second, we, w);
        if (lo != 0xFFFFFFFFu)
          if (auto it = gidToLocal.find(lo); it != gidToLocal.end())
            attach(it->second, we, w);
      }
    }
    // Force-engine (Hertz–Mindlin) history. Pack the LIVE cached-pair springs (hertzKeys/hertzXi
    // hold the current values; keys are gid-based) onto their locally-owned endpoints, and each
    // particle's wall-history slots verbatim.
    if (P.hertzNumPairs > 0) {
      const int hn = P.hertzNumPairs;
      auto h_hk = Kokkos::create_mirror_view(P.hertzKeys);
      auto h_hx = Kokkos::create_mirror_view(P.hertzXi);
      Kokkos::deep_copy(h_hk, P.hertzKeys);
      Kokkos::deep_copy(h_hx, P.hertzXi);
      auto attachHertz = [&](int i, const HertzPairEntry& e, float w) {
        MigratePack& m = packs[(std::size_t)i];
        if (m.numHertz < kWarmCarryMax) {
          m.hertz[m.numHertz++] = e;
          return;
        }
        int worst = 0;
        float worstW = 1e30f;
        for (int s = 0; s < kWarmCarryMax; ++s) {
          const float ws = std::fabs(m.hertz[s].xi[0]) + std::fabs(m.hertz[s].xi[1]) +
                           std::fabs(m.hertz[s].xi[2]);
          if (ws < worstW) {
            worstW = ws;
            worst = s;
          }
        }
        if (w > worstW)
          m.hertz[worst] = e;
      };
      for (int e = 0; e < hn; ++e) {
        HertzPairEntry he;
        he.key = h_hk(e);
        he.xi[0] = h_hx(e, 0);
        he.xi[1] = h_hx(e, 1);
        he.xi[2] = h_hx(e, 2);
        if (he.xi[0] == 0.0f && he.xi[1] == 0.0f && he.xi[2] == 0.0f)
          continue;  // open / historyless pair: nothing worth carrying
        const float w = std::fabs(he.xi[0]) + std::fabs(he.xi[1]) + std::fabs(he.xi[2]);
        const unsigned hi = static_cast<unsigned>(he.key >> 32);
        const unsigned lo = static_cast<unsigned>(he.key & 0xFFFFFFFFu);
        if (auto it = gidToLocal.find(hi); it != gidToLocal.end())
          attachHertz(it->second, he, w);
        if (auto it = gidToLocal.find(lo); it != gidToLocal.end())
          attachHertz(it->second, he, w);
      }
    }
    {
      auto h_xw = Kokkos::create_mirror_view(P.hertzXiWall);
      auto h_sw = Kokkos::create_mirror_view(P.hertzSnWall);
      Kokkos::deep_copy(h_xw, P.hertzXiWall);
      Kokkos::deep_copy(h_sw, P.hertzSnWall);
      for (int i = 0; i < no; ++i)
        for (int wi = 0; wi < Particles::kHertzMaxWalls; ++wi) {
          const int slot = i * Particles::kHertzMaxWalls + wi;
          MigratePack& m = packs[(std::size_t)i];
          m.hertzXiWall[wi][0] = h_xw(slot, 0);
          m.hertzXiWall[wi][1] = h_xw(slot, 1);
          m.hertzXiWall[wi][2] = h_xw(slot, 2);
          m.hertzSnWall[wi] = h_sw(slot);
        }
    }
    for (int i = 0; i < no; ++i)
      std::memcpy(&payload[(std::size_t)i * sizeof(MigratePack)], &packs[(std::size_t)i],
                  sizeof(MigratePack));
  }
  // Unpack the migrated particles back into the SoA [0,newN) and upload; rebuild the rank's
  // persistent-contact ledger (prevPairKeys sorted + aligned impulse stores) from the union of the
  // arriving particles' carried slices.
  void unpackState(Particles& P, const std::vector<peclet::core::Vec<3>>& pos,
                   const std::vector<char>& payload, std::size_t newN) {
    auto h_pos = Kokkos::create_mirror_view(P.pos);
    auto h_quat = Kokkos::create_mirror_view(P.quat);
    auto h_vel = Kokkos::create_mirror_view(P.vel);
    auto h_angVel = Kokkos::create_mirror_view(P.angVel);
    auto h_invI = Kokkos::create_mirror_view(P.invInertia);
    auto h_invM = Kokkos::create_mirror_view(P.invMass);
    auto h_scale = Kokkos::create_mirror_view(P.scale);
    auto h_tScale = Kokkos::create_mirror_view(P.targetScale);
    auto h_shape = Kokkos::create_mirror_view(P.shapeId);
    auto h_pf = Kokkos::create_mirror_view(P.planeFriction);
    auto h_gid = Kokkos::create_mirror_view(P.gid);
    auto h_mat = Kokkos::create_mirror_view(P.materialId);
    auto h_grd = Kokkos::create_mirror_view(P.groundedLevel);
    auto h_orp = Kokkos::create_mirror_view(P.bodyOrphan);
    auto h_ovp = Kokkos::create_mirror_view(P.bodyOrphanVPeak);
    Kokkos::deep_copy(h_orp, P.bodyOrphan);  // slots past newN keep defined values
    Kokkos::deep_copy(h_ovp, P.bodyOrphanVPeak);
    std::vector<WarmPairEntry> ledger;
    ledger.reserve(newN * 4);
    std::vector<HertzPairEntry> hertzLedger;
    hertzLedger.reserve(newN * 4);
    auto h_xw = Kokkos::create_mirror_view(P.hertzXiWall);
    auto h_sw = Kokkos::create_mirror_view(P.hertzSnWall);
    Kokkos::deep_copy(h_xw, P.hertzXiWall);  // slots past newN keep defined values on CUDA mirrors
    Kokkos::deep_copy(h_sw, P.hertzSnWall);
    for (std::size_t i = 0; i < newN; ++i) {
      h_pos((int)i, 0) = pos[i][0];
      h_pos((int)i, 1) = pos[i][1];
      h_pos((int)i, 2) = pos[i][2];
      MigratePack m;
      std::memcpy(&m, &payload[i * sizeof(MigratePack)], sizeof(MigratePack));
      h_quat((int)i, 0) = m.quat.x;
      h_quat((int)i, 1) = m.quat.y;
      h_quat((int)i, 2) = m.quat.z;
      h_quat((int)i, 3) = m.quat.w;
      h_vel((int)i, 0) = m.vel.x;
      h_vel((int)i, 1) = m.vel.y;
      h_vel((int)i, 2) = m.vel.z;
      h_angVel((int)i, 0) = m.angVel.x;
      h_angVel((int)i, 1) = m.angVel.y;
      h_angVel((int)i, 2) = m.angVel.z;
      h_invI((int)i, 0) = m.invInertia.x;
      h_invI((int)i, 1) = m.invInertia.y;
      h_invI((int)i, 2) = m.invInertia.z;
      h_invM((int)i) = m.invMass;
      h_scale((int)i) = m.scale;
      h_tScale((int)i) = m.targetScale;
      h_shape((int)i) = m.shapeId;
      h_pf((int)i, 0) = m.planeFric0;
      h_pf((int)i, 1) = m.planeFric1;
      h_gid((int)i) = m.gid;
      h_mat((int)i) = m.materialId;
      h_grd((int)i) = m.groundedLevel;
      h_orp((int)i) = m.orphan;
      h_ovp((int)i) = m.orphanVPeak;
      for (int s = 0; s < (int)m.numWarm && s < kWarmCarryMax; ++s)
        ledger.push_back(m.warm[s]);
      for (int s = 0; s < (int)m.numHertz && s < kWarmCarryMax; ++s)
        hertzLedger.push_back(m.hertz[s]);
      for (int wi = 0; wi < Particles::kHertzMaxWalls; ++wi) {
        const int slot = (int)i * Particles::kHertzMaxWalls + wi;
        h_xw(slot, 0) = m.hertzXiWall[wi][0];
        h_xw(slot, 1) = m.hertzXiWall[wi][1];
        h_xw(slot, 2) = m.hertzXiWall[wi][2];
        h_sw(slot) = m.hertzSnWall[wi];
      }
    }
    Kokkos::deep_copy(P.pos, h_pos);
    Kokkos::deep_copy(P.quat, h_quat);
    Kokkos::deep_copy(P.vel, h_vel);
    Kokkos::deep_copy(P.angVel, h_angVel);
    Kokkos::deep_copy(P.invInertia, h_invI);
    Kokkos::deep_copy(P.invMass, h_invM);
    Kokkos::deep_copy(P.scale, h_scale);
    Kokkos::deep_copy(P.targetScale, h_tScale);
    Kokkos::deep_copy(P.shapeId, h_shape);
    Kokkos::deep_copy(P.planeFriction, h_pf);
    Kokkos::deep_copy(P.gid, h_gid);
    Kokkos::deep_copy(P.materialId, h_mat);
    Kokkos::deep_copy(P.groundedLevel, h_grd);
    Kokkos::deep_copy(P.bodyOrphan, h_orp);
    Kokkos::deep_copy(P.bodyOrphanVPeak, h_ovp);

    // Ledger rebuild: sort by key, dedupe (a pair arrives once per locally-received endpoint; the
    // duplicates carry identical values), clamp to the store capacity, upload sorted + aligned —
    // exactly the layout the warm-start gather's binary search expects.
    std::sort(ledger.begin(), ledger.end(),
              [](const WarmPairEntry& a, const WarmPairEntry& b) { return a.key < b.key; });
    ledger.erase(
        std::unique(ledger.begin(), ledger.end(),
                    [](const WarmPairEntry& a, const WarmPairEntry& b) { return a.key == b.key; }),
        ledger.end());
    const int nl = std::min<int>((int)ledger.size(), (int)P.prevPairKeys.extent(0));
    {
      auto hk = Kokkos::create_mirror_view(P.prevPairKeys);
      auto hl = Kokkos::create_mirror_view(P.prevLambda);
      auto hlt = Kokkos::create_mirror_view(P.prevLambdaT);
      auto hpi = Kokkos::create_mirror_view(P.prevPosImpulse);
      auto hrb = Kokkos::create_mirror_view(P.prevRestBank);
      auto hrv = Kokkos::create_mirror_view(P.prevRestVPeak);
      Kokkos::deep_copy(hk, P.prevPairKeys);  // preserve tail entries beyond nl
      Kokkos::deep_copy(hl, P.prevLambda);
      Kokkos::deep_copy(hlt, P.prevLambdaT);
      Kokkos::deep_copy(hpi, P.prevPosImpulse);
      Kokkos::deep_copy(hrb, P.prevRestBank);
      Kokkos::deep_copy(hrv, P.prevRestVPeak);
      for (int e = 0; e < nl; ++e) {
        hk(e) = ledger[(std::size_t)e].key;
        hl(e) = ledger[(std::size_t)e].lambda;
        hlt(e, 0) = ledger[(std::size_t)e].lambdaT[0];
        hlt(e, 1) = ledger[(std::size_t)e].lambdaT[1];
        hlt(e, 2) = ledger[(std::size_t)e].lambdaT[2];
        hpi(e) = ledger[(std::size_t)e].posImpulse;
        hrb(e) = ledger[(std::size_t)e].restBank;
        hrv(e) = ledger[(std::size_t)e].restVPeak;
      }
      Kokkos::deep_copy(P.prevPairKeys, hk);
      Kokkos::deep_copy(P.prevLambda, hl);
      Kokkos::deep_copy(P.prevLambdaT, hlt);
      Kokkos::deep_copy(P.prevPosImpulse, hpi);
      Kokkos::deep_copy(P.prevRestBank, hrb);
      Kokkos::deep_copy(P.prevRestVPeak, hrv);
    }
    P.prevPairCount = nl;

    // Force-engine history rebuild: wall slots verbatim per (new local index, wall); the pair
    // ledger sorted + deduped into the hertzPrev store (the exact layout hertzRebuildPairs'
    // key-carry binary search expects), and the cached pair LIST invalidated — local pair slots
    // reference pre-migration indices, so the next force step must rebuild (and re-gathers the
    // halo then). A fresh rank may never have allocated the prev store: size it here.
    Kokkos::deep_copy(P.hertzXiWall, h_xw);
    Kokkos::deep_copy(P.hertzSnWall, h_sw);
    std::sort(hertzLedger.begin(), hertzLedger.end(),
              [](const HertzPairEntry& a, const HertzPairEntry& b) { return a.key < b.key; });
    hertzLedger.erase(std::unique(hertzLedger.begin(), hertzLedger.end(),
                                  [](const HertzPairEntry& a, const HertzPairEntry& b) {
                                    return a.key == b.key;
                                  }),
                      hertzLedger.end());
    const int nh = (int)hertzLedger.size();
    if ((int)P.hertzPrevKeys.extent(0) < nh) {
      P.hertzPrevKeys = Kokkos::View<unsigned long long*, CpMem>("hertzPrevKeys", nh);
      P.hertzPrevXi = Kokkos::View<float* [3], CpMem>("hertzPrevXi", nh);
    }
    if (nh > 0) {
      auto hk = Kokkos::create_mirror_view(P.hertzPrevKeys);
      auto hx = Kokkos::create_mirror_view(P.hertzPrevXi);
      Kokkos::deep_copy(hk, P.hertzPrevKeys);
      Kokkos::deep_copy(hx, P.hertzPrevXi);
      for (int e = 0; e < nh; ++e) {
        hk(e) = hertzLedger[(std::size_t)e].key;
        hx(e, 0) = hertzLedger[(std::size_t)e].xi[0];
        hx(e, 1) = hertzLedger[(std::size_t)e].xi[1];
        hx(e, 2) = hertzLedger[(std::size_t)e].xi[2];
      }
      Kokkos::deep_copy(P.hertzPrevKeys, hk);
      Kokkos::deep_copy(P.hertzPrevXi, hx);
    }
    P.hertzPrevCount = nh;
    P.hertzNumPairs = -1;

    P.numReal = (int)newN;
    P.numParticles = (int)newN;
  }

 public:
  // owner slice [0,numReal) -> ghost slots [numReal,..), verbatim (velocity / angular velocity).
  void forward(V3 field) {
    if (numGhost_ == 0)
      return;
    haloPackF3(field, ownedF3_, numReal_);
    dev_.forward(ownedF3_, ghostF3_);
    haloUnpackF3(field, ghostF3_, shiftDev_, numReal_, numGhost_, /*doShift=*/false);
  }
  // owner slice -> ghost slots with the periodic image shift added (positions).
  void forwardPositions(V3 field) {
    if (numGhost_ == 0)
      return;
    haloPackF3(field, ownedF3_, numReal_);
    dev_.forward(ownedF3_, ghostF3_);
    haloUnpackF3(field, ghostF3_, shiftDev_, numReal_, numGhost_, /*doShift=*/true);
  }
  // owner slice -> ghost slots, verbatim (quaternions).
  void forward4(V4 field) {
    if (numGhost_ == 0)
      return;
    haloPackF4(field, ownedF4_, numReal_);
    dev_.forward(ownedF4_, ghostF4_);
    haloUnpackF4(field, ghostF4_, numReal_, numGhost_);
  }

  void selfMapReals(Vi realIndices, int no) {
    Kokkos::parallel_for(
        "peclet::dem::halo::selfMapReals", Kokkos::RangePolicy<CpExec>(0, no),
        KOKKOS_LAMBDA(int i) { realIndices(i) = i; });
  }

 private:
  void allocBuffers(int no, int ng) {
    // Exact-sized: ParticleHalo::forward host-stages a deep_copy into the ghost View, so
    // its extent must equal numGhost; owned is indexed by sendIdx in [0,numReal).
    ownedF3_ = peclet::core::View<F3>("peclet::dem::halo::ownedF3", no);
    ghostF3_ = peclet::core::View<F3>("peclet::dem::halo::ghostF3", ng);
    ownedF4_ = peclet::core::View<F4>("peclet::dem::halo::ownedF4", no);
    ghostF4_ = peclet::core::View<F4>("peclet::dem::halo::ghostF4", ng);
    ownedPack_ = peclet::core::View<MpiGatherPack>("peclet::dem::halo::ownedPack", no);
    ghostPack_ = peclet::core::View<MpiGatherPack>("peclet::dem::halo::ghostPack", ng);
  }
  // Max Euclidean displacement of any owned particle since the last topology build (device reduce +
  // one scalar read-back) — the Verlet-skin reuse criterion.
  // public: nvcc forbids an extended __host__ __device__ (KOKKOS_LAMBDA) lambda inside a private
  // method.
 public:
  float maxOwnedDisplacement(const V3& pos, int no) const {
    if (no <= 0 || refPos_.extent(0) < static_cast<std::size_t>(no))
      return 1e30f;
    float md = 0.0f;
    V3 p = pos, r = refPos_;
    Kokkos::parallel_reduce(
        "peclet::dem::halo::maxdisp", Kokkos::RangePolicy<CpExec>(0, no),
        KOKKOS_LAMBDA(const int i, float& m) {
          const float dx = p(i, 0) - r(i, 0), dy = p(i, 1) - r(i, 1), dz = p(i, 2) - r(i, 2);
          const float d = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
          if (d > m)
            m = d;
        },
        Kokkos::Max<float>(md));
    return md;
  }

 private:
  void uploadShift() {
    auto t = halo_.flatten();
    std::vector<F3> hs(t.shift.size());
    for (std::size_t i = 0; i < t.shift.size(); ++i)
      hs[i] = F3{static_cast<float>(t.shift[i][0]), static_cast<float>(t.shift[i][1]),
                 static_cast<float>(t.shift[i][2])};
    shiftDev_ = peclet::core::toDevice(hs, "peclet::dem::halo::shift");
  }

  bool inited_ = false;
  int rank_ = 0, numReal_ = 0, numGhost_ = 0;
  // Verlet-skin reuse state (D2): skin width, the build-time owned positions, and the cache-valid
  // flags.
  float verletSkin_ = 0.0f;
  bool haveTopo_ = false;
  int lastNumReal_ = -1;
  long nRebuild_ = 0, nGather_ = 0;
  V3 refPos_;
  MPI_Comm comm_ = MPI_COMM_NULL;
  peclet::core::decomp::BlockDecomposer<3> dec_;
  peclet::core::halo::DomainMap<3> map_;  // physical<->cell mapping (for migrateTo)
  peclet::core::halo::ParticleMigrator<3> mig_;
  peclet::core::halo::ParticleHaloTopology<3> halo_;
  peclet::core::halo::ParticleHalo<3> dev_;
  peclet::core::View<F3> ownedF3_, ghostF3_, shiftDev_;
  peclet::core::View<F4> ownedF4_, ghostF4_;
  peclet::core::View<MpiGatherPack> ownedPack_, ghostPack_;
};

}  // namespace peclet::dem

#endif  // PECLET_DEM_MPI
#endif  // PECLET_DEM_MPI_HALO_HPP
