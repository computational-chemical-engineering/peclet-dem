/// @file
/// @brief dem — portable (Kokkos) owner<->ghost particle halo for the distributed XPBD step.
///
/// Kokkos counterpart of src/mpi/mpi_halo.h (MpiParticleHalo): a thin wrapper over transport-core's
/// tpx::halo::ParticleHalo<3> (host topology, periodic image shift) + DeviceParticleHaloKokkos<3>
/// (on-device gather/scatter + host-staged MPI). Rebuilt each substep from the owned positions; it
/// gathers ghost copies of the owners' FULL state into the Particles SoA ghost slots and refreshes
/// them owner->ghost during the velocity/position solves -- the EXACT distributed scheme (ghosts carry
/// REAL mass; every owned particle sees all its neighbours so it computes its full serial XPBD delta
/// locally; ghost deltas are discarded via a self-mapped realIndices).
///
/// Faithful Kokkos port of Simulation::mpi_gather_ghosts / mpi_forward_positions / mpi_forward4. Gated
/// behind DEM_MPI (mirrors cfd's CFD_MPI): the default dem module never
/// includes this, so it stays byte-identical when the macro is off.
#ifndef DEM_MPI_HALO_HPP
#define DEM_MPI_HALO_HPP
#ifdef DEM_MPI

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/common/view.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_halo.hpp"
#include "tpx/halo/particle_halo_kokkos.hpp"
#include "tpx/halo/particle_migrator.hpp"

#include "dem_portable.hpp"        // F3, F4
#include "particles.hpp"    // Particles, V3/V4/Vf/Vi, CpExec/CpMem

namespace dem {

// All non-position per-particle state in one record, so the substep gather is a single MPI exchange
// (latency dominates the host-staged path). Mirrors MpiParticleHalo::GatherPack; positions are
// forwarded separately (they need the periodic image shift). invMass is its own field here (the CUDA
// SoA carries it in pos.w; the Kokkos SoA keeps it in a separate array). POD => MPI_BYTE-copyable.
struct MpiGatherPack {
  F3 vel, velPred, angVel, angVelPred, invInertia;
  F4 quat, quatPred;
  float scale, invMass;
  int shape;
};

// --- free-function pack/unpack kernels (namespace scope: nvcc forbids KOKKOS_LAMBDA in member fns) ---

inline void haloPackF3(V3 field, tpx::View<F3> owned, int n) {
  Kokkos::parallel_for("dem::halo::packF3", Kokkos::RangePolicy<CpExec>(0, n),
                       KOKKOS_LAMBDA(int i) { owned(i) = F3{field(i, 0), field(i, 1), field(i, 2)}; });
}
inline void haloPackF4(V4 field, tpx::View<F4> owned, int n) {
  Kokkos::parallel_for(
      "dem::halo::packF4", Kokkos::RangePolicy<CpExec>(0, n),
      KOKKOS_LAMBDA(int i) { owned(i) = F4{field(i, 0), field(i, 1), field(i, 2), field(i, 3)}; });
}
// ghost[g] -> field(no+g,:), optionally adding the per-ghost periodic image shift (positions only).
inline void haloUnpackF3(V3 field, tpx::View<F3> ghost, tpx::View<F3> shift, int no, int ng,
                         bool doShift) {
  Kokkos::parallel_for("dem::halo::unpackF3", Kokkos::RangePolicy<CpExec>(0, ng),
                       KOKKOS_LAMBDA(int g) {
                         F3 v = ghost(g);
                         if (doShift) { v.x += shift(g).x; v.y += shift(g).y; v.z += shift(g).z; }
                         field(no + g, 0) = v.x; field(no + g, 1) = v.y; field(no + g, 2) = v.z;
                       });
}
inline void haloUnpackF4(V4 field, tpx::View<F4> ghost, int no, int ng) {
  Kokkos::parallel_for("dem::halo::unpackF4", Kokkos::RangePolicy<CpExec>(0, ng), KOKKOS_LAMBDA(int g) {
    field(no + g, 0) = ghost(g).x; field(no + g, 1) = ghost(g).y;
    field(no + g, 2) = ghost(g).z; field(no + g, 3) = ghost(g).w;
  });
}

inline void haloPackGather(V3 vel, V3 velPred, V3 angVel, V3 angVelPred, V3 invInertia, V4 quat,
                           V4 quatPred, Vf scale, Vf invMass, Vi shapeId,
                           tpx::View<MpiGatherPack> owned, int n) {
  Kokkos::parallel_for("dem::halo::packGather", Kokkos::RangePolicy<CpExec>(0, n), KOKKOS_LAMBDA(int i) {
    MpiGatherPack g;
    g.vel = F3{vel(i, 0), vel(i, 1), vel(i, 2)};
    g.velPred = F3{velPred(i, 0), velPred(i, 1), velPred(i, 2)};
    g.angVel = F3{angVel(i, 0), angVel(i, 1), angVel(i, 2)};
    g.angVelPred = F3{angVelPred(i, 0), angVelPred(i, 1), angVelPred(i, 2)};
    g.invInertia = F3{invInertia(i, 0), invInertia(i, 1), invInertia(i, 2)};
    g.quat = F4{quat(i, 0), quat(i, 1), quat(i, 2), quat(i, 3)};
    g.quatPred = F4{quatPred(i, 0), quatPred(i, 1), quatPred(i, 2), quatPred(i, 3)};
    g.scale = scale(i); g.invMass = invMass(i); g.shape = shapeId(i);
    owned(i) = g;
  });
}
// Unpack the gathered owner state into the ghost slots [no, no+ng) and self-map realIndices (the
// owner is remote, so velocity/position deltas landing on the ghost slot are discarded next forward).
inline void haloUnpackGather(V3 vel, V3 velPred, V3 angVel, V3 angVelPred, V3 invInertia, V4 quat,
                             V4 quatPred, Vf scale, Vf invMass, Vi shapeId, Vi realIndices,
                             tpx::View<MpiGatherPack> ghost, int no, int ng) {
  Kokkos::parallel_for("dem::halo::unpackGather", Kokkos::RangePolicy<CpExec>(0, ng), KOKKOS_LAMBDA(int g) {
    const MpiGatherPack p = ghost(g);
    const int s = no + g;
    vel(s, 0) = p.vel.x; vel(s, 1) = p.vel.y; vel(s, 2) = p.vel.z;
    velPred(s, 0) = p.velPred.x; velPred(s, 1) = p.velPred.y; velPred(s, 2) = p.velPred.z;
    angVel(s, 0) = p.angVel.x; angVel(s, 1) = p.angVel.y; angVel(s, 2) = p.angVel.z;
    angVelPred(s, 0) = p.angVelPred.x; angVelPred(s, 1) = p.angVelPred.y; angVelPred(s, 2) = p.angVelPred.z;
    invInertia(s, 0) = p.invInertia.x; invInertia(s, 1) = p.invInertia.y; invInertia(s, 2) = p.invInertia.z;
    quat(s, 0) = p.quat.x; quat(s, 1) = p.quat.y; quat(s, 2) = p.quat.z; quat(s, 3) = p.quat.w;
    quatPred(s, 0) = p.quatPred.x; quatPred(s, 1) = p.quatPred.y; quatPred(s, 2) = p.quatPred.z; quatPred(s, 3) = p.quatPred.w;
    scale(s) = p.scale; invMass(s) = p.invMass; shapeId(s) = p.shape;
    realIndices(s) = s;
  });
}

/// Owner<->ghost halo driver for the distributed Kokkos demStep. Set up once (initMpi), then each
/// substep: gather() (rebuild + populate ghost slots) and per-iteration forward/forwardPositions.
class KokkosParticleHalo {
 public:
  // Block decomposition over the GLOBAL domain (the per-block solver stays non-periodic; the halo
  // supplies the periodic wrap). gsize is the ORB cell grid. Mirrors MpiParticleHalo::init.
  void initMpi(std::array<double, 3> origin, std::array<double, 3> size, std::array<long, 3> gsize,
               std::array<bool, 3> periodic, MPI_Comm comm) {
    comm_ = comm;
    int sz = 1;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &sz);
    dec_.init(static_cast<std::size_t>(sz), tpx::IVec<3>{gsize[0], gsize[1], gsize[2]});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gsize[i]);
      map.periodic[i] = periodic[i];
    }
    mig_.init(dec_, rank_, map, comm_);
    halo_.init(mig_);
    inited_ = true;
  }

  bool inited() const { return inited_; }
  int rank() const { return rank_; }
  int numGhost() const { return numGhost_; }

  // Rebuild the owner<->ghost correspondence over the current owned positions and populate the ghost
  // slots [numReal, numReal+numGhost) with the owners' full forwarded state. Sets P.numParticles.
  // Returns numReal+numGhost. Faithful port of Simulation::mpi_gather_ghosts.
  int gather(Particles& P, double rcut) {
    const int no = P.numReal;
    numReal_ = no;

    // (1) download owned positions, (re)build the host halo topology, capture it on device.
    auto hpos = Kokkos::create_mirror_view(P.pos);
    Kokkos::deep_copy(hpos, P.pos);
    std::vector<tpx::Vec<3>> pv(static_cast<std::size_t>(no));
    for (int i = 0; i < no; ++i) pv[i] = tpx::Vec<3>{hpos(i, 0), hpos(i, 1), hpos(i, 2)};
    // includePeriodicSelf: a rank that owns a full (undecomposed) periodic axis -- a "x1" ORB axis
    // (e.g. z of a 2x2x1 layout) or np=1 -- is its own periodic image on that axis, so the periodic
    // neighbours are local self-ghosts the cross-rank exchange never makes. This supplies them.
    halo_.build(pv, rcut, /*includePeriodicSelf=*/true);
    dev_.init(halo_);
    const int ng = static_cast<int>(halo_.numGhost());
    // The halo topology (forward / device self-gather) writes ALL ng ghost slots [no, no+ng); the
    // Particles SoA must have room for them. Silently truncating ng here would leave the halo writing
    // past the truncated count -> out-of-bounds SoA writes (memory corruption, not a clean drop). So
    // require adequate capacity and fail loudly instead. Size Simulation capacity for the worst-case
    // ghost band (a fully periodic box at rcut needs a thick boundary layer of ghosts).
    if (no + ng > P.capacity)
      throw std::runtime_error(
          "KokkosParticleHalo::gather: ghost overflow -- need capacity >= " + std::to_string(no + ng) +
          " (numReal=" + std::to_string(no) + " + numGhost=" + std::to_string(ng) + "), have " +
          std::to_string(P.capacity) + "; increase the Simulation capacity.");
    numGhost_ = ng;
    P.numParticles = no + ng;

    // self-map realIndices for the reals (owner deltas land on themselves); done every step like demStep.
    selfMapReals(P.realIndices, no);
    if (ng == 0) return no;

    allocBuffers(no, ng);
    uploadShift();

    // (2) positions (committed + predicted) with the periodic image shift. d_pos_pred was already
    // advanced by predict_velocity, so it is forwarded too (NOT copied from pos) -- see CUDA comment.
    forwardPositions(P.pos);
    forwardPositions(P.posPred);

    // (3) all other state packed into one record -> single exchange -> ghost slots + self-mapped idx.
    haloPackGather(P.vel, P.velPred, P.angVel, P.angVelPred, P.invInertia, P.quat, P.quatPred, P.scale,
                   P.invMass, P.shapeId, ownedPack_, no);
    dev_.forward(ownedPack_, ghostPack_);
    haloUnpackGather(P.vel, P.velPred, P.angVel, P.angVelPred, P.invInertia, P.quat, P.quatPred,
                     P.scale, P.invMass, P.shapeId, P.realIndices, ghostPack_, no, ng);
    return no + ng;
  }

  // owner slice [0,numReal) -> ghost slots [numReal,..), verbatim (velocity / angular velocity).
  void forward(V3 field) {
    if (numGhost_ == 0) return;
    haloPackF3(field, ownedF3_, numReal_);
    dev_.forward(ownedF3_, ghostF3_);
    haloUnpackF3(field, ghostF3_, shiftDev_, numReal_, numGhost_, /*doShift=*/false);
  }
  // owner slice -> ghost slots with the periodic image shift added (positions).
  void forwardPositions(V3 field) {
    if (numGhost_ == 0) return;
    haloPackF3(field, ownedF3_, numReal_);
    dev_.forward(ownedF3_, ghostF3_);
    haloUnpackF3(field, ghostF3_, shiftDev_, numReal_, numGhost_, /*doShift=*/true);
  }
  // owner slice -> ghost slots, verbatim (quaternions).
  void forward4(V4 field) {
    if (numGhost_ == 0) return;
    haloPackF4(field, ownedF4_, numReal_);
    dev_.forward(ownedF4_, ghostF4_);
    haloUnpackF4(field, ghostF4_, numReal_, numGhost_);
  }

  void selfMapReals(Vi realIndices, int no) {
    Kokkos::parallel_for("dem::halo::selfMapReals", Kokkos::RangePolicy<CpExec>(0, no),
                         KOKKOS_LAMBDA(int i) { realIndices(i) = i; });
  }

 private:
  void allocBuffers(int no, int ng) {
    // Exact-sized: DeviceParticleHaloKokkos::forward host-stages a deep_copy into the ghost View, so
    // its extent must equal numGhost; owned is indexed by sendIdx in [0,numReal).
    ownedF3_ = tpx::View<F3>("dem::halo::ownedF3", no);
    ghostF3_ = tpx::View<F3>("dem::halo::ghostF3", ng);
    ownedF4_ = tpx::View<F4>("dem::halo::ownedF4", no);
    ghostF4_ = tpx::View<F4>("dem::halo::ghostF4", ng);
    ownedPack_ = tpx::View<MpiGatherPack>("dem::halo::ownedPack", no);
    ghostPack_ = tpx::View<MpiGatherPack>("dem::halo::ghostPack", ng);
  }
  void uploadShift() {
    auto t = halo_.flatten();
    std::vector<F3> hs(t.shift.size());
    for (std::size_t i = 0; i < t.shift.size(); ++i)
      hs[i] = F3{static_cast<float>(t.shift[i][0]), static_cast<float>(t.shift[i][1]),
                 static_cast<float>(t.shift[i][2])};
    shiftDev_ = tpx::toDevice(hs, "dem::halo::shift");
  }

  bool inited_ = false;
  int rank_ = 0, numReal_ = 0, numGhost_ = 0;
  MPI_Comm comm_ = MPI_COMM_NULL;
  tpx::decomp::BlockDecomposer<3> dec_;
  tpx::halo::ParticleMigrator<3> mig_;
  tpx::halo::ParticleHalo<3> halo_;
  tpx::halo::DeviceParticleHaloKokkos<3> dev_;
  tpx::View<F3> ownedF3_, ghostF3_, shiftDev_;
  tpx::View<F4> ownedF4_, ghostF4_;
  tpx::View<MpiGatherPack> ownedPack_, ghostPack_;
};

}  // namespace dem

#endif  // DEM_MPI
#endif  // DEM_MPI_HALO_HPP
