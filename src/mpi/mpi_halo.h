// demgpu — owner<->ghost particle halo for the distributed (MPI-aware) XPBD step.
//
// Thin wrapper over transport-core (sibling repo ../transport-core): a BlockDecomposer + a
// ParticleMigrator + a ParticleHalo, set up once (mpi_init) and rebuilt each substep from the owned
// particle positions. It provides host-staged "forward" of per-particle device-array fields from each
// owner onto its ghost copies:
//
//   forward_positions : owner xyz + periodic image shift  -> ghost xyz  (.w / inv_mass carried verbatim)
//   forward4 / forward_float / forward_int : verbatim      -> ghost      (velocity, quaternion, scale, ...)
//
// This realises the EXACT distributed scheme (see ../mpi/README.md): ghosts carry *real* mass, every
// owned particle sees all of its neighbours (owned or ghost) so it computes its complete serial XPBD
// delta locally, and the ghost copies are refreshed from their owners after each solver iteration.
// No constraint-delta reverse is needed — the only per-iteration communication is this forward.
//
// Host-staged (download owned slice -> NBX exchange -> upload ghost slice). Correctness-grade; a
// device-resident pack/persistent-exchange path is a later optimisation. Compiled only when
// transport-core is found at configure time (DEMGPU_HAVE_TPX, see CMakeLists.txt).
#pragma once
#ifdef DEMGPU_HAVE_TPX

#include <mpi.h>
#include <vector_types.h>

#include <array>
#include <vector>

#include "tpx/common/types.hpp"
#include "tpx/decomp/block_decomposer.hpp"
#include "tpx/halo/particle_halo.hpp"
#include "tpx/halo/particle_migrator.hpp"

#include "mpi/device_particle_halo.cuh"  // on-device gather + device-pointer MPI (CUDA-aware)

class MpiParticleHalo {
 public:
  // Set up the decomposition over the GLOBAL domain. `gsize` is the cell grid the ORB decomposer
  // splits across ranks; `periodic` is the global periodicity (per-block solver stays non-periodic,
  // the halo supplies the wrap). Uses MPI_COMM_WORLD (mpi4py initialises MPI on the Python side).
  void init(std::array<double, 3> origin, std::array<double, 3> size, std::array<long, 3> gsize,
            std::array<bool, 3> periodic) {
    int sz = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    dec_.init(static_cast<std::size_t>(sz), tpx::IVec<3>{gsize[0], gsize[1], gsize[2]});
    tpx::halo::DomainMap<3> map;
    for (int i = 0; i < 3; ++i) {
      map.origin[i] = origin[i];
      map.cellSize[i] = size[i] / static_cast<double>(gsize[i]);
      map.periodic[i] = periodic[i];
    }
    mig_.init(dec_, rank_, map, MPI_COMM_WORLD);
    halo_.init(mig_);
    inited_ = true;
  }

  bool inited() const { return inited_; }
  int rank() const { return rank_; }
  int owner_of(float x, float y, float z) const {
    return mig_.ownerOf(tpx::Vec<3>{x, y, z});
  }

  // (Re)establish the owner<->ghost correspondence from this rank's `n` owned positions (float4 .xyz).
  // Returns the number of ghosts this rank will receive. Call once per substep, after migration.
  int build(const float4* owned, int n, double rcut) {
    pv_.resize(n);
    for (int i = 0; i < n; ++i) pv_[i] = {owned[i].x, owned[i].y, owned[i].z};
    halo_.build(pv_, rcut);
    num_owned_ = n;
    num_ghost_ = static_cast<int>(halo_.numGhost());
    // upload the flattened topology for the on-device (CUDA-aware) forward path
    dev_.setTopology(halo_.flatten(), num_owned_, num_ghost_, halo_.comm());
    return num_ghost_;
  }

  int num_owned() const { return num_owned_; }
  int num_ghost() const { return num_ghost_; }

  // Device-resident forwards: owned in d_field[0..num_owned), ghosts in d_field[num_owned..]. Fully
  // on-device (gather kernel + device-pointer MPI), no host staging -- needs CUDA-aware MPI.
  void device_forward4(float4* d_field) { dev_.forward4(d_field); }
  void device_forward_positions(float4* d_field) { dev_.forwardPositions(d_field); }

  // owned positions -> ghost positions: xyz receives the periodic image shift, .w (inv_mass) verbatim.
  // `ghost` must have num_ghost() slots.
  void forward_positions(const float4* owned, float4* ghost) {
    vo_.resize(num_owned_);
    for (int i = 0; i < num_owned_; ++i) vo_[i] = {owned[i].x, owned[i].y, owned[i].z};
    vg_.assign(num_ghost_, tpx::Vec<3>{});
    halo_.forwardPositions(vo_.data(), vg_.data());

    fo_.resize(num_owned_);
    for (int i = 0; i < num_owned_; ++i) fo_[i] = owned[i].w;  // inv_mass
    fg_.assign(num_ghost_, 0.0f);
    halo_.forward(fo_.data(), fg_.data());

    for (int i = 0; i < num_ghost_; ++i) {
      ghost[i].x = static_cast<float>(vg_[i][0]);
      ghost[i].y = static_cast<float>(vg_[i][1]);
      ghost[i].z = static_cast<float>(vg_[i][2]);
      ghost[i].w = fg_[i];
    }
  }

  // Verbatim forwards for translation-invariant per-particle fields.
  void forward4(const float4* owned, float4* ghost) { halo_.forward(owned, ghost); }
  void forward_float(const float* owned, float* ghost) { halo_.forward(owned, ghost); }
  void forward_int(const int* owned, int* ghost) { halo_.forward(owned, ghost); }

  // All non-position per-particle state packed into one record, so the substep gather is a single
  // MPI exchange instead of one per field (latency, not bandwidth, dominates the host-staged path).
  struct GatherPack {
    float4 vel, vel_pred, quat, quat_pred, ang_vel, ang_vel_pred, inv_inertia;
    float scale;
    int shape;
  };
  void forward_pack(const GatherPack* owned, GatherPack* ghost) { halo_.forward(owned, ghost); }

 private:
  bool inited_ = false;
  int rank_ = 0, num_owned_ = 0, num_ghost_ = 0;
  tpx::decomp::BlockDecomposer<3> dec_;
  tpx::halo::ParticleMigrator<3> mig_;
  tpx::halo::ParticleHalo<3> halo_;
  DeviceParticleHalo dev_;
  std::vector<tpx::Vec<3>> pv_, vo_, vg_;
  std::vector<float> fo_, fg_;
};

#endif  // DEMGPU_HAVE_TPX
