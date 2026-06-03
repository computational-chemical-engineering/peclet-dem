// demgpu -- device-resident particle halo exchange (CUDA-aware MPI).
//
// The host-staged MpiParticleHalo forwards download the owned slice, exchange on the host, and upload
// the ghost slice -- a synchronous cudaMemcpy each way, ~18 per substep. This keeps the data on the
// GPU: a CUDA kernel gathers the sent owned values (at the halo's flattened send indices) into a
// contiguous device buffer, MPI transfers it device->device (needs CUDA-aware MPI), and the received
// ghosts land directly in the contiguous ghost region of the field (no scatter); for positions a
// kernel adds the per-ghost periodic image shift. Built from a tpx::halo::ParticleHalo<3>::flatten().
//
// Declarations only (no __global__), so this header is includable from host TUs (simulation.cpp); the
// kernels + device MPI live in device_particle_halo.cu (compiled by nvcc, linked against the MPI).
#pragma once
#ifdef DEMGPU_HAVE_TPX

#include <mpi.h>
#include <vector_types.h>

#include <vector>

#include "tpx/halo/particle_halo.hpp"

class DeviceParticleHalo {
 public:
  ~DeviceParticleHalo();

  // (Re)upload the send/recv topology and size the device send buffer. num_owned/num_ghost are the
  // owned and ghost counts (ghosts occupy field slots [num_owned, num_owned+num_ghost)).
  void setTopology(const tpx::halo::ParticleHalo<3>::FlatTopo& topo, int num_owned, int num_ghost,
                   MPI_Comm comm);

  int num_ghost() const { return n_ghost_; }

  // owned (field[0..num_owned)) -> ghost slots, fully on-device. forward4 is verbatim (velocity,
  // quaternion, ...); forwardPositions adds the periodic shift to xyz (.w carried verbatim).
  void forward4(float4* d_field);
  void forwardPositions(float4* d_field);

 private:
  void forwardImpl(float4* d_field, bool positions);

  MPI_Comm comm_ = MPI_COMM_WORLD;
  int n_owned_ = 0, n_ghost_ = 0, n_send_ = 0;
  std::vector<int> sendRanks_, sendCounts_, sendOffsets_;
  std::vector<int> recvRanks_, recvCounts_, recvOffsets_;
  int* d_sendIdx_ = nullptr;     // n_send_ ints: flattened owned indices to gather
  float4* d_shift_ = nullptr;    // n_ghost_: per-ghost xyz periodic shift (.w unused)
  float4* d_sendBuf_ = nullptr;  // n_send_: gathered owned values to send
  int cap_send_ = 0, cap_ghost_ = 0;
};

#endif  // DEMGPU_HAVE_TPX
