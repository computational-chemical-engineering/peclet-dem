#ifdef DEMGPU_HAVE_TPX
#include <cuda_runtime.h>
#include <vector_functions.h>

#include <cstdio>
#include <vector>

#include "device_particle_halo.cuh"

namespace {
// gather owned values at the (flattened) send indices into a contiguous device send buffer
__global__ void gatherFloat4(const float4* __restrict__ field, const int* __restrict__ idx,
                             float4* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = field[idx[i]];
}
// add the per-ghost periodic image shift to the freshly-received ghost positions (xyz only)
__global__ void addShiftFloat4(float4* __restrict__ ghost, const float4* __restrict__ shift, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    ghost[i].x += shift[i].x;
    ghost[i].y += shift[i].y;
    ghost[i].z += shift[i].z;
  }
}
}  // namespace

void DeviceParticleHalo::setTopology(const tpx::halo::ParticleHalo<3>::FlatTopo& t, int no, int ng,
                                     MPI_Comm comm) {
  comm_ = comm;
  n_owned_ = no;
  n_ghost_ = ng;
  sendRanks_ = t.sendRanks;
  sendCounts_ = t.sendCounts;
  sendOffsets_ = t.sendOffsets;
  recvRanks_ = t.recvRanks;
  recvCounts_ = t.recvCounts;
  recvOffsets_.assign(t.recvOffsets.begin(), t.recvOffsets.end());
  n_send_ = static_cast<int>(t.sendIdx.size());

  if (n_send_ > cap_send_) {  // grow-only device buffers (rebuilt each substep, sizes vary)
    cudaFree(d_sendIdx_);
    cudaFree(d_sendBuf_);
    cudaMalloc(&d_sendIdx_, n_send_ * sizeof(int));
    cudaMalloc(&d_sendBuf_, n_send_ * sizeof(float4));
    cap_send_ = n_send_;
  }
  if (n_send_) {
    std::vector<int> idx(t.sendIdx.begin(), t.sendIdx.end());  // Index(int64) -> int
    cudaMemcpy(d_sendIdx_, idx.data(), n_send_ * sizeof(int), cudaMemcpyHostToDevice);
  }
  if (ng > cap_ghost_) {
    cudaFree(d_shift_);
    cudaMalloc(&d_shift_, ng * sizeof(float4));
    cap_ghost_ = ng;
  }
  if (ng) {
    std::vector<float4> sh(ng);
    for (int i = 0; i < ng; ++i)
      sh[i] = make_float4(static_cast<float>(t.shift[i][0]), static_cast<float>(t.shift[i][1]),
                          static_cast<float>(t.shift[i][2]), 0.0f);
    cudaMemcpy(d_shift_, sh.data(), ng * sizeof(float4), cudaMemcpyHostToDevice);
  }
}

void DeviceParticleHalo::forwardImpl(float4* d_field, bool positions) {
  const int blk = 256;
  if (n_send_)
    gatherFloat4<<<(n_send_ + blk - 1) / blk, blk>>>(d_field, d_sendIdx_, d_sendBuf_, n_send_);
  cudaDeviceSynchronize();  // pack must complete before MPI reads d_sendBuf_

  const int ns = static_cast<int>(sendRanks_.size());
  const int nr = static_cast<int>(recvRanks_.size());
  std::vector<MPI_Request> rreq(nr, MPI_REQUEST_NULL), sreq(ns, MPI_REQUEST_NULL);
  // receive DIRECTLY into the contiguous ghost region of the device field (no scatter)
  for (int p = 0; p < nr; ++p)
    MPI_Irecv(d_field + n_owned_ + recvOffsets_[p], recvCounts_[p] * static_cast<int>(sizeof(float4)),
              MPI_BYTE, recvRanks_[p], 9100, comm_, &rreq[p]);
  for (int k = 0; k < ns; ++k)
    MPI_Isend(d_sendBuf_ + sendOffsets_[k], sendCounts_[k] * static_cast<int>(sizeof(float4)),
              MPI_BYTE, sendRanks_[k], 9100, comm_, &sreq[k]);
  MPI_Waitall(nr, rreq.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(ns, sreq.data(), MPI_STATUSES_IGNORE);

  if (positions && n_ghost_)
    addShiftFloat4<<<(n_ghost_ + blk - 1) / blk, blk>>>(d_field + n_owned_, d_shift_, n_ghost_);
  cudaDeviceSynchronize();
}

void DeviceParticleHalo::forward4(float4* d_field) { forwardImpl(d_field, false); }
void DeviceParticleHalo::forwardPositions(float4* d_field) { forwardImpl(d_field, true); }

DeviceParticleHalo::~DeviceParticleHalo() {
  cudaFree(d_sendIdx_);
  cudaFree(d_shift_);
  cudaFree(d_sendBuf_);
}
#endif  // DEMGPU_HAVE_TPX
