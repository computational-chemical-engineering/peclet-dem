// Validate the device-resident particle halo (gather kernel + device-pointer MPI) against the
// already-validated host-staged forward, on the SAME halo. Both must produce identical ghost data.
//
// Build (against the CUDA-aware MPI; see transport-core/docs/cuda-aware-mpi.md):
//   source ~/opt/cudampi-env.sh
//   nvcc -DDEMGPU_HAVE_TPX=1 -std=c++17 -arch=sm_120 -O2 \
//        -I src -I ../transport-core/include -I $CUDAMPI_HOME/include \
//        mpi/test_device_halo.cu src/mpi/device_particle_halo.cu \
//        -L $CUDAMPI_HOME/lib -lmpi -o /tmp/test_device_halo
//   TPX_CUDA_AWARE_MPI=1 mpirun -x TPX_CUDA_AWARE_MPI -np 2 /tmp/test_device_halo
#include <cuda_runtime.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "mpi/mpi_halo.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  cudaSetDevice(ndev > 0 ? rank % ndev : 0);

  const double L = 8.0, rcut = 1.0;
  const int N = 240;
  std::array<double, 3> dmin{0, 0, 0}, dsize{L, L, L};
  std::array<long, 3> gsize{16, 16, 16};
  std::array<bool, 3> periodic{true, true, true};

  MpiParticleHalo halo;
  halo.init(dmin, dsize, gsize, periodic);

  // deterministic global particles; this rank owns those whose owner is `rank`
  std::vector<float4> owned;
  for (int i = 0; i < N; ++i) {
    // a fixed pseudo-random lattice, reproducible on every rank
    double x = std::fmod(0.13 + 0.61 * i, 1.0) * L;
    double y = std::fmod(0.27 + 0.37 * i, 1.0) * L;
    double z = std::fmod(0.51 + 0.19 * i, 1.0) * L;
    if (halo.owner_of((float)x, (float)y, (float)z) == rank)
      owned.push_back(make_float4((float)x, (float)y, (float)z, 1.0f + 0.01f * i));
  }
  const int no = (int)owned.size();
  const int ng = halo.build(owned.data(), no, rcut);

  // reference: host-staged forwards
  std::vector<float4> ref_v(ng), ref_p(ng);
  {
    std::vector<float4> vel(no);
    for (int i = 0; i < no; ++i) vel[i] = make_float4(owned[i].y, -owned[i].z, owned[i].x, 7.0f);
    halo.forward4(vel.data(), ref_v.data());
    halo.forward_positions(owned.data(), ref_p.data());
  }

  // device path: field = [owned | ghost]; fill owned, run device forward, read back ghost slots
  float4* d_field = nullptr;
  cudaMalloc(&d_field, (no + ng) * sizeof(float4));
  auto run_dev = [&](const std::vector<float4>& ownedVals, bool positions, std::vector<float4>& out) {
    cudaMemcpy(d_field, ownedVals.data(), no * sizeof(float4), cudaMemcpyHostToDevice);
    if (positions)
      halo.device_forward_positions(d_field);
    else
      halo.device_forward4(d_field);
    out.resize(ng);
    cudaMemcpy(out.data(), d_field + no, ng * sizeof(float4), cudaMemcpyDeviceToHost);
  };
  std::vector<float4> dev_v, dev_p, vel(no);
  for (int i = 0; i < no; ++i) vel[i] = make_float4(owned[i].y, -owned[i].z, owned[i].x, 7.0f);
  run_dev(vel, false, dev_v);
  run_dev(owned, true, dev_p);

  // compare
  double maxv = 0, maxp = 0;
  for (int i = 0; i < ng; ++i) {
    maxv = fmax(maxv, fmax(fabs(dev_v[i].x - ref_v[i].x),
                           fmax(fabs(dev_v[i].y - ref_v[i].y), fabs(dev_v[i].z - ref_v[i].z))));
    maxp = fmax(maxp, fmax(fabs(dev_p[i].x - ref_p[i].x),
                           fmax(fabs(dev_p[i].y - ref_p[i].y),
                                fmax(fabs(dev_p[i].z - ref_p[i].z), fabs(dev_p[i].w - ref_p[i].w)))));
  }
  double gmaxv = 0, gmaxp = 0;
  long gng = 0, lng = ng;
  MPI_Reduce(&maxv, &gmaxv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&maxp, &gmaxp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&lng, &gng, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    bool ok = gmaxv == 0.0 && gmaxp == 0.0;
    printf("np=%d ghosts(total)=%ld  device-vs-host forward4 max|d|=%.3e  forward_positions max|d|=%.3e"
           "  %s\n", size, gng, gmaxv, gmaxp, ok ? "OK (identical)" : "MISMATCH");
  }

  // --- profile: full host-staged forward (D2H + host MPI + H2D) vs device (kernel + device MPI) ---
  const int M = 400;
  std::vector<float4> oh(no), gh(ng);
  cudaMemcpy(d_field, vel.data(), no * sizeof(float4), cudaMemcpyHostToDevice);
  for (int m = 0; m < 30; ++m) {  // warmup (UCX IPC handle caching, page-in)
    cudaMemcpy(oh.data(), d_field, no * sizeof(float4), cudaMemcpyDeviceToHost);
    halo.forward4(oh.data(), gh.data());
    cudaMemcpy(d_field + no, gh.data(), ng * sizeof(float4), cudaMemcpyHostToDevice);
    halo.device_forward4(d_field);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  cudaDeviceSynchronize();
  double t0 = MPI_Wtime();
  for (int m = 0; m < M; ++m) {
    cudaMemcpy(oh.data(), d_field, no * sizeof(float4), cudaMemcpyDeviceToHost);
    halo.forward4(oh.data(), gh.data());
    cudaMemcpy(d_field + no, gh.data(), ng * sizeof(float4), cudaMemcpyHostToDevice);
  }
  cudaDeviceSynchronize();
  double t_host = (MPI_Wtime() - t0) / M * 1e3;  // ms / forward
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  for (int m = 0; m < M; ++m) halo.device_forward4(d_field);
  cudaDeviceSynchronize();
  double t_dev = (MPI_Wtime() - t0) / M * 1e3;
  double gth = 0, gtd = 0;
  MPI_Reduce(&t_host, &gth, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_dev, &gtd, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0)
    printf("  forward4 cost/call: host-staged %.4f ms  vs  device-resident %.4f ms  (%.1fx faster)\n",
           gth, gtd, gth / gtd);

  cudaFree(d_field);
  MPI_Finalize();
  return 0;
}
