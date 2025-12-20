#include "OutputGenerator.hpp"
#include "../cuda/math_utils.cuh" // For helper types if needed
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include <algorithm> // Added

namespace dem {

// External Kernel Launchers
void launch_init_grid(float *d_grid, int *d_state, int3 dims, float3 origin,
                      float3 voxel_size);
void launch_splat_particles(ParticleSystemData ps, float *d_grid, int *d_state,
                            int3 dims, float3 origin, float3 voxel_size,
                            float3 d_min, float3 d_max);
void launch_eikonal_update(float *d_in, float *d_out, int *d_state, int3 dims,
                           float3 voxel_size, bool px, bool py, bool pz);

OutputGenerator::OutputGenerator(Simulation *sim)
    : sim_(sim), d_grid_ping_(nullptr), d_grid_pong_(nullptr),
      d_state_(nullptr), allocated_voxels_(0) {}

OutputGenerator::~OutputGenerator() { free(); }

void OutputGenerator::allocate(int3 dims) {
  size_t voxels = dims.x * dims.y * dims.z;
  if (voxels > allocated_voxels_) {
    free();
    cudaMalloc(&d_grid_ping_, voxels * sizeof(float));
    cudaMalloc(&d_grid_pong_, voxels * sizeof(float));
    cudaMalloc(&d_state_, voxels * sizeof(int));
    allocated_voxels_ = voxels;
  }
}

void OutputGenerator::free() {
  if (d_grid_ping_)
    cudaFree(d_grid_ping_);
  if (d_grid_pong_)
    cudaFree(d_grid_pong_);
  if (d_state_)
    cudaFree(d_state_);
  d_grid_ping_ = nullptr;
  d_grid_pong_ = nullptr;
  d_state_ = nullptr;
  allocated_voxels_ = 0;
}

void OutputGenerator::generateAndSaveVTI(const std::string &filename,
                                         int3 resolution, float3 bounds_min,
                                         float3 bounds_max) {
  allocate(resolution);

  float3 origin = bounds_min;
  float3 domain_size =
      make_float3(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y,
                  bounds_max.z - bounds_min.z);
  float3 voxel_size =
      make_float3(domain_size.x / resolution.x, domain_size.y / resolution.y,
                  domain_size.z / resolution.z);

  // 1. Init Grid (Ping)
  launch_init_grid(d_grid_ping_, d_state_, resolution, origin, voxel_size);
  cudaDeviceSynchronize();

  // 2. Splat Particles (Ping)
  // Access ps_ directly via friend class
  ParticleSystemData ps = sim_->ps_;

  launch_splat_particles(ps, d_grid_ping_, d_state_, resolution, origin,
                         voxel_size, bounds_min, bounds_max);
  cudaDeviceSynchronize();

  // 3. Eikonal Solve Loop (Ping-Pong)
  int iterations =
      std::max(std::max(resolution.x, resolution.y), resolution.z) *
      4; // Conservative
  // Limit for performance, FIM usually converges fast
  // iterations = 50; // Removed manual limit

  float *current_in = d_grid_ping_;
  float *current_out = d_grid_pong_;

  for (int i = 0; i < iterations; ++i) {
    launch_eikonal_update(current_in, current_out, d_state_, resolution,
                          voxel_size, ps.periodic_x, ps.periodic_y,
                          ps.periodic_z);
    std::swap(current_in, current_out);
  }
  // Final result is in current_in (because we swapped after LAST write to
  // current_out) Wait: loop writes In -> Out. Then Swap(In, Out). So In becomes
  // the new source (which was Out). After loop, the valid data is in
  // 'current_in'.

  // 4. Download
  size_t total_voxels = resolution.x * resolution.y * resolution.z;
  std::vector<float> h_grid(total_voxels);
  cudaMemcpy(h_grid.data(), current_in, total_voxels * sizeof(float),
             cudaMemcpyDeviceToHost);

  // 5. Write VTI
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << std::endl;
    return;
  }

  file << "<VTKFile type=\"ImageData\" version=\"1.0\" "
          "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
  file << "  <ImageData WholeExtent=\"0 " << resolution.x - 1 << " 0 "
       << resolution.y - 1 << " 0 " << resolution.z - 1 << "\""
       << " Origin=\"" << origin.x << " " << origin.y << " " << origin.z << "\""
       << " Spacing=\"" << voxel_size.x << " " << voxel_size.y << " "
       << voxel_size.z << "\">\n";
  file << "    <Piece Extent=\"0 " << resolution.x - 1 << " 0 "
       << resolution.y - 1 << " 0 " << resolution.z - 1 << "\">\n";
  file << "      <PointData Scalars=\"SDF\">\n";
  file << "        <DataArray type=\"Float32\" Name=\"SDF\" "
          "format=\"appended\" offset=\"0\"/>\n";
  file << "      </PointData>\n";
  file << "    </Piece>\n";
  file << "  </ImageData>\n";

  uint64_t num_bytes = total_voxels * sizeof(float);
  file << "  <AppendedData encoding=\"raw\">\n";
  file << "_"; // Spacer?
  file.write(reinterpret_cast<const char *>(&num_bytes), sizeof(uint64_t));
  file.write(reinterpret_cast<const char *>(h_grid.data()), num_bytes);
  file << "\n  </AppendedData>\n";
  file << "</VTKFile>\n";
  file.close();

  std::cout << "SDF exported to " << filename << std::endl;
}

std::vector<float> OutputGenerator::generateSDF(int3 resolution,
                                                float3 bounds_min,
                                                float3 bounds_max) {
  allocate(resolution);

  float3 origin = bounds_min;
  float3 domain_size =
      make_float3(bounds_max.x - bounds_min.x, bounds_max.y - bounds_min.y,
                  bounds_max.z - bounds_min.z);
  float3 voxel_size =
      make_float3(domain_size.x / resolution.x, domain_size.y / resolution.y,
                  domain_size.z / resolution.z);

  // 1. Init Grid (Ping)
  launch_init_grid(d_grid_ping_, d_state_, resolution, origin, voxel_size);
  cudaDeviceSynchronize();

  // 2. Splat Particles (Ping)
  ParticleSystemData ps = sim_->ps_;

  launch_splat_particles(ps, d_grid_ping_, d_state_, resolution, origin,
                         voxel_size, bounds_min, bounds_max);
  cudaDeviceSynchronize();

  // 3. Eikonal Solve Loop (Ping-Pong)
  int iterations =
      std::max(std::max(resolution.x, resolution.y), resolution.z) *
      4; // Conservative

  float *current_in = d_grid_ping_;
  float *current_out = d_grid_pong_;

  for (int i = 0; i < iterations; ++i) {
    launch_eikonal_update(current_in, current_out, d_state_, resolution,
                          voxel_size, ps.periodic_x, ps.periodic_y,
                          ps.periodic_z);
    std::swap(current_in, current_out);
  }

  // 4. Download
  size_t total_voxels = resolution.x * resolution.y * resolution.z;
  std::vector<float> h_grid(total_voxels);
  cudaMemcpy(h_grid.data(), current_in, total_voxels * sizeof(float),
             cudaMemcpyDeviceToHost);

  return h_grid;
}

} // namespace dem
