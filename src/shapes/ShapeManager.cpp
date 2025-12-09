#include "ShapeManager.hpp"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK_SM(call)                                                    \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error in ShapeManager: " << cudaGetErrorString(err)   \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

namespace dem {

ShapeManager::ShapeManager() {}

ShapeManager::~ShapeManager() {
  if (d_shapes_)
    CUDA_CHECK_SM(cudaFree(d_shapes_));
  if (d_points_)
    CUDA_CHECK_SM(cudaFree(d_points_));
}

int ShapeManager::createAnalyticShape(ShapeType type, float4 params,
                                      const std::vector<float4> &shell_points) {
  ShapeDescriptor desc;
  desc.type = type;
  desc.params = params;
  desc.sdf_texture = 0;
  desc.aabb_min = make_float3(0, 0, 0);
  desc.aabb_max = make_float3(0, 0, 0); // Update if needed

  // Point Shell Logic
  int start_idx = (int)all_shell_points_.size();
  int count = (int)shell_points.size();

  all_shell_points_.insert(all_shell_points_.end(), shell_points.begin(),
                           shell_points.end());

  desc.num_points = count;
  // d_fine_points pointer will be set during upload

  ShapePointRange range = {start_idx, count};
  point_ranges_.push_back(range);

  int id = (int)shapes_.size();
  shapes_.push_back(desc);
  return id;
}

void ShapeManager::uploadToGPU() {
  // 1. Reallocate Points Buffer if needed
  int total_points = (int)all_shell_points_.size();
  if (total_points > capacity_points_) {
    if (d_points_)
      CUDA_CHECK_SM(cudaFree(d_points_));
    capacity_points_ = total_points + 1024; // Growth buffer
    CUDA_CHECK_SM(cudaMalloc(&d_points_, capacity_points_ * sizeof(float4)));
  }

  if (total_points > 0) {
    CUDA_CHECK_SM(cudaMemcpy(d_points_, all_shell_points_.data(),
                             total_points * sizeof(float4),
                             cudaMemcpyHostToDevice));
  }

  // 2. Update Descriptors with Device Pointers
  // We modify the Host vector temporarily or create a temp upload buffer?
  // We can modify 'shapes_' directly because d_fine_points is only valid on GPU
  // anyway.
  for (size_t i = 0; i < shapes_.size(); ++i) {
    int start = point_ranges_[i].start;
    int count = point_ranges_[i].count;
    if (count > 0) {
      shapes_[i].d_fine_points = d_points_ + start;
    } else {
      shapes_[i].d_fine_points = nullptr;
    }
  }

  // 3. Reallocate Descriptors Buffer
  int num_shapes = (int)shapes_.size();
  if (num_shapes > capacity_shapes_) {
    // ...
    if (d_shapes_)
      CUDA_CHECK_SM(cudaFree(d_shapes_));
    capacity_shapes_ = num_shapes + 32;
    CUDA_CHECK_SM(
        cudaMalloc(&d_shapes_, capacity_shapes_ * sizeof(ShapeDescriptor)));
  }

  if (num_shapes > 0) {
    CUDA_CHECK_SM(cudaMemcpy(d_shapes_, shapes_.data(),
                             num_shapes * sizeof(ShapeDescriptor),
                             cudaMemcpyHostToDevice));
  }
}

} // namespace dem
