#pragma once
#include "../cuda/ParticleSystem.cuh" // For ShapeDescriptor, ShapeType
#include <cuda_runtime.h>
#include <vector>

namespace dem {

class ShapeManager {
public:
  ShapeManager();
  ~ShapeManager();

  // Create an Analytic Shape (Sphere, Cylinder)
  // Returns the shape_id
  int createAnalyticShape(ShapeType type, float4 params,
                          const std::vector<float4> &shell_points);

  // Create a Grid SDF Shape
  // int createGridShape(...); // Deferred

  // Upload Data to GPU
  // Must be called before simulation step if shapes changed
  void uploadToGPU();

  // Accessors
  ShapeDescriptor *getDeviceShapes() const { return d_shapes_; }
  int getNumShapes() const { return (int)shapes_.size(); }

private:
  // Host Data
  std::vector<ShapeDescriptor> shapes_;
  std::vector<float4> all_shell_points_;

  // Internal tracking for point ranges
  struct ShapePointRange {
    int start;
    int count;
  };
  std::vector<ShapePointRange> point_ranges_;

  // Device Data
  ShapeDescriptor *d_shapes_ = nullptr;
  float4 *d_points_ = nullptr;
  int capacity_shapes_ = 0;
  int capacity_points_ = 0;
};

} // namespace dem
