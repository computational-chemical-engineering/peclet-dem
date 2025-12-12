#include "sdf_wrapper.h"
#include "shapes/sdf_analytic.cuh"

// These functions just forward to the CUDA/C++ shared implementation
// Since we added __host__ to the headers, this file can be compiled
// by NVCC and linked into the main application.

float evaluate_sdf_hollow_cylinder(float3 p, float4 params) {
  return dem::sdf_hollow_cylinder(p, params);
}

float evaluate_sdf_sphere(float3 p, float4 params) {
  return dem::sdf_sphere(p, params);
}
