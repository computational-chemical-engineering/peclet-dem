#pragma once
#include <vector_types.h>

// Wrapper functions callable from standard C++ code
// that invoke the __host__ __device__ SDF functions
float evaluate_sdf_hollow_cylinder(float3 p, float4 params);
float evaluate_sdf_sphere(float3 p, float4 params);
