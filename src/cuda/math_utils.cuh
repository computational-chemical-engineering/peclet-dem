#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ inline void atomicAddVector(float4 *address, float4 val) {
  float *f_addr = reinterpret_cast<float *>(address);
  atomicAdd(f_addr, val.x);
  atomicAdd(f_addr + 1, val.y);
  atomicAdd(f_addr + 2, val.z);
  atomicAdd(f_addr + 3, val.w);
}

__device__ inline void atomicAddVector3(float4 *address, float3 val) {
  float *f_addr = reinterpret_cast<float *>(address);
  atomicAdd(f_addr, val.x);
  atomicAdd(f_addr + 1, val.y);
  atomicAdd(f_addr + 2, val.z);
  // Ignore w
}

__device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator*(const float3 &a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float3 operator*(float s, const float3 &a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float3 operator/(const float3 &a, float s) {
  float inv = 1.0f / s;
  return a * inv;
}
__device__ inline void operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}
