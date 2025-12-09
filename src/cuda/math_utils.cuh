#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __CUDACC__
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
#endif

__device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(float3 v) { return sqrtf(dot(v, v)); }

__device__ inline float length(float2 v) {
  return sqrtf(v.x * v.x + v.y * v.y);
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
__device__ inline float3 vec_cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

// -----------------------------------------------------------------------------
// Quaternion Math
// -----------------------------------------------------------------------------
__device__ inline float3 rotate_vector(float4 q, float3 v) {
  float3 q_vec = make_float3(q.x, q.y, q.z);
  float3 t = 2.0f * vec_cross(q_vec, v);
  return v + q.w * t + vec_cross(q_vec, t);
}

__device__ inline float3 inv_rotate_vector(float4 q, float3 v) {
  float4 inv_q = make_float4(-q.x, -q.y, -q.z, q.w);
  return rotate_vector(inv_q, v);
}

#ifdef __CUDACC__
__device__ inline float atomicMinFloat(float *pt, float val) {
  int *address_as_i = (int *)pt;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ inline float atomicMaxFloat(float *pt, float val) {
  int *address_as_i = (int *)pt;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}
#endif
