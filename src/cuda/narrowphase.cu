#include "ParticleSystem.cuh"
#include "shapes/sdf_cylinder.cuh"
#include <cmath>
#include <cuda_runtime.h>

// Transforms a point from World to Local frame of particle
__device__ inline float3 world_to_local(float3 p, float3 pos, float4 quat) {
  float3 rel = make_float3(p.x - pos.x, p.y - pos.y, p.z - pos.z);

  // Inverse rotation (conjugate quaternion for unit quat)
  float4 q_inv = make_float4(-quat.x, -quat.y, -quat.z, quat.w);

  // Rotate rel by q_inv
  // v' = q * v * q_inv
  // Standard implementation details omitted for brevity, using simple formula
  float ix = q_inv.w * rel.x + q_inv.y * rel.z - q_inv.z * rel.y;
  float iy = q_inv.w * rel.y + q_inv.z * rel.x - q_inv.x * rel.z;
  float iz = q_inv.w * rel.z + q_inv.x * rel.y - q_inv.y * rel.x;
  float iw = -q_inv.x * rel.x - q_inv.y * rel.y - q_inv.z * rel.z;

  float3 res;
  res.x = ix * q_inv.w + iw * -q_inv.x + iy * -q_inv.z - iz * -q_inv.y;
  res.y = iy * q_inv.w + iw * -q_inv.y + iz * -q_inv.x - ix * -q_inv.z;
  res.z = iz * q_inv.w + iw * -q_inv.z + ix * -q_inv.y - iy * -q_inv.x;

  return res;
}

// Rotate vector from Local to World
__device__ inline float3 local_to_world_vec(float3 v, float4 q) {
  float ix = q.w * v.x + q.y * v.z - q.z * v.y;
  float iy = q.w * v.y + q.z * v.x - q.x * v.z;
  float iz = q.w * v.z + q.x * v.y - q.y * v.x;
  float iw = -q.x * v.x - q.y * v.y - q.z * v.z;

  float3 res;
  res.x = ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y;
  res.y = iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z;
  res.z = iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x;
  return res;
}

__global__ void generate_contacts_kernel(
    ParticleSystemData ps,
    int *potential_collisions, // List of pairs from Broadphase (placeholder for
                               // now)
    int num_potential, ShapeData shapes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_potential)
    return;

  // For Phase 2, we assume we have a list of pairs (A, B)
  // Here we implement the logic for ONE pair.
  // In reality, this would be part of a loop over neighbors found by cuBQL
  // query.

  // Placeholder logic for Phase 2 scaffold
  // 1. Load A and B
  // 2. Loop over points of A
  // 3. Transform point to B's frame
  // 4. Eval SDF(B)
  // 5. If dist < 0, generate constraint
}

// NOTE: Actual integration with cuBQL query results will happen in
// broadphase.cu This file is reserved for the complex geometry logic.
