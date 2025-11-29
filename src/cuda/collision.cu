// collision.cu - SDF point-shell collision logic (placeholder)
#include <cuda_runtime.h>
#include "memory_utils.cuh"

namespace demgpu {

__device__ float sampleSDF(float3 p) {
    // Placeholder sphere SDF centered at origin radius 1
    float len = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
    return len - 1.0f;
}

__global__ void collideKernel(float* px, float* py, float* pz, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float3 p{px[i], py[i], pz[i]};
    float d = sampleSDF(p);
    if (d < 0.f) {
        // push outward along gradient (approx using normalization)
        float len = sqrtf(p.x*p.x + p.y*p.y + p.z*p.z) + 1e-6f;
        px[i] = p.x / len;
        py[i] = p.y / len;
        pz[i] = p.z / len;
    }
}

void resolveCollisions(float* px, float* py, float* pz, int count) {
    int threads = 128;
    int blocks = (count + threads - 1) / threads;
    collideKernel<<<blocks, threads>>>(px, py, pz, count);
    cudaDeviceSynchronize();
}

} // namespace demgpu
