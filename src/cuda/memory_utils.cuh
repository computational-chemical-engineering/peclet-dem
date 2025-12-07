#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s in %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  }

template <typename T> void allocate_device(T *&ptr, size_t count) {
  CUDA_CHECK(cudaMalloc((void **)&ptr, count * sizeof(T)));
  CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
}

template <typename T> void free_device(T *&ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
  }
}
