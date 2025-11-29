#pragma once
// memory_utils.cuh - Helpers for cudaMalloc, SoA structs
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace demgpu {

inline void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Error: ") + msg + ": " + cudaGetErrorString(err));
    }
}

// Simple RAII device buffer
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { release(); }

    void allocate(size_t count) {
        release();
        count_ = count;
        cudaCheck(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc failed");
    }

    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }

    void upload(const T* host, size_t n) {
        if (n > count_) throw std::runtime_error("Upload exceeds buffer size");
        cudaCheck(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "Memcpy H2D failed");
    }

    void download(T* host, size_t n) const {
        if (n > count_) throw std::runtime_error("Download exceeds buffer size");
        cudaCheck(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost), "Memcpy D2H failed");
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace demgpu
