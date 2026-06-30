#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", \
            err, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct CudaBuffer {
    void* dev = nullptr;
    size_t size = 0;

    bool alloc(size_t bytes) {
        if (dev) free();
        size = bytes;
        CUDA_CHECK(cudaMalloc(&dev, bytes));
        return dev != nullptr;
    }

    void free() {
        if (dev) {
            cudaFree(dev);
            dev = nullptr;
        }
        size = 0;
    }

    void to_device(const void* host, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
    }

    void to_host(void* host, size_t bytes) {
        CUDA_CHECK(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
    }

    ~CudaBuffer() { free(); }
};

#endif
