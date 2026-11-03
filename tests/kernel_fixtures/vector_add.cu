// Written for CUMLX tests. A template kernel in one file with its host program: the first wall that M0 found.
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
__global__ void vector_add(const T* a, const T* b, T* out);

template <typename T>
__global__ void vector_add(const T* a, const T* b, T* out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = a[i] + b[i];
}

static void report(const std::vector<float>& values) {
    for (float v : values) std::cout << v << "\n";
}

int main() {
    const int n = 8;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    cudaMalloc(&a, n * sizeof(float));
    cudaMalloc(&b, n * sizeof(float));
    cudaMalloc(&c, n * sizeof(float));
    vector_add<float><<<1, n>>>(a, b, c);
    cudaDeviceSynchronize();
    std::vector<float> host(n);
    cudaMemcpy(host.data(), c, n * sizeof(float), cudaMemcpyDeviceToHost);
    report(host);
    return 0;
}
