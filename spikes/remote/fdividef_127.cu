// __fdividef(127.0f, |h|) for every fp16 bit pattern h, as bitsandbytes' kInt8VectorQuant computes its row scale
// (csrc/kernels.cu: `const float scale = __fdividef(127.0f, smem_row_absmax);`, where the absmax is the float of an fp16).
// Writes 65536 little-endian float32 values, indexed by the 16 bits of h. Run by spikes/remote/m6_fdiv.sh.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

__global__ void scale_table(float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) {
        __half h = __ushort_as_half(static_cast<unsigned short>(i));
        float absmax = fabsf(__half2float(h));
        out[i] = __fdividef(127.0f, absmax);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s OUT.bin\n", argv[0]); return 2; }
    float* device = nullptr;
    float* host = static_cast<float*>(malloc(65536 * sizeof(float)));
    if (cudaMalloc(&device, 65536 * sizeof(float)) != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); return 3; }
    scale_table<<<256, 256>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(host, device, 65536 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "kernel or copy failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 3;
    }
    FILE* f = fopen(argv[1], "wb");
    if (!f || fwrite(host, sizeof(float), 65536, f) != 65536) { fprintf(stderr, "write failed\n"); return 4; }
    fclose(f);
    printf("wrote 65536 values to %s\n", argv[1]);
    return 0;
}
