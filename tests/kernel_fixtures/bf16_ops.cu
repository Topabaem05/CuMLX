// bfloat16 conversions, arithmetic and a comparison, for tests/test_bf16.py. Its golden there comes from the model in
// cumlx.verify.bf16, not from a CUDA run.
#include <cuda_bf16.h>

extern "C" __global__ void bf16_ops(const __nv_bfloat16* x, const __nv_bfloat16* y, const float* f, __nv_bfloat16* fused,
                                    __nv_bfloat16* sum, __nv_bfloat16* toward_zero, unsigned int* less) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __nv_bfloat16 s = __float2bfloat16(f[0]);
    fused[i] = __hfma(s, x[i], y[i]);
    sum[i] = x[i] + y[i];
    toward_zero[i] = __float2bfloat16_rz(f[i]);
    less[i] = x[i] < y[i] ? 1u : 0u;
}
