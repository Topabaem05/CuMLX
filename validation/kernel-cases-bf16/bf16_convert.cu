// The four float-to-bfloat16 conversions of cuda_bf16.h. scripts/make_bf16_cases.py writes the inputs.
#include <cuda_bf16.h>

extern "C" __global__ void bf16_convert(const float* x, __nv_bfloat16* rn, __nv_bfloat16* rz, __nv_bfloat16* rd, __nv_bfloat16* ru) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    rn[i] = __float2bfloat16(x[i]);
    rz[i] = __float2bfloat16_rz(x[i]);
    rd[i] = __float2bfloat16_rd(x[i]);
    ru[i] = __float2bfloat16_ru(x[i]);
}
