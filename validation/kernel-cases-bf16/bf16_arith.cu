// The bfloat16 arithmetic of cuda_bf16.h, which needs sm80 or later. scripts/make_bf16_cases.py writes the inputs.
#include <cuda_bf16.h>

extern "C" __global__ void bf16_arith(const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c,
                                      __nv_bfloat16* add, __nv_bfloat16* sub, __nv_bfloat16* mul, __nv_bfloat16* fma,
                                      __nv_bfloat16* neg, __nv_bfloat16* abs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    add[i] = __hadd(a[i], b[i]);
    sub[i] = __hsub(a[i], b[i]);
    mul[i] = __hmul(a[i], b[i]);
    fma[i] = __hfma(a[i], b[i], c[i]);
    neg[i] = __hneg(a[i]);
    abs[i] = __habs(a[i]);
}
