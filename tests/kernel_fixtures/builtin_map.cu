// Written for CUMLX tests. Every built-in that `cumlx kernel port` rewrites, in one kernel, so that a run of the original
// and a run of the prepared source on one GPU show whether each replacement computes the same thing.
#include <math.h>
#include <cuda_fp16.h>

__global__ void builtin_map(const float* a, const float* b, const float* c, const int* m, const int* n, float* out, int* ints) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x = a[i], y = b[i], z = c[i];
    int p = m[i], q = n[i];
    out[i * 12u + 0u] = __fadd_rn(x, y);
    out[i * 12u + 1u] = __fsub_rn(x, y);
    out[i * 12u + 2u] = __fmul_rn(x, y);
    out[i * 12u + 3u] = __fdiv_rn(x, y);
    out[i * 12u + 4u] = __fmaf_rn(x, y, z);
    out[i * 12u + 5u] = __frcp_rn(y);
    out[i * 12u + 6u] = __fsqrt_rn(fabsf(x));
    out[i * 12u + 7u] = __frsqrt_rn(fabsf(y));
    out[i * 12u + 8u] = rsqrtf(fabsf(y));
    out[i * 12u + 9u] = __saturatef(x);
    out[i * 12u + 10u] = __fdividef(x, y);
    out[i * 12u + 11u] = (float)__half2float(__float2half(x));
    ints[i * 8u + 0u] = __clz(p);
    ints[i * 8u + 1u] = __popc((unsigned int)p);
    ints[i * 8u + 2u] = (int)__brev((unsigned int)p);
    ints[i * 8u + 3u] = __ffs(p);
    ints[i * 8u + 4u] = __hadd(p, q);
    ints[i * 8u + 5u] = __rhadd(p, q);
    ints[i * 8u + 6u] = __mulhi(p, q);
    ints[i * 8u + 7u] = (int)__umulhi((unsigned int)p, (unsigned int)q);
}
