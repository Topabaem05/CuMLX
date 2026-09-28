#include <math.h>
#include <stdint.h>
// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
__device__ __forceinline__ float cumlx_e2m1(unsigned int c) {
    unsigned int e = (c & 7u) >> 1u;
    unsigned int m = c & 1u;
    float v = e == 0u ? 0.5f * float(m) : ldexpf(1.0f + 0.5f * float(m), int(e) - 1);
    return copysignf(v, (c & 8u) ? -1.0f : 1.0f);
}
__device__ __forceinline__ float cumlx_e4m3(unsigned int c) {
    if (c > 126u) return NAN;
    unsigned int e = c >> 3u, m = c & 7u;
    return e == 0u ? float(m) * 0.001953125f : ldexpf(1.0f + float(m) * 0.125f, int(e) - 7);
}
__device__ __forceinline__ float cumlx_e8m0(unsigned int c) {
    return c == 255u ? NAN : ldexpf(1.0f, int(c) - 127);
}
__device__ __forceinline__ float cumlx_scale(unsigned int c, int fmt) {
    return fmt == 0 ? cumlx_e4m3(c) : cumlx_e8m0(c);
}
__device__ __forceinline__ float cumlx_restore(unsigned int c, unsigned int s, float global, int fmt) {
    // Explicit order matches the reference decoder. No half-precision weight rounding.
    return (cumlx_e2m1(c) * cumlx_scale(s, fmt)) * global;
}
__device__ __forceinline__ unsigned int cumlx_encode_e2m1(float x) {
    float a = fminf(fabsf(x), 6.0f); // saturate before distance comparisons
    unsigned int best = 0u;
    float distance = a;
    for (unsigned int c = 1u; c < 8u; ++c) {
        float d = fabsf(a - cumlx_e2m1(c));
        if (d < distance || (d == distance && (c & 1u) == 0u)) {
            distance = d; best = c;
        }
    }
    return best | (signbit(x) ? 8u : 0u);
}
__device__ __forceinline__ unsigned int cumlx_encode_e4m3(float x) {
    if (x <= 0.0f) return 0u;
    if (x >= 448.0f) return 126u;
    unsigned int lo = 0u, hi = 126u;
    while (lo < hi) {
        unsigned int mid = (lo + hi) / 2u;
        if (cumlx_e4m3(mid) < x) lo = mid + 1u; else hi = mid;
    }
    unsigned int right = lo, left = right == 0u ? 0u : right - 1u;
    float dl = fabsf(x - cumlx_e4m3(left));
    float dr = fabsf(cumlx_e4m3(right) - x);
    unsigned int result = dr < dl || (dr == dl && (right & 1u) == 0u) ? right : left;
    // Pin the CUMLX nonzero-block scale floor; not an implicit vendor recipe.
    return result == 0u ? 1u : result;
}
__device__ __forceinline__ unsigned int cumlx_encode_e8m0_ceil(float x) {
    unsigned int lo = 0u, hi = 254u;
    while (lo < hi) {
        unsigned int mid = (lo + hi) / 2u;
        if (cumlx_e8m0(mid) < x) lo = mid + 1u; else hi = mid;
    }
    return lo;
}
__device__ __forceinline__ float cumlx_silu(float x) {
    float z = expf(-fabsf(x));
    float sigmoid = x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
    return x * sigmoid;
}
