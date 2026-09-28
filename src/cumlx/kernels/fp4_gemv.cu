// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// BM=1: GEMV. BM=4: reuse the decoded pair across up to four activation rows.
// One 128-lane workgroup owns one output column and BM rows. No warp assumptions.
unsigned int lane = threadIdx.x;
unsigned int group = blockIdx.x;
unsigned int n = group % N;
unsigned int m0 = (group / N) * BM;
__shared__ float partial[BM * 128];
float accum[BM];
for (unsigned int r = 0u; r < BM; ++r) accum[r] = 0.0f;
for (unsigned int k = lane * 2u; k < K; k += 256u) {
    unsigned int pair = wq[n * WP + k / 2u];
    unsigned int scale = ws[n * WSG + k / WG];
    float w0 = cumlx_restore(pair & 15u, scale, wglobal[0], WF);
    float w1 = k + 1u < K ? cumlx_restore(pair >> 4u, scale, wglobal[0], WF) : 0.0f;
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            float a0 = float(x[(m0 + r) * K + k]);
            accum[r] = fmaf(a0, w0, accum[r]);
            if (k + 1u < K) {
                float a1 = float(x[(m0 + r) * K + k + 1u]);
                accum[r] = fmaf(a1, w1, accum[r]);
            }
        }
    }
}
for (unsigned int r = 0u; r < BM; ++r) partial[r * 128u + lane] = accum[r];
__syncthreads();
// All 128 lanes execute every barrier, including lanes with no K contribution.
for (unsigned int step = 64u; step > 0u; step >>= 1u) {
    if (lane < step) {
        for (unsigned int r = 0u; r < BM; ++r)
            partial[r * 128u + lane] += partial[r * 128u + lane + step];
    }
    __syncthreads();
}
if (lane == 0u) {
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            float result = partial[r * 128u];
            if (HAS_BIAS) result += bias[n];
            if (APPLY_SILU) result = cumlx_silu(result);
            y[(m0 + r) * N + n] = result;
        }
    }
}
