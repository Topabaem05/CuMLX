// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// One workgroup per row. Residual addition and accumulation take place in FP32.
unsigned int lane = threadIdx.x, row = blockIdx.x;
__shared__ float square_sum[128];
float partial = 0.0f;
for (unsigned int k = lane; k < K; k += 128u) {
    float v = float(x[row * K + k]);
    if (HAS_RESIDUAL) v += float(residual[row * K + k]);
    partial = fmaf(v, v, partial);
}
square_sum[lane] = partial;
__syncthreads();
for (unsigned int step = 64u; step > 0u; step >>= 1u) {
    if (lane < step) square_sum[lane] += square_sum[lane + step];
    __syncthreads();
}
float inverse_rms = 1.0f / sqrtf(square_sum[0] / float(K) + epsilon[0]);
for (unsigned int k = lane; k < K; k += 128u) {
    float v = float(x[row * K + k]);
    if (HAS_RESIDUAL) v += float(residual[row * K + k]);
    y[row * K + k] = (v * inverse_rms) * float(weight[k]);
}
