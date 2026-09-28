// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// One scalar lane owns one block and its scale byte; writes never overlap.
// The global-amax reduction is supplied by the MLX graph, outside this kernel.
unsigned int block = thread_position_in_grid.x;
if (block >= N * SG) return;
unsigned int n = block / SG, b = block % SG, k0 = b * G;
float amax = 0.0f;
for (unsigned int j = 0u; j < G && k0 + j < K; ++j)
    amax = metal::max(amax, metal::abs(float(x[n * K + k0 + j])));
unsigned int sc;
if (F == 0) {
    float raw = (amax / global[0]) / 6.0f;
    sc = cumlx_encode_e4m3(raw);
} else {
    sc = amax == 0.0f ? 127u : cumlx_encode_e8m0_ceil(amax / 6.0f);
}
outs[block] = sc;
float divisor = cumlx_scale(sc, F);
for (unsigned int j = 0u; j < G && k0 + j < K; j += 2u) {
    float v0 = float(x[n * K + k0 + j]);
    float z0 = divisor > 0.0f ? (v0 / global[0]) / divisor : metal::copysign(0.0f, v0);
    unsigned int c0 = cumlx_encode_e2m1(metal::min(6.0f, metal::max(-6.0f, z0)));
    unsigned int c1 = 0u; // canonical zero high nibble for an odd K tail
    if (k0 + j + 1u < K) {
        float v1 = float(x[n * K + k0 + j + 1u]);
        float z1 = divisor > 0.0f ? (v1 / global[0]) / divisor : metal::copysign(0.0f, v1);
        c1 = cumlx_encode_e2m1(metal::min(6.0f, metal::max(-6.0f, z1)));
    }
    outq[n * P + (k0 + j) / 2u] = c0 | (c1 << 4u);
}
