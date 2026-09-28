// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// BM=8, BN=32, BK=32, 128 lanes; each lane accumulates two output columns.
// Shared memory: (8*32 + 32*32)*4 = 5120 bytes, independent of full model size.
unsigned int lane = threadIdx.x;
unsigned int group = blockIdx.x;
unsigned int tiles_n = (N + 31u) / 32u;
unsigned int m0 = (group / tiles_n) * 8u;
unsigned int n0 = (group % tiles_n) * 32u;
unsigned int row = lane / 16u, col = lane % 16u;
__shared__ float tile_a[8 * 32];
__shared__ float tile_w[32 * 32];
float acc0 = 0.0f, acc1 = 0.0f;
for (unsigned int k0 = 0u; k0 < K; k0 += 32u) {
    for (unsigned int t = lane; t < 8u * 32u; t += 128u) {
        unsigned int r = t / 32u, kk = t % 32u;
        float value = 0.0f;
        if (m0 + r < M && k0 + kk < K) {
            value = float(x[(m0 + r) * K + k0 + kk]);
        }
        tile_a[t] = value;
    }
    for (unsigned int t = lane; t < 32u * 32u; t += 128u) {
        unsigned int n = n0 + t / 32u, k = k0 + t % 32u;
        float value = 0.0f;
        if (n < N && k < K) {
            unsigned int pair = wq[n * WP + k / 2u];
            unsigned int code = (pair >> ((k & 1u) * 4u)) & 15u;
            value = cumlx_restore(code, ws[n * WSG + k / WG], wglobal[0], WF);
        }
        tile_w[t] = value;
    }
    __syncthreads();
    // There is no early return on partial M/N tiles: all threads must synchronize.
    for (unsigned int kk = 0u; kk < 32u; ++kk) {
        float a = tile_a[row * 32u + kk];
        acc0 = fmaf(a, tile_w[col * 32u + kk], acc0);
        acc1 = fmaf(a, tile_w[(col + 16u) * 32u + kk], acc1);
    }
    __syncthreads(); // Finish all reads before the next K tile overwrites shared memory.
}
if (m0 + row < M) {
    if (n0 + col < N) {
        if (HAS_BIAS) acc0 += bias[n0 + col];
        if (APPLY_SILU) acc0 = cumlx_silu(acc0);
        y[(m0 + row) * N + n0 + col] = acc0;
    }
    if (n0 + col + 16u < N) {
        if (HAS_BIAS) acc1 += bias[n0 + col + 16u];
        if (APPLY_SILU) acc1 = cumlx_silu(acc1);
        y[(m0 + row) * N + n0 + col + 16u] = acc1;
    }
}
