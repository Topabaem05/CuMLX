// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
// Diagnostic full reconstruction only. Packed GEMV/GEMM never invoke this kernel.
unsigned int i = thread_position_in_grid.x;
if (i >= N * K) return;
unsigned int n = i / K, k = i % K;
unsigned int p = wq[n * WP + k / 2u];
unsigned int code = (p >> ((k & 1u) * 4u)) & 15u;
y[i] = cumlx_restore(code, ws[n * WSG + k / WG], wglobal[0], WF);
