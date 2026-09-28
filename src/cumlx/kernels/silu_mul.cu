// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
if (i >= SIZE) return;
float g = float(gate[i]), u = float(up[i]);
y[i] = cumlx_silu(g) * u;
