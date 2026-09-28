// CUMLX v0.3. Inference-only candidate; physical GPU validation required.
unsigned int i = thread_position_in_grid.x;
if (i >= SIZE) return;
float g = float(gate[i]), u = float(up[i]);
y[i] = cumlx_silu(g) * u;
