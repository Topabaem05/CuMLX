---
name: cumlx-verifier
description: Verify that an MLX port computes the same as its original PyTorch model and locate the first module that differs. Use after converting or porting a model to MLX, when an MLX model gives different results from the original, or before publishing a port.
---

# Verify an MLX port with cumlx

Prove a port before you report it as correct. A successful conversion or a plausible generated text is not proof.

## Procedure

1. Check the environment. Follow `errors[].hint` if the command fails.

   ```bash
   cumlx doctor --require capture,diff,weights
   ```

2. Check the weights. This takes seconds and names transposed, rescaled and reordered weights.

   ```bash
   cumlx weights <original> <port>
   ```

3. Record a golden trace from the original.

   ```bash
   cumlx capture <original> --prompt "The capital of France is" -o golden/
   ```

   For a quantized port, add `--weights-from-mlx <port>`. Add it also for a low-precision port if step 4 reports `W_WEIGHT_ROUNDING_NOISE`.

4. Compare the port.

   ```bash
   cumlx diff golden/ <port> -o report/
   ```

5. Branch on `exit_code`.

   | exit_code | Meaning | Action |
   |---|---|---|
   | 0 | Every compared module passed | Report the coverage and `data.execution` with the result |
   | 1 | Diverged | Read `data.first_divergence`, fix that module, and run again |
   | 5 | Incomplete | This is not a pass. Map the modules in `data.unverified` with `--mapping`, or ignore them explicitly |
   | 2, 3, 4 | Usage, environment, input | Follow `errors[].hint` |

## Read the first divergence

- `basis: isolated`, `confidence: high`: the module itself is wrong on its golden input. This includes RoPE, masks and residuals.
- `phase: decode`: prefill is correct, and the KV cache handling is wrong (position offset, mask, cache update).
- `basis: chain+isolated`: the module could not be replayed, and its own code between its submodules is wrong.
- `basis: chain` in `--mode chain`: rerun with `--mode isolated`.
- A name that ends in `.mlp.experts`: the experts are wrong on the golden routing. A name that ends in `.mlp.gate`: the router is wrong. `cumlx weights` names the expert.

## Do not

- Do not report `incomplete`, a skipped check or an unverified module as a pass.
- Do not raise `--rel-l2` to clear a failure. Find the cause first.
- Do not treat `chain_drift` as a defect. It is rounding in the native dtype. `data.summary.native_output` shows the effect on predictions.
- Do not compare a quantized port with a reference golden and call the difference a bug. Use `--weights-from-mlx`.
- Do not pass `--tf32 on` unless you measure the default MLX behavior on purpose. It raises every threshold.

## MCP

`cumlx-mcp` serves the same commands as tools named `cumlx_doctor`, `cumlx_weights`, `cumlx_capture`, `cumlx_diff` and so on. Each tool returns the same JSON result as the CLI.
