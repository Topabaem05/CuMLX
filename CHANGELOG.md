# Changelog

## 0.4.0 (2026-11-02)

The first release of the porting parity verifier. CUMLX does not convert models. It proves that an MLX port computes the same as the original PyTorch model and locates the first module that differs.

### Added

- `cumlx capture`, `diff`, `report`: golden traces, isolated replay of every module, chain comparison in the native dtype, and JSON, Markdown and HTML reports.
- Replay of attention and decoder layers with a causal mask, of the decode phase on the port's own KV cache, and of MoE experts on the golden routing.
- `cumlx weights`: static comparison that names transposed, sign-flipped, rescaled and reordered weights, per expert for MoE.
- `cumlx conform`: FP4 decode conformance of MLX on CPU and GPU. `--checkpoint` compares a ModelOpt NVFP4 checkpoint with an MLX repack, byte by byte.
- `cumlx tracediff`: how far a golden moves between devices.
- `cumlx.check_op`: parity of one op, for hand-written kernels.
- `cumlx selftest`: 11 injected faults must be located in the current environment.
- `action.yml`, `cumlx-mcp`, `skills/cumlx-verifier`, and an agent-friendly CLI contract (`cumlx.result.v1`).
- `--device` and `--tf32`. MLX runs float32 matmul in TF32 by default on M5 and CUDA, so comparisons default to full float32 and record the measured error.
- Thresholds from measured clean ports. Each run sets them from its median leaf error, so a defect cannot loosen its own threshold.
- `docs/TRACE-FORMAT.md`: the trace specification.

### Changed

- The kernel lab accepts MLX 0.32.2 and later 0.32.x. Its 35 GPU tests passed on Apple M5 Pro with 0.32.2 and 0.32.4.
- The FP4 kernel lab moved to `docs/FP4-KERNEL-LAB.md`. The README is organized by use case.

### Known limits

- Not run: CUDA goldens, Apple GPU generations other than M5, trained MoE models, padded inputs, models that are not LLMs.
- The thresholds come from two model families on one machine.
- Not published to PyPI.
