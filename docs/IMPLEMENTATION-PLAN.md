# CUMLX FP4 kernels implementation plan

**Goal:** Deliver inspectable NVFP4/MXFP4 codecs, packed GPU kernels, MLX launch wrappers, and honest verification evidence.
**Architecture:** Canonical row-major low-nibble-first packed inputs. Independent NumPy reference and explicit Metal/CUDA providers. No automatic CPU fallback, PyTorch masquerading, full-weight dequantization in the packed fast paths, or external checkpoint layout guessing.
**Tech stack:** Python >=3.10, NumPy, pytest; optional MLX 0.32.2. Metal Shading Language and CUDA C++ JIT function bodies.
**Spec:** Original v0.2 specification is in the conversation-provided `CUMLX-design-v0.2.zip`; its checksum is retained in `provenance/SOURCE-ARCHIVES.json`. The archive is not duplicated in Git.

## Global constraints and decisions
- Original archives remain byte-for-byte unchanged.
- This release implements the G1/G2 kernel slice, not G3/G4 CUDA-facing PyTorch.
- NVFP4: finite E2M1, unsigned E4M3 block scale, G=16, positive FP32 tensor reconstruction multiplier.
- MXFP4: finite E2M1, E8M0 block scale, G=32, no extra global scale.
- Canonical shape W[N,K], payload[N,ceil(K/2)], scales[N,ceil(K/G)]. Odd tails are zero padded.
- Each kernel uses FP32 weight reconstruction and FP32 accumulation; no native FP4 or Tensor Core claim.
- NxFP4/NF4 and producer-specific 2D/swizzled layouts are rejected explicitly, not reinterpreted.
- M,N,K > 0. Inference only. Single selected GPU; no external pointer/stream bridge.
- Metal/CUDA are device-unvalidated in this Linux CPU-only authoring environment.

## Review focus
1. Nonunit global scales, negative zero, exceptional FP8 scale codes, odd K and partial groups.
2. Reduction barriers: every launched lane reaches every barrier; edge lanes contribute zero.
3. Output ownership and complete coverage for tails, independent blocks, small-M reuse.
4. W4A4 must decode both quantized operands, not silently keep floating activations.
5. Timing must evaluate a fresh graph each iteration and include synchronization; no speed claim from source inspection.

## Tasks
1. Tests first: payload, scale, canonical descriptor validation, quantization, CPU reference, serialization. Then implement `quant.py`, `reference.py`.
2. Tests first: dispatch plans and scalar schedule simulations. Implement `planning.py`, `simulation.py`.
3. Write GPU integration tests before source: decode, GEMV/small-M, tiled GEMM, packed W4A4, block quantize, RMSNorm, SiLU multiply. Implement separate `.metal`/`.cu` bodies plus helpers.
4. Implement explicit MLX backend, prepared packed inputs, strict schema checks and optional native MLX QMM benchmark candidate. Test CPU-visible failure paths and packaging. GPU tests must skip (not pass) when unavailable.
5. Add deterministic CPU example, GPU validation runner, benchmark runner with numerical gates, docs/source references, build wheel and zip. Run entire suite, compileall, wheel content checks, isolated wheel smoke test, integrity checks.

## Execution record
- Initial evidence: Linux x86_64, MLX absent, `torch.cuda.is_available() == False`.
- Ruling: no external repository or GPU access was requested/provided; work in this isolated artifact directory without remote writes.
- Ruling: use explicit source-level portable shared-memory reductions before hardware-specific SIMD/matrix optimizations, because no Apple/NVIDIA device is available for execution validation.

## Completed implementation record (CPU/source scope only)
- Codec/reference tests: deliberate missing-module red -> implemented green.
- Launch planner/simulator: deliberate red -> schedule and coverage tests green.
- GPU bodies/integration tests: source written; device tests skipped only because MLX is absent.
- Wrapper preflight: deliberate missing-module red -> fail-closed/registry tests green.
- Reproduced tiny-scale normalization overflow warning; changed CPU/device encoder arithmetic order to explicit FP32 normalization with intentional saturation handling.
- Reproduced maximum-finite E2M1 saturation error in both NumPy and actual C++ helper; fixed by clamping magnitude before nearest-distance comparison. Regression tests retained.
- Ruling: CPU reference normalization and GPU candidate use FP32 arithmetic, while matrix oracle dot remains float64. Extreme subnormal behavior still requires device certification.
- Ruling: original NxFP4 research codec and CUDA-facing frontend remain unimplemented rather than fabricating a format or hardware identity.
- Ruling: 0.32.2 API pin is verified from public docs, not a tested GPU installation.

## Repository import

The historical execution record above describes artifact authoring. On 2026-09-28, the user explicitly requested a main-branch commit in Topabaem05/CUMLX. The import preserves kernel semantics, adds CPU/lint/build CI and a separate manual physical-GPU workflow. See [CI.md](CI.md) and [the current local validation report](../validation/REPORT.md).
