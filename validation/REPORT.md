# CUMLX validation report

This file separates what ran from what did not. The newest evidence is first.

## Apple Silicon and real models (2026-11-02, local, uncommitted working tree on `eaa132c`)

Environment: Apple M5 Pro (`applegpu_g17s`), macOS 27.0.1, Python 3.13.15, torch 2.15.0, transformers 5.22.1,
mlx 0.32.4 (0.32.2 for the kernel lab), mlx-lm 0.32.2. This is a local run, not a GitHub Actions result.

| Check | Observed result | Boundary |
|---|---|---|
| `pytest -m 'not gpu'` | 170 passed, 35 deselected | One Mac, Metal default device |
| `pytest -m e2e --require-e2e` | 31 passed | Tiny fixtures; two tests also start TF32 subprocesses |
| `cumlx selftest` (auto, cpu, gpu) | 10 passed, 0 failed, 0 skipped on each | Tiny model |
| `pytest -m gpu --backend metal --require-gpu` | 35 passed, skip 0 | MLX 0.32.2, compile and numerics; not speed |
| The same GPU tests on MLX 0.32.4 | 35 passed, with TF32 on and off | The version pin was overridden for this probe only; the pin is unchanged |
| Real clean ports (`dogfood`) | 34 runs, 0 false alarms, 100% parametrized coverage | Qwen2.5-0.5B, SmolLM2-135M; prefill only |
| Real fault injection (`dogfood`) | 41 runs, 0 missed; 3 not applicable | F1-F8 and F10; no F9 (MoE) |
| Public port | `mlx-community/Qwen2.5-0.5B-bf16`: ok, same numbers as a local conversion | One port |
| `cumlx weights` | All clean ports pass; about 3.3 s for the 0.5B model | Does not check code; 1B model not measured |
| `cumlx conform --target mlx` | Exit 1 on the known NVFP4 `global_scale` convention; decode is bit-exact on CPU; GPU flushes 64 subnormal results | Finding about MLX, not a CUMLX failure |
| Ruff, compileall, kernel regeneration | Passed, no drift | Not a Metal or CUDA compile |
| sdist, wheel, `twine check`, isolated wheel smoke | Passed | Core wheel only; no extras installed |
| CUDA, other Apple GPU generations, MoE, KV-cache decode | **Not run** | |
| GitHub Actions on these changes | **Not run** (not pushed). The prior commit `eaa132c` passed every CI job, including the verifier job | |

Evidence: [real-model results](dogfood-macos-m5-pro.json), reproduced by `scripts/dogfood_real_model.py`.
The numbers and the reasoning for thresholds are in [the CLI document](../docs/CLI.md#검증-기록-2026-11-02).

### Defects that real hardware and real models exposed

| Found by | Defect | Fix | Regression test |
|---|---|---|---|
| Metal selftest | MLX default TF32 made a clean float32 port fail (7.5e-4 against a 1e-4 threshold) | `--tf32`, default off; measured error in the report | `test_multi_input_modules...`, `test_tf32_on...`, `test_full_precision_request...` |
| Metal conform | GPU subnormal flushing reported as a decode failure | Classified as `device_behavior`, only when the sign and the magnitude limit match | `test_conform_flush_classification...` |
| Qwen2.5 | `lm_head` tied to the embedding stayed `unverified` | Capture records the tie; diff replays `as_linear` | `test_tied_lm_head...` |
| Qwen2.5 bf16 | Clean bf16 port reported as diverged because of chain rounding | Attention and decoder replay; isolated in float32; `chain_drift` | `test_low_precision_port...`, `test_non_finite_native_output...` |
| Qwen2.5 4bit+bf16 | Golden weights about 0.5% off: `mx.dequantize` rounds in the dtype of its scales | Scales go to float32 before dequantize | `test_quantized_port_with_bf16_scales...` |
| Detection limits | A 1% weight error raised its own threshold through a max-based noise estimate, and passed | Median-based noise estimate | `test_one_defective_leaf...`, `test_noise_thresholds...` |
| transformers 5 | Trace `revision` was null | Read from the HF cache | Observed in the Qwen2.5 trace; no unit test |

---

## v0.3 repository import

Date: 2026-09-28. This report records the local pre-push verification, not a
GitHub Actions result. For remote checks, inspect the exact main commit in
[GitHub Actions](https://github.com/Topabaem05/CUMLX/actions/workflows/ci.yml).

| Check | Observed result | Boundary |
|---|---|---|
| CPU suite | 109 passed, 35 GPU tests deselected | Python 3.13.5, Linux x86_64 |
| Full suite | 109 passed, 35 skipped | All skips require MLX/GPU |
| Python compileall | Passed | Not CUDA or Metal compilation |
| Actual CUDA helper compiled as host C++ | Passed in CPU suite | Not NVRTC or GPU execution |
| Kernel regeneration | 18 source/header files byte-identical | Source consistency only |
| Wheel build | Passed using installed setuptools build backend | GPU source wheel, not compiled GPU binary |
| Installed-wheel smoke | Passed outside checkout in Python isolated mode | Local venv reused system NumPy; remote CI uses a clean venv |
| Local Ruff/build frontend install | Blocked by local network DNS restriction | Remote CI installs and runs them separately |
| GPU execution and performance | Not run | No MLX/physical Metal or CUDA environment |

Evidence: [CPU log](local-cpu-pytest.txt), [wheel smoke](local-wheel-smoke.json),
[environment](local-environment.json). Original development reports and red/green
logs remain in the unchanged CUMLX-kernels-v0.3.zip; its hash and older archive
hashes are recorded in [provenance](../provenance/SOURCE-ARCHIVES.json).

No CUDA-facing PyTorch, external CUDA package certification, NxFP4 codec, native
FP4 MMA, or GPU speedup was added or certified by this repository import.

### Reproduce

```bash
python -m pip install -e '.[dev]'
python -m pytest -q -m 'not gpu'
# Real supported Apple GPU only:
python -m pip install -e '.[dev,metal]'
python -m pytest -q -m gpu --backend metal --require-gpu
```

The separate manual GPU workflow requires a trusted physical self-hosted runner;
missing hardware must fail instead of silently falling back to the CPU.
