# CUMLX validation report

This file separates what ran from what did not. The newest evidence is first.

## Published MLX ports with a silent RoPE mismatch (2026-11-07 and 2026-11-08, local)

Environment: Apple M5 Pro, mlx 0.32.4, mlx-lm 0.32.2 (the latest release), torch 2.15.0, transformers 5.22.1.

| Check | Observed result | Boundary |
|---|---|---|
| Config scan (`upstream/config_scan.py`, `upstream/config-scan.json`) | 6000 recent `mlx`-tagged repositories listed, 4864 configs read, 587 with `rope_parameters`. In 8 ports from 7 uploaders, mlx-lm's `ModelArgs` takes a different RoPE base from the one transformers resolves from the same config (llama 6, mistral 1, qwen3_moe 1) | Config only; no model was run in this step. 195 configs name a model type that mlx-lm 0.32.2 does not implement |
| The change of ml-explore/mlx-lm#1964 on the same configs | Fixes 7 of the 8. It does not fix a config whose top-level `rope_theta` (1e6) disagrees with `rope_parameters` (1e7): transformers takes `rope_parameters`, and #1964 keeps the top-level value | Logic restated from the pull request's diff (`upstream/pr1964_logic.py`), not its branch |
| Published ports run against a golden (`upstream/port_rope_check.py`, `upstream/ports/`) | 3 finetune ports (two MiniCPM5-2B, one Voltaire-2.5) diverged at `model.layers.0.self_attn` (isolated, high). The same ports with only a top-level `rope_theta` added to config.json passed (0 fail, worst 8.0e-6, 5.1e-6, 4.0e-6) | The golden runs the original architecture from its config with the port's weights; 2 prompts. Voltaire names no original, so its own config stood for it |
| Larger published ports, first 2 decoder layers (`port_rope_check.py --layers 2`, 2026-11-08) | MiniCPM5-2B-Heretic-ARA, Sentiment-Analysis-7B and Nanbeige-4.3-33B (also `--vocab-from-prompts`, 11 of 166,912 tokens) diverged at `model.layers.0.self_attn` (isolated, high); controls passed (worst 4.6e-6, 2.6e-6, 2.9e-6). On Voltaire, the full check and both cuts gave the same rel_l2 (0.038986802532664726), and a port without the defect (Qwen2.5-0.5B-Instruct-4bit) passed both. The 7B published port, run whole in MLX, differs from its control in top-1 at 19% to 24% of positions (16 to 2048 tokens) | Verdicts cover the first 2 layers. The 8B LLM2Vec repo is not an mlx-lm port (`LlamaBiModel`, no `lm_head`, quantization outside config.json), so it is excluded; the 78 GB qwen3_moe port is config level only |
| Effect on longer input (`upstream/long_context_effect.py`) | On non-repeating text, the published port's top-1 differs from the control at 31% of positions at 16 tokens and 46% at 2048 | One port, one text; not task quality |
| Reported upstream | **No.** mlx-lm's contribution policy requires a person to write the report | |

## Apple Silicon and real models (2026-11-02, local)

Environment: Apple M5 Pro (`applegpu_g17s`), macOS 27.0.1, Python 3.13.15, torch 2.15.0, transformers 5.22.1,
mlx 0.32.4 (kernel tests also on 0.32.2), mlx-lm 0.32.2. These are local runs; CI results are named as such.

| Check | Observed result | Boundary |
|---|---|---|
| `pytest` (all markers) | 259 passed, 0 skipped | One Mac, Metal default device |
| `pytest -m e2e --require-e2e` with no GPU emulated (`gpu_available` false, CPU default) | 67 passed | Emulation on the Mac; the real Linux result is the CI job |
| README followed in a new clone and venv | doctor, selftest, weights, capture, diff, quantized path, conform, check_op and MCP all ran as documented | SmolLM2-135M; Python 3.12 |
| `pytest -m gpu --backend metal --require-gpu` | 36 passed, skip 0 on 0.32.4; the 35 kernel tests also passed on 0.32.2 | Compile and numerics; not speed |
| `cumlx selftest` (auto, cpu, gpu) | 11 passed, 0 failed, 0 skipped | Tiny models |
| Real clean ports (`dogfood`) | 38 runs, 0 false alarms, 100% parametrized coverage; 2 runs could not start (`gather_mm` on CPU needs float32) | Qwen2.5-0.5B, SmolLM2-135M, public tiny MoE; 5 prompts |
| Real fault injection (`dogfood`) | 50 runs, 0 missed; 3 not applicable | F1-F11 |
| Public ports (`public-ports.json`) | 9 of 10 `mlx-community` ports ok; 1 could not load (2024 file name) | One revision of each |
| NVFP4 repack against ModelOpt (`checkpoint-dory-nvfp4.json`) | 288 quantized tensors with identical payload and scales, worst 9.7e-8; 219 other tensors identical | One checkpoint |
| Real mismatch found (`upstream/`) | mlx-lm 0.32.2 silently ignores `rope_parameters` for llama, qwen2, mistral, phi3, starcoder2 | Tiny models; **not reported upstream** |
| Golden between devices (`cumlx tracediff`) | Repeat runs identical on CPU and on MPS; MPS against CPU worst 4.3e-5 | **CUDA not run** |
| Large capture (`large-capture-qwen2.5-7b.json`) | 7.6B parameters, 3 of 28 layers: 322 MB, 12.1 s | Random weights, bf16; architecture only |
| Agent over MCP (`mcp-agent-demo.md`) | Real agent located an injected fault in 4 turns | One run on a tiny fixture |
| GitHub Actions on `3584292` | CI: all 7 jobs passed, including the Linux verifier. Action test: clean port ok; faulty port failed at `model.layers.1.self_attn` | Tiny fixture |
| CUDA golden on Tesla V100 (`cuda-golden-v100.json`) | Repeat runs identical; CUDA against CPU worst 1.9e-5; against M5 Pro CPU 4.5e-5; an MLX bf16 port on the Mac passes against it | One model, one GPU |
| Recent HF models (`latest-models.json`) | 12 models: 7 ok on both the implementation and a 4-bit conversion; 4 refused as partial checkpoints; 1 not loadable by mlx-lm | Short prompts; one revision of each |
| `cumlx scan` on 5 repositories (`scan-cuda-repos.json`) | IsaacSim `platform`; warp, flash-attention, nvdiffrast, sam2 `kernel` | Text scan; nothing was built or run |
| `cumlx port` on SmolLM2-135M, 4 bits | ok: 730 pass, 0 fail, and the port generated text | One run |
| Isaac Sim on macOS | Install fails: no wheel exists, and the 6.0.0.0 stub raises `Didn't find wheel` | Executed with uv |
| Isaac Sim 6.2 in a Linux arm64 container on the Mac (`isaac-sim.json`) | Kit starts in about 30 s, CPU PhysX runs, and an Isaac Lab cartpole falls under gravity only. Repeat runs match. RTX and cameras fail, with about 600 error lines | No GPU; NVIDIA does not support this; 2 full runs |
| Isaac Sim 6.2 on the Tesla V100 (`isaac-sim.json`) | Kit and CPU PhysX run. RTX refuses the GPU (`Skipping unsupported non-RTX GPU`), and the camera returns nothing | 1 run |
| CPU PhysX, Mac container against V100 | Free fall height and 4 pole angles identical in every printed digit | arm64 against x86_64; 2 small scenes; GPU PhysX not run |
| Same cartpole USD, PhysX against Genesis | Pole at 1 s: 3.31 rad against 1.97 rad | Not the same physics; cause not investigated |
| Physics on this Mac | MuJoCo 3.17: about 722,000 steps/s on a cartpole. Newton 1.7 on Warp 1.19: a falling box rests at z = 0.2; `cuda:0` is refused | CPU only; one small scene each |
| Robot policy in MLX (`examples/robot-policy`) | rsl_rl default shape, batches 1, 16 and 4096: worst 5.2e-7; a wrong activation fails | Random weights; no real checkpoint |
| Platform candidates, M0 (`spikes.json`) | Mitsuba Metal against CPU: 0.008% mean difference, 52.1 dB. Quadrants Metal against CPU: 7.0e-8. Isaac Lab cartpole USD loads into Genesis and falls under gravity only. kernelport: 12 Metal ports pass and a wrong port fails | One Mac, Python 3.12, small scenes; kernelport originals not run on CUDA |
| Genesis throughput, M0 | Go2, 4096 environments, random actions: Metal 81,000 to 133,000 env-steps/s, CPU 13,000 to 14,000, 3 runs. At 512 environments Metal is only 1.0 to 1.9 times the CPU | Repeat runs of the standing scene differ by up to 3 times, cause not found; no policy inference |
| woof translator, M0 | 3 CUMLX kernels correct on Metal, worst 5.0e-8; a wrong kernel measured 1.08. Of 17 foreign kernels, none compiled as they are. All refusals named a reason | Foreign kernels were not compared by value; one was run after a manual edit |
| Isaac Lab on the Mac without Isaac Sim (`isaac-lab-mac.json`) | Cartpole with Newton on the CPU: 100 iterations in about 18 s, about 40,000 steps/s, 256 environments. Final episode length 300, 293, 246 of 300 over 3 seeds | Hand-built environment that NVIDIA does not support; one placeholder asset; `uv sync` refuses macOS; one task; not compared with PhysX |
| Real Isaac Lab checkpoint in MLX (`isaac-lab-mac.json`) | Against rsl_rl's own outputs: worst 7.9e-8. Wrong checkpoint 1.08, wrong activation 0.66 and one weight 1% off (2.2e-4) all fail | Cartpole 4-32-32-1; no normalizer |
| MLX policy controls Newton (`isaac-lab-mac.json`) | Over the last 100 of 900 steps: upright 100% in 3 of 3 seeds. No force: 0%. First-iteration checkpoint: 5% | MLX on the GPU, physics on the CPU; 256 environments |
| Isaac Sim streamed to a Mac | **Not run.** No RTX machine; the client is documented for macOS | |
| Other NVIDIA and Apple GPU generations, padded inputs, large MoE | **Not run** | |

Evidence: [real models](dogfood-macos-m5-pro.json), [public ports](public-ports.json), [NVFP4 repack](checkpoint-dory-nvfp4.json), [large capture](large-capture-qwen2.5-7b.json), [agent demo](mcp-agent-demo.md), [upstream finding](upstream/README.md), [platform spikes](spikes.json), [Isaac Lab on the Mac](isaac-lab-mac.json). `scripts/dogfood_real_model.py`, `scripts/public_port_campaign.py` and `scripts/measure_large_capture.py` reproduce them.
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
| SmolLM2 and tiny MoE on CPU | A bf16 MoE port on CPU ended as an internal error | `E_MODEL_RUN` with a hint | `test_op_missing_on_this_device...` |
| Dory1.0 NVFP4 | The safetensors reader refused FP8 scale tensors | Read as raw codes | `test_safetensors_reads_fp8...` |
| Qwen3-0.6B | A stored `lm_head.weight` stayed unverified against a tied port | Compared with the embedding | `test_static_weights_compare_a_stored_head...` |
| LFM2.5-230M | A forced causal mask broke hybrid convolutions; their decode input was labeled as tokens; conv weights were compared in the wrong layout | Replay the mask the port really passed; label by dtype; compare in torch layout | `test_replay_uses_the_mask...`, `test_a_float_input...`, `test_conv_weights...` |
| Gemma4 | A factor that the parent applies was reported as divergence | Accepted only when the parent matches | `test_constant_factor_moved...` |
| Mixtral | `mlp` stayed unverified, and quantized goldens silently kept the original experts | Name rule; a partial golden is now an error | `test_mixtral_moe...`, `test_partial_port_weights...`, `test_quantized_mixtral...` |
| Draft checkpoints | transformers filled missing weights with random values, so a golden could be random | `capture` refuses | `test_a_checkpoint_with_missing_weights...` |

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
