# Changelog

## Unreleased

### Added

- `cumlx run` (experimental): run an unmodified PyTorch script that assumes CUDA on MPS, and record each redirected call in `run-report.json`. A CUDA call with no equivalent, a second GPU, an operation that MPS lacks and float64 all end with an error. `PYTORCH_ENABLE_MPS_FALLBACK` is removed unless you pass `--allow-cpu-fallback`. Importing `cumlx` changes nothing in torch. MCP shows this command only with `--allow-code-execution`.
- `spikes/m2_scripts.sh`: run 5 open-source scripts under `cumlx run`, once per seed for those that train. `--target cuda` records the golden.
- `spikes/m2_substitutes.sh`: evaluate `mps-*`, metalsplat, msplat and mlx3D, each in its own venv at a pinned version, with checks against known numbers.
- `cumlx rundiff` (experimental): compare the loss traces of `cumlx run` from CUDA and from the target, and give grade A (first two steps) or B (training curve over 3 or more seeds). `cumlx run` gains `--trace-losses`, `--seed` and `--target cuda`.
- `validation/isaac-lab-go2.json`: Unitree Go2 flat velocity on a Mac CPU with Newton. With 4096 envs and 300 iterations, the policy in MLX measures +0.74 m/s at a +0.8 command and -0.87 m/s at -0.8, in simulation, with one seed.
- `spikes/isaac_lab_mac.sh`: train Isaac Lab cartpole on a Mac CPU with Newton, without Isaac Sim. NVIDIA does not support this, and Isaac Lab's `uv sync` refuses macOS, so it builds the environment by hand.
- `spikes/rsl_rl_golden.py` and `verify_policy.py --golden`: compare an MLX policy with rsl_rl's own outputs. Before this, the reference was a torch version written in the same file.
- `spikes/isaac_lab_mlx_loop.py`: let an MLX policy control an Isaac Lab environment, with controls.

### Fixed

- `examples/robot-policy` refused real rsl_rl 5 checkpoints: it expected `model_state_dict` and `actor.N`, but rsl_rl 5 writes `actor_state_dict` and `mlp.N`. Only random weights had been tried. It now also refuses an observation normalizer, which it would otherwise have left out with no error.
- The README said that `pip install isaacsim` installs an empty package on a Mac. It fails.

## 0.5.0 (2026-11-02)

### Added

- `cumlx recipe`: name what to use on Apple Silicon for a CUDA dependency. The catalog has 33 recipes. Each option states whether it was run on Apple Silicon (`evidence: run`, with its file under `validation/` and a measured grade) or only read about (`evidence: docs`). `--verified-only` keeps what was run. A test refuses a catalog that claims a run without evidence.
- `cumlx scan` reads declared dependencies from `requirements*.txt`, `pyproject.toml` and `environment*.yml`, and links each finding to its recipes. New rules: `gsplat`, `pytorch3d`, `mmcv-ops`, `nvdiffrast`, `triton`, `isaac-lab`.
- `cumlx port` names a converter for a vision model or a diffusers pipeline when mlx-lm does not implement it.
- `spikes/isaac_sim_container.sh`: run Isaac Sim 6.2 in a Linux arm64 container on a Mac. Kit and CPU PhysX run; RTX and cameras do not. NVIDIA does not support this. The script never accepts the EULA for you.
- `docs/ROADMAP-PLATFORM.md`: the next goal, running RTX and CUDA workloads on Apple Silicon, with the measured M0 results in `validation/spikes.json`.

## 0.4.1 (2026-11-02)

### Added

- `cumlx scan`: rank the work to run a CUDA project or model on Apple Silicon, from a text scan. It never runs the scanned code.
- `cumlx port`: convert with mlx-lm and prove the result in one command. It refuses a model that mlx-lm does not implement.
- `docs/PORTING-CUDA-MODELS.md`: what can be converted, and how to use Isaac Sim from a Mac.
- `examples/robot-policy`: move a robot policy from Isaac Lab to MLX and prove that it computes the same.
- Mixtral (`block_sparse_moe`) and channels-last convolution support in `weights`, `diff` and `--weights-from-mlx`.

### Fixed

Each came from running recent models, and each has a test.

- Replay now uses the mask that the port itself passed. A causal mask broke the LFM2 convolution.
- A decode-phase module with a float input is a hidden state, not tokens.
- A constant factor that the parent module applies, as in Gemma embeddings, is no longer a false alarm. A real scale error still fails, because the parent then differs.
- `capture` refuses a checkpoint with missing weights. transformers only logs them and fills them with random values, so a golden from it was meaningless.
- `--weights-from-mlx` refuses to write a golden that leaves out part of the port's weights.

### Measured

- CUDA golden on a Tesla V100: repeat runs are identical, CUDA against CPU is at most 1.9e-5, and an MLX port on a Mac passes against it.
- 12 recent Hugging Face models: 7 ok, 4 not comparable (partial checkpoints), 1 not loadable by mlx-lm.

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

- Not run: Apple GPU generations other than M5, padded inputs, models that are not LLMs.
- The thresholds come from two model families on one machine.
- Not published to PyPI.
