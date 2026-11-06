# AGENTS.md — using and developing CUMLX with an AI agent

CUMLX verifies that an MLX port computes the same thing as the original
PyTorch/CUDA model, module by module, and locates the first divergence.
It does not convert models; use mlx-lm or another converter for that.

## Using the `cumlx` CLI

Start with these two commands. Both are fast and safe:

```bash
cumlx describe --format json               # full spec: commands, arguments, exit/error codes
cumlx doctor --require capture,diff        # exit 3 + install hint if something is missing
```

### Contract

| Rule | Detail |
|---|---|
| Output | With `--format json` (default when stdout is not a TTY), stdout is exactly one JSON document, schema `cumlx.result.v1`. Logs go to stderr. |
| Envelope | `ok`, `status` (`ok`/`diverged`/`incomplete`/`error`), `exit_code`, `summary`, `data`, `artifacts`, `warnings`, `errors`, `next_steps`, `truncated`. |
| Exit codes | `0` ok · `1` diverged · `2` usage · `3` environment · `4` input · `5` incomplete (NOT a pass) · `10` internal bug. |
| Errors | `errors[].code` is stable (`E_DEP_MISSING`, `E_OUTPUT_EXISTS`, ...). `errors[].hint` is the exact fix. |
| Size | Lists are capped by `--max-items` (default 20, `-1` = all). Full data is in `artifacts` files. |
| Safety | Nothing is overwritten without `--overwrite`. Non-CUMLX directories are never deleted. `--dry-run` validates without heavy work. Remote model code runs only with `--trust-remote-code`. |

### Workflow

```bash
cumlx weights <original-hf-id-or-dir> <mlx-port>                       # seconds; names transposes and scale errors
cumlx capture <original-hf-id-or-dir> --prompt "Hello" -o golden/      # or --token-ids 1,2,3
cumlx diff golden/ <mlx-port> -o report/                               # isolated mode by default
cumlx report report/ --max-items -1                                    # re-read every failure
```

Use a golden with the port's own weights for a quantized port, and for any low-precision port
when `W_WEIGHT_ROUNDING_NOISE` appears. It removes quantization loss and rounding from the
comparison, so every module is judged at `rel_l2 <= 1e-4`. Check the weights themselves with `cumlx weights`.

```bash
cumlx capture <original> --prompt "Hello" --weights-from-mlx <mlx-port> -o golden-p/
```

On `exit_code == 1`, read `data.first_divergence` first:

- `basis: isolated`, `confidence: high` → the module itself is wrong. Attention and decoder layers are replayed too, so this covers RoPE, masking and residuals.
- `basis: chain+isolated`, `confidence: medium` → the module could not be replayed; its submodules are right and the defect is in its own code between them.
- `basis: chain` → in `--mode chain`, rerun with `--mode isolated`. In isolated mode, the output is non-finite or wrongly shaped in the native dtype only.
- `phase: decode` → prefill is right and the module's KV cache handling is wrong (position offset, mask, cache update).
- A module that ends in `.mlp.experts` → the experts are wrong on the golden routing. `.mlp.gate` → the router is wrong. `cumlx weights` names the expert.

Read these before you trust a result:

- `data.execution`: the device and the measured float32 matmul error. The default is the MLX default device with TF32 off. MLX runs float32 matmul in TF32 by default on M5 and CUDA (about 1e-3 error); `--tf32 on` measures that behavior and raises the thresholds.
- `data.summary.chain_drift`: outputs that exceeded the chain threshold while the module passed on its golden input. This is accumulated native-dtype rounding, not a defect; torch's own bf16 shows the same size.
- `data.summary.native_output`: top-1 agreement of the native-dtype logits with the golden.
- `W_WEIGHT_ROUNDING_NOISE`: thresholds rose with the rounding noise of this port; smaller defects can pass.
- `W_PARTIAL_TRACE`: the golden holds only some modules (`--layers`); the verdict covers only those.

- `W_MODULE_BOUNDARY_DIFFERS`: an output is a constant multiple of the golden and its parent matches. The factor sits in another module in that framework. This is not a defect.

To start from a CUDA project: `cumlx scan <path or HF id>` ranks the work (model, device, library, kernel, platform) without running the code. `cumlx recipe <name, path or scan report>` names what to use on Apple Silicon. Read `evidence` on each option: `run` was executed on Apple Silicon within the stated `limits`, and `docs` was only read about. Do not report a `docs` option as working. `outcome: none` means there is no substitute. `cumlx port <model> -o out/` converts with mlx-lm and proves the result in one command. It refuses a model that mlx-lm does not implement. See `docs/PORTING-CUDA-MODELS.md`.

`cumlx run train.py -- --lr 1e-3` runs an unmodified PyTorch script that assumes CUDA on MPS, and writes each redirected call to `run-report.json`. It executes the file. `torch.cuda.is_available()` is True there, and the device name says MPS. A call with no equivalent ends with `E_RUN_UNSUPPORTED`, and an operation that MPS lacks ends with `E_RUN_MPS_OP_MISSING` unless you pass `--allow-cpu-fallback`. `W_RUN_NOT_VERIFIED` is always present: a script that runs is not a script that is right. To grade such a run, record it with `--trace-losses --seed N` on `--target cuda` and on the target, then run `cumlx rundiff golden/ candidate/`. Grade A covers the first two steps only. See `docs/CLI.md`.

To move a CUDA kernel: `cumlx kernel capture kernels/ --cases cases.json -o kgolden/ --require-idle-gpu` compiles and runs the original with nvcc on an NVIDIA machine and records a `cumlx.opgolden.v1` golden. It executes the code, so MCP hides it unless the server has `--allow-code-execution`. `cumlx kernel port foo.cu -o foo.metal` handles headers, templates, host code and PTX built-ins, and calls a translator through `TranslatorAdapter`. CUMLX does not install one: a missing translator is `E_KERNEL_TRANSLATOR_MISSING` (exit 3), and `--preprocess-only` needs none. `E_KERNEL_UNPORTABLE` lists each feature with no Metal equivalent and its reason in `errors[].details.refused`. `cumlx kernel verify foo.metal kgolden/` runs the kernel on the golden inputs. Exit 0 with `grade: A` needs a golden with `kind: cuda`. An unknown kernel, a gap in a folder, a failed compile and float64 are all exit 5, not a pass. Read `data.execution.differences` for fp64, FTZ, FMA and execution order. A fault that the inputs do not reach cannot be caught, so a pass covers only what the cases ran. See `docs/CLI.md`.

`cumlx.sim` holds task definitions on Genesis and MuJoCo, in the style of Isaac Lab and not Isaac Lab itself. `cumlx.render` holds camera and raycast sensors on Mitsuba Metal. `cumlx.render.usd` reads a small part of USD (meshes, one perspective camera, RectLight, DistantLight, a diffuse material) into a Mitsuba scene, and refuses anything else with `E_RENDER_USD_UNSUPPORTED`. `cumlx doctor --stack --env PYTHON` reports a stack spread over environments, and it does not show that the parts load together.

Other commands: `cumlx conform --checkpoint <modelopt> --against <mlx-repack>` compares an NVFP4 repack with its original, byte by byte. `cumlx tracediff A B` measures how far a golden moves between devices. `cumlx.check_op(torch_fn, mlx_fn, inputs)` compares one op. `cumlx-mcp` serves every command as an MCP tool. `skills/cumlx-verifier/SKILL.md` is a skill for porting agents. `docs/TRACE-FORMAT.md` is the trace specification.

On `exit_code == 5`, some parametrized modules were not compared. Map them with
`--mapping rules.json` (`{"rules": [{"match": REGEX, "replace": STR}], "ignore": [REGEX]}`)
or ignore them explicitly. Never report `incomplete` as success.

### The Rust binary `cumlx-rs`

`rust/` holds the core that needs neither torch nor MLX: the library crate `cumlx` (envelope, error codes, trace
format, safetensors, metrics and threshold profiles) and the binary `cumlx-rs`. Build it with
`cargo build --release --manifest-path rust/Cargo.toml`.

- `cumlx-rs tracediff A B` and `cumlx-rs weights ORIGINAL PORT` give the same document, exit code, error codes,
  verdicts and numbers (within 1e-12) as the Python commands. `cumlx-rs describe` lists every command with
  `implemented: true` or `false`.
- Every other command ends with exit 3 and `E_PYTHON_ONLY`; `errors[].hint` is the Python command. Run that.
- `cumlx-rs weights` reads the port's `model*.safetensors` as stored. It refuses a quantized port, `--mapping`,
  `--loader python:...` and a Hugging Face id (`E_PYTHON_ONLY`), and marks a channels-last convolution weight
  `unverified`. It does not prove that mlx-lm loads the port.
- A change to `metrics.py`, `weights.py`, `trace.py`, `trace_compare.py` or `ERROR_CODES` needs the same change in
  `rust/`. Then run `python scripts/rust_vectors.py` and `python -m pytest tests/test_rust_parity.py --require-rust`.
  See `rust/README.md`.

`cumlx selftest` proves within seconds that the verifier works in the
current environment: a clean port passes and 11 injected faults are located.

`cumlx doctor --stack` reports each platform layer in this Python environment: MLX, mlx-lm, torch with MPS, the Metal device, Genesis, Mitsuba Metal, warp-metal and the translator. Any layer that is not `ok` gives exit 3 with `E_DEP_MISSING` and an install hint. It installs nothing, uses no network and does not look for nvcc. A part that a spike installed in its own environment reads as missing here. `ok` says that a part is present, not that its results are right.

## Developing this repository

```bash
python -m pip install -e '.[dev]'                       # core (NumPy only)
python -m pip install -e '.[dev,capture,diff]'          # + torch, transformers, mlx, mlx-lm
python -m pytest -q -m 'not gpu'                        # e2e tests skip without the extras
python -m pytest -q -m e2e --require-e2e                # e2e must run (CI verifier job)
python -m ruff check . && python scripts/generate_kernels.py && git diff --exit-code -- src/cumlx/kernels
python scripts/dogfood_real_model.py -o evidence.json   # real checkpoints; needs network and disk (L2)
cd rust && cargo fmt --all --check && cargo clippy --locked --all-targets -- -D warnings && cargo test --locked && cd ..
cargo build --release --manifest-path rust/Cargo.toml && python -m pytest -q tests/test_rust_parity.py --require-rust --require-e2e
```

`MLXBackend` accepts MLX 0.32.2 and later 0.32.x. The GPU tests ran on 0.32.2 and 0.32.4; another 0.32.x gives a warning.

- `src/cumlx/cli/`: argument parsing, envelope, command handlers. Keep the stdout contract.
- `src/cumlx/verify/`: trace format, capture (torch), MLX probe, diff, static weights, conform, NVFP4 checkpoints, check_op, reports, faults, selftest.
- `src/cumlx/mcp_server.py`: MCP stdio server. Its tool schemas come from the CLI parser.
- `src/cumlx/kernels/`, `mlx_backend.py`: FP4 kernel lab, run on one M5 Pro and one V100 only. Regenerate kernels only with `scripts/generate_kernels.py`.
- A new error code goes in `ERROR_CODES` (`src/cumlx/errors.py`); `describe` and tests pick it up.
- A new fault goes in `verify/faults.py` with an expected target, and it must be located in `tests/test_verify_e2e.py`. A wrong kernel translation goes in `KERNEL_FAULTS` there, with a right case on the same golden.
- `src/cumlx/verify/opgolden.py`, `kernel_*.py`: op-level goldens, capture, verify, and the pre-translation layer. `validation/kernel-golden-v100*` are real V100 runs: re-record them with `scripts/kernel_capture_remote.sh`, and never write one by hand. `tests/test_kernel_gpu.py` fails when `kernel_port.py` changes a prepared source that a golden covers.
- Never turn a skip, an unmapped module or a missing dependency into a pass.
- Do not raise a threshold to clear a failure. Find the cause first, and record any new threshold's measurement in `verify/metrics.py`.
