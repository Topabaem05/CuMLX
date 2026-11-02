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

Read these before you trust a result:

- `data.execution`: the device and the measured float32 matmul error. The default is the MLX default device with TF32 off. MLX runs float32 matmul in TF32 by default on M5 and CUDA (about 1e-3 error); `--tf32 on` measures that behavior and raises the thresholds.
- `data.summary.chain_drift`: outputs that exceeded the chain threshold while the module passed on its golden input. This is accumulated native-dtype rounding, not a defect; torch's own bf16 shows the same size.
- `data.summary.native_output`: top-1 agreement of the native-dtype logits with the golden.
- `W_WEIGHT_ROUNDING_NOISE`: thresholds rose with the rounding noise of this port; smaller defects can pass.

On `exit_code == 5`, some parametrized modules were not compared. Map them with
`--mapping rules.json` (`{"rules": [{"match": REGEX, "replace": STR}], "ignore": [REGEX]}`)
or ignore them explicitly. Never report `incomplete` as success.

`cumlx selftest` proves within seconds that the verifier works in the
current environment: a clean port passes and 9 injected faults are located.

## Developing this repository

```bash
python -m pip install -e '.[dev]'                       # core (NumPy only)
python -m pip install -e '.[dev,capture,diff]'          # + torch, transformers, mlx, mlx-lm
python -m pytest -q -m 'not gpu'                        # e2e tests skip without the extras
python -m pytest -q -m e2e --require-e2e                # e2e must run (CI verifier job)
python -m ruff check . && python scripts/generate_kernels.py && git diff --exit-code -- src/cumlx/kernels
python scripts/dogfood_real_model.py -o evidence.json   # real checkpoints; needs network and disk (L2)
```

On Apple Silicon, `.[dev,metal]` pins MLX 0.32.2 for the kernel lab. Use a separate venv from `.[capture,diff]`,
which installs the newest MLX; `MLXBackend` refuses any other version.

- `src/cumlx/cli/`: argument parsing, envelope, command handlers. Keep the stdout contract.
- `src/cumlx/verify/`: trace format, capture (torch), MLX probe, diff, static weights, conform, faults, selftest.
- `src/cumlx/kernels/`, `mlx_backend.py`: FP4 kernel lab (GPU-unvalidated). Regenerate kernels only with `scripts/generate_kernels.py`.
- A new error code goes in `ERROR_CODES` (`src/cumlx/errors.py`); `describe` and tests pick it up.
- A new fault goes in `verify/faults.py` with an expected target, and it must be located in `tests/test_verify_e2e.py`.
- Never turn a skip, an unmapped module or a missing dependency into a pass.
- Do not raise a threshold to clear a failure. Find the cause first, and record any new threshold's measurement in `verify/metrics.py`.
