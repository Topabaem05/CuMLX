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
cumlx capture <original-hf-id-or-dir> --prompt "Hello" -o golden/      # or --token-ids 1,2,3
cumlx diff golden/ <mlx-port> -o report/                               # isolated mode by default
cumlx report report/ --max-items -1                                    # re-read every failure
```

For a quantized port, capture a quantization-aware golden so that
quantization loss is not reported as a bug:

```bash
cumlx capture <original> --prompt "Hello" --weights-from-mlx <mlx-port> -o golden-q/
```

On `exit_code == 1`, read `data.first_divergence` first:

- `basis: isolated`, `confidence: high` → the module itself is wrong.
- `basis: chain+isolated`, `confidence: medium` → the defect is in the module's own code between its recorded submodules (RoPE, attention math, masking, residuals).
- `basis: chain` → rerun with `--mode isolated` to narrow it down.

On `exit_code == 5`, some parametrized modules were not compared. Map them with
`--mapping rules.json` (`{"rules": [{"match": REGEX, "replace": STR}], "ignore": [REGEX]}`)
or ignore them explicitly. Never report `incomplete` as success.

`cumlx selftest` proves within seconds that the verifier works in the
current environment: a clean port passes and 8 injected faults are located.

## Developing this repository

```bash
python -m pip install -e '.[dev]'                       # core (NumPy only)
python -m pip install -e '.[dev,capture,diff]'          # + torch, transformers, mlx, mlx-lm
python -m pytest -q -m 'not gpu'                        # e2e tests skip without the extras
python -m pytest -q -m e2e --require-e2e                # e2e must run (CI verifier job)
python -m ruff check . && python scripts/generate_kernels.py && git diff --exit-code -- src/cumlx/kernels
```

- `src/cumlx/cli/`: argument parsing, envelope, command handlers. Keep the stdout contract.
- `src/cumlx/verify/`: trace format, capture (torch), MLX probe, diff, conform, faults, selftest.
- `src/cumlx/kernels/`, `mlx_backend.py`: FP4 kernel lab (GPU-unvalidated). Regenerate kernels only with `scripts/generate_kernels.py`.
- A new error code goes in `ERROR_CODES` (`src/cumlx/errors.py`); `describe` and tests pick it up.
- A new fault goes in `verify/faults.py` with an expected target, and it must be located in `tests/test_verify_e2e.py`.
- Never turn a skip, an unmapped module or a missing dependency into a pass.
