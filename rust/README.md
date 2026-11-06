# cumlx-rs: the Rust core of CUMLX

This workspace holds the parts of the CUMLX verifier that need neither PyTorch nor MLX, in Rust:

| Crate | Kind | What it holds |
|---|---|---|
| `cumlx` | library | the `cumlx.result.v1` envelope, the error and exit codes, a safetensors reader, the trace format `cumlx.trace.v1`, the metrics and threshold profiles, the trace comparison and the static weight comparison |
| `cumlx-rs` | binary | the command line: `describe`, `tracediff` and `weights` |

Every verdict, code and number is meant to equal the Python `cumlx`. `tests/test_rust_parity.py` runs both
command lines on the same inputs and fails on any difference larger than 1e-12 relative.

## What runs here, and what does not

| Command | In `cumlx-rs` |
|---|---|
| `describe` | Runs. Lists every command of the Python CLI, with `implemented: true` or `false` |
| `tracediff A B` | Runs. Same verdicts, numbers, warnings and errors as Python |
| `weights ORIGINAL PORT` | Runs for an unquantized port in a local folder. See the limits below |
| `capture`, `diff`, `run`, `port`, `conform`, `fixture`, `selftest`, `doctor` | Refused: they need torch, MLX or a Python environment |
| `kernel capture`, `kernel verify` | Refused: they need nvcc and an NVIDIA GPU, or Metal through MLX or torch |
| `scan`, `recipe`, `rundiff`, `report`, `kernel port` | Refused: not ported |

A refused command ends with exit code 3 and the error `E_PYTHON_ONLY`. Its `hint` is the install command and the
same command for the Python `cumlx`. The binary runs no part of such a command and never reports it as passed.

## Build and use

```bash
cargo build --release --manifest-path rust/Cargo.toml
rust/target/release/cumlx-rs describe
rust/target/release/cumlx-rs tracediff golden-cuda/ golden-cpu/
rust/target/release/cumlx-rs weights ./original ./port-mlx -o weights.json
```

The contract is the Python contract:

- With `--format json`, which is the default when stdout is not a terminal, stdout holds exactly one JSON document.
  Logs (`-v`) go to stderr.
- Exit codes: `0` ok, `1` diverged, `2` usage, `3` environment, `4` input, `5` incomplete (not a pass), `10` internal.
- `errors[].code` comes from the same table as `ERROR_CODES` in `src/cumlx/errors.py`.
- `--max-items` caps lists (default 20, `-1` = all). `CUMLX_FORMAT` and `CUMLX_MAX_ITEMS` set the defaults.
- `-o` never replaces a file without `--overwrite`.

From Rust:

```rust
let report = cumlx::trace_compare::compare_traces("golden-a", "golden-b", 1e-4)?.report;
let t = cumlx::metrics::resolve_threshold("bfloat16", None, false)?;
```

## Limits of `weights`

The Python command loads the port with mlx-lm. This one reads the port's `model*.safetensors` files, the same files
that mlx-lm loads, as they are stored. So:

- **A quantized port is refused** (`E_PYTHON_ONLY`): dequantizing it needs MLX. The binary looks for
  `quantization` or `quantization_config` in `config.json`, `hf_quant_config.json` and `*.scales` tensors.
- **mlx-lm's `sanitize` step does not run.** Give the folder that mlx-lm wrote. For a Hugging Face folder used as
  the port, the names can differ from what mlx-lm would load (for example, MoE experts that mlx-lm stacks).
- **The module type is unknown.** A 3-D or 4-D weight whose shape is the channels-last layout of the original
  convolution is `unverified` (exit 5). Python compares it in that layout.
- **It does not prove that mlx-lm loads the port.** A wrongly shaped weight is reported as `shape_mismatch`
  (with `transposed` when it is), where Python stops at the load with `E_MODEL_LOAD`.
- `--mapping`, `--loader python:...` and a Hugging Face id are refused with `E_PYTHON_ONLY`. `--revision` is accepted
  and, as in Python, has no effect on a local checkpoint.

## Other known differences

- Sums use eight partial sums, and NumPy uses BLAS. The parity tests allow 1e-12 relative.
- On some malformed inputs the Python CLI raises an unexpected exception (`E_INTERNAL`, exit 10): a manifest that
  is not UTF-8, a `golden` that is not an object, a `samples[].file` that is not a string. `cumlx-rs` reports
  `E_INPUT_INVALID` (exit 4).
- The message of a malformed manifest quotes the Python exception in Python and says the same thing in other words
  here. The code is the same. Usage error messages differ in the same way; their code and hint are the same.
- A sample, record or tensor reference with an unknown field is rejected, as the Python reader does it, although
  `docs/TRACE-FORMAT.md` says that a reader must ignore unknown fields.
- `SHA-256` of the sample files is computed by a short implementation in `cumlx/src/sha256.rs`, checked against the
  FIPS 180-4 examples and, through the parity tests, against Python's `hashlib`.

## The proof

| Check | Where | What it requires |
|---|---|---|
| Unit tests | `cargo test` | formatting as Python prints it, safetensors round trip, SHA-256 examples, argument parsing |
| Python test vectors | `cumlx/tests/python_vectors.rs` reads `cumlx/tests/data/python-vectors.json` | the Rust functions equal the Python ones: `compare` on 22 cases, `compare_kernel_output` on 19, the weight comparison on 21, `resolve_threshold` on 16 and 4 refusals, `noise_thresholds` on 160; equal constants, error codes and exit codes |
| Vectors are current | `python scripts/rust_vectors.py --check` | the committed vectors are what today's Python gives |
| Cross-implementation | `tests/test_rust_parity.py` | both command lines on the same folders: same envelope, verdicts, codes, hints and first failure, numbers within 1e-12 |

The cross-implementation test builds its inputs with the Python implementation: synthetic traces with
`TraceWriter`, the selftest's tiny Llama and its golden traces captured with PyTorch, and small checkpoints with an
injected transpose, scale and sign fault, written by MLX. Each fault fails in both. CI runs it in
`.github/workflows/rust.yml`, where a skip is a failure.

To see that the tests catch a wrong implementation, three faults were put into the Rust code by hand on 2026-11-06:
a missing record that no longer made the verdict `incomplete`, a lost `transposed` diagnosis, and a leaf noise
factor of 6.5 in place of 6.0. Each made the tests fail, and the code was restored.

## Dependencies

One direct dependency, `serde_json` 1.0.154 (pinned with `=`), with `preserve_order` (keys keep the Python order) and
`float_roundtrip`. `Cargo.lock` pins the rest:

| Crate | Version | Licence |
|---|---|---|
| serde_json | 1.0.154 | MIT OR Apache-2.0 |
| serde_core | 1.0.229 | MIT OR Apache-2.0 |
| indexmap | 2.14.2 | Apache-2.0 OR MIT |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 |
| equivalent | 1.0.3 | Apache-2.0 OR MIT |
| foldhash | 0.2.0 | Zlib |
| itoa | 1.0.18 | MIT OR Apache-2.0 |
| memchr | 2.8.3 | Unlicense OR MIT |
| zmij | 1.0.23 | MIT |

There is no `clap` and no `safetensors` crate. The argument parser in `cumlx-rs/src/args.rs` follows argparse
(options anywhere, `--opt=value`, unique prefixes, `-1` as a value), and the command table in
`cumlx-rs/src/spec.rs` feeds both the parser and `describe`. The safetensors reader follows the Python reader
check by check, including the FP4 scale types `F8_E4M3` and `F8_E8M0`.

The minimum Rust version is 1.85 (`rust-version`); CI builds with 1.85.1 and 1.99.1.

## Development

```bash
cd rust
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --release --locked
cd ..
python scripts/rust_vectors.py            # after a change to metrics.py or weights.py
python -m pytest -q tests/test_rust_parity.py --require-rust --require-e2e
```

A change to a Python constant, metric or verdict needs the same change here: the vectors test and the parity test
fail until both agree. Do not raise a threshold to clear a failure. Find the cause first.
