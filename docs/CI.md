# Continuous integration and kernel-validation boundaries

The `CI` workflow runs on main pushes, main pull requests, or manual dispatch.
It uses read-only repository permissions, SHA-pinned actions, no persisted checkout
credentials, and 10-minute job timeouts. No deployment or PyPI publication occurs.

## Automatic checks

- Python lint: Ruff 0.11.13. The verifier, the CLI and new code use F, B, I, UP, C4 and SIM.
  The kernel lab keeps the E9/F63/F7/F82 correctness gate. See `pyproject.toml`.
- Type check: mypy on the modules listed in the workflow. The other modules are not held to it yet.
- Python compileall for src/tests/scripts/benchmarks.
- Regenerate all 16 kernel bodies and two headers; any tracked or untracked drift fails.
- CPU reference tests on Python 3.10, 3.11, 3.12, and 3.13 on Ubuntu 24.04.
  A host C++ compiler is required. GPU tests are deselected, not passed.
- Build an sdist and a wheel, check their metadata using twine, install the wheel
  into a clean venv, and run `scripts/wheel_smoke.py` from outside the checkout
  with Python isolated mode. Validate all 16 packaged sources and both FP4 formats.
- Store JUnit/environment reports and Python distributions as 14-day Actions artifacts.
- Verifier end-to-end (`verifier` job): install CPU-only PyTorch and MLX CPU, require
  `cumlx doctor --require capture,diff,conform`, run `pytest -m e2e --require-e2e`
  (skips fail), `cumlx selftest`, and `cumlx conform --target mlx`. Conform findings
  about MLX (exit 1) are stored as evidence and do not fail CI; other non-zero exits do.
  This proves the verifier on tiny random models only, not on real checkpoints or Metal.
  Real checkpoints and Metal run through `scripts/dogfood_real_model.py` on a developer Mac (roadmap L2);
  no workflow runs them yet. See [the validation report](../validation/REPORT.md).

## Physical GPU tests (separate, manual)

Register a trusted self-hosted runner with its physical-backend label `metal` or
`cuda`. Only dispatch the GPU workflow from trusted main. It never runs arbitrary
pull-request code on a self-hosted machine. MLX is pinned to the source's 0.32.2
API target. A matching supported OS/driver and a real GPU are prerequisites.

`--require-gpu` converts missing MLX/device availability into failure. Compilation
and numerical failures after device detection are not skipped. These tests do
not claim CUDA library compatibility, PyTorch CUDA-facing support, or native FP4 MMA.
No GPU runner is provisioned or started by this repository setup.

## Local reproduction

```bash
python -m pip install -e '.[dev]' ruff==0.11.13 mypy==2.4.1 build twine
python -m ruff check .
python -m compileall -q src tests scripts benchmarks
python scripts/generate_kernels.py
git diff --exit-code -- src/cumlx/kernels
python -m pytest -q -m 'not gpu'
python -m pip install -e '.[dev,capture,diff]'   # verifier e2e (Linux: mlx[cpu])
python -m pytest -q -m e2e --require-e2e
cumlx selftest
python -m build
python -m twine check dist/*
```

On Apple Silicon, MLX defaults to TF32 for float32 matmul on M5 and later, so a test that expects
float32 precision must set `MLX_ENABLE_TF32=0` before its first matmul; `tests/conftest.py` does this.

A green CPU CI is not a GPU validation. Check exact commit SHA, workflow jobs,
backend, and test counts before changing a validation status.
