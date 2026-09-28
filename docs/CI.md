# Continuous integration and kernel-validation boundaries

The `CI` workflow runs on main pushes, main pull requests, or manual dispatch.
It uses read-only repository permissions, SHA-pinned actions, no persisted checkout
credentials, and 10-minute job timeouts. No deployment or PyPI publication occurs.

## Automatic checks

- Python lint correctness: Ruff 0.11.13, E9/F63/F7/F82. This is an initial
  correctness gate, not a claim of full style, formatting, type, or security coverage.
- Python compileall for src/tests/scripts/benchmarks.
- Regenerate all 16 kernel bodies and two headers; any tracked or untracked drift fails.
- CPU reference tests on Python 3.10, 3.11, 3.12, and 3.13 on Ubuntu 24.04.
  A host C++ compiler is required. GPU tests are deselected, not passed.
- Build an sdist and a wheel, check their metadata using twine, install the wheel
  into a clean venv, and run `scripts/wheel_smoke.py` from outside the checkout
  with Python isolated mode. Validate all 16 packaged sources and both FP4 formats.
- Store JUnit/environment reports and Python distributions as 14-day Actions artifacts.

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
python -m pip install -e '.[dev]' ruff==0.11.13 build twine
python -m ruff check .
python -m compileall -q src tests scripts benchmarks
python scripts/generate_kernels.py
git diff --exit-code -- src/cumlx/kernels
python -m pytest -q -m 'not gpu'
python -m build
python -m twine check dist/*
```

A green CPU CI is not a GPU validation. Check exact commit SHA, workflow jobs,
backend, and test counts before changing a validation status.
