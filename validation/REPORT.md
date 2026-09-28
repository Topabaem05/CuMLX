# CUMLX v0.3 repository import validation

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

## Reproduce

```bash
python -m pip install -e '.[dev]'
python -m pytest -q -m 'not gpu'
# Real supported Apple GPU only:
python -m pip install -e '.[dev,metal]'
python -m pytest -q -m gpu --backend metal --require-gpu
```

The separate manual GPU workflow requires a trusted physical self-hosted runner;
missing hardware must fail instead of silently falling back to the CPU.
