"""Verify an installed wheel from outside the source checkout (use python -I)."""
from importlib.resources import files
from pathlib import Path
import json
import sys
import numpy as np
import cumlx
from cumlx import quantize, reference
from cumlx.mlx_backend import KERNELS, kernel_source


def main():
    checkout = Path(__file__).resolve().parents[1]
    installed = Path(cumlx.__file__).resolve()
    if checkout in installed.parents:
        raise AssertionError(f'Imported checkout instead of installed wheel: {installed}')
    assert 'mlx.core' not in sys.modules, 'CPU import must not require MLX'
    for backend in ('metal', 'cuda'):
        for name in KERNELS:
            header, source, digest = kernel_source(name, backend)
            assert header and source and len(digest) == 64
    assert files('cumlx').joinpath('py.typed').is_file()
    rng = np.random.default_rng(28)
    x = rng.normal(size=(3, 67)).astype(np.float32)
    w = rng.normal(size=(35, 67)).astype(np.float32)
    for fmt in ('nvfp4', 'mxfp4'):
        q = quantize(w, fmt)
        y = reference.linear(x, q)
        assert y.shape == (3, 35) and np.isfinite(y).all()
    print(json.dumps({'installed_from': str(installed), 'version': cumlx.__version__,
                      'kernel_sources': len(KERNELS) * 2, 'gpu_executed': False,
                      'status': 'WHEEL_CPU_SMOKE_PASSED'}, indent=2))


if __name__ == '__main__':
    main()
