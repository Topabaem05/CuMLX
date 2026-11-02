import os
import pytest

# MLX 0.32 reads this once per process, at the first matmul. A test that wants TF32 uses a subprocess.
os.environ.setdefault('MLX_ENABLE_TF32','0')


def pytest_addoption(parser):
    parser.addoption('--require-gpu',action='store_true',default=False,help='Fail, do not skip, if the selected physical GPU is unavailable')
    parser.addoption('--backend',default=os.environ.get('CUMLX_TEST_BACKEND','metal'),choices=['metal','cuda'])
    parser.addoption('--require-e2e',action='store_true',default=False,help='Fail, do not skip, if torch/transformers/mlx/mlx-lm are missing')


@pytest.fixture(scope='session')
def gpu(request):
    backend=request.config.getoption('--backend')
    try:
        import mlx.core as mx
        device_api=getattr(mx,backend,None)
        available=device_api is not None and bool(device_api.is_available())
    except (ImportError,OSError,RuntimeError) as exc:
        available=False; reason=f'{type(exc).__name__}: {exc}'
    else:
        reason=f'MLX physical {backend} device not available'
    if not available:
        if request.config.getoption('--require-gpu'): pytest.fail(reason)
        pytest.skip(reason)
    # Once a GPU is present, compilation or numerical failures MUST NOT become skips.
    from cumlx.mlx_backend import MLXBackend
    return MLXBackend(backend,allow_unvalidated=True)


@pytest.fixture(scope='session')
def e2e(request):
    """Verifier end-to-end dependencies. Missing deps skip, or fail with --require-e2e."""
    import importlib.util
    missing=[m for m in ('torch','transformers','mlx','mlx_lm') if importlib.util.find_spec(m) is None]
    if missing:
        msg=f'verifier e2e needs {missing}; pip install -e ".[capture,diff]"'
        if request.config.getoption('--require-e2e'): pytest.fail(msg)
        pytest.skip(msg)
    return True
