import os
import pytest

# MLX 0.32 reads this once per process, at the first matmul. A test that wants TF32 uses a subprocess.
os.environ.setdefault('MLX_ENABLE_TF32','0')


def pytest_addoption(parser):
    parser.addoption('--require-gpu',action='store_true',default=False,help='Fail, do not skip, if the selected physical GPU is unavailable')
    parser.addoption('--backend',default=os.environ.get('CUMLX_TEST_BACKEND','metal'),choices=['metal','cuda'])
    parser.addoption('--require-e2e',action='store_true',default=False,help='Fail, do not skip, if torch/transformers/mlx/mlx-lm are missing')
    parser.addoption('--require-tf4',action='store_true',default=False,help='Fail, do not skip, if CUMLX_TF4_SITE does not hold transformers 4')


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


@pytest.fixture(scope='session')
def tf4_site(request, e2e):
    """A directory that holds transformers 4 (one module per MoE expert), put first on PYTHONPATH for a subprocess.

    Create it with: python -m pip install --target DIR 'transformers>=4.57,<5' 'huggingface-hub<1.0'
    """
    import json, subprocess, sys
    site = os.environ.get('CUMLX_TF4_SITE')
    version = None
    if site and os.path.isdir(os.path.join(site, 'transformers')):
        probe = subprocess.run([sys.executable, '-c', 'import transformers; print(transformers.__version__)'],
                               capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': site})
        version = probe.stdout.strip() or None
    if not version or not version.startswith('4.'):
        msg = f'CUMLX_TF4_SITE must hold transformers 4 (found {version!r} at {site!r})'
        if request.config.getoption('--require-tf4'): pytest.fail(msg)
        pytest.skip(msg)
    return site
