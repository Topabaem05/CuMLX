import os
import pytest


def pytest_addoption(parser):
    parser.addoption('--require-gpu',action='store_true',default=False,help='Fail, do not skip, if the selected physical GPU is unavailable')
    parser.addoption('--backend',default=os.environ.get('CUMLX_TEST_BACKEND','metal'),choices=['metal','cuda'])


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
