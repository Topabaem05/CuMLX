from pathlib import Path
import importlib

ROOT=Path(__file__).resolve().parents[1]
NAMES=['fp4_decode','fp4_gemv','fp4_small_m','fp4_tiled','fp4_w4a4','fp4_quantize','rms_norm','silu_mul']


def test_kernel_sources_are_present():
    for name in NAMES:
        for ext in ['metal','cu']:
            path=ROOT/'src/cumlx/kernels'/f'{name}.{ext}'
            assert path.exists(), f'missing implementation: {path.name}'
            text=path.read_text()
            assert '@' not in text and len(text)>150
            assert 'TODO' not in text


def test_backend_not_imported_at_package_import():
    import subprocess, sys
    subprocess.run([sys.executable, '-c', "import cumlx, sys; assert 'mlx.core' not in sys.modules"], check=True, env={**__import__('os').environ, 'PYTHONPATH':str(ROOT/'src')})
