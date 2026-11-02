"""Environment inspection for ``cumlx doctor``. Imports nothing heavy unless probe=True."""
from __future__ import annotations

from importlib import metadata, util
import platform
import sys
from typing import Any

DEPENDENCIES = {
    # module: (distribution, extra)
    'numpy': ('numpy', None),
    'torch': ('torch', 'capture'),
    'transformers': ('transformers', 'capture'),
    'mlx': ('mlx', 'diff'),
    'mlx_lm': ('mlx-lm', 'diff'),
}
# command -> required modules
REQUIREMENTS = {
    'describe': [], 'doctor': [], 'report': [], 'selftest': [],
    'fixture': ['torch', 'transformers'],
    'capture': ['torch', 'transformers'],
    'diff': ['mlx', 'mlx_lm'],
    'conform': ['mlx'],
}


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def inspect_environment(*, probe: bool = False) -> dict[str, Any]:
    deps = {}
    for mod, (dist, extra) in DEPENDENCIES.items():
        present = util.find_spec(mod) is not None
        deps[mod] = {'installed': present, 'version': _version(dist) if present else None, 'extra': extra}
    ready = {cmd: all(deps[m]['installed'] for m in mods) for cmd, mods in REQUIREMENTS.items()}
    missing = {cmd: [m for m in mods if not deps[m]['installed']] for cmd, mods in REQUIREMENTS.items()}
    report: dict[str, Any] = {
        'python': sys.version.split()[0], 'platform': platform.platform(), 'machine': platform.machine(),
        'dependencies': deps, 'commands_ready': ready, 'missing_by_command': {k: v for k, v in missing.items() if v},
    }
    if probe:
        devices: dict[str, Any] = {}
        if deps['mlx']['installed']:
            try:
                import mlx.core as mx
                devices['mlx_default_device'] = str(mx.default_device())
                devices['mlx_metal_available'] = bool(getattr(mx, 'metal', None) and mx.metal.is_available())
                cuda = getattr(mx, 'cuda', None)
                devices['mlx_cuda_available'] = bool(cuda and cuda.is_available())
            except Exception as exc:  # noqa: BLE001
                devices['mlx_error'] = f'{type(exc).__name__}: {exc}'
                report['commands_ready'].update(diff=False, conform=False)
        if deps['torch']['installed']:
            try:
                import torch
                devices['torch_cuda_available'] = bool(torch.cuda.is_available())
                mps = getattr(torch.backends, 'mps', None)
                devices['torch_mps_available'] = bool(mps and mps.is_available())
            except Exception as exc:  # noqa: BLE001
                devices['torch_error'] = f'{type(exc).__name__}: {exc}'
                report['commands_ready'].update(capture=False, fixture=False)
        report['devices'] = devices
    # The FP4 GPU kernel lab is separate from the verifier and still hardware-unvalidated.
    from ..mlx_backend import TARGET_MLX
    report['kernel_lab'] = {'mlx_api_pin': TARGET_MLX, 'device_validated': False,
                            'note': 'MLXBackend kernels are not used by capture/diff/conform'}
    return report


def install_hint(modules: list[str]) -> str:
    extras = sorted({DEPENDENCIES[m][1] for m in modules if DEPENDENCIES[m][1]})
    return f"python -m pip install 'cumlx-lab[{','.join(extras)}]'" if extras else 'python -m pip install numpy'
