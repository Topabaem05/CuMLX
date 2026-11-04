"""Environment inspection for ``cumlx doctor``. Imports nothing heavy unless probe=True."""
from __future__ import annotations

import importlib
from importlib import metadata, util
import json
from pathlib import Path
import platform
import shutil
import subprocess
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
    'describe': [], 'doctor': [], 'report': [], 'selftest': [], 'tracediff': [], 'rundiff': [],
    'fixture': ['torch', 'transformers'],
    'capture': ['torch', 'transformers'],
    'diff': ['mlx', 'mlx_lm'],
    'conform': ['mlx'],
    'weights': ['mlx', 'mlx_lm'],
    'scan': [],
    'recipe': [],
    'run': ['torch'],
    'port': ['torch', 'transformers', 'mlx', 'mlx_lm'],
    'kernel capture': [],  # needs the tools below
    'kernel verify': ['mlx'],
    'kernel port': [],  # the translator is checked by the command itself
}
# command -> programs that must be on PATH. They come from the CUDA toolkit, not from a Python extra.
REQUIRED_PROGRAMS = {'kernel capture': ['nvcc', 'nvidia-smi']}


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
    missing = {cmd: [m for m in mods if not deps[m]['installed']] for cmd, mods in REQUIREMENTS.items()}
    for cmd, programs in REQUIRED_PROGRAMS.items():
        missing[cmd] += [p for p in programs if shutil.which(p) is None]
    ready = {cmd: not missing[cmd] for cmd in REQUIREMENTS}
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
                from .mlx_port import REDUCED_PRECISION_NOISE, measure_gemm_noise
                noise = measure_gemm_noise(mx, mx.default_device())
                devices['mlx_default_gemm_noise_rel_l2'] = noise
                devices['mlx_default_reduced_precision_matmul'] = noise > REDUCED_PRECISION_NOISE
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
        report['memory'] = _memory()
    # The FP4 GPU kernel lab is separate from the verifier and still hardware-unvalidated.
    from ..mlx_backend import TARGET_MLX
    report['kernel_lab'] = {'mlx_api_pin': TARGET_MLX, 'device_validated': False,
                            'note': 'MLXBackend kernels are not used by capture/diff/conform. The 35 GPU tests passed '
                                    'on one device (Apple M5 Pro, MLX 0.32.2, 2026-11-02); that is not a general '
                                    'certification, and speed was not measured'}
    return report


def _memory() -> dict[str, Any]:
    """Memory on this machine, whatever its size. An unknown reading is None."""
    # Here, because `doctor --env` loads this file alone in another Python, with no cumlx package.
    from .. import sysmem

    out: dict[str, Any] = {'used_gb': sysmem.used_gb(), 'total_gb': sysmem.total_gb(), 'limit_gb': None}
    try:
        out['limit_gb'] = sysmem.limit_gb()
    except ValueError as exc:
        out['limit_error'] = str(exc)
    return out


# ---------------------------------------------------------------------- whole stack, `cumlx doctor --stack`
STACK_SCOPE = ('This Python environment only. A part that a spike installed in its own environment reads as missing here. '
               '"ok" means present, and for the layers with checked "device" that the device or variant answered. It does not '
               'mean that results are right: see `cumlx selftest` and docs/UNVERIFIED.md.')
STACK_NOT_CHECKED = [
    'nvcc and any NVIDIA machine: not probed. `cumlx doctor --require "kernel capture"` looks for the programs on this machine.',
    'Genesis, warp-metal and the translator: presence only. They were not imported, so no Metal device was opened through them.',
    'A translator given as python:MODULE:ATTRIBUTE: only woof is looked for.',
]
STACK_VARIANT = 'metal_ad_rgb'  # The same as cumlx.render.sensors.DEFAULT_VARIANT. It is here so that `_STACK_PROBE` can read this file alone.
_PIP_RENDER = "python -m pip install 'cumlx-lab[render]'"
_PIP_DIFF = "python -m pip install 'cumlx-lab[diff]'"
_PIP_CAPTURE = "python -m pip install 'cumlx-lab[capture]'"


def _import(name: str) -> Any:
    return importlib.import_module(name)


def _stack_entry(status: str, checked: str, detail: str, needed_by: str, hint: str, version: str | None = None) -> dict[str, Any]:
    return {'status': status, 'ok': status == 'ok', 'checked': checked, 'version': version, 'detail': detail,
            'needed_by': needed_by, 'install_hint': None if status == 'ok' else hint}


def _package(module: str, dist: str, needed_by: str, hint: str, other_dist: str | None = None) -> dict[str, Any]:
    """A package counts as present when its module and its own distribution are both here. A module of that name from
    another distribution is not this package: `warp` from warp-lang has no Metal backend."""
    if util.find_spec(module) is None:
        return _stack_entry('missing', 'presence', f'no module {module}', needed_by, hint)
    version = _version(dist)
    if version is None:
        also = f', and it comes from {other_dist}, which is not this package' if other_dist and _version(other_dist) else ''
        return _stack_entry('missing', 'presence', f'module {module} exists, but distribution {dist} is not installed{also}', needed_by, hint)
    return _stack_entry('ok', 'presence', f'{dist} {version} is installed', needed_by, hint, version)


def _torch_mps() -> dict[str, Any]:
    needed_by = 'cumlx run'
    if util.find_spec('torch') is None:
        return _stack_entry('missing', 'device', 'no module torch', needed_by, _PIP_CAPTURE)
    version = _version('torch')
    try:
        available = bool(_import('torch').backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        return _stack_entry('broken', 'device', f'importing torch failed: {type(exc).__name__}: {exc}', needed_by, _PIP_CAPTURE, version)
    if not available:
        return _stack_entry('unavailable', 'device', 'torch is installed, and torch.backends.mps.is_available() is False', needed_by,
                            'install a torch build with MPS on Apple Silicon', version)
    return _stack_entry('ok', 'device', 'torch.backends.mps.is_available() is True', needed_by, _PIP_CAPTURE, version)


def _metal_device() -> dict[str, Any]:
    """Ask MLX, then torch. With neither here nothing can answer, so the state is unknown, which is not a pass."""
    needed_by = 'every Metal layer'
    hint = 'needs Apple Silicon, and MLX or torch to ask it: ' + _PIP_DIFF
    problems: list[str] = []
    if util.find_spec('mlx') is not None:
        try:
            mx = _import('mlx.core')
            if mx.metal.is_available():
                try:
                    name = (getattr(mx, 'device_info', None) or mx.metal.device_info)().get('device_name')
                except Exception:  # noqa: BLE001  # The name is only a detail.
                    name = None
                return _stack_entry('ok', 'device', f"MLX reports Metal: {name or 'name not reported'}", needed_by, hint)
            problems.append('MLX reports no Metal device')
        except Exception as exc:  # noqa: BLE001
            problems.append(f'importing mlx.core failed: {type(exc).__name__}: {exc}')
    if util.find_spec('torch') is not None:
        try:
            if _import('torch').backends.mps.is_available():
                return _stack_entry('ok', 'device', 'torch reports MPS, which runs on Metal', needed_by, hint)
            problems.append('torch reports no MPS')
        except Exception as exc:  # noqa: BLE001
            problems.append(f'importing torch failed: {type(exc).__name__}: {exc}')
    if not problems:
        return _stack_entry('unknown', 'device', 'neither MLX nor torch is here to ask', needed_by, hint)
    return _stack_entry('unavailable', 'device', '; '.join(problems), needed_by, hint)


def _mitsuba_metal() -> dict[str, Any]:
    needed_by = 'cumlx.render'
    if util.find_spec('mitsuba') is None:
        return _stack_entry('missing', 'device', 'no module mitsuba', needed_by, _PIP_RENDER)
    version = _version('mitsuba')
    try:
        variants = list(_import('mitsuba').variants())
    except Exception as exc:  # noqa: BLE001
        return _stack_entry('broken', 'device', f'importing mitsuba failed: {type(exc).__name__}: {exc}', needed_by, _PIP_RENDER, version)
    if STACK_VARIANT not in variants:
        return _stack_entry('unavailable', 'device', f'Mitsuba has no variant {STACK_VARIANT} on this machine: {variants}', needed_by,
                            f'{STACK_VARIANT} needs Apple Silicon', version)
    return _stack_entry('ok', 'device', f'variant {STACK_VARIANT} is listed. It was not set, and nothing was rendered', needed_by, _PIP_RENDER, version)


def inspect_stack() -> dict[str, Any]:
    """Report each platform layer. It reads metadata and imports only what can answer a device question: no install, no
    network, and no subprocess. Only the status "ok" counts as ready."""
    layers: dict[str, dict[str, Any]] = {
        'mlx': _package('mlx', 'mlx', 'cumlx diff, conform, kernel verify', _PIP_DIFF),
        'mlx_lm': _package('mlx_lm', 'mlx-lm', 'cumlx diff, weights, port', _PIP_DIFF),
        'torch_mps': _torch_mps(),
        'metal_device': _metal_device(),
        'genesis': _package('genesis', 'genesis-world', 'physics in examples/physics-camera', 'python -m pip install genesis-world'),
        'mitsuba_metal': _mitsuba_metal(),
        'warp_metal': _package('warp', 'warp-metal', 'Warp physics on metal:0, a third-party package',
                               'not on PyPI on 2026-11-03: the wheel is at https://github.com/innate-inc/warp-metal', other_dist='warp-lang'),
        'translator': _package('woof', 'metal-woof', 'cumlx kernel port, cumlx kernel verify --engine woof',
                               'install it yourself if you accept its terms: python -m pip install metal-woof. '
                               'spikes/m3_kernels.sh uses its own environment'),
    }
    not_ready = [name for name, layer in layers.items() if not layer['ok']]
    return {'layers': layers, 'ready': [name for name in layers if name not in not_ready], 'not_ready': not_ready,
            'scope': STACK_SCOPE, 'not_checked': STACK_NOT_CHECKED}


# This file, loaded alone in another environment's Python. -I keeps that Python from reading the current folder or PYTHONPATH.
_STACK_PROBE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('cumlx_stack_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules['cumlx_stack_probe'] = module
spec.loader.exec_module(module)
print('CUMLX-STACK ' + json.dumps(module.inspect_stack()))
"""
STACK_ENV_TIMEOUT_SECONDS = 180
STACK_ENVS_SCOPE = ('Each named Python ran this detection in its own process. A layer counts as ready if one of them reports "ok". '
                    'This reports a stack spread over environments: it does not show that the parts load together in one process, '
                    'or that any result is right. Each named Python runs code of its environment, so name only your own.')


def _probe_environment(python: str) -> dict[str, Any]:
    """One environment's `inspect_stack`. A Python that cannot answer gives 'error', which never counts as ready."""
    try:
        done = subprocess.run([python, '-I', '-c', _STACK_PROBE, str(Path(__file__).resolve())], capture_output=True, text=True,
                              timeout=STACK_ENV_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}
    lines = [line for line in done.stdout.splitlines() if line.startswith('CUMLX-STACK ')]
    if done.returncode != 0 or not lines:
        return {'error': f'exit {done.returncode}: {done.stderr.strip()[-300:]}'}
    try:
        return json.loads(lines[-1][len('CUMLX-STACK '):])
    except ValueError as exc:
        return {'error': f'unreadable answer: {exc}'}


def inspect_stack_across(pythons: list[str]) -> dict[str, Any]:
    """Merge `inspect_stack` over environments. A layer takes its first "ok". With none, it keeps the first answer, and it is not ready."""
    answers = {python: _probe_environment(python) for python in pythons}
    usable = {python: answer for python, answer in answers.items() if 'error' not in answer}
    layers: dict[str, dict[str, Any]] = {}
    for answer in usable.values():
        for name, entry in answer['layers'].items():
            layers.setdefault(name, entry)
    for name in layers:
        statuses = {python: answer['layers'].get(name, {}).get('status', 'absent') for python, answer in usable.items()}
        ready_in = [python for python, status in statuses.items() if status == 'ok']
        if ready_in:
            layers[name] = usable[ready_in[0]]['layers'][name]
        layers[name] = {**layers[name], 'found_in': ready_in[0] if ready_in else None, 'environments': statuses}
    not_ready = [name for name, layer in layers.items() if not layer['ok']]
    return {'layers': layers, 'ready': [name for name in layers if name not in not_ready], 'not_ready': not_ready, 'scope': STACK_ENVS_SCOPE,
            'not_checked': STACK_NOT_CHECKED, 'environments': pythons,
            'unreadable': {python: answer['error'] for python, answer in answers.items() if 'error' in answer}}


def install_hint(modules: list[str]) -> str:
    programs = [m for m in modules if m not in DEPENDENCIES]
    if programs:
        return f"install the CUDA toolkit on a machine with an NVIDIA GPU: {', '.join(programs)} must be on PATH"
    extras = sorted({DEPENDENCIES[m][1] for m in modules if DEPENDENCIES[m][1]})
    return f"python -m pip install 'cumlx-lab[{','.join(extras)}]'" if extras else 'python -m pip install numpy'
