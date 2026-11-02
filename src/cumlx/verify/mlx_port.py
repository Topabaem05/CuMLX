"""MLX side: load a port, record module outputs, replay single modules, read weights.

MLX ``nn.Module`` has no forward hooks. ``probe`` temporarily swaps the class of
each selected module instance for a generated subclass whose ``__call__``
records the output. Parameters, attribute access and ``isinstance`` checks are
unchanged, and the original classes are restored on exit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..errors import CumlxError, missing_dependency


def mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        raise missing_dependency('mlx', 'diff', purpose='MLX port execution') from None
    return mx, nn


def load_port(path: str, loader: str = 'mlx_lm'):
    """Return an MLX ``nn.Module``. loader = 'mlx_lm' or 'python:pkg.module:function'."""
    mlx()
    if loader.startswith('python:'):
        target = loader[len('python:'):]
        mod_name, _, fn_name = target.partition(':')
        if not mod_name or not fn_name:
            raise CumlxError('E_USAGE', f'invalid --loader {loader!r}', hint="use 'python:pkg.module:function'")
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
        except (ImportError, AttributeError) as exc:
            raise CumlxError('E_MODEL_LOAD', f'cannot import loader {target!r}: {exc}') from None
        model = fn(path)
    elif loader == 'mlx_lm':
        try:
            from mlx_lm import utils as mlx_lm_utils
        except ImportError:
            raise missing_dependency('mlx_lm', 'diff', purpose='loading mlx-lm ports') from None
        p = Path(path)
        try:
            if not p.exists():
                p = Path(mlx_lm_utils.snapshot_download(path))
            model, _config = mlx_lm_utils.load_model(p)
        except Exception as exc:  # noqa: BLE001 - mlx-lm raises many types
            raise CumlxError('E_MODEL_LOAD', f'mlx-lm could not load {path!r}: {type(exc).__name__}: {exc}',
                             hint='check the path; custom architectures need --loader python:pkg.mod:fn') from None
    else:
        raise CumlxError('E_USAGE', f'unknown --loader {loader!r}', hint="use 'mlx_lm' or 'python:pkg.mod:fn'")
    _, nn = mlx()
    if not isinstance(model, nn.Module):
        raise CumlxError('E_MODEL_LOAD', f'loader returned {type(model).__name__}, not mlx.nn.Module')
    model.eval()
    return model


def named_modules(model) -> dict[str, Any]:
    return {name: m for name, m in model.named_modules()}


def first_array(obj):
    mx, _ = mlx()
    if isinstance(obj, mx.array):
        return obj
    if isinstance(obj, (tuple, list)) and obj and isinstance(obj[0], mx.array):
        return obj[0]
    return None


_PROBED: dict[type, type] = {}
_ACTIVE: dict[int, tuple['Recorder', str]] = {}


class Recorder:
    def __init__(self):
        self.calls: dict[str, int] = defaultdict(int)
        self.outputs: dict[str, Any] = {}

    def record(self, name: str, out) -> None:
        key = f'{name}#{self.calls[name]}'
        self.calls[name] += 1
        arr = first_array(out)
        if arr is not None:
            self.outputs[key] = arr


def _probed_class(cls: type) -> type:
    if cls not in _PROBED:
        base_call = cls.__call__

        def __call__(self, *args, **kwargs):
            out = base_call(self, *args, **kwargs)
            active = _ACTIVE.get(id(self))
            if active is not None:
                active[0].record(active[1], out)
            return out

        _PROBED[cls] = type(cls.__name__, (cls,), {'__call__': __call__, '__module__': cls.__module__,
                                                   '__qualname__': cls.__qualname__})
    return _PROBED[cls]


@contextlib.contextmanager
def probe(model, names: set[str] | None = None) -> Iterator[Recorder]:
    rec = Recorder()
    mods = named_modules(model)
    chosen = [(n, m) for n, m in mods.items() if names is None or n in names]
    original = []
    try:
        for n, m in chosen:
            original.append((m, type(m)))
            m.__class__ = _probed_class(type(m))
            _ACTIVE[id(m)] = (rec, n)
        yield rec
    finally:
        for m, cls in original:
            _ACTIVE.pop(id(m), None)
            m.__class__ = cls


# float32 GEMM error above this means the backend does not compute in full float32.
REDUCED_PRECISION_NOISE = 1e-5


def gpu_available(mx) -> bool:
    for backend in ('metal', 'cuda'):
        api = getattr(mx, backend, None)
        if api is not None and api.is_available():
            return True
    return False


def _device_name(mx) -> str | None:
    try:
        return (mx.device_info() if hasattr(mx, 'device_info') else mx.metal.device_info()).get('device_name')
    except Exception:  # noqa: BLE001 - only informative
        return None


def measure_gemm_noise(mx, device) -> float:
    """rel_l2 of a float32 multi-row matmul against a float64 reference, on ``device``."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 64)).astype(np.float32)
    b = rng.normal(size=(64, 64)).astype(np.float32)
    y = mx.matmul(mx.array(a), mx.array(b), stream=device)
    mx.eval(y)
    exact = a.astype(np.float64) @ b.astype(np.float64)
    return float(np.linalg.norm(np.array(y).astype(np.float64) - exact) / np.linalg.norm(exact))


def configure_execution(device: str = 'auto', tf32: str = 'off') -> dict[str, Any]:
    """Select the MLX device and matmul precision, then measure what actually took effect.

    By default MLX runs float32 matmul, attention and related ops in TF32 on hardware with matrix
    units (Apple M5 Neural Accelerators, CUDA): about 1e-3 relative error, far above float32 rounding.
    Comparisons therefore default to full float32. MLX 0.32 reads the setting once per process, at
    the first matmul, so the measurement below is the proof, not the request.
    The returned dict is the report block, plus ``handle``, the MLX device to run on.
    """
    if device not in ('auto', 'cpu', 'gpu'):
        raise CumlxError('E_USAGE', f'unknown device {device!r}', hint='use auto, cpu or gpu')
    if tf32 not in ('off', 'on'):
        raise CumlxError('E_USAGE', f'unknown tf32 setting {tf32!r}', hint='use off or on')
    mx, _ = mlx()
    os.environ['MLX_ENABLE_TF32'] = '0' if tf32 == 'off' else '1'
    config = getattr(mx, 'config', None)
    if config is not None and hasattr(config, 'update'):  # MLX releases after 0.32.4
        config.update('MLX_ENABLE_TF32', 0 if tf32 == 'off' else 1)
    if device == 'gpu' and not gpu_available(mx):
        raise CumlxError('E_BACKEND_UNAVAILABLE', 'MLX has no GPU backend here; refusing to fall back to the CPU',
                         hint='use --device cpu')
    handle = {'auto': mx.default_device(), 'cpu': mx.Device(mx.cpu), 'gpu': mx.Device(mx.gpu)}[device]
    is_gpu = handle == mx.Device(mx.gpu)
    noise = measure_gemm_noise(mx, handle)
    reduced = noise > REDUCED_PRECISION_NOISE
    if tf32 == 'off' and reduced:
        raise CumlxError('E_BACKEND_UNAVAILABLE',
                         f'full float32 matmul was requested but the measured error is {noise:.2e}: '
                         'this process already ran a matmul with TF32 on',
                         hint='start a new process with MLX_ENABLE_TF32=0, or pass --tf32 on')
    return {'device': 'gpu' if is_gpu else 'cpu', 'device_requested': device,
            'device_name': _device_name(mx) if is_gpu else None, 'tf32': tf32,
            'gemm_noise_rel_l2': noise, 'reduced_precision_matmul': reduced, 'handle': handle}


@contextlib.contextmanager
def promoted_to_float32(model) -> Iterator[bool]:
    """Run ``model`` in float32 arithmetic with its own weight values; restore the parameters on exit.

    Isolated replay checks weights and logic. Rounding that native low-precision arithmetic
    accumulates says nothing about either one; the chain run observes it.
    """
    mx, _ = mlx()
    from mlx.utils import tree_flatten, tree_unflatten
    saved = tree_flatten(model.parameters())
    if not any(mx.issubdtype(v.dtype, mx.floating) and v.dtype != mx.float32 for _, v in saved):
        yield False
        return
    model.set_dtype(mx.float32)
    try:
        yield True
    finally:
        model.update(tree_unflatten(saved))


def replay(module, x, kind: str | None):
    """Call a port module on a golden input. ``hidden`` modules also get a causal mask."""
    mx, _ = mlx()
    if kind != 'hidden':
        return first_array(module(x))
    params = list(inspect.signature(type(module).__call__).parameters.values())[2:]
    if not any(q.name == 'mask' for q in params):
        required = [q.name for q in params if q.default is inspect.Parameter.empty
                    and q.kind not in (q.VAR_POSITIONAL, q.VAR_KEYWORD)]
        if required:
            raise TypeError(f'no adapter for the required arguments {required}')
        return first_array(module(x))
    n = x.shape[1]
    if n == 1:
        return first_array(module(x, mask=None))
    try:
        return first_array(module(x, mask='causal'))
    except (TypeError, ValueError, AttributeError):  # this port wants an array mask
        return first_array(module(x, mask=mx.tril(mx.ones((n, n), dtype=mx.bool_))))


def to_numpy(arr) -> np.ndarray:
    mx, _ = mlx()
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float32)
    return np.array(arr)


def port_profile(model) -> dict:
    """Describe the port's parameter dtypes; choose a threshold profile."""
    mx, nn = mlx()
    from mlx.utils import tree_flatten
    quantized = sorted(n for n, m in named_modules(model).items() if hasattr(m, 'scales') and hasattr(m, 'bits'))
    counts: Counter = Counter()
    for _, v in tree_flatten(model.parameters()):
        if mx.issubdtype(v.dtype, mx.floating):
            counts[str(v.dtype).replace('mlx.core.', '')] += int(v.size)
    dominant = counts.most_common(1)[0][0] if counts else 'float32'
    profile = 'quantized' if quantized else (dominant if dominant in ('float32', 'float16', 'bfloat16') else 'float32')
    modes = sorted({str(getattr(named_modules(model)[n], 'mode', 'affine')) for n in quantized})
    return {'profile': profile, 'float_param_dtypes': dict(counts), 'quantized_modules': len(quantized),
            'quantization_modes': modes}


def dequantized_weights(model) -> dict[str, np.ndarray]:
    """All parameters as float32 NumPy, with quantized weights dequantized (for quantization-aware goldens)."""
    mx, nn = mlx()
    from mlx.utils import tree_flatten
    out: dict[str, np.ndarray] = {}
    for prefix, m in named_modules(model).items():
        if hasattr(m, 'scales') and hasattr(m, 'bits'):
            # In float32 on the CPU stream. mx.dequantize does its arithmetic in the dtype of the scales, so bf16
            # scales would give weights about 0.5% away from what quantized_matmul computes. Metal flushes
            # subnormals to zero. The golden must hold the exact values.
            def exact(a):  # FP4 scales are uint8 codes and stay as they are
                return a.astype(mx.float32) if a is not None and mx.issubdtype(a.dtype, mx.floating) else a
            w = mx.dequantize(m['weight'], exact(m['scales']), exact(m.get('biases')), group_size=m.group_size,
                              bits=m.bits, mode=getattr(m, 'mode', 'affine'), dtype=mx.float32, stream=mx.cpu)
            out[(prefix + '.' if prefix else '') + 'weight'] = np.array(w)
    for name, v in tree_flatten(model.parameters()):
        leaf = name.rsplit('.', 1)[-1]
        owner = name.rsplit('.', 1)[0] if '.' in name else ''
        m = named_modules(model).get(owner)
        if m is not None and hasattr(m, 'scales') and hasattr(m, 'bits') and leaf in ('weight', 'scales', 'biases'):
            continue
        if mx.issubdtype(v.dtype, mx.floating):
            out[name] = to_numpy(v.astype(mx.float32))
    return out
