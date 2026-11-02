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
            w = mx.dequantize(m['weight'], m['scales'], m.get('biases'), group_size=m.group_size,
                              bits=m.bits, mode=getattr(m, 'mode', 'affine'), dtype=mx.float32)
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
