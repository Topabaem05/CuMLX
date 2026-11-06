"""MLX side: load a port, record module outputs, replay single modules, read weights.

MLX ``nn.Module`` has no forward hooks. ``probe`` temporarily swaps the class of
each selected module instance for a generated subclass whose ``__call__``
records the output. Parameters, attribute access and ``isinstance`` checks are
unchanged, and the original classes are restored on exit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
import contextlib
import importlib
import inspect
import os
from pathlib import Path
import re
from typing import Any

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
        if not p.exists() and str(path).startswith(('/', '.', '~')):  # a local path, not a Hub id
            raise CumlxError('E_INPUT_NOT_FOUND', f'MLX port directory not found: {path}',
                             hint='pass an existing directory, or a Hugging Face id such as mlx-community/<name>')
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
    return dict(model.named_modules())


def first_array(obj):
    mx, _ = mlx()
    if isinstance(obj, mx.array):
        return obj
    if isinstance(obj, (tuple, list)) and obj and isinstance(obj[0], mx.array):
        return obj[0]
    return None


_PROBED: dict[type, type] = {}
_ACTIVE: dict[int, tuple[Recorder, str]] = {}


class Recorder:
    def __init__(self):
        self.calls: dict[str, int] = defaultdict(int)
        self.outputs: dict[str, Any] = {}
        self.masks: dict[str, Any] = {}

    def record(self, name: str, out, args: tuple = (), kwargs: dict | None = None) -> None:
        key = f'{name}#{self.calls[name]}'
        self.calls[name] += 1
        kwargs = kwargs or {}
        # The call is (x, mask, cache); a module called with x alone had no mask.
        self.masks[key] = kwargs['mask'] if 'mask' in kwargs else (args[1] if len(args) > 1 else None)
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
                active[0].record(active[1], out, args, kwargs)
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


def replay_experts(module, x, indices, scores):
    """Run MoE experts with the golden routing and combine them as the original does."""
    y = module(x[None], indices[None])
    return (y * scores[None, ..., None]).sum(axis=-2)


# Mixtral names its expert projections w1 (gate), w3 (up), w2 (down) in transformers 4.
EXPERT_PROJ = {'w1': 'gate_proj', 'w3': 'up_proj', 'w2': 'down_proj'}
_ONE_EXPERT = re.compile(r'^(.*\.switch_mlp)\.(\d+)(?:\.(\w+))?$')


def expert_target(name: str, mods: dict) -> tuple[str, int, str | None] | None:
    """Resolve one expert of a transformers 4 ``ModuleList`` to the port's stacked experts.

    ``name`` is already mapped, so ``mlp.experts.3.up_proj`` reads ``mlp.switch_mlp.3.up_proj``. Returns
    (switch module name, expert index, projection or None for the whole expert), or None if the port has no such
    expert. The port stacks every expert in one ``switch_mlp``, so expert 3 is index 3 of each stacked weight.
    """
    m = _ONE_EXPERT.match(name)
    if not m or m.group(1) not in mods:
        return None
    switch, index, proj = m.group(1), int(m.group(2)), m.group(3)
    if proj is not None:
        proj = EXPERT_PROJ.get(proj, proj)
        if f'{switch}.{proj}' not in mods:
            return None
    linear = mods.get(f'{switch}.{proj or "gate_proj"}')
    count = getattr(linear, 'num_experts', None)
    if count is None or index >= count:
        return None
    return switch, index, proj


def replay_one_expert(mods: dict, target: tuple[str, int, str | None], x):
    """Run expert ``index`` alone on the tokens the original routed to it, without a routing weight.

    A transformers 4 expert returns its output before the routing weight is applied, so the comparison is exact.
    """
    mx, _ = mlx()
    switch, index, proj = target
    indices = mx.full((*x.shape[:-1], 1), index, dtype=mx.uint32)
    if proj is None:
        return replay_experts(mods[switch], x, indices, mx.ones(indices.shape, dtype=x.dtype))
    y = mods[f'{switch}.{proj}'](mx.expand_dims(x, (-2, -3)), indices)  # the layout SwitchGLU passes
    return y.reshape(*x.shape[:-1], y.shape[-1])


NO_RECORDED_MASK = object()


def replay(module, x, kind: str | None, mask: Any = NO_RECORDED_MASK):
    """Call a port module on a golden input.

    A ``hidden`` module gets the mask that the port itself passed to it in its own forward pass. A convolution in a
    hybrid model may take a padding mask or none, so a causal mask cannot be assumed. A causal mask is the fallback for
    a module that the port's forward pass did not reach. A wrong mask from the parent still shows: the output differs.
    """
    mx, _ = mlx()
    if kind != 'hidden':
        return first_array(module(x))
    if mask is not NO_RECORDED_MASK:
        params = list(inspect.signature(type(module).__call__).parameters.values())[2:]
        if any(q.name == 'mask' for q in params):
            return first_array(module(x, mask=mask))
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


def _causal_mask(x):
    return 'causal' if x.shape[1] > 1 else None


def make_caches(model) -> list:
    """The port's own KV caches, one per layer. Raises if this port has no mlx-lm cache support."""
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def _layer_cache(caches: list, name: str):
    match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
    if match is None or int(match.group(1)) >= len(caches):
        raise ValueError(f'no KV cache for module {name!r}')
    return caches[int(match.group(1))]


def replay_cached(model, module, name: str, earlier: list, x, kind: str, masks: list | None = None):
    """Fill the port's own KV cache with the golden inputs of the earlier steps, then run the new input.

    Every input is golden, so a failure means this module's own cache handling is wrong
    (position offset, mask, cache update), not an upstream value.
    """
    caches = make_caches(model)
    if kind == 'tokens_cached':
        for step in earlier:
            module(step, cache=caches)
        return first_array(module(x, cache=caches))
    cache = _layer_cache(caches, name)
    steps = [*earlier, x]
    use = masks if masks is not None and len(masks) == len(steps) else [_causal_mask(s) for s in steps]
    for step, mask in zip(earlier, use):
        module(step, mask=mask, cache=cache)
    return first_array(module(x, mask=use[-1], cache=cache))


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
            kwargs = dict(group_size=m.group_size, bits=m.bits, mode=getattr(m, 'mode', 'affine'), dtype=mx.float32,
                          stream=mx.cpu)
            expert_scale = getattr(m, 'global_scale', None)  # NVFP4 MoE: one tensor scale per expert
            if expert_scale is not None:
                w = mx.stack([mx.dequantize(m['weight'][e], m['scales'][e], global_scale=expert_scale[e], **kwargs)
                              for e in range(m['weight'].shape[0])])
            else:
                w = mx.dequantize(m['weight'], exact(m['scales']), exact(m.get('biases')), **kwargs)
            out[(prefix + '.' if prefix else '') + 'weight'] = np.array(w)
    for name, v in tree_flatten(model.parameters()):
        leaf = name.rsplit('.', 1)[-1]
        owner = name.rsplit('.', 1)[0] if '.' in name else ''
        m = named_modules(model).get(owner)
        if m is not None and hasattr(m, 'scales') and hasattr(m, 'bits') and leaf in ('weight', 'scales', 'biases', 'global_scale'):
            continue
        if mx.issubdtype(v.dtype, mx.floating):
            out[name] = to_numpy(v.astype(mx.float32))
    return out
