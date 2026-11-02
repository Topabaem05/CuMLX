"""Single-op parity: compare a PyTorch function with an MLX function on the same inputs.

Use this for a hand-written Metal or CUDA kernel, or any op too small to need a model trace.

    report = cumlx.check_op(torch.nn.functional.silu, lambda x: mx.fast.metal_kernel(...)(x),
                            inputs=[np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)])
    assert report['verdict'] == 'ok'
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import missing_dependency
from . import mlx_port
from .metrics import compare


def _as_tuple(sample) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(a) for a in (sample if isinstance(sample, (tuple, list)) else (sample,)))


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def check_op(torch_fn: Callable, mlx_fn: Callable, inputs: Sequence[Any], *, rel_l2: float = 1e-4,
             device: str = 'auto', tf32: str = 'off', name: str | None = None) -> dict[str, Any]:
    """Run both functions on every sample and return a report.

    ``inputs`` is a list of samples; a sample is an array or a tuple of positional arrays. PyTorch runs on the
    CPU in float32. ``verdict`` is ok, diverged, or incomplete when ``inputs`` is empty, which is not a pass.
    """
    try:
        import torch
    except ImportError:
        raise missing_dependency('torch', 'capture', purpose='check_op') from None
    mx, _ = mlx_port.mlx()
    execution = mlx_port.configure_execution(device, tf32)
    handle = execution.pop('handle')
    threshold = {'rel_l2': float(rel_l2), 'zero_atol': float(rel_l2)}
    cases = []
    with mx.stream(handle):
        for i, sample in enumerate(inputs):
            arrays = _as_tuple(sample)
            with torch.no_grad():
                expected = _first(torch_fn(*[torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]))
            actual = _first(mlx_fn(*[mx.array(a) for a in arrays]))
            mx.eval(actual)
            row = compare(mlx_port.to_numpy(actual), expected.detach().to(torch.float32).numpy(), threshold)
            cases.append({'sample': i, 'input_shapes': [list(a.shape) for a in arrays], **row})
    failed = [c for c in cases if c['verdict'] != 'pass']
    verdict = 'incomplete' if not cases else ('diverged' if failed else 'ok')
    worst = [c['rel_l2'] for c in cases if c.get('rel_l2') is not None]
    return {'schema': 'cumlx.check-op.v1', 'name': name or getattr(mlx_fn, '__name__', 'op'), 'verdict': verdict,
            'threshold_rel_l2': float(rel_l2), 'execution': execution, 'worst_rel_l2': max(worst, default=None),
            'cases': cases, 'failed': len(failed)}
