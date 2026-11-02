"""Byte-level check of an MLX NVFP4 repack against the ModelOpt checkpoint it came from.

The original holds, per quantized parameter ``X``:

    X.weight          U8  [N, K/2]   E2M1 nibbles, low nibble first
    X.weight_scale    U8  [N, K/16]  E4M3 block scales
    X.weight_scale_2  F32 []         tensor multiplier

The repack holds ``X.weight`` (U32 [N, K/8], the same bytes), ``X.scales`` and optionally ``X.global_scale``.
Nothing here runs a model. Each tensor gets independent evidence: the raw bytes, and the values that
MLX decodes from the repack against what the CUMLX codec decodes from the original.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CumlxError
from ..quant import FP4Matrix, dequantize
from . import mlx_port
from .conform import NV_FACTOR
from .safetensors_io import SafetensorsReader
from .weights import checkpoint_files

SCHEMA = 'cumlx.checkpoint-report.v1'
SCALE_TOLERANCE = 1e-4  # decoded values may differ by float32 rounding of the tensor scale


def _readers(source: str, revision: str | None = None) -> dict[str, SafetensorsReader]:
    out: dict[str, SafetensorsReader] = {}
    for f in checkpoint_files(source, revision):
        r = SafetensorsReader(f)
        for name in r:
            out[name] = r
    return out


def decode_original(payload: np.ndarray, scales: np.ndarray, multiplier: float) -> np.ndarray:
    """Decode ModelOpt NVFP4 bytes with the CUMLX codec, independently of MLX."""
    if payload.ndim != 2 or scales.ndim != 2:
        raise ValueError(f'expected 2D payload and scales, got {payload.shape} and {scales.shape}')
    n, packed = payload.shape
    q = FP4Matrix('nvfp4', (n, packed * 2), payload, scales, float(multiplier))
    return dequantize(q)


def _mlx_decode(mx, weight: np.ndarray, scales: np.ndarray, global_scale: float | None) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if global_scale is not None:
        kwargs['global_scale'] = mx.array(np.float32(global_scale))
    return np.array(mx.dequantize(mx.array(weight), mx.array(scales), mode='nvfp4', dtype=mx.float32,
                                  stream=mx.cpu, **kwargs))


def compare_tensor(mx, payload: np.ndarray, scales: np.ndarray, multiplier: float, repack_weight: np.ndarray,
                   repack_scales: np.ndarray, repack_global: float | None) -> dict[str, Any]:
    """Compare one quantized parameter. ``verdict`` is pass or fail; ``reason`` names the first problem."""
    row: dict[str, Any] = {'multiplier': float(multiplier), 'repack_global_scale': repack_global}
    packed_bytes = np.ascontiguousarray(repack_weight).view(np.uint8) if repack_weight.dtype == np.uint32 else repack_weight
    if packed_bytes.size != payload.size:
        return {**row, 'verdict': 'fail', 'reason': 'payload_shape', 'repack_shape': list(repack_weight.shape),
                'original_shape': list(payload.shape)}
    row['payload_identical'] = bool(np.array_equal(packed_bytes.reshape(payload.shape), payload))
    row['scales_identical'] = bool(repack_scales.shape == scales.shape and np.array_equal(repack_scales, scales))
    expected = decode_original(payload, scales, multiplier)
    actual = _mlx_decode(mx, repack_weight, repack_scales, repack_global)
    if actual.shape != expected.shape:
        return {**row, 'verdict': 'fail', 'reason': 'decoded_shape', 'decoded_shape': list(actual.shape),
                'expected_shape': list(expected.shape)}
    e, a = expected.astype(np.float64), actual.astype(np.float64)
    en = float(np.linalg.norm(e))
    row['rel_l2'] = float(np.linalg.norm(a - e)) / en if en > 0 else float(np.linalg.norm(a))
    row['scale'] = float(np.dot(a.ravel(), e.ravel()) / (en * en)) if en > 0 else 1.0
    row['bit_identical'] = bool(np.array_equal(actual.view(np.uint32), expected.view(np.uint32)))
    if not row['payload_identical']:
        row.update(verdict='fail', reason='payload_bytes_differ')
    elif not row['scales_identical']:
        row.update(verdict='fail', reason='block_scales_differ')
    elif row['rel_l2'] > SCALE_TOLERANCE:
        row.update(verdict='fail', reason='decoded_values_differ')
        if abs(row['scale'] * NV_FACTOR - 1) < 1e-3:
            row['diagnosis'] = ('global_scale holds the ModelOpt multiplier; MLX divides it by 2688, so every weight '
                                'is 2688x too small. Store multiplier * 2688')
        elif repack_global is None and abs(row['scale'] * float(multiplier) - 1) < 1e-3:
            row['diagnosis'] = 'the tensor multiplier weight_scale_2 was dropped'
    else:
        row['verdict'] = 'pass'
    return row


def run_checkpoint(original: str, repack: str, *, revision: str | None = None, max_examples: int = 5) -> dict[str, Any]:
    mx, _ = mlx_port.mlx()
    orig, port = _readers(original, revision), _readers(repack)
    quantized = sorted(k[:-len('.weight_scale')] for k in orig if k.endswith('.weight_scale'))
    if not quantized:
        raise CumlxError('E_INPUT_INVALID', f'{original} has no ModelOpt NVFP4 tensors (*.weight_scale)',
                         hint='pass the original ModelOpt checkpoint first, then the MLX repack')
    rows: list[dict[str, Any]] = []
    for base in quantized:
        name = f'{base}.weight'
        if name not in port or f'{base}.scales' not in port:
            rows.append({'name': base, 'status': 'unverified', 'reason': 'no such quantized parameter in the repack'})
            continue
        multiplier_key = f'{base}.weight_scale_2'
        if multiplier_key not in orig:
            rows.append({'name': base, 'status': 'unverified', 'reason': 'the original has no weight_scale_2'})
            continue
        global_key = f'{base}.global_scale'
        repack_global = float(port[global_key].get(global_key)) if global_key in port else None
        c = compare_tensor(mx, orig[name].get(name), orig[f'{base}.weight_scale'].get(f'{base}.weight_scale'),
                           float(orig[multiplier_key].get(multiplier_key)), port[name].get(name),
                           port[f'{base}.scales'].get(f'{base}.scales'), repack_global)
        rows.append({'name': base, 'status': c.pop('verdict'), **c})
    plain = [k for k in orig if k.rsplit('.', 1)[0] not in set(quantized) or not k.rsplit('.', 1)[-1].startswith('weight')]
    identical = 0
    differ: list[str] = []
    missing: list[str] = []
    for k in plain:
        if k not in port:
            missing.append(k)
            continue
        a, b = orig[k].get(k), port[k].get(k)
        if a.shape == b.shape and np.array_equal(a.astype(np.float64), b.astype(np.float64), equal_nan=True):
            identical += 1
        else:
            differ.append(k)
    counts = {k: sum(r['status'] == k for r in rows) for k in ('pass', 'fail', 'unverified')}
    verdict = 'diverged' if counts['fail'] or differ else ('incomplete' if counts['unverified'] or missing else 'ok')
    ratios = sorted({round(r['repack_global_scale'] / r['multiplier'], 3) for r in rows
                     if r.get('repack_global_scale') and r.get('multiplier')})
    return {'schema': SCHEMA, 'verdict': verdict, 'original': original, 'repack': repack,
            'summary': {**counts, 'quantized': len(rows), 'plain_identical': identical,
                        'plain_differ': len(differ), 'plain_missing': len(missing),
                        'bit_identical': sum(bool(r.get('bit_identical')) for r in rows),
                        'worst_rel_l2': max((r['rel_l2'] for r in rows if r.get('rel_l2') is not None), default=None),
                        'global_scale_to_multiplier_ratios': ratios[:8]},
            'first_failure': next((r for r in rows if r['status'] == 'fail'), None),
            'plain': {'identical': identical, 'differ': differ[:max_examples * 4], 'missing': missing[:max_examples * 4]},
            'parameters': rows}


def write_report(rep: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(rep, indent=1) + '\n')
