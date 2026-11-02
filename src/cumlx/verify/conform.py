"""FP4 decode conformance: compare a target's NVFP4/MXFP4 dequantization with the
bit-exact CUMLX codec (``cumlx.quant``), exhaustively over scale and element codes.

Checks (``kind``):

* ``bitwise``    - must match exactly (including the sign of zero). Failure = defect.
* ``convention`` - checks the CUMLX/NVIDIA meaning of the NVFP4 tensor scale:
                   W = e2m1 * e4m3_block_scale * global_scale (global is a multiplier,
                   e.g. ModelOpt ``weight_scale_2``). Failure = the target uses a different
                   convention; ``finding`` explains how to convert.
* ``device_behavior`` - decode differs from the codec only because the GPU flushes subnormal
                   scales and results to zero. Reported, never a failure: every value is at most 3.6e-38.
* ``info``       - informative only (encoder recipe agreement); never fails.

The format checks run on the CPU stream, where MLX keeps subnormals. Decode also runs on the
GPU stream when a GPU exists, because that is where a port executes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..quant import FP4Matrix, dequantize, quantize, unpack_nibbles
from . import mlx_port

NV_FACTOR = 6.0 * 448.0  # 2688
FLUSH_LIMIT = 6.0 * 2.0 ** -127  # largest FP4 element times the largest subnormal power-of-two scale


def _vectors(fmt: str) -> tuple[list[FP4Matrix], list[int], list[int]]:
    """One row per scale code; each row holds every element code twice (two groups)."""
    g = 16 if fmt == 'nvfp4' else 32
    codes = np.concatenate([np.arange(16), np.arange(15, -1, -1)] * (g // 16)).astype(np.uint8)  # 2*g codes
    payload_row = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    scale_codes = range(127) if fmt == 'nvfp4' else range(255)
    rows, used, excluded = [], [], []
    for sc in scale_codes:
        try:
            rows.append(FP4Matrix(fmt, (1, 2 * g), payload_row[None, :].copy(),
                                  np.full((1, 2), sc, np.uint8), 1.0))
            used.append(sc)
        except ValueError:  # reconstruction would overflow float32: rejected by the CUMLX descriptor
            excluded.append(sc)
    return rows, used, excluded


def _stack(rows: list[FP4Matrix]) -> FP4Matrix:
    fmt, k = rows[0].format, rows[0].shape[1]
    return FP4Matrix(fmt, (len(rows), k), np.concatenate([r.payload for r in rows]),
                     np.concatenate([r.scales for r in rows]), 1.0)


def classify_mismatches(actual: np.ndarray, expected: np.ndarray) -> tuple[int, int]:
    """Return (bit mismatches, how many are explained by flushing subnormals to zero).

    A flushed value is a zero of the expected sign. Its expected magnitude can be a normal number, up
    to 6 * 2**-127, because the E8M0 scale 2**-127 is itself subnormal and flushes before the product.
    """
    bad = actual.view(np.uint32) != expected.view(np.uint32)
    flushed = (bad & (expected != 0) & (np.abs(expected) <= FLUSH_LIMIT) & (actual == 0)
               & (np.signbit(actual) == np.signbit(expected)))
    return int(bad.sum()), int(flushed.sum())


def _gpu_decode_check(mx, fmt: str, q: FP4Matrix, expected: np.ndarray, used: list[int], max_examples: int) -> dict:
    with mx.stream(mx.gpu):
        actual = _mlx_dequant(mx, q)
    n_bad, n_flushed = classify_mismatches(actual, expected)
    _, ex = _bitwise(actual, expected, q, used, max_examples)
    common = {'id': f'decode_bitwise.{fmt}.gpu', 'compared': int(expected.size), 'mismatches': n_bad, 'examples': ex}
    if n_bad == 0:
        return {**common, 'kind': 'bitwise', 'verdict': 'pass'}
    if n_bad == n_flushed:
        return {**common, 'kind': 'device_behavior', 'verdict': 'info', 'subnormal_flushed': n_flushed,
                'finding': f'The GPU flushed {n_flushed} results (magnitude <= {FLUSH_LIMIT:.2e}, from a subnormal scale or '
                           'product) to a zero of the same sign; the CPU stream keeps them. Every other value '
                           'is bit-identical.'}
    return {**common, 'kind': 'bitwise', 'verdict': 'fail', 'subnormal_flushed': n_flushed,
            'finding': f'{n_bad - n_flushed} GPU decode mismatches are not explained by subnormal flushing.'}


def _mlx_dequant(mx, q: FP4Matrix, global_scale: float | None = None) -> np.ndarray:
    kwargs: dict[str, Any] = {}
    if global_scale is not None:
        kwargs['global_scale'] = mx.array(np.float32(global_scale))
    w = mx.dequantize(mx.array(np.ascontiguousarray(q.payload).view(np.uint32)), mx.array(q.scales),
                      mode=q.format, dtype=mx.float32, **kwargs)
    return np.array(w)


def _bitwise(actual: np.ndarray, expected: np.ndarray, q: FP4Matrix, used: list[int], max_examples: int):
    same = (actual.view(np.uint32) == expected.view(np.uint32))
    bad = np.argwhere(~same)
    examples = []
    for r, c in bad[:max_examples]:
        byte = int(q.payload[r, c // 2])
        code = (byte >> (4 * (c % 2))) & 15
        examples.append({'scale_code': used[r] if r < len(used) else None, 'element_code': int(code),
                         'expected': float(expected[r, c]), 'actual': float(actual[r, c]),
                         'expected_bits': f'0x{int(expected.view(np.uint32)[r, c]):08x}',
                         'actual_bits': f'0x{int(actual.view(np.uint32)[r, c]):08x}'})
    return int(bad.shape[0]), examples


def run_conform(target: str = 'mlx', formats: tuple[str, ...] = ('nvfp4', 'mxfp4'),
                max_examples: int = 5) -> dict[str, Any]:
    if target != 'mlx':
        raise ValueError('only target=mlx is implemented')
    mx, _ = mlx_port.mlx()
    has_gpu = mlx_port.gpu_available(mx)
    with mx.stream(mx.cpu):
        checks = _format_checks(mx, formats, max_examples, has_gpu)
    failed = [c for c in checks if c['verdict'] == 'fail']
    try:
        from importlib.metadata import version
        mlx_version = version('mlx')
    except Exception:  # noqa: BLE001
        mlx_version = None
    return {'schema': 'cumlx.conform-report.v1', 'target': target, 'target_version': mlx_version,
            'devices': ['cpu', 'gpu'] if has_gpu else ['cpu'],
            'verdict': 'diverged' if failed else 'ok', 'checks': checks,
            'summary': {'checks': len(checks), 'failed': len(failed),
                        'failed_ids': [c['id'] for c in failed]}}


def _format_checks(mx, formats: tuple[str, ...], max_examples: int, has_gpu: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for fmt in formats:
        rows, used, excluded = _vectors(fmt)
        q = _stack(rows)
        expected = dequantize(q)
        actual = _mlx_dequant(mx, q)
        n_bad, ex = _bitwise(actual, expected, q, used, max_examples)
        checks.append({'id': f'decode_bitwise.{fmt}', 'kind': 'bitwise', 'verdict': 'pass' if n_bad == 0 else 'fail',
                       'compared': int(expected.size), 'mismatches': n_bad, 'examples': ex,
                       'scale_codes_tested': f'{used[0]}..{used[-1]} ({len(used)})',
                       'scale_codes_excluded': excluded,
                       'note': 'every (scale code, element code) pair; excluded codes overflow float32 '
                               'and are rejected by the CUMLX descriptor'})
        if fmt == 'nvfp4':
            checks.extend(_nv_global_checks(mx, q, expected, used, max_examples))
        if has_gpu:
            checks.append(_gpu_decode_check(mx, fmt, q, expected, used, max_examples))
        checks.append(_encoder_agreement(mx, fmt))
    return checks


def _nv_global_checks(mx, q: FP4Matrix, base: np.ndarray, used: list[int], max_examples: int) -> list[dict]:
    out = []
    gs_values = [0.5, 2.0 ** -12, 0.0011556312619220642, 3.0]
    rows = []
    worst_rel = 0.0
    ratios = set()
    for gs in gs_values:
        expected = (base * np.float32(gs)).astype(np.float32)  # CUMLX: (code*scale)*global
        actual = _mlx_dequant(mx, q, gs)
        nz = expected != 0
        r = np.unique(np.round(actual[nz] / expected[nz], 9)) if nz.any() else np.array([1.0])
        ratios.update(float(x) for x in r[:3])
        n_bad = int((actual.view(np.uint32) != expected.view(np.uint32)).sum())
        rows.append({'global_scale': gs, 'mismatches': n_bad})
    multiplier_ok = all(r['mismatches'] == 0 for r in rows)
    finding = None
    if not multiplier_ok:
        if ratios and all(abs(x * NV_FACTOR - 1) < 1e-6 for x in ratios):
            finding = ('The target divides global_scale by 2688 (= 6 * 448): it expects the tensor amax '
                       'convention, not a reconstruction multiplier. Passing a ModelOpt-style weight_scale_2 '
                       'directly shrinks every weight by 2688x. Convert with global_scale_target = multiplier * 2688.')
        else:
            finding = f'The target applies global_scale differently; observed ratios {sorted(ratios)[:4]}.'
    out.append({'id': 'nvfp4.global_scale.multiplier_convention', 'kind': 'convention',
                'verdict': 'pass' if multiplier_ok else 'fail', 'cases': rows, 'finding': finding})
    conv_rows = []
    for gs in gs_values:
        expected = (base * np.float32(gs)).astype(np.float32)
        actual = _mlx_dequant(mx, q, float(np.float32(gs) * np.float32(NV_FACTOR)))
        diff = np.abs(actual.view(np.int32).astype(np.int64) - expected.view(np.int32).astype(np.int64))
        worst = int(diff.max()) if diff.size else 0
        worst_rel = max(worst_rel, worst)
        conv_rows.append({'global_scale': gs, 'passed_to_target': float(np.float32(gs) * np.float32(NV_FACTOR)),
                          'mismatches': int((diff != 0).sum()), 'max_ulp': worst})
    exact = all(r['mismatches'] == 0 for r in conv_rows)
    out.append({'id': 'nvfp4.global_scale.converted_amax', 'kind': 'bitwise' if exact else 'info',
                'verdict': 'pass' if exact else 'info', 'cases': conv_rows,
                'finding': None if exact else
                'After converting with x2688 the result is within the listed ULPs but not bit-exact for '
                'non power-of-two global scales: the target rounds global/2688 in float32.'})
    return out


def _encoder_agreement(mx, fmt: str) -> dict:
    rng = np.random.default_rng(7)
    w = rng.normal(size=(64, 256)).astype(np.float32)
    ours = quantize(w, fmt)
    gs_target = None
    if fmt == 'nvfp4':
        gs_target = mx.array(np.float32(np.abs(w).max()))
    res = mx.quantize(mx.array(w), mode=fmt, **({'global_scale': gs_target} if gs_target is not None else {}))
    t_payload = np.array(res[0]).view(np.uint8)
    t_scales = np.array(res[1])
    kwargs = {'global_scale': gs_target} if gs_target is not None else {}
    t_deq = np.array(mx.dequantize(res[0], res[1], mode=fmt, dtype=mx.float32, **kwargs))
    o_deq = dequantize(ours)
    rel = lambda a: float(np.linalg.norm(a - w) / np.linalg.norm(w))  # noqa: E731
    t_codes = unpack_nibbles(t_payload, w.shape[1])
    o_codes = unpack_nibbles(ours.payload, w.shape[1])
    diff = t_codes != o_codes
    pairs = {}
    for a, b in zip(t_codes[diff].tolist(), o_codes[diff].tolist()):
        pairs[f'{a}->{b}'] = pairs.get(f'{a}->{b}', 0) + 1
    zero_only = bool(diff.any()) and set(pairs) <= {'0->8', '8->0'}
    return {'id': f'encoder_agreement.{fmt}', 'kind': 'info', 'verdict': 'info',
            'element_codes_equal_fraction': float(1 - diff.mean()),
            'scale_bytes_equal_fraction': float((t_scales == ours.scales).mean()),
            'differing_code_pairs_target_to_cumlx': dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:8]),
            'decoded_values_equal': bool(np.array_equal(t_deq, o_deq)),
            'roundtrip_rel_l2': {'target': rel(t_deq), 'cumlx': rel(o_deq)},
            'note': ('all code differences are signed zeros: the target writes +0 for tiny negatives, CUMLX keeps -0'
                     if zero_only else
                     'encoders may legitimately use different scale/rounding recipes; decode conformance is what matters')}
