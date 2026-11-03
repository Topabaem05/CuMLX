"""Numerical comparison of a ported output against a golden output.

All metrics are computed in float64. ``rel_l2`` is the primary verdict metric;
``max_abs`` and ``cosine`` are reported for diagnosis.

Thresholds come from measured clean ports (Qwen2.5-0.5B and SmolLM2-135M, Apple M5 Pro
GPU and CPU, 2026-11-02; docs/CLI.md has the table). Isolated replay always runs in float32
arithmetic, so its only low-precision noise is the rounding of the port's weights:

* ``isolated_rel_l2``        upper limit for leaf modules (Linear, Embedding, Norm).
* ``composite_rel_l2_max``   upper limit for modules that contain other parametrized modules, where
  rounding noise can grow (cancellation in attention and residuals).

The diff sets both from the MEDIAN leaf error of that run, between ``noise_floor`` and the limit.
Rounding noise reaches every leaf about equally (max/median <= 3 measured), and one defective leaf
cannot move a median, so a defect cannot loosen its own threshold. A lossless port keeps the floor.
* ``rel_l2``                 chain outputs in the native dtype. In isolated mode an excess is
  ``chain_drift``, not a failure; only ``--mode chain`` fails on it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Chosen by the dtype of the PORT's parameters. Measured worst clean values (float32-origin weights):
# float32 2.8e-5; float16 leaf 5.9e-4, composite 3.5e-3; bfloat16 leaf 4.7e-3, composite 7.5e-2.
THRESHOLD_PROFILES: dict[str, dict[str, float]] = {
    'float32': {'rel_l2': 1e-4, 'isolated_rel_l2': 1e-4, 'composite_rel_l2_max': 1e-4, 'zero_atol': 1e-6},
    'float16': {'rel_l2': 5e-3, 'isolated_rel_l2': 2e-3, 'composite_rel_l2_max': 1e-2, 'zero_atol': 1e-3},
    'bfloat16': {'rel_l2': 3e-2, 'isolated_rel_l2': 1.5e-2, 'composite_rel_l2_max': 2e-1, 'zero_atol': 1e-2},
    'quantized': {'rel_l2': 3e-2, 'isolated_rel_l2': 3e-2, 'composite_rel_l2_max': 3e-2, 'zero_atol': 1e-2},
}
# A golden that holds the port's own weights leaves only float32 arithmetic noise.
PORT_WEIGHTS_RELL2 = THRESHOLD_PROFILES['float32']['isolated_rel_l2']
# Twice the worst measured ratio to the median leaf error, on 10 clean float32-origin ports:
# leaf max/median 3.0, composite max/median 48 (one Qwen2.5 decoder layer near cancellation).
LEAF_NOISE_FACTOR = 6.0
COMPOSITE_NOISE_FACTOR = 100.0


def resolve_threshold(profile: str, rel_l2: float | None = None, *, port_weights_golden: bool = False) -> dict[str, Any]:
    if profile not in THRESHOLD_PROFILES:
        raise ValueError(f'unknown threshold profile {profile!r}')
    t: dict[str, Any] = dict(THRESHOLD_PROFILES[profile])
    t['noise_floor'] = PORT_WEIGHTS_RELL2
    if port_weights_golden:
        t['isolated_rel_l2'] = t['composite_rel_l2_max'] = PORT_WEIGHTS_RELL2
    if rel_l2 is not None:
        if not (math.isfinite(rel_l2) and rel_l2 > 0):
            raise ValueError('rel_l2 threshold must be positive and finite')
        t['rel_l2'] = t['isolated_rel_l2'] = t['composite_rel_l2_max'] = float(rel_l2)
        t['fixed_by_user'] = True
    t['profile'] = profile
    return t


def noise_thresholds(threshold: dict[str, Any], median_leaf_error: float) -> tuple[float, float]:
    """(leaf, composite) isolated thresholds for this run, from its median leaf error."""
    leaf_limit, composite_limit = threshold['isolated_rel_l2'], threshold['composite_rel_l2_max']
    if threshold.get('fixed_by_user'):
        return leaf_limit, composite_limit
    floor = threshold['noise_floor']
    return (min(max(LEAF_NOISE_FACTOR * median_leaf_error, floor), max(leaf_limit, floor)),
            min(max(COMPOSITE_NOISE_FACTOR * median_leaf_error, floor), max(composite_limit, floor)))


def _finite_or_none(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


def compare(actual: np.ndarray, expected: np.ndarray, threshold: dict[str, Any]) -> dict[str, Any]:
    """Return metrics plus ``verdict`` in {'pass', 'fail'} and a ``reason`` for failures."""
    a = np.asarray(actual)
    e = np.asarray(expected)
    out: dict[str, Any] = {'shape': list(a.shape), 'expected_shape': list(e.shape)}
    if a.shape != e.shape:
        if a.size == e.size and a.size > 0:
            a = a.reshape(e.shape)
            out['reshaped'] = True
        else:
            out.update(verdict='fail', reason='shape_mismatch', rel_l2=None, max_abs=None, cosine=None)
            return out
    a64 = a.astype(np.float64)
    e64 = e.astype(np.float64)
    if not np.isfinite(a64).all():
        bad = int((~np.isfinite(a64)).sum())
        out.update(verdict='fail', reason='nonfinite_output', nonfinite_count=bad,
                   rel_l2=None, max_abs=None, cosine=None)
        return out
    if not np.isfinite(e64).all():
        out.update(verdict='fail', reason='nonfinite_golden', rel_l2=None, max_abs=None, cosine=None)
        return out
    diff = a64 - e64
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    en = float(np.linalg.norm(e64))
    an = float(np.linalg.norm(a64))
    dn = float(np.linalg.norm(diff))
    if en > 0:
        rel = dn / en
    else:
        rel = 0.0 if dn == 0 else math.inf
    cos = float(np.dot(a64.ravel(), e64.ravel()) / (an * en)) if an > 0 and en > 0 else (1.0 if an == en else 0.0)
    out.update(rel_l2=_finite_or_none(rel), max_abs=max_abs, cosine=_finite_or_none(cos),
               scale=_finite_or_none(float(np.dot(a64.ravel(), e64.ravel()) / (en * en))) if en > 0 else None)
    if en == 0:
        ok = max_abs <= threshold['zero_atol']
        out.update(verdict='pass' if ok else 'fail', basis='zero_atol')
        if not ok:
            out['reason'] = 'golden_is_zero_but_output_is_not'
        return out
    ok = rel <= threshold['rel_l2']
    out.update(verdict='pass' if ok else 'fail', basis='rel_l2')
    if not ok:
        out['reason'] = 'rel_l2_above_threshold'
    return out


# ---------------------------------------------------------------------------- op-level goldens
# `cumlx kernel verify` compares a Metal kernel with an op-level golden from a real CUDA run.
# These limits were written on 2026-11-03, BEFORE any CUDA output of these kernels had been recorded or compared.
# The reason is arithmetic, not a measurement: float32 unit roundoff is 6e-8, and these kernels are short reductions
# with an explicit FMA order, so 1e-5 leaves about 170 roundoffs for libm differences and a different lane split.
# The model-level limit of 1e-4 is ten times wider because it also holds the noise of a whole graph. `max_rel` is the
# worst element divided by the largest golden element, so one wrong element cannot hide in a long output.
# Integer outputs must match exactly. Do not raise these to clear a failure: find the cause first.
KERNEL_THRESHOLDS: dict[str, float] = {'rel_l2': 1e-5, 'max_rel': 1e-5, 'zero_atol': 1e-6}
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)


def _ulp_index(values: np.ndarray) -> np.ndarray:
    """Map float32 values to integers in which neighbouring floats differ by one. -0 and +0 are the same point."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.int64)
    magnitude = bits & 0x7fffffff
    return np.where(bits >> 31 == 1, -magnitude, magnitude)


def _first(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(mask)), mask.shape)) if mask.shape else ()


def compare_kernel_output(actual: np.ndarray, expected: np.ndarray,
                          limits: dict[str, float] | None = None) -> dict[str, Any]:
    """Compare one kernel output with its golden. Returns metrics, ``verdict`` and, on failure, ``reason``.

    NaN and infinity must sit at the same elements with the same sign. Everything else is computed in float64.
    """
    limits = KERNEL_THRESHOLDS if limits is None else limits
    a, e = np.asarray(actual), np.asarray(expected)
    out: dict[str, Any] = {'shape': list(a.shape), 'expected_shape': list(e.shape), 'dtype': str(e.dtype),
                           'rel_l2': None, 'max_rel': None, 'max_abs': None, 'max_ulp': None}
    if a.shape != e.shape:
        return {**out, 'verdict': 'fail', 'reason': 'shape_mismatch'}
    if a.dtype != e.dtype:
        return {**out, 'verdict': 'fail', 'reason': 'dtype_mismatch', 'actual_dtype': str(a.dtype)}
    if not np.issubdtype(e.dtype, np.floating):
        wrong = a != e
        out['mismatched'] = int(wrong.sum())
        if out['mismatched']:
            at = _first(wrong)
            return {**out, 'verdict': 'fail', 'reason': 'integer_mismatch', 'index': list(at),
                    'golden': e[at].item(), 'port': a[at].item()}
        return {**out, 'verdict': 'pass', 'basis': 'exact'}
    a64, e64 = a.astype(np.float64), e.astype(np.float64)
    a_nan, e_nan = np.isnan(a64), np.isnan(e64)
    out['nan_in_golden'] = int(e_nan.sum())
    if (a_nan != e_nan).any():
        at = _first(a_nan != e_nan)
        return {**out, 'verdict': 'fail', 'reason': 'nan_mismatch', 'mismatched': int((a_nan != e_nan).sum()),
                'index': list(at), 'golden': float(e64[at]), 'port': float(a64[at])}
    a_inf, e_inf = np.isinf(a64), np.isinf(e64)
    wrong = (a_inf != e_inf) | (e_inf & (a64 != e64))
    if wrong.any():
        at = _first(wrong)
        return {**out, 'verdict': 'fail', 'reason': 'infinity_mismatch', 'mismatched': int(wrong.sum()),
                'index': list(at), 'golden': float(e64[at]), 'port': float(a64[at])}
    finite = ~(e_nan | e_inf)
    out['bitwise_identical'] = a.tobytes() == e.tobytes()
    subnormal = finite & (np.abs(e64) > 0) & (np.abs(e64) < _FLOAT32_TINY)
    out['subnormal_in_golden'] = int(subnormal.sum())
    out['subnormal_flushed'] = int((subnormal & (a64 == 0)).sum())
    if not finite.any():
        return {**out, 'verdict': 'pass', 'basis': 'all_special_values'}
    diff = np.zeros(e64.shape)
    diff[finite] = a64[finite] - e64[finite]
    worst = _first(np.abs(diff) == np.max(np.abs(diff)))
    max_abs = float(np.abs(diff[worst]))
    scale = float(np.max(np.abs(e64[finite])))
    norm = float(np.linalg.norm(e64[finite]))
    out.update(max_abs=max_abs, index=list(worst), golden=float(e64[worst]), port=float(a64[worst]))
    if e.dtype == np.float32:
        out['max_ulp'] = int(np.max(np.abs(_ulp_index(a[finite]) - _ulp_index(e[finite]))))
    if norm == 0:
        if max_abs <= limits['zero_atol']:
            return {**out, 'verdict': 'pass', 'basis': 'zero_atol'}
        return {**out, 'verdict': 'fail', 'reason': 'golden_is_zero_but_output_is_not'}
    out['rel_l2'] = float(np.linalg.norm(diff) / norm)
    out['max_rel'] = max_abs / scale
    if out['rel_l2'] > limits['rel_l2']:
        return {**out, 'verdict': 'fail', 'reason': 'rel_l2_above_limit'}
    if out['max_rel'] > limits['max_rel']:
        return {**out, 'verdict': 'fail', 'reason': 'max_rel_above_limit'}
    return {**out, 'verdict': 'pass', 'basis': 'rel_l2+max_rel'}
