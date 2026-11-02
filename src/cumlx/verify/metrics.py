"""Numerical comparison of a ported output against a golden output.

All metrics are computed in float64. ``rel_l2`` is the primary verdict metric;
``max_abs`` and ``cosine`` are reported for diagnosis. Thresholds are initial
beta values and must be recalibrated on clean ports (roadmap W2).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Primary threshold on rel_l2, chosen by the dtype of the PORT's parameters.
# Basis: float32 CPU reductions differ by ~1e-7 relative per op; bf16 has an
# 8-bit significand (~4e-3 unit roundoff), fp16 11-bit (~5e-4).
THRESHOLD_PROFILES: dict[str, dict[str, float]] = {
    'float32': {'rel_l2': 1e-4, 'zero_atol': 1e-6},
    'float16': {'rel_l2': 5e-3, 'zero_atol': 1e-3},
    'bfloat16': {'rel_l2': 3e-2, 'zero_atol': 1e-2},
    'quantized': {'rel_l2': 3e-2, 'zero_atol': 1e-2},
}


def resolve_threshold(profile: str, rel_l2: float | None = None) -> dict[str, float]:
    if profile not in THRESHOLD_PROFILES:
        raise ValueError(f'unknown threshold profile {profile!r}')
    t = dict(THRESHOLD_PROFILES[profile])
    if rel_l2 is not None:
        if not (math.isfinite(rel_l2) and rel_l2 > 0):
            raise ValueError('rel_l2 threshold must be positive and finite')
        t['rel_l2'] = float(rel_l2)
    t['profile'] = profile  # type: ignore[assignment]
    return t


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
    out.update(rel_l2=_finite_or_none(rel), max_abs=max_abs, cosine=_finite_or_none(cos))
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
