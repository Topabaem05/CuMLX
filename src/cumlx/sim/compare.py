"""Compare two runs of one task from the same start under the same actions. Grade B: limits on the spread over worlds."""
from __future__ import annotations

from typing import Any

import numpy as np


def worst_per_world(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Return, for each world, the largest absolute difference over steps and values. The shape is steps x worlds x values."""
    reference, other = np.asarray(reference, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if reference.shape != other.shape or reference.ndim != 3:
        raise ValueError(f'the runs have shapes {reference.shape} and {other.shape}: steps x worlds x values is needed in both')
    return np.abs(reference - other).max(axis=(0, 2))


def compare_trajectories(reference: np.ndarray, other: np.ndarray, *, median_limit: float, p90_limit: float) -> dict[str, Any]:
    """A non-finite value fails. The largest world is reported, and these limits do not bound it."""
    worst = worst_per_world(reference, other)
    finite = bool(np.isfinite(worst).all())
    median, p90 = (float(np.median(worst)), float(np.quantile(worst, 0.9))) if finite else (float('inf'), float('inf'))
    reasons = []
    if not finite:
        reasons.append('a value is not finite')
    if median > median_limit:
        reasons.append(f'median {median:.3g} above {median_limit}')
    if p90 > p90_limit:
        reasons.append(f'p90 {p90:.3g} above {p90_limit}')
    return {'median': median, 'p90': p90, 'max': float(worst.max()) if finite else float('inf'), 'worlds': int(worst.size),
            'median_limit': median_limit, 'p90_limit': p90_limit, 'pass': not reasons, 'reasons': reasons}
