"""Compare path-traced images from two devices, and give grade B or refuse.

A path tracer gives a noisy estimate, and two devices do not share their random numbers. So the question is
statistical: is the difference between the devices as small as the difference between two seeds on one device?

The limits below were committed before any Mitsuba image was rendered for this comparison, and they must not change
after a result is seen (docs/ROADMAP-PLATFORM.md, section 2). The earlier M0 run gave one number: Metal against
scalar_rgb, Cornell box, 1024 spp, 52.1 dB. The scale of the limits comes from that, and validation/m5-render-limits.json
has the reasons and the synthetic cases.

Images are floating-point linear RGB, height x width x 3. PSNR is computed on values clipped to [0, 1].
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ..errors import CumlxError

MIN_SEEDS = 3
PSNR_FLOOR_DB = 40.0  # Every pair of images, on a device or across devices, must reach this. Below it the noise can hide a real error.
NOISE_RATIO = 2.0  # The mean squared error across devices may be this many times the mean within a device (about 3 dB).
STANDARD_ERRORS = 3.0  # Each channel mean may differ by this many standard errors of the difference in means ...
REL_FLOOR = 0.01  # ... and never by less than this part of the golden mean.


def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise CumlxError('E_INPUT_INVALID', f'image shapes differ: {a.shape} and {b.shape}')
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return math.inf
    return float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))


def _db(error: float) -> float:
    if not math.isfinite(error):
        return -math.inf
    return -10 * math.log10(error) if error > 0 else math.inf


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return _db(mse(a, b))


def _within(images: list[np.ndarray]) -> list[float]:
    return [mse(a, b) for a, b in itertools.combinations(images, 2)]


def compare_images(golden: dict[int, np.ndarray], candidate: dict[int, np.ndarray]) -> dict[str, Any]:
    """golden and candidate map a seed to one image of one scene. The two sides must use separate seeds."""
    for side, images in (('golden', golden), ('candidate', candidate)):
        if len(images) < MIN_SEEDS:
            raise CumlxError('E_INPUT_INVALID', f'{side} has {len(images)} images, and the comparison needs {MIN_SEEDS} seeds')
    if set(golden) & set(candidate):
        raise CumlxError('E_INPUT_INVALID', 'the two sides share seeds; use separate seeds, so that shared random numbers cannot flatter the result')
    across = [mse(g, c) for g, c in itertools.product(golden.values(), candidate.values())]
    within_golden, within_candidate = _within(list(golden.values())), _within(list(candidate.values()))
    within = float(np.mean(within_golden + within_candidate))
    across_mean = float(np.mean(across))
    finite = all(math.isfinite(x) for x in [*across, *within_golden, *within_candidate])
    if not finite:
        return _result('diverged', ['an image is not finite'], golden, candidate, -math.inf, -math.inf, across_mean, within, None, None)

    g_means = np.stack([np.clip(x, 0, 1).mean(axis=(0, 1)) for x in golden.values()])
    c_means = np.stack([np.clip(x, 0, 1).mean(axis=(0, 1)) for x in candidate.values()])
    standard_error = np.sqrt(g_means.var(axis=0, ddof=1) / len(g_means) + c_means.var(axis=0, ddof=1) / len(c_means))
    allowed = np.maximum(STANDARD_ERRORS * standard_error, REL_FLOOR * g_means.mean(axis=0))
    difference = np.abs(g_means.mean(axis=0) - c_means.mean(axis=0))

    worst_within_db = _db(max(within_golden + within_candidate))
    worst_across_db = _db(max(across))
    failures = []
    # When two seeds on one device are under the floor too, the noise explains it. That case ends as inconclusive below.
    if worst_across_db < PSNR_FLOOR_DB <= worst_within_db:
        failures.append(f'worst PSNR across devices {worst_across_db:.2f} dB is under {PSNR_FLOOR_DB} dB')
    if within > 0 and across_mean > NOISE_RATIO * within:
        failures.append(f'MSE across devices is {across_mean / within:.2f} times the MSE within a device, limit {NOISE_RATIO}')
    if within == 0 and across_mean > 0:
        failures.append('each device repeats exactly across seeds, and the two devices differ')
    for channel, name in enumerate('RGB'):
        if difference[channel] > allowed[channel]:
            failures.append(f'{name} mean differs by {difference[channel]:.3g}, allowed {allowed[channel]:.3g}')

    if failures:
        verdict = 'diverged'
    elif worst_within_db < PSNR_FLOOR_DB:
        # Two seeds on one device already differ too much: more samples are needed. This is not a pass.
        verdict = 'inconclusive'
        failures.append(f'worst PSNR within a device {worst_within_db:.2f} dB is under {PSNR_FLOOR_DB} dB, so the noise is too high to judge')
    else:
        verdict = 'ok'
    return _result(verdict, failures, golden, candidate, worst_across_db, worst_within_db, across_mean, within, difference, allowed)


def _result(verdict: str, failures: list[str], golden: dict[int, np.ndarray], candidate: dict[int, np.ndarray], worst_across_db: float,
            worst_within_db: float, across_mean: float, within: float, difference: Any, allowed: Any) -> dict[str, Any]:
    return {
        'verdict': verdict,
        'grade': 'B' if verdict == 'ok' else None,
        'failures': failures,
        'worst_psnr_across_db': worst_across_db,
        'worst_psnr_within_db': worst_within_db,
        'mean_mse_across': across_mean,
        'mean_mse_within': within,
        'mse_ratio': across_mean / within if within > 0 and math.isfinite(across_mean) else None,
        'channel_mean_difference': None if difference is None else difference.tolist(),
        'channel_mean_allowed': None if allowed is None else allowed.tolist(),
        'seeds': {'golden': sorted(golden), 'candidate': sorted(candidate)},
        'limits': {'psnr_floor_db': PSNR_FLOOR_DB, 'noise_ratio': NOISE_RATIO, 'standard_errors': STANDARD_ERRORS,
                   'rel_floor': REL_FLOOR, 'min_seeds': MIN_SEEDS},
    }
