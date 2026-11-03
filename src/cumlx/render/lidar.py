"""A LiDAR-style sensor: rays from one point on a fixed pattern, traced by Mitsuba on Metal. It measures geometry only.

There is no beam width, no reflectivity, no noise and no motion during a sweep. A real sensor has all of these. `analytic_ranges`
gives the exact distances on planes, boxes and spheres, so a scan can be checked without a second renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .sensors import DEFAULT_VARIANT, load_mitsuba


@dataclass(frozen=True)
class LidarPattern:
    azimuths: int = 1024
    elevations_degrees: tuple[float, ...] = tuple(float(v) for v in np.linspace(-25.0, 5.0, 16))
    max_range: float = 20.0

    def directions(self) -> np.ndarray:
        """Unit directions, elevations x azimuths x 3, with the z axis up. The first azimuth is along +x."""
        azimuth = np.arange(self.azimuths) * 2 * math.pi / self.azimuths
        elevation = np.radians(np.asarray(self.elevations_degrees))[:, None]
        return np.stack((np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.broadcast_to(np.sin(elevation), (len(self.elevations_degrees), self.azimuths))), axis=-1)


class Lidar:
    """Trace on a loaded Mitsuba scene. A miss, or a hit past `max_range`, is NaN. Each call waits for the GPU."""

    def __init__(self, scene: Any, pattern: LidarPattern | None = None, variant: str = DEFAULT_VARIANT):
        self.mi = load_mitsuba(variant)
        self.scene = scene
        self.pattern = pattern or LidarPattern()
        self._directions = self.pattern.directions()

    def scan(self, origin: Any) -> np.ndarray:
        mi = self.mi
        flat = self._directions.reshape(-1, 3).astype(np.float32)
        count = len(flat)
        o = np.asarray(origin, dtype=np.float32)
        ray = mi.Ray3f(o=mi.Point3f(mi.Float(np.full(count, o[0])), mi.Float(np.full(count, o[1])), mi.Float(np.full(count, o[2]))),
                       d=mi.Vector3f(mi.Float(flat[:, 0]), mi.Float(flat[:, 1]), mi.Float(flat[:, 2])))
        hit = self.scene.ray_intersect(ray)
        ranges = np.array(hit.t, dtype=np.float64)
        ranges[~np.array(hit.is_valid(), dtype=bool) | (ranges > self.pattern.max_range)] = np.nan
        return ranges.reshape(self._directions.shape[:2])


@dataclass(frozen=True)
class Floor:
    height: float = 0.0


@dataclass(frozen=True)
class Box:
    center: tuple[float, float, float]
    half: tuple[float, float, float]


@dataclass(frozen=True)
class Sphere:
    center: tuple[float, float, float]
    radius: float


def _floor(origin: np.ndarray, d: np.ndarray, shape: Floor) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (shape.height - origin[2]) / d[..., 2]
    return np.where(t > 0, t, np.inf)


def _box(origin: np.ndarray, d: np.ndarray, shape: Box) -> np.ndarray:
    center, half = np.asarray(shape.center), np.asarray(shape.half)
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = (center - half - origin) / d, (center + half - origin) / d
    near, far = np.nanmax(np.minimum(a, b), axis=-1), np.nanmin(np.maximum(a, b), axis=-1)
    return np.where((near <= far) & (near > 0), near, np.inf)


def _sphere(origin: np.ndarray, d: np.ndarray, shape: Sphere) -> np.ndarray:
    relative = origin - np.asarray(shape.center)
    b = d @ relative
    disc = b * b - (relative @ relative - shape.radius ** 2)
    with np.errstate(invalid='ignore'):
        t = -b - np.sqrt(disc)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def analytic_ranges(origin: Any, pattern: LidarPattern, shapes: list[Floor | Box | Sphere]) -> np.ndarray:
    """The exact distance on each ray, in float64. A miss, or a hit past `max_range`, is NaN. The origin must lie outside each shape."""
    directions = pattern.directions()
    o = np.asarray(origin, dtype=np.float64)
    nearest = np.full(directions.shape[:2], np.inf)
    for shape in shapes:
        if isinstance(shape, Floor):
            hit = _floor(o, directions, shape)
        elif isinstance(shape, Box):
            hit = _box(o, directions, shape)
        else:
            hit = _sphere(o, directions, shape)
        nearest = np.minimum(nearest, hit)
    nearest[nearest > pattern.max_range] = np.nan
    return nearest


def compare_ranges(measured: np.ndarray, expected: np.ndarray, *, range_limit: float, hit_mismatch_limit: float) -> dict[str, Any]:
    """Check a scan against exact distances. Rays must agree on hit or miss, and hits must agree on the distance. No hit at all is a failure."""
    measured_hit, expected_hit = np.isfinite(measured), np.isfinite(expected)
    both = measured_hit & expected_hit
    mismatch = float((measured_hit != expected_hit).mean())
    worst = float(np.abs(measured[both] - expected[both]).max()) if both.any() else float('inf')
    p99 = float(np.quantile(np.abs(measured[both] - expected[both]), 0.99)) if both.any() else float('inf')
    reasons = []
    if not both.any():
        reasons.append('no ray hit on both sides')
    if worst > range_limit:
        reasons.append(f'worst range error {worst:.3g} m above {range_limit}')
    if mismatch > hit_mismatch_limit:
        reasons.append(f'hit or miss differs on {mismatch:.3%} of rays, limit {hit_mismatch_limit:.3%}')
    return {'rays': int(measured.size), 'hits': int(both.sum()), 'worst_range_error_m': worst, 'p99_range_error_m': p99, 'hit_mismatch_fraction': mismatch,
            'range_limit_m': range_limit, 'hit_mismatch_limit': hit_mismatch_limit, 'pass': not reasons, 'reasons': reasons}
