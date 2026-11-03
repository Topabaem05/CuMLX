#!/usr/bin/env python3
"""M5: a LiDAR-style raycast on Mitsuba Metal against exact distances. The limits were committed in validation/m5-lidar-limits.json before the first run.

    python spikes/m5_lidar.py --out result.json

Exit status 0 means that the right scan passed and that each wrong case failed. It renders nothing, and it measures geometry only.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx.render.lidar import Box, Floor, Lidar, LidarPattern, Sphere, analytic_ranges, compare_ranges  # noqa: E402
from cumlx.render.sensors import load_mitsuba, process_gpu_seconds  # noqa: E402

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm5-lidar-limits.json').read_text())['limits']
ORIGIN = (0.0, 0.0, 1.2)
BOX = Box(center=(4.0, 1.0, 0.5), half=(0.5, 1.0, 0.5))
SPHERE = Sphere(center=(-3.0, 2.0, 0.7), radius=0.7)
SHAPES = [Floor(0.0), BOX, SPHERE]
PATTERN = LidarPattern()


def scene(mi: Any, box: Box = BOX) -> Any:
    t = mi.ScalarTransform4f()
    return mi.load_dict({
        'type': 'scene',
        'floor': {'type': 'rectangle', 'to_world': t.scale([30.0, 30.0, 1.0])},
        'box': {'type': 'cube', 'to_world': t.translate(list(box.center)).scale(list(box.half))},
        'sphere': {'type': 'sphere', 'center': list(SPHERE.center), 'radius': SPHERE.radius},
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variant', default='metal_ad_rgb')
    args = parser.parse_args()
    mi = load_mitsuba(args.variant)
    lidar = Lidar(scene(mi), PATTERN, args.variant)
    judge = lambda measured, expected: compare_ranges(measured, expected, range_limit=LIMITS['range_limit_m'], hit_mismatch_limit=LIMITS['hit_mismatch_limit'])  # noqa: E731
    expected = analytic_ranges(ORIGIN, PATTERN, SHAPES)
    begin = time.perf_counter()
    first = lidar.scan(ORIGIN)
    first_seconds = time.perf_counter() - begin
    times = []
    for _ in range(20):
        begin = time.perf_counter()
        lidar.scan(ORIGIN)
        times.append(time.perf_counter() - begin)
    moved = (0.5, -0.5, 0.9)
    wrong_elevation = replace(PATTERN, elevations_degrees=tuple(v + 0.5 for v in PATTERN.elevations_degrees))
    moved_box = Box(center=(BOX.center[0] + 0.05, *BOX.center[1:]), half=BOX.half)
    faults = {
        'origin_5_cm_off': judge(lidar.scan((ORIGIN[0] + 0.05, ORIGIN[1], ORIGIN[2])), expected),
        'elevation_0_5_degrees_off': judge(Lidar(lidar.scene, wrong_elevation, args.variant).scan(ORIGIN), expected),
        'box_5_cm_off': judge(Lidar(scene(mi, moved_box), PATTERN, args.variant).scan(ORIGIN), expected),
    }
    right = {'origin': judge(first, expected), 'another_origin': judge(lidar.scan(moved), analytic_ranges(moved, PATTERN, SHAPES))}
    result = {
        'variant': args.variant, 'mitsuba': mi.__version__, 'rays_per_scan': int(first.size), 'first_scan_seconds': round(first_seconds, 4),
        'scans_per_second': {'min': round(1 / max(times), 1), 'max': round(1 / min(times), 1), 'repeats': len(times)},
        'right': right, 'faults': faults, 'process_gpu_seconds': process_gpu_seconds(os.getpid()),
        'verdict': 'pass' if all(r['pass'] for r in right.values()) and not any(f['pass'] for f in faults.values()) else 'fail',
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ('verdict', 'rays_per_scan', 'scans_per_second', 'process_gpu_seconds')}))
    print(json.dumps({k: {n: v[n] for n in ('worst_range_error_m', 'hit_mismatch_fraction', 'pass')} for k, v in {**right, **faults}.items()}))
    return 0 if result['verdict'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
