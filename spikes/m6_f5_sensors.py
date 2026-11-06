#!/usr/bin/env python3
"""F5: the camera sensor of cumlx.render (RGB, depth, segmentation) on one Mitsuba variant. The rules are in validation/m6-f5-cuda-limits.json.

    python spikes/m6_f5_sensors.py --variant cuda_ad_rgb --seeds 1,2,3 --out cuda/        # the V100, Mitsuba 3.8.0
    python spikes/m6_f5_sensors.py --variant metal_ad_rgb --seeds 11,12,13 --out metal/    # the Mac, Mitsuba 3.9.1
    ... --fault box_5cm | light_2pct                                                     # controls, on the Mac

For each pose of the box, it writes the RGB images of each seed, and depth and segmentation from one sample at each pixel centre
(a stratified sampler without jitter, so that both machines sample the same points), with the exact depth and label of each pixel
centre in float64 from spikes/m5_lidar.py's ray tests. spikes/m6_f5_judge.py compares two outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx.render.lidar import Box, Floor, _box, _floor  # noqa: E402
from cumlx.render.sensors import Camera, Sensors, box_scene, load_mitsuba  # noqa: E402

SPP = 1024
HALF = 0.5
POSES = {'centre': (0.0, 0.0, 0.5), 'left_low': (-0.8, 0.4, 0.5), 'far_raised': (0.6, -0.9, 1.4)}
CAMERAS = {'front': Camera(origin=(3.0, -4.0, 2.5), target=(0.0, 0.0, 0.5)), 'top': Camera(origin=(0.4, 0.3, 6.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))}
FAULTS = ('box_5cm', 'light_2pct')


def exact(camera: Camera, position: tuple[float, float, float], shape_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The distance from the camera to the first surface along each pixel-centre ray, and that surface's index in shape_ids."""
    rays = camera.ray_grid().astype(np.float64)
    origin = np.asarray(camera.origin, dtype=np.float64)
    floor = _floor(origin, rays, Floor(0.0))
    floor = np.where((np.abs(origin[0] + floor * rays[..., 0]) <= 30) & (np.abs(origin[1] + floor * rays[..., 1]) <= 30), floor, np.inf)
    box = _box(origin, rays, Box(center=position, half=(HALF, HALF, HALF)))
    depth = np.minimum(floor, box)
    label = np.where(np.isinf(depth), -1, np.where(box <= floor, shape_ids.index('box'), shape_ids.index('floor')))
    return np.where(np.isinf(depth), 0.0, depth), label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--fault', choices=FAULTS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    seeds = [int(s) for s in args.seeds.split(',')]
    mi = load_mitsuba(args.variant)
    arrays: dict[str, np.ndarray] = {}
    summary = {'variant': args.variant, 'mitsuba': mi.__version__, 'seeds': seeds, 'spp': SPP, 'fault': args.fault, 'poses': list(POSES), 'cameras': list(CAMERAS)}
    for camera_name, camera in CAMERAS.items():
        for pose_name, position in POSES.items():
            placed = (position[0] + 0.05, position[1], position[2]) if args.fault == 'box_5cm' else position
            scene = box_scene(mi, camera, HALF, placed)
            if args.fault == 'light_2pct':
                scene['light']['radiance']['value'] = [0.8 * 1.02] * 3
            key = f'{camera_name}/{pose_name}'
            rgb = Sensors(scene, args.variant, camera)
            for seed in seeds:
                arrays[f'rgb/{key}/{seed}'] = rgb.rgb(spp=SPP, seed=seed)
            # A new dict, because Mitsuba 3.8's transforms cannot be deep-copied (attempt 1, 2026-11-06).
            centred = box_scene(mi, camera, HALF, placed)
            if args.fault == 'light_2pct':
                centred['light']['radiance']['value'] = [0.8 * 1.02] * 3
            centred['sensor']['sampler'] = {'type': 'stratified', 'sample_count': 1, 'jitter': False}
            sensor = Sensors(centred, args.variant, camera)
            depth, segmentation = sensor.depth_segmentation(seed=0)
            want_depth, want_label = exact(camera, position, sensor.shape_ids)
            arrays[f'depth/{key}'], arrays[f'segmentation/{key}'] = depth, segmentation
            arrays[f'exact_depth/{key}'], arrays[f'exact_label/{key}'] = want_depth, want_label
            summary.setdefault('shape_ids', sensor.shape_ids)
    np.savez(args.out / 'sensors.npz', **arrays)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'variant': args.variant, 'mitsuba': mi.__version__, 'images': len(arrays), 'fault': args.fault}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
