#!/usr/bin/env python3
"""Physics and a camera in one loop: Genesis drops a cube, and Mitsuba renders RGB, depth and segmentation at each pose.

    pip install 'cumlx-lab[render]' genesis-world==1.4.5 torch
    python drop_box.py --out run/

Both run on Metal in one process. It is a small test scene, and it proves the path, not a robot sensor suite.

A camera image that looks right proves little, so each frame is checked against the physics state: the cube's pixels
must lie where the camera formula puts the physics position, and floor depth must match the ray and plane formula.

Four scenarios run:
    drop         the cube falls and comes to rest. Every frame must pass.
    no_gravity   the control: the cube must stay where it is, in the state and in the image.
    stale_pose   wrong on purpose: the renderer keeps the first pose. The check must fail.
    wrong_camera wrong on purpose: the renderer's camera is 0.3 m away from the camera that the check uses. The check must fail.

Exit status 0 means that the two right scenarios passed and the two wrong ones failed.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cumlx.render.sensors import (  # noqa: E402
    Camera,
    Sensors,
    box_scene,
    centroid,
    check_frame,
    load_mitsuba,
    process_gpu_seconds,
    system_used_gb,
)

HALF_SIZE = 0.25
START = (0.0, 0.0, 1.5)
START_EULER = (15.0, 20.0, 0.0)  # degrees. The cube lands on a corner and tumbles.
DT = 0.01
STEPS = 300
STEPS_PER_FRAME = 5
RGB_SPP = 16
CAMERA = Camera(origin=(4.0, -4.0, 2.5), target=(0.0, 0.0, 0.6))
# Limits, committed before the first run.
REST_HEIGHT_LIMIT = 0.02  # m, from half the side of the cube.
REST_SPEED_LIMIT = 0.02  # m/s, the vertical speed over the last frame.
CONTROL_PIXEL_LIMIT = 0.5  # The largest movement of the cube's pixels in the control, in pixels.
MAX_SYSTEM_USED_GB = 43.0


def as_array(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)


def run(gs: Any, scenario: str, out: Path | None) -> dict[str, Any]:
    gravity = (0.0, 0.0, 0.0) if scenario == 'no_gravity' else (0.0, 0.0, -9.81)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT, gravity=gravity), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    body = scene.add_entity(gs.morphs.Box(size=(2 * HALF_SIZE,) * 3, pos=START, euler=START_EULER))
    begin = time.perf_counter()
    scene.build()
    build_seconds = time.perf_counter() - begin

    mi = load_mitsuba('metal_ad_rgb')
    render_camera = replace(CAMERA, origin=(CAMERA.origin[0] + 0.3, CAMERA.origin[1], CAMERA.origin[2])) if scenario == 'wrong_camera' else CAMERA
    sensors = Sensors(box_scene(mi, render_camera, HALF_SIZE, position=START), 'metal_ad_rgb')
    body_index, floor_index = sensors.shape_ids.index('box'), sensors.shape_ids.index('floor')

    frames: list[dict[str, Any]] = []
    images = []
    seconds = {'physics': 0.0, 'pose': 0.0, 'rgb': 0.0, 'depth_segmentation': 0.0, 'check': 0.0}
    first_pose = None
    for step in range(STEPS + 1):
        if step % STEPS_PER_FRAME == 0:
            used = system_used_gb()
            if used is not None and used > MAX_SYSTEM_USED_GB:
                raise SystemExit(f'stopped: the system uses {used:.1f} GB, over the limit of {MAX_SYSTEM_USED_GB} GB')
            position, quaternion = as_array(body.get_pos()), as_array(body.get_quat())
            if first_pose is None:
                first_pose = (position.copy(), quaternion.copy())
            shown = first_pose if scenario == 'stale_pose' else (position, quaternion)
            mark = time.perf_counter()
            sensors.set_pose('box', shown[0], shown[1])
            seconds['pose'] += time.perf_counter() - mark
            mark = time.perf_counter()
            rgb = sensors.rgb(spp=RGB_SPP, seed=step)
            seconds['rgb'] += time.perf_counter() - mark
            mark = time.perf_counter()
            depth, segmentation = sensors.depth_segmentation(seed=step)
            seconds['depth_segmentation'] += time.perf_counter() - mark
            mark = time.perf_counter()
            check = check_frame(CAMERA, depth, segmentation, body_index, floor_index, position, HALF_SIZE)
            seconds['check'] += time.perf_counter() - mark
            frames.append({'step': step, 'position': position.tolist(), 'quaternion_wxyz': quaternion.tolist(), 'ok': check['ok'], 'failures': check['failures'],
                           'pixel_error': check['pixel_error'], 'floor_mean_rel': check['floor_mean_rel'], 'floor_median_rel': check['floor_median_rel'],
                           'centroid': centroid(segmentation, body_index), 'rgb_mean': rgb.mean(axis=(0, 1)).tolist()})
            if out is not None and scenario == 'drop' and step in (0, 60, 120, STEPS):
                images.append((step, rgb, depth, segmentation))
        if step < STEPS:
            mark = time.perf_counter()
            scene.step()
            seconds['physics'] += time.perf_counter() - mark

    if out is not None:
        for step, rgb, depth, segmentation in images:
            mi.util.write_bitmap(str(out / f'{scenario}_{step:03d}_rgb.png'), mi.Bitmap(rgb))
            np.savez_compressed(out / f'{scenario}_{step:03d}_sensors.npz', depth=depth, segmentation=segmentation)
    positions = np.array([f['position'] for f in frames])
    centroids = np.array([f['centroid'] for f in frames if f['centroid'] is not None])
    return {
        'scenario': scenario,
        'frames': frames,
        'frames_passed': sum(f['ok'] for f in frames),
        'frames_total': len(frames),
        'worst_pixel_error': max((f['pixel_error'] for f in frames if f['pixel_error'] is not None), default=None),
        'final_height': float(positions[-1, 2]),
        'height_range': [float(positions[:, 2].min()), float(positions[:, 2].max())],
        'centroid_travel_px': float(np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0))) if len(centroids) else None,
        'build_seconds': round(build_seconds, 2),
        'seconds': {key: round(value, 3) for key, value in seconds.items()},
        'seconds_per_frame': {key: round(seconds[key] / len(frames), 4) for key in ('pose', 'rgb', 'depth_segmentation')},
    }


def judge(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    drop, control, stale, wrong = (results[name] for name in ('drop', 'no_gravity', 'stale_pose', 'wrong_camera'))
    final_speed = abs(drop['frames'][-1]['position'][2] - drop['frames'][-2]['position'][2]) / (DT * STEPS_PER_FRAME)
    checks = {
        'drop: every frame passes': drop['frames_passed'] == drop['frames_total'],
        'drop: the cube fell': drop['height_range'][1] - drop['height_range'][0] > 0.5,
        f'drop: it rests within {REST_HEIGHT_LIMIT} m of half its side': abs(drop['final_height'] - HALF_SIZE) <= REST_HEIGHT_LIMIT,
        f'drop: its last vertical speed is under {REST_SPEED_LIMIT} m/s': final_speed <= REST_SPEED_LIMIT,
        'drop: the cube moved in the image': drop['centroid_travel_px'] > 10,
        'no_gravity: every frame passes': control['frames_passed'] == control['frames_total'],
        f'no_gravity: the pixels moved under {CONTROL_PIXEL_LIMIT} px': control['centroid_travel_px'] < CONTROL_PIXEL_LIMIT,
        'stale_pose: the check fails': stale['frames_passed'] < stale['frames_total'],
        'wrong_camera: the check fails': wrong['frames_passed'] < wrong['frames_total'],
    }
    return {'checks': checks, 'ok': all(checks.values()), 'final_speed': final_speed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', help='folder for the JSON result, four PNG images and depth and segmentation arrays')
    args = parser.parse_args()
    out = Path(args.out) if args.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    import genesis as gs
    gs.init(backend=gs.metal, precision='32', logging_level='warning', seed=7)
    results = {name: run(gs, name, out) for name in ('drop', 'no_gravity', 'stale_pose', 'wrong_camera')}
    verdict = judge(results)
    record = {**verdict, 'gpu_seconds_of_this_process': process_gpu_seconds(os.getpid()), 'results': results}
    if out is not None:
        (out / 'result.json').write_text(json.dumps(record, indent=1))
    print(json.dumps({'ok': record['ok'], 'checks': record['checks'], 'final_speed': record['final_speed'], 'gpu_seconds': record['gpu_seconds_of_this_process'],
                      'scenarios': {name: {k: r[k] for k in ('frames_passed', 'frames_total', 'worst_pixel_error', 'final_height', 'centroid_travel_px', 'seconds_per_frame')} for name, r in results.items()}}, indent=1))
    return 0 if record['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
