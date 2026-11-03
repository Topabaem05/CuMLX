#!/usr/bin/env python3
"""A camera on a `cumlx.sim` task: cartpole on Genesis or MuJoCo, and RGB, depth and segmentation from Mitsuba on Metal at each pose.

    python sim_camera.py --engine genesis --out result.json

The limits are in validation/m4-sim-limits.json, camera_surface: each link's pixels must lie on the box at its physics pose, and floor depth must match
the ray and plane formula. The first rule, a centroid within 3 px, failed on the pole because the cart hides its base, and it is still reported. Three scenarios:
    right         every frame must pass
    stale_pose    wrong on purpose: the renderer keeps the first pose. The check must fail
    wrong_camera  wrong on purpose: the renderer's camera is 0.3 m away from the camera that the check uses. The check must fail

Exit status 0 means that the right scenario passed and that the two wrong ones failed. The links are boxes, not meshes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cumlx.render.sensors import Camera  # noqa: E402
from cumlx.sim import get_task  # noqa: E402
from cumlx.sim.backends import GenesisBackend, MujocoBackend  # noqa: E402
from cumlx.sim.camera import TaskCamera  # noqa: E402

CAMERA = Camera(origin=(0.0, -6.0, 2.5), target=(0.0, 0.0, 1.5))
STEPS = 60
STEPS_PER_FRAME = 3
SEED = 8


def run(backend: Any, scenario: str) -> dict[str, Any]:
    import torch
    task = backend.task
    backend.reset(torch.arange(backend.num_envs, device=backend.device), task.reset(backend.num_envs, torch.Generator().manual_seed(SEED), task))
    shown = replace(CAMERA, origin=(CAMERA.origin[0] + 0.3, CAMERA.origin[1], CAMERA.origin[2])) if scenario == 'wrong_camera' else CAMERA
    sensor = TaskCamera(backend, shown, check_camera=CAMERA)
    first = None
    frames = []
    actions = torch.full((backend.num_envs, task.num_actions), 0.1, device=backend.device)
    for step in range(STEPS + 1):
        if step % STEPS_PER_FRAME == 0:
            poses = sensor.poses()
            first = first or poses
            sensor.show(first if scenario == 'stale_pose' else poses)
            rgb, depth, segmentation = sensor.frame(seed=step)
            check = sensor.check(depth, segmentation, poses)
            frames.append({'step': step, 'ok': check['ok'], 'centroid_ok': check['centroid_ok'], 'surface': check['surface'], 'pixel_error': {name: row['pixel_error'] for name, row in check['links'].items()},
                           'failures': [f for row in check['links'].values() for f in row['failures']], 'rgb_mean': float(rgb.mean())})
        backend.apply(actions)
        backend.step()
    errors = [v for frame in frames for v in frame['pixel_error'].values() if v is not None]
    return {'scenario': scenario, 'frames': len(frames), 'frames_ok': sum(f['ok'] for f in frames), 'worst_pixel_error': max(errors) if errors else None,
            'first_failures': next(({'surface': f['surface'], 'failures': f['failures']} for f in frames if not f['ok']), None), 'frames_centroid_ok': sum(f['centroid_ok'] for f in frames),
            'worst_surface_p99_m': max((row['p99_m'] for f in frames for row in f['surface'].values() if row['p99_m'] is not None), default=None),
            'pole_angle_moved': float(abs(backend.state().joint_pos[0, 1])), 'frame_rows': frames}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--engine', required=True, choices=['genesis', 'mujoco'])
    parser.add_argument('--physics', default='metal', choices=['metal', 'cpu'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    task = get_task('cartpole')
    backend = GenesisBackend(task, 1, physics=args.physics, seed=SEED) if args.engine == 'genesis' else MujocoBackend(task, 1)
    results = {scenario: run(backend, scenario) for scenario in ('right', 'stale_pose', 'wrong_camera')}
    right = results['right']
    verdict = right['frames_ok'] == right['frames'] and all(results[s]['frames_ok'] < results[s]['frames'] for s in ('stale_pose', 'wrong_camera'))
    Path(args.out).write_text(json.dumps({'engine': args.engine, 'verdict': 'pass' if verdict else 'fail', 'scenarios': results}, indent=1))
    print(json.dumps({'engine': args.engine, 'verdict': 'pass' if verdict else 'fail', **{k: {n: v[n] for n in ('frames_ok', 'frames', 'frames_centroid_ok', 'worst_pixel_error', 'worst_surface_p99_m')} for k, v in results.items()}})[:1500])
    return 0 if verdict else 1


if __name__ == '__main__':
    np.seterr(all='ignore')
    raise SystemExit(main())
