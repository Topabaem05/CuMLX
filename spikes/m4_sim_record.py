#!/usr/bin/env python3
"""Write validation/m4-sim.json and validation/m5-lidar.json from the raw outputs of spikes/m4_sim.py, sim_camera.py and m5_lidar.py.

    python spikes/m4_sim_record.py ~/cumlx-sim --date 2026-11-04

Every judgment here applies a rule that validation/m4-sim-limits.json states. A part with no output is 'not_run', which is not a pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIMITS = json.loads((ROOT / 'validation' / 'm4-sim-limits.json').read_text())


def load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def spread(reference: Path, other: Path, median_limit: float, p90_limit: float) -> dict[str, Any] | None:
    if not reference.exists() or not other.exists():
        return None
    a, b = np.load(reference)['values'], np.load(other)['values']
    worst = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=(0, 2))
    median, p90 = float(np.median(worst)), float(np.quantile(worst, 0.9))
    return {'median': median, 'p90': p90, 'max': float(worst.max()), 'pass': median <= median_limit and p90 <= p90_limit}


def trajectory(raw: Path, name: str, faults: list[str]) -> dict[str, Any]:
    spec = LIMITS['trajectory'][name]
    folder = raw / 'compare' / name
    rows: dict[str, Any] = {}
    for seed in spec['seeds']:
        rows[str(seed)] = {
            'genesis_vs_mujoco': spread(folder / f'mujoco_{seed}.npz', folder / f'genesis_{seed}.npz', spec['median_limit'], spec['p90_limit']),
            'time_step_control': spread(folder / f'mujoco_fine_{seed}.npz', folder / f'mujoco_{seed}.npz', spec['median_limit'], spec['p90_limit']),
            'faults': {fault: spread(folder / f'mujoco_{seed}.npz', folder / f'{fault}_{seed}.npz', spec['median_limit'], spec['p90_limit']) for fault in faults},
        }
    clean = [row['genesis_vs_mujoco'] for row in rows.values()]
    fault_rows = [v for row in rows.values() for v in row['faults'].values()]
    if any(row is None for row in [*clean, *fault_rows]):
        verdict = 'not_run'
    elif all(row['pass'] for row in clean) and not any(row['pass'] for row in fault_rows):
        verdict = 'pass'
    else:
        verdict = 'fail'
    return {'verdict': verdict, 'limits': {k: spec[k] for k in ('median_limit', 'p90_limit')}, 'seeds': rows}


def training(raw: Path, task: str, engine: str) -> dict[str, Any]:
    folder = raw / 'train' / f'{task}_{engine}'
    summary = load(folder / 'train.json')
    if summary is None:
        return {'verdict': 'not_run'}
    out: dict[str, Any] = {'train': summary, 'usage': load(folder / 'train.usage.json')}
    runs = {path.stem.removeprefix('eval_'): load(path) for path in sorted(folder.glob('eval_*.json'))}
    out['eval'] = runs
    reasons = []
    if task == 'cartpole':
        if 'trained' not in runs:
            return {**out, 'verdict': 'not_run'}
        if runs['trained']['upright_fraction_last_100'] < 0.95:
            reasons.append('trained upright fraction under 0.95')
        for control in ('early', 'zero'):
            if runs[control]['upright_fraction_last_100'] > 0.20:
                reasons.append(f'{control} control over 0.20')
    else:
        need = {'fwd': 0.8, 'back': -0.8, 'stop': 0.0, 'early': 0.8, 'zero': 0.8}
        if any(name not in runs for name in need):
            return {**out, 'verdict': 'not_run'}
        for name, command in need.items():
            speed = runs[name]['mean_forward_speed']
            if name in ('fwd', 'back') and abs(speed - command) > 0.2:
                reasons.append(f'{name} speed {speed:.3f}')
            if name not in ('fwd', 'back') and abs(speed) > 0.1:
                reasons.append(f'{name} speed {speed:.3f}, near 0 is needed')
        for name in ('fwd', 'back', 'stop'):
            if runs[name]['falls_per_world'] > 0.1:
                reasons.append(f'{name} falls {runs[name]["falls_per_world"]:.3f}')
    checks = [v for run in runs.values() for v in [run.get('check_op_against_rsl_rl')] if v is not None]
    wrong = [v for run in runs.values() for v in [run.get('check_op_wrong_activation')] if v is not None]
    absolute = [run['absolute_error'] for run in runs.values() if 'absolute_error' in run]
    if not absolute or any(a['right'] > a['limit'] for a in absolute):
        reasons.append('the policy in MLX is outside the absolute rule, or the rule did not run')
    if any(a['wrong_activation'] <= a['limit'] for a in absolute) or not wrong or any(c['verdict'] != 'diverged' for c in wrong):
        reasons.append('a wrong activation was not caught')
    out['verdict'], out['reasons'] = ('pass' if not reasons else 'fail'), reasons
    out['relative_rule'] = {'limit': LIMITS['training']['policy_in_mlx']['limit_rel_l2'], 'worst_rel_l2': max(c['worst_rel_l2'] for c in checks) if checks else None,
                            'verdict': 'pass' if checks and all(c['verdict'] == 'ok' for c in checks) else 'fail'}
    out['worst_absolute_error'] = max(a['right'] for a in absolute) if absolute else None
    return out


def first_failed(raw: Path) -> dict[str, Any]:
    """The failed runs, kept for the record."""
    bug = raw / 'compare' / 'first-run-bug' / 'cartpole'
    camera = load(raw / 'camera' / 'first-rule' / 'mujoco.json')
    out: dict[str, Any] = {}
    if bug.exists():
        spec = LIMITS['trajectory']['cartpole']
        out['cartpole_with_the_hinge_driven_too'] = {
            'cause': 'A bug of the task: the action also drove the hinge, so the pole spun. Fixed with `actuated`.',
            'genesis_vs_mujoco_seed_101': spread(bug / 'mujoco_101.npz', bug / 'genesis_101.npz', spec['median_limit'], spec['p90_limit']),
            'time_step_control_seed_101': spread(bug / 'mujoco_fine_101.npz', bug / 'mujoco_101.npz', spec['median_limit'], spec['p90_limit'])}
    failed_eval = load(raw / 'failed' / 'cartpole_mujoco_eval_with_nn_elu.json')
    if failed_eval:
        out['policy_relative_rule_on_cartpole'] = {
            'rule': 'worst rel_l2 at most 1e-4 against rsl_rl', 'result': failed_eval['check_op_against_rsl_rl'],
            'finding': 'The absolute error stays at 3e-7 to 1e-6. When the pole is balanced the action is near 8e-4, so rounding of float32 reaches this limit: torch in float32 is itself 8e-5 from torch in float64 on those batches.'}
    if camera:
        right = camera['scenarios']['right']
        out['camera_centroid_rule'] = {
            'rule': 'The pixels of a link have a centroid within 3 px of its projected physics position',
            'frames_ok': right['frames_ok'], 'frames': right['frames'], 'worst_pixel_error': right['worst_pixel_error'],
            'finding': 'The cart stays within 0.35 px. The pole breaks the limit, and the cart hides its base, which moves its centroid. Read from the geometry: this was not tested alone.'}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('raw', type=Path)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    raw = args.raw.expanduser()
    cameras = {engine: load(raw / 'camera' / f'{engine}.json') for engine in ('mujoco', 'genesis')}
    record = {
        'schema': 'cumlx.m4_sim.v1', 'date': args.date, 'limits': 'validation/m4-sim-limits.json', 'code': 'src/cumlx/sim, spikes/m4_sim.py, examples/physics-camera/sim_camera.py',
        'environments': 'Genesis 1.4.5, MuJoCo 3.17.0, torch 2.15.0, rsl-rl-lib 5.5.2, MLX 0.32.4, in existing environments. Nothing was installed.',
        'machine': 'Apple M5 Pro, 48 GB. Other jobs ran at the same time.',
        'gates_of_m4_changed': 'None. This builds the items above the gates.',
        'trajectory': {
            'cartpole_as_written': trajectory(raw, 'cartpole', ['fault_gravity', 'fault_action']),
            'cartpole_before_rail': trajectory(raw, 'cartpole_before_rail', ['fault_gravity', 'fault_action']),
            'go2_flat': trajectory(raw, 'go2_flat', ['fault_kp']),
            'findings': [
                'On cartpole, Genesis on Metal and MuJoCo agree to about 1e-6 until the carts reach the end of the rail, and then they part. The written test stays failed.',
                'The time step control is far outside the limits at the same time: a 4 times smaller time step moves MuJoCo more than changing the engine does. So this agreement says that both use one integrator at one time step, not that either is accurate.',
                'Go2 lands from 0.42 m and takes random small actions for 1 s. Each engine has its own contact model.',
                'The Go2 bodies differ: Genesis totals 15.019 kg and MuJoCo 15.208 kg from one URDF, because Genesis leaves out two small links of each calf when it merges fixed joints. The cause in Genesis was read from the link masses, and the code was not read.',
            ],
        },
        'failed_runs': first_failed(raw),
        'training': {'cartpole_mujoco': training(raw, 'cartpole', 'mujoco'), 'cartpole_genesis': training(raw, 'cartpole', 'genesis'), 'go2_flat_genesis': training(raw, 'go2_flat', 'genesis')},
        'camera': {engine: None if doc is None else {'verdict': doc['verdict'], **{name: {k: run[k] for k in ('frames_ok', 'frames', 'frames_centroid_ok', 'worst_pixel_error', 'worst_surface_p99_m')} for name, run in doc['scenarios'].items()}} for engine, doc in cameras.items()},
        'camera_notes': ['The right scenario\'s worst p99 is near its limit, at 8.8 mm against 10 mm. A pixel at this distance is about 19 mm wide, and the depth comes from one sample inside it, while the check uses the pixel centre. This reading was not tested alone.',
                         'The wrong camera case fails on 12 of 21 frames, not on all. A camera 0.3 m away is only weakly caught.'],
        'not_shown': ['Isaac Lab, its assets, or PhysX', 'USD assets: no USD reader is in these environments', 'A real robot', 'MuJoCo training for Go2: MujocoBackend is a Python loop over worlds', 'Meshes in camera images: links are boxes', 'A flagship reproduction on a new Mac'],
    }
    rate_log = raw / 'train' / 'go2_flat_genesis' / 'rate.log'
    if rate_log.exists():
        record['go2_rate_zero_actions'] = next(json.loads(line) for line in rate_log.read_text().splitlines() if line.startswith('{"task"'))
        record['go2_rate_note'] = 'Standing still with zero actions, no policy and no learner, one run. This is not the M4 throughput measure. Training on 4096 envs ran at the rate in training.go2_flat_genesis.train.'
    (ROOT / 'validation' / 'm4-sim.json').write_text(json.dumps(record, indent=1).replace(str(Path.home()), '~') + '\n')
    lidar = load(raw / 'lidar' / 'result.json')
    if lidar is not None:
        limits = json.loads((ROOT / 'validation' / 'm5-lidar-limits.json').read_text())
        lidar = {'schema': 'cumlx.m5_lidar.v1', 'date': args.date, 'limits': 'validation/m5-lidar-limits.json', **lidar, 'usage': load(raw / 'lidar' / 'usage.json'), 'not_shown': limits['not_shown'],
                 'gpu_note': 'accumulatedGPUTime is sampled every few seconds and this run lasts under a second, so process_gpu_seconds is a low lower bound.'}
        (ROOT / 'validation' / 'm5-lidar.json').write_text(json.dumps(lidar, indent=1).replace(str(Path.home()), '~') + '\n')
    print(json.dumps({'cartpole_as_written': record['trajectory']['cartpole_as_written']['verdict'], 'cartpole_before_rail': record['trajectory']['cartpole_before_rail']['verdict'],
                      'go2_flat': record['trajectory']['go2_flat']['verdict'], 'training': {k: v.get('verdict') for k, v in record['training'].items()},
                      'camera': {k: (v or {}).get('verdict') for k, v in record['camera'].items()}, 'lidar': lidar and lidar['verdict']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
