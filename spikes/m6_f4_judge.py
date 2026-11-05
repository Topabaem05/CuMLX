#!/usr/bin/env python3
"""Judge one F4 run from its folder. The rules are in validation/m6-f4-limits.json, which takes each limit from an earlier, committed rule.

    python spikes/m6_f4_judge.py --run ~/cumlx-repro/result/f4 --output verdict.json    # exit 0 pass, 1 fail, 5 a part has no output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m4_sim_record import LIMITS as M4  # noqa: E402
from m4_sim_record import spread

SAME_MAC = json.loads((HERE.parent / 'validation' / 'm6-same-mac-limits.json').read_text())['f4']
ABSOLUTE = M4['training']['policy_in_mlx']['absolute_rule']['limit']
GO2 = M4['trajectory']['go2_flat']


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def part(reasons: list[str], missing: bool = False, **measured: Any) -> dict[str, Any]:
    return {'state': 'not_run' if missing else ('pass' if not reasons else 'fail'), 'reasons': reasons, **measured}


def judge(run: Path) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    code = (run / 'train.exit').read_text().strip() if (run / 'train.exit').exists() else None
    parts['train'] = part([] if code == '0' else [f'exit {code}'], code is None)
    evals = {name: read_json(run / f'eval_{name}.json') for name in ('fwd', 'back', 'stop', 'zero')}
    reasons, speeds = [], {}
    for name, command in SAME_MAC['runs'].items():
        result = evals.get(name)
        if result is None:
            reasons.append(f'{name}: no result')
            continue
        speed = result.get('mean_forward_speed')
        speeds[name] = {'speed': speed, 'falls_per_world': result.get('falls_per_world')}
        target = 0.0 if command is None else command
        tolerance = SAME_MAC['near_zero_speed'] if target == 0.0 else SAME_MAC['command_speed_tolerance']
        if speed is None or abs(speed - target) > tolerance:
            reasons.append(f'{name}: {speed} m/s against {target} +- {tolerance}')
        if name != 'zero' and (result.get('falls_per_world') is None or result['falls_per_world'] > SAME_MAC['max_falls_per_world']):
            reasons.append(f'{name}: {result.get("falls_per_world")} falls per world')
    parts['commands'] = part(reasons, all(v is None for v in evals.values()), measured=speeds)
    reasons, errors = [], {}
    for name in ('fwd', 'back', 'stop'):
        absolute = (evals.get(name) or {}).get('absolute_error')
        if absolute is None:
            reasons.append(f'{name}: the policy was not checked')
            continue
        errors[name] = absolute
        if not absolute['right'] <= ABSOLUTE:
            reasons.append(f'{name}: MLX is {absolute["right"]:.3g} from rsl_rl, limit {ABSOLUTE}')
        if not absolute['wrong_activation'] > ABSOLUTE:
            reasons.append(f'{name}: the ReLU control is within the limit ({absolute["wrong_activation"]:.3g})')
    parts['inference'] = part(reasons, measured=errors)
    reasons, rows = [], {}
    folder = run / 'compare'
    for seed in GO2['seeds']:
        clean = spread(folder / f'mujoco_{seed}.npz', folder / f'genesis_{seed}.npz', GO2['median_limit'], GO2['p90_limit'])
        fault = spread(folder / f'mujoco_{seed}.npz', folder / f'fault_kp_{seed}.npz', GO2['median_limit'], GO2['p90_limit'])
        control = spread(folder / f'mujoco_fine_{seed}.npz', folder / f'mujoco_{seed}.npz', GO2['median_limit'], GO2['p90_limit'])
        rows[str(seed)] = {'genesis_vs_mujoco': clean, 'fault_kp': fault, 'time_step_control': control}
        if clean is None or fault is None:
            reasons.append(f'seed {seed}: no trajectory')
            continue
        if not clean['pass']:
            reasons.append(f'seed {seed}: median {clean["median"]:.3g}, p90 {clean["p90"]:.3g}')
        if fault['pass']:
            reasons.append(f'seed {seed}: the stiffness fault passed')
    parts['physics'] = part(reasons, measured=rows)
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('not_run' if 'not_run' in states and 'fail' not in states else 'fail')
    return {'schema': 'cumlx.m6_f4_verdict.v1', 'rules': 'validation/m6-f4-limits.json', 'verdict': verdict,
            'grade': {'inference': 'A', 'physics': 'B'} if verdict == 'pass' else None, 'parts': parts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = judge(args.run)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({'verdict': result['verdict'], 'parts': {k: v['state'] for k, v in result['parts'].items()}}))
    return {'pass': 0, 'fail': 1}.get(result['verdict'], 5)


if __name__ == '__main__':
    raise SystemExit(main())
