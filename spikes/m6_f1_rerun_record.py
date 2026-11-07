#!/usr/bin/env python3
"""Write validation/m6-f1-rerun.json from the two folders of an F1 run under validation/m6-f1-rerun-limits.json. It reruns
spikes/m6_f1_judge.py on them and checks the conditions that the rerun rules add. It measures nothing itself.

    python spikes/m6_f1_rerun_record.py --cuda ~/cumlx-m6-f1rerun --mac ~/cumlx-repro/result --out validation/m6-f1-rerun.json

--cuda is the folder that spikes/m6_cuda_v100.sh fetch wrote (its f1rerun holds the F1 output). --mac is the result folder of
spikes/m6_same_mac.sh part f1 (its f1 holds the F1 output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import m6_f1_judge as j1  # noqa: E402

RULES = REPO / 'validation' / 'm6-f1-rerun-limits.json'


def text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def leaf_numbers(result: dict[str, Any] | None, golden: dict[str, Any] | None) -> dict[str, Any] | None:
    """The leaves of one replay: the worst, how many are over the limit, how many are equal to the bit, and the judge's state."""
    if result is None or golden is None:
        return None
    leaves = result['leaves']
    worst = max(leaves, key=lambda name: leaves[name]['rel_l2'])
    limit = j1.LIMITS['parts']['leaves']['rel_l2_at_most']
    return {'worst': worst, 'worst_rel_l2': leaves[worst]['rel_l2'], 'over_limit': sum(v['rel_l2'] > limit for v in leaves.values()),
            'bit_equal': sum(v['equal_fraction'] == 1.0 for v in leaves.values()), 'of': len(leaves),
            'weights_cb_differ_worst': max((w['cb_differ_fraction'] for w in result['weights'].values()), default=None),
            'leaves_state': j1.JUDGES['leaves'](result, golden)['state'], 'weights_state': j1.JUDGES['weights'](result, golden)['state']}


def assess(cuda: Path, mac: Path, rules: dict[str, Any], rules_in_commit: bool) -> dict[str, Any]:
    cuda_f1, mac_f1 = cuda / 'f1rerun', mac / 'f1'
    verdict = j1.judge(cuda_f1, mac_f1)
    golden = j1.read_json(cuda_f1 / 'golden' / 'summary.json')
    mac_golden = j1.read_json(mac_f1 / 'mac-golden' / 'summary.json')
    commits = {'cuda': text(cuda / 'cumlx_commit.txt'), 'mac': text(mac / 'cumlx_commit.txt')}
    # A reason here makes the run not complete: it is recorded, and it can be run again (complete_run in the rules).
    not_complete = []
    if not commits['cuda'] or commits['cuda'] != commits['mac']:
        not_complete.append(f'the machines ran other commits: {commits}')
    if not rules_in_commit:
        not_complete.append('the commit does not hold validation/m6-f1-rerun-limits.json as it is now')
    if (golden or {}).get('prompt') != rules['prompt']:
        not_complete.append(f'the CUDA golden has the prompt {(golden or {}).get("prompt")!r}, not the one of the rules')
    if (mac_golden or {}).get('prompt') != rules['prompt']:
        not_complete.append(f'the Mac golden has the prompt {(mac_golden or {}).get("prompt")!r}, not the one of the rules')
    if (cuda / 'busy_at_start.txt').exists():
        not_complete.append('the GPU was busy at the start')
    if (cuda / 'incomplete.txt').exists() or not (cuda / 'done.txt').exists():
        not_complete.append('the CUDA side did not finish')
    if verdict['verdict'] == 'not_run':
        not_complete.append('a part has no output')
    complete = not not_complete
    reported = {'ieee_row_scale': leaf_numbers(j1.read_json(mac_f1 / 'reported_ieee_row_scale.json'), golden),
                'ieee_row_scale_exit': j1.read_exit(mac_f1, 'reported_ieee_row_scale'),
                'right_adapter': leaf_numbers(j1.read_json(mac_f1 / 'replay.json'), golden),
                'golden_text': (golden or {}).get('text'), **verdict.get('reported_not_judged', {})}
    return {
        'schema': 'cumlx.m6_f1_rerun.v1',
        'rules': 'validation/m6-f1-rerun-limits.json',
        'prompt': rules['prompt'],
        'commits': commits,
        'complete': complete,
        'not_complete_because': not_complete,
        'verdict': verdict['verdict'],
        'grade': 'A' if complete and verdict['verdict'] == 'pass' else None,
        'parts': {name: {k: v for k, v in part.items() if k != 'measured'} for name, part in verdict['parts'].items()},
        'reported_not_judged': reported,
        'cuda_versions': j1.read_json(cuda_f1 / 'versions.json'),
        'gpu_at_start': text(cuda / 'machine_start.txt'),
        'mac_peak_memory_gb': text(mac / 'peak_used_gb.txt'),
        'mac_verdict': text(mac / 'verdict.txt'),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', required=True, type=Path)
    parser.add_argument('--mac', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    rules = json.loads(RULES.read_text())
    commit = text(args.cuda / 'cumlx_commit.txt') or 'HEAD'
    shown = subprocess.run(['git', '-C', str(REPO), 'show', f'{commit}:validation/m6-f1-rerun-limits.json'], capture_output=True, check=False)
    record = assess(args.cuda, args.mac, rules, shown.returncode == 0 and shown.stdout == RULES.read_bytes())
    # The machine text names the GPU and the driver; the host name never enters the record.
    record['gpu_at_start'] = '\n'.join(line for line in (record['gpu_at_start'] or '').splitlines() if 'Tesla' in line or 'Driver Version' in line) or None
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('complete', 'verdict', 'grade', 'not_complete_because')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
