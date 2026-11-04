#!/usr/bin/env python3
"""Judge one run of spikes/m6_vm_guest.sh from its output folder. The rules are in validation/m6-vm-limits.json, written before the first run.

    python spikes/m6_vm_judge.py OUT       # writes OUT/verdict.json. Exit 0 pass, 1 fail, 5 a part has no output.

A part with no output is 'not_run', which is not a pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-vm-limits.json').read_text())


def read_exit(out: Path, name: str) -> int | None:
    path = out / f'{name}.exit'
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pytest_counts(out: Path, name: str) -> dict[str, int]:
    """The counts in pytest's last line. Absent words count as 0."""
    try:
        lines = (out / f'{name}.txt').read_text().strip().splitlines()
    except OSError:
        return {}
    last = lines[-1] if lines else ''
    return {word: int(count) for count, word in re.findall(r'(\d+) (passed|failed|skipped|error)', last)}


def part(reasons: list[str], missing: bool) -> dict[str, Any]:
    state = 'not_run' if missing else ('pass' if not reasons else 'fail')
    return {'state': state, 'reasons': reasons}


def judge_base(out: Path) -> dict[str, Any]:
    doctor, selftest, tests = read_exit(out, 'doctor'), read_exit(out, 'selftest'), read_exit(out, 'tests')
    counts = pytest_counts(out, 'tests')
    reasons = []
    if doctor != 0:
        reasons.append(f'doctor exit {doctor}')
    if selftest != 0:
        reasons.append(f'selftest exit {selftest}')
    if tests != 0 or counts.get('passed', 0) < 100:
        reasons.append(f'tests exit {tests}, {counts}')
    return part(reasons, None in (doctor, selftest, tests))


def judge_f3(out: Path) -> dict[str, Any]:
    code = read_exit(out, 'f3')
    counts = pytest_counts(out, 'f3')
    reasons = []
    if code != 0 or counts.get('passed', 0) < 5:
        reasons.append(f'exit {code}, {counts}')
    if counts.get('skipped', 0):
        reasons.append(f'{counts["skipped"]} skipped, and a skip is not a pass')
    return part(reasons, code is None)


def judge_f4(out: Path) -> dict[str, Any]:
    rule = LIMITS['f4']
    train = read_exit(out, 'f4_train')
    reasons, missing = [], train is None
    if train not in (0, None):
        reasons.append(f'train exit {train}')
    for name, command in rule['runs'].items():
        try:
            run = json.loads((out / f'f4_{name}.json').read_text())
        except (OSError, ValueError):
            missing = True
            continue
        speed = run['mean_forward_speed']
        if command is not None and command != 0.0:
            if abs(speed - command) > rule['command_speed_tolerance']:
                reasons.append(f'{name} speed {speed:.3f}')
        elif abs(speed) > rule['near_zero_speed']:
            reasons.append(f'{name} speed {speed:.3f}, near 0 is needed')
        if name != 'zero' and run['falls_per_world'] > rule['max_falls_per_world']:
            reasons.append(f'{name} falls {run["falls_per_world"]:.3f}')
    return part(reasons, missing)


def judge_f5(out: Path) -> dict[str, Any]:
    code = read_exit(out, 'f5')
    try:
        status = json.loads((out / 'f5' / 'summary.json').read_text()).get('status')
    except (OSError, ValueError):
        return part([], True)
    reasons = [] if code == 0 and status == 'rendered' else [f'exit {code}, status {status}']
    return part(reasons, False)


def judge(out: Path) -> dict[str, Any]:
    parts = {'base': judge_base(out), 'f3': judge_f3(out), 'f4': judge_f4(out), 'f5': judge_f5(out)}
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('fail' if 'fail' in states else 'not_run')
    return {'schema': 'cumlx.m6_vm_verdict.v1', 'verdict': verdict, 'parts': parts, 'scope': LIMITS['scope'], 'does_not_cover': LIMITS['does_not_cover']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = judge(args.out)
    (args.out / 'verdict.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return {'pass': 0, 'fail': 1}.get(result['verdict'], 5)


if __name__ == '__main__':
    raise SystemExit(main())
