#!/usr/bin/env python3
"""Judge one F1 and F2 run from its output folder. The rules are in validation/m6-f1-f2-limits.json, written before the first run.

    python spikes/m6_f1_f2_judge.py OUT    # writes OUT/verdict.json. Exit 0 pass, 1 fail, 5 a part has no output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f1-f2-limits.json').read_text())


def read_exit(out: Path, name: str) -> int | None:
    try:
        return int((out / f'{name}.exit').read_text().strip())
    except (OSError, ValueError):
        return None


def read_json(out: Path, name: str) -> dict[str, Any] | None:
    try:
        return json.loads((out / f'{name}.json').read_text())
    except (OSError, ValueError):
        return None


def part(reasons: list[str], missing: bool) -> dict[str, Any]:
    return {'state': 'not_run' if missing else ('pass' if not reasons else 'fail'), 'reasons': reasons}


def f1_mac_port(out: Path) -> dict[str, Any]:
    port, port8, bad = read_exit(out, 'f1_port'), read_exit(out, 'f1_port8'), read_exit(out, 'f1_bad')
    reasons = []
    if port != 0:
        reasons.append(f'port exit {port}')
    if port8 != 0:
        reasons.append(f'8-bit port exit {port8}')
    status = (read_json(out, 'f1_bad') or {}).get('status')
    if bad != 1 or status != 'diverged':
        reasons.append(f'a wrong port gave exit {bad} and status {status}, and exit 1 with status diverged is needed')
    return part(reasons, None in (port, port8, bad))


def f1_original(out: Path) -> dict[str, Any]:
    install, run = read_exit(out, 'f1_program_install'), read_exit(out, 'f1_program')
    if install is None:
        return part([], True)
    if install != 0:
        return part([f'install exit {install}'], False)
    reasons = [] if run == 0 else [f'exit {run}']
    result = read_json(out, 'f1_program') or {}
    if run == 0 and result.get('generated_tokens') != 20:
        reasons.append(f'{result.get("generated_tokens")} tokens, and 20 are needed')
    return part(reasons, run is None)


def f2_mac_training(out: Path) -> dict[str, Any]:
    rule = LIMITS['f2']['mac_training']
    install = read_exit(out, 'f2_install')
    if install is None:
        return part([], True)
    if install != 0:
        return part([f'install exit {install}'], False)
    right, frozen, wrong = read_json(out, 'f2_right'), read_json(out, 'f2_frozen'), read_json(out, 'f2_wrong_cameras')
    if None in (right, frozen, wrong):
        return part([], True)
    reasons = []
    end, start = right['heldout_psnr_end'], right['heldout_psnr_start']
    if end < rule['heldout_psnr_at_least_db']:
        reasons.append(f'held-out PSNR {end:.2f} dB')
    if end - start < rule['heldout_psnr_rise_at_least_db']:
        reasons.append(f'PSNR rose {end - start:.2f} dB')
    if right['device'] != 'mps':
        reasons.append(f'device {right["device"]}')
    if frozen['heldout_psnr_end'] >= rule['frozen_control_below_db']:
        reasons.append(f'the frozen control reached {frozen["heldout_psnr_end"]:.2f} dB')
    if end - wrong['heldout_psnr_end'] < rule['wrong_cameras_at_least_db_below_right']:
        reasons.append(f'the wrong cameras reached {wrong["heldout_psnr_end"]:.2f} dB, against {end:.2f} dB')
    return part(reasons, False)


def f2_original(out: Path) -> dict[str, Any]:
    install, run = read_exit(out, 'f2_install'), read_exit(out, 'f2_program')
    if install is None:
        return part([], True)
    if install != 0:
        return part([f'install exit {install}'], False)
    reasons = [] if run == 0 else [f'exit {run}']
    result = read_json(out, 'f2_program') or {}
    if run == 0 and not result.get('last_loss', float('inf')) < result.get('first_loss', 0.0):
        reasons.append('the loss did not fall')
    return part(reasons, run is None)


def judge(out: Path) -> dict[str, Any]:
    parts = {'f1_original_program': f1_original(out), 'f1_mac_port': f1_mac_port(out), 'f2_original_program': f2_original(out), 'f2_mac_training': f2_mac_training(out)}
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('fail' if 'fail' in states else 'not_run')
    return {'schema': 'cumlx.m6_f1_f2_verdict.v1', 'verdict': verdict, 'parts': parts, 'scope': LIMITS['scope']}


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
