#!/usr/bin/env python3
"""Judge one gsplat adapter run from its output folder. The rules are in validation/m6-f2-adapter-limits.json, written before the adapter first ran.

    python spikes/m6_f2_adapter_judge.py OUT    # writes OUT/verdict.json. Exit 0 pass, 1 fail, 5 a part has no output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f2-adapter-limits.json').read_text())
GRADIENT_NAMES = {'means', 'quats', 'scales', 'opacities', 'colors'}


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


def part(reasons: list[str], missing: bool = False) -> dict[str, Any]:
    return {'state': 'not_run' if missing else ('pass' if not reasons else 'fail'), 'reasons': reasons}


def conformance(result: dict[str, Any]) -> dict[str, Any]:
    rule = LIMITS['conformance']
    reasons = []
    scenes = result.get('conformance') or {}
    if not scenes:
        return part([], True)
    for name, scene in scenes.items():
        if scene['image_rel_l2'] > rule['image_rel_l2_at_most']:
            reasons.append(f'{name}: image {scene["image_rel_l2"]:.4f}')
        if scene['alpha_rel_l2'] > rule['alpha_rel_l2_at_most']:
            reasons.append(f'{name}: alpha {scene["alpha_rel_l2"]:.4f}')
        if scene['peak_error_px'] > rule['peak_error_px_at_most']:
            reasons.append(f'{name}: peak {scene["peak_error_px"]} px')
        if scene['device'] != rule['device']:
            reasons.append(f'{name}: device {scene["device"]}')
    return part(reasons)


def controls(result: dict[str, Any]) -> dict[str, Any]:
    found = result.get('controls') or {}
    if not found:
        return part([], True)
    return part([f'{name} was not caught' for name in LIMITS['controls']['must_fail'] if not (found.get(name) or {}).get('caught')])


def gradients(result: dict[str, Any]) -> dict[str, Any]:
    found = result.get('gradients') or {}
    if not found:
        return part([], True)
    limit = LIMITS['gradients']['relative_error_at_most']
    reasons = [f'{name}: {item["relative_error"]:.4f}' for name, item in found.items() if item['relative_error'] > limit]
    have = {name.split('.')[1] for name in found if name.startswith('color_loss.')}
    reasons += [f'{name} was not measured' for name in sorted(GRADIENT_NAMES - have)]
    return part(reasons)


def refusals(result: dict[str, Any]) -> dict[str, Any]:
    found = result.get('refusals') or {}
    if not found:
        return part([], True)
    return part([f'{name} did not raise' for name in LIMITS['refusals']['must_raise'] if not str(found.get(name, '')).startswith('raised')])


def original_program(out: Path, install: int) -> dict[str, Any]:
    if install != 0:
        return part([f'install exit {install}'])
    run = read_exit(out, 'program')
    if run is None:
        return part([], True)
    reasons = [] if run == 0 else [f'exit {run}']
    result = read_json(out, 'program') or {}
    if run == 0 and not result.get('last_loss', float('inf')) < result.get('first_loss', 0.0):
        reasons.append('the loss did not fall')
    return part(reasons)


def training(out: Path, install: int) -> dict[str, Any]:
    rule = LIMITS['training']
    if install != 0:
        return part([f'install exit {install}'])
    right, frozen, wrong = read_json(out, 'train_right'), read_json(out, 'train_frozen'), read_json(out, 'train_wrong_cameras')
    if None in (right, frozen, wrong):
        return part([], True)
    reasons = []
    end, start = right['heldout_psnr_end'], right['heldout_psnr_start']
    if end < rule['heldout_psnr_at_least_db']:
        reasons.append(f'held-out PSNR {end:.2f} dB')
    if end - start < rule['heldout_psnr_rise_at_least_db']:
        reasons.append(f'PSNR rose {end - start:.2f} dB')
    if right['device'] != LIMITS['conformance']['device']:
        reasons.append(f'device {right["device"]}')
    if frozen['heldout_psnr_end'] >= rule['frozen_control_below_db']:
        reasons.append(f'the frozen control reached {frozen["heldout_psnr_end"]:.2f} dB')
    if end - wrong['heldout_psnr_end'] < rule['wrong_cameras_at_least_db_below_right']:
        reasons.append(f'the wrong cameras reached {wrong["heldout_psnr_end"]:.2f} dB, against {end:.2f} dB')
    return part(reasons)


def judge(out: Path) -> dict[str, Any]:
    install = read_exit(out, 'install')
    if install is None:
        parts = {name: part([], True) for name in ('conformance', 'controls', 'gradients', 'refusals', 'original_program', 'training')}
    else:
        result = read_json(out, 'conformance') or {}
        if install != 0:
            missing = part([f'install exit {install}'])
            parts = {'conformance': missing, 'controls': missing, 'gradients': missing, 'refusals': missing}
        else:
            parts = {'conformance': conformance(result), 'controls': controls(result), 'gradients': gradients(result), 'refusals': refusals(result)}
        parts |= {'original_program': original_program(out, install), 'training': training(out, install)}
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('fail' if 'fail' in states else 'not_run')
    return {'schema': 'cumlx.m6_f2_adapter_verdict.v1', 'verdict': verdict, 'parts': parts, 'scope': LIMITS['scope']}


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
