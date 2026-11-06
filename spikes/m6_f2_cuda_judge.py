#!/usr/bin/env python3
"""Judge F2 against gsplat on CUDA, from the V100's output folder and the Mac's. The rules are in validation/m6-f2-cuda-limits.json, written before either ran.

    python spikes/m6_f2_cuda_judge.py --cuda ~/cumlx-m6-cuda/f2 --mac ~/cumlx-repro/result/f2cuda --output verdict.json
    Exit 0 pass (grade B), 1 fail, 5 a part has no output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f2-cuda-limits.json').read_text())
PEAK_SCENES = ('one_gaussian', 'off_center_rotated', 'overlap_background', 'small_classic', 'small_antialiased', 'near_and_far')
HAND_SCENES = PEAK_SCENES + ('opaque_wall',)
ALL_SCENES = HAND_SCENES + ('learned_like_train_view', 'learned_like_heldout_view')
GRADIENT_SCENES = ('covering', 'learned_like')
PARAMETERS = ('means', 'quats', 'scales', 'opacities', 'colors')
SEEDS = (0, 1, 2, 3, 4)
RENDER_FAULTS = tuple(LIMITS['controls']['renders_must_fail'])
GRADIENT_FAULTS = tuple(LIMITS['controls']['gradients_must_fail'])


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


def read_npz(out: Path, name: str) -> dict[str, np.ndarray] | None:
    try:
        with np.load(out / f'{name}.npz') as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError):
        return None


def part(reasons: list[str], missing: bool = False, **measured: Any) -> dict[str, Any]:
    return {'state': 'not_run' if missing else ('pass' if not reasons else 'fail'), 'reasons': reasons, **measured}


def compare(image: np.ndarray, alpha: np.ndarray, want_image: np.ndarray, want_alpha: np.ndarray, background: list[float] | None) -> dict[str, float]:
    base = np.zeros(3) if background is None else np.asarray(background, dtype=np.float64)
    foreground = want_image - base
    peak = np.unravel_index(np.argmax(np.abs(foreground).sum(-1)), foreground.shape[:2])
    found = np.unravel_index(np.argmax(np.abs(image - base).sum(-1)), foreground.shape[:2])
    return {'image_rel_l2': float(np.linalg.norm(image - want_image) / max(np.linalg.norm(foreground), 1e-30)),
            'alpha_rel_l2': float(np.linalg.norm(alpha - want_alpha) / max(np.linalg.norm(want_alpha), 1e-30)),
            'peak_error_px': float(max(abs(int(peak[0]) - int(found[0])), abs(int(peak[1]) - int(found[1]))))}


def within(metrics: dict[str, float], rule: dict[str, Any], scene: str) -> list[str]:
    out = []
    # `not x <= limit`, so that a NaN fails: NaN > limit is False.
    if not metrics['image_rel_l2'] <= rule['image_rel_l2_at_most']:
        out.append(f'{scene}: image {metrics["image_rel_l2"]:.3g}')
    if not metrics['alpha_rel_l2'] <= rule['alpha_rel_l2_at_most']:
        out.append(f'{scene}: alpha {metrics["alpha_rel_l2"]:.3g}')
    if scene in PEAK_SCENES and not metrics['peak_error_px'] <= rule['peak_error_px_at_most']:
        out.append(f'{scene}: peak {metrics["peak_error_px"]:.0f} px')
    return out


def backgrounds(summary: dict[str, Any] | None) -> dict[str, Any]:
    return {name: item.get('background') for name, item in ((summary or {}).get('scenes') or {}).items()}


def between(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], background: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    rule = LIMITS['renders']['between_machines']
    reasons, measured = [], {}
    for scene in ALL_SCENES:
        keys = [f'image/{scene}', f'alpha/{scene}']
        if any(key not in candidate or key not in reference for key in keys):
            reasons.append(f'{scene}: missing')
            continue
        metrics = compare(candidate[keys[0]], candidate[keys[1]], reference[keys[0]], reference[keys[1]], background.get(scene))
        measured[scene] = metrics
        reasons += within(metrics, rule, scene)
    return reasons, measured


def against_hand(record: dict[str, np.ndarray], background: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    rule = LIMITS['renders']['against_the_hand_render']
    reasons, measured = [], {}
    for scene in HAND_SCENES:
        keys = [f'image/{scene}', f'alpha/{scene}', f'hand_image/{scene}', f'hand_alpha/{scene}']
        if any(key not in record for key in keys):
            reasons.append(f'{scene}: missing')
            continue
        metrics = compare(*(record[key] for key in keys), background.get(scene))
        measured[scene] = metrics
        reasons += within(metrics, rule, scene)
    return reasons, measured


def gradients(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[list[str], dict[str, float]]:
    limit = LIMITS['gradients']['rel_l2_at_most']
    reasons, measured = [], {}
    for scene in GRADIENT_SCENES:
        for name in PARAMETERS:
            key = f'grad/{scene}/{name}'
            if key not in candidate or key not in reference:
                reasons.append(f'{key}: missing')
                continue
            rel = float(np.linalg.norm(candidate[key] - reference[key]) / max(np.linalg.norm(reference[key]), 1e-30))
            measured[f'{scene}.{name}'] = rel
            if not rel <= limit:
                reasons.append(f'{scene}.{name}: {rel:.3g}')
    return reasons, measured


def training_runs(out: Path, mode: str) -> dict[int, dict[str, Any] | None]:
    return {seed: (read_json(out, f'train_{mode}_{seed}') if read_exit(out, f'train_{mode}_{seed}') == 0 else None) for seed in SEEDS}


def each_machine(out: Path) -> dict[str, Any]:
    right, frozen, wrong = (training_runs(out, mode) for mode in ('right', 'frozen', 'wrong_cameras'))
    reasons = []
    for seed in SEEDS:
        r, f, w = right[seed], frozen[seed], wrong[seed]
        if r is None or f is None or w is None:
            reasons.append(f'seed {seed}: a mode has no result')
            continue
        if not r['heldout_psnr_end'] >= 25:
            reasons.append(f'seed {seed}: right {r["heldout_psnr_end"]:.2f} dB')
        if not r['heldout_psnr_end'] - r['heldout_psnr_start'] >= 3:
            reasons.append(f'seed {seed}: right rose {r["heldout_psnr_end"] - r["heldout_psnr_start"]:.2f} dB')
        if not f['heldout_psnr_end'] < 25:
            reasons.append(f'seed {seed}: frozen {f["heldout_psnr_end"]:.2f} dB')
        if not r['heldout_psnr_end'] - w['heldout_psnr_end'] >= 3:
            reasons.append(f'seed {seed}: wrong cameras only {r["heldout_psnr_end"] - w["heldout_psnr_end"]:.2f} dB below')
    devices = sorted({item['device'] for item in right.values() if item})
    return part(reasons, all(item is None for item in right.values()), devices=devices,
                right=[None if right[s] is None else right[s]['heldout_psnr_end'] for s in SEEDS],
                frozen=[None if frozen[s] is None else frozen[s]['heldout_psnr_end'] for s in SEEDS],
                wrong_cameras=[None if wrong[s] is None else wrong[s]['heldout_psnr_end'] for s in SEEDS])


def mean_and_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def equivalent(candidate: list[float | None], reference: list[float | None]) -> dict[str, Any]:
    if any(v is None for v in candidate) or any(v is None for v in reference):
        return {'measured': False}
    m_c, se_c = mean_and_se([float(v) for v in candidate])
    m_r, se_r = mean_and_se([float(v) for v in reference])
    tolerance = max(3 * math.sqrt(se_c ** 2 + se_r ** 2), 0.01 * m_r)
    return {'measured': True, 'mean_candidate': m_c, 'mean_reference': m_r, 'se_candidate': se_c, 'se_reference': se_r, 'difference': m_c - m_r, 'tolerance': tolerance, 'within': abs(m_c - m_r) <= tolerance}


def judge(cuda: Path, mac: Path) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    setup, absent = [], []
    for side, out, names in (('cuda', cuda, ('install', 'build', 'record')), ('mac', mac, ('install', 'record'))):
        for name in names:
            code = read_exit(out, name)
            if code is None:
                absent.append(f'{side} {name} did not run')
            elif code != 0:
                setup.append(f'{side} {name} exit {code}')
    parts['setup'] = part(setup + absent, bool(absent) and not setup)
    cuda_record, mac_record = read_npz(cuda, 'record'), read_npz(mac, 'record')
    background = backgrounds(read_json(cuda, 'record'))
    if cuda_record is None or mac_record is None:
        for name in ('renders_between', 'hand_cuda', 'hand_mac', 'gradients', 'controls'):
            parts[name] = part([], True)
    else:
        reasons, measured = between(mac_record, cuda_record, background)
        parts['renders_between'] = part(reasons, measured=measured)
        for side, record in (('hand_cuda', cuda_record), ('hand_mac', mac_record)):
            reasons, measured = against_hand(record, background)
            parts[side] = part(reasons, measured=measured)
        reasons, measured = gradients(mac_record, cuda_record)
        parts['gradients'] = part(reasons, measured=measured)
        control_reasons, caught = [], {}
        for fault in RENDER_FAULTS + GRADIENT_FAULTS:
            record = read_npz(mac, f'fault_{fault}') if read_exit(mac, f'fault_{fault}') == 0 else None
            if record is None:
                control_reasons.append(f'{fault}: no output')
                continue
            failed = (between(record, cuda_record, background)[0] if fault in RENDER_FAULTS else gradients(record, cuda_record)[0])
            caught[fault] = {'caught': bool(failed), 'by': failed[:3]}
            if not failed:
                control_reasons.append(f'{fault} was not caught')
        parts['controls'] = part(control_reasons, caught=caught)
    program, code = read_json(cuda, 'program'), read_exit(cuda, 'program')
    if code != 0 or program is None:
        parts['original_program'] = part([f'exit {code}'])
    else:
        parts['original_program'] = part([] if program['last_loss'] < program['first_loss'] else ['the loss did not fall'], first_loss=program['first_loss'], last_loss=program['last_loss'], device=program['device'])
    parts['training_cuda'] = each_machine(cuda)
    parts['training_mac'] = each_machine(mac)
    for side, expected in (('training_cuda', 'cuda'), ('training_mac', 'mps')):
        if parts[side]['state'] != 'not_run' and parts[side]['devices'] != [expected]:
            parts[side]['reasons'].append(f'devices {parts[side]["devices"]}')
            parts[side]['state'] = 'fail'
    right_cuda = parts['training_cuda'].get('right', [None] * len(SEEDS))
    test = equivalent(parts['training_mac'].get('right', [None] * len(SEEDS)), right_cuda)
    parts['equivalence'] = part([] if test.get('within') else ['not measured' if not test['measured'] else f'difference {test["difference"]:.3f} dB, tolerance {test["tolerance"]:.3f} dB'], not test['measured'], **test)
    zero = [None if r is None else r['heldout_psnr_end'] for r in training_runs(mac, 'colors_gradient_zero').values()]
    must_fail = {'wrong_cameras': equivalent(parts['training_mac'].get('wrong_cameras', [None] * len(SEEDS)), right_cuda), 'colors_gradient_zero': equivalent(zero, right_cuda)}
    parts['equivalence_controls'] = part([f'{name} was not caught' if item['measured'] else f'{name} has no result' for name, item in must_fail.items() if not item['measured'] or item['within']], **must_fail)
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('not_run' if 'not_run' in states and 'fail' not in states else 'fail')
    return {'schema': 'cumlx.m6_f2_cuda_verdict.v1', 'rules': 'validation/m6-f2-cuda-limits.json', 'verdict': verdict, 'grade': 'B' if verdict == 'pass' else None,
            'cuda_versions': read_json(cuda, 'versions'), 'cuda_commit': _text(cuda / 'cumlx_commit.txt'), 'mac_commit': _text(mac / 'cumlx_commit.txt') or _text(mac.parent / 'cumlx_commit.txt'), 'parts': parts}


def _text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', required=True, type=Path)
    parser.add_argument('--mac', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = judge(args.cuda, args.mac)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({'verdict': result['verdict'], 'parts': {k: v['state'] for k, v in result['parts'].items()}}))
    return {'pass': 0, 'fail': 1}.get(result['verdict'], 5)


if __name__ == '__main__':
    raise SystemExit(main())
