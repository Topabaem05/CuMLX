#!/usr/bin/env python3
"""Judge F1 against CUDA, from the V100's output folder and the Mac's. The rules are in validation/m6-f1-limits.json, written before either ran.

    python spikes/m6_f1_judge.py --cuda ~/cumlx-m6-cuda/f1 --mac ~/cumlx-repro/result --output verdict.json
    Exit 0 pass (grade A), 1 fail, 5 a part has no output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f1-limits.json').read_text())
PARTS = LIMITS['parts']
FAULTS = tuple(LIMITS['controls']['must_fail'])
NEW_TOKENS = 20
LEAVES, LAYERS, PREFILL_CALLS, DECODE_CALLS = 168, 24, 24, 72
SUBSTITUTES = {'flash_attn': 'cumlx.compat.flash_attn_sdpa', 'bitsandbytes': 'cumlx.compat.bitsandbytes_int8'}
DISTRIBUTIONS = {'flash-attn': '2.8.3+cumlx', 'bitsandbytes': '0.50.2+cumlx'}


def read_exit(out: Path, name: str) -> int | None:
    try:
        return int((out / f'{name}.exit').read_text().strip())
    except (OSError, ValueError):
        return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def part(reasons: list[str], missing: bool = False, **measured: Any) -> dict[str, Any]:
    return {'state': 'not_run' if missing else ('pass' if not reasons else 'fail'), 'reasons': reasons[:20], 'reason_count': len(reasons), **measured}


def bad(value: Any, limit: float) -> bool:
    return not (isinstance(value, (int, float)) and math.isfinite(value) and value <= limit)


def program(out: Path, side: str) -> tuple[dict[str, Any], list[int] | None]:
    code, result = read_exit(out, 'program'), read_json(out / 'program.json')
    if code is None:
        return part([], True), None
    reasons = [] if code == 0 else [f'exit {code}']
    tokens = (result or {}).get('token_ids')
    if not tokens or len(tokens) != NEW_TOKENS:
        reasons.append(f'{0 if not tokens else len(tokens)} tokens')
    if side == 'mac':
        report = read_json(out / 'program.run-report.json') or {}
        calls = report.get('calls') or []
        for name, module in SUBSTITUTES.items():
            if not any(c['kind'] == 'substituted' and c['name'] == name and c['to'] == module for c in calls):
                reasons.append(f'the run report does not list {name} -> {module}')
        for name, version in DISTRIBUTIONS.items():
            if not any(c['kind'] == 'distribution' and c['name'] == name and version in (c['to'] or '') for c in calls):
                reasons.append(f'the run report does not list the distribution {name} {version}')
    return part(reasons, text=(result or {}).get('text'), device=(result or {}).get('device')), tokens


def leaves(replay: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    limit = PARTS['leaves']['rel_l2_at_most']
    names = golden.get('leaves') or []
    reasons = [] if len(names) == LEAVES else [f'the golden has {len(names)} Linear8bitLt, not {LEAVES}']
    found = replay.get('leaves') or {}
    worst = 0.0
    for name in names:
        item = found.get(name)
        if item is None:
            reasons.append(f'{name}: not replayed')
            continue
        if bad(item['rel_l2'], limit) or not item.get('finite', True):
            reasons.append(f'{name}: {item["rel_l2"]:.3g}')
        worst = max(worst, item['rel_l2'] if math.isfinite(item['rel_l2']) else math.inf)
    return part(reasons, worst_rel_l2=worst, least_equal_fraction=min((v['equal_fraction'] for v in found.values()), default=None))


def weights(replay: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    rule = PARTS['weights']
    found = replay.get('weights') or {}
    reasons = []
    for name in golden.get('leaves') or []:
        item = found.get(name)
        if item is None:
            reasons.append(f'{name}: not compared')
            continue
        if not item['shape_equal']:
            reasons.append(f'{name}: shape')
            continue
        if not item['scb_equal']:
            reasons.append(f'{name}: SCB differs')
        if item['cb_max_step'] is None or item['cb_max_step'] > rule['cb_step_at_most']:
            reasons.append(f'{name}: a weight {item["cb_max_step"]} steps away')
        if item['cb_differ_fraction'] > rule['cb_differ_fraction_at_most']:
            reasons.append(f'{name}: {item["cb_differ_fraction"]:.2e} of the weights differ')
    return part(reasons, most_differ_fraction=max((v['cb_differ_fraction'] for v in found.values()), default=None))


def layers(replay: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    limit = PARTS['layers']['rel_l2_at_most']
    found = replay.get('layers') or {}
    names = list(golden.get('layers') or []) + ['norm', 'lm_head']
    reasons = [] if len(golden.get('layers') or []) == LAYERS else [f'the golden has {len(golden.get("layers") or [])} decoder layers, not {LAYERS}']
    for name in names:
        item = found.get(name)
        if item is None:
            reasons.append(f'{name}: not replayed')
        elif bad(item['rel_l2'], limit) or not item.get('finite', True):
            reasons.append(f'{name}: {item["rel_l2"]:.3g}')
    return part(reasons, worst_rel_l2=max((v['rel_l2'] for v in found.values()), default=None))


def attention(replay: dict[str, Any], golden: dict[str, Any]) -> dict[str, Any]:
    limit = PARTS['attention']['rel_l2_at_most']
    calls = golden.get('attention') or []
    reasons = []
    phases = [c['phase'] for c in calls]
    if phases.count('prefill') != PREFILL_CALLS or phases.count('decode') != DECODE_CALLS:
        reasons.append(f'the golden has {phases.count("prefill")} prefill and {phases.count("decode")} decode calls, not {PREFILL_CALLS} and {DECODE_CALLS}')
    if golden.get('varlen_calls'):
        reasons.append(f'{golden["varlen_calls"]} varlen calls, which this test does not record')
    found = replay.get('attention') or {}
    for call in calls:
        item = found.get(call['key'])
        if item is None:
            reasons.append(f'{call["key"]}: not replayed')
            continue
        for side in ('against_exact', 'golden_against_exact'):
            if bad(item[side]['rel_l2'], limit) or not item[side].get('finite', True):
                reasons.append(f'{call["key"]} ({call["phase"]}, layer {call["layer"]}) {side}: {item[side]["rel_l2"]:.3g}')
    return part(reasons, worst_mac=max((v['against_exact']['rel_l2'] for v in found.values()), default=None),
                worst_cuda=max((v['golden_against_exact']['rel_l2'] for v in found.values()), default=None))


JUDGES = {'leaves': leaves, 'weights': weights, 'layers': layers, 'attention': attention}


def judge(cuda: Path, mac: Path) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    steps = [(f'{side} {name}', read_exit(out, name)) for side, out, names in (('cuda', cuda, ('install', 'capture')), ('mac', mac, ('install', 'clean', 'replay')))
             for name in names]
    failed = [f'{label} exit {code}' for label, code in steps if code is not None and code != 0]
    absent = [f'{label} did not run' for label, code in steps if code is None]
    parts['setup'] = part(failed + absent, bool(absent) and not failed)
    parts['program_cuda'], cuda_tokens = program(cuda, 'cuda')
    parts['program_mac'], mac_tokens = program(mac, 'mac')
    if cuda_tokens and mac_tokens:
        parts['first_token'] = part([] if cuda_tokens[0] == mac_tokens[0] else [f'{mac_tokens[0]} on the Mac, {cuda_tokens[0]} on CUDA'],
                                    equal_tokens=sum(a == b for a, b in zip(cuda_tokens, mac_tokens, strict=False)), of=NEW_TOKENS)
    else:
        parts['first_token'] = part([], True)
    golden = read_json(cuda / 'golden' / 'summary.json')
    replay = read_json(mac / 'replay.json')
    if golden is None or replay is None:
        for name in JUDGES:
            parts[name] = part([], True)
    else:
        if not replay.get('input_ids_equal'):
            parts['setup']['reasons'].append('the Mac tokenized the prompt to other ids')
            parts['setup']['state'] = 'fail'
        for name, run in JUDGES.items():
            parts[name] = run(replay, golden)
    control_reasons, caught = [], {}
    for fault, where in LIMITS['controls']['must_fail'].items():
        result = read_json(mac / f'fault_{fault}.json') if read_exit(mac, f'fault_{fault}') == 0 else None
        if result is None or golden is None:
            control_reasons.append(f'{fault}: no output')
            continue
        verdict = JUDGES[where](result, golden)
        caught[fault] = {'by': where, 'caught': verdict['state'] == 'fail', 'reasons': verdict['reasons'][:3]}
        if verdict['state'] != 'fail':
            control_reasons.append(f'{fault} was not caught by {where}')
    parts['controls'] = part(control_reasons, caught=caught)
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('not_run' if 'not_run' in states and 'fail' not in states else 'fail')
    reported = {'prefill_chain': (replay or {}).get('prefill_chain'), 'cuda_text': (read_json(cuda / 'program.json') or {}).get('text'),
                'mac_text': (read_json(mac / 'program.json') or {}).get('text'), 'cuda_bitsandbytes': (golden or {}).get('bitsandbytes'), 'cuda_torch': (golden or {}).get('torch')}
    return {'schema': 'cumlx.m6_f1_verdict.v1', 'rules': 'validation/m6-f1-limits.json', 'verdict': verdict, 'grade': 'A' if verdict == 'pass' else None, 'parts': parts, 'reported_not_judged': reported}


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
