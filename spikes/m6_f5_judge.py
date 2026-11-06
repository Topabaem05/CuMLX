#!/usr/bin/env python3
"""Judge F5 against CUDA, from the V100's output folder and the Mac's. The rules are in validation/m6-f5-cuda-limits.json, written before either ran.

    python spikes/m6_f5_judge.py --cuda ~/cumlx-m6-cuda/f5 --mac ~/cumlx-repro/result/f5 --output verdict.json
    Exit 0 pass (grade B), 1 fail, 5 a part has no output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx.render.compare import compare_images  # noqa: E402

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f5-cuda-limits.json').read_text())
RULE = LIMITS['depth_and_segmentation']
VIEWS = tuple(f'{c}/{p}' for c in ('front', 'top') for p in ('centre', 'left_low', 'far_raised'))


def read_exit(out: Path, name: str) -> int | None:
    try:
        return int((out / f'{name}.exit').read_text().strip())
    except (OSError, ValueError):
        return None


def read_npz(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError):
        return None


def part(reasons: list[str], missing: bool = False, **measured: Any) -> dict[str, Any]:
    return {'state': 'not_run' if missing else ('pass' if not reasons else 'fail'), 'reasons': reasons[:20], 'reason_count': len(reasons), **measured}


def geometry(depth: np.ndarray, label: np.ndarray, want_depth: np.ndarray, want_label: np.ndarray) -> dict[str, float]:
    same = (label == want_label) & (label >= 0)
    worst = float(np.abs(depth[same].astype(np.float64) - want_depth[same]).max()) if same.any() else math.inf
    return {'depth_worst_m': worst, 'label_mismatch': float(np.mean(label != want_label))}


def within(metrics: dict[str, float]) -> bool:
    return metrics['depth_worst_m'] <= RULE['depth_limit_m'] and metrics['label_mismatch'] <= RULE['label_mismatch_limit']


def images(record: dict[str, np.ndarray], view: str) -> dict[int, np.ndarray]:
    prefix = f'rgb/{view}/'
    return {int(key[len(prefix):]): value for key, value in record.items() if key.startswith(prefix)}


def judge(cuda: Path, mac: Path) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    steps = [(f'{side} {name}', read_exit(out, name)) for side, out, names in (('cuda', cuda, ('install', 'sensors')), ('mac', mac, ('install', 'sensors'))) for name in names]
    failed = [f'{label} exit {code}' for label, code in steps if code is not None and code != 0]
    absent = [f'{label} did not run' for label, code in steps if code is None]
    parts['setup'] = part(failed + absent, bool(absent) and not failed)
    c, m = read_npz(cuda / 'sensors' / 'sensors.npz'), read_npz(mac / 'sensors' / 'sensors.npz')
    if c is None or m is None:
        for name in ('rgb', 'geometry_cuda', 'geometry_mac', 'geometry_between', 'controls'):
            parts[name] = part([], True)
    else:
        rgb_reasons, rgb = [], {}
        for view in VIEWS:
            try:
                result = compare_images(images(c, view), images(m, view))
            except Exception as exc:  # noqa: BLE001 - a view that cannot be compared is a failure of this part
                rgb_reasons.append(f'{view}: {exc}')
                continue
            rgb[view] = {k: result[k] for k in ('verdict', 'worst_psnr_across_db', 'worst_psnr_within_db', 'mse_ratio', 'failures')}
            if result['verdict'] != 'ok':
                rgb_reasons.append(f'{view}: {result["verdict"]}: {"; ".join(result["failures"])[:200]}')
        parts['rgb'] = part(rgb_reasons, views=rgb)
        for side, record in (('geometry_cuda', c), ('geometry_mac', m)):
            reasons, measured = [], {}
            for view in VIEWS:
                if f'depth/{view}' not in record:
                    reasons.append(f'{view}: missing')
                    continue
                metrics = geometry(record[f'depth/{view}'], record[f'segmentation/{view}'], record[f'exact_depth/{view}'], record[f'exact_label/{view}'])
                measured[view] = metrics
                if not within(metrics):
                    reasons.append(f'{view}: depth {metrics["depth_worst_m"]:.3g} m, labels {metrics["label_mismatch"]:.2e}')
            parts[side] = part(reasons, measured=measured)
        reasons, measured = [], {}
        for view in VIEWS:
            if f'depth/{view}' not in c or f'depth/{view}' not in m:
                reasons.append(f'{view}: missing')
                continue
            metrics = geometry(m[f'depth/{view}'], m[f'segmentation/{view}'], c[f'depth/{view}'].astype(np.float64), c[f'segmentation/{view}'])
            measured[view] = metrics
            if not within(metrics):
                reasons.append(f'{view}: depth {metrics["depth_worst_m"]:.3g} m, labels {metrics["label_mismatch"]:.2e}')
        parts['geometry_between'] = part(reasons, measured=measured)
        control_reasons, caught = [], {}
        box = read_npz(mac / 'fault_box_5cm' / 'sensors.npz') if read_exit(mac, 'fault_box_5cm') == 0 else None
        if box is None:
            control_reasons.append('box_5cm: no output')
        else:
            failed = [v for v in VIEWS if not within(geometry(box[f'depth/{v}'], box[f'segmentation/{v}'], c[f'exact_depth/{v}'], c[f'exact_label/{v}']))]
            caught['box_5cm'] = {'caught': bool(failed), 'views': failed}
            if not failed:
                control_reasons.append('box_5cm was not caught')
        light = read_npz(mac / 'fault_light_2pct' / 'sensors.npz') if read_exit(mac, 'fault_light_2pct') == 0 else None
        if light is None:
            control_reasons.append('light_2pct: no output')
        else:
            failed = [v for v in VIEWS if compare_images(images(c, v), images(light, v))['verdict'] == 'diverged']
            caught['light_2pct'] = {'caught': bool(failed), 'views': failed}
            if not failed:
                control_reasons.append('light_2pct was not caught')
        parts['controls'] = part(control_reasons, caught=caught)
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('not_run' if 'not_run' in states and 'fail' not in states else 'fail')
    versions = {side: json.loads((out / 'sensors' / 'summary.json').read_text()).get('mitsuba') if (out / 'sensors' / 'summary.json').exists() else None for side, out in (('cuda', cuda), ('mac', mac))}
    return {'schema': 'cumlx.m6_f5_verdict.v1', 'rules': 'validation/m6-f5-cuda-limits.json', 'verdict': verdict, 'grade': 'B' if verdict == 'pass' else None, 'mitsuba': versions, 'parts': parts}


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
