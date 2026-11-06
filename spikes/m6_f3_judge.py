#!/usr/bin/env python3
"""Judge F3 against CUDA, from the V100's output folder and the Mac's. The rules are in validation/m6-f3-limits.json, written before either ran.

    python spikes/m6_f3_judge.py --cuda ~/cumlx-m6-cuda/f3 --mac ~/cumlx-repro/result/f3 --output verdict.json
    Exit 0 pass (grade A), 1 fail, 5 a part has no output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f3-limits.json').read_text())
SIZES = ('32x32', '64x64', '128x96', '256x256')
KINDS = ('dense', 'sparse', 'blobs', 'spiral', 'empty', 'full')
FAULTS = tuple(LIMITS['controls']['must_fail'])


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


def partition(labels: np.ndarray) -> np.ndarray:
    """Each pixel's label replaced by the first flat index that has that label, so that two labelings of one partition are equal."""
    flat = labels.reshape(-1)
    first = {}
    for index, value in enumerate(flat.tolist()):
        if value and value not in first:
            first[value] = index + 1
    return np.array([first.get(v, 0) for v in flat.tolist()], dtype=np.int64).reshape(labels.shape)


def masks_of(record: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    out = []
    for size in SIZES:
        labels, counts = record.get(f'labels/{size}'), record.get(f'counts/{size}')
        if labels is None or counts is None:
            continue
        for index, kind in enumerate(KINDS):
            out.append((size, kind, labels[index, 0], counts[index, 0]))
    return out


def exact(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> list[str]:
    reasons = []
    found = {(s, k): (lab, cnt) for s, k, lab, cnt in masks_of(candidate)}
    for size, kind, labels, counts in masks_of(reference):
        if (size, kind) not in found:
            reasons.append(f'{size} {kind}: missing')
            continue
        lab, cnt = found[(size, kind)]
        if lab.shape != labels.shape or not np.array_equal(lab, labels):
            reasons.append(f'{size} {kind}: {int(np.sum(lab != labels)) if lab.shape == labels.shape else "shape"} labels differ')
        if cnt.shape != counts.shape or not np.array_equal(cnt, counts):
            reasons.append(f'{size} {kind}: {int(np.sum(cnt != counts)) if cnt.shape == counts.shape else "shape"} counts differ')
    if len(masks_of(reference)) != len(SIZES) * len(KINDS):
        reasons.append(f'{len(masks_of(reference))} masks in the reference, not {len(SIZES) * len(KINDS)}')
    return reasons


def same_partition(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> list[str]:
    reasons = []
    found = {(s, k): (lab, cnt) for s, k, lab, cnt in masks_of(candidate)}
    for size, kind, labels, counts in masks_of(reference):
        if (size, kind) not in found:
            reasons.append(f'{size} {kind}: missing')
            continue
        lab, cnt = found[(size, kind)]
        if lab.shape != labels.shape or not np.array_equal(partition(lab), partition(labels)):
            reasons.append(f'{size} {kind}: another partition')
        if not np.array_equal(cnt, counts):
            reasons.append(f'{size} {kind}: other counts')
    return reasons


def judge(cuda: Path, mac: Path) -> dict[str, Any]:
    parts: dict[str, Any] = {}
    steps = [(f'{side} {name}', read_exit(out, name)) for side, out, names in (('cuda', cuda, ('fetch', 'cuda')), ('mac', mac, ('install', 'fetch', 'mac', 'reference')))
             for name in names]
    failed = [f'{label} exit {code}' for label, code in steps if code is not None and code != 0]
    absent = [f'{label} did not run' for label, code in steps if code is None]
    parts['setup'] = part(failed + absent, bool(absent) and not failed)
    translated = read_exit(mac, 'translate') == 0 and (mac / 'connected_components.metal').exists() and (mac / 'connected_components.port.json').exists()
    parts['translate'] = part([] if translated else [f'cumlx kernel port exit {read_exit(mac, "translate")}'])
    cuda_record, mac_record, reference = read_npz(cuda / 'cuda.npz'), read_npz(mac / 'mac.npz'), read_npz(mac / 'reference.npz')
    if cuda_record is None or mac_record is None:
        parts['cuda_vs_mac'] = part([], True)
    else:
        reasons = exact(mac_record, cuda_record)
        parts['cuda_vs_mac'] = part(reasons, masks=len(masks_of(cuda_record)))
    if reference is None or cuda_record is None or mac_record is None:
        parts['reference'] = part([], True)
    else:
        parts['reference'] = part([f'cuda: {r}' for r in same_partition(cuda_record, reference)] + [f'mac: {r}' for r in same_partition(mac_record, reference)],
                                  labels_equal_to_the_bit=not exact(cuda_record, reference))
    control_reasons, caught = [], {}
    if cuda_record is None:
        parts['controls'] = part([], True)
    for fault in FAULTS if cuda_record is not None else ():
        record = read_npz(mac / f'fault_{fault}.npz') if read_exit(mac, f'fault_{fault}') == 0 else None
        if record is None or cuda_record is None:
            control_reasons.append(f'{fault}: no output')
            continue
        failed = exact(record, cuda_record)
        caught[fault] = {'caught': bool(failed), 'by': failed[:3]}
        if not failed:
            control_reasons.append(f'{fault} was not caught')
    if cuda_record is not None:
        parts['controls'] = part(control_reasons, caught=caught)
    states = [item['state'] for item in parts.values()]
    verdict = 'pass' if all(s == 'pass' for s in states) else ('not_run' if 'not_run' in states and 'fail' not in states else 'fail')
    return {'schema': 'cumlx.m6_f3_verdict.v1', 'rules': 'validation/m6-f3-limits.json', 'verdict': verdict, 'grade': 'A' if verdict == 'pass' else None, 'parts': parts}


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
