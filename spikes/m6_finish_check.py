#!/usr/bin/env python3
"""Recompute the key numbers of the M6 finish verdicts from the raw files, with code of its own: it imports no judge. Prints JSON.

    python spikes/m6_finish_check.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

H = Path.home(); C = H / 'cumlx-m6-cuda'; M = H / 'cumlx-repro/result-20261106-101702'
def load(path):
    return json.loads(Path(path).read_text())


def rel(a, b, base=0.0):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b - base))
out = {}
# whole run
out['commits'] = {str(p.relative_to(H)): p.read_text().strip()[:7] for p in [C / 'cumlx_commit.txt', M / 'cumlx_commit.txt', *sorted(C.glob('f*/cumlx_commit.txt'))]}
out['gpu_at_start'] = [l for l in (C / 'machine_start.txt').read_text().splitlines() if 'pid' in l or 'MiB' in l][-2:]
out['mac_peak_gb'] = (M / 'peak_used_gb.txt').read_text().strip()
out['base_exits'] = {n: (M / 'base' / f'{n}.exit').read_text().strip() for n in ('doctor', 'selftest', 'tests')}
out['base_tests_tail'] = (M / 'base' / 'tests.txt').read_text().strip().splitlines()[-1]
# F2
c, m = np.load(C / 'f2/record.npz'), np.load(M / 'f2cuda/record.npz')
bgs = {k: v['background'] for k, v in load(C / 'f2/record.json')['scenes'].items()}
worst = 0.0
for s, bg in bgs.items():
    base = np.zeros(3) if bg is None else np.asarray(bg)
    worst = max(worst, rel(m[f'image/{s}'], c[f'image/{s}'], base), rel(m[f'alpha/{s}'], c[f'alpha/{s}']))
gw = max(rel(m[k], c[k]) for k in c.files if k.startswith('grad/'))
right = lambda d: [load(d / f'train_right_{i}.json')['heldout_psnr_end'] for i in range(5)]
rc, rm = right(C / 'f2'), right(M / 'f2cuda')
se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
dev = {load(C / f'f2/train_right_{i}.json')['device'] for i in range(5)} | {load(M / f'f2cuda/train_right_{i}.json')['device'] for i in range(5)}
out['f2'] = {'renders_worst_rel_l2': worst, 'gradients_worst_rel_l2': gw, 'psnr_mean_cuda': float(np.mean(rc)), 'psnr_mean_mac': float(np.mean(rm)),
             'tolerance': float(max(3 * np.hypot(se(rc), se(rm)), 0.01 * np.mean(rc))), 'devices': sorted(dev)}
# F3
c3, m3, r3 = np.load(C / 'f3/cuda.npz'), np.load(M / 'f3/mac.npz'), np.load(M / 'f3/reference.npz')
same = all(np.array_equal(c3[k], m3[k]) and np.array_equal(c3[k], r3[k]) for k in c3.files)
comps = {k: int(len(set(c3[k].reshape(-1).tolist()) - {0})) for k in c3.files if k.startswith('labels/')}
faults = {f.stem: any(not np.array_equal(np.load(f)[k], c3[k]) for k in c3.files) for f in (M / 'f3').glob('fault_*.npz')}
out['f3'] = {'cuda_mac_reference_all_equal': same, 'masks': sum(np.load(C / 'f3/cuda.npz')[k].shape[0] for k in c3.files if k.startswith('labels/')), 'components_per_batch': comps, 'faults_differ': faults,
             'mac_engine': load(M / 'f3/mac.json').get('engine', {}).get('engine'), 'cuda_source_sha256': load(C / 'f3/cuda.json')['source_sha256'][:12]}
# F4
f4 = M / 'f4/compare'
def spread(a, b):
    w = np.abs(np.load(a)['values'] - np.load(b)['values']).max(axis=(0, 2))
    return round(float(np.median(w)), 4), round(float(np.quantile(w, 0.9)), 4)
out['f4'] = {s: {'genesis_vs_mujoco': spread(f4 / f'mujoco_{s}.npz', f4 / f'genesis_{s}.npz'), 'fault_kp': spread(f4 / f'mujoco_{s}.npz', f4 / f'fault_kp_{s}.npz')} for s in (101, 102, 103)}
out['f4']['speeds'] = {n: round(load(M / f'f4/eval_{n}.json')['mean_forward_speed'], 3) for n in ('fwd', 'back', 'stop', 'zero')}
out['f4']['mlx_abs_error'] = {n: load(M / f'f4/eval_{n}.json')['absolute_error']['right'] for n in ('fwd', 'back', 'stop')}
# F5
c5, m5 = np.load(C / 'f5/sensors/sensors.npz'), np.load(M / 'f5/sensors/sensors.npz')
views = sorted({k.split('/', 1)[1].rsplit('/', 1)[0] for k in c5.files if k.startswith('rgb/')})
psnr = lambda a, b: -10 * np.log10(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
across, depth, labels, seg_cover = [], 0.0, 0.0, []
for v in views:
    gi = [c5[k] for k in c5.files if k.startswith(f'rgb/{v}/')]; ci = [m5[k] for k in m5.files if k.startswith(f'rgb/{v}/')]
    across.append(min(psnr(a, b) for a, b in itertools.product(gi, ci)))
    for side in (c5, m5):
        same_shape = (side[f'segmentation/{v}'] == side[f'exact_label/{v}']) & (side[f'segmentation/{v}'] >= 0)
        depth = max(depth, float(np.abs(side[f'depth/{v}'][same_shape] - side[f'exact_depth/{v}'][same_shape]).max()))
        labels = max(labels, float(np.mean(side[f'segmentation/{v}'] != side[f'exact_label/{v}'])))
    seg_cover.append(float(np.mean(c5[f'segmentation/{v}'] == 1)))
out['f5'] = {'worst_psnr_across_db': round(float(min(across)), 2), 'worst_depth_vs_exact_m': depth, 'worst_label_mismatch': labels, 'box_pixel_fraction_per_view': [round(x, 3) for x in seg_cover]}
# F1 attempt 3: the replay JSON against the claims, and the fault effects
r3j = load(H / 'cumlx-repro/result-20261106-102011/f1/replay.json')
out['f1_attempt3'] = {'worst_leaf': max(v['rel_l2'] for v in r3j['leaves'].values()), 'leaves': len(r3j['leaves']), 'cb_differ_max': max(v['cb_differ_fraction'] for v in r3j['weights'].values()),
                      'worst_layer': max(v['rel_l2'] for v in r3j['layers'].values()), 'worst_attention_vs_exact': max(v['against_exact']['rel_l2'] for v in r3j['attention'].values()),
                      'tokens_equal': sum(a == b for a, b in zip(load(C / 'f1/program.json')['token_ids'], load(H / 'cumlx-repro/result-20261106-102011/f1/program.json')['token_ids']))}
print(json.dumps(out, indent=1, default=str))
