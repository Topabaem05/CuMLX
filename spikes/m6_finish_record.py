#!/usr/bin/env python3
"""Write validation/m6-finish-run.json from the outputs of the M6 finish runs. It reruns each judge, so the record holds what the
judges say about the folders named here, and it measures nothing itself.

    python spikes/m6_finish_record.py --out validation/m6-finish-run.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m6_f1_judge as j1  # noqa: E402
import m6_f2_cuda_judge as j2  # noqa: E402
import m6_f3_judge as j3  # noqa: E402
import m6_f4_judge as j4  # noqa: E402
import m6_f5_judge as j5  # noqa: E402

HOME = Path.home()
CUDA = HOME / 'cumlx-m6-cuda'
MAC2 = HOME / 'cumlx-repro' / 'result-20261106-101702'
MAC3 = HOME / 'cumlx-repro' / 'result-20261106-102011'
ALT_CUDA = HOME / 'cumlx-m6-f1alt' / 'f1alt'
ALT_MAC = HOME / 'cumlx-repro' / 'result' / 'f1'


def text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def brief(verdict: dict[str, Any]) -> dict[str, Any]:
    keep = ('worst_rel_l2', 'least_equal_fraction', 'most_differ_fraction', 'worst_mac', 'worst_cuda', 'equal_tokens', 'of', 'text', 'device',
            'mean_candidate', 'mean_reference', 'difference', 'tolerance', 'right', 'labels_equal_to_the_bit', 'masks')
    return {'verdict': verdict['verdict'], 'grade': verdict.get('grade'),
            'parts': {name: {'state': part['state'], 'reasons': part['reasons'][:5], **{k: part[k] for k in keep if k in part}} for name, part in verdict['parts'].items()}}


def f2_numbers(verdict: dict[str, Any]) -> dict[str, Any]:
    parts = verdict['parts']
    renders = parts['renders_between'].get('measured', {})
    return {'renders_between_worst_image_rel_l2': max((v['image_rel_l2'] for v in renders.values()), default=None),
            'renders_between_worst_alpha_rel_l2': max((v['alpha_rel_l2'] for v in renders.values()), default=None),
            'gradients_worst_rel_l2': max(parts['gradients'].get('measured', {}).values(), default=None),
            'controls_caught': {k: v['caught'] for k, v in parts['controls'].get('caught', {}).items()},
            'training_right_psnr': {'cuda': parts['training_cuda'].get('right'), 'mac': parts['training_mac'].get('right')},
            'equivalence': {k: parts['equivalence'].get(k) for k in ('mean_candidate', 'mean_reference', 'difference', 'tolerance')},
            'equivalence_controls': {k: {x: v.get(x) for x in ('difference', 'tolerance', 'within')} for k, v in parts['equivalence_controls'].items() if isinstance(v, dict) and 'within' in v}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    v1_2, v1_3, v1_alt = j1.judge(CUDA / 'f1', MAC2 / 'f1'), j1.judge(CUDA / 'f1', MAC3 / 'f1'), j1.judge(ALT_CUDA, ALT_MAC)
    v2, v3 = j2.judge(CUDA / 'f2', MAC2 / 'f2cuda'), j3.judge(CUDA / 'f3', MAC2 / 'f3')
    v4, v5 = j4.judge(MAC2 / 'f4'), j5.judge(CUDA / 'f5', MAC2 / 'f5')
    versions = {name: json.loads(text(CUDA / name / 'versions.json') or 'null') for name in ('f1', 'f2')}
    record = {
        'schema': 'cumlx.m6_finish_run.v1',
        'date': '2026-11-06',
        'rules': ['validation/m6-f1-limits.json', 'validation/m6-f2-cuda-limits.json', 'validation/m6-f3-limits.json', 'validation/m6-f4-limits.json', 'validation/m6-f5-cuda-limits.json'],
        'how': 'spikes/m6_finish.sh: the CUDA parts on the shared V100 through spikes/m6_cuda_v100.sh with an idle GPU, the Mac parts from the public repository with an empty home folder through spikes/m6_same_mac.sh part m6, and the judges spikes/m6_f*_judge.py. This file was written by spikes/m6_finish_record.py, which reruns the judges on the folders below',
        'machines': {'cuda': 'Tesla V100-SXM2-16GB, driver 580.178.04, nvcc 12.0, sm70, python 3.12 with torch 2.10.0+cu128', 'mac': 'the one Apple M5 Pro, 48 GB, macOS 27, the same Mac as every earlier record'},
        'versions': versions,
        'attempts': [
            {'attempt': 1, 'commit': '840c43c', 'verdict': 'setup errors, nothing measured', 'what': 'F1 on the V100: uv found a Python 3.11 for --python python3. F2: gsplat 1.5.3 with packed=True rejects [C, 3] backgrounds, and the Mac work made no folder. F3: no ninja on the V100, no cmake on the clean Mac PATH. F5: Mitsuba 3.8 transforms cannot be deep-copied. The launcher held ssh open and fetched nothing. Fixed in 28f2143 and 4bf9b41 with the rules unchanged; a review of five readers and two skeptics per finding found the rest before attempt 2'},
            {'attempt': 2, 'commit': '4bf9b41', 'verdict': 'f1 fail, f2 f3 f4 f5 pass', 'the_first_complete_run': True,
             'cuda_commit': text(CUDA / 'cumlx_commit.txt'), 'mac_commit': text(MAC2 / 'cumlx_commit.txt'), 'gpu_busy_at_start': (CUDA / 'busy_at_start.txt').exists(),
             'mac_peak_memory_gb': text(MAC2 / 'peak_used_gb.txt'), 'mac_verdict': text(MAC2 / 'verdict.txt'),
             'base': {name: text(MAC2 / 'base' / f'{name}.exit') for name in ('doctor', 'selftest', 'tests')}},
            {'attempt': 3, 'commit': 'd1717d6', 'parts': 'F1 only: a clean Mac run of the F1 part against attempt 2\'s CUDA golden, which did not change', 'verdict': v1_3['verdict'], 'mac_commit': text(MAC3 / 'cumlx_commit.txt'),
             'what_changed': 'cumlx.compat.bitsandbytes_int8 takes the row scale 127/absmax from a table of CUDA __fdividef values measured on the V100 (validation/bnb-fdividef-sm70.json), where it divided exactly. The limits did not change'},
        ],
        'f1': {'target': 'A', 'attempt_2': brief(v1_2), 'attempt_3': brief(v1_3),
               'cause_of_attempt_2': 'bitsandbytes takes 127/absmax with the approximate __fdividef, 1 ulp from the quotient for 29% of fp16 inputs. Of 840 rows of the 168 int8 layers, 835 were equal to CUDA to the bit; in 4 more, 9 activations within 6e-5 of a rounding half went to the other int8 step; 1 element differed in the bfloat16 outlier sum. One such activation at row 2 of model.layers.14.self_attn.o_proj (-63.500004 on the Mac, -63.499996 on CUDA) moved 58% of that row by one bfloat16 step',
               'second_prompt': {'what': 'A second CUDA golden with the prompt "Water boils at a lower temperature on a mountain because", recorded at d1717d6 after the fix, and its replay in a clean Mac run. The rules name one prompt, so this is reported and not judged. Its judge verdict is not_run only because this golden has no CUDA run of the original program', 'result': brief(v1_alt)}},
        'f2': {'target': 'B', **brief(v2), 'numbers': f2_numbers(v2)},
        'f3': {'target': 'A', **brief(v3)},
        'f4': {'target': 'inference A, physics B', **brief(v4)},
        'f5': {'target': 'B', **brief(v5), 'mitsuba': v5.get('mitsuba')},
        'grades': {'f1': 'A at attempt 3, after a fix that attempt 2\'s failure showed; attempt 2 is a fail', 'f2': v2.get('grade'), 'f3': v3.get('grade'), 'f4': v4.get('grade'), 'f5': v5.get('grade')},
        'ticks_m6_1': False,
        'why_not_ticked': 'The condition of M6-1 names a new Mac, and every part ran on the one Mac that there is. On 2026-11-06 the owner decided to keep M6-1 unticked on this result',
        'does_not_show': ['Another Mac or another Apple GPU generation', 'FlashAttention-2 itself: it needs compute capability 8.0 and the V100 has 7.0, so F1 judges attention against exact attention in float64',
                          'Other GPU generations for the __fdividef table', 'Real datasets, densification or spherical harmonics in F2', 'Isaac Lab, PhysX or a real robot in F4', 'An RTX sensor in F5'],
    }
    check = subprocess.run([sys.executable, str(HERE / 'm6_finish_check.py')], capture_output=True, text=True, check=True)
    record['independent_recompute'] = {'what': 'spikes/m6_finish_check.py: the key numbers again, from the raw files, with code that imports no judge', 'result': json.loads(check.stdout)}
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record['grades'][k] for k in record['grades']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
