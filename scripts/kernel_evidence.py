"""Measure `cumlx kernel verify` against the V100 goldens under validation/, and write validation/kernel-translation.json.

It runs each kernel on this Mac: the right translation and each wrong one, through MLX and through PyTorch.
Usage: python scripts/kernel_evidence.py -o validation/kernel-translation.json
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

from cumlx.verify.faults import KERNEL_FAULTS
from cumlx.verify.kernel_verify import MLXEngine, TorchEngine, verify_kernels
from cumlx.verify.metrics import KERNEL_THRESHOLDS, compare_kernel_output
from cumlx.verify.opgolden import OpGolden

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'src/cumlx/kernels'
GOLDEN = ROOT / 'validation/kernel-golden-v100'
PREPARED_LAB = ROOT / 'validation/kernel-golden-v100-prepared-lab'
FIXTURES = ROOT / 'validation/kernel-golden-v100-fixtures'
ORIGINAL = ROOT / 'validation/kernel-golden-v100-fixtures-original'


def right_and_wrong(engine: MLXEngine, work: Path) -> list[dict]:
    rows = []
    header = str(LAB / 'common_metal.h')
    for fault in KERNEL_FAULTS.values():
        source = (LAB / f'{fault.kernel}.metal').read_text()
        folder = work / fault.id
        folder.mkdir()
        path = folder / f'{fault.kernel}.metal'
        path.write_text(source)
        right = verify_kernels(path, GOLDEN, header=header, engine=engine)
        path.write_text(fault.apply(source))
        wrong = verify_kernels(path, GOLDEN, header=header, engine=engine)
        first = wrong['first_divergence'] or {}
        rows.append({
            'fault': fault.id, 'kind': fault.kind, 'title': fault.title, 'kernel': fault.kernel,
            'right': {'verdict': right['verdict'], 'grade': right['kernels'][fault.kernel]['grade'],
                      'cases_passed': right['kernels'][fault.kernel]['passed'], 'cases': right['kernels'][fault.kernel]['cases'],
                      'worst_rel_l2': right['kernels'][fault.kernel]['worst_rel_l2']},
            'wrong': {'verdict': wrong['verdict'], 'cases_failed': wrong['summary']['cases']['fail'],
                      'cases': wrong['kernels'][fault.kernel]['cases'], 'first_case': first.get('case'), 'reason': first.get('reason'),
                      'rel_l2': first.get('rel_l2'), 'max_rel': first.get('max_rel'), 'mismatched': first.get('mismatched')}})
    return rows


def summarize(report: dict) -> dict:
    return {'verdict': report['verdict'], 'summary': report['summary'], 'kernels': report['kernels'],
            'failures': [{'case': r['case'], 'outputs': {k: {f: m.get(f) for f in ('reason', 'rel_l2', 'max_rel', 'max_ulp', 'index', 'golden', 'port')}
                                                         for k, m in r['outputs'].items() if m['verdict'] == 'fail'}}
                         for r in report['records'] if r['status'] == 'fail'],
            'unverified': [{'case': r['case'], 'stage': r.get('stage'), 'reason': (r.get('reason') or '')[:400]}
                           for r in report['records'] if r['status'] == 'unverified']}


def prepared_lab() -> list[dict]:
    lab, prepared = OpGolden(GOLDEN), OpGolden(PREPARED_LAB)
    rows = []
    for case in prepared.cases:
        _, ours = prepared.load(case)
        _, original = lab.load(next(c for c in lab.cases if c.id == case.id))
        rows.append({'case': case.id, 'bytes_identical': all(ours[k].tobytes() == original[k].tobytes() for k in ours)})
    return rows


def rewritten_builtins() -> dict:
    prepared, original = OpGolden(FIXTURES), OpGolden(ORIGINAL)
    _, ours = prepared.load(next(c for c in prepared.cases if c.kernel == 'builtin_map'))
    _, theirs = original.load(original.cases[0])
    floats = compare_kernel_output(ours['out'], theirs['out'])
    names = ['__fadd_rn', '__fsub_rn', '__fmul_rn', '__fdiv_rn', '__fmaf_rn', '__frcp_rn', '__fsqrt_rn', '__frsqrt_rn', 'rsqrtf', '__saturatef', '__fdividef', 'half round trip']
    differ = np.nonzero(ours['out'] != theirs['out'])[0].tolist()
    return {'integer_outputs_identical': bool(np.array_equal(ours['ints'], theirs['ints'])), 'integer_elements': int(ours['ints'].size),
            'float_verdict': floats['verdict'], 'float_elements': int(ours['out'].size), 'float_rel_l2': floats['rel_l2'], 'float_max_ulp': floats['max_ulp'],
            'float_elements_that_differ': len(differ), 'builtins_that_differ': sorted({names[i % 12] for i in differ})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    mlx = MLXEngine()
    torch = TorchEngine()
    golden = OpGolden(GOLDEN)
    right = verify_kernels(LAB, GOLDEN, engine=mlx)
    through_torch = verify_kernels(LAB, GOLDEN, engine=torch, repeats=1)
    with tempfile.TemporaryDirectory() as tmp:
        faults = right_and_wrong(mlx, Path(tmp))
    doc = {
        'schema': 'cumlx.kernel-translation.v1', 'date': date.today().isoformat(),
        'golden_source': golden.source, 'golden_floating_point': golden.floating_point,
        'metal': {'machine': platform.machine(), 'system': platform.platform(), **mlx.describe()},
        'limits': KERNEL_THRESHOLDS,
        'limits_written_before_measuring': 'commit f93ac26, before the first CUDA or Metal output of these kernels was compared',
        'golden_count': {'lab_kernels': len({c.kernel for c in golden.cases}), 'lab_cases': len(golden.cases),
                         'test_kernels': len({c.kernel for c in OpGolden(FIXTURES).cases}), 'm3_gate_needs': 20},
        'hand_written_metal_against_v100': {
            'mlx': summarize(right), 'torch': summarize(through_torch), 'execution': right['execution']['differences'],
            'measured_on_metal': right['execution']['measured'], 'warnings': right['warnings']},
        'wrong_and_right_translations': faults,
        'pre_translation_on_v100': {'lab_kernels_wrapped_against_original': prepared_lab(), 'rewritten_builtins_against_original': rewritten_builtins()},
        'limits_of_this_evidence': ('One V100 and one M5 Pro. The kernels are this repository\'s own, written for portability, and they use no warp-level '
                                    'behaviour. No third-party translator ran: `cumlx kernel port` was tested with a stub. The inputs are normal floats, plus '
                                    'subnormals in one case and NaN scales in two. Subnormal and overflow regions are not certified.')}
    Path(args.output).write_text(json.dumps(doc, indent=1, default=str) + '\n')
    wrong_caught = sum(1 for f in faults if f['wrong']['verdict'] == 'diverged')
    right_ok = sum(1 for f in faults if f['right']['verdict'] == 'ok')
    print(f"MLX {right['verdict']} {right['summary']['cases']}; torch {through_torch['verdict']} {through_torch['summary']['cases']}; "
          f'{wrong_caught}/{len(faults)} wrong caught; {right_ok}/{len(faults)} right pass')


if __name__ == '__main__':
    main()
