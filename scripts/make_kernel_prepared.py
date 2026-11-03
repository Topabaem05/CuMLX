"""Write sources prepared by `cumlx kernel port --preprocess-only`, so that a CUDA run can check the pre-translation layer.

  <out>/fixtures/   the test kernels, prepared, with their cases
  <out>/lab/        one case of each FP4 kernel, wrapped in a complete kernel with its constants

Record both with `cumlx kernel capture`. A prepared source must compute what its original computes on the same GPU.
Usage: python scripts/make_kernel_prepared.py -o OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from cumlx.verify.kernel_port import preprocess_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/kernel_fixtures'
LAB = ROOT / 'src/cumlx/kernels'
LAB_CASES = ROOT / 'validation/kernel-cases'
FIXTURE_KERNELS = ('vector_add', 'norm_mixed', 'builtin_map')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    (out / 'fixtures').mkdir(parents=True)
    (out / 'lab').mkdir()
    for name in FIXTURE_KERNELS:
        (out / 'fixtures' / f'{name}.cu').write_text(preprocess_file(FIXTURES / f'{name}.cu').source)
    shutil.copy(FIXTURES / 'cases.json', out / 'fixtures' / 'cases.json')
    doc = json.loads((LAB_CASES / 'cases.json').read_text())
    chosen: dict[str, dict] = {}
    def size(case: dict) -> int:
        return max(max(t['shape']) for t in case['outputs'])

    for case in doc['cases']:  # the largest case of each kernel exercises the most of it
        if case['kernel'] not in chosen or size(case) > size(chosen[case['kernel']]):
            chosen[case['kernel']] = case
    for kernel, case in chosen.items():
        pre = preprocess_file(LAB / f'{kernel}.cu', cases_file=str(LAB_CASES / 'cases.json'), case_id=case['id'])
        (out / 'lab' / f'{kernel}.cu').write_text(pre.source)
    (out / 'lab' / 'cases.json').write_text(json.dumps({'schema': doc['schema'], 'cases': list(chosen.values())}, indent=1) + '\n')
    shutil.copy(LAB_CASES / 'inputs.safetensors', out / 'lab' / 'inputs.safetensors')
    print(f'{len(FIXTURE_KERNELS)} test kernels and {len(chosen)} lab kernels prepared in {out}')


if __name__ == '__main__':
    main()
