#!/usr/bin/env python3
"""Check the command-line contract in a base install, with no MLX, torch or NVIDIA tools. Standard library only.

    python scripts/compat_check.py --python PATH_TO_PYTHON     # exit 0 only if every check holds
    python scripts/compat_check.py --python PY -o result.json

Give it the Python of an environment that holds only the wheel and NumPy. It refuses an environment that has MLX or torch,
because then the checks would not test what they name. Each command must give one JSON document on stdout, its exit code
must match the envelope, and a missing part must give the documented code and a hint. A missing part is never a pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

SCHEMA = 'cumlx.result.v1'
FORBIDDEN = ('mlx', 'mlx_lm', 'torch', 'transformers', 'mitsuba', 'genesis', 'warp', 'woof')
KERNEL = '__global__ void relu(float *x, int n) {\n    x[0] = x[0] > 0.0f ? x[0] : 0.0f;\n}\n'


def present_modules(python: str) -> list[str]:
    code = f'import importlib.util as u; print(",".join(m for m in {FORBIDDEN!r} if u.find_spec(m)))'
    out = subprocess.run([python, '-I', '-c', code], capture_output=True, text=True, check=True, timeout=60).stdout.strip()
    return [m for m in out.split(',') if m]


def cli(python: str, argv: list[str], cwd: str) -> tuple[int, dict[str, Any] | None, str]:
    done = subprocess.run([python, '-I', '-m', 'cumlx', *argv, '--format', 'json'], capture_output=True, text=True, timeout=180, cwd=cwd, check=False)
    try:
        doc = json.loads(done.stdout)
    except json.JSONDecodeError:
        doc = None
    return done.returncode, doc, done.stdout + done.stderr


def check(name: str, python: str, argv: list[str], cwd: str, exit_code: int, error: str | None = None, hint: str | None = None) -> dict[str, Any]:
    code, doc, raw = cli(python, argv, cwd)
    problems = []
    if doc is None:
        problems.append(f'stdout is not one JSON document: {raw[:300]}')
    else:
        if doc.get('schema') != SCHEMA:
            problems.append(f'schema is {doc.get("schema")!r}')
        if doc.get('exit_code') != code:
            problems.append(f'envelope says exit {doc.get("exit_code")}, and the process gave {code}')
        codes = [e.get('code') for e in doc.get('errors', [])]
        if error is not None and error not in codes:
            problems.append(f'expected {error}, got {codes}')
        if hint is not None and not any(hint in str(e.get('hint')) for e in doc.get('errors', [])):
            problems.append(f'no error has a hint with {hint!r}: {doc.get("errors")}')
    if code != exit_code:
        problems.append(f'exit {code}, expected {exit_code}')
    return {'name': name, 'argv': argv, 'ok': not problems, 'exit_code': code, 'problems': problems}


def run_checks(python: str) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        kernel = work / 'relu.cu'
        kernel.write_text(KERNEL)
        repo = work / 'project'
        repo.mkdir()
        (repo / 'train.py').write_text("import torch\nprint(torch.cuda.is_available())\n")
        cwd = str(work)
        return [
            check('describe', python, ['describe'], cwd, 0),
            check('doctor', python, ['doctor'], cwd, 0),
            check('selftest core', python, ['selftest', '--core-only'], cwd, 0),
            check('doctor names missing extras', python, ['doctor', '--require', 'capture,diff'], cwd, 3, 'E_DEP_MISSING', 'pip install'),
            check('doctor --stack is not a pass', python, ['doctor', '--stack'], cwd, 3, 'E_DEP_MISSING', 'pip install'),
            check('capture without torch', python, ['capture', 'some-model', '--token-ids', '1,2', '-o', str(work / 'g')], cwd, 3, 'E_DEP_MISSING', 'capture'),
            check('diff without mlx', python, ['diff', str(work / 'g'), 'some-port', '-o', str(work / 'r')], cwd, 3, 'E_DEP_MISSING', 'diff'),
            check('run without torch', python, ['run', str(repo / 'train.py')], cwd, 3, 'E_DEP_MISSING', 'capture'),
            check('kernel capture without nvcc', python, ['doctor', '--require', 'kernel capture'], cwd, 3, 'E_DEP_MISSING', 'CUDA toolkit'),
            check('kernel verify without mlx', python, ['doctor', '--require', 'kernel verify'], cwd, 3, 'E_DEP_MISSING', 'diff'),
            check('kernel port needs no translator to prepare', python, ['kernel', 'port', str(kernel), '--preprocess-only', '-o', str(work / 'out.cu')], cwd, 0),
            check('kernel port without a translator', python, ['kernel', 'port', str(kernel), '-o', str(work / 'translated.metal')], cwd, 3, 'E_KERNEL_TRANSLATOR_MISSING'),
            check('scan works without the extras', python, ['scan', str(repo)], cwd, 0),
            check('recipe works without the extras', python, ['recipe', 'bitsandbytes'], cwd, 0),
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python', required=True)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()
    extra = present_modules(args.python)
    if extra:
        print(f'error: this environment has {extra}, so it is not a base install. Nothing was checked.', file=sys.stderr)
        return 3
    results = run_checks(args.python)
    failed = [r for r in results if not r['ok']]
    report = {'checks': len(results), 'failed': len(failed), 'results': results}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'checks': len(results), 'failed': [{'name': r['name'], 'problems': r['problems']} for r in failed]}, indent=1))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
