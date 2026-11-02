#!/usr/bin/env python3
"""The body of the CUMLX GitHub Action. Settings come from the environment; see action.yml.

Exit status is the verdict: only `ok` passes. `incomplete` fails the job, because it is not a pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path('cumlx-output')


def run(*argv: str) -> dict:
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json'], stdout=subprocess.PIPE, text=True, check=False)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'exit_code': p.returncode or 10, 'status': 'error', 'summary': 'no JSON result', 'errors': [], 'data': {}}


def output(key: str, value: str) -> None:
    path = os.environ.get('GITHUB_OUTPUT')
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f'{key}={value}\n')


def summary(lines: list[str]) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with open(path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


def fail(doc: dict, step: str) -> int:
    print(f"::error title=cumlx {step}::{doc.get('summary')}")
    for e in doc.get('errors', []):
        print(f"::error::{e.get('code')}: {e.get('message')} {e.get('hint', '')}")
    summary([f"## CUMLX: {step} failed", '', str(doc.get('summary'))])
    return int(doc.get('exit_code') or 1)


def main() -> int:
    env = os.environ
    original, port = env['ORIGINAL'], env['PORT']
    loader = env.get('LOADER') or 'mlx_lm'
    OUT.mkdir(exist_ok=True)
    if env.get('CHECK_WEIGHTS', 'true') == 'true':
        doc = run('weights', original, port, '--loader', loader, '-o', str(OUT / 'weights.json'), '--overwrite')
        print(doc.get('summary'))
        if doc.get('exit_code') != 0:
            first = (doc.get('data') or {}).get('first_failure') or {}
            output('verdict', (doc.get('data') or {}).get('verdict', 'error'))
            output('first-divergence', first.get('name', ''))
            return fail(doc, 'weights')
    inputs = []
    if env.get('PROMPT'):
        inputs += ['--prompt', env['PROMPT']]
    if env.get('TOKEN_IDS'):
        inputs += ['--token-ids', env['TOKEN_IDS']]
    capture = ['capture', original, *inputs, '-o', str(OUT / 'golden'), '--overwrite']
    if env.get('PORT_WEIGHTS') == 'true':
        capture += ['--weights-from-mlx', port]
    doc = run(*capture)
    print(doc.get('summary'))
    if doc.get('exit_code') != 0:
        return fail(doc, 'capture')
    doc = run('diff', str(OUT / 'golden'), port, '--loader', loader, '-o', str(OUT / 'report'), '--overwrite')
    print(doc.get('summary'))
    data = doc.get('data') or {}
    fd = data.get('first_divergence') or {}
    output('verdict', data.get('verdict', 'error'))
    output('first-divergence', fd.get('module', ''))
    output('report-dir', str(OUT / 'report'))
    report_md = OUT / 'report' / 'report.md'
    summary([f"## CUMLX: {data.get('verdict', 'error')}", '', str(doc.get('summary')), ''] +
            ([f"First divergence: `{fd.get('module') or '(root)'}` ({fd.get('basis')}, {fd.get('confidence')})", '',
              str(fd.get('explanation')), ''] if fd else []) +
            ([report_md.read_text(encoding='utf-8')[:6000]] if report_md.is_file() else []))
    if doc.get('exit_code') != 0:
        return fail(doc, 'diff')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
