#!/usr/bin/env python3
"""L4: verify published MLX ports against their originals and write one JSON evidence file.

    python scripts/public_port_campaign.py -o validation/public-ports.json

Weights are always compared. A quantized port uses a golden with its own weights, so quantization loss is not
reported as a bug. A port's result says nothing about ports this script did not run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

PROMPTS = ['The capital of France is', 'def fibonacci(n):', 'Explain why the sky is blue in one sentence.',
           '한국의 수도는 어디인가요?', '1, 1, 2, 3, 5, 8, 13, 21,']
PORTS = [
    ('Qwen/Qwen2.5-0.5B', 'mlx-community/Qwen2.5-0.5B-bf16', False),
    ('Qwen/Qwen2.5-0.5B', 'mlx-community/Qwen2.5-0.5B-8bit', True),
    ('Qwen/Qwen2.5-0.5B', 'mlx-community/Qwen2.5-0.5B-4bit', True),
    ('Qwen/Qwen2.5-0.5B-Instruct', 'mlx-community/Qwen2.5-0.5B-Instruct-4bit', True),
    ('Qwen/Qwen2.5-Coder-0.5B-Instruct', 'mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit', True),
    ('HuggingFaceTB/SmolLM2-135M-Instruct', 'mlx-community/SmolLM2-135M-Instruct', False),
    ('HuggingFaceTB/SmolLM2-135M-Instruct', 'mlx-community/SmolLM2-135M-Instruct-8bit', True),
    ('Qwen/Qwen3-0.6B', 'mlx-community/Qwen3-0.6B-bf16', False),
    ('Qwen/Qwen3-0.6B', 'mlx-community/Qwen3-0.6B-4bit', True),
    ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit', True),
]


CLEAN = re.compile(r"""/(?:private/)?var/folders/[^\s"']+|/Users/[^/\s"']+""")  # local paths stay out of committed evidence


def clean(text: str) -> str:
    return CLEAN.sub('<local>', text)


def cumlx(*argv: str) -> dict:
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '8'],
                       capture_output=True, text=True, check=False)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'exit_code': p.returncode, 'status': 'error', 'summary': 'no JSON result',
                'errors': [{'code': 'E_NO_JSON', 'message': p.stderr[-600:]}], 'data': {}}


def brief(doc: dict) -> dict:
    return {'exit_code': doc.get('exit_code'), 'status': doc.get('status'), 'summary': doc.get('summary'),
            'errors': [{'code': e.get('code'), 'message': str(e.get('message'))[:400]} for e in doc.get('errors', [])],
            'warnings': [w.get('code') for w in doc.get('warnings', [])]}


def verify(original: str, port: str, quantized: bool, prompts: Path, work: Path) -> dict:
    row: dict = {'original': original, 'port': port, 'golden': 'port-weights' if quantized else 'reference'}
    weights = cumlx('weights', original, port)
    row['weights'] = {**brief(weights), 'first_failure': (weights.get('data') or {}).get('first_failure')}
    golden = work / port.replace('/', '-')
    capture = ['capture', original, '--prompts-file', str(prompts), '--max-tokens', '48', '-o', str(golden), '--overwrite']
    if quantized:
        capture += ['--weights-from-mlx', port]
    captured = cumlx(*capture)
    row['capture'] = brief(captured)
    row['revision'] = ((captured.get('data') or {}).get('source') or {}).get('revision')
    if captured.get('exit_code') != 0:
        return row
    diff = cumlx('diff', str(golden), port)
    data = diff.get('data') or {}
    summary = data.get('summary') or {}
    row['diff'] = {**brief(diff), 'verdict': data.get('verdict'), 'first_divergence': data.get('first_divergence'),
                   'coverage': summary.get('parametrized_coverage'), 'worst_isolated_rel_l2': summary.get('worst_isolated_rel_l2'),
                   'worst_chain_rel_l2': summary.get('worst_chain_rel_l2'), 'chain_drift': summary.get('chain_drift'),
                   'decode_records': summary.get('decode_records'), 'native_output': summary.get('native_output'),
                   'threshold': data.get('threshold'), 'profile': (data.get('port') or {}).get('profile'),
                   'quantization_modes': (data.get('port') or {}).get('quantization_modes'),
                   'unverified': (data.get('unverified') or {}).get('items', [])[:5]}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--port', action='append', metavar='ORIGINAL=PORT[:q]',
                    help='check this pair instead of the built-in list; ":q" makes the golden use the port weights')
    args = ap.parse_args()
    ports = PORTS
    if args.port:
        ports = []
        for item in args.port:
            original, _, port = item.partition('=')
            quantized = port.endswith(':q')
            ports.append((original, port.removesuffix(':q'), quantized))
    rows = []
    with TemporaryDirectory(prefix='cumlx-campaign-') as tmp:
        work = Path(tmp)
        prompts = work / 'prompts.txt'
        prompts.write_text('\n'.join(PROMPTS) + '\n', encoding='utf-8')
        current = None
        for original, port, quantized in ports:
            print(f'verifying {port}', file=sys.stderr, flush=True)
            if original != current:  # each original gets a new download cache, to keep disk use low
                shutil.rmtree(work / 'hf', ignore_errors=True)
                current = original
            os.environ['HF_HOME'] = str(work / 'hf')
            rows.append(verify(original, port, quantized, prompts, work))
            Path(args.output).write_text(clean(json.dumps({'schema': 'cumlx.public-ports.v1', 'ports': rows}, indent=1, ensure_ascii=False)) + '\n',
                                         encoding='utf-8')
            for golden in work.iterdir():
                if golden.is_dir() and golden.name != 'hf':
                    shutil.rmtree(golden, ignore_errors=True)
    print(json.dumps([{'port': r['port'], 'weights': r['weights']['status'],
                       'diff': (r.get('diff') or {}).get('verdict')} for r in rows]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
