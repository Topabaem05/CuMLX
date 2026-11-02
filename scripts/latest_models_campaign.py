#!/usr/bin/env python3
"""Verify mlx-lm's implementation of recent Hugging Face models, then delete each model.

Each model loads twice from the same checkpoint: transformers makes the golden, and mlx-lm is the port. A divergence
means the mlx-lm code computes something else than the transformers code, or the config means something else to it.
A second run converts the model to 4 bits and compares it with a golden that holds its own weights.

Remote code is never trusted, so a model that needs it is recorded as not run.

    python scripts/latest_models_campaign.py -o validation/latest-models.json
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
MODELS = ['Octen/Octen-WebGR-0.6B', 'inclusionAI/Ling-3.0-flash-UBTree', 'udgamlabs/Laghu', 'udgamlabs/Vemana-Base',
          'nkthebass/tinybrainbot-61m-a10m-v4.1-moe-base', 'nkthebass/tinybrainbot-360m-a75m-v5.1-instruct',
          'nkthebass/tinybrainbot-100m-v5.1-base', 'openbmb/MiniCPM-V-4.7-1B-DSpark', 'CohereLabs/North-Mini-Code-1.0-eagle',
          'LiquidAI/LFM2.5-230M', 'incoai/Qwen3.8-Flash-Next-DFlash2', 'Prience91/Climate-1000-Granite-0.35b']
MIN_FREE_BYTES = 4 * 1024**3


CLEAN = re.compile(r'/(?:private/)?var/folders/[^\\s"\\']+|/Users/[^/\\s"\\']+')  # local paths stay out of committed evidence


def clean(text: str) -> str:
    return CLEAN.sub('<local>', text)


def cumlx(*argv: str) -> dict:
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '8'],
                       capture_output=True, text=True, check=False)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'exit_code': p.returncode, 'status': 'error', 'summary': 'no JSON result',
                'errors': [{'code': 'E_NO_JSON', 'message': p.stderr[-500:]}], 'warnings': [], 'data': {}}


def errors(doc: dict) -> list[dict]:
    return [{'code': e.get('code'), 'message': str(e.get('message'))[:500]} for e in doc.get('errors', [])]


def diff_row(doc: dict) -> dict:
    data = doc.get('data') or {}
    s = data.get('summary') or {}
    fd = data.get('first_divergence')
    return {'status': doc.get('status'), 'exit_code': doc.get('exit_code'), 'summary': doc.get('summary'), 'errors': errors(doc),
            'verdict': data.get('verdict'), 'warnings': [w.get('code') for w in doc.get('warnings', [])],
            'first_divergence': fd and {k: fd.get(k) for k in ('module', 'phase', 'basis', 'confidence', 'explanation')}
            | {'rel_l2': (fd.get('metrics') or {}).get('rel_l2'), 'reason': (fd.get('metrics') or {}).get('reason')},
            'coverage': s.get('parametrized_coverage'), 'pass': s.get('pass'), 'fail': s.get('fail'), 'unverified': s.get('unverified'),
            'worst_isolated_rel_l2': s.get('worst_isolated_rel_l2'), 'worst_chain_rel_l2': s.get('worst_chain_rel_l2'),
            'chain_drift': s.get('chain_drift'), 'decode_records': s.get('decode_records'), 'native_output': s.get('native_output'),
            'profile': (data.get('port') or {}).get('profile'), 'threshold': data.get('threshold'),
            'unverified_items': [(u.get('name'), u.get('note')) for u in (data.get('unverified') or {}).get('items', [])[:6]],
            'failures': [(f.get('name'), f.get('isolated_rel_l2')) for f in (data.get('failures') or {}).get('items', [])[:8]]}


def verify(model: str, work: Path, prompts: Path) -> dict:
    row: dict = {'model': model}
    golden = work / 'golden'
    captured = cumlx('capture', model, '--prompts-file', str(prompts), '--max-tokens', '48', '-o', str(golden), '--overwrite')
    token_input: list[str] = []
    if captured.get('exit_code') != 0 and 'produced no tokens' in str(captured.get('summary')):
        # A draft model repository can have an empty tokenizer. The model is still comparable.
        token_input = ['--token-ids', '1,5,9,23,7,99,3,64,12,8', '--token-ids', '11,3,40,2']
        row['input'] = 'token ids: the repository tokenizer is empty'
        captured = cumlx('capture', model, *token_input, '-o', str(golden), '--overwrite')
    if captured.get('exit_code') != 0:
        row['result'] = 'not_run'
        row['capture'] = {'summary': captured.get('summary'), 'errors': errors(captured)}
        return row
    source = (captured.get('data') or {}).get('source') or {}
    row.update(revision=source.get('revision'), architecture=source.get('architecture'), records=captured['data'].get('records'))
    weights = cumlx('weights', model, model)
    data = weights.get('data') or {}
    row['weights'] = {'verdict': data.get('verdict'), 'summary': weights.get('summary'), 'errors': errors(weights),
                      'first_failure': data.get('first_failure'), 'unverified': (data.get('unverified') or {}).get('items', [])[:6]}
    row['implementation'] = diff_row(cumlx('diff', str(golden), model))
    row['result'] = row['implementation'].get('verdict') or 'port_error'
    port = work / 'port-4bit'
    convert = subprocess.run([sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', model, '--mlx-path', str(port), '-q', '--q-bits', '4'],
                             capture_output=True, text=True, check=False)
    if convert.returncode != 0:
        row['quantized'] = {'status': 'convert_failed', 'message': convert.stderr[-400:]}
        return row
    aware = work / 'golden-q'
    inputs = token_input or ['--prompts-file', str(prompts), '--max-tokens', '48']
    captured = cumlx('capture', model, *inputs, '-o', str(aware), '--overwrite', '--weights-from-mlx', str(port))
    if captured.get('exit_code') != 0:
        row['quantized'] = {'status': 'capture_failed', 'summary': captured.get('summary'), 'errors': errors(captured)}
        return row
    row['quantized'] = diff_row(cumlx('diff', str(aware), str(port)))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--model', action='append')
    args = ap.parse_args()
    rows: list[dict] = []
    with TemporaryDirectory(prefix='cumlx-latest-') as tmp:
        root = Path(tmp)
        prompts = root / 'prompts.txt'
        prompts.write_text('\n'.join(PROMPTS) + '\n', encoding='utf-8')
        for model in args.model or MODELS:
            if shutil.disk_usage(tmp).free < MIN_FREE_BYTES:
                rows.append({'model': model, 'result': 'not_run', 'reason': 'not enough free disk'})
                continue
            print(f'verifying {model}', file=sys.stderr, flush=True)
            work = root / 'work'
            work.mkdir()
            os.environ['HF_HOME'] = str(work / 'hf')  # all of it goes away with the next line of this loop
            try:
                rows.append(verify(model, work, prompts))
            finally:
                shutil.rmtree(work, ignore_errors=True)
            Path(args.output).write_text(clean(json.dumps({'schema': 'cumlx.latest-models.v1', 'date': '2026-11-02', 'models': rows}, indent=1,
                                                          ensure_ascii=False)) + '\n', encoding='utf-8')
    print(json.dumps({r['model']: r['result'] for r in rows}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
