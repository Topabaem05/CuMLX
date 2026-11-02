#!/usr/bin/env python3
"""Run open-source PyTorch scripts that assume CUDA under `cumlx run`, with no edit to any of them.

This is the M2 gate in docs/ROADMAP-PLATFORM.md. It clones each repository at a fixed commit, builds a
venv, and runs each script. One failure does not stop the others. A failure stays a failure.

    python spikes/m2_scripts.py --work ~/cumlx-m2-scripts                  # on the Mac, target mps
    python spikes/m2_scripts.py --work ~/cumlx-m2-scripts --target cuda    # the golden, on a CUDA machine
    cumlx rundiff cuda-reports/nanogpt_train mps-reports/nanogpt_train     # the comparison

A script that trains runs once per seed, with its losses recorded. It runs third-party code with your
permissions, and it downloads MNIST.
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
import time

ROOT = Path(__file__).resolve().parents[1]
TORCH = ('torch==2.15.0', 'torchvision==0.29.2')
REPOSITORIES = {
    'nanoGPT': ('https://github.com/karpathy/nanoGPT.git', '3adf61e154c3fe3fca428ad6bc3818b27a3b8291'),
    'minGPT': ('https://github.com/karpathy/minGPT.git', '37baab71b9abea1b76ab957409a1cc2fbfba8a26'),
    'PyTorch-GAN': ('https://github.com/eriklindernoren/PyTorch-GAN.git', '36d3c77e5ff20ebe0aeefd322326a134a279b93e'),
    'pytorch-tutorial': ('https://github.com/yunjey/pytorch-tutorial.git', '0500d3df5a2a8080ccfccbc00aca0eacc21818db'),
}
# A small model and few iterations are arguments of these scripts, not edits to them.
SCRIPTS = [
    {'name': 'nanogpt_train', 'repo': 'nanoGPT', 'cwd': '.', 'before': ['data/shakespeare_char/prepare.py'], 'script': 'train.py',
     'args': ['config/train_shakespeare_char.py', '--n_layer=2', '--n_head=2', '--n_embd=64', '--block_size=32', '--batch_size=12',
              '--max_iters=200', '--eval_interval=100', '--eval_iters=2', '--compile=False', '--dropout=0.0', '--always_save_checkpoint=True'],
     'timeout': 1200, 'trains': True, 'progress': r'iter (\d+): loss ([\d.]+)'},
    {'name': 'nanogpt_sample', 'repo': 'nanoGPT', 'cwd': '.', 'script': 'sample.py',
     'args': ['--out_dir=out-shakespeare-char', '--num_samples=1', '--max_new_tokens=40', '--compile=False'], 'timeout': 300},
    {'name': 'mingpt_adder', 'repo': 'minGPT', 'cwd': '.', 'script': 'projects/adder/adder.py', 'pythonpath': '.',
     'args': ['--trainer.max_iters=50', '--trainer.batch_size=64', '--model.model_type=gpt-nano'], 'timeout': 600, 'trains': True,
     'progress': r'iter (\d+): train loss ([\d.]+)'},
    {'name': 'gan_mnist', 'repo': 'PyTorch-GAN', 'cwd': 'implementations/gan', 'script': 'gan.py', 'args': ['--n_epochs', '1'], 'timeout': 1200, 'trains': True,
     'progress': r'\[D loss: ([\d.]+)\]'},
    {'name': 'cnn_mnist', 'repo': 'pytorch-tutorial', 'cwd': 'tutorials/02-intermediate/convolutional_neural_network', 'script': 'main.py', 'args': [], 'timeout': 1200, 'trains': True,
     'progress': r'Loss: ([\d.]+)', 'final': r'Test Accuracy[^:]*: ([\d.]+) ?%'},
]


def progress_of(text: str, pattern: str | None, final: str | None = None) -> dict:
    """Numbers from the script's own output. Absent numbers stay absent."""
    out: dict = {}
    if pattern:
        found = re.findall(pattern, text)
        if found:
            values = [float(item[-1] if isinstance(item, tuple) else item) for item in found]
            out = {'first': values[0], 'last': values[-1], 'samples': len(values)}
    if final:
        match = re.findall(final, text)
        if match:
            out['final'] = float(match[-1])
    return out


def run_one(entry: dict, sources: Path, python: Path, reports: Path, target: str, seed: int | None) -> dict:
    cwd = sources / entry['repo'] / entry['cwd']
    name = entry['name']
    before_code = 0
    for script in entry.get('before', []):
        before_code = subprocess.run([str(python), script], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode
        if before_code:
            return {'name': name, 'status': 'setup_failed', 'setup_script': script, 'exit_code': before_code}
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    if entry.get('pythonpath'):
        env['PYTHONPATH'] = entry['pythonpath']
    folder = reports / name
    folder.mkdir(exist_ok=True)
    label = 'run' if seed is None else f'seed{seed}'
    report = folder / f'{label}.json'
    argv = [str(python), '-m', 'cumlx', 'run', entry['script'], '--target', target, '--report', str(report), '--overwrite', '--format', 'json',
            '--max-items', '-1', '--timeout', str(entry['timeout'])]
    if seed is not None:
        argv += ['--seed', str(seed), '--trace-losses']
    if entry['args']:
        argv += ['--', *entry['args']]
    start = time.monotonic()
    out_path, err_path = folder / f'{label}.out', folder / f'{label}.err'
    with out_path.open('w') as stdout, err_path.open('w') as stderr:
        code = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False).returncode
    seconds = round(time.monotonic() - start, 1)
    text = err_path.read_text(errors='replace')
    try:
        doc = json.loads(out_path.read_text())
    except json.JSONDecodeError:
        return {'name': name, 'seed': seed, 'status': 'no_result', 'exit_code': code, 'seconds': seconds, 'tail': text[-800:]}
    kinds = doc['data'].get('kinds') or {}
    return {'name': name, 'seed': seed, 'target': target, 'status': doc['status'], 'exit_code': code, 'seconds': seconds, 'errors': doc['errors'], 'warnings': [w['code'] for w in doc['warnings']],
            'kinds': kinds, 'processes': doc['data'].get('processes'), 'progress': progress_of(text, entry.get('progress'), entry.get('final')),
            'tail': text[-600:] if code else ''}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--target', choices=['mps', 'cpu', 'cuda'], default='mps')
    parser.add_argument('--seeds', default='0,1,2', help='comma-separated seeds for the scripts that train')
    parser.add_argument('--only', help='comma-separated script names')
    parser.add_argument('--manifest', type=Path, help='JSON list of entries in place of SCRIPTS, with sources already in place. Used by the tests')
    args = parser.parse_args(argv)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    sources, reports = work / 'sources', work / 'reports'
    sources.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    summary_path = work / 'summary.json'
    if args.manifest:
        entries, python, commits = json.loads(args.manifest.read_text()), Path(sys.executable), {}
    else:
        if shutil.which('uv') is None:
            print('error: uv not found', file=sys.stderr)
            return 3
        python = work / 'venv' / 'bin' / 'python'
        print('[setup] venv and packages', flush=True)
        subprocess.run(['uv', 'venv', '--clear', '-q', '--python', '3.12', str(work / 'venv')], check=True)
        subprocess.run(['uv', 'pip', 'install', '-q', '--python', str(python), *TORCH, 'numpy', 'requests', 'tiktoken', '-e', str(ROOT)], check=True)
        for name, (url, commit) in REPOSITORIES.items():
            target = sources / name
            if not (target / '.git').exists():
                print(f'[setup] {name}', flush=True)
                subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', url, str(target)], check=True)
            subprocess.run(['git', '-C', str(target), '-c', 'advice.detachedHead=false', 'checkout', '-q', commit], check=True)
        entries, commits = SCRIPTS, {k: v[1] for k, v in REPOSITORIES.items()}
    if args.only:
        wanted = set(args.only.split(','))
        entries = [entry for entry in entries if entry['name'] in wanted]
    seeds = [int(seed) for seed in args.seeds.split(',')]
    plan = [(entry, seed) for entry in entries for seed in (seeds if entry.get('trains') else [None])]
    results = []
    for entry, seed in plan:
        print(f"[{entry['name']} seed {seed}] running", flush=True)
        result = run_one(entry, sources, python, reports, args.target, seed)
        results.append(result)
        print(f"[{entry['name']} seed {seed}] {result['status']} exit {result['exit_code']} {result.get('seconds', '')}s", flush=True)
        summary_path.write_text(json.dumps({'schema': 'cumlx.spike.m2_scripts.v1', 'commits': commits, 'edited_scripts': 0, 'results': results}, indent=1) + '\n')
    passed = sum(r['exit_code'] == 0 for r in results)
    print(f'[done] {passed} of {len(results)} ran to the end. Result: {summary_path}', flush=True)
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
