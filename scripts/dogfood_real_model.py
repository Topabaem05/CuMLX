#!/usr/bin/env python3
"""L2 dogfooding: run the verifier on real checkpoints and write one JSON evidence file.

Needs network access or a filled HF cache, `.[capture,diff]`, and about 8 GB of disk for each model.

    MLX_ENABLE_TF32=0 python scripts/dogfood_real_model.py --work /tmp/dogfood -o evidence.json

Three parts, each independently meaningful:

* ports:    mlx-lm converts the original to each dtype; every port must pass with no false alarm.
* rounding: weights are perturbed so they are not representable in bf16 (like a float32 checkpoint),
            so the rounding noise of a low-precision port is real; clean ports must still pass.
* faults:   each fault goes into one mid-stack layer; the verifier must name that exact module.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

os.environ.setdefault('MLX_ENABLE_TF32', '0')  # MLX 0.32 reads this once per process
PROMPTS = ['The capital of France is', 'def fibonacci(n):', 'Explain why the sky is blue in one sentence.',
           '한국의 수도는 어디인가요? 그리고 인구는 얼마나 되나요?', '1, 1, 2, 3, 5, 8, 13, 21,']
PLAIN = {'f32': 'float32', 'bf16': 'bfloat16', 'f16': 'float16'}
QUANTIZED = {'4bit': (4, 'float32'), '4bit-bf16': (4, 'bfloat16'), '8bit': (8, 'float32')}
MAX_TOKENS = 48


def cumlx(*argv: str) -> dict:
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '5'],
                       capture_output=True, text=True, check=False)
    doc = json.loads(p.stdout)
    assert doc['exit_code'] == p.returncode, 'exit code and envelope disagree'
    return doc


def convert(hf: str, out: Path, dtype: str, bits: int | None = None) -> None:
    if out.exists():
        return
    cmd = [sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', hf, '--mlx-path', str(out), '--dtype', dtype]
    if bits:
        cmd += ['-q', '--q-bits', str(bits)]
    subprocess.run(cmd, check=True, capture_output=True)


def capture(hf: str, out: Path, prompts: Path, weights_from: Path | None = None) -> dict:
    argv = ['capture', hf, '--prompts-file', str(prompts), '-o', str(out), '--max-tokens', str(MAX_TOKENS), '--overwrite']
    if weights_from:
        argv += ['--weights-from-mlx', str(weights_from)]
    return cumlx(*argv)


def row(doc: dict, **extra) -> dict:
    d = doc['data']
    s, t = d['summary'], d['threshold']
    fd = d.get('first_divergence')
    return {**extra, 'verdict': d['verdict'], 'exit_code': doc['exit_code'], 'coverage': s['parametrized_coverage'],
            'records': s['records'], 'pass': s['pass'], 'fail': s['fail'], 'unverified': s['unverified'],
            'chain_drift': s['chain_drift'], 'worst_isolated_rel_l2': s['worst_isolated_rel_l2'],
            'worst_chain_rel_l2': s['worst_chain_rel_l2'], 'native_output': s['native_output'],
            'median_leaf_error': t.get('median_leaf_error'), 'leaf_rel_l2': t.get('leaf_rel_l2'),
            'composite_rel_l2': t.get('composite_rel_l2'), 'warnings': [w['code'] for w in doc['warnings']],
            'first_divergence': fd and {k: fd[k] for k in ('module', 'basis', 'confidence')},
            'measured_gemm_noise': d['execution']['gemm_noise_rel_l2'], 'elapsed_ms': doc['elapsed_ms']}


def ports(hf: str, key: str, work: Path, prompts: Path, devices: list[str], public: list[str]) -> list[dict]:
    rows = []
    golden = work / f'{key}-golden'
    capture(hf, golden, prompts)
    for name, dtype in PLAIN.items():
        path = work / f'{key}-{name}'
        convert(hf, path, dtype)
        w = cumlx('weights', hf, str(path))
        for device in devices:
            rows.append(row(cumlx('diff', str(golden), str(path), '--device', device), model=hf, port=name,
                            golden='reference', device=device, weights_verdict=w['data']['verdict']))
    for name, (bits, dtype) in QUANTIZED.items():
        path = work / f'{key}-{name}'
        convert(hf, path, dtype, bits)
        w = cumlx('weights', hf, str(path))
        g = work / f'{key}-{name}-golden'
        capture(hf, g, prompts, path)
        for device in devices:
            rows.append(row(cumlx('diff', str(g), str(path), '--device', device), model=hf, port=name,
                            golden='port-weights', device=device, weights_verdict=w['data']['verdict']))
    for repo in public:
        w = cumlx('weights', hf, repo)
        for device in devices:
            rows.append(row(cumlx('diff', str(golden), repo, '--device', device), model=hf, port=repo,
                            golden='reference', device=device, weights_verdict=w['data']['verdict']))
    return rows


def rounding(hf: str, key: str, work: Path, devices: list[str]) -> list[dict]:
    import mlx.core as mx
    import torch
    from transformers import AutoTokenizer

    from cumlx.verify import diff, mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    torch.manual_seed(0)
    model = load_hf_model(hf)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.mul_(1 + 1e-2 * torch.randn_like(p))
    weights = {k: v.numpy() for k, v in model.state_dict().items() if 'inv_freq' not in k}
    tok = AutoTokenizer.from_pretrained(hf)
    samples = [np.asarray(tok(p)['input_ids'][:MAX_TOKENS]) for p in PROMPTS]
    g = prepare_output_dir(work / f'{key}-origin32-golden', overwrite=True)
    w = TraceWriter(g, source={'framework': 'torch', 'model': hf + ' (weights perturbed by 1%)'},
                    golden={'kind': 'reference', 'dtype': 'float32'}, max_bytes=1 << 31)
    capture_model(model, samples, w)
    w.finish()
    rows = []
    for name, dt in (('f16', mx.float16), ('bf16', mx.bfloat16)):
        for device in devices:
            port = mlx_port.load_port(str(work / f'{key}-f32'))
            port.load_weights([(k, mx.array(v)) for k, v in weights.items() if k != 'lm_head.weight'], strict=False)
            port.set_dtype(dt)
            r = diff.run_diff(str(g), name, model=port, device=device)
            t = r['threshold']
            rows.append({'model': hf, 'port': name, 'device': device, 'verdict': r['verdict'],
                         'worst_isolated_rel_l2': r['summary']['worst_isolated_rel_l2'],
                         'median_leaf_error': t['median_leaf_error'], 'leaf_rel_l2': t['leaf_rel_l2'],
                         'composite_rel_l2': t['composite_rel_l2'], 'warnings': [x['code'] for x in r['warnings']]})
    return rows


def fault_suite(hf: str, key: str, work: Path, layer: int) -> list[dict]:
    import mlx.nn as nn
    import torch  # noqa: F401 - loaded before the helpers below for a clear import error

    from transformers import AutoTokenizer

    from cumlx.errors import CumlxError
    from cumlx.verify import diff, faults, mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    a = f'model.layers.{layer}'
    targets = {'F1': f'{a}.self_attn.q_proj', 'F2': f'{a}.self_attn.k_proj', 'F3': f'{a}.input_layernorm',
               'F6': f'{a}.mlp', 'F7': f'{a}.self_attn', 'F8': f'{a}.self_attn'}
    golden = str(work / f'{key}-golden')
    rows = []

    def record(label, fid, target, rep):
        fd = rep['first_divergence']
        rows.append({'port': label, 'fault': fid, 'target': target, 'located': fd and fd['module'],
                     'basis': fd and fd['basis'], 'confidence': fd and fd['confidence'],
                     'rel_l2': fd and fd['metrics'].get('rel_l2'),
                     'ok': rep['verdict'] == 'diverged' and bool(fd) and fd['module'] == target and fd['basis'] == 'isolated'})

    for label in PLAIN:
        path = str(work / f'{key}-{label}')
        for fid, target in targets.items():
            m = mlx_port.load_port(path)
            try:
                faults.inject(m, fid, target)
            except CumlxError as exc:  # e.g. F2 on a model without attention bias: neither a pass nor a miss
                rows.append({'port': label, 'fault': fid, 'target': target, 'ok': None, 'not_applicable': exc.message})
                continue
            record(label, fid, target, diff.run_diff(golden, path, model=m))
    tok = AutoTokenizer.from_pretrained(hf)
    samples = [np.asarray(tok(p)['input_ids'][:MAX_TOKENS]) for p in PROMPTS]
    for fid, mode, gs, target in (('F4', 'mxfp4', 32, f'{a}.mlp.up_proj'), ('F5', 'nvfp4', 16, f'{a}.mlp.down_proj'),
                                  ('F10', 'nvfp4', 16, f'{a}.self_attn.v_proj'), ('F10', 'mxfp4', 32, f'{a}.self_attn.v_proj')):
        def port():
            m = mlx_port.load_port(str(work / f'{key}-f32'))
            nn.quantize(m, group_size=gs, bits=4, mode=mode, class_predicate=lambda p, x: isinstance(x, nn.Linear))
            return m
        clean = port()
        torch_model = load_hf_model(hf)
        load_mlx_weights_into(torch_model, mlx_port.dequantized_weights(clean))
        g = prepare_output_dir(work / f'{key}-{mode}-golden', overwrite=True)
        w = TraceWriter(g, source={'framework': 'torch'}, golden={'kind': 'quantization-aware', 'dtype': 'float32'},
                        max_bytes=1 << 31)
        capture_model(torch_model, samples, w)
        w.finish()
        rep = diff.run_diff(str(g), mode, model=clean)
        assert rep['verdict'] == 'ok', f'clean {mode} port: {rep["first_divergence"]}'
        bad = port()
        faults.inject(bad, fid, target)
        record(mode, fid, target, diff.run_diff(str(g), mode, model=bad))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--output', required=True, help='evidence JSON to write')
    ap.add_argument('--work', help='work directory (kept); default: a temporary directory')
    ap.add_argument('--model', action='append', help='HF id=fault-layer, repeatable')
    ap.add_argument('--public-port', action='append', default=[], help='public MLX port of the first model')
    ap.add_argument('--devices', default='gpu,cpu')
    args = ap.parse_args()
    models = args.model or ['Qwen/Qwen2.5-0.5B=11', 'HuggingFaceTB/SmolLM2-135M-Instruct=14']
    devices = args.devices.split(',')

    def run(work: Path) -> dict:
        prompts = work / 'prompts.txt'
        prompts.write_text('\n'.join(PROMPTS) + '\n', encoding='utf-8')
        out = {'schema': 'cumlx.dogfood.v1', 'ports': [], 'rounding': [], 'faults': []}
        for i, spec in enumerate(models):
            hf, _, layer = spec.partition('=')
            key = f'm{i}'
            out['ports'] += ports(hf, key, work, prompts, devices, args.public_port if i == 0 else [])
            out['rounding'] += rounding(hf, key, work, devices)
            out['faults'] += [{'model': hf, **r} for r in fault_suite(hf, key, work, int(layer or 1))]
            for extra in work.glob(f'{key}-*-golden'):
                shutil.rmtree(extra, ignore_errors=True)
        return out

    if args.work:
        Path(args.work).mkdir(parents=True, exist_ok=True)
        evidence = run(Path(args.work))
    else:
        with TemporaryDirectory(prefix='cumlx-dogfood-') as tmp:
            evidence = run(Path(tmp))
    import mlx.core as mx
    import mlx_lm
    import torch
    import transformers
    evidence['environment'] = {
        'platform': platform.platform(), 'python': platform.python_version(), 'mlx': mx.__version__,
        'mlx_lm': mlx_lm.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': mx.device_info().get('device_name') if hasattr(mx, 'device_info') and mx.metal.is_available() else None}
    false_alarms = [r for r in evidence['ports'] + evidence['rounding'] if r['verdict'] != 'ok']
    missed = [r for r in evidence['faults'] if r['ok'] is False]
    evidence['summary'] = {'clean_runs': len(evidence['ports']) + len(evidence['rounding']),
                           'false_alarms': len(false_alarms),
                           'fault_runs': sum(r['ok'] is not None for r in evidence['faults']), 'missed': len(missed),
                           'not_applicable': sum(r['ok'] is None for r in evidence['faults'])}
    Path(args.output).write_text(json.dumps(evidence, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(evidence['summary']))
    return 1 if false_alarms or missed else 0


if __name__ == '__main__':
    raise SystemExit(main())
