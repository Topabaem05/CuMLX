#!/usr/bin/env python3
"""Confirm a config_scan.py candidate on the published port itself, and prove the cause with a control.

    MLX_ENABLE_TF32=0 python validation/upstream/port_rope_check.py ORIGINAL PORT -o result.json [--work DIR]

1. Download the original's config.json only, and the whole published MLX port. When ORIGINAL is the port itself (a port
   whose original is not named), the port's config without its quantization entries stands for the original's.
2. Build the original architecture in transformers from the original's config (from_config), and load every
   dequantized weight of the port into it. Loading refuses a missing weight, an unused weight and a shape mismatch.
   This is the golden of `cumlx capture --weights-from-mlx`, without downloading the original's weights.
3. `cumlx diff` the golden against the port as published.
4. Control: the same port, with only a top-level `rope_theta` (and `rope_scaling`) copied from `rope_parameters` into
   its config.json. If the port diverges and the control is ok, the RoPE setting is the cause.

The weights in the golden are the port's, so this checks the port's code and config, not its quantization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault('MLX_ENABLE_TF32', '0')
PROMPTS = ['The capital of France is', 'def fibonacci(n):']
CLEAN = re.compile(r"""/(?:private/)?var/folders/[^\s"']+|/(?:private/)?tmp/[^\s"']+|/Users/[^/\s"']+""")


def cumlx(*argv: str) -> dict:
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '5'],
                       capture_output=True, text=True, check=False)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'exit_code': p.returncode, 'status': 'error', 'summary': p.stderr[-600:]}


def brief(doc: dict) -> dict:
    data = doc.get('data') or {}
    summary = data.get('summary') or {}
    fd = data.get('first_divergence') or {}
    return {'exit_code': doc.get('exit_code'), 'status': doc.get('status'), 'summary': doc.get('summary'),
            'first_divergence': {k: fd.get(k) for k in ('module', 'basis', 'confidence', 'phase', 'metrics')} if fd else None,
            'native_output': summary.get('native_output'), 'worst_isolated_rel_l2': summary.get('worst_isolated_rel_l2'),
            'warnings': [w.get('code') for w in doc.get('warnings', [])],
            'errors': [{'code': e.get('code'), 'message': str(e.get('message'))[:300]} for e in doc.get('errors', [])]}


def capture(original_config: dict, port_dir: Path, out: Path) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from cumlx.verify import mlx_port
    from cumlx.verify.capture_torch import capture_model, load_mlx_weights_into
    from cumlx.verify.trace import TraceWriter, prepare_output_dir

    cfg_dir = out.parent / 'original-config'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    plain = {k: v for k, v in original_config.items() if k not in ('quantization', 'quantization_config')}
    (cfg_dir / 'config.json').write_text(json.dumps(plain))
    config = AutoConfig.from_pretrained(cfg_dir)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    port = mlx_port.load_port(str(port_dir))
    info = load_mlx_weights_into(model, mlx_port.dequantized_weights(port))
    del port
    tok = AutoTokenizer.from_pretrained(port_dir)
    import numpy as np
    samples = [np.asarray(tok(p)['input_ids'], dtype=np.int64) for p in PROMPTS]
    source = {'framework': 'torch', 'framework_version': torch.__version__, 'transformers_version': metadata.version('transformers'),
              'model': 'original config + port weights', 'architecture': type(model).__name__, 'device': 'cpu',
              'inputs': [{'kind': 'prompt', 'text': p} for p in PROMPTS]}
    golden = {'kind': 'quantization-aware', 'dtype': 'float32', 'store_dtype': 'float32',
              'weights_from_mlx': {'loaded': info['loaded'], 'port': str(port_dir)}}
    writer = TraceWriter(prepare_output_dir(out, overwrite=True), source=source, golden=golden, max_bytes=8 << 30)
    stats = capture_model(model, samples, writer, decode_steps=1)
    writer.finish()
    return {'architecture': type(model).__name__, 'rope_theta_in_transformers': getattr(config, 'rope_parameters', None)
            or getattr(config, 'rope_theta', None), 'weights_loaded': info['loaded'], 'records': stats['records']}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original')
    ap.add_argument('port')
    ap.add_argument('-o', '--output', type=Path, required=True)
    ap.add_argument('--work', type=Path)
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    work = Path(tempfile.mkdtemp(prefix='cumlx-rope-', dir=args.work))
    try:
        api = HfApi()
        original_rev = api.model_info(args.original).sha
        port_rev = api.model_info(args.port).sha
        original_config = json.loads(Path(hf_hub_download(args.original, 'config.json', revision=original_rev,
                                                          cache_dir=work / 'hf')).read_text())
        port_dir = Path(snapshot_download(args.port, revision=port_rev, cache_dir=work / 'hf'))
        port_config = json.loads((port_dir / 'config.json').read_text())
        ignore = {'quantization', 'quantization_config', 'transformers_version', 'torch_dtype', 'dtype'}
        config_differences = sorted(k for k in set(original_config) | set(port_config) if k not in ignore
                                    and original_config.get(k) != port_config.get(k))
        row = {'schema': 'cumlx.port-rope-check.v1', 'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
               'original': args.original, 'original_revision': original_rev, 'port': args.port, 'port_revision': port_rev,
               'original_transformers_version': original_config.get('transformers_version'),
               'rope_parameters': port_config.get('rope_parameters'), 'top_level_rope_theta': port_config.get('rope_theta'),
               'config_differences_original_vs_port': config_differences,
               'versions': {p: metadata.version(p) for p in ('mlx', 'mlx-lm', 'torch', 'transformers')}}
        from mlx_lm.utils import load_model
        mlx_model, _ = load_model(port_dir)
        row['rope_base_in_mlx'] = next((getattr(m.rope, 'base', None) or getattr(m.rope, '_base', None)
                                        for _, m in mlx_model.named_modules() if hasattr(m, 'rope')), None)
        del mlx_model
        row['golden'] = capture(original_config, port_dir, work / 'golden')
        row['port_as_published'] = brief(cumlx('diff', str(work / 'golden'), str(port_dir)))

        control = work / 'control'
        control.mkdir()
        for f in port_dir.iterdir():
            if f.name != 'config.json':
                (control / f.name).symlink_to(f.resolve())
        fixed = dict(port_config)
        rp = fixed.get('rope_parameters') or {}
        fixed['rope_theta'] = rp.get('rope_theta')
        if rp.get('rope_type') not in (None, 'default'):
            fixed['rope_scaling'] = rp
        (control / 'config.json').write_text(json.dumps(fixed, indent=1))
        row['control_top_level_rope_theta'] = {'added': {k: fixed[k] for k in ('rope_theta', 'rope_scaling') if k in fixed},
                                               **brief(cumlx('diff', str(work / 'golden'), str(control)))}
        row['confirmed'] = row['port_as_published']['exit_code'] == 1 and row['control_top_level_rope_theta']['exit_code'] == 0
        args.output.write_text(CLEAN.sub('<local>', json.dumps(row, indent=1, ensure_ascii=False)) + '\n')
        print(json.dumps({'port': args.port, 'confirmed': row['confirmed'],
                          'published': row['port_as_published']['summary'],
                          'control': row['control_top_level_rope_theta']['summary']}, ensure_ascii=False))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
