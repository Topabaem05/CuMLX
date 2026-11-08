#!/usr/bin/env python3
"""Confirm a config_scan.py candidate on the published port itself, and prove the cause with a control.

    MLX_ENABLE_TF32=0 python validation/upstream/port_rope_check.py ORIGINAL PORT -o result.json [--work DIR] [--layers K [--vocab-from-prompts]]

1. Download the original's config.json only, and the whole published MLX port. When ORIGINAL is the port itself (a port
   whose original is not named), the port's config without its quantization entries stands for the original's.
2. Build the original architecture in transformers from the original's config (from_config), and load every
   dequantized weight of the port into it. Loading refuses a missing weight, an unused weight and a shape mismatch.
   This is the golden of `cumlx capture --weights-from-mlx`, without downloading the original's weights.
3. `cumlx diff` the golden against the port as published.
4. Control: the same port, with only a top-level `rope_theta` (and `rope_scaling`) copied from `rope_parameters` into
   its config.json. If the port diverges and the control is ok, the RoPE setting is the cause.

The weights in the golden are the port's, so this checks the port's code and config, not its quantization.

--layers K checks the first K decoder layers of a port too large for a float32 golden in memory. It downloads only the
shards that hold those layers, the embedding, the final norm and the head, and writes a port with them and
num_hidden_layers = K. The original's config gets the same K. Every step above then runs on that port, and the control
is the same cut port. A RoPE base defect shows in layer 0, so the cut can locate it. The verdict covers those K layers,
and the output metrics are those of the cut model, not of the published one.

--vocab-from-prompts also keeps only the rows of the embedding and the head for the token ids of the prompts, renumbered
from 0. A 4-bit row is quantized on its own, so the kept rows are unchanged, every activation equals that of the uncut
model, and the logits are those of the kept tokens. It is for a vocabulary whose float32 embedding and head do not fit
in memory beside the golden.
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
VOCAB = 'cumlx-vocab.json'  # in a cut port: the published token ids that the kept rows hold, in order
LAYER = re.compile(r'(?:^|\.)layers\.(\d+)\.')
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
    if (port_dir / VOCAB).exists():
        renumber = {t: i for i, t in enumerate(json.loads((port_dir / VOCAB).read_text())['kept_token_ids'])}
        samples = [np.asarray([renumber[int(t)] for t in s], dtype=np.int64) for s in samples]
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


def cut(config: dict, layers: int) -> dict:
    """The config of the first `layers` decoder layers."""
    total = config['num_hidden_layers']
    if not 0 < layers < total:
        raise SystemExit(f'--layers must be between 1 and {total - 1}')
    out = {**config, 'num_hidden_layers': layers}
    for key, value in config.items():  # per-layer lists, such as layer_types
        if isinstance(value, list) and len(value) == total:
            out[key] = value[:layers]
    return out


def cut_port(api, repo: str, rev: str, layers: int, work: Path, vocab_from_prompts: bool = False) -> Path:
    """Download only what the first `layers` decoder layers need, and write them as a port."""
    import mlx.core as mx
    from huggingface_hub import hf_hub_download

    keep = lambda key: (m := LAYER.search(key)) is None or int(m.group(1)) < layers  # noqa: E731
    names = [s.rfilename for s in api.model_info(repo, revision=rev).siblings]
    small = [n for n in names if '/' not in n and not n.endswith('.safetensors') and n != 'model.safetensors.index.json']
    weight_map = api.get_safetensors_metadata(repo, revision=rev).weight_map
    shards = sorted({f for k, f in weight_map.items() if keep(k)})
    get = lambda n: Path(hf_hub_download(repo, n, revision=rev, cache_dir=work / 'hf'))  # noqa: E731
    out = work / f'port-first-{layers}-layers'
    out.mkdir()
    for name in small:
        if name != 'config.json':
            (out / name).symlink_to(get(name).resolve())
    weights = {}
    for shard in shards:
        weights.update({k: v for k, v in mx.load(str(get(shard))).items() if keep(k)})
    config = cut(json.loads(get('config.json').read_text()), layers)
    if vocab_from_prompts:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(out)
        kept = sorted({int(t) for p in PROMPTS for t in tok(p)['input_ids']})
        rows = mx.array(kept)
        table = re.compile(r'(?:^|\.)(?:embed_tokens|lm_head)\.(?:weight|scales|biases)$')
        weights = {k: v[rows] if table.search(k) else v for k, v in weights.items()}
        (out / VOCAB).write_text(json.dumps({'kept_token_ids': kept, 'vocab_size_published': config['vocab_size']}))
        config['vocab_size'] = len(kept)
        for key in ('pad_token_id', 'bos_token_id', 'eos_token_id'):  # ids of the published vocabulary
            config[key] = None
    mx.save_safetensors(str(out / 'model.safetensors'), weights, metadata={'format': 'mlx'})
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('original')
    ap.add_argument('port')
    ap.add_argument('-o', '--output', type=Path, required=True)
    ap.add_argument('--work', type=Path)
    ap.add_argument('--layers', type=int, help='check only the first K decoder layers (see above)')
    ap.add_argument('--vocab-from-prompts', action='store_true', help='with --layers: keep only the prompt tokens')
    args = ap.parse_args()
    if args.vocab_from_prompts and not args.layers:
        ap.error('--vocab-from-prompts needs --layers')
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    work = Path(tempfile.mkdtemp(prefix='cumlx-rope-', dir=args.work))
    try:
        api = HfApi()
        original_rev = api.model_info(args.original).sha
        port_rev = api.model_info(args.port).sha
        original_config = json.loads(Path(hf_hub_download(args.original, 'config.json', revision=original_rev,
                                                          cache_dir=work / 'hf')).read_text())
        if args.layers:
            port_dir = cut_port(api, args.port, port_rev, args.layers, work, args.vocab_from_prompts)
            published = json.loads(Path(hf_hub_download(args.port, 'config.json', revision=port_rev,
                                                        cache_dir=work / 'hf')).read_text())
        else:
            port_dir = Path(snapshot_download(args.port, revision=port_rev, cache_dir=work / 'hf'))
            published = json.loads((port_dir / 'config.json').read_text())
        port_config = published
        ignore = {'quantization', 'quantization_config', 'transformers_version', 'torch_dtype', 'dtype'}
        config_differences = sorted(k for k in set(original_config) | set(port_config) if k not in ignore
                                    and original_config.get(k) != port_config.get(k))
        row = {'schema': 'cumlx.port-rope-check.v1', 'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
               'original': args.original, 'original_revision': original_rev, 'port': args.port, 'port_revision': port_rev,
               'original_transformers_version': original_config.get('transformers_version'),
               'rope_parameters': port_config.get('rope_parameters'), 'top_level_rope_theta': port_config.get('rope_theta'),
               'config_differences_original_vs_port': config_differences,
               'versions': {p: metadata.version(p) for p in ('mlx', 'mlx-lm', 'torch', 'transformers')}}
        if args.layers:
            row['layers'] = {'checked': args.layers, 'published': published['num_hidden_layers'],
                             'does_not_show': ['layers after the first K', 'the output of the whole port']}
            port_config = json.loads((port_dir / 'config.json').read_text())
            original_config = cut(original_config, args.layers)
            if args.vocab_from_prompts:
                kept = json.loads((port_dir / VOCAB).read_text())['kept_token_ids']
                row['vocabulary'] = {'kept_token_ids': kept, 'published': published['vocab_size'],
                                     'does_not_show': ['logits of the tokens that were not kept']}
                original_config = {**original_config, **{k: port_config[k] for k in
                                                         ('vocab_size', 'pad_token_id', 'bos_token_id', 'eos_token_id')}}
        from mlx_lm.utils import load_model
        mlx_model, _ = load_model(port_dir)
        row['rope_base_in_mlx'] = next((getattr(m.rope, 'base', None) or getattr(m.rope, '_base', None)
                                        for _, m in mlx_model.named_modules() if hasattr(m, 'rope')), None)
        del mlx_model
        row['golden'] = capture(original_config, port_dir, work / 'golden')
        row['port_as_published'] = brief(cumlx('diff', str(work / 'golden'), str(port_dir)))

        if not port_config.get('rope_parameters'):  # nothing to copy: the control does not apply
            row['control_top_level_rope_theta'] = {'skipped': 'the config has no rope_parameters'}
            row['confirmed'] = False
            args.output.write_text(CLEAN.sub('<local>', json.dumps(row, indent=1, ensure_ascii=False)) + '\n')
            print(json.dumps({'port': args.port, 'confirmed': False, 'published': row['port_as_published']['summary']}))
            return 0
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
