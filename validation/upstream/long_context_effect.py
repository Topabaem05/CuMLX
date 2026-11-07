#!/usr/bin/env python3
"""Measure how a missing rope_parameters.rope_theta changes a published port's output as the input grows.

    MLX_ENABLE_TF32=0 python validation/upstream/long_context_effect.py PORT -o result.json

It loads the port twice with mlx-lm: as published, and with only a top-level rope_theta copied from rope_parameters
(the control that port_rope_check.py proved equal to the transformers golden). Both run on the same token ids, a
non-repeating English text from this repository (AGENTS.md, validation/REPORT.md) tokenized
by the port's own tokenizer, and the script reports, per context length, the share of
positions where the top-1 next token differs and the relative L2 error of the logits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault('MLX_ENABLE_TF32', '0')
TEXT_FILES = ['AGENTS.md', 'validation/REPORT.md']  # non-repeating English text from this repository
LENGTHS = [16, 64, 256, 1024, 2048]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('port')
    ap.add_argument('-o', '--output', type=Path, required=True)
    ap.add_argument('--work', type=Path)
    args = ap.parse_args()
    import mlx.core as mx
    from huggingface_hub import HfApi, snapshot_download
    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    work = Path(tempfile.mkdtemp(prefix='cumlx-long-', dir=args.work))
    try:
        rev = HfApi().model_info(args.port).sha
        port_dir = Path(snapshot_download(args.port, revision=rev, cache_dir=work / 'hf'))
        config = json.loads((port_dir / 'config.json').read_text())
        control = work / 'control'
        control.mkdir()
        for f in port_dir.iterdir():
            if f.name != 'config.json':
                (control / f.name).symlink_to(f.resolve())
        (control / 'config.json').write_text(json.dumps({**config, 'rope_theta': config['rope_parameters']['rope_theta']}))
        tok = AutoTokenizer.from_pretrained(port_dir)
        root = Path(__file__).resolve().parents[2]
        git = ['git', '-C', str(root)]
        if subprocess.run([*git, 'diff', '--quiet', 'HEAD', '--', *TEXT_FILES]).returncode != 0:
            raise SystemExit(f'{TEXT_FILES} have uncommitted changes; the record names the commit the text came from')
        commit = subprocess.run([*git, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        text = '\n\n'.join((root / f).read_text(encoding='utf-8') for f in TEXT_FILES)
        ids = tok(text)['input_ids'][:max(LENGTHS)]
        assert len(ids) == max(LENGTHS), f'the text gives only {len(ids)} tokens'
        x = mx.array([ids])
        logits = {}
        for name, path in (('published', port_dir), ('control', control)):
            model, _ = load_model(path)
            out = model(x).astype(mx.float32)
            mx.eval(out)
            logits[name] = out
            del model
        rows = []
        a, b = logits['published'], logits['control']
        for n in LENGTHS:
            pa, pb = a[0, :n], b[0, :n]
            differ = float(mx.mean(mx.argmax(pa, -1) != mx.argmax(pb, -1)).item())
            tail = slice(n // 2, n)  # the second half, where positions are far from the start
            rel = float((mx.linalg.norm(pa[tail] - pb[tail]) / mx.linalg.norm(pb[tail])).item())
            rows.append({'tokens': n, 'top1_differs': round(differ, 4), 'logits_rel_l2_second_half': round(rel, 4)})
        doc = {'schema': 'cumlx.rope-long-context.v1', 'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
               'port': args.port, 'port_revision': rev, 'rope_parameters': config.get('rope_parameters'),
               'reference': 'the same port with a top-level rope_theta (equal to the transformers golden, see port_rope_check.py)',
               'text': TEXT_FILES, 'text_commit': commit, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'versions': {p: metadata.version(p) for p in ('mlx', 'mlx-lm')}, 'lengths': rows,
               'does_not_show': ['task quality on real data', 'other ports or prompts']}
        args.output.write_text(json.dumps(doc, indent=1) + '\n')
        print(json.dumps(rows))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
