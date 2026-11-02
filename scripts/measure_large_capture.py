#!/usr/bin/env python3
"""Measure the time and disk use of capture on a large model, with --layers.

The weights are random: only the config is downloaded. Trace size and capture time depend on the architecture, so
this answers the size question without a large download. It says nothing about the accuracy of a real checkpoint.

    python scripts/measure_large_capture.py Qwen/Qwen2.5-7B --layers 0,13,27 --work /tmp/large -o result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from cumlx.verify.capture_torch import capture_model
from cumlx.verify.trace import TraceWriter, prepare_output_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model')
    ap.add_argument('--layers', default='0,13,27')
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--samples', type=int, default=3)
    ap.add_argument('--work', required=True)
    ap.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32',
                    help='parameter dtype. bfloat16 needs half the memory, and CPU arithmetic is usually slower')
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(',')]
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_config(config, dtype=getattr(torch, args.dtype)).eval()
    build = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(0)
    samples = [rng.integers(0, config.vocab_size, size=args.tokens) for _ in range(args.samples)]
    out = prepare_output_dir(Path(args.work) / 'golden', overwrite=True)
    writer = TraceWriter(out, source={'framework': 'torch', 'model': args.model, 'weights': 'random'},
                         golden={'kind': 'reference', 'dtype': 'float32'}, max_bytes=20 * 1024**3)
    t1 = time.perf_counter()
    stats = capture_model(model, samples, writer, layers=layers, decode_steps=1)
    writer.finish()
    capture_s = time.perf_counter() - t1
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    result = {'schema': 'cumlx.large-capture.v1', 'model': args.model, 'weights': 'random, architecture only',
              'parameters': int(params), 'layers_recorded': layers, 'num_hidden_layers': config.num_hidden_layers,
              'hidden_size': config.hidden_size, 'samples': args.samples, 'tokens_per_sample': args.tokens, 'decode_steps': 1,
              'records': stats['records'], 'trace_bytes': size, 'capture_seconds': round(capture_s, 1),
              'model_build_seconds': round(build, 1), 'device': 'cpu', 'dtype': args.dtype,
              'machine': platform.platform(), 'torch': torch.__version__}
    shutil.rmtree(out)
    Path(args.output).write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
