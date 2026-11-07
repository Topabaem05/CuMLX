#!/usr/bin/env python3
"""Write validation/kernel-cases-bf16: the inputs of bf16_convert.cu, for `cumlx kernel capture` on an NVIDIA machine.

65536 float32 bit patterns: every sign and exponent with the mantissas where rounding changes (the low 16 bits 0, 1,
0x7FFF, 0x8000, 0x8001 and 0xFFFF, under upper bits 0, 1, 0x3F, 0x40, 0x7E and 0x7F), and the rest at random. That covers
zeros, subnormals, the largest finite values, infinities and NaNs.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cumlx.verify.safetensors_io import save_file  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'validation' / 'kernel-cases-bf16'
N = 65536


def inputs(seed: int = 20261107) -> np.ndarray:
    top = np.arange(512, dtype=np.uint32) << np.uint32(23)
    upper = np.array([0, 1, 0x3F, 0x40, 0x7E, 0x7F], np.uint32) << np.uint32(16)
    low = np.array([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF], np.uint32)
    grid = (top[:, None, None] | upper[None, :, None] | low[None, None, :]).ravel()
    rest = np.random.default_rng(seed).integers(0, 1 << 32, N - grid.size, dtype=np.uint64).astype(np.uint32)
    return np.concatenate([grid, rest]).view(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    save_file({'x': inputs()}, OUT / 'inputs.safetensors')
    outputs = [{'name': n, 'dtype': 'bfloat16', 'shape': [N]} for n in ('rn', 'rz', 'rd', 'ru')]
    doc = {'schema': 'cumlx.kernel-cases.v1', 'cases': [
        {'id': 'bf16_convert.65536', 'kernel': 'bf16_convert',
         'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': [N], 'file': 'inputs.safetensors'}],
         'outputs': outputs, 'grid': [N, 1, 1], 'threadgroup': [256, 1, 1]}]}
    (OUT / 'cases.json').write_text(json.dumps(doc, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
