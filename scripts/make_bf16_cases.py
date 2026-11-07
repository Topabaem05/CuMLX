#!/usr/bin/env python3
"""Write validation/kernel-cases-bf16: the inputs of bf16_convert.cu and bf16_arith.cu, for `cumlx kernel capture`.

cases.json holds bf16_convert only, which sm70 runs. cases-sm80.json holds bf16_convert and bf16_arith: the arithmetic
of cuda_bf16.h needs sm80 or later.

bf16_convert:

65536 float32 bit patterns: every sign and exponent with the mantissas where rounding changes (the low 16 bits 0, 1,
0x7FFF, 0x8000, 0x8001 and 0xFFFF, under upper bits 0, 1, 0x3F, 0x40, 0x7E and 0x7F), and the rest at random. That covers
zeros, subnormals, the largest finite values, infinities and NaNs.

bf16_arith, 262144 triples (a, b, c) of bfloat16 bit patterns in four blocks of 65536:
1. a takes every bit pattern, b every bit pattern in random order, c at random.
2. a takes every bit pattern, b and c cycle through SPECIAL (zeros, subnormals, 1, the largest finite, inf, NaN).
3. a + b cancels: b is -a moved by -2 to 2 steps.
4. a * b + c cancels: c is -(a * b rounded) moved by -2 to 2 steps.
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cumlx.verify.safetensors_io import save_file  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'validation' / 'kernel-cases-bf16'
N = 65536
M = 4 * N
SPECIAL = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x3F81, 0x7F7F, 0xFF7F,
                    0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x0040, 0x3B80, 0x4380], np.uint16)


def inputs(seed: int = 20261107) -> np.ndarray:
    top = np.arange(512, dtype=np.uint32) << np.uint32(23)
    upper = np.array([0, 1, 0x3F, 0x40, 0x7E, 0x7F], np.uint32) << np.uint32(16)
    low = np.array([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF], np.uint32)
    grid = (top[:, None, None] | upper[None, :, None] | low[None, None, :]).ravel()
    rest = np.random.default_rng(seed).integers(0, 1 << 32, N - grid.size, dtype=np.uint64).astype(np.uint32)
    return np.concatenate([grid, rest]).view(np.float32)


def arith_inputs(seed: int = 20261108) -> dict[str, np.ndarray]:
    from cumlx.verify import bf16

    rng = np.random.default_rng(seed)
    every = np.arange(N, dtype=np.uint32).astype(np.uint16)
    rand = lambda n: rng.integers(0, N, n, dtype=np.uint32).astype(np.uint16)  # noqa: E731
    step = lambda n: rng.integers(-2, 3, n).astype(np.int32)  # noqa: E731
    a1, b1, c1 = every, rng.permutation(every), rand(N)
    a2, b2, c2 = every, SPECIAL[np.arange(N) % SPECIAL.size], SPECIAL[(np.arange(N) // SPECIAL.size) % SPECIAL.size]
    a3 = rand(N)
    b3, c3 = ((a3.astype(np.int32) ^ 0x8000) + step(N)).astype(np.uint16), rand(N)
    a4, b4 = rand(N), rand(N)
    with np.errstate(over='ignore', invalid='ignore'):
        product = bf16.from_float_bits((bf16.bits_to_float(a4) * bf16.bits_to_float(b4)).view(np.uint32), 'rn')
    c4 = ((product.astype(np.int32) ^ 0x8000) + step(N)).astype(np.uint16)
    return {'a': np.concatenate([a1, a2, a3, a4]), 'b': np.concatenate([b1, b2, b3, b4]), 'c': np.concatenate([c1, c2, c3, c4])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    save_file({'x': inputs()}, OUT / 'inputs.safetensors')
    outputs = [{'name': n, 'dtype': 'bfloat16', 'shape': [N]} for n in ('rn', 'rz', 'rd', 'ru')]
    doc = {'schema': 'cumlx.kernel-cases.v1', 'cases': [
        {'id': 'bf16_convert.65536', 'kernel': 'bf16_convert',
         'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': [N], 'file': 'inputs.safetensors'}],
         'outputs': outputs, 'grid': [N, 1, 1], 'threadgroup': [256, 1, 1]}]}
    (OUT / 'cases.json').write_text(json.dumps(doc, indent=1) + '\n')
    save_file(arith_inputs(), OUT / 'arith-inputs.safetensors')
    arith = {'id': f'bf16_arith.{M}', 'kernel': 'bf16_arith',
             'inputs': [{'name': n, 'dtype': 'bfloat16', 'shape': [M], 'file': 'arith-inputs.safetensors'} for n in 'abc'],
             'outputs': [{'name': n, 'dtype': 'bfloat16', 'shape': [M]} for n in ('add', 'sub', 'mul', 'fma', 'neg', 'abs')],
             'grid': [M, 1, 1], 'threadgroup': [256, 1, 1]}
    (OUT / 'cases-sm80.json').write_text(json.dumps({**doc, 'cases': [*doc['cases'], arith]}, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
