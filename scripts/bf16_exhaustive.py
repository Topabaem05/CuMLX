#!/usr/bin/env python3
"""Compare the Metal header of cumlx.verify.bf16 with its model on every input, on this Mac's GPU.

    python scripts/bf16_exhaustive.py -o validation/bf16-exhaustive.json

- The four conversions from float, on all 2^32 float32 bit patterns, against cuda_bf16.hpp (CUDA 12.0, sm70 path).
- __hadd, __hsub and __hmul on all 2^32 pairs of bf16 bit patterns, against one rounding of the float64 result.
- __hfma on --fma-samples random triples and on the special values, against exact rational arithmetic.

Non-NaN results must be equal to the bit. NaN results must be NaN. The model is not a CUDA run: it says what the PTX
ISA and cuda_bf16.hpp specify, and only an NVIDIA GPU can say what one computes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cumlx.verify import bf16  # noqa: E402

CHUNK = 1 << 24
SPECIAL = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x8080, 0x00FF, 0x3F80, 0xBF80, 0x3F81, 0x4000, 0x7F7F, 0xFF7F,
                    0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x7FFF, 0x0040, 0x3B80, 0x0180, 0x4380], np.uint16)


def kernel(mx, name: str, inputs: list[str], body: str):
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=['y'], source=body, header=bf16.METAL_HEADER,
                                ensure_row_contiguous=True, compile_options={'math_mode': 'safe'})


def run(k, mx, arrays, n):
    (y,) = k(inputs=[mx.array(a) for a in arrays], grid=(n, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(n,)],
             output_dtypes=[mx.uint16])
    return np.array(y)


def compare(port: np.ndarray, model: np.ndarray) -> tuple[int, int, int]:
    """(bits that differ where the model is not NaN, NaN class mismatches, NaN payload differences)."""
    pn, mn = bf16.is_nan(port), bf16.is_nan(model)
    return int(((port != model) & ~mn).sum()), int((pn != mn).sum()), int(((port != model) & mn & pn).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--out', type=Path, required=True)
    parser.add_argument('--fma-samples', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    from importlib import metadata

    import mlx.core as mx
    record: dict = {'schema': 'cumlx.bf16_exhaustive.v1', 'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                    'device': mx.device_info(mx.gpu).get('device_name') if hasattr(mx, 'device_info') else None,
                    'machine': platform.platform(), 'mlx': metadata.version('mlx'), 'math_mode': 'safe',
                    'header_sha256': __import__('hashlib').sha256(bf16.METAL_HEADER.encode()).hexdigest(),
                    'model': 'conversions: cuda_bf16.hpp of CUDA 12.0 (sm70 path); add, sub, mul, fma: the exact result rounded once '
                             'to nearest even with subnormals kept (PTX ISA fma.rn.bf16). Not a CUDA run.',
                    'results': {}}
    t0 = time.time()
    conv = {m: kernel(mx, f'cumlx_bf16_cv_{m}', ['x'], f'uint i = thread_position_in_grid.x; y[i] = cumlx_bf16_{m}_bits(x[i]);')
            for m in ('rn', 'rz', 'rd', 'ru')}
    for mode, k in conv.items():
        totals = [0, 0, 0]
        for start in range(0, 1 << 32, CHUNK):
            x = np.arange(start, start + CHUNK, dtype=np.uint64).astype(np.uint32)
            got = compare(run(k, mx, [x], CHUNK), bf16.from_float_bits(x, mode))
            totals = [a + b for a, b in zip(totals, got)]
        record['results'][f'float2bfloat16_{mode}'] = {'inputs': 1 << 32, 'bits_differ': totals[0], 'nan_class_differs': totals[1],
                                                       'nan_payload_differs': totals[2]}
        print(mode, record['results'][f'float2bfloat16_{mode}'], f'{time.time() - t0:.0f}s', flush=True)
    ops = {op: kernel(mx, f'cumlx_bf16_op_{op}', ['a', 'b'],
                      'uint i = thread_position_in_grid.x; '
                      + {'add': 'y[i] = cumlx_bf16_fma_bits(a[i], 0x3f80u, b[i]);', 'sub': 'y[i] = cumlx_bf16_fma_bits(b[i], 0xbf80u, a[i]);',
                         'mul': 'y[i] = cumlx_bf16_fma_bits(a[i], b[i], 0x8000u);'}[op]) for op in ('add', 'sub', 'mul')}
    rows = CHUNK >> 16
    every = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    for op, k in ops.items():
        totals = [0, 0, 0]
        for a0 in range(0, 1 << 16, rows):
            a = np.repeat(np.arange(a0, a0 + rows, dtype=np.uint32).astype(np.uint16), 1 << 16)
            b = np.tile(every, rows)
            got = compare(run(k, mx, [a, b], CHUNK), bf16.model_op(op, a, b))
            totals = [x + y for x, y in zip(totals, got)]
        record['results'][f'h{op}'] = {'pairs': 1 << 32, 'bits_differ': totals[0], 'nan_class_differs': totals[1], 'nan_payload_differs': totals[2]}
        print(op, record['results'][f'h{op}'], f'{time.time() - t0:.0f}s', flush=True)
    # Controls: two ways to do it that are not bit-exact. Each must differ, or this check could not see a difference.
    native = kernel(mx, 'cumlx_bf16_ctl_native', ['x'],
                    'uint i = thread_position_in_grid.x; y[i] = as_type<ushort>(bfloat(as_type<float>(x[i])));')
    differ = 0
    for start in range(0, 1 << 32, CHUNK):
        x = np.arange(start, start + CHUNK, dtype=np.uint64).astype(np.uint32)
        differ += compare(run(native, mx, [x], CHUNK), bf16.from_float_bits(x, 'rn'))[0]
    record['controls'] = {'native_bfloat_conversion': {'what': 'bfloat(float) of MSL', 'inputs': 1 << 32, 'bits_differ': differ}}
    print('control native', differ, flush=True)
    through_float = kernel(mx, 'cumlx_bf16_ctl_float', ['a', 'b'],
                           'uint i = thread_position_in_grid.x; '
                           'y[i] = cumlx_bf16_rn_bits(as_type<uint>(as_type<float>(uint(a[i]) << 16u) + as_type<float>(uint(b[i]) << 16u)));')
    differ = 0
    for a0 in range(0, 1 << 16, rows):
        a = np.repeat(np.arange(a0, a0 + rows, dtype=np.uint32).astype(np.uint16), 1 << 16)
        b = np.tile(every, rows)
        differ += compare(run(through_float, mx, [a, b], CHUNK), bf16.model_op('add', a, b))[0]
    record['controls']['add_through_metal_float'] = {'what': 'float add on Metal, then round to nearest even', 'pairs': 1 << 32, 'bits_differ': differ}
    print('control float add', differ, flush=True)
    rng = np.random.default_rng(args.seed)
    triples = rng.integers(0, 1 << 16, size=(3, args.fma_samples), dtype=np.uint32).astype(np.uint16)
    special = np.array(np.meshgrid(SPECIAL, SPECIAL, SPECIAL)).reshape(3, -1)
    # Values near each other, so that a*b and c cancel: the cases where one rounding and two differ.
    near = rng.integers(0, 1 << 16, size=(2, args.fma_samples // 4), dtype=np.uint32).astype(np.uint16)
    product = bf16.from_float_bits((bf16.bits_to_float(near[0]) * bf16.bits_to_float(near[1])).view(np.uint32), 'rn')
    c_near = (product ^ np.uint16(0x8000)) + rng.integers(-2, 3, size=product.shape).astype(np.uint16)
    triples = np.concatenate([triples, special, np.stack([near[0], near[1], c_near])], axis=1)
    n = triples.shape[1]
    k = kernel(mx, 'cumlx_bf16_fma', ['a', 'b', 'c'], 'uint i = thread_position_in_grid.x; if (i < n) y[i] = cumlx_bf16_fma_bits(a[i], b[i], c[i]);'.replace('n)', f'{n}u)'))
    padded = -(-n // 256) * 256
    pad = [np.concatenate([t, np.zeros(padded - n, np.uint16)]) for t in triples]
    got = run(k, mx, pad, padded)[:n]
    model = bf16.model_fma(*triples)
    differ, nan_class, payload = compare(got, model)
    record['results']['hfma'] = {'triples': n, 'random': args.fma_samples, 'special_values': int(special.shape[1]),
                                 'near_cancellation': int(near.shape[1]), 'seed': args.seed, 'bits_differ': differ,
                                 'nan_class_differs': nan_class, 'nan_payload_differs': payload}
    print('fma', record['results']['hfma'], f'{time.time() - t0:.0f}s', flush=True)
    record['seconds'] = round(time.time() - t0)
    record['pass'] = (all(r['bits_differ'] == 0 and r['nan_class_differs'] == 0 for r in record['results'].values())
                      and all(c['bits_differ'] > 0 for c in record['controls'].values()))
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'pass': record['pass']}))
    return 0 if record['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
