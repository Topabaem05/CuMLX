"""Write the cases of the 8 FP4 kernels for `cumlx kernel capture`: cases.json and inputs.safetensors.

The launch parameters follow `cumlx.mlx_backend`, so the same case runs on CUDA and on Metal. The inputs are pinned by seed.
Usage: python scripts/make_kernel_cases.py -o validation/kernel-cases
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cumlx.planning import plan_linear
from cumlx.quant import quantize
from cumlx.verify.safetensors_io import save_file

THREADS = 128


def grid_for(elements: int) -> list[int]:
    return [((elements + THREADS - 1) // THREADS) * THREADS, 1, 1]


class Builder:
    def __init__(self) -> None:
        self.cases: list[dict] = []
        self.tensors: dict[str, np.ndarray] = {}

    def add(self, case_id: str, kernel: str, inputs: dict[str, np.ndarray], outputs: dict[str, tuple[str, tuple[int, ...]]],
            constants: dict, grid: list[int]) -> None:
        specs = []
        for name, array in inputs.items():
            tensor = f'{case_id}/{name}'
            self.tensors[tensor] = array
            specs.append({'name': name, 'dtype': str(array.dtype), 'shape': list(array.shape),
                          'file': 'inputs.safetensors', 'tensor': tensor})
        self.cases.append({'id': case_id, 'kernel': kernel, 'inputs': specs,
                           'outputs': [{'name': k, 'dtype': d, 'shape': list(s)} for k, (d, s) in outputs.items()],
                           'constants': constants, 'grid': grid, 'threadgroup': [THREADS, 1, 1]})


def weight_params(q, prefix: str = 'W') -> dict:
    _, k = q.shape
    g = q.group_size
    return {prefix + 'P': (k + 1) // 2, prefix + 'SG': (k + g - 1) // g, prefix + 'G': g,
            prefix + 'F': 0 if q.format == 'nvfp4' else 1}


def weight_arrays(q, *, names=('wq', 'ws', 'wglobal')) -> dict[str, np.ndarray]:
    return {names[0]: q.payload.astype(np.uint8), names[1]: q.scales.astype(np.uint8),
            names[2]: np.array([q.global_scale], np.float32)}


def normal(rng, *shape, scale=1.0) -> np.ndarray:
    return (rng.normal(size=shape) * scale).astype(np.float32)


def add_decode(b: Builder, rng) -> None:
    for fmt, n, k in [('nvfp4', 7, 17), ('mxfp4', 5, 65)]:
        q = quantize(normal(rng, n, k), fmt)
        b.add(f'fp4_decode.{fmt}.{n}x{k}', 'fp4_decode', weight_arrays(q), {'y': ('float32', (n, k))},
              dict(N=n, K=k, **weight_params(q)), grid_for(n * k))
    # Every code byte with a plain scale. Then scales that decode to NaN: 127 in E4M3 and 255 in E8M0.
    payload = np.arange(256, dtype=np.uint8)[:, None]
    for fmt, scale, label, wf, wg in [('nvfp4', 64, 'allbytes', 0, 16), ('mxfp4', 128, 'allbytes', 1, 32),
                                      ('nvfp4', 127, 'nan-scale', 0, 16), ('mxfp4', 255, 'nan-scale', 1, 32)]:
        inputs = {'wq': payload, 'ws': np.full((256, 1), scale, np.uint8),
                  'wglobal': np.array([0.25 if fmt == 'nvfp4' else 1.0], np.float32)}
        b.add(f'fp4_decode.{fmt}.{label}', 'fp4_decode', inputs, {'y': ('float32', (256, 2))},
              dict(N=256, K=2, WP=1, WSG=1, WG=wg, WF=wf), grid_for(512))


def add_linear(b: Builder, rng) -> None:
    table = [('fp4_gemv', 'gemv', 'nvfp4', 1, 7, 17, True, False), ('fp4_gemv', 'gemv', 'mxfp4', 1, 33, 65, True, False),
             ('fp4_gemv', 'gemv', 'nvfp4', 1, 5, 601, False, True),  # K above 256: a lane visits more than one pair
             ('fp4_small_m', 'small_m', 'nvfp4', 2, 17, 31, True, False), ('fp4_small_m', 'small_m', 'mxfp4', 3, 33, 67, True, False),
             ('fp4_small_m', 'small_m', 'nvfp4', 7, 5, 9, False, True),
             ('fp4_tiled', 'tiled', 'nvfp4', 3, 33, 67, True, False), ('fp4_tiled', 'tiled', 'mxfp4', 9, 35, 63, True, False),
             ('fp4_tiled', 'tiled', 'nvfp4', 17, 65, 129, False, True), ('fp4_tiled', 'tiled', 'mxfp4', 1, 1, 1, True, False)]
    for kernel, algorithm, fmt, m, n, k, has_bias, silu in table:
        q = quantize(normal(rng, n, k), fmt)
        bias = normal(rng, n) if has_bias else np.zeros(1, np.float32)
        plan = plan_linear(m, n, k, algorithm=algorithm)
        constants = dict(M=m, N=n, K=k, **weight_params(q), HAS_BIAS=has_bias, APPLY_SILU=silu)
        if algorithm in ('gemv', 'small_m'):
            constants['BM'] = plan.tile_m
        inputs = {'x': normal(rng, m, k), **weight_arrays(q), 'bias': bias}
        suffix = '.silu' if silu else ''
        b.add(f'{kernel}.{fmt}.{m}x{n}x{k}{suffix}', kernel, inputs, {'y': ('float32', (m, n))}, constants, list(plan.grid))
    q = quantize(normal(rng, 9, 17), 'nvfp4')
    plan = plan_linear(1, 9, 17, algorithm='gemv')
    x = normal(rng, 1, 17).astype(np.float16)
    b.add('fp4_gemv.nvfp4.float16-input', 'fp4_gemv', {'x': x, **weight_arrays(q), 'bias': np.zeros(1, np.float32)},
          {'y': ('float32', (1, 9))}, dict(M=1, N=9, K=17, **weight_params(q), HAS_BIAS=False, APPLY_SILU=False, BM=1), list(plan.grid))


def add_w4a4(b: Builder, rng) -> None:
    for af, wf, m, n, k in [('nvfp4', 'nvfp4', 9, 35, 67), ('mxfp4', 'mxfp4', 3, 33, 65), ('nvfp4', 'mxfp4', 9, 35, 67)]:
        qa, qw = quantize(normal(rng, m, k), af), quantize(normal(rng, n, k), wf)
        plan = plan_linear(m, n, k, algorithm='tiled')
        inputs = {**weight_arrays(qa, names=('aq', 'ascales', 'aglobal')), **weight_arrays(qw), 'bias': np.zeros(1, np.float32)}
        b.add(f'fp4_w4a4.{af}x{wf}.{m}x{n}x{k}', 'fp4_w4a4', inputs, {'y': ('float32', (m, n))},
              dict(M=m, N=n, K=k, **weight_params(qa, 'A'), **weight_params(qw), HAS_BIAS=False, APPLY_SILU=False), list(plan.grid))


def add_quantize(b: Builder, rng) -> None:
    for fmt, n, k, g, f, scale in [('nvfp4', 3, 67, 16, 0, 0.125), ('mxfp4', 3, 67, 32, 1, 1.0)]:
        sg, p = (k + g - 1) // g, (k + 1) // 2
        inputs = {'x': normal(rng, n, k), 'global': np.array([scale], np.float32)}
        b.add(f'fp4_quantize.{fmt}.{n}x{k}', 'fp4_quantize', inputs, {'outq': ('uint8', (n, p)), 'outs': ('uint8', (n, sg))},
              dict(N=n, K=k, SG=sg, G=g, F=f, P=p), grid_for(n * sg))


def add_rms(b: Builder, rng) -> None:
    for label, m, k, residual, scale in [('plain', 4, 129, False, 1.0), ('residual', 3, 300, True, 1.0), ('small-input', 2, 67, False, 1e-3)]:
        inputs = {'x': normal(rng, m, k, scale=scale), 'residual': normal(rng, m, k, scale=scale) if residual else np.zeros(1, np.float32),
                  'weight': (1 + 0.1 * rng.normal(size=k)).astype(np.float32), 'epsilon': np.array([1e-5], np.float32)}
        b.add(f'rms_norm.{label}.{m}x{k}', 'rms_norm', inputs, {'y': ('float32', (m, k))},
              dict(K=k, HAS_RESIDUAL=residual), [m * THREADS, 1, 1])


def add_silu(b: Builder, rng) -> None:
    size = 300
    b.add('silu_mul.float32', 'silu_mul', {'gate': normal(rng, size, scale=2.0), 'up': normal(rng, size, scale=2.0)},
          {'y': ('float32', (size,))}, dict(SIZE=size), grid_for(size))
    b.add('silu_mul.float16-input', 'silu_mul',
          {'gate': normal(rng, size, scale=2.0).astype(np.float16), 'up': normal(rng, size, scale=2.0).astype(np.float16)},
          {'y': ('float32', (size,))}, dict(SIZE=size), grid_for(size))
    # 16 subnormal gates among normal ones, so that FTZ shows in a count and does not decide the verdict.
    gate = normal(rng, size, scale=2.0)
    gate[:16] = np.float32(1e-40) * np.where(np.arange(16) % 2, 1, -1)
    b.add('silu_mul.subnormal-gates', 'silu_mul', {'gate': gate, 'up': np.ones(size, np.float32)},
          {'y': ('float32', (size,))}, dict(SIZE=size), grid_for(size))


def add_second_pass(b: Builder, seed: int) -> None:
    """K above 256, so that each lane of small_m runs its loop more than once.

    With K <= 256 the accumulator starts at 0 and takes one pass, so fma(a, w, 0) and a * w are the same number: the fault
    K7 changed no value in the first 3 small_m cases and passed. It has its own generator, so the other inputs stay as they are.
    """
    rng = np.random.default_rng(seed + 1)
    m, n, k = 2, 5, 601
    q = quantize(normal(rng, n, k), 'nvfp4')
    plan = plan_linear(m, n, k, algorithm='small_m')
    constants = dict(M=m, N=n, K=k, **weight_params(q), HAS_BIAS=False, APPLY_SILU=False, BM=plan.tile_m)
    inputs = {'x': normal(rng, m, k), **weight_arrays(q), 'bias': np.zeros(1, np.float32)}
    b.add(f'fp4_small_m.nvfp4.{m}x{n}x{k}', 'fp4_small_m', inputs, {'y': ('float32', (m, n))}, constants, list(plan.grid))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', required=True, help='folder to write')
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    b = Builder()
    for add in (add_decode, add_linear, add_w4a4, add_quantize, add_rms, add_silu):
        add(b, rng)
    add_second_pass(b, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_file(b.tensors, out / 'inputs.safetensors')
    (out / 'cases.json').write_text(json.dumps({'schema': 'cumlx.kernel-cases.v1', 'seed': args.seed, 'cases': b.cases}, indent=1) + '\n')
    print(f'{len(b.cases)} cases, {len({c["kernel"] for c in b.cases})} kernels, written to {out}')


if __name__ == '__main__':
    main()
