#!/usr/bin/env python3
"""Measure how this Mac runs bfloat16: which model operations run on MPS and MLX, how far they are from float64, and what
happens to subnormal values. It writes one JSON record.

    python scripts/bf16_mac_probe.py -o validation/bf16-mac-probe.json \
        [--activations golden/tensors.npz --activations-label "what the file is"]

--activations counts the subnormal values in the float32 or bfloat16 arrays of an .npz, for example the activations of a
CUMLX golden. The record keeps the label and the SHA-256 of the file, not its path.

This is a measurement on one machine. It does not compare with a CUDA run; validation/kernel-golden-a4000-bf16 does that for
the kernel header.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import warnings

import numpy as np

SUBNORMAL_F32 = np.float32(1.1754944e-38)  # the smallest normal float32 and bfloat16 value, 2^-126


def rel_l2(out, ref) -> float:
    import mlx.core as mx
    import torch

    if not torch.is_tensor(out):
        out = torch.from_numpy(np.array(out.astype(mx.float32)))
    out = out.cpu().double()
    return float((out - ref).norm() / ref.norm())


def ops() -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    x, w = torch.randn(256, 512), torch.randn(512, 384)
    q, k, v = (torch.randn(1, 8, 128, 64) for _ in range(3))
    cases = {
        'matmul': (lambda d, t: x.to(d, t) @ w.to(d, t), x.double() @ w.double()),
        'softmax': (lambda d, t: torch.softmax(x.to(d, t), -1), torch.softmax(x.double(), -1)),
        'layer_norm': (lambda d, t: F.layer_norm(x.to(d, t), (512,)), F.layer_norm(x.double(), (512,))),
        'rms_norm': (lambda d, t: F.rms_norm(x.to(d, t), (512,)), F.rms_norm(x.double(), (512,))),
        'silu': (lambda d, t: F.silu(x.to(d, t)), F.silu(x.double())),
        'sdpa_causal': (lambda d, t: F.scaled_dot_product_attention(q.to(d, t), k.to(d, t), v.to(d, t), is_causal=True),
                        F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)),
        'cumsum': (lambda d, t: torch.cumsum(x.to(d, t), -1), torch.cumsum(x.double(), -1)),
        'topk': (lambda d, t: torch.topk(x.to(d, t), 5, -1).values, torch.topk(x.double(), 5, -1).values),
    }
    result = {}
    for name, (fn, ref) in cases.items():
        row = {}
        for label, device, dtype in (('cpu_bfloat16', 'cpu', torch.bfloat16), ('mps_bfloat16', 'mps', torch.bfloat16),
                                     ('mps_float16', 'mps', torch.float16)):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    out = fn(device, dtype)
                row[label] = {'rel_l2': rel_l2(out, ref), 'warnings': [str(c.message)[:200] for c in caught]}
            except Exception as error:  # an operation that does not run is a result here
                row[label] = {'error': f'{type(error).__name__}: {str(error)[:200]}'}
        result[name] = row
    import mlx.core as mx

    mq, mk, mv = (mx.array(t.numpy()).astype(mx.bfloat16) for t in (q, k, v))
    result['matmul']['mlx_gpu_bfloat16'] = {'rel_l2': rel_l2(mx.array(x.numpy()).astype(mx.bfloat16) @ mx.array(w.numpy()).astype(mx.bfloat16),
                                                             cases['matmul'][1])}
    result['sdpa_causal']['mlx_gpu_bfloat16'] = {'rel_l2': rel_l2(mx.fast.scaled_dot_product_attention(mq, mk, mv, scale=64 ** -0.5, mask='causal'),
                                                                  cases['sdpa_causal'][1])}
    return result


def autocast() -> dict:
    import torch

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with torch.autocast('mps', dtype=torch.bfloat16):
            y = torch.randn(8, 8, device='mps') @ torch.randn(8, 8, device='mps')
    return {'output_dtype': str(y.dtype), 'warnings': [str(c.message)[:200] for c in caught]}


def subnormals() -> dict:
    import mlx.core as mx
    import torch

    values = [2.0 ** -130, 2.0 ** -133, -2.0 ** -128]
    f = torch.tensor(values, dtype=torch.float32)
    bits = lambda t: [hex(b & 0xFFFF) for b in t.cpu().view(torch.int16).tolist()]  # noqa: E731
    s = torch.tensor([2.0 ** -130], dtype=torch.bfloat16)
    m = mx.array(np.array(values, np.float32))
    return {
        'inputs': ['2^-130', '2^-133', '-2^-128'],
        'float32_to_bfloat16_bits': {'cpu': bits(f.to(torch.bfloat16)), 'mps': bits(f.to('mps').to(torch.bfloat16)),
                                     'mlx_gpu': [hex(b) for b in np.array(m.astype(mx.bfloat16).view(mx.uint16)).tolist()],
                                     'mlx_cpu': [hex(b) for b in np.array(mx.astype(m, mx.bfloat16, stream=mx.cpu).view(mx.uint16)).tolist()]},
        'bfloat16_add_2^-130_twice': {'cpu': (s + s).float().item(), 'mps': (s.to('mps') + s.to('mps')).float().cpu().item()},
        'cuda': 'CUDA keeps these values: validation/kernel-golden-v100-bf16 and validation/kernel-golden-a4000-bf16',
    }


def count_subnormals(path: Path, label: str) -> dict:
    data = np.load(path)
    total = found = arrays = 0
    where = []
    for name in data.files:
        a = data[name]
        if a.dtype == np.uint16:
            magnitude = a & np.uint16(0x7FFF)
            n = int(((magnitude != 0) & (magnitude < 0x0080)).sum())
        elif a.dtype in (np.float32, np.float16):
            f = a.astype(np.float32)
            n = int(((f != 0) & (np.abs(f) < SUBNORMAL_F32)).sum())
        else:
            continue
        arrays += 1
        total += a.size
        found += n
        if n:
            where.append({'array': name, 'subnormal_values': n})
    return {'label': label, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'arrays': arrays, 'values': total,
            'subnormal_values': found, 'arrays_with_subnormals': where[:20]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--out', type=Path, required=True)
    parser.add_argument('--activations', type=Path)
    parser.add_argument('--activations-label', default='')
    args = parser.parse_args()
    import mlx.core as mx
    import torch

    record = {
        'schema': 'cumlx.bf16_mac_probe.v1',
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'macos': platform.mac_ver()[0], 'machine': platform.machine(),
        'device': mx.device_info(mx.gpu).get('device_name') if hasattr(mx, 'device_info') else None,
        'torch': torch.__version__, 'mlx': metadata.version('mlx'),
        'reference': 'float64 on the CPU, from the same float32 inputs',
        'ops': ops(), 'autocast_mps_bfloat16': autocast(), 'subnormals': subnormals(),
        'does_not_show': ['a CUDA run of these operations', 'other Macs, macOS, torch or MLX versions',
                          'operations other than the ones listed, or a whole model'],
    }
    if args.activations:
        record['activations'] = count_subnormals(args.activations, args.activations_label)
    args.out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'ops_with_errors': [n for n, row in record['ops'].items() if any('error' in c for c in row.values())],
                      'subnormal_values_in_activations': record.get('activations', {}).get('subnormal_values')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
