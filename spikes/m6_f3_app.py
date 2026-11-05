#!/usr/bin/env python3
"""F3: SAM2's connected components, a custom .cu kernel app, on CUDA and on a Mac. The rules are in validation/m6-f3-limits.json.

    python spikes/m6_f3_app.py fetch --out src/                       # sam2/csrc/connected_components.cu at the pinned commit, hash checked
    python spikes/m6_f3_app.py cuda --source src/connected_components.cu --build build/ --output cuda.npz
    cumlx kernel port src/connected_components.cu -o metal/connected_components.metal --translator woof
    python spikes/m6_f3_app.py mac --metal metal/connected_components.metal --output mac.npz [--fault skip_compression]
    python spikes/m6_f3_app.py reference --output reference.npz

cuda builds the original file as a torch extension with nvcc and calls its host function get_connected_componnets (the misspelling
is sam2's), so the six kernels launch as SAM2 launches them. mac runs the six kernels that woof translated, through cumlx's kernel
runner, in the order and with the launch geometry of that host function. The host function is C++ with ATen, so it does not run on
the Mac: this driver is its port. reference is a serial NumPy version of the same algorithm, from spikes/m3_kernels.py. Each side
writes the labels and counts of the same masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
import urllib.request

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
KERNEL = 'connected_components'
BLOCK = 16
FAULTS = ('skip_compression', 'half_grid', 'swapped_wh')
SCALARS = {'init_labeling': 'uint32', 'merge': 'uint32', 'compression': 'int32', 'final_labeling': 'int32', 'init_counting': 'int32', 'final_counting': 'int32'}


def mask_set() -> dict[str, np.ndarray]:
    """[N, 1, H, W] uint8 batches, one per size. Each batch holds the six kinds below in this order."""
    from m3_kernels import cc_dense_image, cc_sparse_image
    out = {}
    for index, (h, w) in enumerate(((32, 32), (64, 64), (128, 96), (256, 256))):
        rng = np.random.default_rng(100 + index)
        yy, xx = np.mgrid[0:h, 0:w]
        blobs = np.zeros((h, w), np.uint8)
        for _ in range(6):  # ellipses, as masks from a segmenter look
            cy, cx, ry, rx = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(2, h / 4), rng.uniform(2, w / 4)
            blobs |= (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.uint8)
        angle = np.arctan2(yy - h / 2, xx - w / 2)
        radius = np.hypot(yy - h / 2, xx - w / 2)
        spiral = ((np.floor((radius - 3 * angle) / 4) % 2) == 0).astype(np.uint8)  # long arms, so that unions run across the image
        kinds = [cc_dense_image(h, w, 200 + index), cc_sparse_image(h, w), blobs, spiral, np.zeros((h, w), np.uint8), np.ones((h, w), np.uint8)]
        out[f'{h}x{w}'] = np.stack(kinds)[:, None]
    return out


KINDS = ('dense', 'sparse', 'blobs', 'spiral', 'empty', 'full')


def fetch(out: Path) -> int:
    doc = json.loads((REPO / 'validation' / 'kernel-cases-m3' / 'sources.json').read_text())
    commit, spec = doc['repositories']['sam2']['commit'], doc['sam2_file']
    data = urllib.request.urlopen(f"https://raw.githubusercontent.com/facebookresearch/sam2/{commit}/{spec['path']}", timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec['sha256']:
        raise SystemExit(f"sam2 connected_components.cu has sha256 {digest}, not {spec['sha256']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{KERNEL}.cu').write_bytes(data)
    print(json.dumps({'commit': commit, 'sha256': digest, 'path': str(out / f'{KERNEL}.cu')}))
    return 0


def save(output: Path, labels: dict[str, np.ndarray], counts: dict[str, np.ndarray], info: dict[str, Any]) -> None:
    arrays = {**{f'labels/{k}': v for k, v in labels.items()}, **{f'counts/{k}': v for k, v in counts.items()}}
    np.savez(output, **arrays)
    output.with_suffix('.json').write_text(json.dumps({**info, 'sizes': sorted(labels), 'kinds': list(KINDS)}, indent=1))


def run_cuda(source: Path, build: Path, output: Path) -> int:
    import torch
    from torch.utils.cpp_extension import load
    build.mkdir(parents=True, exist_ok=True)
    ext = load(name='sam2_connected_components_cumlx', sources=[str(source)], build_directory=str(build), verbose=False)
    labels, counts = {}, {}
    for size, masks in mask_set().items():
        lab, cnt = ext.get_connected_componnets(torch.from_numpy(masks).cuda())
        torch.cuda.synchronize()
        labels[size], counts[size] = lab.cpu().numpy(), cnt.cpu().numpy()
    save(output, labels, counts, {'side': 'cuda', 'device': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    print(json.dumps({'side': 'cuda', 'sizes': len(labels)}))
    return 0


def grid_for(entry: str, h: int, w: int, fault: str | None) -> tuple[int, int, int]:
    if entry in ('init_counting', 'final_counting'):
        blocks = [(w + BLOCK) // BLOCK, (h + BLOCK) // BLOCK]
    else:
        blocks = [math.ceil(((w + 1) // 2) / BLOCK), math.ceil(((h + 1) // 2) / BLOCK)]
    if fault == 'half_grid':
        blocks = [max(1, blocks[0] // 2), blocks[1]]
    return (blocks[0] * BLOCK, blocks[1] * BLOCK, 1)


def run_mac(metal: Path, output: Path, fault: str | None) -> int:
    from cumlx.verify.kernel_verify import WoofEngine
    from cumlx.verify.opgolden import KernelCase, TensorSpec
    engine = WoofEngine()
    engine.use_translations({KERNEL: json.loads(metal.with_name(f'{KERNEL}.port.json').read_text())})
    source = metal.read_text()

    def launch(entry: str, h: int, w: int, inputs: dict[str, np.ndarray], out_name: str, start: np.ndarray, order: list[str]) -> np.ndarray:
        scalar = SCALARS[entry]
        W, H = (h, w) if fault == 'swapped_wh' else (w, h)
        specs = [TensorSpec(k, str(v.dtype), tuple(v.shape)) for k, v in inputs.items()]
        specs += [TensorSpec(f'{out_name}_in', str(start.dtype), tuple(start.shape)), TensorSpec('W', scalar, ()), TensorSpec('H', scalar, ())]
        case = KernelCase(id=f'{entry}.{h}x{w}', kernel=KERNEL, inputs=specs, outputs=[TensorSpec(out_name, str(start.dtype), tuple(start.shape))],
                          grid=grid_for(entry, h, w, fault), threadgroup=(BLOCK, BLOCK, 1), entry=entry, call=tuple(order), initial={out_name: f'{out_name}_in'})
        values = {**inputs, f'{out_name}_in': start, 'W': np.array(W, dtype=scalar), 'H': np.array(H, dtype=scalar)}
        return engine.run(case, source, '', values)[out_name]

    labels, counts = {}, {}
    for size, masks in mask_set().items():
        n, _, h, w = masks.shape
        lab_out, cnt_out = np.zeros(masks.shape, np.int32), np.zeros(masks.shape, np.int32)
        for index in range(n):  # one image at a time, as the host function's loop over n does
            img = np.ascontiguousarray(masks[index, 0])
            zero = np.zeros((h, w), np.int32)
            label = launch('init_labeling', h, w, {}, 'label', zero, ['label', 'W', 'H'])
            label = launch('merge', h, w, {'img': img}, 'label', label, ['img', 'label', 'W', 'H'])
            if fault != 'skip_compression':
                label = launch('compression', h, w, {}, 'label', label, ['label', 'W', 'H'])
            label = launch('final_labeling', h, w, {'img': img}, 'label', label, ['img', 'label', 'W', 'H'])
            count_init = launch('init_counting', h, w, {'label': label}, 'count_init', zero, ['label', 'count_init', 'W', 'H'])
            count_final = launch('final_counting', h, w, {'label': label, 'count_init': count_init}, 'count_final', zero, ['label', 'count_init', 'count_final', 'W', 'H'])
            lab_out[index, 0], cnt_out[index, 0] = label, count_final
        labels[size], counts[size] = lab_out, cnt_out
    save(output, labels, counts, {'side': 'mac', 'fault': fault, 'engine': engine.describe(), 'metal_sha256': hashlib.sha256(metal.read_bytes()).hexdigest()})
    print(json.dumps({'side': 'mac', 'fault': fault, 'sizes': len(labels)}))
    return 0


def run_reference(output: Path) -> int:
    from m3_kernels import cc_compress, cc_count, cc_final, cc_init, cc_merge
    labels, counts = {}, {}
    for size, masks in mask_set().items():
        lab_out, cnt_out = np.zeros(masks.shape, np.int32), np.zeros(masks.shape, np.int32)
        for index in range(masks.shape[0]):
            img = masks[index, 0]
            label = cc_final(img, cc_compress(cc_merge(img, cc_init(*img.shape))))
            per_label = cc_count(label).reshape(-1)
            lab_out[index, 0] = label
            cnt_out[index, 0] = np.where(label > 0, per_label[np.maximum(label, 1) - 1], 0)
        labels[size], counts[size] = lab_out, cnt_out
    save(output, labels, counts, {'side': 'reference'})
    print(json.dumps({'side': 'reference', 'sizes': len(labels)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('fetch').add_argument('--out', required=True, type=Path)
    c = sub.add_parser('cuda')
    c.add_argument('--source', required=True, type=Path)
    c.add_argument('--build', required=True, type=Path)
    c.add_argument('--output', required=True, type=Path)
    m = sub.add_parser('mac')
    m.add_argument('--metal', required=True, type=Path)
    m.add_argument('--output', required=True, type=Path)
    m.add_argument('--fault', choices=FAULTS)
    sub.add_parser('reference').add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'fetch':
        return fetch(args.out)
    if args.mode == 'cuda':
        return run_cuda(args.source, args.build, args.output)
    if args.mode == 'mac':
        return run_mac(args.metal, args.output, args.fault)
    return run_reference(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
