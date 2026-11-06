#!/usr/bin/env python3
"""Draw fixed scenes and their gradients through whatever `gsplat` this process has, on the device that the program calls cuda.

On the V100 that is gsplat itself. On a Mac it is the adapter, through cumlx run:

    python spikes/m6_f2_cuda_record.py --output cuda.npz                                                   # NVIDIA machine, real gsplat
    cumlx run spikes/m6_f2_cuda_record.py --substitute gsplat=cumlx.compat.gsplat_metalsplat -- --output mac.npz
    CUMLX_F2_FAULT=no_eps2d cumlx run spikes/m6_f2_cuda_record.py --substitute gsplat=m6_f2_fault_gsplat -- --output fault.npz

It has no Mac-specific line. The rules are in validation/m6-f2-cuda-limits.json, and spikes/m6_f2_cuda_judge.py compares two outputs.

It passes packed=False, which chooses gsplat's memory layout and not its result. gsplat 1.5.3 with the default packed=True takes the image
size from the packed means2d in rasterize_to_pixels, and so rejects the [C, 3] backgrounds that its own docstring asks for (attempt 1, 2026-11-06).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import gsplat
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_f2_hand_render import look_at  # noqa: E402
from m6_f2_hand_render import render as hand_render

DEVICE = 'cuda'
PARAMETERS = ('means', 'quats', 'scales', 'opacities', 'colors')


def conformance_scenes() -> dict[str, dict]:
    """The six scenes of spikes/m6_f2_conformance.py, copied here so that this file needs no cumlx, and two more."""
    rng = np.random.default_rng(0)
    rotated = look_at([1.3, -0.7, -3.1], [0.1, 0.05, 0.0], [0.0, 1.0, 0.0])
    plain_K = np.array([[128.0, 0.0, 64.0], [0.0, 128.0, 64.0], [0.0, 0.0, 1.0]])
    off_K = np.array([[150.0, 0.0, 40.0], [0.0, 110.0, 30.0], [0.0, 0.0, 1.0]])
    n = 6
    scenes = {
        'one_gaussian': dict(means=[[0.21, -0.13, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[0.25, 0.25, 0.25]], opacities=[0.8], colors=[[0.9, 0.3, 0.2]], viewmat=np.eye(4), K=plain_K, width=128, height=128),
        'off_center_rotated': dict(means=[[0.13, -0.07, 0.1]], quats=[[0.82, 0.31, -0.42, 0.21]], scales=[[0.45, 0.09, 0.16]], opacities=[0.9], colors=[[0.2, 0.7, 0.9]], viewmat=rotated, K=off_K, width=96, height=64),
        'overlap_background': dict(means=rng.uniform(-0.5, 0.5, (n, 3)) + [0.0, 0.0, 3.5], quats=rng.normal(size=(n, 4)), scales=rng.uniform(0.08, 0.3, (n, 3)), opacities=rng.uniform(0.4, 0.9, n), colors=rng.uniform(0.0, 1.0, (n, 3)), viewmat=np.eye(4), K=plain_K, width=128, height=128, background=[0.2, 0.4, 0.6]),
        'small_classic': dict(means=[[0.031, -0.017, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[0.016, 0.016, 0.016]], opacities=[0.9], colors=[[1.0, 1.0, 1.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128),
        'small_antialiased': dict(means=[[0.031, -0.017, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[0.016, 0.016, 0.016]], opacities=[0.9], colors=[[1.0, 1.0, 1.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128, antialiased=True),
        'near_and_far': dict(means=[[0.2, 0.1, 0.05], [0.0, 0.0, 3.0], [-0.3, 0.2, 9.0]], quats=[[1.0, 0.0, 0.0, 0.0]] * 3, scales=[[0.2, 0.2, 0.2]] * 3, opacities=[0.9, 0.7, 0.9], colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128, near_plane=0.5, far_plane=6.0),
    }
    # One Gaussian far wider than the frame, at opacity 1, in front of white: each pixel's raw alpha is above 0.999 (the corner's is about
    # 0.99938), so the clamp sets the whole image. The first form had scales 20 and only 28% of its pixels above 0.999 (review, 2026-11-06).
    scenes['opaque_wall'] = dict(means=[[0.0, 0.0, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[60.0, 60.0, 0.5]], opacities=[1.0], colors=[[0.0, 0.0, 0.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128, background=[1.0, 1.0, 1.0])
    return scenes


def camera_at(azimuth: float, elevation: float) -> np.ndarray:
    eye = 3.0 * np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
    return look_at(eye, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])


def learned_like(view: np.ndarray) -> dict:
    """200 Gaussians where the training program starts and where it may go: some outside the frame, some opaque, some faint."""
    rng = np.random.default_rng(7)
    n = 200
    return dict(means=rng.uniform(-1.0, 1.0, (n, 3)), quats=rng.normal(size=(n, 4)), scales=rng.uniform(0.02, 0.2, (n, 3)), opacities=rng.uniform(0.05, 0.999, n), colors=rng.uniform(0.0, 1.0, (n, 3)),
                viewmat=view, K=np.array([[128.0, 0.0, 64.0], [0.0, 128.0, 64.0], [0.0, 0.0, 1.0]]), width=128, height=128)


def all_scenes() -> dict[str, dict]:
    scenes = conformance_scenes()
    scenes['learned_like_train_view'] = learned_like(camera_at(2 * math.pi * 3 / 8, 0.5))
    scenes['learned_like_heldout_view'] = learned_like(camera_at(2 * math.pi * 3.5 / 8, 0.25))
    return scenes


def gradient_scenes() -> dict[str, dict]:
    """covering: from spikes/m6_f2_conformance.py, where no pixel is near the 1/255 cutoff. learned_like: the scene above, on a 64 x 64 frame."""
    rng = np.random.default_rng(1)
    n = 5
    covering = dict(means=rng.uniform(-0.3, 0.3, (n, 3)) + [0.0, 0.0, 3.0], quats=rng.normal(size=(n, 4)), scales=rng.uniform(0.95, 1.6, (n, 3)), opacities=rng.uniform(0.3, 0.7, n), colors=rng.uniform(0.2, 0.8, (n, 3)),
                    viewmat=np.eye(4), K=np.array([[64.0, 0.0, 16.0], [0.0, 64.0, 16.0], [0.0, 0.0, 1.0]]), width=32, height=32)
    wide = learned_like(camera_at(2 * math.pi * 3 / 8, 0.5))
    wide.update(K=np.array([[64.0, 0.0, 32.0], [0.0, 64.0, 32.0], [0.0, 0.0, 1.0]]), width=64, height=64)
    return {'covering': covering, 'learned_like': wide}


def tensor(value, grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(value), dtype=torch.float32, device=DEVICE, requires_grad=grad)


def draw(scene: dict, leaves: dict[str, torch.Tensor] | None = None):
    values = leaves or {name: tensor(scene[name]) for name in PARAMETERS}
    backgrounds = None if scene.get('background') is None else tensor(scene['background'])[None]
    image, alpha, _ = gsplat.rasterization(values['means'], values['quats'], values['scales'], values['opacities'], values['colors'], tensor(scene['viewmat'])[None], tensor(scene['K'])[None],
                                           scene['width'], scene['height'], near_plane=scene.get('near_plane', 0.01), far_plane=scene.get('far_plane', 1e10), backgrounds=backgrounds,
                                           rasterize_mode='antialiased' if scene.get('antialiased') else 'classic', packed=False)
    return image, alpha


def by_hand(scene: dict):
    return hand_render(scene['means'], scene['quats'], scene['scales'], scene['opacities'], scene['colors'], scene['viewmat'], scene['K'], scene['width'], scene['height'],
                       near_plane=scene.get('near_plane', 0.01), far_plane=scene.get('far_plane', 1e10), antialiased=scene.get('antialiased', False), background=scene.get('background'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='an .npz file. A .json file of the same name gets the summary')
    parser.add_argument('--no-hand', action='store_true', help='skip the hand render, which takes some seconds')
    args = parser.parse_args()
    arrays: dict[str, np.ndarray] = {}
    summary: dict = {'gsplat_module': getattr(gsplat, '__name__', '?'), 'gsplat_version': getattr(gsplat, '__version__', None), 'torch': torch.__version__, 'scenes': {}, 'gradients': {}}
    for name, scene in all_scenes().items():
        with torch.no_grad():
            image, alpha = draw(scene)
        summary['device'] = image.device.type
        arrays[f'image/{name}'] = image[0].detach().cpu().double().numpy()
        arrays[f'alpha/{name}'] = alpha[0, ..., 0].detach().cpu().double().numpy()
        if not args.no_hand:
            want_image, want_alpha = by_hand(scene)
            arrays[f'hand_image/{name}'] = want_image
            arrays[f'hand_alpha/{name}'] = want_alpha
        summary['scenes'][name] = {'width': scene['width'], 'height': scene['height'], 'background': scene.get('background')}
    generator = torch.Generator().manual_seed(0)
    for name, scene in gradient_scenes().items():
        target = torch.rand(scene['height'], scene['width'], 3, generator=generator, dtype=torch.float64)
        leaves = {key: tensor(scene[key], grad=True) for key in PARAMETERS}
        image, alpha = draw(scene, leaves)
        loss = ((image[0].cpu().double() - target) ** 2).mean() + alpha.cpu().double().mean()
        loss.backward()
        summary['gradients'][name] = {'loss': float(loss)}
        for key in PARAMETERS:
            arrays[f'grad/{name}/{key}'] = leaves[key].grad.detach().cpu().double().numpy()
    np.savez(args.output, **arrays)
    Path(args.output).with_suffix('.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'device': summary.get('device'), 'arrays': len(arrays), 'gsplat_module': summary['gsplat_module']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
