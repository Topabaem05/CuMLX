#!/usr/bin/env python3
"""The gsplat adapter against the hand render, wrong conventions, gradients and refusals. The rules are in validation/m6-f2-adapter-limits.json.

    python spikes/m6_f2_conformance.py --output out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_f2_hand_render import look_at  # noqa: E402
from m6_f2_hand_render import render as hand_render

from cumlx.compat.gsplat_metalsplat import rasterization  # noqa: E402

DEVICE = 'mps'


def scene_set() -> dict[str, dict]:
    rng = np.random.default_rng(0)
    rotated = look_at([1.3, -0.7, -3.1], [0.1, 0.05, 0.0], [0.0, 1.0, 0.0])
    plain_K = np.array([[128.0, 0.0, 64.0], [0.0, 128.0, 64.0], [0.0, 0.0, 1.0]])
    off_K = np.array([[150.0, 0.0, 40.0], [0.0, 110.0, 30.0], [0.0, 0.0, 1.0]])
    n = 6
    return {
        'one_gaussian': dict(means=[[0.21, -0.13, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[0.25, 0.25, 0.25]], opacities=[0.8], colors=[[0.9, 0.3, 0.2]], viewmat=np.eye(4), K=plain_K, width=128, height=128),
        'off_center_rotated': dict(means=[[0.13, -0.07, 0.1]], quats=[[0.82, 0.31, -0.42, 0.21]], scales=[[0.45, 0.09, 0.16]], opacities=[0.9], colors=[[0.2, 0.7, 0.9]], viewmat=rotated, K=off_K, width=96, height=64),
        'overlap_background': dict(means=rng.uniform(-0.5, 0.5, (n, 3)) + [0.0, 0.0, 3.5], quats=rng.normal(size=(n, 4)), scales=rng.uniform(0.08, 0.3, (n, 3)), opacities=rng.uniform(0.4, 0.9, n), colors=rng.uniform(0.0, 1.0, (n, 3)), viewmat=np.eye(4), K=plain_K, width=128, height=128, background=[0.2, 0.4, 0.6]),
        'small_classic': dict(means=[[0.031, -0.017, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[0.016, 0.016, 0.016]], opacities=[0.9], colors=[[1.0, 1.0, 1.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128),
        'small_antialiased': dict(means=[[0.031, -0.017, 3.0]], quats=[[1.0, 0.0, 0.0, 0.0]], scales=[[0.016, 0.016, 0.016]], opacities=[0.9], colors=[[1.0, 1.0, 1.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128, antialiased=True),
        'near_and_far': dict(means=[[0.2, 0.1, 0.05], [0.0, 0.0, 3.0], [-0.3, 0.2, 9.0]], quats=[[1.0, 0.0, 0.0, 0.0]] * 3, scales=[[0.2, 0.2, 0.2]] * 3, opacities=[0.9, 0.7, 0.9], colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], viewmat=np.eye(4), K=plain_K, width=128, height=128, near_plane=0.5, far_plane=6.0),
    }


def tensor(value, grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(value), dtype=torch.float32, device=DEVICE, requires_grad=grad)


def through_adapter(scene: dict, wrong: str | None = None, **changes):
    quats = np.asarray(scene['quats'], dtype=np.float64)
    viewmat = np.asarray(scene['viewmat'], dtype=np.float64)
    K = np.asarray(scene['K'], dtype=np.float64).copy()
    antialiased = scene.get('antialiased', False)
    eps2d = 0.3
    if wrong == 'quaternion_xyzw':
        quats = quats[:, [1, 2, 3, 0]]
    elif wrong == 'camera_to_world_view_matrix':
        viewmat = np.linalg.inv(viewmat)
    elif wrong == 'fx_fy_swapped':
        K[0, 0], K[1, 1] = K[1, 1], K[0, 0]
    elif wrong == 'no_eps2d':
        eps2d = 0.0
    elif wrong == 'antialiased_for_classic':
        antialiased = True
    backgrounds = None if scene.get('background') is None else tensor(scene['background'])[None]
    with torch.no_grad():
        image, alpha, _ = rasterization(tensor(scene['means']), tensor(quats), tensor(scene['scales']), tensor(scene['opacities']), tensor(scene['colors']), tensor(viewmat)[None], tensor(K)[None],
                                        scene['width'], scene['height'], near_plane=scene.get('near_plane', 0.01), far_plane=scene.get('far_plane', 1e10), eps2d=eps2d, backgrounds=backgrounds,
                                        rasterize_mode='antialiased' if antialiased else 'classic', **changes)
    torch.mps.synchronize()
    return image[0].cpu().numpy().astype(np.float64), alpha[0, ..., 0].cpu().numpy().astype(np.float64), image.device.type


def by_hand(scene: dict, pixel_offset: float = 0.5):
    return hand_render(scene['means'], scene['quats'], scene['scales'], scene['opacities'], scene['colors'], scene['viewmat'], scene['K'], scene['width'], scene['height'],
                       near_plane=scene.get('near_plane', 0.01), far_plane=scene.get('far_plane', 1e10), antialiased=scene.get('antialiased', False), background=scene.get('background'), pixel_offset=pixel_offset)


def compare(ours, expected, background) -> dict[str, float]:
    image, alpha = ours
    want_image, want_alpha = expected
    base = np.zeros(3) if background is None else np.asarray(background)
    foreground = want_image - base
    peak = np.unravel_index(np.argmax(np.abs(foreground).sum(-1)), foreground.shape[:2])
    found = np.unravel_index(np.argmax(np.abs(image - base).sum(-1)), foreground.shape[:2])
    return {'image_rel_l2': float(np.linalg.norm(image - want_image) / np.linalg.norm(foreground)), 'alpha_rel_l2': float(np.linalg.norm(alpha - want_alpha) / np.linalg.norm(want_alpha)),
            'peak_error_px': float(max(abs(int(peak[0]) - int(found[0])), abs(int(peak[1]) - int(found[1]))))}


def within(metrics: dict[str, float], rule: dict) -> bool:
    return metrics['image_rel_l2'] <= rule['image_rel_l2_at_most'] and metrics['alpha_rel_l2'] <= rule['alpha_rel_l2_at_most'] and metrics['peak_error_px'] <= rule['peak_error_px_at_most']


def conformance(rule: dict) -> dict:
    scenes = scene_set()
    out = {}
    for name, scene in scenes.items():
        image, alpha, device = through_adapter(scene)
        metrics = compare((image, alpha), by_hand(scene), scene.get('background'))
        out[name] = {**metrics, 'device': device, 'within': within(metrics, rule)}
    return out


def controls(rule: dict, names: list[str]) -> dict:
    scenes = scene_set()
    out = {}
    for name in names:
        results = {}
        for scene_name, scene in scenes.items():
            if name == 'half_pixel_shift':
                image, alpha, _ = through_adapter(scene)
                expected = by_hand(scene, pixel_offset=0.0)
            else:
                image, alpha, _ = through_adapter(scene, wrong=name)
                expected = by_hand(scene)
            metrics = compare((image, alpha), expected, scene.get('background'))
            results[scene_name] = {**metrics, 'within': within(metrics, rule)}
        out[name] = {'caught': not all(item['within'] for item in results.values()), 'scenes': results}
    return out


def gradient_scene() -> dict:
    """Each Gaussian covers the whole frame with an alpha far above 1/255. A pixel that crosses that cutoff jumps, in gsplat too, and a central difference sees the jump where autograd has no term for it. The first attempt had such pixels: see validation/m6-f2-adapter-run.json."""
    rng = np.random.default_rng(1)
    n = 5
    return dict(means=rng.uniform(-0.3, 0.3, (n, 3)) + [0.0, 0.0, 3.0], quats=rng.normal(size=(n, 4)), scales=rng.uniform(0.95, 1.6, (n, 3)), opacities=rng.uniform(0.3, 0.7, n), colors=rng.uniform(0.2, 0.8, (n, 3)))


def gradients(epsilon: float = 2e-3) -> dict:
    scene = gradient_scene()
    size = 32
    K = tensor([[64.0, 0.0, size / 2], [0.0, 64.0, size / 2], [0.0, 0.0, 1.0]])[None]
    view = tensor(np.eye(4))[None]
    names = ['means', 'quats', 'scales', 'opacities', 'colors']
    generator = torch.Generator().manual_seed(0)
    target = torch.rand(size, size, 3, generator=generator, dtype=torch.float64)

    def losses(values: dict[str, torch.Tensor]):
        image, alpha, _ = rasterization(values['means'], values['quats'], values['scales'], values['opacities'], values['colors'], view, K, size, size)
        return ((image[0].cpu().double() - target) ** 2).mean(), alpha.cpu().double().mean()  # in float64, so that rounding does not hide a small difference

    out = {}
    for which, pick in (('color_loss', 0), ('alpha_loss', 1)):
        leaves = {name: tensor(scene[name], grad=True) for name in names}
        losses(leaves)[pick].backward()
        for name in names:
            if which == 'alpha_loss' and name == 'colors':
                continue
            direction = torch.randn(leaves[name].shape, generator=generator).to(DEVICE)
            direction = direction / direction.norm()
            analytic = float((leaves[name].grad * direction).sum())
            with torch.no_grad():
                values = {key: value.detach() for key, value in leaves.items()}
                plus = float(losses({**values, name: values[name] + epsilon * direction})[pick])
                minus = float(losses({**values, name: values[name] - epsilon * direction})[pick])
            numeric = (plus - minus) / (2 * epsilon)
            out[f'{which}.{name}'] = {'autograd': analytic, 'central_difference': numeric, 'relative_error': abs(analytic - numeric) / max(abs(numeric), 1e-12)}
    return out


def refusals() -> dict:
    scene = scene_set()['one_gaussian']
    base = dict(means=tensor(scene['means']), quats=tensor(scene['quats']), scales=tensor(scene['scales']), opacities=tensor(scene['opacities']), colors=tensor(scene['colors']), viewmats=tensor(scene['viewmat'])[None], Ks=tensor(scene['K'])[None], width=64, height=64)
    cases = {
        'sh_degree': {'sh_degree': 3}, 'render_mode': {'render_mode': 'RGB+ED'}, 'camera_model': {'camera_model': 'fisheye'}, 'covars': {'covars': torch.eye(3, device=DEVICE)[None]},
        'sparse_grad': {'sparse_grad': True}, 'absgrad': {'absgrad': True}, 'distributed': {'distributed': True}, 'with_ut': {'with_ut': True}, 'segmented': {'segmented': True},
        'batched_means': {'means': base['means'][None]}, 'viewmats_require_grad': {'viewmats': base['viewmats'].clone().requires_grad_()}, 'colors_not_rgb': {'colors': torch.zeros(1, 6, device=DEVICE)}, 'unknown_option': {'no_such_option': True},
    }
    out = {}
    for name, change in cases.items():
        try:
            rasterization(**{**base, **change})
            out[name] = 'ran'
        except NotImplementedError as error:
            out[name] = f'raised: {error}'
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    limits = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm6-f2-adapter-limits.json').read_text())
    result = {'conformance': conformance(limits['conformance']), 'controls': controls(limits['conformance'], limits['controls']['must_fail']), 'gradients': gradients(), 'refusals': refusals()}
    Path(args.output).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
