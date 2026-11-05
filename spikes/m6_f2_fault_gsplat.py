"""The gsplat adapter with one fault, named by CUMLX_F2_FAULT, for the controls of validation/m6-f2-cuda-limits.json.

    CUMLX_F2_FAULT=no_eps2d cumlx run spikes/m6_f2_cuda_record.py --substitute gsplat=m6_f2_fault_gsplat -- --output fault.npz

A comparison that passes the right adapter says something only when it fails each of these. This module is for those controls, and nothing else imports it.
"""
from __future__ import annotations

import os

import torch

from cumlx.compat import gsplat_metalsplat

FAULTS = ('quaternion_xyzw', 'camera_to_world_view_matrix', 'fx_fy_swapped', 'half_pixel_shift', 'no_eps2d', 'antialiased_for_classic', 'alpha_clamp_0.99',
          'means_gradient_1.1', 'colors_gradient_zero')
FAULT = os.environ.get('CUMLX_F2_FAULT', '')
if FAULT not in FAULTS:
    raise RuntimeError(f'CUMLX_F2_FAULT must be one of {FAULTS}, got {FAULT!r}')
__version__ = f'fault:{FAULT}'


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, factor):
        ctx.factor = factor
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


def rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **options):
    if FAULT == 'quaternion_xyzw':
        quats = quats[:, [1, 2, 3, 0]]
    elif FAULT == 'camera_to_world_view_matrix':
        viewmats = torch.linalg.inv(viewmats.cpu()).to(viewmats.device)
    elif FAULT == 'fx_fy_swapped':
        Ks = Ks.clone()
        Ks[:, 0, 0], Ks[:, 1, 1] = Ks[:, 1, 1].clone(), Ks[:, 0, 0].clone()
    elif FAULT == 'half_pixel_shift':
        # Pixel centers at j in place of j + 0.5 is the same image as every mean moved by +0.5 pixel.
        Ks = Ks.clone()
        Ks[:, 0, 2] += 0.5
        Ks[:, 1, 2] += 0.5
    elif FAULT == 'no_eps2d':
        options['eps2d'] = 0.0
    elif FAULT == 'antialiased_for_classic' and options.get('rasterize_mode', 'classic') == 'classic':
        options['rasterize_mode'] = 'antialiased'
    elif FAULT == 'alpha_clamp_0.99':
        # Mark metalsplat's own library as already set, so that its clamp of 0.99 stays.
        from metalsplat.ops import rasterize as metalsplat_rasterize
        metalsplat_rasterize._cumlx_alpha_max = gsplat_metalsplat.GSPLAT_ALPHA_MAX
    elif FAULT == 'means_gradient_1.1':
        means = _ScaleGradient.apply(means, 1.1)
    elif FAULT == 'colors_gradient_zero':
        colors = colors.detach()
    return gsplat_metalsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, **options)
