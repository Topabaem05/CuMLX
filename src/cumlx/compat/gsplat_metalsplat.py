"""gsplat.rasterization on metalsplat, for an unmodified gsplat program on a Mac.

    cumlx run train.py --substitute gsplat=cumlx.compat.gsplat_metalsplat

This takes the activated values that gsplat takes, and it calls metalsplat's own kernels. An option that this does not implement raises NotImplementedError, because a program that lost one in silence would get a wrong result. It covers pinhole cameras, RGB, and the classic and antialiased modes, and nothing else in the gsplat package. The rules and results are in validation/m6-f2-adapter-limits.json and validation/m6-f2-adapter-run.json. It has not been compared with gsplat on CUDA.
"""

from __future__ import annotations

from typing import Any

import torch

_ONLY_DEFAULT = {
    'sh_degree': None, 'render_mode': 'RGB', 'camera_model': 'pinhole', 'radius_clip': 0.0, 'sparse_grad': False, 'absgrad': False, 'distributed': False,
    'segmented': False, 'covars': None, 'with_ut': False, 'with_eval3d': False, 'return_normals': False, 'global_z_order': True, 'rays': None,
}


def _refuse(what: str) -> NotImplementedError:
    return NotImplementedError(f'cumlx.compat.gsplat_metalsplat does not implement {what}, so it stops here and does not ignore it')


def _check_shapes(means: torch.Tensor, opacities: torch.Tensor, colors: torch.Tensor | None, viewmats: torch.Tensor, Ks: torch.Tensor) -> tuple[int, int]:
    if means.dim() != 2 or opacities.dim() != 1:
        raise _refuse(f'batched Gaussians (means {tuple(means.shape)}, opacities {tuple(opacities.shape)})')
    if viewmats.dim() != 3 or Ks.dim() != 3 or viewmats.shape[0] != Ks.shape[0]:
        raise _refuse(f'cameras of this shape (viewmats {tuple(viewmats.shape)}, Ks {tuple(Ks.shape)})')
    if viewmats.requires_grad:
        raise _refuse('gradients to viewmats')
    n, cameras = means.shape[0], viewmats.shape[0]
    if colors is None or colors.shape not in ((n, 3), (cameras, n, 3)):
        raise _refuse(f'colors of shape {None if colors is None else tuple(colors.shape)}: it takes [N, 3] or [C, N, 3]')
    return n, cameras


def rasterization(means: torch.Tensor, quats: torch.Tensor, scales: torch.Tensor, opacities: torch.Tensor, colors: torch.Tensor | None, viewmats: torch.Tensor, Ks: torch.Tensor,
                  width: int, height: int, near_plane: float = 0.01, far_plane: float = 1e10, eps2d: float = 0.3, packed: bool = True, tile_size: int | None = None,
                  backgrounds: torch.Tensor | None = None, rasterize_mode: str = 'classic', channel_chunk: int = 32, **options: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Returns ([C, H, W, 3] colors, [C, H, W, 1] alphas, meta). packed and channel_chunk choose how gsplat uses its memory, and they do not change a result, so they are accepted."""
    for name, value in options.items():
        if name not in _ONLY_DEFAULT and value is not None:
            raise _refuse(f'the option {name}')
        if name in _ONLY_DEFAULT and value != _ONLY_DEFAULT[name] and not (value is None and _ONLY_DEFAULT[name] is None):
            raise _refuse(f'{name}={value!r}')
    if rasterize_mode not in ('classic', 'antialiased'):
        raise _refuse(f'rasterize_mode={rasterize_mode!r}')
    _, cameras = _check_shapes(means, opacities, colors, viewmats, Ks)

    from metalsplat.ops.project import project_gaussians
    from metalsplat.ops.rasterize import rasterize_gaussians
    from metalsplat.ops.tiling import DEFAULT_TILE_SIZE

    tiles = DEFAULT_TILE_SIZE if tile_size is None else tile_size
    unit_quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    images, alphas = [], []
    for camera in range(cameras):
        K = Ks[camera]
        means2d, depths, conics, radii, valid, compensation = project_gaussians(
            means, scales, unit_quats, viewmats[camera, :3, :3], viewmats[camera, :3, 3], float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), width, height, near=near_plane, eps2d=eps2d)
        kept = valid * (depths < far_plane).to(valid.dtype)
        strength = opacities * compensation if rasterize_mode == 'antialiased' else opacities
        rgb = colors if colors.dim() == 2 else colors[camera]
        background = None if backgrounds is None else backgrounds[camera]
        image, _, final_T = rasterize_gaussians(means2d, depths, conics, strength, rgb, radii, kept, width, height, tile_size=tiles, background=background, return_aux=True)
        if torch.is_grad_enabled():
            # gsplat's alpha has gradients and final_T has none. White Gaussians on black draw 1 - T, with gradients, for the price of a second pass.
            alpha = rasterize_gaussians(means2d, depths, conics, strength, torch.ones_like(rgb), radii, kept, width, height, tile_size=tiles)[..., :1]
        else:
            alpha = (1.0 - final_T)[..., None]
        images.append(image)
        alphas.append(alpha)
    meta = {'width': width, 'height': height, 'n_cameras': cameras, 'near_plane': near_plane, 'far_plane': far_plane, 'eps2d': eps2d, 'rasterize_mode': rasterize_mode, 'tile_width': tiles, 'tile_height': tiles}
    return torch.stack(images), torch.stack(alphas), meta
