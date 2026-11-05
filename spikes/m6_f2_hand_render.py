"""What gsplat's documented formula draws, in NumPy, with no metalsplat code. The adapter is checked against this, and the training program takes its target images from it.

World-to-camera view matrix, wxyz quaternions, pixel centers at +0.5, covariance J R diag(s^2) R^T J^T + eps2d, alpha min(alpha_max, opacity exp(-d^T cov^-1 d / 2)), alpha below 1/255 skipped, front to back. It is meant for Gaussians that lie well inside the frame.

alpha_max is 0.999, from gsplat 1.5.3's source (RasterizeToPixels3DGSFwd.cu). It was 0.99, from the original 3DGS paper, until 2026-11-06. No scene of validation/m6-f2-adapter-run.json and no training target has an alpha above 0.9, so that change does not move a value there.
"""
from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def look_at(eye, target, up) -> np.ndarray:
    """A world-to-camera matrix in gsplat's convention: +x right, +y down, +z forward."""
    eye, target, up = (np.asarray(item, dtype=np.float64) for item in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    view = np.eye(4)
    view[:3, :3] = np.stack([right, down, forward])
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def render(means, quats, scales, opacities, colors, viewmat, K, width, height, near_plane=0.01, far_plane=1e10, eps2d=0.3, antialiased=False, background=None, pixel_offset=0.5, alpha_max=0.999):
    """Returns ([H, W, 3] image, [H, W] alpha) as float64 arrays."""
    means, quats, scales, opacities, colors, viewmat, K = (np.asarray(item, dtype=np.float64) for item in (means, quats, scales, opacities, colors, viewmat, K))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    camera_points = means @ R.T + t
    xs, ys = np.meshgrid(np.arange(width) + pixel_offset, np.arange(height) + pixel_offset)
    color = np.zeros((height, width, 3))
    transmittance = np.ones((height, width))
    for index in np.argsort(camera_points[:, 2], kind='stable'):
        x, y, z = camera_points[index]
        if not near_plane < z < far_plane:
            continue
        rotation = R @ quat_wxyz_to_matrix(quats[index])
        covariance = rotation @ np.diag(scales[index] ** 2) @ rotation.T
        jacobian = np.array([[fx / z, 0.0, -fx * x / z ** 2], [0.0, fy / z, -fy * y / z ** 2]])
        raw = jacobian @ covariance @ jacobian.T
        blurred = raw + eps2d * np.eye(2)
        strength = opacities[index]
        if antialiased:
            strength = strength * np.sqrt(np.clip(np.linalg.det(raw) / np.linalg.det(blurred), 0.0, 1.0))
        conic = np.linalg.inv(blurred)
        dx, dy = xs - (fx * x / z + cx), ys - (fy * y / z + cy)
        power = -0.5 * (conic[0, 0] * dx * dx + 2 * conic[0, 1] * dx * dy + conic[1, 1] * dy * dy)
        alpha = np.minimum(alpha_max, strength * np.exp(power))
        alpha[alpha < 1.0 / 255.0] = 0.0
        color += (transmittance * alpha)[..., None] * colors[index]
        transmittance *= 1.0 - alpha
    if background is not None:
        color += transmittance[..., None] * np.asarray(background, dtype=np.float64)
    return color, 1.0 - transmittance
