#!/usr/bin/env python3
"""F2 with metalsplat's own API: 3D Gaussian Splatting on a synthetic scene. The rules are in validation/m6-f1-f2-limits.json.

    python spikes/m6_f2_train.py --mode right --output out.json
    --mode frozen          the learning rate is 0, so the render stays at its start
    --mode wrong_cameras   each training camera gets another view's image, so nothing true can be learned
"""
from __future__ import annotations

import argparse
import json
import math
import time

SIZE = 128


def camera_at(Camera, torch, azimuth: float, elevation: float):
    eye = 3.0 * torch.tensor([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)], dtype=torch.float32)
    return Camera.look_at(eye, torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]), fx=SIZE, fy=SIZE, cx=SIZE / 2, cy=SIZE / 2, img_width=SIZE, img_height=SIZE).to('mps')


def psnr(torch, image, target) -> float:
    return float(-10 * torch.log10(((image - target) ** 2).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['right', 'frozen', 'wrong_cameras'], required=True)
    parser.add_argument('--iterations', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    from metalsplat import Camera, GaussianModel, render
    import torch

    torch.manual_seed(args.seed)
    truth = GaussianModel(
        (torch.rand(48, 3) * 1.4 - 0.7).to('mps'),
        scales=(torch.rand(48, 3) * 0.1 + 0.05).to('mps'),
        colors=torch.rand(48, 3).to('mps'),
    ).to('mps')
    rings = [-0.5, 0.0, 0.5]
    train_cameras = [camera_at(Camera, torch, 2 * math.pi * i / 8, e) for e in rings for i in range(8)]
    test_cameras = [camera_at(Camera, torch, 2 * math.pi * (i + 0.5) / 8, 0.25) for i in range(8)]

    def draw(model, camera):
        with torch.no_grad():
            image = render(model, camera)
            torch.mps.synchronize()
        return image

    train_images = [draw(truth, camera) for camera in train_cameras]
    test_images = [draw(truth, camera) for camera in test_cameras]
    if args.mode == 'wrong_cameras':
        train_images = train_images[5:] + train_images[:5]

    model = GaussianModel.random(n=200, bound=1.0, device='mps')

    def heldout() -> float:
        return sum(psnr(torch, draw(model, camera), image) for camera, image in zip(test_cameras, test_images, strict=True)) / len(test_cameras)

    start_psnr = heldout()
    rate = 0.0 if args.mode == 'frozen' else 1.0
    optimizer = torch.optim.Adam([
        {'params': [model.means], 'lr': 0.01 * rate},
        {'params': [model.raw_scales], 'lr': 0.02 * rate},
        {'params': [model.raw_quats], 'lr': 0.01 * rate},
        {'params': [model.raw_opacities], 'lr': 0.05 * rate},
        {'params': [model.raw_colors], 'lr': 0.02 * rate},
    ])
    order = torch.randperm(args.iterations) % len(train_cameras)
    begin = time.perf_counter()
    first_loss = last_loss = None
    device = None
    for index in order.tolist():
        image = render(model, train_cameras[index])
        device = image.device.type
        loss = (image - train_images[index]).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss)
        first_loss = last_loss if first_loss is None else first_loss
    seconds = time.perf_counter() - begin
    result = {'mode': args.mode, 'iterations': args.iterations, 'seed': args.seed, 'device': device, 'heldout_psnr_start': start_psnr, 'heldout_psnr_end': heldout(),
              'first_loss': first_loss, 'last_loss': last_loss, 'seconds': seconds, 'gaussians': 200, 'views_train': len(train_cameras), 'views_heldout': len(test_cameras)}
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
