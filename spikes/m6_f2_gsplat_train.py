"""An ordinary gsplat training program for a CUDA machine, on a synthetic scene. It has no Mac-specific line, and its target images come from a NumPy hand render, so the rasterizer does not draw its own answers.

    python spikes/m6_f2_gsplat_train.py --mode right --output out.json
    --mode frozen          the learning rate is 0, so the render stays at its start
    --mode wrong_cameras   each training camera gets another view's image, so nothing true can be learned
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import gsplat
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6_f2_hand_render import look_at  # noqa: E402
from m6_f2_hand_render import render as hand_render

SIZE = 128
K = np.array([[SIZE, 0.0, SIZE / 2], [0.0, SIZE, SIZE / 2], [0.0, 0.0, 1.0]])

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['right', 'frozen', 'wrong_cameras'], required=True)
parser.add_argument('--iterations', type=int, default=1500)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--output', required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)


def camera_at(azimuth, elevation):
    eye = 3.0 * np.array([math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
    return look_at(eye, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])


truth = {'means': rng.uniform(-0.7, 0.7, (48, 3)), 'quats': rng.normal(size=(48, 4)), 'scales': rng.uniform(0.05, 0.15, (48, 3)), 'opacities': rng.uniform(0.5, 0.9, 48), 'colors': rng.uniform(0.0, 1.0, (48, 3))}
train_views = [camera_at(2 * math.pi * i / 8, e) for e in (-0.5, 0.0, 0.5) for i in range(8)]
test_views = [camera_at(2 * math.pi * (i + 0.5) / 8, 0.25) for i in range(8)]


def target(view):
    return torch.tensor(hand_render(truth['means'], truth['quats'], truth['scales'], truth['opacities'], truth['colors'], view, K, SIZE, SIZE)[0], dtype=torch.float32, device=device)


def as_camera(view):
    return torch.tensor(view, dtype=torch.float32, device=device)[None]


train_images = [target(view) for view in train_views]
test_images = [target(view) for view in test_views]
if args.mode == 'wrong_cameras':
    train_images = train_images[5:] + train_images[:5]
train_cameras = [as_camera(view) for view in train_views]
test_cameras = [as_camera(view) for view in test_views]
Ks = torch.tensor(K, dtype=torch.float32, device=device)[None]

n = 200
means = torch.nn.Parameter(torch.rand(n, 3, device=device) * 2 - 1)
quats = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1))
log_scales = torch.nn.Parameter(torch.full((n, 3), math.log(0.02), device=device))
raw_opacities = torch.nn.Parameter(torch.zeros(n, device=device))
raw_colors = torch.nn.Parameter(torch.logit(torch.rand(n, 3, device=device).clamp(0.01, 0.99)))


def draw(camera):
    return gsplat.rasterization(means, quats, log_scales.exp(), raw_opacities.sigmoid(), raw_colors.sigmoid(), camera, Ks, SIZE, SIZE)[0][0]


def heldout():
    with torch.no_grad():
        return sum(float(-10 * torch.log10(((draw(camera) - image) ** 2).mean())) for camera, image in zip(test_cameras, test_images, strict=True)) / len(test_cameras)


start_psnr = heldout()
rate = 0.0 if args.mode == 'frozen' else 1.0
optimizer = torch.optim.Adam([
    {'params': [means], 'lr': 0.01 * rate},
    {'params': [log_scales], 'lr': 0.02 * rate},
    {'params': [quats], 'lr': 0.01 * rate},
    {'params': [raw_opacities], 'lr': 0.05 * rate},
    {'params': [raw_colors], 'lr': 0.02 * rate},
])
order = torch.randperm(args.iterations) % len(train_cameras)
begin = time.perf_counter()
first_loss = last_loss = None
image_device = None
for index in order.tolist():
    image = draw(train_cameras[index])
    image_device = image.device.type
    loss = (image - train_images[index]).abs().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    last_loss = float(loss)
    first_loss = last_loss if first_loss is None else first_loss
seconds = time.perf_counter() - begin
result = {'mode': args.mode, 'iterations': args.iterations, 'seed': args.seed, 'device': image_device, 'heldout_psnr_start': start_psnr, 'heldout_psnr_end': heldout(), 'first_loss': first_loss, 'last_loss': last_loss, 'seconds': seconds, 'gaussians': n, 'views_train': len(train_cameras), 'views_heldout': len(test_cameras)}
with open(args.output, 'w') as handle:
    json.dump(result, handle, indent=1)
print(json.dumps(result))
