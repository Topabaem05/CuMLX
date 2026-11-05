"""An ordinary gsplat training program for a CUDA machine. It has no Mac-specific line, because F2 asks whether such a program runs as it is."""
import argparse
import json

import gsplat
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
n = 64
means = torch.nn.Parameter(torch.rand(n, 3, device=device) * 2 - 1)
quats = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1))
scales = torch.nn.Parameter(torch.full((n, 3), 0.1, device=device))
opacities = torch.nn.Parameter(torch.full((n,), 0.5, device=device))
colors = torch.nn.Parameter(torch.rand(n, 3, device=device))
viewmat = torch.eye(4, device=device)[None]
viewmat[0, 2, 3] = 4.0
K = torch.tensor([[[128.0, 0.0, 64.0], [0.0, 128.0, 64.0], [0.0, 0.0, 1.0]]], device=device)
target = torch.zeros(1, 128, 128, 3, device=device)
target[:, 40:88, 40:88] = torch.tensor([0.8, 0.2, 0.1], device=device)
optimizer = torch.optim.Adam([means, quats, scales, opacities, colors], lr=0.01)
losses = []
for _ in range(50):
    image, _, _ = gsplat.rasterization(means, quats, scales, opacities, colors, viewmat, K, 128, 128)
    loss = (image - target).abs().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(float(loss))
with open(args.output, 'w') as handle:
    json.dump({'first_loss': losses[0], 'last_loss': losses[-1], 'device': device}, handle)
