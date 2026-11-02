#!/usr/bin/env python3
"""Move a robot policy trained in Isaac Lab to MLX, and prove that it computes the same.

Isaac Sim itself does not run on a Mac. A policy is an ordinary network, so it can: train and simulate on an NVIDIA
machine, then run inference on Apple Silicon. The default rsl_rl actor is an MLP with ELU activations.

    python verify_policy.py                                # random weights, to prove the path
    python verify_policy.py --checkpoint model_1000.pt     # a real rsl_rl checkpoint

Exit status 0 means the policies match on every test observation.
"""
from __future__ import annotations

import argparse
import json
import sys

import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
import torch

from cumlx import check_op

ACTOR_PREFIX = 'actor.'


def actor_layers(state: dict[str, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(weight, bias) of each linear layer of the actor, in order."""
    numbers = sorted({int(k[len(ACTOR_PREFIX):].split('.')[0]) for k in state if k.startswith(ACTOR_PREFIX)})
    if not numbers:
        raise SystemExit(f'no {ACTOR_PREFIX}* parameters; keys start with {sorted(state)[:3]}')
    return [(state[f'{ACTOR_PREFIX}{n}.weight'], state[f'{ACTOR_PREFIX}{n}.bias']) for n in numbers]


def random_state(obs: int = 48, hidden: tuple[int, ...] = (512, 256, 128), actions: int = 12) -> dict[str, torch.Tensor]:
    """The default rsl_rl actor shape, for a quadruped velocity task."""
    torch.manual_seed(0)
    sizes = [obs, *hidden, actions]
    state = {}
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layer = torch.nn.Linear(a, b)
        state[f'{ACTOR_PREFIX}{2 * i}.weight'] = layer.weight.detach()
        state[f'{ACTOR_PREFIX}{2 * i}.bias'] = layer.bias.detach()
    return state


def torch_policy(layers):
    def run(x):
        for i, (w, b) in enumerate(layers):
            x = torch.nn.functional.linear(x, w.to(torch.float32), b.to(torch.float32))
            if i < len(layers) - 1:
                x = torch.nn.functional.elu(x)
        return x
    return run


def mlx_policy(layers, activation=mxnn.elu):
    converted = [(mx.array(w.to(torch.float32).numpy()), mx.array(b.to(torch.float32).numpy())) for w, b in layers]

    def run(x):
        for i, (w, b) in enumerate(converted):
            x = x @ w.T + b
            if i < len(converted) - 1:
                x = activation(x)
        return x
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', help='rsl_rl checkpoint. It is read with weights_only=True, so it runs no code')
    args = ap.parse_args()
    if args.checkpoint:
        loaded = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = loaded.get('model_state_dict', loaded)
    else:
        state = random_state()
    layers = actor_layers(state)
    obs = layers[0][0].shape[1]
    rng = np.random.default_rng(0)
    # Batches of several sizes, because MLX takes another matmul path for one row than for many.
    inputs = [rng.normal(size=(n, obs)).astype(np.float32) for n in (1, 16, 4096)]
    report = check_op(torch_policy(layers), mlx_policy(layers), inputs, name='policy', rel_l2=1e-4)
    result = {k: report[k] for k in ('verdict', 'worst_rel_l2', 'failed')} | {
        'weights': args.checkpoint or 'random', 'observation_size': obs, 'action_size': int(layers[-1][0].shape[0]),
        'layers': len(layers), 'device': report['execution']['device'], 'batches': [len(x) for x in inputs]}
    print(json.dumps(result))
    return 0 if report['verdict'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
