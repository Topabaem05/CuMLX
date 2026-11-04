#!/usr/bin/env python3
"""Move a robot policy trained in Isaac Lab to MLX, and prove that it computes the same.

Isaac Sim itself does not run on a Mac. A policy is an ordinary network, so it can: train it elsewhere, then run
inference on Apple Silicon. The default rsl_rl actor is an MLP with ELU activations. A real checkpoint is in
validation/isaac-lab-mac/.

    python verify_policy.py                                # random weights, to prove the path
    python verify_policy.py --checkpoint model_1000.pt     # a real rsl_rl checkpoint
    python verify_policy.py --checkpoint model.pt --golden golden.npz   # against rsl_rl's own outputs

Without --golden the reference is a torch version written in this file, so an error in reading the
architecture would be in both sides. validation/isaac-lab-mac/golden.npz holds rsl_rl's own outputs, recorded once.

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

# rsl_rl 5 stores the actor alone in actor_state_dict, as mlp.N. Earlier versions store actor.N in model_state_dict.
ACTOR_PREFIXES = ('mlp.', 'actor.')


def actor_state(loaded: dict) -> dict[str, torch.Tensor]:
    """The part of a loaded checkpoint that holds the actor."""
    for key in ('actor_state_dict', 'model_state_dict'):
        if key in loaded:
            return loaded[key]
    return loaded


def actor_layers(state: dict[str, torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(weight, bias) of each linear layer of the actor, in order.

    A policy with an observation normalizer is refused. This file does not port one, and leaving it out
    would give wrong actions with no error.
    """
    normalizers = sorted(k for k in state if 'normalizer' in k)
    if normalizers:
        raise SystemExit(f'observation normalizer is not supported: {normalizers[:3]}')
    for prefix in ACTOR_PREFIXES:
        numbers = sorted({int(k[len(prefix):].split('.')[0]) for k in state if k.startswith(prefix)})
        if numbers:
            return [(state[f'{prefix}{n}.weight'], state[f'{prefix}{n}.bias']) for n in numbers]
    raise SystemExit(f'no {" or ".join(ACTOR_PREFIXES)} parameters; keys start with {sorted(state)[:3]}')


def random_state(obs: int = 48, hidden: tuple[int, ...] = (512, 256, 128), actions: int = 12) -> dict[str, torch.Tensor]:
    """The default rsl_rl actor shape, for a quadruped velocity task."""
    torch.manual_seed(0)
    sizes = [obs, *hidden, actions]
    state = {}
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layer = torch.nn.Linear(a, b)
        state[f'mlp.{2 * i}.weight'] = layer.weight.detach()
        state[f'mlp.{2 * i}.bias'] = layer.bias.detach()
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


def golden_policy(golden):
    """The outputs recorded from rsl_rl. It refuses an observation that was not recorded."""
    def run(x):
        size = x.shape[0]
        if not np.array_equal(x.numpy(), golden[f'obs_{size}']):
            raise ValueError(f'observation batch {size} is not the recorded one')
        return torch.from_numpy(golden[f'action_{size}'])
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', help='rsl_rl checkpoint. It is read with weights_only=True, so it runs no code')
    ap.add_argument('--golden', help="an npz of rsl_rl's own outputs on recorded observations, such as validation/isaac-lab-mac/golden.npz")
    args = ap.parse_args()
    if args.golden and not args.checkpoint:
        ap.error('--golden needs --checkpoint')
    if args.checkpoint:
        state = actor_state(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    else:
        state = random_state()
    layers = actor_layers(state)
    obs = layers[0][0].shape[1]
    if args.golden:
        golden = dict(np.load(args.golden))
        inputs = [golden[k] for k in sorted((k for k in golden if k.startswith('obs_')), key=lambda k: int(k[4:]))]
        if not inputs or inputs[0].shape[1] != obs:
            raise SystemExit(f'{args.golden} does not hold observations of size {obs}')
        reference = golden_policy(golden)
    else:
        rng = np.random.default_rng(0)
        # Batches of several sizes, because MLX takes another matmul path for one row than for many.
        inputs = [rng.normal(size=(n, obs)).astype(np.float32) for n in (1, 16, 4096)]
        reference = torch_policy(layers)
    report = check_op(reference, mlx_policy(layers), inputs, name='policy', rel_l2=1e-4)
    result = {k: report[k] for k in ('verdict', 'worst_rel_l2', 'failed')} | {
        'weights': args.checkpoint or 'random', 'reference': 'rsl_rl' if args.golden else 'this file',
        'observation_size': obs, 'action_size': int(layers[-1][0].shape[0]),
        'layers': len(layers), 'device': report['execution']['device'], 'batches': [len(x) for x in inputs]}
    print(json.dumps(result))
    return 0 if report['verdict'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
