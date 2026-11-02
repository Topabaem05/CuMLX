#!/usr/bin/env python3
"""Record what rsl_rl's own MLPModel computes from a checkpoint, as a reference for an MLX port.

Run it with the Python of the environment that trained the policy:

    venv/bin/python spikes/rsl_rl_golden.py model.pt golden.npz

The architecture comes from the arguments, and load_state_dict is strict. A wrong shape or a key
that the model does not expect is an error.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--activation", default="elu")
    args = parser.parse_args()

    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = loaded["actor_state_dict"]
    obs_size = state["mlp.0.weight"].shape[1]
    actions = state[f"mlp.{2 * len(args.hidden)}.weight"].shape[0]
    example = TensorDict({"policy": torch.zeros(1, obs_size)}, batch_size=[1])
    actor = MLPModel(
        example,
        {"actor": ["policy"]},
        "actor",
        actions,
        hidden_dims=args.hidden,
        activation=args.activation,
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    actor.load_state_dict(state, strict=True)
    actor.eval()

    rng = np.random.default_rng(0)
    arrays = {}
    with torch.no_grad():
        for size in (1, 16, 4096):
            obs = rng.normal(size=(size, obs_size)).astype(np.float32)
            batch = TensorDict({"policy": torch.from_numpy(obs)}, batch_size=[size])
            arrays[f"obs_{size}"] = obs
            arrays[f"action_{size}"] = actor(batch).numpy()
    np.savez(args.output, **arrays)
    print(f"{args.output}: observation {obs_size}, action {actions}, torch {torch.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
