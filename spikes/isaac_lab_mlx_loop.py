#!/usr/bin/env python3
"""Let an MLX policy control an Isaac Lab environment on a Mac, with Newton on the CPU.

Run it with the Python of the environment from spikes/isaac_lab_mac.py, with mlx added to it:

    ISAACSIM_ASSET_ROOT=assets venv/bin/python spikes/isaac_lab_mlx_loop.py --controller mlx \
        --checkpoint model_99.pt --output out.json --device cpu presets=newton_mjwarp

--controller zero applies no force. A checkpoint from early in training is the other control.
Each run is one process, because one process holds one simulation.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TASK = "Isaac-Cartpole-Direct"


def load_example():
    spec = importlib.util.spec_from_file_location(
        "verify_policy", REPO / "examples" / "robot-policy" / "verify_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    import gymnasium as gym
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_rl.entrypoints.common import apply_env_overrides
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("mlx", "zero"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task", default=TASK)
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args, hydra_args = setup_preset_cli(parser, None)
    sys.argv = [sys.argv[0], *hydra_args]
    if (args.controller == "mlx") != bool(args.checkpoint):
        parser.error("--checkpoint belongs with --controller mlx, and only there")

    run_mlx = run_torch = None
    if args.controller == "mlx":
        import mlx.core as mx

        from cumlx.verify import mlx_port

        example = load_example()
        layers = example.actor_layers(
            example.actor_state(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        )
        execution = mlx_port.configure_execution("auto", "off")
        run_mlx = example.mlx_policy(layers)
        run_torch = example.torch_policy(layers)

    torch.manual_seed(args.seed)
    env_cfg, _ = resolve_task_config(args.task, "")
    apply_env_overrides(args, env_cfg)

    final: dict = {}
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        env = gym.make(args.task, cfg=env_cfg)
        cleanup.callback(env.close)
        obs, _ = env.reset(seed=args.seed)
        count = env.unwrapped.num_envs
        rewards, terminated, truncated = 0.0, 0, 0
        last_angles, last_upright = [], []
        worst, inference = 0.0, 0.0
        start = time.monotonic()
        with torch.inference_mode():
            for step in range(args.steps):
                state = obs["policy"]
                if run_mlx is None:
                    action = torch.zeros((count, 1))
                else:
                    tick = time.perf_counter()
                    out = run_mlx(mx.array(state.numpy()))
                    mx.eval(out)
                    inference += time.perf_counter() - tick
                    action = torch.from_numpy(np.array(out)).to(state.device)
                    # The same observations through torch: the observations a real rollout visits.
                    reference = run_torch(state).numpy()
                    diff = np.linalg.norm(np.array(out) - reference)
                    worst = max(worst, float(diff / max(np.linalg.norm(reference), 1e-12)))
                obs, reward, done, cut, _ = env.step(action)
                rewards += float(reward.sum())
                terminated += int(done.sum())
                truncated += int(cut.sum())
                if step >= args.steps - 100:
                    angle = torch.atan2(torch.sin(state[:, 1]), torch.cos(state[:, 1])).abs()
                    last_angles.append(float(angle.mean()))
                    last_upright.append(float((angle < 0.1).float().mean()))
        final = {
            "controller": args.controller,
            "checkpoint": Path(args.checkpoint).name if args.checkpoint else None,
            "envs": count,
            "steps": args.steps,
            "seed": args.seed,
            "device": str(env.unwrapped.device),
            "reward_per_env_step": rewards / (count * args.steps),
            "terminated_per_env": terminated / count,
            "timed_out_per_env": truncated / count,
            "mean_abs_pole_rad_last_100": sum(last_angles) / len(last_angles),
            "upright_fraction_last_100": sum(last_upright) / len(last_upright),
            "seconds": round(time.monotonic() - start, 2),
            "pole_limit_note": f"pi/2 = {math.pi / 2:.3f} rad",
        }
        if run_mlx is not None:
            final["mlx_vs_torch_worst_rel_l2"] = worst
            final["mlx_inference_ms_per_step"] = round(1000 * inference / args.steps, 4)
            final["mlx_device"] = execution["device"] if isinstance(execution, dict) else str(execution)
    Path(args.output).write_text(json.dumps(final, indent=1) + "\n")
    print(json.dumps(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
