#!/usr/bin/env python3
"""M4: drive cumlx.sim from the command line. Run each command through spikes/run_guarded.py, in the Python of an environment with the engines.

    m4_sim.py smoke   --task cartpole --engine mujoco --envs 4 --steps 20
    m4_sim.py record  --task cartpole --engine genesis --seed 101 --out a.npz [--gravity -10.3] [--action-scale 1.1] [--kp-scale 0.5]
    m4_sim.py compare --task cartpole a.npz b.npz          # exit 0 inside the limits, 1 outside
    m4_sim.py train   --task cartpole --engine genesis --envs 256 --iterations 150 --seed 1 --log-dir runs/a
    m4_sim.py eval    --task cartpole --engine genesis --checkpoint runs/a/model_149.pt --output e.json [--controller mlx|zero] [--forward 0.8]
    m4_sim.py rate    --task go2_flat --engine genesis --envs 4096 --steps 300

The limits are in validation/m4-sim-limits.json. They were committed before the first comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx.sim import compare_trajectories, get_task  # noqa: E402
from cumlx.sim.backends import GenesisBackend, MujocoBackend  # noqa: E402
from cumlx.sim.env import TaskEnv  # noqa: E402

LIMITS = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm4-sim-limits.json').read_text())
UPRIGHT_RADIANS = 0.2


def make_backend(task: Any, args: argparse.Namespace) -> Any:
    gravity = getattr(args, 'gravity', -9.81)
    if args.engine == 'genesis':
        return GenesisBackend(task, args.envs, physics=args.physics, seed=args.seed, gravity=gravity)
    return MujocoBackend(task, args.envs, gravity=gravity, substep_scale=getattr(args, 'substep_scale', 1))


def cmd_smoke(args: argparse.Namespace) -> int:
    import torch
    task = get_task(args.task)
    env = TaskEnv(task, make_backend(task, args), seed=args.seed)
    begin = time.perf_counter()
    done_count = 0
    for _ in range(args.steps):
        obs, reward, done, _ = env.step(torch.zeros((args.envs, task.num_actions), device=env.device))
        done_count += int(done.sum())
    print(json.dumps({'task': task.name, 'engine': args.engine, 'obs_shape': list(obs['policy'].shape), 'finite': bool(obs['policy'].isfinite().all()),
                      'reward_mean': float(reward.mean()), 'done': done_count, 'seconds': round(time.perf_counter() - begin, 3), 'device': str(env.device)}))
    return 0


def open_loop_actions(task: Any, spec: dict[str, Any], seed: int, worlds: int) -> np.ndarray:
    """The same actions for every engine: new random values every `action_hold_steps`, from NumPy and the seed alone."""
    rng = np.random.default_rng(seed)
    holds = math.ceil(spec['control_steps'] / spec['action_hold_steps'])
    shape = (holds, worlds, task.num_actions)
    values = rng.uniform(-spec['action_range'], spec['action_range'], shape) if 'action_range' in spec else rng.normal(0.0, spec['action_std'], shape)
    return np.repeat(values, spec['action_hold_steps'], axis=0)[:spec['control_steps']].astype(np.float32)


def cmd_record(args: argparse.Namespace) -> int:
    import torch
    task = get_task(args.task)
    if args.kp_scale != 1.0:
        task = replace(task, actuation=replace(task.actuation, kp=task.actuation.kp * args.kp_scale))
    if args.action_scale != 1.0:
        task = replace(task, actuation=replace(task.actuation, scale=task.actuation.scale * args.action_scale))
    spec = LIMITS['trajectory'][args.measure or args.task]
    args.envs = spec['worlds']
    backend = make_backend(task, args)
    backend.reset(torch.arange(args.envs, device=backend.device), task.reset(args.envs, torch.Generator().manual_seed(args.seed), task))
    actions = open_loop_actions(task, spec, args.seed, args.envs)
    rows = []
    begin = time.perf_counter()
    for step_actions in actions:
        backend.apply(torch.from_numpy(step_actions).to(backend.device))
        backend.step()
        state = backend.state()
        values = [state.joint_pos]
        if state.base_pos is not None:
            values.append(state.base_pos)
        rows.append(torch.cat(values, dim=1).detach().cpu().numpy().astype(np.float64))
    np.savez_compressed(args.out, values=np.stack(rows), engine=args.engine, device=str(backend.device), seconds=time.perf_counter() - begin)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    spec = LIMITS['trajectory'][args.measure or args.task]
    a, b = np.load(args.reference), np.load(args.other)
    result = compare_trajectories(a['values'], b['values'], median_limit=spec['median_limit'], p90_limit=spec['p90_limit'])
    print(json.dumps({'reference': Path(args.reference).name, 'other': Path(args.other).name, **result}))
    return 0 if result['pass'] else 1


RSL_RL_CFG: dict[str, Any] = {
    'algorithm': {'class_name': 'PPO', 'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001,
                  'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 'use_clipped_value_loss': True,
                  'value_loss_coef': 1.0},
    'actor': {'class_name': 'MLPModel', 'hidden_dims': [], 'activation': 'elu',
              'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'scalar'}},
    'critic': {'class_name': 'MLPModel', 'hidden_dims': [], 'activation': 'elu'},
    'obs_groups': {'actor': ['policy'], 'critic': ['policy']},
    'num_steps_per_env': 24, 'save_interval': 50, 'logger': 'tensorboard', 'run_name': 'cumlx_sim',
}
HIDDEN = {'cartpole': [32, 32], 'go2_flat': [512, 256, 128]}


def cmd_train(args: argparse.Namespace) -> int:
    import copy

    from rsl_rl.runners import OnPolicyRunner
    task = get_task(args.task)
    env = TaskEnv(task, make_backend(task, args), seed=args.seed)
    cfg = copy.deepcopy(RSL_RL_CFG)
    cfg['actor']['hidden_dims'] = cfg['critic']['hidden_dims'] = HIDDEN[args.task]
    cfg['seed'] = args.seed
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, cfg, str(log_dir), device=args.learner or str(env.device))
    begin = time.perf_counter()
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    seconds = time.perf_counter() - begin
    summary = {'task': task.name, 'engine': args.engine, 'physics': args.physics if args.engine == 'genesis' else 'cpu', 'learner': args.learner or str(env.device), 'envs': args.envs,
               'iterations': args.iterations, 'seed': args.seed, 'seconds': round(seconds, 2),
               'env_steps_per_second': round(args.envs * cfg['num_steps_per_env'] * args.iterations / seconds)}
    (log_dir / 'train.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch
    run_mlx = wrong = reference = None
    verdict_inputs: list[np.ndarray] = []
    if args.checkpoint:
        import mlx.core as mx
        import mlx.nn as nn

        from cumlx.sim.policy import load_actor_layers, mlx_policy, rsl_rl_actor, verify_policy
        from cumlx.verify import mlx_port
        layers = load_actor_layers(args.checkpoint)
        mlx_port.configure_execution('auto', 'off')
        run_mlx, wrong, reference = mlx_policy(layers), mlx_policy(layers, activation=nn.relu), rsl_rl_actor(args.checkpoint)
    task = get_task(args.task)
    command = (args.forward, 0.0, 0.0) if task.command_ranges else None
    env = TaskEnv(task, make_backend(task, args), seed=args.seed, fixed_command=command)
    obs = env.reset()
    tail, speeds, errors = [], [], []
    falls, rewards = 0, 0.0
    inference = 0.0
    with torch.inference_mode():
        for step in range(args.steps):
            state = obs['policy'].detach().cpu().numpy().astype(np.float32)
            if step % max(1, args.steps // 8) == 0:
                verdict_inputs.append(state.copy())
            if run_mlx is None:
                actions = torch.zeros((args.envs, task.num_actions), device=env.device)
            else:
                begin = time.perf_counter()
                out = run_mlx(mx.array(state))
                mx.eval(out)
                inference += time.perf_counter() - begin
                actions = torch.from_numpy(np.array(out)).to(env.device)
            obs, reward, done, extras = env.step(actions)
            rewards += float(reward.mean())
            falls += int(extras['terminated'].sum())
            if task.name == 'cartpole' and step >= args.steps - 100:
                tail.append(float((env.state.joint_pos[:, 1].abs() < UPRIGHT_RADIANS).float().mean()))
            if task.name == 'go2_flat' and step >= min(100, args.steps // 2):
                from cumlx.sim.task import rotate_inverse
                velocity = rotate_inverse(env.state.base_quat, env.state.base_lin_vel)
                speeds.append(float(velocity[:, 0].mean()))
                errors.append(float((env.commands[:, :2] - velocity[:, :2]).norm(dim=1).mean()))
    result: dict[str, Any] = {'task': task.name, 'engine': args.engine, 'controller': 'mlx' if run_mlx else 'zero', 'checkpoint': Path(args.checkpoint).name if args.checkpoint else None,
                              'envs': args.envs, 'steps': args.steps, 'seed': args.seed, 'reward_per_step': rewards / args.steps, 'falls_per_world': falls / args.envs}
    if tail:
        result['upright_fraction_last_100'] = sum(tail) / len(tail)
    if speeds:
        result.update(forward_command=args.forward, mean_forward_speed=sum(speeds) / len(speeds), mean_velocity_error=sum(errors) / len(errors))
    if run_mlx is not None:
        limit = LIMITS['training']['policy_in_mlx']['limit_rel_l2']
        result['mlx_ms_per_step'] = round(1000 * inference / args.steps, 3)
        result['check_op_against_rsl_rl'] = verify_policy(reference, run_mlx, verdict_inputs, rel_l2=limit)
        result['check_op_wrong_activation'] = verify_policy(reference, wrong, verdict_inputs, rel_l2=limit)
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, dict)}))
    return 0


def cmd_rate(args: argparse.Namespace) -> int:
    import torch
    task = get_task(args.task)
    env = TaskEnv(task, make_backend(task, args), seed=args.seed)
    actions = torch.zeros((args.envs, task.num_actions), device=env.device)
    for _ in range(20):
        env.step(actions)
    begin = time.perf_counter()
    for _ in range(args.steps):
        env.step(actions)
    seconds = time.perf_counter() - begin
    print(json.dumps({'task': task.name, 'engine': args.engine, 'envs': args.envs, 'steps': args.steps, 'env_steps_per_second': round(args.envs * args.steps / seconds), 'seconds': round(seconds, 3),
                      'actions': 'zero, no learner, no policy', 'includes': 'task code in cumlx.sim'}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    def add(name: str, handler: Any, *, engine: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument('--task', required=True)
        if engine:
            sub.add_argument('--engine', required=True, choices=['genesis', 'mujoco'])
            sub.add_argument('--physics', default='metal', choices=['metal', 'cpu'])
            sub.add_argument('--seed', type=int, default=1)
            sub.add_argument('--envs', type=int, default=4)
        return sub

    sub = add('smoke', cmd_smoke)
    sub.add_argument('--steps', type=int, default=20)
    sub = add('record', cmd_record)
    sub.add_argument('--out', required=True)
    sub.add_argument('--measure', help='a measure in validation/m4-sim-limits.json, if not the one named after the task')
    sub.add_argument('--gravity', type=float, default=-9.81)
    sub.add_argument('--action-scale', type=float, default=1.0)
    sub.add_argument('--kp-scale', type=float, default=1.0)
    sub.add_argument('--substep-scale', type=int, default=1)
    sub = add('compare', cmd_compare, engine=False)
    sub.add_argument('--measure')
    sub.add_argument('reference')
    sub.add_argument('other')
    sub = add('train', cmd_train)
    sub.add_argument('--iterations', type=int, required=True)
    sub.add_argument('--log-dir', required=True)
    sub.add_argument('--learner')
    sub = add('eval', cmd_eval)
    sub.add_argument('--checkpoint', help='without it, the no-action control runs')
    sub.add_argument('--steps', type=int, default=500)
    sub.add_argument('--forward', type=float, default=0.8)
    sub.add_argument('--output', required=True)
    sub = add('rate', cmd_rate)
    sub.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
