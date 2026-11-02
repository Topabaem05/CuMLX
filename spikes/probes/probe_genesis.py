"""Does Genesis run parallel quadrupeds on Metal? Gate: 4096 environments, at least 3x the CPU at the same size.

The timer starts after a warm-up, so kernel compilation is reported apart from throughput.
The scene is the bundled Go2 on a plane under PD control. 'stand' holds one pose, so every environment is the same and
contacts settle: this flatters the throughput. 'random' sends each environment different joint targets at each step.
"""
import json
import time

import genesis as gs
import numpy as np

JOINTS = {
    'FL_hip_joint': 0.0, 'FR_hip_joint': 0.0, 'RL_hip_joint': 0.0, 'RR_hip_joint': 0.0,
    'FL_thigh_joint': 0.8, 'FR_thigh_joint': 0.8, 'RL_thigh_joint': 1.0, 'RR_thigh_joint': 1.0,
    'FL_calf_joint': -1.5, 'FR_calf_joint': -1.5, 'RL_calf_joint': -1.5, 'RR_calf_joint': -1.5,
}


def measure(backend, n, mode, warm=20, steps=100):
    gs.init(backend=backend, logging_level='warning')
    start = time.perf_counter()
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.02, substeps=2))
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf', pos=(0, 0, 0.42), quat=(1, 0, 0, 0)))
    scene.build(n_envs=n)
    dofs = [robot.get_joint(name).dofs_idx_local[0] for name in JOINTS]
    robot.set_dofs_kp([20.0] * len(dofs), dofs)
    robot.set_dofs_kv([0.5] * len(dofs), dofs)
    target = np.tile(np.array(list(JOINTS.values()), dtype=np.float32), (n, 1))
    robot.control_dofs_position(target, dofs)
    for _ in range(warm):
        scene.step()
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.3, (steps, n, len(dofs))).astype(np.float32) if mode == 'random' else None
    ready = time.perf_counter()
    for step in range(steps):
        if noise is not None:
            robot.control_dofs_position(target + noise[step], dofs)
        scene.step()
    height = robot.get_pos()[..., 2]
    height = height.cpu().numpy() if hasattr(height, 'cpu') else np.asarray(height)  # forces the GPU to finish
    end = time.perf_counter()
    result = {'env_steps_per_s': steps * n / (end - ready), 'compile_and_warm_s': ready - start,
              'height_mean': float(height.mean()), 'height_std': float(height.std())}
    gs.destroy()
    return result


for mode in ('stand', 'random'):
    rates = {}
    for name, backend in (('metal', gs.metal), ('cpu', gs.cpu)):
        for n in (512, 4096):
            label = {'probe': 'genesis', 'scene': 'go2', 'mode': mode, 'backend': name, 'envs': n}
            try:
                rates[(name, n)] = measure(backend, n, mode)
                print(json.dumps({**label, **rates[(name, n)]}), flush=True)
            except Exception as exc:  # noqa: BLE001 - a failure is the result
                print(json.dumps({**label, 'error': f'{type(exc).__name__}: {exc}'[:300]}), flush=True)
                gs.destroy()
    for n in (512, 4096):
        if ('metal', n) in rates and ('cpu', n) in rates:
            ratio = rates[('metal', n)]['env_steps_per_s'] / rates[('cpu', n)]['env_steps_per_s']
            print(json.dumps({'probe': 'genesis', 'mode': mode, 'envs': n, 'metal_over_cpu': ratio, 'gate': 'at least 3x at 4096',
                              'pass': ratio >= 3 if n == 4096 else None}), flush=True)
