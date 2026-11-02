"""Does Genesis load a real Isaac Lab USD articulation and simulate it? Usage: probe_usd_genesis.py <file.usd>

Gate: the two joints load, the pole released at 0.1 rad falls past 0.5 rad under gravity, and it stays
within 0.01 rad of 0.1 with gravity off. The second case stops a joint that never moves from passing.
"""
import json
import sys

import genesis as gs
import numpy as np


def plain(values):
    return np.asarray(values.cpu() if hasattr(values, 'cpu') else values, dtype=float).ravel().tolist()


def release(gravity):
    gs.init(backend=gs.metal, logging_level='warning')
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01, gravity=gravity))
    entities = scene.add_stage(gs.morphs.USD(file=sys.argv[1]))
    scene.build()
    entities = entities if isinstance(entities, (list, tuple)) else [entities]
    robot = max(entities, key=lambda e: e.n_dofs)
    joints = [j.name for j in robot.joints]
    pole = next(j for j in robot.joints if j.name.endswith('cart_to_pole'))
    index = pole.dofs_idx_local[0]
    robot.set_dofs_position([0.1], [index])
    history = []
    for step in range(100):
        scene.step()
        if step % 25 == 24:
            history.append(plain(robot.get_dofs_position([index]))[0])
    gs.destroy()
    return joints, history


joints, falling = release((0, 0, -9.81))
_, still = release((0, 0, 0))
print(json.dumps({'probe': 'usd_genesis', 'joints': joints, 'pole_rad_with_gravity': falling, 'pole_rad_without_gravity': still}), flush=True)
passed = len(joints) == 2 and abs(falling[-1]) > 0.5 and abs(still[-1] - 0.1) < 0.01
print(json.dumps({'probe': 'usd_genesis', 'gate': 'falls past 0.5 rad; stays at 0.1 rad without gravity', 'pass': passed}), flush=True)
