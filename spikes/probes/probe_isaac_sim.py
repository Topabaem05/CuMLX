"""Isaac Sim: does Kit start, do PhysX and an Isaac Lab articulation run, and does a camera render?

Run it with Isaac Sim's own Python environment. It prints one `RESULT {json}` line per step. A failed step is a result.
The control cases turn gravity off: a body that never moves must not count as physics that works.
"""
import json
import os
import platform
import time
import traceback
import urllib.request


def say(**fields):
    print('RESULT ' + json.dumps(fields), flush=True)


start = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})  # Isaac Sim requires this before its other imports
say(step='kit', seconds=round(time.time() - start, 1), machine=platform.machine())

ASSET = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Classic/Cartpole/'
WORK = os.environ.get('ISAAC_WORK', '/tmp/isaac-assets')
os.makedirs(f'{WORK}/Props', exist_ok=True)
for name in ('cartpole.usd', 'Props/instanceable_meshes.usd'):
    if not os.path.exists(f'{WORK}/{name}'):
        urllib.request.urlretrieve(ASSET + name, f'{WORK}/{name}')

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
import numpy as np  # noqa: E402


def attempt(step, function):
    try:
        function()
    except Exception as exc:  # noqa: BLE001 - a failure is the result
        say(step=step, error=f'{type(exc).__name__}: {exc}'[:400], where=traceback.format_exc().splitlines()[-3][:200])


def new_world(dt, gravity):
    World.clear_instance()
    world = World(physics_dt=dt, rendering_dt=dt, backend='numpy', device='cpu')
    world.get_physics_context().set_gravity(gravity)
    return world


def fall(gravity):
    world = new_world(1 / 240, gravity)
    cube = world.scene.add(DynamicCuboid(prim_path='/World/cube', name='cube', position=np.array([0, 0, 100.0]), size=0.1))
    world.reset()
    begin = time.time()
    for _ in range(240):
        world.step(render=False)
    rate = 240 / (time.time() - begin)
    return float(cube.get_world_pose()[0][2]), rate


def freefall():
    height, rate = fall(-9.81)
    exact = 100 - 0.5 * 9.81
    say(step='freefall', z_after_240_steps=height, exact_1s=exact, abs_error=abs(height - exact), steps_per_s=rate)
    height, _ = fall(0.0)
    say(step='freefall_no_gravity', z_after_240_steps=height, moved=abs(height - 100))


def pole(gravity):
    world = new_world(0.01, gravity)
    add_reference_to_stage(f'{WORK}/cartpole.usd', '/World/cartpole')
    robot = Articulation('/World/cartpole')
    world.scene.add(robot)
    world.reset()
    names = list(robot.dof_names)
    index = names.index('cart_to_pole')
    positions = np.array(robot.get_joint_positions()).reshape(1, -1)
    positions[0, index] = 0.1
    robot.set_joint_positions(positions)
    history = []
    for step in range(100):
        world.step(render=False)
        if step % 25 == 24:
            history.append(float(np.array(robot.get_joint_positions()).reshape(-1)[index]))
    return names, history


def cartpole():
    names, falling = pole(-9.81)
    _, still = pole(0.0)
    say(step='cartpole', dofs=names, pole_rad_with_gravity=falling, pole_rad_without_gravity=still)


def camera():
    from isaacsim.sensors.camera import Camera

    world = new_world(1 / 60, -9.81)
    world.scene.add_default_ground_plane()
    sensor = Camera(prim_path='/World/camera', position=np.array([0, 0, 3.0]), resolution=(128, 128), frequency=20)
    world.reset()
    sensor.initialize()
    begin = time.time()
    for _ in range(30):
        world.step(render=True)
    rgba = sensor.get_rgba()
    empty = rgba is None or np.size(rgba) == 0
    say(step='camera', shape=None if rgba is None else list(np.shape(rgba)), mean=None if empty else float(np.mean(np.asarray(rgba)[..., :3])),
        seconds=round(time.time() - begin, 1))


attempt('freefall', freefall)
attempt('cartpole', cartpole)
attempt('camera', camera)
say(step='done')
app.close()
