"""What a task is, apart from any engine. Cartpole and Go2 flat, in the style of Isaac Lab's direct tasks."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib.util
import math
from pathlib import Path
from typing import Any

from ..errors import CumlxError, missing_dependency

ASSETS = Path(__file__).resolve().parent / 'assets'


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        raise missing_dependency('torch', 'sim', purpose='cumlx.sim') from None
    return torch


@dataclass(frozen=True)
class Asset:
    """`file` is under cumlx/sim/assets, or under Genesis's own assets when `source` is 'genesis'."""
    kind: str  # 'mjcf', 'urdf' or 'usd'
    file: str
    source: str = 'cumlx'
    base_pos: tuple[float, float, float] | None = None  # None: the asset's base is fixed to the world.
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z

    def path(self) -> Path:
        if self.source == 'cumlx':
            root = ASSETS
        elif self.source == 'genesis':
            spec = importlib.util.find_spec('genesis')
            if spec is None or spec.origin is None:
                raise missing_dependency('genesis', 'sim', purpose=f'The asset {self.file}')
            root = Path(spec.origin).parent / 'assets'
        else:
            raise CumlxError('E_INPUT_INVALID', f'unknown asset source {self.source!r}')
        path = root / self.file
        if not path.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'the asset {path} does not exist')
        return path


@dataclass(frozen=True)
class Actuation:
    """'force': action x scale is the force on each joint. 'position': default + action x scale is the target of a PD drive."""
    kind: str
    scale: float
    kp: float = 0.0
    kd: float = 0.0
    default: tuple[float, ...] = ()
    armature: float | None = None  # Rotor inertia on each joint. Genesis adds its own to a URDF, so this states one value for both engines.


@dataclass
class State:
    """Torch tensors, num_envs first. The base entries are in the world frame, and quaternions are w, x, y, z."""
    joint_pos: Any
    joint_vel: Any
    base_pos: Any = None
    base_quat: Any = None
    base_lin_vel: Any = None
    base_ang_vel: Any = None


@dataclass
class ResetState:
    joint_pos: Any
    joint_vel: Any


@dataclass
class StepContext:
    state: State
    actions: Any
    last_actions: Any
    commands: Any
    task: Task
    terminated: Any = None


@dataclass(frozen=True)
class Task:
    name: str
    asset: Asset
    joints: tuple[str, ...]
    actuation: Actuation
    dt: float  # The physics step.
    decimation: int  # Physics steps per action.
    episode_seconds: float
    observation_size: int
    observe: Callable[[StepContext], Any]
    reward: Callable[[StepContext], Any]
    terminated: Callable[[StepContext], Any]
    reset: Callable[[int, Any, Task], ResetState]  # (number of worlds to reset, torch generator, task)
    action_clip: float = 1.0
    action_latency: bool = False  # The drive gets the previous action, as on a real robot.
    command_ranges: tuple[tuple[float, float], ...] = ()
    command_seconds: float = 0.0
    substeps: int = 1  # Solver substeps inside one physics step.
    links: dict[str, tuple[str, tuple[float, float, float]]] = field(default_factory=dict)  # A name: its link, and the offset of its centre.
    actuated: tuple[str, ...] = ()  # The joints that the action drives. Empty: all of `joints`.

    @property
    def action_joints(self) -> tuple[str, ...]:
        return self.actuated or self.joints

    @property
    def num_actions(self) -> int:
        return len(self.action_joints)

    @property
    def control_dt(self) -> float:
        return self.dt * self.decimation

    @property
    def max_episode_steps(self) -> int:
        return math.ceil(self.episode_seconds / self.control_dt)


def rotate_inverse(quat: Any, vectors: Any) -> Any:
    """Rotate world vectors into the frame of a body with quaternion w, x, y, z. Both have num_envs first."""
    torch = require_torch()
    w, q = quat[:, :1], quat[:, 1:]
    t = 2.0 * torch.cross(q, vectors, dim=1)
    return vectors - w * t + torch.cross(q, t, dim=1)


# -------------------------------------------------------------------- cartpole
CARTPOLE_POLE_RANGE = 0.25 * math.pi


def _cartpole_observe(c: StepContext) -> Any:
    torch = require_torch()
    p, v = c.state.joint_pos, c.state.joint_vel
    return torch.stack((p[:, 0], p[:, 1], v[:, 0], v[:, 1]), dim=1)


def _cartpole_reward(c: StepContext) -> Any:
    """Isaac-Cartpole-Direct's five terms and scales."""
    torch = require_torch()
    p, v = c.state.joint_pos, c.state.joint_vel
    return (1.0 * (~c.terminated).float() - 2.0 * c.terminated.float() - 1.0 * torch.square(p[:, 1])
            - 0.01 * torch.abs(v[:, 0]) - 0.005 * torch.abs(v[:, 1]))


def _cartpole_terminated(c: StepContext) -> Any:
    p = c.state.joint_pos
    return (p[:, 0].abs() > 3.0) | (p[:, 1].abs() > math.pi / 2)


def _cartpole_reset(count: int, generator: Any, task: Task) -> ResetState:
    torch = require_torch()
    pos = torch.zeros((count, 2))
    pos[:, 1] = (torch.rand(count, generator=generator) * 2 - 1) * CARTPOLE_POLE_RANGE
    return ResetState(pos, torch.zeros((count, 2)))


CARTPOLE = Task(
    name='cartpole', asset=Asset('mjcf', 'cartpole.xml'), joints=('slider', 'hinge'), actuation=Actuation('force', 20.0, armature=0.0),
    dt=1 / 120, decimation=2, episode_seconds=5.0, observation_size=4, observe=_cartpole_observe, reward=_cartpole_reward,
    terminated=_cartpole_terminated, reset=_cartpole_reset, actuated=('slider',), links={'cart': ('cart', (0.0, 0.0, 0.0)), 'pole': ('pole', (0.0, 0.0, 0.5))})


# -------------------------------------------------------------------- Go2 flat
GO2_JOINTS = ('FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
              'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint')
GO2_DEFAULT = (0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5)
GO2_TILT_LIMIT = math.sin(math.radians(10.0))  # Roll or pitch, read from gravity in the body frame.
GO2_HEIGHT = 0.3


def _gravity(state: State) -> Any:
    torch = require_torch()
    down = torch.zeros_like(state.base_pos)
    down[:, 2] = -1.0
    return rotate_inverse(state.base_quat, down)


def _go2_observe(c: StepContext) -> Any:
    torch = require_torch()
    s = c.state
    default = s.joint_pos.new_tensor(c.task.actuation.default)
    ang = rotate_inverse(s.base_quat, s.base_ang_vel)
    scale = c.commands.new_tensor((2.0, 2.0, 0.25))
    return torch.cat((ang * 0.25, _gravity(s), c.commands * scale, s.joint_pos - default, s.joint_vel * 0.05, c.actions), dim=1)


def _go2_reward(c: StepContext) -> Any:
    """Genesis's Go2 walking terms and scales, for one control step."""
    torch = require_torch()
    s = c.state
    default = s.joint_pos.new_tensor(c.task.actuation.default)
    lin, ang = rotate_inverse(s.base_quat, s.base_lin_vel), rotate_inverse(s.base_quat, s.base_ang_vel)
    tracking = torch.exp(-torch.sum(torch.square(c.commands[:, :2] - lin[:, :2]), dim=1) / 0.25)
    turning = torch.exp(-torch.square(c.commands[:, 2] - ang[:, 2]) / 0.25)
    return c.task.control_dt * (1.0 * tracking + 0.2 * turning - 1.0 * torch.square(lin[:, 2])
                                - 50.0 * torch.square(s.base_pos[:, 2] - GO2_HEIGHT)
                                - 0.005 * torch.sum(torch.square(c.last_actions - c.actions), dim=1)
                                - 0.1 * torch.sum(torch.abs(s.joint_pos - default), dim=1))


def _go2_terminated(c: StepContext) -> Any:
    gravity = _gravity(c.state)
    return (gravity[:, 0].abs() > GO2_TILT_LIMIT) | (gravity[:, 1].abs() > GO2_TILT_LIMIT)


def _go2_reset(count: int, generator: Any, task: Task) -> ResetState:
    torch = require_torch()
    return ResetState(torch.tensor(task.actuation.default, dtype=torch.float32).repeat(count, 1), torch.zeros((count, task.num_actions)))


GO2_FLAT = Task(
    name='go2_flat', asset=Asset('urdf', 'urdf/go2/urdf/go2.urdf', source='genesis', base_pos=(0.0, 0.0, 0.42)), joints=GO2_JOINTS,
    actuation=Actuation('position', 0.25, kp=20.0, kd=0.5, default=GO2_DEFAULT, armature=0.1), dt=0.02, decimation=1, substeps=2,
    episode_seconds=20.0, observation_size=45, observe=_go2_observe, reward=_go2_reward, terminated=_go2_terminated, reset=_go2_reset,
    action_clip=100.0, action_latency=True, command_ranges=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)), command_seconds=4.0,
    links={'base': ('base', (0.0, 0.0, 0.0))})

_TASKS = {task.name: task for task in (CARTPOLE, GO2_FLAT)}


def available_tasks() -> list[str]:
    return sorted(_TASKS)


def get_task(name: str) -> Task:
    if name not in _TASKS:
        raise CumlxError('E_INPUT_INVALID', f'unknown task {name!r}. Tasks: {available_tasks()}')
    return _TASKS[name]
