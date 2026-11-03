"""Engines for `cumlx.sim`. Each holds `num_envs` worlds of one task. Genesis runs on Metal or the CPU, and MuJoCo on the CPU."""
from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from ..errors import CumlxError, missing_dependency
from .task import ResetState, State, Task, require_torch


class GenesisBackend:
    """`physics` is 'metal' or 'cpu'. `gravity` exists for fault tests. Genesis allows one init in a process."""

    engine = 'genesis'

    def __init__(self, task: Task, num_envs: int, *, physics: str = 'metal', seed: int = 0, gravity: float = -9.81):
        try:
            import genesis as gs
        except ModuleNotFoundError:
            raise missing_dependency('genesis', 'sim', purpose='GenesisBackend') from None
        if task.asset.kind == 'usd':
            try:
                import pxr  # noqa: F401
            except ModuleNotFoundError:
                raise missing_dependency('pxr', 'sim', purpose='A USD asset') from None
        if physics not in ('metal', 'cpu'):
            raise CumlxError('E_USAGE', f'physics must be metal or cpu, not {physics!r}')
        morph = {'mjcf': gs.morphs.MJCF, 'urdf': gs.morphs.URDF, 'usd': gs.morphs.USD}.get(task.asset.kind)
        if morph is None:
            raise CumlxError('E_SIM_UNSUPPORTED', f'Genesis cannot load an asset of kind {task.asset.kind!r}')
        torch = require_torch()
        if not getattr(gs, '_initialized', False):
            gs.init(backend=gs.metal if physics == 'metal' else gs.cpu, seed=seed, logging_level='warning')
        self.task, self.num_envs, self.device = task, num_envs, gs.device
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=task.dt, substeps=task.substeps, gravity=(0.0, 0.0, gravity)),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, max_collision_pairs=20, tolerance=1e-5), show_viewer=False)
        options: dict[str, Any] = {'file': str(task.asset.path())}
        self.floating = task.asset.base_pos is not None
        if self.floating:
            options.update(pos=task.asset.base_pos, quat=task.asset.base_quat)
            self.scene.add_entity(gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))
        if task.actuation.armature is not None:
            options['default_armature'] = task.actuation.armature
        self.entity = self.scene.add_entity(morph(**options))
        self.scene.build(n_envs=num_envs)
        self.dofs = torch.tensor([self.entity.get_joint(n).dofs_idx_local[0] for n in task.joints], dtype=torch.long, device=self.device)
        self.qs = torch.tensor([self.entity.get_joint(n).qs_idx_local[0] for n in task.joints], dtype=torch.long, device=self.device)
        self.action_dofs = torch.tensor([self.entity.get_joint(n).dofs_idx_local[0] for n in task.action_joints], dtype=torch.long, device=self.device)
        if task.actuation.kind == 'position':
            self.entity.set_dofs_kp([task.actuation.kp] * task.num_actions, self.action_dofs)
            self.entity.set_dofs_kv([task.actuation.kd] * task.num_actions, self.action_dofs)
        self.base_q = torch.zeros(self.entity.n_qs, device=self.device)
        if self.floating:
            self.base_q[:3] = self.base_q.new_tensor(task.asset.base_pos)
            self.base_q[3:7] = self.base_q.new_tensor(task.asset.base_quat)

    def reset(self, envs: Any, values: ResetState) -> None:
        q = self.base_q.repeat(len(envs), 1)
        q[:, self.qs] = values.joint_pos.to(self.device).to(q.dtype)
        self.entity.set_qpos(q, envs_idx=envs, zero_velocity=True)
        if bool(values.joint_vel.ne(0).any()):
            self.entity.set_dofs_velocity(values.joint_vel.to(self.device).to(q.dtype), self.dofs, envs_idx=envs)

    def apply(self, actions: Any) -> None:
        act = self.task.actuation
        if act.kind == 'force':
            self.entity.control_dofs_force(actions * act.scale, self.action_dofs)
        else:
            self.entity.control_dofs_position(actions * act.scale + actions.new_tensor(act.default), self.action_dofs)

    def step(self) -> None:
        for _ in range(self.task.decimation):
            self.scene.step()

    def state(self) -> State:
        out = State(self.entity.get_dofs_position(self.dofs), self.entity.get_dofs_velocity(self.dofs))
        if self.floating:
            out.base_pos, out.base_quat = self.entity.get_pos(), self.entity.get_quat()
            out.base_lin_vel, out.base_ang_vel = self.entity.get_vel(), self.entity.get_ang()
        return out

    def failed_worlds(self) -> Any:
        return self.scene.rigid_solver.get_error_envs_mask()

    def link_pose(self, link: str, world: int = 0) -> tuple[np.ndarray, np.ndarray]:
        body = self.entity.get_link(link)
        return (body.get_pos().detach().cpu().numpy()[world].astype(np.float64),
                body.get_quat().detach().cpu().numpy()[world].astype(np.float64))


def urdf_for_mujoco(path: str) -> str:
    """A URDF with its visuals taken out, because MuJoCo does not read their .dae files. The collision shapes stay."""
    root = ET.parse(path).getroot()
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
    settings = ET.SubElement(root, 'mujoco')
    ET.SubElement(settings, 'compiler', {'discardvisual': 'true', 'fusestatic': 'false', 'balanceinertia': 'true'})
    return ET.tostring(root, encoding='unicode')


class MujocoBackend:
    """A loop over one MjData per world, in float64 on the CPU. A PD drive is an explicit force at each physics step.

    `gravity` and `substep_scale` exist for fault tests and for a control of the time step.
    """

    engine = 'mujoco'

    def __init__(self, task: Task, num_envs: int, *, gravity: float = -9.81, substep_scale: int = 1):
        try:
            import mujoco
        except ModuleNotFoundError:
            raise missing_dependency('mujoco', 'sim', purpose='MujocoBackend') from None
        self.mujoco, self.torch = mujoco, require_torch()
        self.task, self.num_envs, self.device = task, num_envs, 'cpu'
        kind, path = task.asset.kind, str(task.asset.path())
        if kind == 'mjcf':
            spec = mujoco.MjSpec.from_file(path)
        elif kind == 'urdf':
            spec = mujoco.MjSpec.from_string(urdf_for_mujoco(path))
        else:
            raise CumlxError('E_SIM_UNSUPPORTED', f'MujocoBackend cannot load an asset of kind {kind!r}: use mjcf or urdf')
        self.floating = task.asset.base_pos is not None
        if self.floating:
            next(body for body in spec.bodies if body.name != 'world').add_freejoint()
            floor = spec.worldbody.add_geom()
            floor.type = mujoco.mjtGeom.mjGEOM_PLANE
            floor.size = [0.0, 0.0, 1.0]
        if task.actuation.armature is not None:
            for joint in spec.joints:
                if joint.name in task.joints:
                    joint.armature = task.actuation.armature
        self.model = spec.compile()
        self.model.opt.timestep = task.dt / task.substeps / substep_scale
        self.model.opt.gravity[:] = (0.0, 0.0, gravity)
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.steps_per_action = task.decimation * task.substeps * substep_scale
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.joint_q = np.array([self.model.joint(n).qposadr[0] for n in task.joints])
        self.joint_v = np.array([self.model.joint(n).dofadr[0] for n in task.joints])
        self.action_q = np.array([self.model.joint(n).qposadr[0] for n in task.action_joints])
        self.action_v = np.array([self.model.joint(n).dofadr[0] for n in task.action_joints])
        self.targets = np.zeros((num_envs, task.num_actions))
        self.default = np.array(task.actuation.default) if task.actuation.kind == 'position' else None

    def reset(self, envs: Any, values: ResetState) -> None:
        q, v = values.joint_pos.numpy().astype(np.float64), values.joint_vel.numpy().astype(np.float64)
        for row, env in enumerate(envs.tolist()):
            data = self.data[env]
            self.mujoco.mj_resetData(self.model, data)
            if self.floating:
                data.qpos[:3] = self.task.asset.base_pos
                data.qpos[3:7] = self.task.asset.base_quat
            data.qpos[self.joint_q] = q[row]
            data.qvel[self.joint_v] = v[row]
            self.mujoco.mj_forward(self.model, data)

    def apply(self, actions: Any) -> None:
        self.targets = actions.detach().cpu().numpy().astype(np.float64) * self.task.actuation.scale
        if self.default is not None:
            self.targets = self.targets + self.default

    def step(self) -> None:
        act = self.task.actuation
        for env, data in enumerate(self.data):
            for _ in range(self.steps_per_action):
                if act.kind == 'force':
                    data.qfrc_applied[self.action_v] = self.targets[env]
                else:
                    data.qfrc_applied[self.action_v] = act.kp * (self.targets[env] - data.qpos[self.action_q]) - act.kd * data.qvel[self.action_v]
                self.mujoco.mj_step(self.model, data)

    def state(self) -> State:
        torch = self.torch
        q = np.stack([d.qpos[self.joint_q] for d in self.data])
        v = np.stack([d.qvel[self.joint_v] for d in self.data])
        out = State(torch.from_numpy(q).float(), torch.from_numpy(v).float())
        if self.floating:
            pos = np.stack([d.qpos[:3] for d in self.data])
            quat = np.stack([d.qpos[3:7] for d in self.data])
            lin = np.stack([d.qvel[:3] for d in self.data])
            local = np.stack([d.qvel[3:6] for d in self.data])  # MuJoCo keeps a free joint's angular velocity in the body frame.
            w, xyz = quat[:, :1], quat[:, 1:]
            t = 2.0 * np.cross(xyz, local)
            ang = local + w * t + np.cross(xyz, t)
            out.base_pos, out.base_quat = torch.from_numpy(pos).float(), torch.from_numpy(quat).float()
            out.base_lin_vel, out.base_ang_vel = torch.from_numpy(lin).float(), torch.from_numpy(ang).float()
        return out

    def failed_worlds(self) -> Any:
        return self.torch.tensor([not np.isfinite(d.qpos).all() for d in self.data])

    def link_pose(self, link: str, world: int = 0) -> tuple[np.ndarray, np.ndarray]:
        body = self.data[world].body(link)
        return np.array(body.xpos, dtype=np.float64), np.array(body.xquat, dtype=np.float64)
