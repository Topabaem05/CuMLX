"""A camera on a `cumlx.sim` task: each named link of the task is a box in a Mitsuba scene, and each frame takes its pose from the physics state.

The boxes are stand-ins with the sizes in the task. A robot's meshes are not rendered, so this shows the path and not a robot's looks.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..render.sensors import Camera, Sensors, check_frame, load_mitsuba, rotate


class TaskCamera:
    def __init__(self, backend: Any, camera: Camera, *, check_camera: Camera | None = None, variant: str = 'metal_ad_rgb', world: int = 0):
        """`camera` renders. `check_camera` is what `check` assumes, and differs from `camera` only in a wrong case."""
        self.backend, self.camera, self.world = backend, camera, world
        self.check_camera = check_camera or camera
        self.links = backend.task.links
        if not self.links:
            raise ValueError(f'the task {backend.task.name} names no links to show')
        mi = load_mitsuba(variant)
        t = mi.ScalarTransform4f()
        scene: dict[str, Any] = {
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 3},
            'sensor': camera.mitsuba_dict(mi),
            'light': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}},
            'floor': {'type': 'rectangle', 'to_world': t.scale([30.0, 30.0, 1.0]), 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}},
        }
        for name, link in self.links.items():
            scene[name] = {'type': 'cube', 'to_world': t.scale(list(link.half_extents)), 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.1, 0.1]}}}
        self.sensors = Sensors(scene, variant, camera=camera)
        self.floor_index = self.sensors.shape_ids.index('floor')
        self.indices = {name: self.sensors.shape_ids.index(name) for name in self.links}

    def poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The physics state: each named link's centre in the world, and its quaternion."""
        out = {}
        for name, link in self.links.items():
            position, quaternion = self.backend.link_pose(link.link, self.world)
            out[name] = (position + rotate(quaternion, np.asarray([link.offset], dtype=np.float64))[0], quaternion)
        return out

    def show(self, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        for name, (position, quaternion) in poses.items():
            self.sensors.set_pose(name, position, quaternion)

    def frame(self, *, spp: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth, segmentation = self.sensors.depth_segmentation(seed=seed)
        return self.sensors.rgb(spp=spp, seed=seed), depth, segmentation

    def check(self, depth: np.ndarray, segmentation: np.ndarray, poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        """Each link's pixels must lie where the camera formula puts its physics position, and floor depth must match the ray and plane formula."""
        rows = {}
        for name, (position, _) in poses.items():
            half = float(np.linalg.norm(self.links[name].half_extents)) / math.sqrt(3)  # check_frame's reach is half x sqrt(3): the box's corner.
            rows[name] = check_frame(self.check_camera, depth, segmentation, self.indices[name], self.floor_index, position, half)
        return {'ok': all(row['ok'] for row in rows.values()), 'links': {name: {'ok': row['ok'], 'pixel_error': row['pixel_error'], 'failures': row['failures']} for name, row in rows.items()}}
