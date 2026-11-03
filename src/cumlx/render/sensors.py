"""Camera sensors on Mitsuba: RGB, depth and segmentation. Mitsuba is an optional dependency, imported on use.

The camera geometry and the checks need only NumPy, so they can be tested without a renderer.

Depth is the distance along the ray from the camera, in scene units, and not the z of the camera. A ray that hits nothing
has depth 0. Segmentation is the index of the shape in `Sensors.shape_ids`, or -1 for a ray that hits nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import platform
import re
import subprocess
from typing import Any

import numpy as np

from ..errors import CumlxError, missing_dependency

DEFAULT_VARIANT = 'metal_ad_rgb'
# Limits of check_frame. They were committed before the first physics and camera run.
PIXEL_LIMIT = 3.0  # Middle of the shape's pixels against the projected middle of the body, in pixels of a 256 x 256 image.
FLOOR_MEAN_REL_LIMIT = 1e-3  # Mean signed error of floor depth against the ray and plane formula.
FLOOR_MEDIAN_REL_LIMIT = 5e-3  # Median absolute error. One sample per pixel lands anywhere in the pixel, so one pixel can differ by about 1%.


def load_mitsuba(variant: str = DEFAULT_VARIANT) -> Any:
    """Return the mitsuba module with `variant` set. A variant that this machine lacks is an environment error."""
    try:
        import mitsuba as mi
    except ModuleNotFoundError:
        raise missing_dependency('mitsuba', 'render', purpose='Rendering') from None
    if variant not in mi.variants():
        raise CumlxError('E_DEP_MISSING', f'Mitsuba {mi.__version__} has no variant {variant!r} on this machine: {mi.variants()}',
                         hint='metal_ad_rgb needs Apple Silicon, and cuda_ad_rgb needs an NVIDIA GPU with OptiX in its driver')
    mi.set_variant(variant)
    return mi


@dataclass(frozen=True)
class Camera:
    origin: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_x_degrees: float = 45.0
    width: int = 256
    height: int = 256

    def _axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.subtract(self.target, self.origin, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        return forward, right, np.cross(right, forward)

    @property
    def _focal(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.fov_x_degrees) / 2)

    def project(self, point: Any) -> tuple[float, float]:
        """Return the continuous pixel coordinates (x right, y down) of a world point in front of the camera."""
        forward, right, up = self._axes()
        relative = np.subtract(point, self.origin, dtype=np.float64)
        along = float(relative @ forward)
        if along <= 0:
            raise ValueError('the point is behind the camera')
        return (self.width / 2 + self._focal * float(relative @ right) / along,
                self.height / 2 - self._focal * float(relative @ up) / along)

    def ray_direction(self, x: float, y: float) -> np.ndarray:
        """Return the unit direction of the ray through continuous pixel coordinates."""
        forward, right, up = self._axes()
        direction = self._focal * forward + (x - self.width / 2) * right - (y - self.height / 2) * up
        return direction / np.linalg.norm(direction)

    def ray_grid(self) -> np.ndarray:
        """Return height x width x 3 unit directions through the pixel centres."""
        forward, right, up = self._axes()
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        offsets_x, offsets_y = (xs + 0.5 - self.width / 2)[..., None], (ys + 0.5 - self.height / 2)[..., None]
        directions = self._focal * forward + offsets_x * right - offsets_y * up
        return directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    def mitsuba_dict(self, mi: Any) -> dict[str, Any]:
        return {
            'type': 'perspective',
            'fov': self.fov_x_degrees,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f().look_at(origin=list(self.origin), target=list(self.target), up=list(self.up)),
            'film': {'type': 'hdrfilm', 'width': self.width, 'height': self.height, 'pixel_format': 'rgb',
                     'rfilter': {'type': 'box'}},
            'sampler': {'type': 'independent', 'sample_count': 16},
        }


def rotate(quaternion_wxyz: Any, vectors: np.ndarray) -> np.ndarray:
    """Rotate N x 3 vectors by a unit quaternion in w, x, y, z order."""
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    q = np.array([x, y, z])
    return vectors + 2 * w * np.cross(q, vectors) + 2 * np.cross(q, np.cross(q, vectors))


def box_scene(mi: Any, camera: Camera, half_size: float, position: Any = (0.0, 0.0, 0.5)) -> dict[str, Any]:
    """A red cube over a grey floor at z = 0, under a constant light. The cube is the shape named `box`."""
    transform = mi.ScalarTransform4f()
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 3},
        'sensor': camera.mitsuba_dict(mi),
        'light': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}},
        'floor': {'type': 'rectangle', 'to_world': transform.scale([30.0, 30.0, 1.0]),
                  'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}},
        'box': {'type': 'cube', 'to_world': transform.translate(list(position)).scale([half_size] * 3),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.1, 0.1]}}},
    }


class Sensors:
    """One Mitsuba scene with one camera. Every image is a NumPy array on the CPU side, so each call waits for the GPU."""

    def __init__(self, scene_dict: dict[str, Any], variant: str = DEFAULT_VARIANT):
        self.mi = load_mitsuba(variant)
        self.variant = variant
        self.scene = self.mi.load_dict(scene_dict)
        if not variant.split('_')[0] in ('metal', 'cuda', 'llvm'):
            raise CumlxError('E_USAGE', f'Sensors needs a JIT variant, because segmentation reads shape registry ids, and {variant} is not one')
        import drjit as dr
        self.shape_ids = [shape.id() for shape in self.scene.shapes()]
        # The shape_index AOV holds the registry id of the shape, which starts at 1 and need not follow the order of shapes().
        self._registry = {int(dr.reinterpret_array(self.mi.UInt32, self.mi.ShapePtr(shape))[0]): index for index, shape in enumerate(self.scene.shapes())}
        self._aov = self.mi.load_dict({'type': 'aov', 'aovs': 'dd:depth,ss:shape_index'})
        self._params: Any = None
        self._rest: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def rgb(self, spp: int = 16, seed: int = 0) -> np.ndarray:
        return np.array(self.mi.render(self.scene, spp=spp, seed=seed), dtype=np.float32)

    def depth_segmentation(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """One sample per pixel, because an average of two shape indices is not a shape."""
        image = np.array(self.mi.render(self.scene, integrator=self._aov, spp=1, seed=seed), dtype=np.float32)
        depth, index = image[..., -2], image[..., -1]
        segmentation = np.full(index.shape, -1, dtype=np.int32)
        for registry, position in self._registry.items():
            segmentation[(depth > 0) & (np.rint(index).astype(np.int64) == registry)] = position
        return depth, segmentation

    def set_pose(self, shape_id: str, position: Any, quaternion_wxyz: Any = (1.0, 0.0, 0.0, 0.0)) -> None:
        """Move a mesh to a pose relative to the place where the scene loaded it. This rebuilds the acceleration structure."""
        mi = self.mi
        if self._params is None:
            self._params = mi.traverse(self.scene)
        points_key, normals_key = f'{shape_id}.vertex_positions', f'{shape_id}.vertex_normals'
        if points_key not in self._params:
            raise CumlxError('E_INPUT_INVALID', f'{shape_id!r} is not a mesh in this scene: {sorted(self._params.keys())[:8]}')
        if shape_id not in self._rest:
            points = np.array(self._params[points_key]).reshape(-1, 3)
            normals = np.array(self._params[normals_key]).reshape(-1, 3)
            self._rest[shape_id] = (points - points.mean(axis=0), normals)
        points, normals = self._rest[shape_id]
        self._params[points_key] = mi.Float((rotate(quaternion_wxyz, points) + np.asarray(position)).astype(np.float32).ravel())
        self._params[normals_key] = mi.Float(rotate(quaternion_wxyz, normals).astype(np.float32).ravel())
        self._params.update()


def centroid(segmentation: np.ndarray, index: int) -> tuple[float, float] | None:
    """Return the continuous pixel coordinates of the middle of one shape's pixels, or None if it is not in the image."""
    ys, xs = np.nonzero(segmentation == index)
    if len(xs) == 0:
        return None
    return float(xs.mean()) + 0.5, float(ys.mean()) + 0.5


def check_frame(camera: Camera, depth: np.ndarray, segmentation: np.ndarray, body_index: int, floor_index: int, position: Any, half_size: float) -> dict[str, Any]:
    """Does one camera frame agree with the physics state? A floor at z = 0 and a cube with half side `half_size`."""
    failures = []
    middle = centroid(segmentation, body_index)
    projected = camera.project(position)
    pixel_error = None if middle is None else math.hypot(middle[0] - projected[0], middle[1] - projected[1])
    if middle is None:
        failures.append('the body is not in the image')
    elif pixel_error > PIXEL_LIMIT:  # type: ignore[operator]
        failures.append(f'the body is {pixel_error:.2f} px from its projected position, limit {PIXEL_LIMIT}')

    if middle is not None:
        distance = float(np.linalg.norm(np.subtract(position, camera.origin)))
        reach = half_size * math.sqrt(3)
        body_depth = depth[segmentation == body_index]
        if body_depth.min() < distance - reach - 1e-3 or body_depth.max() > distance + reach + 1e-3:
            failures.append(f'the body depth is {body_depth.min():.3f} to {body_depth.max():.3f}, and the body is {distance - reach:.3f} to {distance + reach:.3f} away')

    floor = segmentation == floor_index
    mean_rel = median_rel = None
    if not floor.any():
        failures.append('the floor is not in the image')
    else:
        directions = camera.ray_grid()[floor]
        expected = -camera.origin[2] / directions[:, 2]
        relative = depth[floor] / expected - 1
        mean_rel, median_rel = float(relative.mean()), float(np.median(np.abs(relative)))
        if abs(mean_rel) > FLOOR_MEAN_REL_LIMIT:
            failures.append(f'the mean floor depth error is {mean_rel:.2e}, limit {FLOOR_MEAN_REL_LIMIT}')
        if median_rel > FLOOR_MEDIAN_REL_LIMIT:
            failures.append(f'the median floor depth error is {median_rel:.2e}, limit {FLOOR_MEDIAN_REL_LIMIT}')
    return {'ok': not failures, 'failures': failures, 'centroid': middle, 'projected': projected, 'pixel_error': pixel_error,
            'floor_mean_rel': mean_rel, 'floor_median_rel': median_rel}


def system_used_gb() -> float | None:
    """Memory in use on a Mac: active, wired and compressed pages. None elsewhere."""
    if platform.system() != 'Darwin':
        return None
    out = subprocess.run(['vm_stat'], capture_output=True, text=True, check=False).stdout
    size = re.search(r'page size of (\d+)', out)
    counts = [re.search(rf'{label}:\s+(\d+)', out) for label in ('Pages active', 'Pages wired down', 'Pages occupied by compressor')]
    if size is None or None in counts:
        return None
    return sum(int(c.group(1)) for c in counts if c) * int(size.group(1)) / 1e9


def process_gpu_seconds(pid: int) -> float | None:
    """GPU time of one process on a Mac, from `accumulatedGPUTime` on its Metal clients. None elsewhere."""
    if platform.system() != 'Darwin':
        return None
    out = subprocess.run(['ioreg', '-l', '-w0', '-r', '-c', 'AGXDeviceUserClient'], capture_output=True, text=True, check=False).stdout
    total, seen, pending = 0, False, 0
    for line in out.splitlines():  # ioreg sorts keys, so an entry's AppUsage comes before its IOUserClientCreator.
        if 'AppUsage' in line:
            pending = sum(int(value) for value in re.findall(r'"accumulatedGPUTime"=(\d+)', line))
            continue
        creator = re.search(r'"IOUserClientCreator" = "pid (\d+),', line)
        if creator:
            if int(creator.group(1)) == pid:
                total += pending
                seen = True
            pending = 0
    return total / 1e9 if seen else None
