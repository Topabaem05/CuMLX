"""Read a USD scene and turn it into a Mitsuba scene dictionary. `pxr` and Mitsuba are optional, and they are imported on use.

This covers a small part of USD: polygon meshes, one perspective camera, RectLight and DistantLight, and a diffuse material from
UsdPreviewSurface or `primvars:displayColor`. Anything else in the scene is refused by name, with E_RENDER_USD_UNSUPPORTED, because a
prim that is left out without a word gives a wrong picture with no error. What is kept but simplified is listed in `UsdScene.notes`.

Light units follow the UsdLux schema. The luminance is L = intensity * 2^exposure * color. `normalize` divides it by a size factor:
the world-space area for a RectLight, and the factor in the schema's `inputs:normalize` text for a DistantLight. A DistantLight's
illuminance is E = L * sizeFactor, so with `normalize` on, E is the intensity in lux. It becomes a Mitsuba `directional` emitter with
that irradiance, which is perfectly parallel: a soft shadow from a nonzero angle is not rendered.

USD matrices multiply row vectors, and these functions return them for column vectors. Meshes are written to binary PLY files, so
`to_mitsuba_dict` needs a folder that lasts until `mi.load_dict`. `load_usd` handles that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..errors import CumlxError, missing_dependency

DEFAULT_REFLECTANCE = (0.18, 0.18, 0.18)  # The default of UsdPreviewSurface's diffuseColor.
PREVIEW_SURFACE_DEFAULTS = {'metallic': 0.0, 'opacity': 1.0, 'clearcoat': 0.0, 'displacement': 0.0, 'emissiveColor': (0.0, 0.0, 0.0)}
RIGID_TOLERANCE = 1e-5  # How far a camera's rotation may be from orthonormal, in float32 attributes.


@dataclass
class UsdMesh:
    path: str
    points: np.ndarray  # N x 3, in world space
    triangles: np.ndarray  # M x 3, wound so that the front faces outward in world space
    reflectance: tuple[float, float, float]
    double_sided: bool = False


@dataclass
class UsdCamera:
    path: str
    to_world: np.ndarray  # 4 x 4 for column vectors, in Mitsuba's convention: +Z looks forward and +X is left
    fov_x_degrees: float
    aspect: float  # width / height
    near_clip: float
    far_clip: float


@dataclass
class UsdRectLight:
    path: str
    to_world: np.ndarray  # 4 x 4 that maps Mitsuba's rectangle [-1, 1]^2 with normal +Z onto the light, whose normal is the emitting side
    radiance: tuple[float, float, float]


@dataclass
class UsdDistantLight:
    path: str
    direction: tuple[float, float, float]  # The way the light travels
    irradiance: tuple[float, float, float]


@dataclass
class UsdScene:
    source: str
    meshes: list[UsdMesh]
    camera: UsdCamera
    lights: list[UsdRectLight | UsdDistantLight]
    up_axis: str
    meters_per_unit: float
    notes: list[str] = field(default_factory=list)


def _load_pxr() -> SimpleNamespace:
    try:
        from pxr import Tf, Usd, UsdGeom, UsdLux, UsdShade
    except ModuleNotFoundError:
        raise missing_dependency('pxr', 'usd', purpose='Reading a USD scene') from None
    return SimpleNamespace(Tf=Tf, Usd=Usd, UsdGeom=UsdGeom, UsdLux=UsdLux, UsdShade=UsdShade)


def _open_stage(pxr: SimpleNamespace, source: Any) -> tuple[Any, str]:
    if isinstance(source, pxr.Usd.Stage):
        return source, source.GetRootLayer().identifier
    path = Path(source)
    if not path.exists():
        raise CumlxError('E_INPUT_NOT_FOUND', f'no USD file at {path}')
    try:
        return pxr.Usd.Stage.Open(str(path)), str(path)
    except pxr.Tf.ErrorException as exc:
        raise CumlxError('E_INPUT_INVALID', f'cannot open {path} as USD: {exc}') from None


def _column_matrix(matrix: Any) -> np.ndarray:
    """A Gf.Matrix4d, which multiplies row vectors, as a 4 x 4 array for column vectors."""
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)], dtype=np.float64).T


def _value(attribute: Any, time: Any, default: Any = None) -> Any:
    value = attribute.Get(time) if attribute else None
    return default if value is None else value


def _triple(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_triangles(counts: np.ndarray, indices: np.ndarray, point_count: int) -> np.ndarray:
    """Fan triangulation. It is right for convex polygons, and it does not check that a polygon is convex."""
    if counts.size == 0 or (counts < 3).any():
        raise ValueError('a face has fewer than 3 vertices')
    if counts.sum() != indices.size:
        raise ValueError(f'faceVertexCounts sums to {int(counts.sum())}, and there are {indices.size} faceVertexIndices')
    if indices.min() < 0 or indices.max() >= point_count:
        raise ValueError(f'an index is outside the {point_count} points')
    triangles: list[tuple[int, int, int]] = []
    start = 0
    for count in counts:
        face = indices[start:start + count]
        triangles.extend((face[0], face[i], face[i + 1]) for i in range(1, int(count) - 1))
        start += int(count)
    return np.array(triangles, dtype=np.int32)


def transform_mesh(points: np.ndarray, triangles: np.ndarray, to_world: np.ndarray, left_handed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Points in world space, and triangles wound so that a mirror in the transform or a leftHanded mesh keeps its front side."""
    homogeneous = np.column_stack([points.astype(np.float64), np.ones(len(points))])
    world = (to_world @ homogeneous.T).T[:, :3]
    if left_handed != (np.linalg.det(to_world[:3, :3]) < 0):
        triangles = triangles[:, [0, 2, 1]]
    return world, triangles


def distant_size_factor(angle_degrees: float) -> float:
    """The schema's sizeFactor of a DistantLight, from the text of `inputs:normalize`."""
    theta = min(max(math.radians(angle_degrees) / 2, 0.0), math.pi)
    if theta == 0:
        return 1.0
    sin2 = math.sin(theta) ** 2
    return sin2 * math.pi if theta <= math.pi / 2 else (2 - sin2) * math.pi


class _Reader:
    def __init__(self, pxr: SimpleNamespace, time: Any) -> None:
        self.pxr = pxr
        self.time = time
        self.refused: list[dict[str, str]] = []
        self.notes: set[str] = set()
        self.xforms = pxr.UsdGeom.XformCache(time)

    def refuse(self, prim: Any, reason: str) -> None:
        self.refused.append({'prim': str(prim.GetPath()), 'reason': reason})

    def world(self, prim: Any) -> np.ndarray:
        return _column_matrix(self.xforms.GetLocalToWorldTransform(prim))

    def _material(self, prim: Any) -> tuple[float, float, float] | None:
        pxr = self.pxr
        material = pxr.UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if material:
            surface = material.ComputeSurfaceSource()
            shader = surface[0] if isinstance(surface, tuple) else surface
            if not shader or shader.GetShaderId() != 'UsdPreviewSurface':
                self.refuse(prim, f'its material {material.GetPath()} has no UsdPreviewSurface shader')
                return None
            for name, default in PREVIEW_SURFACE_DEFAULTS.items():
                shader_input = shader.GetInput(name)
                if shader_input and shader_input.HasConnectedSource():
                    self.refuse(prim, f'its material connects {name} to a texture or a node')
                    return None
                value = shader_input.Get(self.time) if shader_input else None
                if value is not None and not np.allclose(np.asarray(value, dtype=np.float64), default):
                    self.refuse(prim, f'its material sets {name}, and only a diffuse color is converted')
                    return None
            color = shader.GetInput('diffuseColor')
            if color and color.HasConnectedSource():
                self.refuse(prim, 'its material connects diffuseColor to a texture or a node')
                return None
            self.notes.add("UsdPreviewSurface becomes a diffuse BSDF: its specular lobe is not rendered")
            return _triple(_value(color, self.time, DEFAULT_REFLECTANCE)) if color else DEFAULT_REFLECTANCE
        primvar = pxr.UsdGeom.PrimvarsAPI(prim).GetPrimvar('displayColor')
        if primvar and primvar.HasAuthoredValue():
            values = primvar.Get(self.time)
            if values is None or len(values) != 1:
                self.refuse(prim, 'its displayColor has more than one value, and only one color is converted')
                return None
            return _triple(values[0])
        self.notes.add(f'A mesh with no material and no displayColor gets the diffuse color {DEFAULT_REFLECTANCE}')
        return DEFAULT_REFLECTANCE

    def mesh(self, prim: Any) -> UsdMesh | None:
        pxr = self.pxr
        mesh = pxr.UsdGeom.Mesh(prim)
        if pxr.UsdGeom.Subset.GetAllGeomSubsets(pxr.UsdGeom.Imageable(prim)):
            self.refuse(prim, 'it has GeomSubsets, and one material per mesh is converted')
            return None
        holes = _value(mesh.GetHoleIndicesAttr(), self.time, [])
        if len(holes):
            self.refuse(prim, 'it has holeIndices')
            return None
        points = np.asarray(_value(mesh.GetPointsAttr(), self.time, []), dtype=np.float64).reshape(-1, 3)
        counts = np.asarray(_value(mesh.GetFaceVertexCountsAttr(), self.time, []), dtype=np.int64)
        indices = np.asarray(_value(mesh.GetFaceVertexIndicesAttr(), self.time, []), dtype=np.int64)
        if not len(points) or not np.isfinite(points).all():
            self.refuse(prim, 'it has no points, or a point that is not finite')
            return None
        try:
            triangles = _as_triangles(counts, indices, len(points))
        except ValueError as exc:
            self.refuse(prim, f'its faces cannot be read: {exc}')
            return None
        to_world = self.world(prim)
        if abs(np.linalg.det(to_world[:3, :3])) < 1e-12:
            self.refuse(prim, 'its transform has no inverse')
            return None
        if _value(mesh.GetSubdivisionSchemeAttr(), self.time) != 'none':
            self.notes.add('A mesh with a subdivision scheme is rendered as its control polygons, not as a subdivision surface')
        reflectance = self._material(prim)
        if reflectance is None:
            return None
        if not all(0.0 <= c <= 1.0 for c in reflectance):
            self.refuse(prim, f'its color {reflectance} is outside [0, 1]')
            return None
        left_handed = _value(mesh.GetOrientationAttr(), self.time) == 'leftHanded'
        world, triangles = transform_mesh(points, triangles, to_world, left_handed)
        double_sided = bool(_value(mesh.GetDoubleSidedAttr(), self.time, False))
        return UsdMesh(str(prim.GetPath()), world, triangles, reflectance, double_sided)

    def camera(self, prim: Any) -> UsdCamera | None:
        camera = self.pxr.UsdGeom.Camera(prim)
        if _value(camera.GetProjectionAttr(), self.time) != 'perspective':
            self.refuse(prim, 'only a perspective camera is converted')
            return None
        if _value(camera.GetHorizontalApertureOffsetAttr(), self.time, 0.0) or _value(camera.GetVerticalApertureOffsetAttr(), self.time, 0.0):
            self.refuse(prim, 'it has an aperture offset')
            return None
        horizontal = float(_value(camera.GetHorizontalApertureAttr(), self.time))
        vertical = float(_value(camera.GetVerticalApertureAttr(), self.time))
        focal = float(_value(camera.GetFocalLengthAttr(), self.time))
        if horizontal <= 0 or vertical <= 0 or focal <= 0:
            self.refuse(prim, 'its aperture or focal length is not positive')
            return None
        to_world = self.world(prim)
        rotation = to_world[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=RIGID_TOLERANCE) or np.linalg.det(rotation) < 0:
            self.refuse(prim, 'its transform has a scale, a shear or a mirror')
            return None
        if float(_value(camera.GetFStopAttr(), self.time, 0.0)) > 0:
            self.notes.add('The camera has an fStop, and depth of field is not rendered')
        near, far = _value(camera.GetClippingRangeAttr(), self.time)
        # USD: looks along -Z, with +X to the right. Mitsuba: looks along +Z, with +X to the left. The same camera, turned half way round Y.
        return UsdCamera(str(prim.GetPath()), to_world @ np.diag([-1.0, 1.0, -1.0, 1.0]), math.degrees(2 * math.atan(horizontal / (2 * focal))),
                         horizontal / vertical, float(near), float(far))

    def _light_radiance(self, prim: Any, light: Any) -> tuple[float, float, float] | None:
        for getter, default, name in ((light.GetDiffuseAttr, 1.0, 'diffuse'), (light.GetSpecularAttr, 1.0, 'specular')):
            value = _value(getter(), self.time, default)
            if value != default:
                self.refuse(prim, f'it sets the {name} multiplier')
                return None
        if _value(light.GetEnableColorTemperatureAttr(), self.time, False):
            self.refuse(prim, 'it uses a color temperature')
            return None
        for api in (self.pxr.UsdLux.ShapingAPI, self.pxr.UsdLux.ShadowAPI):
            if prim.HasAPI(api):
                self.refuse(prim, f'it applies {api.__name__}')
                return None
        scale = float(_value(light.GetIntensityAttr(), self.time)) * 2.0 ** float(_value(light.GetExposureAttr(), self.time, 0.0))
        color = _value(light.GetColorAttr(), self.time, (1.0, 1.0, 1.0))
        return (scale * float(color[0]), scale * float(color[1]), scale * float(color[2]))

    def rect_light(self, prim: Any) -> UsdRectLight | None:
        light = self.pxr.UsdLux.RectLight(prim)
        file_attribute = light.GetTextureFileAttr()
        if file_attribute and file_attribute.HasAuthoredValue():
            self.refuse(prim, 'it has a texture file')
            return None
        radiance = self._light_radiance(prim, light)
        if radiance is None:
            return None
        width = float(_value(light.GetWidthAttr(), self.time))
        height = float(_value(light.GetHeightAttr(), self.time))
        # The rectangle's normal +Z of Mitsuba goes to the light's -Z, which is the side that emits. x changes sign too, to keep a rotation.
        to_world = self.world(prim) @ np.diag([-width / 2, height / 2, -1.0, 1.0])
        if _value(light.GetNormalizeAttr(), self.time, False):
            area = 4 * np.linalg.norm(to_world[:3, 0]) * np.linalg.norm(to_world[:3, 1])
            if area <= 0:
                self.refuse(prim, 'it has no area')
                return None
            radiance = (radiance[0] / area, radiance[1] / area, radiance[2] / area)
        return UsdRectLight(str(prim.GetPath()), to_world, radiance)

    def distant_light(self, prim: Any) -> UsdDistantLight | None:
        light = self.pxr.UsdLux.DistantLight(prim)
        luminance = self._light_radiance(prim, light)
        if luminance is None:
            return None
        angle = float(_value(light.GetAngleAttr(), self.time, 0.0))
        if angle > 0:
            self.notes.add('A DistantLight with an angle becomes a parallel light: its shadows stay hard')
        factor = 1.0 if _value(light.GetNormalizeAttr(), self.time, False) else distant_size_factor(angle)
        direction = self.world(prim)[:3, :3] @ np.array([0.0, 0.0, -1.0])
        norm = np.linalg.norm(direction)
        if norm == 0:
            self.refuse(prim, 'its transform has no direction')
            return None
        return UsdDistantLight(str(prim.GetPath()), _triple(direction / norm), (luminance[0] * factor, luminance[1] * factor, luminance[2] * factor))


def read_usd(source: Any, *, camera: str | None = None, time: float | None = None) -> UsdScene:
    """Read a USD file or an open stage. `camera` is a prim path, and it is required only when the scene has more than one camera."""
    pxr = _load_pxr()
    stage, name = _open_stage(pxr, source)
    timecode = pxr.Usd.TimeCode.Default() if time is None else pxr.Usd.TimeCode(time)
    reader = _Reader(pxr, timecode)
    meshes: list[UsdMesh] = []
    cameras: list[UsdCamera] = []
    lights: list[UsdRectLight | UsdDistantLight] = []
    for prim in pxr.Usd.PrimRange.Stage(stage, pxr.Usd.TraverseInstanceProxies()):
        imageable = pxr.UsdGeom.Imageable(prim)
        if imageable:
            if imageable.ComputeVisibility(timecode) == pxr.UsdGeom.Tokens.invisible:
                continue
            if imageable.ComputePurpose() in (pxr.UsdGeom.Tokens.guide, pxr.UsdGeom.Tokens.proxy):
                continue
        if prim.IsA(pxr.UsdGeom.Mesh):
            meshes.append(reader.mesh(prim))  # type: ignore[arg-type]
        elif prim.IsA(pxr.UsdGeom.Camera):
            cameras.append(reader.camera(prim))  # type: ignore[arg-type]
        elif prim.IsA(pxr.UsdLux.RectLight):
            lights.append(reader.rect_light(prim))  # type: ignore[arg-type]
        elif prim.IsA(pxr.UsdLux.DistantLight):
            lights.append(reader.distant_light(prim))  # type: ignore[arg-type]
        elif prim.HasAPI(pxr.UsdLux.LightAPI):
            reader.refuse(prim, f'{prim.GetTypeName()} is not a converted light type: RectLight and DistantLight are')
        elif prim.IsA(pxr.UsdGeom.Gprim) or prim.IsA(pxr.UsdGeom.PointInstancer):
            reader.refuse(prim, f'{prim.GetTypeName()} is not converted: only Mesh is')
    converted_meshes = [m for m in meshes if m is not None]
    converted_lights = [light for light in lights if light is not None]
    converted_cameras = [c for c in cameras if c is not None]
    if camera is not None:
        converted_cameras = [c for c in converted_cameras if c.path == camera]
    if not reader.refused:
        problems = [('no mesh', not converted_meshes), ('no light', not converted_lights), ('no camera', not converted_cameras),
                    ('more than one camera: name one', len(converted_cameras) > 1)]
        reader.refused.extend({'prim': '/', 'reason': f'the scene has {text}'} for text, bad in problems if bad)
    if reader.refused:
        first = reader.refused[0]
        raise CumlxError('E_RENDER_USD_UNSUPPORTED', f"{name}: {len(reader.refused)} problem(s), first {first['prim']}: {first['reason']}",
                         hint='Change the scene, or leave these prims out of it. Nothing was converted, because a missing prim gives a wrong picture with no error.',
                         details={'refused': reader.refused})
    up_axis = str(pxr.UsdGeom.GetStageUpAxis(stage))
    return UsdScene(name, converted_meshes, converted_cameras[0], converted_lights, up_axis, float(pxr.UsdGeom.GetStageMetersPerUnit(stage)),
                    sorted(reader.notes))


def write_ply(path: Path, points: np.ndarray, triangles: np.ndarray) -> None:
    """A binary little-endian PLY file with positions and triangles only, so that Mitsuba uses the normal of each face."""
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
              f'element face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n')
    faces = np.zeros(len(triangles), dtype=[('count', 'u1'), ('indices', '<i4', 3)])
    faces['count'] = 3
    faces['indices'] = triangles
    with open(path, 'wb') as handle:
        handle.write(header.encode('ascii'))
        handle.write(points.astype('<f4').tobytes())
        handle.write(faces.tobytes())


def _rgb(value: tuple[float, float, float]) -> dict[str, Any]:
    return {'type': 'rgb', 'value': list(value)}


def to_mitsuba_dict(mi: Any, scene: UsdScene, folder: Path, *, width: int = 256, spp: int = 64, max_depth: int = 8) -> dict[str, Any]:
    """A dictionary for `mi.load_dict`. The film's height comes from the camera's aperture. The PLY files in `folder` must last until then."""
    folder = Path(folder)
    height = round(width / scene.camera.aspect)
    out: dict[str, Any] = {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': max_depth},
        'sensor': {
            'type': 'perspective', 'fov': scene.camera.fov_x_degrees, 'fov_axis': 'x',
            'near_clip': scene.camera.near_clip, 'far_clip': scene.camera.far_clip,
            'to_world': mi.ScalarTransform4f(scene.camera.to_world.tolist()),
            'sampler': {'type': 'independent', 'sample_count': spp},
            'film': {'type': 'hdrfilm', 'width': width, 'height': height, 'rfilter': {'type': 'gaussian'},
                     'pixel_format': 'rgb', 'component_format': 'float32'},
        },
    }
    for index, mesh in enumerate(scene.meshes):
        file = folder / f'mesh_{index}.ply'
        write_ply(file, mesh.points, mesh.triangles)
        bsdf: dict[str, Any] = {'type': 'diffuse', 'reflectance': _rgb(mesh.reflectance)}
        if mesh.double_sided:
            bsdf = {'type': 'twosided', 'material': bsdf}
        out[f'mesh_{index}'] = {'type': 'ply', 'filename': str(file), 'face_normals': True, 'bsdf': bsdf}
    for index, light in enumerate(scene.lights):
        if isinstance(light, UsdRectLight):
            out[f'light_{index}'] = {'type': 'rectangle', 'to_world': mi.ScalarTransform4f(light.to_world.tolist()),
                                     'bsdf': {'type': 'diffuse', 'reflectance': _rgb((0.0, 0.0, 0.0))},
                                     'emitter': {'type': 'area', 'radiance': _rgb(light.radiance)}}
        else:
            out[f'light_{index}'] = {'type': 'directional', 'direction': list(light.direction), 'irradiance': _rgb(light.irradiance)}
    return out


def load_usd(mi: Any, source: Any, *, camera: str | None = None, time: float | None = None, width: int = 256, spp: int = 64,
             max_depth: int = 8) -> tuple[Any, UsdScene]:
    """The Mitsuba scene of a USD file or stage, with what was read. Mitsuba's variant must be set already."""
    scene = read_usd(source, camera=camera, time=time)
    with tempfile.TemporaryDirectory(prefix='cumlx-usd-') as folder:
        return mi.load_dict(to_mitsuba_dict(mi, scene, Path(folder), width=width, spp=spp, max_depth=max_depth)), scene
