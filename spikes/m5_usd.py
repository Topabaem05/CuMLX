#!/usr/bin/env python3
"""M5: render a USD scene and the equal scene written by hand as a Mitsuba dictionary, and compare them with the M5 grade B rule.

    python spikes/m5_usd.py run --variant metal_ad_rgb --output FILE.json
    python spikes/m5_usd.py write --output-dir DIR      # the scenes as .usda files, with no renderer

Each stage is written here with USD's own calls, and each reference by hand with Mitsuba's own shapes, so that the converter has to
reach the reference. The limits and the faults are in validation/m5-usd-limits.json, and they were committed before the first render.
Exit 0 means each clean scene is `ok` and each fault is `diverged`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx.render.compare import compare_images  # noqa: E402
from cumlx.render.sensors import load_mitsuba, process_gpu_seconds  # noqa: E402
from cumlx.render.usd import load_usd  # noqa: E402
from cumlx.sysmem import over_limit  # noqa: E402

SPP = 4096  # The same as spikes/m5_render.py.
PASS_SPP = 64
REFERENCE_SEEDS = (1, 2, 3)
USD_SEEDS = (11, 12, 13)
FAULT_SEEDS = (21, 22, 23)
SCENES = ('cornell', 'boxes')
FAULTS = {'cornell': ('wrong_transform', 'missing_mesh', 'wrong_light_size'), 'boxes': ('wrong_transform', 'missing_mesh', 'wrong_light_direction')}

WHITE = (0.885809, 0.698859, 0.666422)
GREEN = (0.105421, 0.37798, 0.076425)
RED = (0.570068, 0.0430135, 0.0443706)
QUAD_POINTS = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
CUBE_POINTS = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
CUBE_FACES = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]  # Quads, wound to face outward.


def _mesh(stage: Any, path: str, points: list[tuple[int, int, int]], counts: list[int], indices: list[int]) -> Any:
    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr('none')
    return mesh


def quad(stage: Any, path: str) -> Any:
    return _mesh(stage, path, QUAD_POINTS, [4], [0, 1, 2, 3])


def cube(stage: Any, path: str) -> Any:
    return _mesh(stage, path, CUBE_POINTS, [4] * 6, CUBE_FACES)


def _display_color(mesh: Any, color: tuple[float, float, float]) -> None:
    from pxr import Gf, Vt

    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))


def material(stage: Any, path: str, color: tuple[float, float, float]) -> Any:
    from pxr import Gf, Sdf, UsdShade

    made = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f'{path}/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    made.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return made


def bind(mesh: Any, made: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(made)


def camera(stage: Any, fov_x_degrees: float) -> Any:
    from pxr import UsdGeom

    made = UsdGeom.Camera.Define(stage, '/camera')
    aperture = 20.0
    made.CreateHorizontalApertureAttr(aperture)
    made.CreateVerticalApertureAttr(aperture)
    made.CreateFocalLengthAttr(float(aperture / (2 * np.tan(np.radians(fov_x_degrees) / 2))))
    made.CreateClippingRangeAttr((0.001, 100.0))
    return made


def cornell_stage(fault: str | None = None) -> Any:
    """Mitsuba's Cornell box, in USD. The small box sits under a parent, and the white parts use displayColor instead of a material."""
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    camera(stage, 39.3077).AddTranslateOp().Set(Gf.Vec3d(0, 0, 3.9))
    walls = {'floor': ((0, -1, 0), 'X', -90.0), 'ceiling': ((0, 1, 0), 'X', 90.0), 'back': ((0, 0, -1), None, 0.0),
             'green_wall': ((1, 0, 0), 'Y', -90.0), 'red_wall': ((-1, 0, 0), 'Y', 90.0)}
    green, red = material(stage, '/materials/green', GREEN), material(stage, '/materials/red', RED)
    for name, (position, axis, angle) in walls.items():
        mesh = quad(stage, f'/room/{name}')
        mesh.AddTranslateOp().Set(Gf.Vec3d(*position))
        if axis == 'X':
            mesh.AddRotateXOp().Set(angle)
        elif axis == 'Y':
            mesh.AddRotateYOp().Set(angle)
        if name == 'green_wall':
            bind(mesh, green)
        elif name == 'red_wall':
            bind(mesh, red)
        else:
            _display_color(mesh, WHITE)
    if fault != 'missing_mesh':
        parent = UsdGeom.Xform.Define(stage, '/room/small_box_parent')
        parent.AddTranslateOp().Set(Gf.Vec3d(0.335 + (0.1 if fault == 'wrong_transform' else 0.0), -0.7, 0.38))
        small = cube(stage, '/room/small_box_parent/small_box')
        small.AddRotateYOp().Set(-17.0)
        small.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.3))
        _display_color(small, WHITE)
    large = cube(stage, '/room/large_box')
    large.AddTranslateOp().Set(Gf.Vec3d(-0.33, -0.4, -0.28))
    large.AddRotateYOp().Set(18.25)
    large.AddScaleOp().Set(Gf.Vec3f(0.3, 0.61, 0.3))
    _display_color(large, WHITE)
    light = UsdLux.RectLight.Define(stage, '/light')
    light.AddTranslateOp().Set(Gf.Vec3d(0, 0.99, 0.01))
    light.AddRotateXOp().Set(-90.0)  # The light emits along -Z, so this points it at the floor.
    light.CreateWidthAttr(0.46 * (1.2 if fault == 'wrong_light_size' else 1.0))
    light.CreateHeightAttr(0.38)
    light.CreateIntensityAttr(18.387 / 2)  # With the exposure, this tests the 2^exposure rule.
    light.CreateExposureAttr(1.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 13.9873 / 18.387, 6.75357 / 18.387))
    return stage


def cornell_reference(mi: Any) -> dict[str, Any]:
    """mi.cornell_box() with a black surface on the light, because a RectLight has no surface."""
    scene = mi.cornell_box()
    scene['light']['bsdf'] = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.0, 0.0, 0.0]}}
    scene['sensor']['fov_axis'] = 'x'
    return scene


def look_at_row_matrix(origin: tuple[float, float, float], target: tuple[float, float, float], up: tuple[float, float, float]) -> Any:
    """A USD camera's matrix, for row vectors: it looks along -Z with +Y up."""
    from pxr import Gf

    forward = np.subtract(target, origin, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    rows = [[*right, 0.0], [*true_up, 0.0], [*(-forward), 0.0], [*origin, 1.0]]
    return Gf.Matrix4d(*[float(value) for row in rows for value in row])


def boxes_stage(fault: str | None = None) -> Any:
    """A box on a floor under a parallel light, Z up, with a mirrored parent and a light that uses normalize."""
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    camera(stage, 45.0).AddTransformOp().Set(look_at_row_matrix((4.0, -4.0, 3.0), (0.0, 0.0, 0.5), (0.0, 0.0, 1.0)))
    floor = quad(stage, '/world/floor')
    floor.AddScaleOp().Set(Gf.Vec3f(4, 4, 1))
    bind(floor, material(stage, '/materials/floor', (0.6, 0.6, 0.5)))
    if fault != 'missing_mesh':
        parent = UsdGeom.Xform.Define(stage, '/world/box_parent')
        parent.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.5, 0.5))
        parent.AddRotateZOp().Set(30.0 + (10.0 if fault == 'wrong_transform' else 0.0))
        parent.AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))  # A mirror: the converter must keep the faces pointing outward.
        box = cube(stage, '/world/box_parent/box')
        box.AddScaleOp().Set(Gf.Vec3f(0.5, 0.75, 0.5))
        bind(box, material(stage, '/materials/box', (0.2, 0.4, 0.7)))
    light = UsdLux.DistantLight.Define(stage, '/sun')
    light.AddRotateXYZOp().Set(Gf.Vec3f(35.0 + (10.0 if fault == 'wrong_light_direction' else 0.0), 0.0, 40.0))
    light.CreateIntensityAttr(3.0)  # With normalize, this is the illuminance, at any angle.
    light.CreateNormalizeAttr(True)
    light.CreateAngleAttr(0.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.9, 0.8))
    return stage


def sun_direction() -> list[float]:
    """rotateXYZ turns about X first, then Y, then Z. The light travels along its -Z. Worked by hand for a Y angle of 0."""
    ax, az = np.radians(35.0), np.radians(40.0)
    return [float(-np.sin(ax) * np.sin(az)), float(np.sin(ax) * np.cos(az)), float(-np.cos(ax))]


def boxes_reference(mi: Any) -> dict[str, Any]:
    t = mi.ScalarTransform4f
    # The same placement with Mitsuba's own cube. A mirror on x leaves the box as it was, so the reference does not need one.
    box = t().translate([0.25, 0.5, 0.5]) @ t().rotate([0, 0, 1], 30.0) @ t().scale([0.5, 0.75, 0.5])
    return {
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 8},
        'sensor': {
            'type': 'perspective', 'fov': 45.0, 'fov_axis': 'x', 'near_clip': 0.001, 'far_clip': 100.0,
            'to_world': t().look_at(origin=[4.0, -4.0, 3.0], target=[0.0, 0.0, 0.5], up=[0.0, 0.0, 1.0]),
            'sampler': {'type': 'independent', 'sample_count': PASS_SPP},
            'film': {'type': 'hdrfilm', 'width': 256, 'height': 256, 'rfilter': {'type': 'gaussian'}, 'pixel_format': 'rgb', 'component_format': 'float32'},
        },
        'floor': {'type': 'rectangle', 'to_world': t().scale([4.0, 4.0, 1.0]), 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.6, 0.6, 0.5]}}},
        'box': {'type': 'cube', 'to_world': box, 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.2, 0.4, 0.7]}}},
        'sun': {'type': 'directional', 'direction': sun_direction(), 'irradiance': {'type': 'rgb', 'value': [3.0, 2.7, 2.4]}},
    }


STAGES = {'cornell': cornell_stage, 'boxes': boxes_stage}
REFERENCES = {'cornell': cornell_reference, 'boxes': boxes_reference}


def render_seeds(mi: Any, scene: Any, seeds: tuple[int, ...]) -> dict[int, np.ndarray]:
    images = {}
    for seed in seeds:
        total = None
        for index in range(SPP // PASS_SPP):
            over = over_limit()
            if over:
                raise SystemExit(f'stopped: {over}')
            image = np.array(mi.render(scene, spp=PASS_SPP, seed=seed * 1000 + index), dtype=np.float64)
            total = image if total is None else total + image
        assert total is not None
        images[seed] = (total * PASS_SPP / SPP).astype(np.float32)
    return images


def run(args: argparse.Namespace) -> int:
    mi = load_mitsuba(args.variant)
    record: dict[str, Any] = {'schema': 'cumlx.m5_usd.v1', 'variant': args.variant, 'spp': SPP, 'mitsuba': mi.__version__, 'scenes': {}}
    code = 0
    for name in SCENES:
        begin = time.perf_counter()
        reference = render_seeds(mi, mi.load_dict(REFERENCES[name](mi)), REFERENCE_SEEDS)
        scene, read = load_usd(mi, STAGES[name](), spp=PASS_SPP)
        clean = compare_images(reference, render_seeds(mi, scene, USD_SEEDS))
        entry: dict[str, Any] = {'clean': clean, 'notes': read.notes, 'meshes': len(read.meshes), 'lights': len(read.lights), 'faults': {}}
        code = max(code, 0 if clean['verdict'] == 'ok' else 1)
        print(f"{name} clean: {clean['verdict']}", flush=True)
        for fault in FAULTS[name]:
            faulty, _ = load_usd(mi, STAGES[name](fault), spp=PASS_SPP)
            result = compare_images(reference, render_seeds(mi, faulty, FAULT_SEEDS))
            entry['faults'][fault] = result
            code = max(code, 0 if result['verdict'] == 'diverged' else 1)
            print(f"{name} {fault}: {result['verdict']}", flush=True)
        entry['seconds'] = round(time.perf_counter() - begin, 1)
        record['scenes'][name] = entry
    record['process_gpu_seconds'] = process_gpu_seconds(os.getpid())
    record['verdict'] = 'pass' if code == 0 else 'fail'
    Path(args.output).write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({'verdict': record['verdict'], 'process_gpu_seconds': record['process_gpu_seconds']}), flush=True)
    return code


def write(args: argparse.Namespace) -> int:
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        stage = STAGES[name]()  # Held here: USD aborts the process if a stage goes away during this call.
        stage.GetRootLayer().Export(str(folder / f'{name}.usda'))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    render = commands.add_parser('run')
    render.add_argument('--variant', default='metal_ad_rgb')
    render.add_argument('--output', required=True)
    render.set_defaults(handler=run)
    export = commands.add_parser('write')
    export.add_argument('--output-dir', required=True)
    export.set_defaults(handler=write)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
