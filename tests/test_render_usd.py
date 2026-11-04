"""cumlx.render.usd. The arithmetic needs only NumPy, and the reader needs real pxr. No renderer runs here: spikes/m5_usd.py renders."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from cumlx.errors import ERROR_CODES, CumlxError
from cumlx.render.usd import _as_triangles, distant_size_factor, read_usd, transform_mesh, write_ply

SPIKE = Path(__file__).resolve().parent.parent / 'spikes' / 'm5_usd.py'


def _pxr():
    """CI sets CUMLX_REQUIRE_USD in the job that installs usd-core, so that a skip there is a failure."""
    if os.environ.get('CUMLX_REQUIRE_USD') and importlib.util.find_spec('pxr') is None:
        pytest.fail('CUMLX_REQUIRE_USD is set, and pxr is not installed')
    return pytest.importorskip('pxr')


@pytest.fixture
def scenes():
    _pxr()
    spec = importlib.util.spec_from_file_location('m5_usd_under_test', SPIKE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


def _face_normal(mesh, triangle):
    a, b, c = mesh.points[mesh.triangles[triangle]]
    normal = np.cross(b - a, c - a)
    return normal / np.linalg.norm(normal)


def _refused(source, **options):
    with pytest.raises(CumlxError) as caught:
        read_usd(source, **options)
    assert caught.value.code == 'E_RENDER_USD_UNSUPPORTED'
    return caught.value.details['refused']


def test_a_missing_pxr_is_the_documented_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pxr', None)
    with pytest.raises(CumlxError) as caught:
        read_usd('anything.usda')
    assert caught.value.code == 'E_DEP_MISSING'
    assert caught.value.details['extra'] == 'usd'
    assert "cumlx-lab[usd]" in caught.value.hint


def test_the_new_error_code_is_registered():
    assert 'E_RENDER_USD_UNSUPPORTED' in ERROR_CODES


def test_a_fan_covers_a_polygon_and_bad_faces_are_refused():
    triangles = _as_triangles(np.array([3, 5]), np.array([0, 1, 2, 3, 4, 5, 6, 7]), 8)
    assert triangles.tolist() == [[0, 1, 2], [3, 4, 5], [3, 5, 6], [3, 6, 7]]
    for counts, indices, why in (([2], [0, 1], 'fewer than 3'), ([3], [0, 1], 'sums to'), ([3], [0, 1, 9], 'outside')):
        with pytest.raises(ValueError, match=why):
            _as_triangles(np.array(counts), np.array(indices), 3)


def test_a_mirror_and_a_left_handed_mesh_keep_their_front_side():
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2]])
    mirror = np.diag([-1.0, 1, 1, 1])
    assert transform_mesh(points, triangles, np.eye(4), False)[1].tolist() == [[0, 1, 2]]
    assert transform_mesh(points, triangles, mirror, False)[1].tolist() == [[0, 2, 1]]
    assert transform_mesh(points, triangles, np.eye(4), True)[1].tolist() == [[0, 2, 1]]
    assert transform_mesh(points, triangles, mirror, True)[1].tolist() == [[0, 1, 2]]  # Two flips cancel.


def test_the_distant_size_factor_follows_the_schema():
    assert distant_size_factor(0.0) == 1.0
    theta = np.radians(0.53) / 2
    assert distant_size_factor(0.53) == pytest.approx(np.pi * np.sin(theta) ** 2)
    assert distant_size_factor(180.0) == pytest.approx(np.pi)  # The two branches meet at a hemisphere.
    assert distant_size_factor(270.0) == pytest.approx((2 - np.sin(np.radians(135)) ** 2) * np.pi)


def test_write_ply_writes_what_it_says(tmp_path):
    path = tmp_path / 'a.ply'
    write_ply(path, np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[0, 1, 2]]))
    data = path.read_bytes()
    header, body = data.split(b'end_header\n')
    assert b'element vertex 3' in header and b'element face 1' in header
    assert np.frombuffer(body[:36], dtype='<f4').tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert body[36] == 3 and np.frombuffer(body[37:49], dtype='<i4').tolist() == [0, 1, 2]
    assert len(body) == 49


def test_the_cornell_stage_reads_to_the_known_numbers(scenes):
    scene = read_usd(scenes.cornell_stage())
    assert scene.up_axis == 'Y' and len(scene.meshes) == 7 and len(scene.lights) == 1
    by_name = {m.path.rsplit('/', 1)[1]: m for m in scene.meshes}
    # Mitsuba's own matrix for the small box, from mi.cornell_box(), applied to one corner of the cube.
    matrix = np.array([[0.286891, 0, -0.0877115, 0.335], [0, 0.3, 0, -0.7], [0.0877115, 0, 0.286891, 0.38]])
    corner = matrix @ np.array([1.0, 1.0, 1.0, 1.0])
    assert np.abs(by_name['small_box'].points - corner).sum(axis=1).min() < 1e-5
    assert by_name['floor'].reflectance == pytest.approx(scenes.WHITE)
    assert by_name['green_wall'].reflectance == pytest.approx(scenes.GREEN)
    assert _face_normal(by_name['floor'], 0) == pytest.approx([0, 1, 0], abs=1e-6)
    assert _face_normal(by_name['green_wall'], 0) == pytest.approx([-1, 0, 0], abs=1e-6)
    assert scene.camera.to_world.tolist() == [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 3.9], [0, 0, 0, 1]]
    assert scene.camera.fov_x_degrees == pytest.approx(39.3077, abs=1e-4)
    light = scene.lights[0]
    assert light.radiance == pytest.approx((18.387, 13.9873, 6.75357), rel=1e-5)  # 2^exposure, and the color.
    assert light.to_world[:3, 2] == pytest.approx([0, -1, 0], abs=1e-6)  # The emitting side faces the floor.
    assert np.linalg.norm(light.to_world[:3, 0]) == pytest.approx(0.23, rel=1e-6) and np.linalg.norm(light.to_world[:3, 1]) == pytest.approx(0.19, rel=1e-6)
    assert np.linalg.det(light.to_world[:3, :3]) > 0


def test_the_box_stage_reads_to_the_known_numbers(scenes):
    scene = read_usd(scenes.boxes_stage())
    assert scene.up_axis == 'Z' and len(scene.meshes) == 2
    box = next(m for m in scene.meshes if m.path.endswith('box'))
    center = box.points.mean(axis=0)
    assert center == pytest.approx([0.25, 0.5, 0.5], abs=1e-9)
    for index in range(len(box.triangles)):  # The parent mirrors the box, and each face must still point away from its center.
        assert np.dot(_face_normal(box, index), box.points[box.triangles[index, 0]] - center) > 0
    light = scene.lights[0]
    assert light.direction == pytest.approx(scenes.sun_direction(), abs=1e-6)
    assert light.irradiance == pytest.approx((3.0, 2.7, 2.4), rel=1e-6)  # normalize: the intensity is the illuminance.
    assert scene.camera.fov_x_degrees == pytest.approx(45.0, abs=1e-4)
    forward = scene.camera.to_world[:3, 2]  # Mitsuba's camera looks along its +Z.
    expected = np.array([-4.0, 4.0, -2.5]) / np.linalg.norm([-4.0, 4.0, -2.5])
    assert forward == pytest.approx(expected, abs=1e-9)


def test_faults_change_what_is_read(scenes):
    clean = read_usd(scenes.cornell_stage())
    assert len(read_usd(scenes.cornell_stage('missing_mesh')).meshes) == len(clean.meshes) - 1
    moved = read_usd(scenes.cornell_stage('wrong_transform'))
    small = [m for m in moved.meshes if m.path.endswith('small_box')][0]
    assert small.points.mean(axis=0)[0] == pytest.approx(0.435, abs=1e-6)


def test_a_file_on_disk_reads_like_the_stage(scenes, tmp_path):
    path = tmp_path / 'cornell.usda'
    stage = scenes.cornell_stage()  # Held here: USD aborts the process if a stage goes away during this call.
    stage.GetRootLayer().Export(str(path))
    scene = read_usd(path)
    assert scene.source == str(path) and len(scene.meshes) == 7


def test_a_missing_or_broken_file_is_an_input_error(scenes, tmp_path):
    with pytest.raises(CumlxError) as caught:
        read_usd(tmp_path / 'none.usda')
    assert caught.value.code == 'E_INPUT_NOT_FOUND'
    broken = tmp_path / 'broken.usda'
    broken.write_text('this is not a USD file')
    with pytest.raises(CumlxError) as caught:
        read_usd(broken)
    assert caught.value.code == 'E_INPUT_INVALID'


def _small_stage(scenes):
    from pxr import Usd, UsdLux

    stage = Usd.Stage.CreateInMemory()
    scenes.camera(stage, 45.0)
    scenes.quad(stage, '/quad')
    UsdLux.RectLight.Define(stage, '/light')
    return stage


def test_a_small_scene_is_accepted(scenes):
    scene = read_usd(_small_stage(scenes))
    assert any('no material' in note for note in scene.notes)  # The default color is said, not silent.


@pytest.mark.parametrize('kind', ['Sphere', 'Cube', 'Cone', 'Capsule', 'Cylinder', 'PointInstancer'])
def test_an_unconverted_geometry_is_refused_by_name_and_not_left_out(scenes, kind):
    stage = _small_stage(scenes)
    stage.DefinePrim('/extra', kind)
    assert any(item['prim'] == '/extra' and kind in item['reason'] for item in _refused(stage))


@pytest.mark.parametrize('kind', ['SphereLight', 'DomeLight', 'DiskLight', 'CylinderLight'])
def test_an_unconverted_light_is_refused(scenes, kind):
    stage = _small_stage(scenes)
    stage.DefinePrim('/extra', kind)
    assert any(item['prim'] == '/extra' for item in _refused(stage))


@pytest.mark.parametrize('name,value', [('metallic', 1.0), ('opacity', 0.5), ('clearcoat', 1.0)])
def test_a_material_beyond_diffuse_is_refused(scenes, name, value):
    from pxr import Sdf, UsdGeom, UsdShade

    stage = _small_stage(scenes)
    made = scenes.material(stage, '/m', (0.1, 0.2, 0.3))
    UsdShade.Shader.Get(stage, '/m/surface').CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    scenes.bind(UsdGeom.Mesh.Get(stage, '/quad'), made)
    assert any(name in item['reason'] for item in _refused(stage))


def test_a_textured_diffuse_color_is_refused(scenes):
    from pxr import Sdf, UsdGeom, UsdShade

    stage = _small_stage(scenes)
    made = scenes.material(stage, '/m', (0.1, 0.2, 0.3))
    texture = UsdShade.Shader.Define(stage, '/m/texture')
    texture.CreateIdAttr('UsdUVTexture')
    output = texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    UsdShade.Shader.Get(stage, '/m/surface').GetInput('diffuseColor').ConnectToSource(output)
    scenes.bind(UsdGeom.Mesh.Get(stage, '/quad'), made)
    assert any('diffuseColor' in item['reason'] for item in _refused(stage))


def test_a_scene_with_no_light_or_no_camera_or_no_mesh_is_refused(scenes):
    from pxr import UsdGeom

    for drop in ('/light', '/camera', '/quad'):
        stage = _small_stage(scenes)
        stage.RemovePrim(drop)
        assert any(f"no {drop.strip('/').replace('quad', 'mesh')}" in item['reason'] for item in _refused(stage))
    stage = _small_stage(scenes)
    UsdGeom.Camera.Define(stage, '/second')
    assert any('more than one camera' in item['reason'] for item in _refused(stage))
    assert read_usd(stage, camera='/second').camera.path == '/second'


def test_an_invisible_mesh_and_a_guide_are_left_out_as_usd_says(scenes):
    stage = _small_stage(scenes)
    scenes.quad(stage, '/hidden').CreateVisibilityAttr('invisible')
    scenes.quad(stage, '/guide').CreatePurposeAttr('guide')
    assert [m.path for m in read_usd(stage).meshes] == ['/quad']


def test_a_camera_that_cannot_be_converted_is_refused(scenes):
    from pxr import Gf, UsdGeom

    stage = _small_stage(scenes)
    UsdGeom.Camera.Get(stage, '/camera').AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    assert any('scale' in item['reason'] for item in _refused(stage))
    stage = _small_stage(scenes)
    UsdGeom.Camera.Get(stage, '/camera').CreateProjectionAttr('orthographic')
    assert any('perspective' in item['reason'] for item in _refused(stage))


def test_a_rect_light_with_normalize_divides_by_its_area(scenes):
    from pxr import Gf, UsdLux

    stage = _small_stage(scenes)
    light = UsdLux.RectLight.Get(stage, '/light')
    light.CreateWidthAttr(2.0)
    light.CreateHeightAttr(3.0)
    light.CreateIntensityAttr(12.0)
    light.CreateNormalizeAttr(True)
    light.AddScaleOp().Set(Gf.Vec3f(2, 1, 1))
    assert read_usd(stage).lights[0].radiance == pytest.approx((1.0, 1.0, 1.0))  # The world-space area is 4 x 3.


def test_a_distant_light_without_normalize_uses_the_size_factor(scenes):
    from pxr import UsdLux

    stage = _small_stage(scenes)
    stage.RemovePrim('/light')
    light = UsdLux.DistantLight.Define(stage, '/sun')
    light.CreateIntensityAttr(50000.0)
    light.CreateAngleAttr(0.53)
    scene = read_usd(stage)
    assert scene.lights[0].irradiance[0] == pytest.approx(50000.0 * distant_size_factor(0.53), rel=1e-5)
    assert scene.lights[0].direction == pytest.approx((0.0, 0.0, -1.0))
    assert any('parallel light' in note for note in scene.notes)


@pytest.mark.parametrize('attribute,value,why', [('CreateDiffuseAttr', 0.5, 'diffuse'), ('CreateSpecularAttr', 0.5, 'specular'), ('CreateEnableColorTemperatureAttr', True, 'temperature')])
def test_a_light_control_that_is_not_converted_is_refused(scenes, attribute, value, why):
    from pxr import UsdLux

    stage = _small_stage(scenes)
    getattr(UsdLux.RectLight.Get(stage, '/light'), attribute)(value)
    assert any(why in item['reason'] for item in _refused(stage))


def test_a_geom_subset_and_holes_are_refused(scenes):
    from pxr import UsdGeom, Vt

    stage = _small_stage(scenes)
    mesh = UsdGeom.Mesh.Get(stage, '/quad')
    UsdGeom.Subset.CreateGeomSubset(UsdGeom.Imageable(mesh), 'part', 'face', Vt.IntArray([0]))
    assert any('GeomSubset' in item['reason'] for item in _refused(stage))
    stage = _small_stage(scenes)
    UsdGeom.Mesh.Get(stage, '/quad').CreateHoleIndicesAttr([0])
    assert any('holeIndices' in item['reason'] for item in _refused(stage))


def test_a_transform_at_a_time_is_read_at_that_time(scenes):
    from pxr import Gf, UsdGeom

    stage = _small_stage(scenes)
    op = UsdGeom.Mesh.Get(stage, '/quad').AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 1.0)
    op.Set(Gf.Vec3d(0, 0, 10), 2.0)
    assert read_usd(stage, time=1.0).meshes[0].points[:, 2].max() == pytest.approx(0.0)
    assert read_usd(stage, time=2.0).meshes[0].points[:, 2].max() == pytest.approx(10.0)
