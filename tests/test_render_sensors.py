"""cumlx.render.sensors: camera geometry and frame checks on synthetic frames, and real renders when Mitsuba is installed."""
import math
import sys

import numpy as np
import pytest

from cumlx.errors import CumlxError
from cumlx.render.sensors import Camera, PIXEL_LIMIT, centroid, check_frame, load_mitsuba, rotate

CAMERA = Camera(origin=(4.0, -4.0, 2.5), target=(0.0, 0.0, 0.6))
HALF = 0.25


def _raycast(camera, position, half):
    """A synthetic renderer in NumPy: an axis-aligned cube over a floor at z = 0. Index 0 is the cube and 1 the floor."""
    directions = camera.ray_grid()
    origin = np.array(camera.origin)
    floor_t = np.where(directions[..., 2] < 0, -origin[2] / directions[..., 2], np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        low = (np.array(position) - half - origin) / directions
        high = (np.array(position) + half - origin) / directions
    near = np.minimum(low, high).max(axis=-1)
    far = np.maximum(low, high).min(axis=-1)
    box = (near <= far) & (far > 0)
    depth = np.where(box, near, np.where(np.isfinite(floor_t), floor_t, 0))
    segmentation = np.where(box, 0, np.where(np.isfinite(floor_t), 1, -1))
    return depth.astype(np.float32), segmentation.astype(np.int32)


def test_the_target_projects_to_the_image_centre_and_rays_round_trip():
    assert CAMERA.project(CAMERA.target) == pytest.approx((128.0, 128.0))
    x, y = CAMERA.project((0.5, 0.2, 0.1))
    origin, direction = np.array(CAMERA.origin), CAMERA.ray_direction(x, y)
    expected = np.subtract((0.5, 0.2, 0.1), origin)
    assert direction == pytest.approx(expected / np.linalg.norm(expected))
    with pytest.raises(ValueError):
        CAMERA.project((8.0, -8.0, 4.4))


def test_right_and_up_in_the_world_are_right_and_up_in_the_image():
    camera = Camera(origin=(0.0, -5.0, 0.0), target=(0.0, 0.0, 0.0))
    centre_x, centre_y = camera.project((0.0, 0.0, 0.0))
    right_x, _ = camera.project((1.0, 0.0, 0.0))
    _, up_y = camera.project((0.0, 0.0, 1.0))
    assert right_x > centre_x and up_y < centre_y


def test_focal_length_follows_the_horizontal_field_of_view():
    camera = Camera(origin=(0.0, -1.0, 0.0), target=(0.0, 0.0, 0.0), fov_x_degrees=90.0)
    assert camera.project((1.0, 0.0, 0.0))[0] == pytest.approx(256.0)


def test_near_offset_is_the_near_clip_at_the_centre_and_longer_at_the_corner():
    offsets = CAMERA.near_offsets()
    assert offsets[128, 128] == pytest.approx(CAMERA.near_clip, rel=1e-4)
    assert offsets[0, 0] > offsets[128, 128]
    half_diagonal = math.hypot(127.5, 127.5) / (128 / math.tan(math.radians(CAMERA.fov_x_degrees) / 2))
    assert offsets[0, 0] == pytest.approx(CAMERA.near_clip * math.hypot(1, half_diagonal))


def test_rotate_by_a_known_quarter_turn_and_by_nothing():
    half = math.sqrt(0.5)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert rotate((half, 0.0, 0.0, half), vectors) == pytest.approx(np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
    assert rotate((1.0, 0.0, 0.0, 0.0), vectors) == pytest.approx(vectors)


def test_centroid_of_a_known_mask_and_of_nothing():
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:4, 4:6] = 3
    assert centroid(mask, 3) == (5.0, 3.0)
    assert centroid(mask, 9) is None


def test_a_right_frame_passes():
    position = (0.2, 0.1, 0.6)
    depth, segmentation = _raycast(CAMERA, position, HALF)
    result = check_frame(CAMERA, depth, segmentation, 0, 1, position, HALF)
    assert result['ok'], result['failures']
    assert result['pixel_error'] < PIXEL_LIMIT


def test_a_stale_pose_fails():
    depth, segmentation = _raycast(CAMERA, (0.0, 0.0, 1.5), HALF)
    result = check_frame(CAMERA, depth, segmentation, 0, 1, (0.0, 0.0, 0.25), HALF)
    assert not result['ok'] and any('projected' in failure for failure in result['failures'])


def test_a_moved_camera_fails():
    position = (0.2, 0.1, 0.6)
    moved = Camera(origin=(4.3, -4.0, 2.5), target=CAMERA.target)
    depth, segmentation = _raycast(moved, position, HALF)
    assert not check_frame(CAMERA, depth, segmentation, 0, 1, position, HALF)['ok']


def test_depth_on_the_wrong_scale_fails():
    position = (0.2, 0.1, 0.6)
    depth, segmentation = _raycast(CAMERA, position, HALF)
    result = check_frame(CAMERA, depth * 1.01, segmentation, 0, 1, position, HALF)
    assert not result['ok'] and any('floor depth' in failure for failure in result['failures'])


def test_a_missing_body_and_a_missing_floor_fail():
    position = (0.2, 0.1, 0.6)
    depth, segmentation = _raycast(CAMERA, position, HALF)
    assert not check_frame(CAMERA, depth, np.where(segmentation == 0, 1, segmentation), 0, 1, position, HALF)['ok']
    assert not check_frame(CAMERA, depth, np.where(segmentation == 1, -1, segmentation), 0, 1, position, HALF)['ok']


def test_a_missing_mitsuba_is_an_environment_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mitsuba', None)
    with pytest.raises(CumlxError) as caught:
        load_mitsuba()
    assert caught.value.code == 'E_DEP_MISSING'
    assert 'render' in caught.value.hint


@pytest.mark.gpu
def test_real_metal_render_agrees_with_the_camera_and_a_stale_pose_fails():
    mi = pytest.importorskip('mitsuba')
    if 'metal_ad_rgb' not in mi.variants():
        pytest.skip('needs Mitsuba on Metal')
    from cumlx.render.sensors import Sensors, box_scene

    mi = load_mitsuba('metal_ad_rgb')
    camera = CAMERA
    sensors = Sensors(box_scene(mi, camera, HALF, position=(0.0, 0.0, 1.0)), 'metal_ad_rgb', camera=camera)
    body, floor = sensors.shape_ids.index('box'), sensors.shape_ids.index('floor')
    for position in ((0.0, 0.0, 1.0), (0.3, -0.2, 0.6)):
        sensors.set_pose('box', position)
        depth, segmentation = sensors.depth_segmentation(seed=1)
        result = check_frame(camera, depth, segmentation, body, floor, position, HALF)
        assert result['ok'], result['failures']
        rgb = sensors.rgb(spp=4, seed=1)
        assert rgb.shape == (256, 256, 3) and rgb.mean() > 0.01
    assert not check_frame(camera, depth, segmentation, body, floor, (0.0, 0.0, 1.0), HALF)['ok']
