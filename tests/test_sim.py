"""cumlx.sim, cumlx.render.lidar and `cumlx doctor --stack --env`. Engines are stubbed or skipped here: real runs are in validation/m4-sim.json."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cumlx.errors import ERROR_CODES, CumlxError
from cumlx.render.lidar import Box, Floor, LidarPattern, Sphere, analytic_ranges, compare_ranges
from cumlx.sim import available_tasks, compare_trajectories, get_task, worst_per_world
from cumlx.sim.task import Asset

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- trajectory comparison
def test_equal_runs_pass_and_a_small_difference_is_read_per_world():
    a = np.zeros((10, 4, 2))
    assert compare_trajectories(a, a.copy(), median_limit=0.01, p90_limit=0.05)['pass']
    b = a.copy()
    b[5, 3, 1] = 0.5  # One world: the largest value is reported, and the median and p90 limits do not bound it.
    assert worst_per_world(a, b).tolist() == [0.0, 0.0, 0.0, 0.5]
    assert compare_trajectories(a, b, median_limit=0.01, p90_limit=1.0)['max'] == 0.5


def test_a_large_difference_or_a_nan_fails():
    a = np.zeros((10, 4, 2))
    report = compare_trajectories(a, a + 0.1, median_limit=0.01, p90_limit=0.05)
    assert not report['pass'] and len(report['reasons']) == 2
    b = a.copy()
    b[0, 0, 0] = np.nan
    assert not compare_trajectories(a, b, median_limit=0.01, p90_limit=0.05)['pass']


def test_runs_of_different_shapes_are_refused():
    with pytest.raises(ValueError, match='shapes'):
        worst_per_world(np.zeros((2, 2, 2)), np.zeros((2, 3, 2)))


# ---------------------------------------------------------------- tasks
def test_each_task_names_its_parts_and_a_wrong_name_is_refused():
    assert available_tasks() == ['cartpole', 'go2_flat']
    with pytest.raises(CumlxError) as caught:
        get_task('nope')
    assert caught.value.code == 'E_INPUT_INVALID'


def test_cartpole_drives_the_slider_only():
    task = get_task('cartpole')  # The first comparison spun the pole, because the action drove the hinge too.
    assert task.joints == ('slider', 'hinge') and task.action_joints == ('slider',) and task.num_actions == 1


def test_go2_has_a_default_for_each_joint_and_the_observation_adds_up():
    task = get_task('go2_flat')
    assert len(task.actuation.default) == task.num_actions == 12
    assert task.observation_size == 3 + 3 + 3 + 12 + 12 + 12


def test_a_missing_asset_is_an_input_error(tmp_path):
    with pytest.raises(CumlxError) as caught:
        Asset('mjcf', 'nope.xml').path()
    assert caught.value.code == 'E_INPUT_NOT_FOUND'
    assert Asset('mjcf', 'cartpole.xml').path().is_file()


def test_the_error_code_for_an_unloadable_asset_exists():
    assert 'E_SIM_UNSUPPORTED' in ERROR_CODES


def _torch():
    return pytest.importorskip('torch')


def test_quaternion_rotation_into_the_body_frame():
    torch = _torch()
    from cumlx.sim.task import rotate_inverse
    half = math.sqrt(0.5)
    quat = torch.tensor([[half, 0.0, 0.0, half]])  # 90 degrees about z.
    out = rotate_inverse(quat, torch.tensor([[1.0, 0.0, 0.0]]))
    assert out.numpy() == pytest.approx(np.array([[0.0, -1.0, 0.0]]), abs=1e-6)


class FakeBackend:
    """Counts its calls. A world's cart moves with the action, so a termination can be driven from the test."""
    engine = 'fake'

    def __init__(self, task, num_envs):
        torch = _torch()
        self.torch, self.task, self.num_envs, self.device = torch, task, num_envs, 'cpu'
        self.pos, self.vel = torch.zeros((num_envs, 2)), torch.zeros((num_envs, 2))
        self.applied, self.steps, self.resets = None, 0, []

    def reset(self, envs, values):
        self.resets.append(envs.tolist())
        self.pos[envs] = values.joint_pos
        self.vel[envs] = values.joint_vel

    def apply(self, actions):
        self.applied = actions.clone()

    def step(self):
        self.steps += 1
        self.pos[:, 0] += self.applied[:, 0]

    def state(self):
        from cumlx.sim import State
        return State(self.pos.clone(), self.vel.clone())

    def failed_worlds(self):
        return None


def _env(num_envs=3):
    pytest.importorskip('tensordict')
    from cumlx.sim.env import TaskEnv
    task = get_task('cartpole')
    backend = FakeBackend(task, num_envs)
    return TaskEnv(task, backend, seed=1), backend


def test_env_ends_and_resets_only_the_world_that_left_the_rail():
    torch = _torch()
    env, backend = _env()
    backend.pos[1, 0] = 3.5
    _, reward, done, extras = env.step(torch.zeros((3, 1)))
    assert done.tolist() == [False, True, False] and extras['terminated'].tolist() == [False, True, False]
    assert reward[1] < 0 < reward[0]  # The termination penalty reaches the world that ended, before its reset.
    assert backend.resets[-1] == [1] and float(backend.pos[1, 0]) == 0.0
    assert env.episode_length_buf.tolist() == [1, 0, 1]


def test_env_times_out_without_calling_it_a_fall():
    torch = _torch()
    env, _ = _env()
    env.episode_length_buf[:] = env.max_episode_length - 1
    _, _, done, extras = env.step(torch.zeros((3, 1)))
    assert done.all() and extras['time_outs'].all() and not extras['terminated'].any()


def test_env_clips_actions_and_the_same_seed_repeats_the_reset():
    torch = _torch()
    env, backend = _env()
    env.step(torch.tensor([[9.0], [-9.0], [0.5]]))
    assert backend.applied.flatten().tolist() == [1.0, -1.0, 0.5]
    first, second = _env()[1], _env()[1]
    assert first.pos.tolist() == second.pos.tolist() and first.pos[:, 1].abs().sum() > 0  # The resets are random, and they repeat.


def test_a_missing_engine_is_a_dependency_error_and_not_a_pass(monkeypatch):
    from cumlx.sim import backends
    monkeypatch.setitem(sys.modules, 'genesis', None)  # An import of a module set to None raises ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, 'mujoco', None)
    for cls in (backends.GenesisBackend, backends.MujocoBackend):
        with pytest.raises(CumlxError) as caught:
            cls(get_task('cartpole'), 1)
        assert caught.value.code == 'E_DEP_MISSING'


def test_mujoco_refuses_usd(tmp_path):
    pytest.importorskip('mujoco')
    from dataclasses import replace

    from cumlx.sim.backends import MujocoBackend
    usd = tmp_path / 'a.usd'
    usd.write_text('')
    task = replace(get_task('cartpole'), asset=Asset('usd', str(usd)))
    with pytest.raises(CumlxError) as caught:
        MujocoBackend(task, 1)
    assert caught.value.code == 'E_SIM_UNSUPPORTED'


def test_mujoco_steps_cartpole_and_gravity_pulls_the_pole_down():
    pytest.importorskip('mujoco')
    torch = _torch()
    from cumlx.sim import ResetState
    from cumlx.sim.backends import MujocoBackend
    task = get_task('cartpole')
    backend = MujocoBackend(task, 2)
    backend.reset(torch.arange(2), ResetState(torch.tensor([[0.0, 0.1], [0.0, 0.0]]), torch.zeros((2, 2))))
    backend.apply(torch.zeros((2, 1)))
    for _ in range(30):
        backend.step()
    pose = backend.state().joint_pos
    assert float(pose[0, 1]) > 0.1 and abs(float(pose[1, 1])) < 1e-9  # A tilted pole falls, and an upright one with no push does not.


# ---------------------------------------------------------------- policy
def _layers():
    torch = _torch()
    torch.manual_seed(0)
    return [(torch.randn(8, 4), torch.randn(8)), (torch.randn(2, 8), torch.randn(2))]


def test_a_checkpoint_with_a_normalizer_is_refused(tmp_path):
    torch = _torch()
    from cumlx.sim.policy import load_actor_layers
    path = tmp_path / 'm.pt'
    torch.save({'actor_state_dict': {'obs_normalizer._mean': torch.zeros(4), 'mlp.0.weight': torch.zeros(1, 4), 'mlp.0.bias': torch.zeros(1)}}, path)
    with pytest.raises(CumlxError, match='normalizer'):
        load_actor_layers(str(path))


def test_a_checkpoint_with_no_actor_is_refused_and_a_good_one_reads_in_order(tmp_path):
    torch = _torch()
    from cumlx.sim.policy import load_actor_layers
    path = tmp_path / 'm.pt'
    torch.save({'other': torch.zeros(1)}, path)
    with pytest.raises(CumlxError, match='no mlp'):
        load_actor_layers(str(path))
    torch.save({'actor_state_dict': {'mlp.2.weight': torch.ones(1, 2), 'mlp.2.bias': torch.ones(1), 'mlp.0.weight': torch.zeros(2, 4), 'mlp.0.bias': torch.zeros(2), 'std': torch.zeros(1)}}, path)
    layers = load_actor_layers(str(path))
    assert [w.shape[0] for w, _ in layers] == [2, 1]


def test_policy_in_mlx_matches_and_a_wrong_activation_fails():
    torch = _torch()
    pytest.importorskip('mlx.core')
    import mlx.nn as nn

    from cumlx.sim.policy import mlx_policy, verify_policy
    layers = _layers()

    def reference(x):
        return torch.nn.functional.linear(torch.nn.functional.elu(torch.nn.functional.linear(x, *layers[0])), *layers[1])
    inputs = [np.random.default_rng(0).normal(size=(16, 4)).astype(np.float32)]
    assert verify_policy(reference, mlx_policy(layers), inputs)['verdict'] == 'ok'
    assert verify_policy(reference, mlx_policy(layers, activation=nn.relu), inputs)['verdict'] == 'diverged'
    assert verify_policy(reference, mlx_policy(layers), [])['verdict'] == 'incomplete'


# ---------------------------------------------------------------- raycast sensor, exact distances
PATTERN = LidarPattern(azimuths=8, elevations_degrees=(0.0, -30.0), max_range=20.0)


def test_exact_distances_on_each_shape():
    down = analytic_ranges((0.0, 0.0, 1.0), PATTERN, [Floor(0.0)])
    assert np.isnan(down[0]).all()  # A level ray never meets the floor.
    assert down[1] == pytest.approx(np.full(8, 2.0))  # sin(30 degrees) is one half.
    box = analytic_ranges((0.0, 0.0, 0.0), PATTERN, [Box((5.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    assert box[0, 0] == pytest.approx(4.0) and np.isnan(box[0, 4])
    sphere = analytic_ranges((0.0, 0.0, 0.0), PATTERN, [Sphere((0.0, 5.0, 0.0), 1.0)])
    assert sphere[0, 2] == pytest.approx(4.0) and np.isnan(sphere[0, 6])


def test_the_nearest_shape_wins_and_the_range_limit_applies():
    both = analytic_ranges((0.0, 0.0, 0.0), PATTERN, [Box((5.0, 0.0, 0.0), (1.0, 1.0, 1.0)), Sphere((10.0, 0.0, 0.0), 1.0)])
    assert both[0, 0] == pytest.approx(4.0)
    near = LidarPattern(azimuths=8, elevations_degrees=(0.0,), max_range=3.0)
    assert np.isnan(analytic_ranges((0.0, 0.0, 0.0), near, [Box((5.0, 0.0, 0.0), (1.0, 1.0, 1.0))])[0, 0])


def test_range_comparison_right_and_wrong():
    expected = np.array([[1.0, 2.0, np.nan, 4.0]])
    assert compare_ranges(expected + 1e-5, expected, range_limit=1e-3, hit_mismatch_limit=0.0)['pass']
    assert not compare_ranges(expected + 0.05, expected, range_limit=1e-3, hit_mismatch_limit=0.0)['pass']
    missing = expected.copy()
    missing[0, 0] = np.nan
    assert not compare_ranges(missing, expected, range_limit=1e-3, hit_mismatch_limit=0.0)['pass']
    nothing = np.full_like(expected, np.nan)
    assert not compare_ranges(nothing, expected, range_limit=1e-3, hit_mismatch_limit=1.0)['pass']  # No hit at all cannot pass.


# ---------------------------------------------------------------- the camera's surface rule
def _camera_stub(half=(0.5, 0.5, 0.5)):
    from cumlx.render.sensors import Camera
    from cumlx.sim import VisualLink
    from cumlx.sim.camera import TaskCamera
    stub = object.__new__(TaskCamera)
    stub.check_camera = Camera(origin=(0.0, -5.0, 0.0), target=(0.0, 0.0, 0.0), width=64, height=64)
    stub.links = {'box': VisualLink('box', half_extents=half)}
    stub.indices = {'box': 1}
    return stub


def _box_image(stub, center):
    """Depth and segmentation of the front face of a box at y = center_y - 0.5, by the ray and plane formula."""
    rays = stub.check_camera.ray_grid()
    t = (center[1] - 0.5 + 5.0) / rays[..., 1]
    points = np.array(stub.check_camera.origin) + rays * t[..., None]
    on_box = (np.abs(points[..., 0] - center[0]) <= 0.5) & (np.abs(points[..., 2] - center[2]) <= 0.5)
    return np.where(on_box, t, 0.0), np.where(on_box, 1, 0)


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def test_pixels_on_the_box_pass_and_a_stale_pose_fails():
    stub = _camera_stub()
    depth, segmentation = _box_image(stub, (0.0, 0.0, 0.0))
    right = stub.surface_excess(depth, segmentation, 'box', (np.zeros(3), np.array(IDENTITY)))
    assert right['ok'] and right['p99_m'] < 1e-6 and right['pixels'] > 20
    wrong = stub.surface_excess(depth, segmentation, 'box', (np.array([1.0, 0.0, 0.0]), np.array(IDENTITY)))
    assert not wrong['ok'] and wrong['p99_m'] > 0.1


def test_a_link_with_too_few_pixels_is_not_a_pass():
    stub = _camera_stub()
    report = stub.surface_excess(np.zeros((64, 64)), np.zeros((64, 64), dtype=int), 'box', (np.zeros(3), np.array(IDENTITY)))
    assert not report['ok'] and report['p99_m'] is None


# ---------------------------------------------------------------- the record
def _record_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('m4_sim_record', ROOT / 'spikes' / 'm4_sim_record.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_part_with_no_output_is_not_run_and_not_a_pass(tmp_path):
    record = _record_module()
    assert record.trajectory(tmp_path, 'go2_flat', ['fault_kp'])['verdict'] == 'not_run'
    assert record.training(tmp_path, 'cartpole', 'mujoco')['verdict'] == 'not_run'


def test_a_trajectory_passes_only_if_its_faults_fail(tmp_path):
    record = _record_module()
    folder = tmp_path / 'compare' / 'go2_flat'
    folder.mkdir(parents=True)
    zero = np.zeros((3, 4, 2))
    for seed in (101, 102, 103):
        for name in ('mujoco', 'mujoco_fine', 'genesis'):
            np.savez(folder / f'{name}_{seed}.npz', values=zero)
        np.savez(folder / f'fault_kp_{seed}.npz', values=zero + 1.0)
    assert record.trajectory(tmp_path, 'go2_flat', ['fault_kp'])['verdict'] == 'pass'
    np.savez(folder / 'fault_kp_102.npz', values=zero)  # A fault that goes unseen: the measure cannot be trusted.
    assert record.trajectory(tmp_path, 'go2_flat', ['fault_kp'])['verdict'] == 'fail'


def test_the_written_limits_of_each_measure_are_there():
    limits = json.loads((ROOT / 'validation' / 'm4-sim-limits.json').read_text())
    assert limits['trajectory']['cartpole']['median_limit'] == 0.02 and limits['trajectory']['cartpole']['p90_limit'] == 0.05
    assert limits['training']['policy_in_mlx']['limit_rel_l2'] == 1e-4
    assert json.loads((ROOT / 'validation' / 'm5-lidar-limits.json').read_text())['limits']['range_limit_m'] == 0.001


# ---------------------------------------------------------------- doctor --stack --env
def _fake_python(path: Path, statuses: dict[str, str]) -> str:
    """A program that answers as an environment does, and writes a mark that proves it ran."""
    layers = {name: {'status': status, 'ok': status == 'ok', 'detail': f'fake {status}', 'install_hint': 'hint', 'needed_by': 'x', 'checked': 'presence', 'version': None}
              for name, status in statuses.items()}
    answer = json.dumps({'layers': layers})
    path.write_text(f"#!/bin/sh\necho ran >> {path}.mark\ncat <<'EOF'\nsome other output\nCUMLX-STACK {answer}\nEOF\n")
    path.chmod(0o755)
    return str(path)


def _doctor(capsys, *args):
    from cumlx.cli.main import main
    code = main(['doctor', '--format', 'json', *args])
    return code, json.loads(capsys.readouterr().out)


def test_a_stack_spread_over_two_environments_is_ready(tmp_path, capsys):
    a = _fake_python(tmp_path / 'a', {'mlx': 'ok', 'genesis': 'missing'})
    b = _fake_python(tmp_path / 'b', {'mlx': 'missing', 'genesis': 'ok'})
    code, doc = _doctor(capsys, '--stack', '--env', a, '--env', b)
    stack = doc['data']['stack']
    assert code == 0 and stack['not_ready'] == [] and stack['unreadable'] == {}
    assert stack['layers']['mlx']['found_in'] == a and stack['layers']['genesis']['found_in'] == b
    assert stack['layers']['genesis']['environments'] == {a: 'missing', b: 'ok'}
    assert 'load together' in stack['scope']
    assert (tmp_path / 'a.mark').exists() and (tmp_path / 'b.mark').exists()  # The fakes answered.


def test_a_layer_ready_nowhere_is_exit_3(tmp_path, capsys):
    a = _fake_python(tmp_path / 'a', {'mlx': 'ok', 'translator': 'missing'})
    b = _fake_python(tmp_path / 'b', {'mlx': 'ok', 'translator': 'broken'})
    code, doc = _doctor(capsys, '--stack', '--env', a, '--env', b)
    assert code == 3 and doc['data']['stack']['not_ready'] == ['translator']
    assert doc['data']['stack']['layers']['translator']['found_in'] is None
    assert any(e['code'] == 'E_DEP_MISSING' and e['details'].get('layer') == 'translator' for e in doc['errors'])


def test_an_environment_that_cannot_answer_is_not_a_pass(tmp_path, capsys):
    good = _fake_python(tmp_path / 'a', {'mlx': 'ok'})
    silent = tmp_path / 'silent'
    silent.write_text('#!/bin/sh\necho no answer\n')
    silent.chmod(0o755)
    for bad in (str(silent), str(tmp_path / 'no-such-python')):
        code, doc = _doctor(capsys, '--stack', '--env', good, '--env', bad)
        assert code == 3 and bad in doc['data']['stack']['unreadable']
        assert any(e['details'].get('python') == bad for e in doc['errors'])


def test_env_without_stack_is_a_usage_error(tmp_path, capsys):
    code, doc = _doctor(capsys, '--env', 'x')
    assert code == 2 and doc['errors'][0]['code'] == 'E_USAGE'


def test_the_probe_reads_env_py_alone_in_a_real_python():
    from cumlx.verify.env import _probe_environment
    answer = _probe_environment(sys.executable)
    assert 'error' not in answer, answer
    assert 'mlx' in answer['layers'] and 'translator' in answer['layers']


def test_the_stack_variant_is_the_render_default():
    from cumlx.render.sensors import DEFAULT_VARIANT
    from cumlx.verify.env import STACK_VARIANT
    assert STACK_VARIANT == DEFAULT_VARIANT


def test_mcp_hides_env_unless_code_execution_is_allowed():
    from cumlx.mcp_server import build_tools
    hidden = json.dumps(build_tools(allow_code=False), default=str)
    shown = json.dumps(build_tools(allow_code=True), default=str)
    assert 'env_python' not in hidden and 'env_python' in shown


# ---------------------------------------------------------------- the memory guard
GUARD = ROOT / 'spikes' / 'run_guarded.py'


def test_guard_runs_a_command_under_its_limits_and_passes_on_its_exit_code(tmp_path):
    done = subprocess.run([sys.executable, str(GUARD), '--stop-gb', '1000', '--start-gb', '1000', '--disk-gb', '0', '--output', str(tmp_path / 'u.json'), '--',
                           sys.executable, '-c', 'raise SystemExit(7)'], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 7
    assert json.loads((tmp_path / 'u.json').read_text())['stopped_for'] is None


def test_guard_stops_a_command_when_free_disk_is_under_its_limit(tmp_path):
    done = subprocess.run([sys.executable, str(GUARD), '--stop-gb', '1000', '--start-gb', '1000', '--process-gb', '1000', '--disk-gb', '1000000000', '--interval', '0.1',
                           '--output', str(tmp_path / 'u.json'), '--', sys.executable, '-c', 'import time; time.sleep(60)'],
                          capture_output=True, text=True, timeout=50, check=False)
    assert done.returncode == 76
    assert 'of disk is free' in json.loads((tmp_path / 'u.json').read_text())['stopped_for']


def test_guard_does_not_call_a_finished_command_stopped(tmp_path):
    """Its exit code passes through even if a limit is broken at the moment that it ends."""
    done = subprocess.run([sys.executable, str(GUARD), '--stop-gb', '0', '--start-gb', '1000', '--interval', '3', '--output', str(tmp_path / 'u.json'), '--',
                           sys.executable, '-c', 'raise SystemExit(7)'], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 7
    assert json.loads((tmp_path / 'u.json').read_text())['stopped_for'] is None


def test_guard_refuses_to_start_over_its_start_limit(tmp_path):
    mark = tmp_path / 'ran'
    done = subprocess.run([sys.executable, str(GUARD), '--start-gb', '0', '--', sys.executable, '-c', f'open({str(mark)!r}, "w")'],
                          capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 75 and not mark.exists()


def test_guard_refuses_when_memory_cannot_be_read(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('run_guarded_under_test', GUARD)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard.sysmem, 'used_gb', lambda: None)
    monkeypatch.setattr(guard.sysmem, 'limit_gb', lambda: None)
    mark = tmp_path / 'ran'
    monkeypatch.setattr(sys, 'argv', ['run_guarded', '--', sys.executable, '-c', f'open({str(mark)!r}, "w")'])
    assert guard.main() == guard.REFUSED and not mark.exists()  # No limit known.
    monkeypatch.setattr(sys, 'argv', ['run_guarded', '--stop-gb', '38', '--', sys.executable, '-c', f'open({str(mark)!r}, "w")'])
    with pytest.raises(SystemExit) as stopped:
        guard.main()
    assert stopped.value.code == guard.REFUSED and not mark.exists()  # A limit, but no reading.


def test_guard_stops_a_command_over_its_stop_limit_and_only_that_command(tmp_path):
    done = subprocess.run([sys.executable, str(GUARD), '--stop-gb', '0', '--start-gb', '1000', '--interval', '0.1', '--output', str(tmp_path / 'u.json'), '--',
                           sys.executable, '-c', 'import time; time.sleep(60)'], capture_output=True, text=True, timeout=50, check=False)
    assert done.returncode == 76
    assert 'over 0' in json.loads((tmp_path / 'u.json').read_text())['stopped_for']
