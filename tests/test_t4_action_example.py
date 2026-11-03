"""spikes/t4_action_example.sh: the refusals that come before any build or network call."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / 'spikes' / 't4_action_example.sh'
EDITOR = REPO / 'spikes' / 't4_edit_workflow.py'
WORKFLOW = REPO / 'examples/port-ci/.github/workflows/verify-port.yml'


def _run(*args, out):
    env = {**os.environ, 'CUMLX_ACTION_EXAMPLE': str(out)}
    return subprocess.run(['sh', str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=30)


def test_prepare_refuses_a_folder_that_exists_and_leaves_it_alone(tmp_path):
    keep = tmp_path / 'keep.txt'
    keep.write_text('mine')
    run = _run('prepare', out=tmp_path)
    assert run.returncode == 4 and 'exists' in run.stderr
    assert [p.name for p in tmp_path.iterdir()] == ['keep.txt'] and keep.read_text() == 'mine'


def test_an_unknown_mode_is_a_usage_error(tmp_path):
    assert _run('publish', out=tmp_path / 'absent').returncode == 2
    assert not (tmp_path / 'absent').exists()


def test_a_repository_with_no_venv_is_an_environment_error_with_the_fix(tmp_path):
    (tmp_path / 'spikes').mkdir()
    shutil.copy(SCRIPT, tmp_path / 'spikes')
    env = {**os.environ, 'CUMLX_ACTION_EXAMPLE': str(tmp_path / 'absent')}
    run = subprocess.run(['sh', str(tmp_path / 'spikes/t4_action_example.sh'), 'prepare'], env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 3 and '.venv' in run.stderr and 'python -m venv' in run.stderr
    assert not (tmp_path / 'absent').exists()


def _editor():
    spec = importlib.util.spec_from_file_location('t4_edit_workflow', EDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_edit_of_the_real_example_workflow():
    out = _editor().edit(WORKFLOW.read_text(), 'cumlx_lab-0-py3-none-any.whl')
    assert 'Topabaem05/CuMLX@main' not in out and '- uses: ./.cumlx-action' in out
    assert '        with:\n          cumlx-package: ${{ github.workspace }}/.cumlx-action/cumlx_lab-0-py3-none-any.whl\n          original: original' in out


def test_a_workflow_without_the_expected_lines_is_refused_and_left_alone(tmp_path):
    with pytest.raises(ValueError):
        _editor().edit('jobs: {}\n', 'x.whl')
    file = tmp_path / 'w.yml'
    file.write_text('jobs: {}\n')
    run = subprocess.run([sys.executable, str(EDITOR), str(file), 'x.whl'], capture_output=True, text=True, timeout=30)
    assert run.returncode == 4 and file.read_text() == 'jobs: {}\n'
