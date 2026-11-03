"""spikes/t4_action_example.sh: the refusals that come before any build or network call."""
import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parent.parent / 'spikes' / 't4_action_example.sh'


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
