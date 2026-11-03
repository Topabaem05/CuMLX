"""spikes/m5_render_v100.sh on a fake ssh. Nothing reaches a real machine, and no third-party code runs.

The first check of each case proves that the shell finds the fake and not the real ssh, and every run names a host
that does not exist, so a real ssh that slipped through would fail too.
"""
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / 'spikes' / 'm5_render_v100.sh'
HOST = 'no-such-host.invalid'


def _fake_ssh(folder: Path, *, busy: str = '', pid_alive: bool = True, has_results: bool = False) -> Path:
    log = folder / 'ssh.log'
    body = f"""#!/bin/sh
printf '%s\\n' "FAKE-SSH $*" >> '{log}'
case "$*" in
    *nvidia-smi*) printf '%s' '{busy}' ;;
    *"tar -x"*) cat > /dev/null ;;
    *"test -f"*) {'exit 0' if has_results else 'exit 1'} ;;
    *"tar -c"*) tar -c -C '{folder}' ssh.log ;;
    *"kill -0"*) {'exit 0' if pid_alive else 'exit 1'} ;;
    *"cat cumlx-m5-render/run.pid"*) echo 4242 ;;
    *"cat cumlx-m5-render/run.log"*) echo boom ;;
esac
exit 0
"""
    ssh = folder / 'ssh'
    ssh.write_text(body)
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    return log


def _env(folder: Path, **extra) -> dict:
    return {**os.environ, 'PATH': f'{folder}{os.pathsep}{os.environ["PATH"]}', 'CUMLX_GPU_HOST': HOST, 'CUMLX_M5_GOLDEN': str(folder / 'golden'), **extra}


def _run(folder: Path, *args) -> subprocess.CompletedProcess:
    env = _env(folder)
    found = subprocess.run(['sh', '-c', 'command -v ssh'], env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert found == str(folder / 'ssh'), f'the fake is not what runs: {found}'
    return subprocess.run(['sh', str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60, check=False)


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_start_on_an_idle_gpu_copies_only_its_own_files_and_pins_the_install(tmp_path):
    log = _fake_ssh(tmp_path)
    done = _run(tmp_path)
    assert done.returncode == 0, done.stderr
    assert 'Started on' in done.stdout and '4242' in done.stdout
    calls = log.read_text()
    assert calls.count('FAKE-SSH') >= 4  # The fake's own lines prove which program ran.
    assert 'mitsuba==3.8.0' in calls and '--variant cuda_ad_rgb --seeds 1,2,3 --out cuda;' in calls
    assert 'pip install -q --python venv/bin/python mitsuba==3.8.0 numpy &&' in calls
    assert '--variant scalar_rgb --seeds 21,22,23 --out scalar;' in calls  # The control of the version difference runs even if CUDA fails.
    remote = re.search(r"bash -lc '(.*)'$", calls, re.M)
    assert remote is not None
    syntax = subprocess.run(['bash', '-n', '-c', remote.group(1)], capture_output=True, text=True, check=False)  # -n reads the command and runs nothing.
    assert syntax.returncode == 0, syntax.stderr
    source = SCRIPT.read_text()
    assert 'archive HEAD spikes/m5_render.py src/cumlx |' in source


def test_it_refuses_a_busy_gpu_and_starts_nothing(tmp_path):
    log = _fake_ssh(tmp_path, busy='1651910')
    done = _run(tmp_path)
    assert done.returncode == 3
    assert '1651910' in done.stderr and 'Nothing was started' in done.stderr
    calls = log.read_text()
    assert calls.count('FAKE-SSH') == 1 and 'nvidia-smi' in calls
    assert 'mitsuba' not in calls and 'tar -x' not in calls


def test_a_run_that_stops_at_once_is_an_error(tmp_path):
    _fake_ssh(tmp_path, pid_alive=False)
    done = _run(tmp_path)
    assert done.returncode == 1
    assert 'stopped at once' in done.stderr and 'boom' in done.stderr


def test_fetch_without_results_fails_and_with_results_copies(tmp_path):
    _fake_ssh(tmp_path, has_results=False)
    done = _run(tmp_path, 'fetch')
    assert done.returncode == 1 and 'no results yet' in done.stderr
    assert not (tmp_path / 'golden').exists()

    _fake_ssh(tmp_path, has_results=True)
    done = _run(tmp_path, 'fetch')
    assert done.returncode == 0, done.stderr
    assert (tmp_path / 'golden' / 'ssh.log').is_file()


def test_status_and_a_wrong_argument(tmp_path):
    log = _fake_ssh(tmp_path)
    assert _run(tmp_path, 'status').returncode == 0
    assert 'run.pid' in log.read_text()
    assert _run(tmp_path, 'nonsense').returncode == 2
