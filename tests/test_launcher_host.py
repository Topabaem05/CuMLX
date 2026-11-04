"""The GPU machine is the user's setting. A launcher with none must stop before it contacts anything. Only fake programs run here."""
from __future__ import annotations

import os
from pathlib import Path
import re
import stat
import subprocess

import pytest

REPO = Path(__file__).resolve().parent.parent
REMOTE = ['spikes/m2_golden_v100.sh', 'spikes/m5_render_v100.sh']


def _sandbox(tmp_path: Path) -> tuple[dict, Path]:
    """A home with no settings, and a fake ssh that writes down each call. The check below proves that the fake is what runs."""
    home = tmp_path / 'home'
    (home / '.local/bin').mkdir(parents=True)
    log = tmp_path / 'ssh.log'
    ssh = home / '.local/bin/ssh'
    ssh.write_text(f'#!/bin/sh\necho "FAKE-SSH $*" >> "{log}"\necho "fake output"\n')
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k not in ('CUMLX_GPU_HOST', 'XDG_CONFIG_HOME')}
    env.update(HOME=str(home), PATH=f'{ssh.parent}{os.pathsep}{env["PATH"]}', CUMLX_M3_ROOT=str(tmp_path / 'root'))
    found = subprocess.run(['sh', '-c', 'command -v ssh'], env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert found == str(ssh), f'the fake is not what runs: {found}'
    return env, log


def _run(script: str, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(['sh', str(REPO / script), *args], env=env, capture_output=True, text=True, timeout=30, check=False)


@pytest.mark.parametrize('script, args', [(REMOTE[0], ()), (REMOTE[0], ('status',)), (REMOTE[1], ()), (REMOTE[1], ('fetch',)), ('spikes/m3_kernels.sh', ('capture',))])
def test_with_no_host_it_stops_before_it_contacts_anything(tmp_path, script, args):
    env, log = _sandbox(tmp_path)
    done = _run(script, env, *args)
    assert done.returncode == 3, done.stderr
    assert 'CUMLX_GPU_HOST' in done.stderr and 'gpu-host' in done.stderr
    assert not log.exists()
    assert not (tmp_path / 'root/run.pid').exists()


@pytest.mark.parametrize('script', REMOTE)
def test_the_file_gives_the_host_and_the_variable_comes_first(tmp_path, script):
    env, log = _sandbox(tmp_path)
    config = Path(env['HOME']) / '.config/cumlx'
    config.mkdir(parents=True)
    (config / 'gpu-host').write_text('host-from-file\n')
    assert _run(script, env, 'status').returncode == 0
    assert 'FAKE-SSH host-from-file ' in log.read_text()
    log.unlink()
    assert _run(script, {**env, 'CUMLX_GPU_HOST': 'host-from-variable'}, 'status').returncode == 0
    assert 'FAKE-SSH host-from-variable ' in log.read_text()


def test_xdg_config_home_is_respected(tmp_path):
    env, log = _sandbox(tmp_path)
    config = tmp_path / 'xdg/cumlx'
    config.mkdir(parents=True)
    (config / 'gpu-host').write_text('host-from-xdg\n')
    assert _run(REMOTE[0], {**env, 'XDG_CONFIG_HOME': str(tmp_path / 'xdg')}, 'status').returncode == 0
    assert 'FAKE-SSH host-from-xdg ' in log.read_text()


def test_no_launcher_names_one_persons_machine():
    default_host = re.compile(r'CUMLX_GPU_HOST:-[^}]')
    named = [path.name for folder in ('spikes', 'scripts') for path in (REPO / folder).glob('*.sh') if default_host.search(path.read_text())]
    assert not named
