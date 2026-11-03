"""The stall limit of `m5_render.py render`, on a fake mitsuba. No real renderer runs here."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parent.parent / 'spikes' / 'm5_render.py'
FAKE_MITSUBA = """
__version__ = 'fake-9'
HANG = {hang!r}
def _spin():
    while True:
        pass
def variants():
    return ['fake_variant']
def set_variant(name):
    if HANG == 'set_variant':
        _spin()
def cornell_box():
    return {{'small-box': 1, 'large-box': 2}}
def load_dict(scene):
    return scene
def render(scene, spp, seed):
    if HANG == 'render':
        _spin()
    import numpy
    return numpy.ones((2, 2, 3))
"""


def _run(tmp_path: Path, hang: str | None, stall: str) -> subprocess.CompletedProcess[str]:
    fake = tmp_path / 'fake'
    fake.mkdir()
    (fake / 'mitsuba.py').write_text(FAKE_MITSUBA.format(hang=hang))
    (fake / 'drjit.py').write_text("__version__ = 'fake-9'\n")
    env = {**os.environ, 'PYTHONPATH': str(fake)}
    return subprocess.run([sys.executable, str(SCRIPT), 'render', '--variant', 'fake_variant', '--seeds', '1', '--out', str(tmp_path / 'out'),
                           '--stall-seconds', stall], capture_output=True, text=True, timeout=120, env=env, check=False)


def test_a_run_with_no_stall_finishes_and_writes_its_summary(tmp_path):
    done = _run(tmp_path, None, '60')
    assert done.returncode == 0, done.stderr
    summary = json.loads((tmp_path / 'out' / 'summary.json').read_text())
    assert summary['mitsuba'] == 'fake-9'  # The fake answered, so no real renderer ran.
    assert summary['status'] == 'rendered'
    assert '"stage": "render cornell_spheres seed 1"' in done.stdout


def test_a_stall_in_loading_ends_the_run_and_names_the_stage(tmp_path):
    done = _run(tmp_path, 'set_variant', '1')
    assert done.returncode != 0
    assert done.stdout.strip().splitlines()[-1] == '{"stage": "load mitsuba"}'
    assert 'set_variant' in done.stderr and 'Timeout' in done.stderr
    assert not (tmp_path / 'out' / 'summary.json').exists()  # A stall is not recorded as a result.


def test_a_stall_in_the_first_render_names_that_stage(tmp_path):
    done = _run(tmp_path, 'render', '1')
    assert done.returncode != 0
    assert done.stdout.strip().splitlines()[-1] == '{"stage": "first render cornell_box"}'
