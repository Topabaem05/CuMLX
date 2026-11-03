"""The CLI with no MLX and no torch, and the checker that CI runs on a fresh install of the wheel."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parent.parent
HEAVY = ('mlx', 'mlx.core', 'mlx_lm', 'torch', 'transformers', 'mitsuba', 'genesis', 'warp', 'woof')
BLOCKED = f"import sys\nfor m in {HEAVY!r}: sys.modules[m] = None\nfrom cumlx.cli import entrypoint\nsys.exit(entrypoint())\n"


def _checker():
    spec = importlib.util.spec_from_file_location('compat_check', REPO / 'scripts/compat_check.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['compat_check'] = module
    spec.loader.exec_module(module)
    return module


def _blocked_cli(*argv: str) -> tuple[int, dict]:
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'src')}
    done = subprocess.run([sys.executable, '-c', BLOCKED, *argv, '--format', 'json'], capture_output=True, text=True, timeout=120, env=env, check=False)
    return done.returncode, json.loads(done.stdout)


def test_with_nothing_heavy_the_cli_still_answers():
    for argv in (['describe'], ['doctor'], ['selftest', '--core-only']):
        code, doc = _blocked_cli(*argv)
        assert code == doc['exit_code'] == 0, doc['summary']
        assert doc['schema'] == 'cumlx.result.v1'


@pytest.mark.parametrize('need, hint', [('capture', 'capture'), ('diff', 'diff'), ('conform', 'diff'), ('run', 'capture')])
def test_a_missing_extra_is_exit_3_with_its_install_line_and_never_a_pass(need, hint):
    code, doc = _blocked_cli('doctor', '--require', need)
    assert code == doc['exit_code'] == 3 and doc['ok'] is False
    errors = doc['errors']
    assert errors[0]['code'] == 'E_DEP_MISSING' and 'pip install' in errors[0]['hint'] and hint in errors[0]['hint']


def test_the_stack_check_with_nothing_heavy_is_not_a_pass():
    code, doc = _blocked_cli('doctor', '--stack')
    assert code == 3 and doc['ok'] is False
    layers = doc['data'].get('stack', doc['data'])['layers']
    assert not any(layer['ok'] for layer in layers.values())
    assert all(layer['install_hint'] for layer in layers.values())


# ---------------------------------------------------------------- the checker itself must be able to fail
def _fake_python(tmp_path: Path, body: str) -> str:
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/sh\n' + body)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return str(fake)


GOOD = {'schema': 'cumlx.result.v1', 'exit_code': 3, 'errors': [{'code': 'E_DEP_MISSING', 'hint': 'python -m pip install x'}]}


def test_the_checker_accepts_a_right_answer(tmp_path):
    fake = _fake_python(tmp_path, f"echo '{json.dumps(GOOD)}'\nexit 3\n")
    assert _checker().check('t', fake, ['x'], str(tmp_path), 3, 'E_DEP_MISSING', 'pip install')['ok']


@pytest.mark.parametrize('stdout, code, reason', [
    ('not json', 3, 'not one JSON'),
    (json.dumps({**GOOD, 'schema': 'other'}), 3, 'schema'),
    (json.dumps({**GOOD, 'exit_code': 0}), 3, 'envelope'),
    (json.dumps({**GOOD, 'exit_code': 0, 'errors': []}), 0, 'expected E_DEP_MISSING'),
    (json.dumps({**GOOD, 'errors': [{'code': 'E_DEP_MISSING', 'hint': None}]}), 3, 'hint'),
])
def test_the_checker_catches_a_wrong_answer(tmp_path, stdout, code, reason):
    fake = _fake_python(tmp_path, f"echo '{stdout}'\nexit {code}\n")
    result = _checker().check('t', fake, ['x'], str(tmp_path), 3, 'E_DEP_MISSING', 'pip install')
    assert not result['ok'] and any(reason in p for p in result['problems'])


def test_the_checker_refuses_an_environment_that_is_not_a_base_install(tmp_path):
    fake = _fake_python(tmp_path, 'echo mlx,torch\n')
    done = subprocess.run([sys.executable, str(REPO / 'scripts/compat_check.py'), '--python', fake], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 3 and 'not a base install' in done.stderr
