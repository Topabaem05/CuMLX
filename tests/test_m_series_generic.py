"""Nothing assumes one chip or one size of RAM. The numbers from the M5 Pro stay in comments and records, and never in a branch."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from cumlx import sysmem
from cumlx.verify import env

REPO = Path(__file__).resolve().parent.parent
# A chip's name or a fixed size of RAM in code that decides something. Comments and plain strings for a reader do not match.
BRANCH = re.compile(r"""(if|elif|while|assert|return)\b[^#\n]*(['"]([^'"\n]*\b(M[1-9]( Pro| Max| Ultra)?|apple[0-9]+)\b[^'"\n]*)['"]\s*(in|==|!=)|(==|!=|\bin\b)\s*['"]([^'"\n]*\b(M[1-9]( Pro| Max| Ultra)?|apple[0-9]+)\b)[^'"\n]*['"])""")
FIXED_RAM = re.compile(r"(?m)^\s*\w*(MAX|LIMIT)\w*GB\w*\s*=\s*[0-9.]+\s*(#.*)?$")


def _sources():
    for folder in ('src/cumlx', 'scripts', 'spikes'):
        yield from sorted((REPO / folder).rglob('*.py'))


def test_no_code_branches_on_a_chip_name():
    found = [f'{path.name}: {m.group(0)[:80]}' for path in _sources() for m in BRANCH.finditer(path.read_text())]
    assert not found


def test_no_code_has_a_fixed_memory_limit():
    found = [f'{path.name}: {m.group(0).strip()[:80]}' for path in _sources() for m in FIXED_RAM.finditer(path.read_text())]
    assert not found


@pytest.mark.parametrize('line', ["if 'M5' in name:", 'if name == "Apple M5 Pro":', "assert device == 'apple9'", "elif 'Apple M3 Max' == chip:"])
def test_the_rule_catches_a_branch_on_a_chip(line):
    assert BRANCH.search(line)


@pytest.mark.parametrize('line', ["# measured on the M5 Pro", "note = 'measured on one M5 Pro'", "if measured > REDUCED_PRECISION_NOISE:"])
def test_the_rule_leaves_a_note_alone(line):
    assert not BRANCH.search(line)


def test_the_rule_catches_a_fixed_limit():
    assert FIXED_RAM.search('MAX_USED_GB = 38.0  # for this Mac\n')
    assert not FIXED_RAM.search('limit_gb = sysmem.limit_gb()\n')


@pytest.mark.parametrize('name', ['Apple M1', 'Apple M2 Max', 'Apple M3 Ultra', 'Apple M5 Pro', 'Apple M9'])
def test_every_chip_name_reads_the_same(monkeypatch, name):
    fake_mlx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True), device_info=lambda: {'device_name': name})
    monkeypatch.setattr(env.util, 'find_spec', lambda module: object() if module == 'mlx' else None)
    monkeypatch.setattr(env, '_import', lambda module: fake_mlx)
    layer = env._metal_device()
    assert layer['ok'] and name in layer['detail']


def test_no_metal_is_not_a_pass_on_any_machine(monkeypatch):
    fake_mlx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(env.util, 'find_spec', lambda module: object() if module == 'mlx' else None)
    monkeypatch.setattr(env, '_import', lambda module: fake_mlx)
    layer = env._metal_device()
    assert not layer['ok'] and layer['status'] == 'unavailable'


@pytest.mark.parametrize('gib', [8, 16, 24, 48, 512])
def test_the_probe_reports_memory_on_any_size(monkeypatch, gib):
    monkeypatch.delenv(sysmem.LIMIT_VARIABLE, raising=False)
    monkeypatch.setattr(sysmem, 'total_gb', lambda: gib * 2**30 / 1e9)
    monkeypatch.setattr(sysmem, 'used_gb', lambda: 3.0)
    memory = env._memory()
    assert memory['used_gb'] == 3.0
    assert memory['limit_gb'] == pytest.approx(gib * 2**30 / 1e9 * sysmem.DEFAULT_LIMIT_FRACTION)


def test_a_bad_limit_is_reported_and_does_not_stop_the_doctor(monkeypatch):
    monkeypatch.setenv(sysmem.LIMIT_VARIABLE, 'many')
    memory = env._memory()
    assert memory['limit_gb'] is None and sysmem.LIMIT_VARIABLE in memory['limit_error']


@pytest.mark.skipif(importlib.util.find_spec('mlx') is None, reason='needs MLX: not checked here')
def test_the_real_device_is_read_without_assuming_its_name():
    layer = env._metal_device()
    assert layer['status'] in ('ok', 'unavailable')  # A Linux or Intel machine, or a runner without Metal, gives unavailable.
