"""The memory reading and limit, on other page sizes, other sizes of RAM and other systems."""
from __future__ import annotations

import platform
import subprocess

import pytest

from cumlx import sysmem

VM_STAT = """Mach Virtual Memory Statistics: (page size of {page} bytes)
Pages free:                               100.
Pages active:                            1000.
Pages inactive:                           500.
Pages wired down:                         200.
Pages occupied by compressor:              50.
"""
MEMINFO = 'MemTotal:       16000000 kB\nMemFree:         1000000 kB\nMemAvailable:    6000000 kB\n'


@pytest.mark.parametrize('page', [4096, 16384])
def test_vm_stat_uses_the_page_size_that_it_states(page):
    assert sysmem.parse_vm_stat(VM_STAT.format(page=page)) == pytest.approx(1250 * page / 1e9)


def test_vm_stat_without_a_page_size_or_a_field_is_not_read():
    assert sysmem.parse_vm_stat('Pages active: 1000.\nPages wired down: 200.\nPages occupied by compressor: 50.\n') is None
    assert sysmem.parse_vm_stat(VM_STAT.format(page=4096).replace('Pages wired down', 'Pages wired')) is None
    assert sysmem.parse_vm_stat('') is None


def test_meminfo_is_total_less_available():
    assert sysmem.parse_meminfo(MEMINFO) == pytest.approx(10.0)
    assert sysmem.parse_meminfo('MemTotal: 16000000 kB\nMemFree: 1000000 kB\n') is None


def test_each_system_reads_its_own_source(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, VM_STAT.format(page=16384), ''))
    assert sysmem.used_gb() == pytest.approx(1250 * 16384 / 1e9)

    def no_tool(*args, **kwargs):
        raise FileNotFoundError('vm_stat')
    monkeypatch.setattr(subprocess, 'run', no_tool)
    assert sysmem.used_gb() is None  # A missing tool is an unknown reading, not a low one.

    monkeypatch.setattr(platform, 'system', lambda: 'FreeBSD')
    assert sysmem.used_gb() is None


@pytest.mark.skipif(platform.system() != 'Linux', reason='reads the real /proc/meminfo')
def test_the_real_meminfo_reads_on_linux():
    used, total = sysmem.used_gb(), sysmem.total_gb()
    assert used is not None and total is not None and 0 < used <= total


@pytest.mark.skipif(platform.system() != 'Darwin', reason='runs the real vm_stat')
def test_the_real_vm_stat_reads_on_a_mac():
    used, total = sysmem.used_gb(), sysmem.total_gb()
    assert used is not None and total is not None and 0 < used <= total


@pytest.mark.parametrize('gib, limit', [(8, 6.27), (16, 12.54), (48, 37.62), (128, 100.33)])
def test_the_default_limit_follows_ram_and_keeps_this_macs_rule(monkeypatch, gib, limit):
    monkeypatch.delenv(sysmem.LIMIT_VARIABLE, raising=False)
    monkeypatch.setattr(sysmem, 'total_gb', lambda: gib * 2**30 / 1e9)
    assert sysmem.limit_gb() == pytest.approx(limit, abs=0.01)
    if gib == 48:
        assert sysmem.limit_gb() <= 38  # The rule on the machine that this was written on.


def test_the_variable_sets_the_limit_and_a_bad_value_is_an_error(monkeypatch):
    monkeypatch.setenv(sysmem.LIMIT_VARIABLE, '12.5')
    assert sysmem.limit_gb() == 12.5
    for bad in ('many', '0', '-3'):
        monkeypatch.setenv(sysmem.LIMIT_VARIABLE, bad)
        with pytest.raises(ValueError, match=sysmem.LIMIT_VARIABLE):
            sysmem.limit_gb()


def test_over_limit_names_the_reason_and_an_unknown_reading_is_not_one(monkeypatch):
    monkeypatch.setattr(sysmem, 'limit_gb', lambda: 10.0)
    monkeypatch.setattr(sysmem, 'used_gb', lambda: 10.5)
    assert '10.5 GB' in sysmem.over_limit()
    monkeypatch.setattr(sysmem, 'used_gb', lambda: 9.5)
    assert sysmem.over_limit() is None
    monkeypatch.setattr(sysmem, 'used_gb', lambda: None)
    assert sysmem.over_limit() is None
