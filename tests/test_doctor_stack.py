"""`cumlx doctor --stack`. Each test stubs the detection, and proves from the stub's own log that the stub answered."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
from types import SimpleNamespace

import pytest

from cumlx.cli.main import main
from cumlx.errors import ExitCode
from cumlx.verify import env

STUB_VERSION = '9.9.9-stub'  # No real package has this version, so seeing it proves that the stub answered.
LAYERS = {'mlx', 'mlx_lm', 'torch_mps', 'metal_device', 'genesis', 'mitsuba_metal', 'warp_metal', 'translator'}
MODULES = {'mlx', 'mlx_lm', 'torch', 'genesis', 'mitsuba', 'warp', 'woof'}
DISTS = {'mlx', 'mlx-lm', 'torch', 'genesis-world', 'mitsuba', 'warp-metal', 'metal-woof'}


def run(capsys, *argv):
    code = main([*argv, '--format', 'json'])
    doc = json.loads(capsys.readouterr().out)  # stdout must be exactly one JSON document
    assert doc['exit_code'] == code
    return code, doc


class Detection:
    """Stands in for find_spec, the metadata lookup and the import, and logs each call."""

    def __init__(self, monkeypatch, modules=frozenset(), dists=frozenset(), imports=None):
        self.modules, self.dists, self.imports = set(modules), set(dists), dict(imports or {})
        self.spec_calls: list[str] = []
        self.version_calls: list[str] = []
        self.import_calls: list[str] = []
        self._real_find_spec = env.util.find_spec
        monkeypatch.setattr(env.util, 'find_spec', self._find_spec)
        monkeypatch.setattr(env, '_version', self._version)
        monkeypatch.setattr(env, '_import', self._import)

    def _find_spec(self, name, *args):
        if name in MODULES:
            self.spec_calls.append(name)
            return object() if name in self.modules else None
        return self._real_find_spec(name, *args)

    def _version(self, dist):
        self.version_calls.append(dist)
        return STUB_VERSION if dist in self.dists else None

    def _import(self, name):
        self.import_calls.append(name)
        if name not in self.imports:
            raise AssertionError(f'unexpected import of {name}')
        value = self.imports[name]
        if isinstance(value, Exception):
            raise value
        return value


def fake_mlx_core(available=True):
    return SimpleNamespace(metal=SimpleNamespace(is_available=lambda: available, device_info=lambda: {'device_name': 'stub gpu'}))


def fake_torch(mps=True):
    return SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)))


def fake_mitsuba(variants=('scalar_rgb', 'metal_ad_rgb')):
    return SimpleNamespace(variants=lambda: list(variants))


def everything(monkeypatch, modules=MODULES, dists=DISTS, imports=None):
    return Detection(monkeypatch, modules, dists,
                     {'mlx.core': fake_mlx_core(), 'torch': fake_torch(), 'mitsuba': fake_mitsuba(), **(imports or {})})


def test_nothing_installed_is_missing_with_a_hint_and_exit_3(capsys, monkeypatch):
    stub = Detection(monkeypatch)
    code, doc = run(capsys, 'doctor', '--stack')
    # The stub answered: every module was asked of it, and nothing was imported.
    assert set(stub.spec_calls) >= MODULES and stub.import_calls == []
    assert code == ExitCode.ENVIRONMENT and doc['ok'] is False and doc['status'] == 'error'
    stack = doc['data']['stack']
    assert set(stack['layers']) == LAYERS and stack['ready'] == [] and set(stack['not_ready']) == LAYERS
    for name, layer in stack['layers'].items():
        assert layer['ok'] is False and layer['install_hint'], name
        assert layer['status'] == ('unknown' if name == 'metal_device' else 'missing'), name
    assert {e['details']['layer'] for e in doc['errors']} == LAYERS
    assert {e['code'] for e in doc['errors']} == {'E_DEP_MISSING'} and all(e['hint'] for e in doc['errors'])
    assert 'cumlx-lab[render]' in stack['layers']['mitsuba_metal']['install_hint']
    assert 'genesis-world' in stack['layers']['genesis']['install_hint']
    assert 'warp-metal' in stack['layers']['warp_metal']['install_hint']
    assert 'metal-woof' in stack['layers']['translator']['install_hint']


def test_everything_present_is_exit_0(capsys, monkeypatch):
    stub = everything(monkeypatch)
    code, doc = run(capsys, 'doctor', '--stack')
    assert set(stub.version_calls) >= DISTS and {'mlx.core', 'torch', 'mitsuba'} <= set(stub.import_calls)
    assert code == 0 and doc['ok'] is True and doc['errors'] == []
    stack = doc['data']['stack']
    assert stack['not_ready'] == [] and set(stack['ready']) == LAYERS
    for name, layer in stack['layers'].items():
        assert layer['status'] == 'ok' and layer['install_hint'] is None, name
    assert stack['layers']['genesis']['version'] == STUB_VERSION  # This came from the stub, not from this machine.
    assert 'stub gpu' in stack['layers']['metal_device']['detail']
    assert '8 of 8 layers ready' in doc['summary']


@pytest.mark.parametrize('layer,status,overrides', [
    ('warp_metal', 'missing', {'dists': DISTS - {'warp-metal'}}),  # `warp` is here, from warp-lang: no Metal backend.
    ('genesis', 'missing', {'dists': DISTS - {'genesis-world'}}),  # A module of that name from another package.
    ('translator', 'missing', {'modules': MODULES - {'woof'}}),
    ('torch_mps', 'unavailable', {'imports': {'torch': fake_torch(mps=False)}}),
    ('torch_mps', 'broken', {'imports': {'torch': ImportError('stub')}}),
    ('mitsuba_metal', 'unavailable', {'imports': {'mitsuba': fake_mitsuba(('scalar_rgb', 'llvm_ad_rgb'))}}),
    ('mitsuba_metal', 'broken', {'imports': {'mitsuba': OSError('stub')}}),
])
def test_one_wrong_layer_fails_and_names_that_layer(capsys, monkeypatch, layer, status, overrides):
    everything(monkeypatch, **overrides)
    code, doc = run(capsys, 'doctor', '--stack')
    stack = doc['data']['stack']
    assert code == ExitCode.ENVIRONMENT and stack['layers'][layer]['status'] == status and not stack['layers'][layer]['ok']
    assert layer in stack['not_ready'] and stack['layers'][layer]['install_hint']
    assert {e['details']['layer'] for e in doc['errors']} == set(stack['not_ready'])
    assert set(stack['ready']) == LAYERS - set(stack['not_ready'])


def test_warp_lang_alone_says_where_the_module_comes_from(monkeypatch):
    everything(monkeypatch, dists=(DISTS - {'warp-metal'}) | {'warp-lang'})
    layer = env.inspect_stack()['layers']['warp_metal']
    assert layer['status'] == 'missing' and 'warp-lang' in layer['detail']


def test_metal_device_asks_torch_when_mlx_is_not_here(monkeypatch):
    stub = everything(monkeypatch, modules=MODULES - {'mlx'})
    assert env.inspect_stack()['layers']['metal_device']['status'] == 'ok'
    assert 'mlx.core' not in stub.import_calls


def test_metal_device_is_not_a_pass_when_both_say_no(monkeypatch):
    everything(monkeypatch, imports={'mlx.core': fake_mlx_core(False), 'torch': fake_torch(False)})
    layer = env.inspect_stack()['layers']['metal_device']
    assert layer['status'] == 'unavailable' and layer['ok'] is False


def test_it_does_not_install_use_the_network_or_look_for_nvcc(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError('the stack check must not do this')
    stub = everything(monkeypatch)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(shutil, 'which', forbidden)
    assert env.inspect_stack()['not_ready'] == [] and stub.import_calls  # It ran, under the guards.
    assert any('nvcc' in line for line in env.STACK_NOT_CHECKED)


def test_a_package_that_prints_on_import_does_not_break_stdout(capsys, monkeypatch):
    class Noisy:
        @property
        def backends(self):
            print('a banner on stdout')
            return fake_torch().backends
    everything(monkeypatch, imports={'torch': Noisy()})
    code, doc = run(capsys, 'doctor', '--stack')  # run() parses stdout as one JSON document
    assert code == 0 and doc['data']['stack']['layers']['torch_mps']['ok'] is True


def test_without_the_flag_nothing_changes(capsys, monkeypatch):
    stub = Detection(monkeypatch)
    code, doc = run(capsys, 'doctor')
    assert 'stack' not in doc['data'] and stub.import_calls == [] and 'genesis' not in stub.spec_calls
    assert code == 0


def test_stack_and_require_are_both_enforced(capsys, monkeypatch):
    everything(monkeypatch, modules=MODULES - {'genesis'})
    code, doc = run(capsys, 'doctor', '--stack', '--require', 'scan')
    assert code == ExitCode.ENVIRONMENT and [e['details']['layer'] for e in doc['errors']] == ['genesis']
    code, _ = run(capsys, 'doctor', '--stack', '--require', 'nonsense')
    assert code == ExitCode.USAGE


def test_describe_and_mcp_show_the_flag(capsys):
    code, doc = run(capsys, 'describe', '--command', 'doctor')
    assert code == 0 and '--stack' in json.dumps(doc['data'])
    from cumlx.mcp_server import _schema
    arguments = next(iter(doc['data']['commands'].values()))['arguments']
    schema = _schema(arguments, allow_code=False)[0]
    assert schema['properties']['stack']['type'] == 'boolean'


def test_on_this_machine_the_exit_code_agrees_with_the_layers(capsys):
    code, doc = run(capsys, 'doctor', '--stack')
    stack = doc['data']['stack']
    assert code == (ExitCode.ENVIRONMENT if stack['not_ready'] else 0)
    assert set(stack['ready']) | set(stack['not_ready']) == LAYERS
    assert not set(stack['ready']) & set(stack['not_ready'])
