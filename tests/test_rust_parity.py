"""The Rust `cumlx-rs` against the Python `cumlx`, on the same inputs.

Every case runs both command lines as subprocesses and requires the same envelope: exit code, status,
error codes and hints, verdicts, first failure or divergence, every key in the same order, and every float
within 1e-12 relative. Each injected fault must fail in both. A missing binary is a skip here and a failure
with ``--require-rust`` (the CI job passes it).

Build the binary first::

    cargo build --release --manifest-path rust/Cargo.toml
    python -m pytest -q tests/test_rust_parity.py --require-rust

Groups:

* NumPy only: the CLI contract, ``describe``, the Python-only refusals, and ``tracediff`` on synthetic traces
  with verdicts ok, diverged and incomplete, and every trace error that Python reports with a code.
* MLX and mlx-lm: ``weights`` on small checkpoints with an injected transpose, scale and sign fault, bf16 and
  f16 ports, tied embeddings and MoE expert stacks (the ``mlx`` fixture).
* torch and transformers too: ``tracediff`` on the selftest's golden traces and ``weights`` on the selftest's
  tiny Llama with fault F1 (the ``e2e`` fixture).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from rust_vectors import same  # noqa: E402

from cumlx import __version__  # noqa: E402
from cumlx.errors import ERROR_CODES  # noqa: E402
from cumlx.verify.safetensors_io import save_file  # noqa: E402
from cumlx.verify.trace import ModuleRecord, TensorRef, TraceWriter, prepare_output_dir  # noqa: E402

RUST = ROOT / 'rust'
TOKENS = [1, 5, 9, 23]
NUMBER = re.compile(r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?')


# ----------------------------------------------------------------------------- running both
@pytest.fixture(scope='session')
def rust_bin(request) -> Path:
    given = os.environ.get('CUMLX_RS_BIN')
    candidates = [Path(given)] if given else [RUST / 'target' / 'release' / 'cumlx-rs', RUST / 'target' / 'debug' / 'cumlx-rs']
    found = next((p for p in candidates if p.is_file()), None)
    if found is None:
        msg = 'cumlx-rs is not built: cargo build --release --manifest-path rust/Cargo.toml'
        if request.config.getoption('--require-rust'):
            pytest.fail(msg)
        pytest.skip(msg)
    if not given:  # a stale binary must not pass for the current source
        newest = max(p.stat().st_mtime for p in RUST.rglob('*.rs') if 'target' not in p.parts)
        assert found.stat().st_mtime >= newest, f'{found} is older than the Rust source; rebuild it'
    return found


@pytest.fixture(scope='session')
def mlx(request):
    missing = [m for m in ('mlx', 'mlx_lm') if importlib.util.find_spec(m) is None]
    if missing:
        msg = f'the weights parity tests need {missing}; pip install -e ".[diff]"'
        if request.config.getoption('--require-rust') or request.config.getoption('--require-e2e'):
            pytest.fail(msg)
        pytest.skip(msg)
    return True


def _env(**extra: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith('CUMLX_')}
    env.update({'PYTHONPATH': str(ROOT / 'src'), 'CUMLX_FORMAT': 'json', **extra})
    return env


def _run(cmd: list[str], cwd: Path | None = None, **env: str) -> tuple[int, dict]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=_env(**env), cwd=cwd, timeout=600)
    try:
        doc = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise AssertionError(f'{cmd[:3]} did not print one JSON document:\n{p.stdout}\n{p.stderr}') from None
    assert doc['exit_code'] == p.returncode, (doc, p.returncode)
    return p.returncode, doc


def py(*args: str, cwd: Path | None = None, **env: str) -> tuple[int, dict]:
    return _run([sys.executable, '-m', 'cumlx', *args], cwd=cwd, **env)


def rs(binary: Path, *args: str, cwd: Path | None = None, **env: str) -> tuple[int, dict]:
    return _run([str(binary), *args], cwd=cwd, **env)


def _norm(v):
    """The Rust binary names itself in hints; everything else must be equal."""
    if isinstance(v, str):
        return v.replace('cumlx-rs', 'cumlx')
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def same_text(a: str, b: str) -> bool:
    """Equal text, where each number may differ by 1e-12 relative (Python prints floats with repr)."""
    if NUMBER.sub('#', a) != NUMBER.sub('#', b):
        return False
    xs, ys = NUMBER.findall(a), NUMBER.findall(b)
    return all(x == y or math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=0.0) for x, y in zip(xs, ys))


def assert_parity(p: tuple[int, dict], r: tuple[int, dict], *, messages: bool = True) -> dict:
    """The two envelopes agree. Returns the Python document."""
    (pc, pd), (rc, rd) = p, r
    rd = _norm(rd)
    pd = _norm(pd)
    assert (pc, pd['status'], pd['ok']) == (rc, rd['status'], rd['ok']), (pd, rd)
    assert list(pd) == list(rd), (list(pd), list(rd))
    for key in ('schema', 'command', 'truncated', 'cumlx_version', 'warnings', 'artifacts', 'next_steps'):
        assert pd[key] == rd[key], (key, pd[key], rd[key])
    assert same_text(pd['summary'], rd['summary']) or not messages, (pd['summary'], rd['summary'])
    assert [e['code'] for e in pd['errors']] == [e['code'] for e in rd['errors']], (pd['errors'], rd['errors'])
    for pe, re_ in zip(pd['errors'], rd['errors']):
        assert pe.get('hint') == re_.get('hint'), (pe, re_)
        if messages:
            assert pe['message'] == re_['message'], (pe, re_)
            assert pe.get('details') == re_.get('details'), (pe, re_)
    diffs = same(pd['data'], rd['data'])
    assert not diffs, '\n'.join(diffs[:20])
    return pd


def both(binary: Path, *args: str, messages: bool = True, cwd: Path | None = None, **env: str) -> dict:
    return assert_parity(py(*args, cwd=cwd, **env), rs(binary, *args, cwd=cwd, **env), messages=messages)


# ----------------------------------------------------------------------------- contract and describe
def test_versions_and_error_tables_are_one(rust_bin):
    cargo = (RUST / 'Cargo.toml').read_text()
    assert re.search(r'^version = "([^"]+)"', cargo, re.M).group(1) == __version__
    _, doc = rs(rust_bin, 'describe')
    data = doc['data']
    assert data['tool'] == 'cumlx-rs' and data['version'] == __version__
    assert list(data['error_codes']) == list(ERROR_CODES)
    assert {k: (v['exit_code'], v['meaning']) for k, v in data['error_codes'].items()} == \
        {k: (int(v[0]), v[1]) for k, v in ERROR_CODES.items()}
    _, pydoc = py('describe')
    assert data['exit_codes'] == pydoc['data']['exit_codes']
    assert data['error_codes'] == pydoc['data']['error_codes']
    assert data['result_envelope'] == pydoc['data']['result_envelope']
    assert data['env'] == pydoc['data']['env']


def test_describe_lists_every_python_command_and_the_same_arguments(rust_bin):
    _, pydoc = py('describe')
    _, doc = rs(rust_bin, 'describe')
    pc, rc = pydoc['data']['commands'], doc['data']['commands']
    assert list(rc) == list(pc), 'cumlx-rs must list every Python command, in the same order'
    implemented = sorted(k for k, v in rc.items() if v['implemented'])
    assert implemented == ['describe', 'tracediff', 'weights']
    assert sorted(doc['data']['python_only']) == sorted(set(pc) - set(implemented))
    for name in implemented:
        strip = [{k: v for k, v in a.items() if k != 'help'} for a in pc[name]['arguments']]
        ours = [{k: v for k, v in a.items() if k != 'help'} for a in rc[name]['arguments']]
        assert ours == strip, name
    for name in doc['data']['python_only']:
        for key in ('summary', 'requires', 'stage', 'examples'):
            assert rc[name][key] == pc[name][key], (name, key)
        assert 'arguments' not in rc[name] and rc[name]['python_command'] == f'cumlx {name}'


@pytest.mark.parametrize('argv', [
    ['capture', 'Qwen/Qwen2.5-0.5B', '--prompt', 'Hello world', '-o', 'golden/'],
    ['diff', 'golden/', './port'], ['run', 'train.py', '--', '--lr', '1e-3'], ['kernel', 'capture', 'k/', '--cases', 'c.json', '-o', 'g/'],
    ['kernel', 'verify', 'foo.metal', 'g/'], ['kernel', 'port', 'foo.cu', '-o', 'foo.metal'], ['doctor'], ['selftest'],
    ['conform'], ['scan', '.'], ['recipe'], ['rundiff', 'a', 'b'], ['report', 'r.json'], ['port', 'x', '-o', 'y'], ['fixture', 'tiny-llama', '-o', 'x'],
])
def test_python_only_commands_exit_3_and_name_the_python_command(rust_bin, argv):
    code, doc = rs(rust_bin, *argv)
    assert code == 3 and doc['status'] == 'error' and doc['data'] == {}
    (err,) = doc['errors']
    assert err['code'] == 'E_PYTHON_ONLY' and ERROR_CODES['E_PYTHON_ONLY'][0] == 3
    assert err['hint'].endswith(' && cumlx ' + ' '.join(a if re.fullmatch(r'[\w@%+=:,./-]+', a) else f"'{a}'" for a in argv))
    assert err['details']['command'] == doc['command']


def test_usage_errors_are_the_same(rust_bin, tmp_path):
    for argv in (['tracediff', 'a'], ['tracediff', 'a', 'b', 'c'], ['tracediff', 'a', 'b', '--rel-l2', 'x'],
                 ['tracediff', 'a', 'b', '--format', 'xml'], ['weights', 'a'], ['describe', '--command', 'nope'],
                 ['nope'], ['kernel', 'nope'], ['tracediff', 'a', 'b', '--rel-l2', '-1'], ['tracediff', 'a', 'b', '--rel-l2', '0'],
                 ['tracediff', 'a', 'b', '--rel-l2', 'nan'], ['tracediff', 'a', 'b', '--max-items', 'many']):
        p, r = py(*argv), rs(rust_bin, *argv)
        assert (p[0], p[1]['errors'][0]['code']) == (r[0], r[1]['errors'][0]['code']) == (2, 'E_USAGE'), (argv, p, r)
        assert _norm(p[1]['errors'][0].get('hint')) == _norm(r[1]['errors'][0].get('hint')), argv
        assert _norm(p[1]['command']) == _norm(r[1]['command']), argv


def test_bare_call_and_text_format(rust_bin):
    code, doc = rs(rust_bin)
    assert code == 0 and doc['command'] == 'describe'
    out = subprocess.run([str(rust_bin), 'describe', '--command', 'tracediff', '--format', 'text'], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith('[OK] cumlx-rs describe: 1 command(s)')
    assert subprocess.run([str(rust_bin), '--version'], capture_output=True, text=True).stdout.strip() == f'cumlx-rs {__version__}'


# ----------------------------------------------------------------------------- synthetic traces
MODULES = [  # key, name, type, shape, phase
    ('model.embed_tokens#0', 'model.embed_tokens', 'Embedding', (1, 4, 8), 'prefill'),
    ('model.layers.0.self_attn#0', 'model.layers.0.self_attn', 'Attention', (1, 4, 8), 'prefill'),
    ('model.layers.0.mlp#0', 'model.layers.0.mlp', 'MLP', (1, 4, 8), 'prefill'),
    ('model.layers.0#0', 'model.layers.0', 'DecoderLayer', (1, 4, 8), 'prefill'),
    ('model.norm#0', 'model.norm', 'RMSNorm', (1, 4, 8), 'prefill'),
    ('lm_head#0', 'lm_head', 'Linear', (1, 4, 16), 'prefill'),
    ('#0', '', 'ForCausalLM', (1, 4, 16), 'prefill'),
    ('model.layers.0.self_attn#1', 'model.layers.0.self_attn', 'Attention', (1, 1, 8), 'decode'),
    ('lm_head#1', 'lm_head', 'Linear', (1, 1, 16), 'decode'),
]


def golden_outputs(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {key: rng.normal(size=shape).astype(np.float32) for key, _, _, shape, _ in MODULES}


def write_trace(path: Path, outputs: dict[str, np.ndarray], *, tokens=TOKENS, drop_output: tuple[str, ...] = (),
                extra_records: list[tuple[str, np.ndarray]] = (), samples: int = 1, source: dict | None = None) -> Path:
    out = prepare_output_dir(path, overwrite=False)
    w = TraceWriter(out, source=source or {'framework': 'synthetic', 'model': 'tiny', 'device': 'cpu'},
                    golden={'kind': 'reference', 'dtype': 'float32', 'store_dtype': 'float32'}, max_bytes=1 << 30)
    rows = [(k, n, t, outputs[k], ph) for k, n, t, _, ph in MODULES if k in outputs]
    rows += [(k, k.split('#')[0], 'Extra', a, 'prefill') for k, a in extra_records]
    for _ in range(samples):
        tensors: dict[str, np.ndarray] = {'token_ids': np.asarray([tokens], np.int64), 'decode_token_ids': np.asarray([[7]], np.int64)}
        records = []
        for i, (key, name, typ, arr, phase) in enumerate(rows):
            tname = f'r{i:05d}.out'
            tensors[tname] = arr
            ref = None if key in drop_output else TensorRef(tname, list(arr.shape), str(arr.dtype), 'tensor')
            records.append(ModuleRecord(key=key, name=name, call=int(key.split('#')[1]), order=i, type=typ, has_params=True,
                                        replayable=False, output=ref, phase=phase))
        w.add_sample(tensors, records, num_tokens=len(tokens), decode_steps=1)
    w.finish()
    return out


@pytest.fixture
def golden(tmp_path) -> Path:
    return write_trace(tmp_path / 'golden', golden_outputs())


def _variant(tmp_path: Path, name: str, change) -> Path:
    outs = golden_outputs()
    change(outs)
    return write_trace(tmp_path / name, outs)


def test_tracediff_identical_traces_pass(rust_bin, golden, tmp_path):
    other = write_trace(tmp_path / 'same', golden_outputs())
    doc = both(rust_bin, 'tracediff', str(golden), str(other))
    assert doc['data']['verdict'] == 'ok' and doc['data']['summary']['worst_rel_l2'] == 0.0
    p = subprocess.run([sys.executable, '-m', 'cumlx', 'tracediff', str(golden), str(other), '--format', 'text'],
                       capture_output=True, text=True, env=_env())
    r = subprocess.run([str(rust_bin), 'tracediff', str(golden), str(other), '--format', 'text'], capture_output=True, text=True)
    plines, rlines = p.stdout.splitlines(), r.stdout.replace('[OK] cumlx-rs ', '[OK] cumlx ', 1).splitlines()
    assert len(plines) == len(rlines) and plines[0] == rlines[0] and plines[-1] == rlines[-1] == '  exit_code=0'
    assert [x.split(':')[0] for x in plines] == [x.split(':')[0] for x in rlines]
    assert plines[3] == rlines[3] == '  threshold_rel_l2: 0.0001'


def test_tracediff_noise_below_the_threshold_passes_with_the_same_numbers(rust_bin, golden, tmp_path):
    def noise(o):
        rng = np.random.default_rng(9)
        for k in o:
            o[k] = (o[k] * (1 + 1e-6 * rng.normal(size=o[k].shape))).astype(np.float32)
    doc = both(rust_bin, 'tracediff', str(golden), str(_variant(tmp_path, 'noise', noise)))
    assert doc['data']['verdict'] == 'ok' and 0 < doc['data']['summary']['worst_rel_l2'] < 1e-4


@pytest.mark.parametrize('key', ['model.layers.0.mlp#0', 'lm_head#1', '#0'])
def test_tracediff_moved_module_diverges_in_both(rust_bin, golden, tmp_path, key):
    def move(o):
        o[key] = (o[key] * np.float32(1.01)).astype(np.float32)
    doc = both(rust_bin, 'tracediff', str(golden), str(_variant(tmp_path, 'moved', move)))
    data = doc['data']
    assert data['verdict'] == 'diverged' and data['summary']['above_threshold'] == 1
    assert data['worst']['items'][0]['key'] == key and data['worst']['items'][0]['reason'] == 'rel_l2_above_threshold'
    loose = both(rust_bin, 'tracediff', str(golden), str(tmp_path / 'moved'), '--rel-l2', '0.02')
    assert loose['data']['verdict'] == 'ok'
    assert both(rust_bin, 'tracediff', str(golden), str(tmp_path / 'moved'), '--rel-l2', 'inf')['data']['verdict'] == 'ok'


def test_tracediff_special_values_shapes_and_zero_goldens(rust_bin, golden, tmp_path):
    def change(o):
        o['model.embed_tokens#0'] = o['model.embed_tokens#0'].reshape(4, 8)  # same size: reshaped, passes
        o['model.layers.0.self_attn#0'] = o['model.layers.0.self_attn#0'][:, :3]  # other size: shape_mismatch
        o['model.layers.0.mlp#0'][0, 0, 0] = np.nan  # in the first trace: nonfinite_output
        o['model.norm#0'] = np.zeros_like(o['model.norm#0'])
        o['lm_head#0'] = o['lm_head#0'].astype(np.float16)
    first = _variant(tmp_path, 'special', change)

    def golden_side(o):
        o['model.layers.0#0'][0, 1, 2] = np.inf  # in the second trace: nonfinite_golden
        o['model.norm#0'] = np.zeros_like(o['model.norm#0'])
    second = _variant(tmp_path, 'special-golden', golden_side)
    doc = both(rust_bin, 'tracediff', str(first), str(second), '--max-items', '-1')
    rows = {x['key']: x for x in doc['data']['worst']['items']}
    assert doc['data']['verdict'] == 'diverged'
    assert rows['model.embed_tokens#0']['reshaped'] is True and rows['model.norm#0']['basis'] == 'zero_atol'
    assert doc['data']['summary']['above_threshold'] >= 3


def test_tracediff_missing_records_are_incomplete_not_a_pass(rust_bin, golden, tmp_path):
    fewer = golden_outputs()
    del fewer['lm_head#1']
    doc = both(rust_bin, 'tracediff', str(golden), str(write_trace(tmp_path / 'fewer', fewer)))
    assert doc['data']['verdict'] == 'incomplete' and doc['data']['summary']['missing'] == 1
    no_output = write_trace(tmp_path / 'no-output', golden_outputs(), drop_output=('model.norm#0',))
    doc = both(rust_bin, 'tracediff', str(no_output), str(golden))
    assert doc['data']['verdict'] == 'incomplete'
    empty = write_trace(tmp_path / 'empty', {})
    doc = both(rust_bin, 'tracediff', str(empty), str(empty))
    assert doc['data']['verdict'] == 'incomplete' and doc['data']['summary']['worst_rel_l2'] is None


def test_tracediff_caps_lists_like_python(rust_bin, tmp_path):
    rng = np.random.default_rng(3)
    extra = [(f'block.{i}#0', rng.normal(size=(2, 3)).astype(np.float32)) for i in range(25)]
    a = write_trace(tmp_path / 'a', golden_outputs(), extra_records=extra, samples=2)
    moved = [(k, (v * (1 + 1e-3 * (i + 1))).astype(np.float32)) for i, (k, v) in enumerate(extra)]
    b = write_trace(tmp_path / 'b', golden_outputs(), extra_records=moved, samples=2)
    for items in ('2', '0', '-1', '20', '100'):
        doc = both(rust_bin, 'tracediff', str(a), str(b), '--max-items', items)
        assert len(doc['data']['worst']['items']) == min(20, 10**6 if int(items) < 0 else int(items))
    assert both(rust_bin, 'tracediff', str(a), str(b), CUMLX_MAX_ITEMS='3')['truncated'] is True


def _manifest(path: Path) -> dict:
    return json.loads((path / 'manifest.json').read_text())


def _write_manifest(path: Path, m: dict) -> None:
    (path / 'manifest.json').write_text(json.dumps(m))


def test_trace_errors_have_the_same_codes(rust_bin, golden, tmp_path):
    ok = str(golden)

    def broken(name: str, edit) -> str:
        dst = tmp_path / name
        shutil.copytree(golden, dst)
        edit(dst)
        return str(dst)

    def manifest_edit(fn):
        def edit(d):
            m = _manifest(d)
            fn(m)
            _write_manifest(d, m)
        return edit

    def resign(d):
        """Write a new SHA-256 into the manifest, so the reader reaches its own checks."""
        import hashlib
        m = _manifest(d)
        for s in m['samples']:
            s['sha256'] = hashlib.sha256((d / s['file']).read_bytes()).hexdigest()
        _write_manifest(d, m)

    def header(d, fn):
        f = d / 'samples' / 's0000.safetensors'
        raw = f.read_bytes()
        n = int.from_bytes(raw[:8], 'little')
        h = json.loads(raw[8:8 + n])
        fn(h)
        new = json.dumps(h).encode()
        new += b' ' * ((8 - len(new) % 8) % 8)
        f.write_bytes(len(new).to_bytes(8, 'little') + new + raw[8 + n:])
        resign(d)

    def first_tensor(h):
        return next(k for k in h if k.startswith('r'))

    cases = {
        'missing_dir': (str(tmp_path / 'nope'), 'E_INPUT_NOT_FOUND', True),
        'not_json': (broken('not-json', lambda d: (d / 'manifest.json').write_text('{oops')), 'E_INPUT_INVALID', False),
        'schema': (broken('schema', manifest_edit(lambda m: m.update(schema='cumlx.trace.v0'))), 'E_INPUT_INVALID', True),
        'no_schema_list': (broken('list', lambda d: (d / 'manifest.json').write_text('[1, 2]')), 'E_INPUT_INVALID', True),
        'no_samples': (broken('no-samples', manifest_edit(lambda m: m.pop('samples'))), 'E_INPUT_INVALID', True),
        'no_source': (broken('no-source', manifest_edit(lambda m: m.pop('source'))), 'E_INPUT_INVALID', True),
        'sample_field_missing': (broken('sample-field', manifest_edit(lambda m: m['samples'][0].pop('sha256'))), 'E_INPUT_INVALID', False),
        'sample_field_unknown': (broken('sample-unknown', manifest_edit(lambda m: m['samples'][0].update(extra=1))), 'E_INPUT_INVALID', False),
        'record_field_unknown': (broken('record-unknown', manifest_edit(lambda m: m['samples'][0]['records'][0].update(extra=1))), 'E_INPUT_INVALID', False),
        'record_field_missing': (broken('record-missing', manifest_edit(lambda m: m['samples'][0]['records'][0].pop('type'))), 'E_INPUT_INVALID', False),
        'tensor_ref_unknown': (broken('ref-unknown', manifest_edit(lambda m: m['samples'][0]['records'][0]['output'].update(x=1))), 'E_INPUT_INVALID', False),
        'golden_kind': (broken('kind', manifest_edit(lambda m: m['golden'].update(kind='guess'))), 'E_INPUT_INVALID', True),
        'golden_kind_missing': (broken('kind-missing', manifest_edit(lambda m: m['golden'].pop('kind'))), 'E_INPUT_INVALID', True),
        'sample_file_missing': (broken('file-missing', lambda d: (d / 'samples' / 's0000.safetensors').unlink()), 'E_INPUT_NOT_FOUND', True),
        'checksum': (broken('checksum', lambda d: (d / 'samples' / 's0000.safetensors').open('ab').write(b'\0')), 'E_INPUT_INVALID', True),
        'short_file': (broken('short', lambda d: ((d / 'samples' / 's0000.safetensors').write_bytes(b'\1\2'), resign(d))), 'E_INPUT_INVALID', True),
        'header_length': (broken('hlen', lambda d: ((d / 'samples' / 's0000.safetensors').write_bytes((10**9).to_bytes(8, 'little') + b'{}'), resign(d))), 'E_INPUT_INVALID', True),
        'header_not_object': (broken('hlist', lambda d: ((d / 'samples' / 's0000.safetensors').write_bytes((8).to_bytes(8, 'little') + b'[]      '), resign(d))), 'E_INPUT_INVALID', True),
        'dtype': (broken('dtype', lambda d: header(d, lambda h: h[first_tensor(h)].update(dtype='F8_E5M2'))), 'E_INPUT_INVALID', True),
        'offsets': (broken('offsets', lambda d: header(d, lambda h: h[first_tensor(h)].update(data_offsets=[0, 4]))), 'E_INPUT_INVALID', True),
        'malformed_entry': (broken('malformed', lambda d: header(d, lambda h: h[first_tensor(h)].pop('shape'))), 'E_INPUT_INVALID', True),
        'negative_shape': (broken('negative', lambda d: header(d, lambda h: h[first_tensor(h)].update(shape=[-1]))), 'E_INPUT_INVALID', True),
        'overlap': (broken('overlap', lambda d: header(d, lambda h: [h[k].update(data_offsets=[0, h[k]['data_offsets'][1] - h[k]['data_offsets'][0]])
                                                                     for k in sorted(h) if k.startswith('r')][:2])), 'E_INPUT_INVALID', True),
        'tensor_missing': (broken('tensor-missing', lambda d: header(d, lambda h: h.pop('r00003.out'))), 'E_INPUT_INVALID', True),
        'tokens_missing': (broken('tokens-missing', lambda d: header(d, lambda h: h.pop('token_ids'))), 'E_INPUT_INVALID', True),
    }
    for name, (path, code, messages) in cases.items():
        for a, b in ((path, ok), (ok, path)):
            p = py('tracediff', a, b)
            assert p[1]['errors'] and p[1]['errors'][0]['code'] == code, (name, p)
            assert_parity(p, rs(rust_bin, 'tracediff', a, b), messages=messages)


def test_traces_of_other_inputs_are_refused_by_both(rust_bin, golden, tmp_path):
    other_tokens = write_trace(tmp_path / 'tokens', golden_outputs(), tokens=[1, 5, 9, 24])
    both(rust_bin, 'tracediff', str(golden), str(other_tokens))
    shorter = write_trace(tmp_path / 'shorter', golden_outputs(), tokens=[1, 5, 9])
    both(rust_bin, 'tracediff', str(golden), str(shorter))
    two = write_trace(tmp_path / 'two', golden_outputs(), samples=2)
    doc = both(rust_bin, 'tracediff', str(golden), str(two))
    assert doc['errors'][0]['code'] == 'E_INPUT_INVALID' and 'number of samples' in doc['errors'][0]['message']


def test_relative_paths_and_odd_spellings_are_reported_alike(rust_bin, tmp_path):
    write_trace(tmp_path / 'g', golden_outputs())
    write_trace(tmp_path / 'h', golden_outputs(1))
    both(rust_bin, 'tracediff', 'g/', './/h', cwd=tmp_path)
    both(rust_bin, 'tracediff', 'g', 'nope/', cwd=tmp_path)


# ----------------------------------------------------------------------------- weights (MLX)
LLAMA = {'model_type': 'llama', 'architectures': ['LlamaForCausalLM'], 'hidden_size': 32, 'num_hidden_layers': 2,
         'intermediate_size': 64, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'rms_norm_eps': 1e-5,
         'vocab_size': 64, 'rope_theta': 10000.0, 'max_position_embeddings': 64, 'tie_word_embeddings': False}
MIXTRAL = {'model_type': 'mixtral', 'architectures': ['MixtralForCausalLM'], 'hidden_size': 32, 'num_hidden_layers': 2,
           'intermediate_size': 24, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'num_local_experts': 4,
           'num_experts_per_tok': 2, 'rms_norm_eps': 1e-5, 'vocab_size': 64, 'rope_theta': 10000.0, 'tie_word_embeddings': False}
QWEN3_MOE = {'model_type': 'qwen3_moe', 'architectures': ['Qwen3MoeForCausalLM'], 'hidden_size': 32, 'num_hidden_layers': 2,
             'intermediate_size': 48, 'moe_intermediate_size': 16, 'num_attention_heads': 4, 'num_key_value_heads': 2,
             'head_dim': 8, 'num_experts': 4, 'num_experts_per_tok': 2, 'decoder_sparse_step': 1, 'mlp_only_layers': [],
             'norm_topk_prob': True, 'rms_norm_eps': 1e-5, 'vocab_size': 64, 'rope_theta': 10000.0,
             'max_position_embeddings': 64, 'tie_word_embeddings': False}


def mlx_params(config: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """Parameters of a fresh mlx-lm model, as float32 NumPy, in the port's names."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes as get_model_classes
    mx.random.seed(seed)
    model_class, args_class = get_model_classes(config=config)
    model = model_class(args_class.from_dict(config))
    rng = np.random.default_rng(seed)
    return {k: (np.array(v.astype(mx.float32)) + 0.1 * rng.normal(size=v.shape)).astype(np.float32)
            for k, v in tree_flatten(model.parameters())}


def save_port(path: Path, params: dict[str, np.ndarray], config: dict, *, dtype: str = 'float32', shards: int = 1) -> Path:
    import mlx.core as mx
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps(config))
    names = sorted(params)
    for i in range(shards):
        part = {k: mx.array(params[k]).astype(getattr(mx, dtype)) for k in names[i::shards]}
        name = 'model.safetensors' if shards == 1 else f'model-{i + 1:05d}-of-{shards:05d}.safetensors'
        mx.save_safetensors(str(path / name), part, metadata={'format': 'mlx'})
    return path


def save_original(path: Path, tensors: dict[str, np.ndarray], config: dict, *, shards: int = 1) -> Path:
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps(config))
    names = sorted(tensors)
    if shards == 1:
        save_file(tensors, path / 'model.safetensors')
        return path
    weight_map = {}
    for i in range(shards):
        f = f'model-{i + 1:05d}-of-{shards:05d}.safetensors'
        save_file({k: tensors[k] for k in names[i::shards]}, path / f)
        weight_map.update({k: f for k in names[i::shards]})
    (path / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {}, 'weight_map': weight_map}))
    return path


@pytest.fixture
def llama(mlx, tmp_path):
    params = mlx_params(LLAMA)
    original = save_original(tmp_path / 'original', params, LLAMA)
    return params, original


def test_weights_clean_ports_pass_in_both(rust_bin, llama, tmp_path):
    params, original = llama
    for dtype, shards in (('float32', 1), ('float16', 1), ('bfloat16', 2)):
        port = save_port(tmp_path / f'port-{dtype}', params, LLAMA, dtype=dtype, shards=shards)
        doc = both(rust_bin, 'weights', str(original), str(port))
        assert doc['data']['verdict'] == 'ok' and doc['data']['summary']['pass'] == len(params), dtype
        assert doc['next_steps'][0]['command'].startswith('cumlx capture ')
    sharded = save_original(tmp_path / 'original-sharded', params, LLAMA, shards=3)
    assert both(rust_bin, 'weights', str(sharded), str(tmp_path / 'port-float32'))['data']['verdict'] == 'ok'
    single = original / 'model.safetensors'
    assert both(rust_bin, 'weights', str(single), str(tmp_path / 'port-float32'))['data']['verdict'] == 'ok'


@pytest.mark.parametrize('fault,target,diagnosis', [
    ('transpose', 'model.layers.0.self_attn.q_proj.weight', 'transposed'),
    ('scale2', 'model.layers.1.self_attn.o_proj.weight', 'scaled_by_2'),
    ('scale1.5', 'model.layers.0.mlp.up_proj.weight', 'scaled_by_1.5'),
    ('sign', 'model.layers.1.mlp.down_proj.weight', 'sign_flipped'),
    ('noise', 'model.norm.weight', None),
])
def test_weights_faults_fail_in_both_and_are_named(rust_bin, llama, tmp_path, fault, target, diagnosis):
    params, original = llama
    bad = dict(params)
    w = params[target]
    bad[target] = {'transpose': lambda: np.ascontiguousarray(w.T), 'scale2': lambda: w * np.float32(2),
                   'scale1.5': lambda: w * np.float32(1.5), 'sign': lambda: -w,
                   'noise': lambda: (w + np.float32(1e-3)).astype(np.float32)}[fault]()
    port = save_port(tmp_path / 'port', bad, LLAMA)
    doc = both(rust_bin, 'weights', str(original), str(port))
    first = doc['data']['first_failure']
    assert doc['data']['verdict'] == 'diverged' and doc['exit_code'] == 1
    assert first['name'] == target and first.get('diagnosis') == diagnosis
    assert doc['data']['summary']['fail'] == 1


def test_weights_several_faults_cap_and_report_file(rust_bin, llama, tmp_path):
    params, original = llama
    bad = {k: (v * np.float32(3) if k.endswith('proj.weight') else v) for k, v in params.items()}
    port = save_port(tmp_path / 'port', bad, LLAMA)
    doc = both(rust_bin, 'weights', str(original), str(port), '--max-items', '2')
    assert doc['truncated'] and doc['data']['failures']['omitted'] > 0
    out_py, out_rs = tmp_path / 'py.json', tmp_path / 'rs.json'
    p = py('weights', str(original), str(port), '-o', str(out_py))
    r = rs(rust_bin, 'weights', str(original), str(port), '-o', str(out_rs))
    assert p[0] == r[0] == 1 and p[1]['artifacts'][0]['kind'] == r[1]['artifacts'][0]['kind'] == 'weights_report'
    assert not same(json.loads(out_py.read_text()), json.loads(out_rs.read_text()))
    exists = both(rust_bin, 'weights', str(original), str(port), '-o', str(out_py))
    assert exists['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    code, _ = rs(rust_bin, 'weights', str(original), str(port), '-o', str(out_rs), '--overwrite')
    assert code == 1 and not same(json.loads(out_py.read_text()), json.loads(out_rs.read_text()))


def test_weights_tied_embeddings_missing_and_extra_tensors(rust_bin, mlx, tmp_path):
    config = {**LLAMA, 'tie_word_embeddings': True}
    params = mlx_params(config)
    assert 'lm_head.weight' not in params
    tied = {**params, 'lm_head.weight': params['model.embed_tokens.weight']}
    original = save_original(tmp_path / 'tied', tied, config)
    port = save_port(tmp_path / 'port', params, config)
    doc = both(rust_bin, 'weights', str(original), str(port))
    assert doc['data']['verdict'] == 'ok'
    wrong = save_original(tmp_path / 'wrong-head', {**tied, 'lm_head.weight': -params['model.embed_tokens.weight']}, config)
    doc = both(rust_bin, 'weights', str(wrong), str(port))
    assert doc['data']['first_failure']['name'] == 'lm_head.weight'
    assert doc['data']['first_failure']['port_name'] == 'model.embed_tokens.weight'
    extra = save_original(tmp_path / 'extra', {**tied, 'model.layers.0.self_attn.rotary_emb.inv_freq': np.ones(4, np.float32),
                                               'model.position_ids': np.arange(8, dtype=np.int64)}, config)
    doc = both(rust_bin, 'weights', str(extra), str(port))
    assert doc['data']['verdict'] == 'incomplete' and doc['data']['summary']['unverified'] == 2 and doc['exit_code'] == 5
    fewer = save_original(tmp_path / 'fewer', {k: v for k, v in tied.items() if 'layers.1.' not in k}, config)
    doc = both(rust_bin, 'weights', str(fewer), str(port))
    assert doc['data']['verdict'] == 'ok' and doc['data']['summary']['extra_in_port'] > 0


def _mixtral_original(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """transformers 4 Mixtral names: one tensor per expert, w1/w2/w3."""
    out = {}
    for k, v in params.items():
        m = re.match(r'(.+)\.block_sparse_moe\.switch_mlp\.(gate|up|down)_proj\.weight$', k)
        if not m:
            out[k] = v
            continue
        w = {'gate': 'w1', 'down': 'w2', 'up': 'w3'}[m.group(2)]
        for e in range(v.shape[0]):
            out[f'{m.group(1)}.block_sparse_moe.experts.{e}.{w}.weight'] = v[e]
    return out


def _qwen3_moe_original(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """transformers 5 fused names: experts.gate_up_proj = gate and up on axis 1, experts.down_proj."""
    out = {k: v for k, v in params.items() if '.switch_mlp.' not in k}
    blocks = {k.split('.switch_mlp.')[0] for k in params if '.switch_mlp.' in k}
    for b in blocks:
        gate, up, down = (params[f'{b}.switch_mlp.{p}.weight'] for p in ('gate_proj', 'up_proj', 'down_proj'))
        out[f'{b}.experts.gate_up_proj'] = np.concatenate([gate, up], axis=1)
        out[f'{b}.experts.down_proj'] = down
    return out


@pytest.mark.parametrize('config,layout', [(MIXTRAL, _mixtral_original), (QWEN3_MOE, _qwen3_moe_original)])
def test_weights_moe_experts_and_rotated_experts(rust_bin, mlx, tmp_path, config, layout):
    params = mlx_params(config)
    stacks = [k for k in params if '.switch_mlp.' in k]
    assert stacks
    original = save_original(tmp_path / 'original', layout(params), config)
    port = save_port(tmp_path / 'port', params, config)
    doc = both(rust_bin, 'weights', str(original), str(port))
    assert doc['data']['verdict'] == 'ok' and doc['data']['summary']['extra_in_port'] == 0
    rotated = dict(params)
    key = sorted(stacks)[0]
    rotated[key] = np.roll(params[key], 1, axis=0)
    doc = both(rust_bin, 'weights', str(original), str(save_port(tmp_path / 'rotated', rotated, config)))
    first = doc['data']['first_failure']
    assert doc['data']['verdict'] == 'diverged' and first['diagnosis'].startswith('expert_order: this expert is at index ')


def test_weights_options_and_inputs_only_python_handles(rust_bin, llama, tmp_path):
    params, original = llama
    port = save_port(tmp_path / 'port', params, LLAMA)
    rules = tmp_path / 'rules.json'
    rules.write_text('{"rules": []}')
    for extra in (['--mapping', str(rules)], ['--loader', 'python:pkg.mod:fn']):
        code, doc = rs(rust_bin, 'weights', str(original), str(port), *extra)
        assert code == 3 and doc['errors'][0]['code'] == 'E_PYTHON_ONLY'
        assert doc['errors'][0]['hint'].endswith(f"cumlx weights {original} {port} {' '.join(extra)}")
    code, doc = rs(rust_bin, 'weights', 'Qwen/Qwen2.5-0.5B', str(port))
    assert code == 3 and doc['errors'][0]['code'] == 'E_PYTHON_ONLY'
    for missing in ('./nope', str(tmp_path / 'nope')):
        code, doc = rs(rust_bin, 'weights', str(original), missing)
        assert code == 4 and doc['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'
        assert assert_parity(py('weights', str(original), missing), (code, doc))
    code, doc = rs(rust_bin, 'weights', str(tmp_path / 'nope'), str(port))
    assert code == 4 and doc['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'
    empty = tmp_path / 'empty'
    empty.mkdir()
    both(rust_bin, 'weights', str(empty), str(port))
    no_tensors = tmp_path / 'no-tensors'
    no_tensors.mkdir()
    save_file({}, no_tensors / 'model.safetensors')
    both(rust_bin, 'weights', str(no_tensors), str(port))


def test_a_quantized_port_is_refused_not_judged(rust_bin, llama, tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import _get_classes as get_model_classes
    params, original = llama
    model_class, args_class = get_model_classes(config=LLAMA)
    model = model_class(args_class.from_dict(LLAMA))
    model.load_weights([(k, mx.array(v)) for k, v in params.items()])
    nn.quantize(model, group_size=32, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear))
    q = tmp_path / 'q'
    q.mkdir()
    mx.save_safetensors(str(q / 'model.safetensors'), dict(tree_flatten(model.parameters())))
    (q / 'config.json').write_text(json.dumps({**LLAMA, 'quantization': {'group_size': 32, 'bits': 8}}))
    code, doc = rs(rust_bin, 'weights', str(original), str(q))
    assert code == 3 and doc['errors'][0]['code'] == 'E_PYTHON_ONLY' and 'quantized' in doc['errors'][0]['message']
    assert py('weights', str(original), str(q))[0] == 0  # Python judges it; Rust refuses it, never passes it
    (q / 'config.json').write_text(json.dumps(LLAMA))  # no config entry: the .scales tensors still refuse it
    assert rs(rust_bin, 'weights', str(original), str(q))[1]['errors'][0]['code'] == 'E_PYTHON_ONLY'


def test_what_rust_reports_where_mlx_lm_refuses_to_load(rust_bin, llama, tmp_path):
    """A non-square transposed weight: mlx-lm refuses to load the port; cumlx-rs reads it and names the fault."""
    params, original = llama
    bad = {**params, 'model.layers.0.self_attn.k_proj.weight': np.ascontiguousarray(params['model.layers.0.self_attn.k_proj.weight'].T)}
    port = save_port(tmp_path / 'port', bad, LLAMA)
    code, doc = rs(rust_bin, 'weights', str(original), str(port))
    first = doc['data']['first_failure']
    assert code == 1 and first['reason'] == 'shape_mismatch' and first['diagnosis'] == 'transposed'
    assert py('weights', str(original), str(port))[1]['errors'][0]['code'] == 'E_MODEL_LOAD'
    conv = {**params, 'model.layers.0.conv.weight': np.ones((8, 3, 4), np.float32)}
    port = save_port(tmp_path / 'conv', {**params, 'model.layers.0.conv.weight': np.ones((8, 4, 3), np.float32)}, LLAMA)
    orig = save_original(tmp_path / 'orig-conv', conv, LLAMA)
    code, doc = rs(rust_bin, 'weights', str(orig), str(port))
    assert code == 5 and doc['data']['unverified']['items'][0]['name'] == 'model.layers.0.conv.weight'


# ----------------------------------------------------------------------------- the selftest's models (torch)
def _capture(tiny: Path, dst: Path, scale: float | None = None) -> Path:
    import torch

    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.selftest import TOKENS as SELFTEST_TOKENS
    model = load_hf_model(str(tiny))
    if scale:
        with torch.no_grad():
            model.lm_head.weight.mul_(scale)
    out = prepare_output_dir(dst, overwrite=False)
    w = TraceWriter(out, source={'framework': 'torch', 'model': str(tiny)}, golden={'kind': 'reference', 'dtype': 'float32'},
                    max_bytes=1 << 30)
    capture_model(model, [np.asarray(SELFTEST_TOKENS)], w, decode_steps=1)
    w.finish()
    return out


def test_selftest_traces_and_weights_agree(rust_bin, e2e, tmp_path):
    """Not marked e2e: the verifier job of ci.yml builds no Rust binary. rust.yml runs it with --require-e2e."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from cumlx.verify import faults, mlx_port
    from cumlx.verify.fixtures import make_tiny_llama
    tiny = tmp_path / 'tiny-llama'
    make_tiny_llama(tiny)
    a, b = _capture(tiny, tmp_path / 'golden'), _capture(tiny, tmp_path / 'golden-again')
    doc = both(rust_bin, 'tracediff', str(a), str(b))
    assert doc['data']['verdict'] == 'ok' and doc['data']['summary']['compared'] > 20
    moved = _capture(tiny, tmp_path / 'golden-moved', scale=1.01)
    doc = both(rust_bin, 'tracediff', str(a), str(moved), '--max-items', '-1')
    assert doc['data']['verdict'] == 'diverged' and doc['data']['worst']['items'][0]['key'].startswith(('lm_head', '#'))

    doc = both(rust_bin, 'weights', str(tiny), str(tiny))  # the selftest's E7 clean case: an HF folder as the port
    assert doc['data']['verdict'] == 'ok'
    for fid, expected in (('F1', 'transposed'),):
        model = mlx_port.load_port(str(tiny))
        target = faults.TINY_LLAMA_TARGETS[fid]
        faults.inject(model, fid, target)
        port = tmp_path / f'port-{fid}'
        port.mkdir()
        shutil.copy(tiny / 'config.json', port / 'config.json')
        mx.save_safetensors(str(port / 'model.safetensors'), dict(tree_flatten(model.parameters())))
        doc = both(rust_bin, 'weights', str(tiny), str(port))
        first = doc['data']['first_failure']
        assert first['name'] == target + '.weight' and first['diagnosis'] == expected


# ----------------------------------------------------------------------------- the vectors that cargo test reads
def test_the_rust_test_vectors_are_current():
    """`cargo test` compares the Rust metrics with rust/cumlx/tests/data/python-vectors.json; it must be today's Python."""
    done = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts' / 'rust_vectors.py'), '--check'],
                          capture_output=True, text=True, env=_env())
    assert done.returncode == 0, 'stale: run python scripts/rust_vectors.py\n' + done.stderr
