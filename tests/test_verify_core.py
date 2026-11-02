"""Verifier building blocks that need only NumPy: safetensors IO, trace container, metrics, mapping."""
import json
import struct

import numpy as np
import pytest

from cumlx.errors import CumlxError
from cumlx.verify.mapping import NameMapper
from cumlx.verify.metrics import compare, resolve_threshold
from cumlx.verify.safetensors_io import SafetensorsReader, load_file, save_file
from cumlx.verify.trace import ModuleRecord, TensorRef, Trace, TraceWriter, prepare_output_dir


def test_safetensors_roundtrip_all_dtypes(tmp_path):
    rng = np.random.default_rng(0)
    data = {'f32': rng.normal(size=(3, 4)).astype(np.float32), 'f16': rng.normal(size=5).astype(np.float16),
            'f64': rng.normal(size=(2,)), 'i64': np.arange(6, dtype=np.int64).reshape(2, 3),
            'i32': np.arange(3, dtype=np.int32), 'u8': np.arange(9, dtype=np.uint8), 'b': np.array([True, False]),
            'empty': np.zeros((0, 3), np.float32), 'scalar': np.array(3.5, np.float32)}
    p = tmp_path / 'x.safetensors'
    save_file(data, p, metadata={'a': 'b'})
    r = SafetensorsReader(p)
    assert r.metadata == {'a': 'b'} and r.keys() == sorted(data)
    back = load_file(p)
    for k, v in data.items():
        assert back[k].dtype == v.dtype and back[k].shape == v.shape and np.array_equal(back[k], v), k
    save_file({'t': np.arange(12, dtype=np.float32).reshape(3, 4).T}, p)  # non-contiguous view
    assert np.array_equal(load_file(p)['t'], np.arange(12, dtype=np.float32).reshape(3, 4).T)
    save_file(data, p, metadata={'a': 'b'})
    header_len = int.from_bytes(p.read_bytes()[:8], 'little')
    assert header_len % 8 == 0  # header padded to 8 bytes so the payload is aligned


def test_safetensors_interop_with_official_package(tmp_path):
    st = pytest.importorskip('safetensors.numpy')
    a = {'w': np.arange(12, dtype=np.float32).reshape(3, 4), 'ids': np.arange(4, dtype=np.int64)}
    save_file(a, tmp_path / 'ours.safetensors')
    theirs = st.load_file(str(tmp_path / 'ours.safetensors'))
    assert all(np.array_equal(theirs[k], a[k]) for k in a)
    st.save_file(a, str(tmp_path / 'theirs.safetensors'))
    ours = load_file(tmp_path / 'theirs.safetensors')
    assert all(np.array_equal(ours[k], a[k]) for k in a)


def test_safetensors_reads_bf16_as_float32(tmp_path):
    vals = np.array([1.0, -2.5, 3.140625], np.float32)
    bits = (vals.view(np.uint32) >> 16).astype('<u2')
    header = json.dumps({'x': {'dtype': 'BF16', 'shape': [3], 'data_offsets': [0, 6]}}).encode()
    (tmp_path / 'b.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + bits.tobytes())
    assert np.array_equal(load_file(tmp_path / 'b.safetensors')['x'], vals)


@pytest.mark.parametrize('corrupt', ['short', 'header_len', 'not_json', 'bad_offsets', 'overlap'])
def test_safetensors_rejects_corrupt_files(tmp_path, corrupt):
    p = tmp_path / 'c.safetensors'
    if corrupt == 'short':
        p.write_bytes(b'123')
    elif corrupt == 'header_len':
        p.write_bytes(struct.pack('<Q', 10**6) + b'{}')
    elif corrupt == 'not_json':
        p.write_bytes(struct.pack('<Q', 3) + b'{x}')
    else:
        entries = {'a': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8 if corrupt == 'overlap' else 4]}}
        if corrupt == 'overlap':
            entries['b'] = {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]}
        h = json.dumps(entries).encode()
        p.write_bytes(struct.pack('<Q', len(h)) + h + b'\0' * 8)
    with pytest.raises(CumlxError) as e:
        SafetensorsReader(p)
    assert e.value.code == 'E_INPUT_INVALID'


def _write_trace(path):
    out = prepare_output_dir(path, overwrite=False)
    w = TraceWriter(out, source={'framework': 'test'}, golden={'kind': 'reference', 'dtype': 'float32'},
                    max_bytes=1 << 20)
    rec = ModuleRecord(key='a#0', name='a', call=0, order=0, type='Linear', has_params=True, replayable=True,
                       inputs=[TensorRef('r00000.in0', [1, 2], 'float32', 'arg0')],
                       output=TensorRef('r00000.out', [1, 3], 'float32', 'tensor'))
    w.add_sample({'token_ids': np.array([[1, 2]]), 'r00000.in0': np.ones((1, 2), np.float32),
                  'r00000.out': np.full((1, 3), 2.0, np.float32)}, [rec], num_tokens=2)
    w.finish()
    return out


def test_trace_roundtrip_and_tamper_detection(tmp_path):
    out = _write_trace(tmp_path / 't')
    t = Trace(out)
    s = t.manifest.samples[0]
    assert t.manifest.schema == 'cumlx.trace.v1' and t.manifest.cumlx_version
    assert s.records[0].inputs[0].source == 'arg0'
    assert np.array_equal(t.tensor(s, 'r00000.out'), np.full((1, 3), 2.0, np.float32))
    f = out / s.file
    raw = bytearray(f.read_bytes())
    raw[-1] ^= 1
    f.write_bytes(bytes(raw))
    with pytest.raises(CumlxError, match='checksum'):
        Trace(out)


def test_trace_rejects_unknown_schema(tmp_path):
    out = _write_trace(tmp_path / 't')
    m = json.loads((out / 'manifest.json').read_text())
    m['schema'] = 'cumlx.trace.v0'
    (out / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(CumlxError) as e:
        Trace(out)
    assert e.value.code == 'E_INPUT_INVALID' and e.value.hint


def test_trace_size_limit(tmp_path):
    out = prepare_output_dir(tmp_path / 't', overwrite=False)
    w = TraceWriter(out, source={}, golden={'kind': 'reference'}, max_bytes=100)
    with pytest.raises(CumlxError) as e:
        w.add_sample({'x': np.zeros(1000, np.float32)}, [], num_tokens=1)
    assert e.value.code == 'E_SIZE_LIMIT' and '--max-bytes' in e.value.hint


def test_metrics_verdicts():
    t = resolve_threshold('float32')
    x = np.linspace(-1, 1, 64, dtype=np.float32).reshape(8, 8)
    ok = compare(x + 1e-7, x, t)
    assert ok['verdict'] == 'pass' and ok['rel_l2'] < 1e-5 and ok['cosine'] > 0.999999
    bad = compare(x * 1.001, x, t)
    assert bad['verdict'] == 'fail' and bad['reason'] == 'rel_l2_above_threshold'
    assert compare(x * 1.001, x, resolve_threshold('bfloat16'))['verdict'] == 'pass'
    assert compare(x.reshape(64), x, t)['verdict'] == 'pass'  # same elements, reshaped
    assert compare(x[:4], x, t)['reason'] == 'shape_mismatch'
    z = np.zeros((2, 2), np.float32)
    assert compare(z, z, t)['verdict'] == 'pass'
    assert compare(z + 1, z, t)['reason'] == 'golden_is_zero_but_output_is_not'
    inf = x.copy()
    inf[0, 0] = np.inf
    assert compare(inf, x, t)['reason'] == 'nonfinite_output'
    assert compare(x, inf, t)['reason'] == 'nonfinite_golden'
    with pytest.raises(ValueError):
        resolve_threshold('float32', rel_l2=-1)


def test_mapping_rules(tmp_path):
    p = tmp_path / 'm.json'
    p.write_text(json.dumps({'rules': [{'match': r'^transformer\.h\.(\d+)', 'replace': r'model.layers.\1'}],
                             'ignore': ['rotary_emb$']}))
    m = NameMapper.from_file(str(p))
    assert m.map('transformer.h.3.attn') == 'model.layers.3.attn'
    assert m.is_ignored('model.rotary_emb') and not m.is_ignored('model.norm')
    p.write_text('{"rules": [{"match": "("}]}')
    with pytest.raises(CumlxError):
        NameMapper.from_file(str(p))
