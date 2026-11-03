"""`cumlx kernel`: the parts that need no GPU. Golden format, comparison limits, pre-translation layer, adapters."""
import json

import numpy as np
import pytest

from cumlx.errors import CumlxError
from cumlx.verify.metrics import KERNEL_THRESHOLDS, compare_kernel_output
from cumlx.verify.opgolden import SCHEMA, KernelCase, OpGolden, TensorSpec, read_cases, write_golden


def _case(case_id='c', kernel='k'):
    return KernelCase(case_id, kernel, [TensorSpec('x', 'float32', (3,))], [TensorSpec('y', 'float32', (3,))],
                      {'N': 3, 'B': True}, (128, 1, 1), (128, 1, 1))


def _fixture_golden(path, *, kind='fixture', y=None):
    x = np.array([1, 2, 3], np.float32)
    y = x * 2 if y is None else y
    return write_golden(path, kind=kind, source={'framework': 'test'}, kernels={'k': {'source_sha256': 'abc'}},
                        floating_point={}, cases=[{'case': _case(), 'inputs': {'x': x}, 'outputs': {'y': y},
                                                   'repeat_identical': True}])


# ---------------------------------------------------------------- comparison limits
def test_the_limit_is_the_one_written_before_the_measurement():
    assert KERNEL_THRESHOLDS == {'rel_l2': 1e-5, 'max_rel': 1e-5, 'zero_atol': 1e-6}


def test_compare_passes_on_rounding_and_fails_just_above_the_limit():
    e = np.full(1000, 1.0, np.float32)
    assert compare_kernel_output(e.copy(), e)['verdict'] == 'pass'
    assert compare_kernel_output(e.copy(), e)['bitwise_identical'] is True
    near = np.nextafter(e, np.float32(2))
    r = compare_kernel_output(near, e)
    assert r['verdict'] == 'pass' and r['max_ulp'] == 1 and r['bitwise_identical'] is False
    off = e.copy()
    off[:] = 1 + 2e-5
    r = compare_kernel_output(off, e)
    assert r['verdict'] == 'fail' and r['reason'] == 'rel_l2_above_limit'


def test_one_wrong_element_cannot_hide_in_a_long_output():
    e = np.full(1_000_000, 1.0, np.float32)
    a = e.copy()
    a[123] = 1.005  # rel_l2 = 5e-6, under the limit by itself
    r = compare_kernel_output(a, e)
    assert r['verdict'] == 'fail' and r['reason'] == 'max_rel_above_limit' and r['index'] == [123]


@pytest.mark.parametrize('actual,expected,reason', [
    ([np.nan, 1.0], [1.0, 1.0], 'nan_mismatch'),
    ([1.0, 1.0], [np.nan, 1.0], 'nan_mismatch'),
    ([np.inf, 1.0], [-np.inf, 1.0], 'infinity_mismatch'),
    ([1.0, 1.0], [np.inf, 1.0], 'infinity_mismatch'),
])
def test_special_values_must_match(actual, expected, reason):
    r = compare_kernel_output(np.array(actual, np.float32), np.array(expected, np.float32))
    assert r['verdict'] == 'fail' and r['reason'] == reason


def test_nan_at_the_same_place_is_not_a_failure():
    a = np.array([np.nan, 1.0, np.inf], np.float32)
    r = compare_kernel_output(a.copy(), a)
    assert r['verdict'] == 'pass' and r['nan_in_golden'] == 1
    r = compare_kernel_output(np.array([np.nan], np.float32), np.array([np.nan], np.float32))
    assert r['verdict'] == 'pass' and r['basis'] == 'all_special_values'


def test_integers_must_match_exactly():
    e = np.arange(10, dtype=np.uint8)
    assert compare_kernel_output(e.copy(), e)['verdict'] == 'pass'
    a = e.copy()
    a[7] = 8
    r = compare_kernel_output(a, e)
    assert r['verdict'] == 'fail' and r['reason'] == 'integer_mismatch' and r['index'] == [7] and r['mismatched'] == 1


def test_shape_and_dtype_are_never_converted():
    e = np.zeros((2, 3), np.float32)
    assert compare_kernel_output(np.zeros(6, np.float32), e)['reason'] == 'shape_mismatch'
    assert compare_kernel_output(np.zeros((2, 3), np.float16), e)['reason'] == 'dtype_mismatch'


def test_subnormals_flushed_to_zero_are_counted():
    e = np.array([1.0, 1e-40, 1e-41], np.float32)
    r = compare_kernel_output(np.array([1.0, 0.0, 0.0], np.float32), e)
    assert r['verdict'] == 'pass' and r['subnormal_in_golden'] == 2 and r['subnormal_flushed'] == 2
    only = np.array([1e-40], np.float32)
    assert compare_kernel_output(np.array([0.0], np.float32), only)['verdict'] == 'fail'


def test_all_zero_golden_uses_the_absolute_limit():
    e = np.zeros(4, np.float32)
    assert compare_kernel_output(np.full(4, 1e-7, np.float32), e)['basis'] == 'zero_atol'
    assert compare_kernel_output(np.full(4, 1e-3, np.float32), e)['reason'] == 'golden_is_zero_but_output_is_not'


# ---------------------------------------------------------------- golden format
def test_golden_roundtrip(tmp_path):
    y = np.array([2, 4, 6], np.float32)
    manifest = _fixture_golden(tmp_path, y=y)
    assert manifest['schema'] == SCHEMA
    golden = OpGolden(tmp_path)
    assert golden.kind == 'fixture' and [c.id for c in golden.cases] == ['c']
    case = golden.cases[0]
    assert case.constants == {'N': 3, 'B': True} and case.blocks == (1, 1, 1)
    inputs, outputs = golden.load(case)
    assert np.array_equal(inputs['x'], [1, 2, 3]) and np.array_equal(outputs['y'], y)
    assert golden.info('c')['repeat_identical'] is True


def test_golden_rejects_a_changed_file(tmp_path):
    _fixture_golden(tmp_path)
    golden = OpGolden(tmp_path)
    path = tmp_path / golden.info('c')['file']
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(CumlxError, match='SHA-256') as exc:
        golden.load(golden.cases[0])
    assert exc.value.code == 'E_INPUT_INVALID'


@pytest.mark.parametrize('change,message', [
    (lambda m: m.update(schema='cumlx.opgolden.v2'), 'expected cumlx.opgolden.v1'),
    (lambda m: m.update(golden={'kind': 'guess'}), 'unknown golden.kind'),
    (lambda m: m.update(golden={'kind': 'cuda'}), 'must record source.compiler'),
    (lambda m: m.update(cases=[]), 'at least one case'),
    (lambda m: m['cases'][0].update(grid=[100, 1, 1]), 'whole number of threadgroups'),
    (lambda m: m['cases'][0]['inputs'][0].update(dtype='complex64'), 'unsupported dtype'),
    (lambda m: m['cases'][0]['inputs'][0].update(name='x;evil'), 'bad tensor name'),
    (lambda m: m['cases'][0].update(outputs=[]), 'at least one output'),
])
def test_golden_rejects_bad_manifests(tmp_path, change, message):
    manifest = _fixture_golden(tmp_path)
    change(manifest)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(CumlxError, match=message) as exc:
        OpGolden(tmp_path)
    assert exc.value.exit_code == 4


def test_golden_rejects_a_path_outside_itself(tmp_path):
    manifest = _fixture_golden(tmp_path)
    manifest['cases'][0]['file'] = '../outside.safetensors'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    golden = OpGolden(tmp_path)
    with pytest.raises(CumlxError, match='bad file'):
        golden.load(golden.cases[0])


def test_cases_file_with_inline_values(tmp_path):
    p = tmp_path / 'cases.json'
    p.write_text(json.dumps({'schema': 'cumlx.kernel-cases.v1', 'cases': [{
        'id': 'a', 'kernel': 'k', 'grid': [1, 1, 1], 'threadgroup': [1, 1, 1],
        'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': [2], 'values': [1.5, 2.5]}],
        'outputs': [{'name': 'y', 'dtype': 'float32', 'shape': [2]}]}]}))
    [(case, arrays)] = read_cases(p)
    assert case.kernel == 'k' and arrays['x'].dtype == np.float32 and arrays['x'].tolist() == [1.5, 2.5]


def test_cases_file_may_not_read_outside_its_folder(tmp_path):
    p = tmp_path / 'cases.json'
    p.write_text(json.dumps({'schema': 'cumlx.kernel-cases.v1', 'cases': [{
        'id': 'a', 'kernel': 'k', 'grid': [1, 1, 1], 'threadgroup': [1, 1, 1],
        'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': [2], 'file': '../x.safetensors'}],
        'outputs': [{'name': 'y', 'dtype': 'float32', 'shape': [2]}]}]}))
    with pytest.raises(CumlxError, match='beside the cases file'):
        read_cases(p)
