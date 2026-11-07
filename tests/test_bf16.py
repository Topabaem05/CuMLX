"""bfloat16 in `cumlx kernel port` and `cumlx kernel verify`: the mapping, the model, the comparison, and a GPU run.

The goldens here come from the model in cumlx.verify.bf16, so they are kind 'fixture' and earn no grade.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from cumlx.cli.main import main
from cumlx.errors import CumlxError
from cumlx.verify import bf16
from cumlx.verify.kernel_capture import generate_harness
from cumlx.verify.kernel_port import preprocess_file
from cumlx.verify.opgolden import KernelCase, OpGolden, TensorSpec, write_golden

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/kernel_fixtures/bf16_ops.cu'
STUB = 'python:kernel_stub_translator:StubTranslator'
SPECIAL = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x3F81, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80,
                    0x7FC0, 0xFFC0, 0x7F81, 0x7FFF, 0x0040, 0x0180], np.uint16)


def run(capsys, *argv):
    code = main([*argv, '--format', 'json'])
    return code, json.loads(capsys.readouterr().out)


def f32(bits):
    return np.array(bits, np.uint32)


# ---------------------------------------------------------------- the model
def test_conversions_follow_cuda_bf16_hpp():
    x = f32([0x3F800000, 0x3F808000, 0x3F818000, 0x3F808001, 0x00008000, 0x00018000, 0x00000001, 0x7F7FFFFF, 0xFF7FFFFF,
             0x7FC00000, 0xFF800001, 0x80000001])
    assert [hex(v) for v in bf16.from_float_bits(x, 'rn')] == [hex(v) for v in (
        0x3F80, 0x3F80, 0x3F82, 0x3F81, 0x0000, 0x0002, 0x0000, 0x7F80, 0xFF80, 0x7FFF, 0x7FFF, 0x8000)]
    assert list(bf16.from_float_bits(x, 'rz'))[:4] == [0x3F80, 0x3F80, 0x3F81, 0x3F80]
    assert list(bf16.from_float_bits(f32([0x00000001, 0x80000001, 0x7F7FFFFF]), 'ru')) == [0x0001, 0x8000, 0x7F80]
    assert list(bf16.from_float_bits(f32([0x00000001, 0x80000001, 0xFF7FFFFF]), 'rd')) == [0x0000, 0x8001, 0xFF80]


def test_the_two_models_of_one_rounding_agree():
    rng = np.random.default_rng(7)
    a = np.concatenate([np.repeat(SPECIAL, SPECIAL.size), rng.integers(0, 1 << 16, 3000, dtype=np.uint32).astype(np.uint16)])
    b = np.concatenate([np.tile(SPECIAL, SPECIAL.size), rng.integers(0, 1 << 16, 3000, dtype=np.uint32).astype(np.uint16)])
    one, minus_zero = np.full(a.shape, 0x3F80, np.uint16), np.full(a.shape, 0x8000, np.uint16)
    for op, fused in (('add', bf16.model_fma(a, one, b)), ('mul', bf16.model_fma(a, b, minus_zero))):
        model = bf16.model_op(op, a, b)
        nan = bf16.is_nan(model)
        assert (nan == bf16.is_nan(fused)).all() and (model[~nan] == fused[~nan]).all(), op


def test_known_results():
    assert bf16.model_op('add', np.array([0x0001], np.uint16), np.array([0x0001], np.uint16))[0] == 0x0002  # subnormals stay
    assert bf16.model_op('add', np.array([0x3F80], np.uint16), np.array([0xBF80], np.uint16))[0] == 0x0000  # x - x = +0
    assert bf16.model_op('add', np.array([0x8000], np.uint16), np.array([0x8000], np.uint16))[0] == 0x8000
    assert bf16.model_op('mul', np.array([0x7F80], np.uint16), np.array([0x0000], np.uint16))[0] == 0x7FFF
    assert bf16.model_op('mul', np.array([0x7F7F], np.uint16), np.array([0x4000], np.uint16))[0] == 0x7F80
    assert bf16.model_op('mul', np.array([0x0080], np.uint16), np.array([0x3F00], np.uint16))[0] == 0x0040  # 2^-126 / 2


# ---------------------------------------------------------------- the comparison
def test_bits_must_be_equal_and_nan_is_compared_by_class():
    golden = np.array([0x3F80, 0x7FFF, 0x0001], np.uint16)
    assert bf16.compare_bits(golden.copy(), golden)['verdict'] == 'pass'
    payload = bf16.compare_bits(np.array([0x3F80, 0xFFC0, 0x0001], np.uint16), golden)
    assert payload['verdict'] == 'pass' and payload['nan_payload_differs'] == 1
    one_ulp = bf16.compare_bits(np.array([0x3F81, 0x7FFF, 0x0001], np.uint16), golden)
    assert one_ulp['verdict'] == 'fail' and one_ulp['reason'] == 'bits_differ' and one_ulp['max_ulp'] == 1
    flushed = bf16.compare_bits(np.array([0x3F80, 0x7FFF, 0x0000], np.uint16), golden)
    assert flushed['reason'] == 'bits_differ' and flushed['golden_bits'] == '0x0001'
    assert bf16.compare_bits(np.array([0x3F80, 0x3F80, 0x0001], np.uint16), golden)['reason'] == 'nan_mismatch'
    assert bf16.compare_bits(np.array([0x7FC0, 0x7FFF, 0x0001], np.uint16), golden)['reason'] == 'nan_mismatch'


# ---------------------------------------------------------------- the port
def test_port_maps_the_type_and_the_intrinsics():
    pre = preprocess_file(FIXTURE)
    assert pre.record['bf16'] is True
    assert pre.source.startswith(f'#include "{bf16.HEADER_NAME}"')
    assert '__nv_bfloat16' not in pre.source and '__hfma' not in pre.source
    assert 'cumlx_hfma(s, x[i], y[i])' in pre.source and 'cumlx_float2bfloat16_rz(f[i])' in pre.source
    assert pre.source.count('extern "C"') == 1  # the source has its own extern "C", and the layer adds none


@pytest.mark.parametrize('line, token', [('__nv_bfloat16 q = __hdiv(x[i], y[i]);', '__hdiv'),
                                          ('__nv_bfloat162 q;', '__nv_bfloat162')])
def test_division_and_pairs_are_refused(tmp_path, line, token):
    src = tmp_path / 'k.cu'
    src.write_text(f'__global__ void k(const __nv_bfloat16* x, const __nv_bfloat16* y) {{ int i = threadIdx.x; {line} }}\n')
    with pytest.raises(CumlxError) as e:
        preprocess_file(src)
    assert e.value.code == 'E_KERNEL_UNPORTABLE' and token in [r['token'] for r in e.value.details['refused']]


def test_preprocess_only_writes_the_header(capsys, tmp_path):
    out = tmp_path / 'bf16_ops.cu'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURE), '-o', str(out), '--preprocess-only')
    assert code == 0, doc
    header = tmp_path / bf16.HEADER_NAME
    assert header.read_text() == bf16.HEADER and doc['data']['preprocess']['bf16'] is True
    assert {'W_KERNEL_BF16_EMULATED', 'W_KERNEL_NOT_TRANSLATED'} <= {w['code'] for w in doc['warnings']}
    header.write_text('// changed\n')
    out.unlink()
    (tmp_path / 'bf16_ops.port.json').unlink()
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURE), '-o', str(out), '--preprocess-only')
    assert code != 0 and doc['errors'][0]['code'] == 'E_OUTPUT_EXISTS' and header.read_text() == '// changed\n'


def test_the_capture_harness_declares_bfloat16():
    case = KernelCase('c', 'k', [TensorSpec('x', 'bfloat16', (4,))], [TensorSpec('y', 'bfloat16', (4,))], {}, (4, 1, 1), (4, 1, 1))
    text = generate_harness(case, 'y[threadIdx.x] = x[threadIdx.x];', '')
    assert '#include <cuda_bf16.h>' in text and '__nv_bfloat16* y' in text


def test_the_header_is_added_only_for_a_kernel_that_needs_it():
    source = '#include "cumlx_bf16.h"\nkernel void k(device cumlx_bf16* y [[buffer(0)]]) {}\n'
    text, head, added = bf16.with_metal_header(source, '', {'cumlx_bf16'})
    assert added and '#include "cumlx_bf16.h"' not in text and head.startswith('#ifndef CUMLX_BF16_H')
    assert bf16.with_metal_header('kernel void k() {}', '', {'k'})[2] is False
    own = 'struct cumlx_bf16 { ushort x; };\n' + source
    assert bf16.with_metal_header(own, '', {'cumlx_bf16'})[2] is False


# ---------------------------------------------------------------- on the GPU
def _golden(path: Path, n: int = 4096) -> None:
    rng = np.random.default_rng(11)
    x = rng.integers(0, 1 << 16, n, dtype=np.uint32).astype(np.uint16)
    y = rng.integers(0, 1 << 16, n, dtype=np.uint32).astype(np.uint16)
    k = SPECIAL.size
    x[:k * k], y[:k * k] = np.repeat(SPECIAL, k), np.tile(SPECIAL, k)
    # The pairs where a*b and c nearly cancel: one rounding and two give different results here.
    m = 512
    product = bf16.model_op('mul', np.full(m, 0x3FC0, np.uint16), x[k * k:k * k + m])
    y[k * k:k * k + m] = (product ^ np.uint16(0x8000)) + rng.integers(-2, 3, m).astype(np.uint16)
    f = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32).view(np.float32)
    f[0] = 1.5  # s = 0x3FC0
    s = np.full(n, 0x3FC0, np.uint16)
    with np.errstate(invalid='ignore'):
        less = (bf16._wide(x) < bf16._wide(y)).astype(np.uint32)
    case = KernelCase('bf16_ops.4096', 'bf16_ops', [TensorSpec('x', 'bfloat16', (n,)), TensorSpec('y', 'bfloat16', (n,)), TensorSpec('f', 'float32', (n,))],
                      [TensorSpec('fused', 'bfloat16', (n,)), TensorSpec('sum', 'bfloat16', (n,)), TensorSpec('toward_zero', 'bfloat16', (n,)),
                       TensorSpec('less', 'uint32', (n,))], {}, (n, 1, 1), (256, 1, 1))
    outputs = {'fused': bf16.model_fma(s, x, y), 'sum': bf16.model_op('add', x, y),
               'toward_zero': bf16.from_float_bits(f.view(np.uint32), 'rz'), 'less': less}
    write_golden(path, kind='fixture', source={'framework': 'cumlx.verify.bf16 model'}, kernels={'bf16_ops': {'features': {}}},
                 floating_point={}, cases=[{'case': case, 'inputs': {'x': x, 'y': y, 'f': f}, 'outputs': outputs}])


@pytest.mark.gpu
def test_a_ported_bf16_kernel_matches_the_model_bit_for_bit_and_a_float_emulation_does_not(gpu, capsys, tmp_path, monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    golden = tmp_path / 'golden'
    _golden(golden)
    assert OpGolden(golden).cases[0].outputs[0].dtype == 'bfloat16'
    out = tmp_path / 'right' / 'bf16_ops.metal'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURE), '-o', str(out), '--translator', STUB)
    assert code == 0, doc
    code, doc = run(capsys, 'kernel', 'verify', str(out), str(golden), '--max-items', '-1', '-o', str(tmp_path / 'report'))
    assert code == 0, (doc['summary'], doc['data']['failures'], doc['data']['unverified'])
    assert doc['data']['execution']['engine'] == 'torch' and 'bf16_header_sha256' in doc['data']['execution']
    assert doc['data']['kernels']['bf16_ops']['grade'] is None  # a fixture golden is not CUDA
    assert 'W_KERNEL_BF16_EMULATED' in [w['code'] for w in doc['warnings']]
    record = json.loads((tmp_path / 'report' / 'report.json').read_text())['records'][0]['outputs']
    assert record['fused']['basis'] == 'bit_exact' and record['fused']['nan_in_golden'] > 0

    # The same kernel with the fused multiply-add done in Metal float arithmetic, which flushes subnormals and rounds twice.
    wrong = tmp_path / 'wrong' / 'bf16_ops.metal'
    wrong.parent.mkdir()
    text = out.read_text().replace('cumlx_hfma(s, x[i], y[i])', 'cumlx_bf16(float(s) * float(x[i]) + float(y[i]))')
    assert text != out.read_text()
    wrong.write_text(text)
    code, doc = run(capsys, 'kernel', 'verify', str(wrong), str(golden), '--max-items', '-1')
    assert code == 1 and doc['data']['first_divergence']['output'] == 'fused', doc['summary']
    assert doc['data']['first_divergence']['reason'] == 'bits_differ'


@pytest.mark.gpu
def test_mlx_cannot_bind_bf16_buffers_so_the_case_is_incomplete(gpu, capsys, tmp_path):
    golden = tmp_path / 'golden'
    n = 4
    case = KernelCase('c', 'copy', [TensorSpec('x', 'bfloat16', (n,))], [TensorSpec('y', 'bfloat16', (n,))], {}, (n, 1, 1), (n, 1, 1))
    write_golden(golden, kind='fixture', source={}, kernels={'copy': {}}, floating_point={},
                 cases=[{'case': case, 'inputs': {'x': np.arange(n, dtype=np.uint16)}, 'outputs': {'y': np.arange(n, dtype=np.uint16)}}])
    body = tmp_path / 'copy.metal'
    body.write_text('uint i = thread_position_in_grid.x; y[i] = x[i];\n')
    code, doc = run(capsys, 'kernel', 'verify', str(body), str(golden), '--engine', 'mlx')
    assert code == 5 and doc['data']['unverified']['items'][0]['stage'] == 'bind', doc['summary']
