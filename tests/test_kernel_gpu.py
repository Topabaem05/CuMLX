"""`cumlx kernel verify` on a real Metal GPU, against goldens recorded on a real Tesla V100 (validation/kernel-golden-v100*).

The faults themselves are in tests/test_verify_e2e.py.
"""
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pytest

from cumlx.cli.main import main
from cumlx.errors import CumlxError, ExitCode
from cumlx.verify import kernel_verify
from cumlx.verify.kernel_port import preprocess_file
from cumlx.verify.metrics import compare_kernel_output
from cumlx.verify.opgolden import OpGolden

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'src/cumlx/kernels'
GOLDEN = ROOT / 'validation/kernel-golden-v100'
PREPARED_LAB = ROOT / 'validation/kernel-golden-v100-prepared-lab'
FIXTURE_GOLDEN = ROOT / 'validation/kernel-golden-v100-fixtures'
ORIGINAL_BUILTIN = ROOT / 'validation/kernel-golden-v100-fixtures-original'
FIXTURES = ROOT / 'tests/kernel_fixtures'
KERNELS = ['fp4_decode', 'fp4_gemv', 'fp4_small_m', 'fp4_tiled', 'fp4_w4a4', 'fp4_quantize', 'rms_norm', 'silu_mul']
STUB = 'python:kernel_stub_translator:StubTranslator'


def run(capsys, *argv):
    code = main([*argv, '--format', 'json'])
    return code, json.loads(capsys.readouterr().out)


def test_every_golden_is_from_a_real_cuda_run():
    for path in (GOLDEN, PREPARED_LAB, FIXTURE_GOLDEN, ORIGINAL_BUILTIN):
        g = OpGolden(path)
        assert g.kind == 'cuda' and g.source['gpu'] == 'Tesla V100-SXM2-16GB' and g.source['driver'] == '580.178.04'
        assert g.source['compiler']['name'] == 'nvcc' and g.source['compiler']['version'] == '12.0.140' and g.source['idle_during_capture'] is True
        assert all(g.info(c.id)['repeat_identical'] for c in g.cases)


def test_hand_written_metal_kernels_match_the_v100(gpu, capsys):
    code, doc = run(capsys, 'kernel', 'verify', str(LAB), str(GOLDEN), '--max-items', '-1')
    assert code == 0 and doc['status'] == 'ok', doc['summary']
    data = doc['data']
    assert sorted(data['summary']['grade_a_kernels']) == sorted(KERNELS)
    assert data['summary']['cases'] == {'pass': 29, 'fail': 0, 'unverified': 0}
    assert data['execution']['engine'] == 'mlx' and data['execution']['math_mode'] == 'safe'
    assert {d['topic'] for d in data['execution']['differences']} == {'fp64', 'ftz', 'fmad', 'execution_order'}


def test_the_hand_written_kernels_also_run_through_torch(gpu, capsys):
    pytest.importorskip('torch')
    code, doc = run(capsys, 'kernel', 'verify', str(LAB), str(GOLDEN), '--engine', 'torch', '--repeats', '1', '--max-items', '-1')
    assert doc['data']['execution'].get('engine') == 'torch'
    # PyTorch sets no compile option, so its library runs in the Metal default math mode. It gave the same verdict when measured.
    assert code == 0, (doc['summary'], doc['data']['failures'], doc['data']['unverified'])


def test_ftz_difference_is_measured_and_counted(gpu):
    rep = kernel_verify.verify_kernels(LAB / 'silu_mul.metal', GOLDEN, cases=['silu_mul.subnormal-gates'])
    assert rep['verdict'] == 'ok'
    ftz = next(d for d in rep['execution']['differences'] if d['topic'] == 'ftz')
    assert ftz['cuda']['input_flushes'] is False
    flushed = rep['records'][0]['outputs']['y']['subnormal_flushed']
    in_golden = rep['records'][0]['outputs']['y']['subnormal_in_golden']
    assert in_golden == 16 and ('W_KERNEL_FTZ_DIFFERS' in [w['code'] for w in rep['warnings']]) == (bool(flushed) or ftz['differs'])


def test_nan_scales_decode_to_nan_at_the_same_places_on_both(gpu):
    rep = kernel_verify.verify_kernels(LAB / 'fp4_decode.metal', GOLDEN, cases=['fp4_decode.nvfp4.nan-scale', 'fp4_decode.mxfp4.nan-scale'])
    assert rep['verdict'] == 'ok'
    assert all(r['outputs']['y']['nan_in_golden'] == 512 and r['outputs']['y']['basis'] == 'all_special_values' for r in rep['records'])


def test_a_kernel_that_writes_nothing_cannot_pass_on_stale_memory(gpu, tmp_path):
    """MLX reuses freed buffers. Run the right kernel first, so that a freed buffer holds the right values."""
    engine = kernel_verify.MLXEngine()
    assert kernel_verify.verify_kernels(LAB / 'silu_mul.metal', GOLDEN, cases=['silu_mul.float32'], engine=engine)['verdict'] == 'ok'
    idle = tmp_path / 'silu_mul.metal'
    idle.write_text('unsigned int i = thread_position_in_grid.x;\nif (i >= SIZE + 1u) { y[0] = float(gate[0]) + float(up[0]); }\n')
    rep = kernel_verify.verify_kernels(idle, GOLDEN, cases=['silu_mul.float32'], header=str(LAB / 'common_metal.h'), engine=engine)
    assert rep['verdict'] == 'diverged', rep['records'][0]
    assert rep['first_divergence']['reason'] == 'nan_mismatch'


def test_a_translation_that_does_not_compile_is_incomplete(gpu, capsys, tmp_path):
    broken = tmp_path / 'silu_mul.metal'
    broken.write_text('this is not metal;\n')
    code, doc = run(capsys, 'kernel', 'verify', str(broken), str(GOLDEN), '--case', 'silu_mul.float32')
    assert code == ExitCode.INCOMPLETE and doc['status'] == 'incomplete'
    assert doc['data']['unverified']['items'][0]['stage'] == 'compile'
    assert doc['data']['kernels']['silu_mul']['grade'] is None


def test_a_library_with_the_wrong_buffers_is_incomplete(gpu, capsys, tmp_path):
    pytest.importorskip('torch')
    lib = tmp_path / 'silu_mul.metal'
    lib.write_text('#include <metal_stdlib>\nusing namespace metal;\nkernel void silu_mul(const constant float* gate [[buffer(0)]], const constant float* up [[buffer(1)]]) { }\n')
    code, doc = run(capsys, 'kernel', 'verify', str(lib), str(GOLDEN), '--case', 'silu_mul.float32')
    assert code == ExitCode.INCOMPLETE and doc['data']['unverified']['items'][0]['stage'] == 'bind'
    assert 'no buffer for output' in doc['data']['unverified']['items'][0]['reason']


def test_changed_golden_is_refused(gpu, capsys, tmp_path):
    copy = tmp_path / 'g'
    shutil.copytree(GOLDEN, copy)
    victim = copy / OpGolden(copy).info('silu_mul.float32')['file']
    data = bytearray(victim.read_bytes())
    data[-3] ^= 0x10
    victim.write_bytes(bytes(data))
    code, doc = run(capsys, 'kernel', 'verify', str(LAB / 'silu_mul.metal'), str(copy), '--case', 'silu_mul.float32')
    assert code == ExitCode.INPUT and 'SHA-256' in doc['errors'][0]['message']


# ---------------------------------------------------------------- the pre-translation layer on real hardware
def test_prepared_lab_kernels_gave_the_same_bytes_on_the_v100():
    """Each lab kernel, wrapped by the pre-translation layer and compiled by nvcc, against its original."""
    lab, prepared = OpGolden(GOLDEN), OpGolden(PREPARED_LAB)
    assert sorted(c.kernel for c in prepared.cases) == sorted(KERNELS)
    for case in prepared.cases:
        _, ours = prepared.load(case)
        _, original = lab.load(next(c for c in lab.cases if c.id == case.id))
        for name in ours:
            assert ours[name].tobytes() == original[name].tobytes(), (case.id, name)


def test_rewritten_builtins_match_the_originals_on_the_v100():
    prepared = OpGolden(FIXTURE_GOLDEN)
    original = OpGolden(ORIGINAL_BUILTIN)
    _, ours = prepared.load(next(c for c in prepared.cases if c.kernel == 'builtin_map'))
    _, theirs = original.load(original.cases[0])
    assert np.array_equal(ours['ints'], theirs['ints'])  # integer built-ins are exact
    r = compare_kernel_output(ours['out'], theirs['out'])
    assert r['verdict'] == 'pass' and r['max_ulp'] <= 1


def test_template_and_mixed_kernels_compute_the_expected_values_on_the_v100():
    golden = OpGolden(FIXTURE_GOLDEN)
    case = next(c for c in golden.cases if c.kernel == 'vector_add')
    inputs, outputs = golden.load(case)
    np.testing.assert_array_equal(outputs['out'], inputs['a'] + inputs['b'])
    case = next(c for c in golden.cases if c.kernel == 'norm_mixed')
    inputs, outputs = golden.load(case)
    x = inputs['x'].astype(np.float64).reshape(2, 4) * 2.5
    expected = (x / np.sqrt((x ** 2).sum(axis=1, keepdims=True))).ravel()
    np.testing.assert_allclose(outputs['out'], expected, atol=2e-6, rtol=2e-6)


# ---------------------------------------------------------------- the whole pipeline, with a stub for the third-party step
@pytest.mark.parametrize('kernel', KERNELS)
def test_stub_translation_of_a_lab_kernel_equals_the_hand_written_one(gpu, capsys, tmp_path, monkeypatch, kernel):
    """Source -> pre-translation -> stub -> `verify` against the V100. It proves the wiring, and it says nothing about woof."""
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    out = tmp_path / f'{kernel}.metal'
    code, doc = run(capsys, 'kernel', 'port', str(LAB / f'{kernel}.cu'), '-o', str(out), '--translator', STUB)
    assert code == 0, doc
    code, doc = run(capsys, 'kernel', 'verify', str(out), str(GOLDEN), '--max-items', '-1')
    assert code == 0 and doc['data']['kernels'][kernel]['grade'] == 'A', doc['summary']
    assert doc['data']['translations'][kernel]['translator']['name'] == 'test-stub'

    def normal(text):
        return re.sub(r'\s+', ' ', re.sub(r'//[^\n]*', '', text)).strip()

    assert normal(out.read_text()) == normal((LAB / f'{kernel}.metal').read_text())


@pytest.mark.parametrize('kernel', ['vector_add', 'norm_mixed', 'builtin_map'])
def test_stub_translation_of_a_test_kernel_matches_the_v100(gpu, capsys, tmp_path, monkeypatch, kernel):
    """Template kernels, host code and built-ins, through a complete Metal library and torch.mps.compile_shader."""
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    out = tmp_path / f'{kernel}.metal'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURES / f'{kernel}.cu'), '-o', str(out), '--translator', STUB)
    assert code == 0, doc
    code, doc = run(capsys, 'kernel', 'verify', str(out), str(FIXTURE_GOLDEN))
    assert doc['data']['execution']['engine'] == 'torch'
    assert code == 0 and doc['data']['kernels'][kernel]['grade'] == 'A', (doc['summary'], doc['data'].get('failures'), doc['data'].get('unverified'))
    assert 'W_KERNEL_SOURCE_MISMATCH' not in [w['code'] for w in doc['warnings']]


def test_wrapped_case_goes_through_the_stub_to_the_v100(gpu, capsys, tmp_path, monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    cases = str(ROOT / 'validation/kernel-cases/cases.json')
    case = 'rms_norm.plain.4x129'
    out = tmp_path / 'rms_norm.metal'
    code, doc = run(capsys, 'kernel', 'port', str(LAB / 'rms_norm.cu'), '-o', str(out), '--translator', STUB, '--cases', cases, '--case', case)
    assert code == 0, doc
    code, doc = run(capsys, 'kernel', 'verify', str(out), str(GOLDEN), '--case', case)
    assert code == 0, (doc['summary'], doc['data'].get('unverified'), doc['data'].get('failures'))
    assert doc['data']['execution']['engine'] == 'torch'


def test_preprocessing_is_stable_on_the_committed_prepared_sources():
    """If this fails, the V100 evidence for the pre-translation layer no longer covers the current code."""
    for kernel in ('vector_add', 'norm_mixed', 'builtin_map'):
        expected = OpGolden(FIXTURE_GOLDEN).kernels[kernel]['source_sha256']
        import hashlib
        assert hashlib.sha256(preprocess_file(FIXTURES / f'{kernel}.cu').source.encode()).hexdigest() == expected, kernel


def test_a_missing_kernel_is_not_covered_by_a_folder(gpu, capsys, tmp_path):
    shutil.copy(LAB / 'silu_mul.metal', tmp_path)
    shutil.copy(LAB / 'common_metal.h', tmp_path)
    code, doc = run(capsys, 'kernel', 'verify', str(tmp_path), str(GOLDEN))
    assert code == ExitCode.INCOMPLETE and len(doc['data']['summary']['not_covered']) == 7
    assert doc['data']['kernels']['silu_mul']['status'] == 'pass'


def test_no_cuda_golden_means_no_grade_even_if_values_match(gpu, tmp_path):
    copy = tmp_path / 'g'
    shutil.copytree(GOLDEN, copy)
    manifest = json.loads((copy / 'manifest.json').read_text())
    manifest['golden']['kind'] = 'fixture'
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    rep = kernel_verify.verify_kernels(LAB / 'silu_mul.metal', copy, cases=['silu_mul.float32'])
    assert rep['verdict'] == 'ok' and rep['kernels']['silu_mul']['grade'] is None


def test_a_golden_that_lies_about_its_origin_is_rejected_for_missing_provenance(tmp_path):
    copy = tmp_path / 'g'
    shutil.copytree(GOLDEN, copy)
    manifest = json.loads((copy / 'manifest.json').read_text())
    del manifest['source']['driver']
    (copy / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(CumlxError, match='source.driver'):
        OpGolden(copy)
