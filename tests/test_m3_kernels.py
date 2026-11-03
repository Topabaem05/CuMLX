"""The M3 path without third-party code: call and initial, the woof engine on a stub, the driver, the gates and the launcher.

No test here installs woof, reads a real zoo kernel, or uses a GPU. The stub package is not woof, so a pass here says only
that the wiring holds. validation/kernel-m3.json says what ran for real.
"""
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

import numpy as np
import pytest

from cumlx.cli.main import main
from cumlx.errors import CumlxError, ExitCode
from cumlx.verify import kernel_port, kernel_verify
from cumlx.verify.kernel_capture import generate_harness, wrap_body
from cumlx.verify.kernel_port import preprocess_file
from cumlx.verify.kernel_source import is_complete_metal
from cumlx.verify.opgolden import KernelCase, OpGolden, TensorSpec, parse_case, read_cases, write_golden

ROOT = Path(__file__).resolve().parents[1]
SPIKE = ROOT / 'spikes' / 'm3_kernels.py'
LAUNCHER = ROOT / 'spikes' / 'm3_kernels.sh'
CASES = ROOT / 'validation/kernel-cases-m3'


@pytest.fixture(scope='module')
def driver():
    spec = importlib.util.spec_from_file_location('m3_kernels', SPIKE)
    module = importlib.util.module_from_spec(spec)
    sys.modules['m3_kernels'] = module
    spec.loader.exec_module(module)
    return module


def saxpy_case(**extra):
    return {'id': 's', 'kernel': 'saxpy', 'entry': 'saxpy<float>', 'grid': [256, 1, 1], 'threadgroup': [256, 1, 1],
            'inputs': [{'name': 'n', 'dtype': 'int32', 'shape': []}, {'name': 'a', 'dtype': 'float32', 'shape': []},
                       {'name': 'x', 'dtype': 'float32', 'shape': [3]}, {'name': 'y_in', 'dtype': 'float32', 'shape': [3]}],
            'outputs': [{'name': 'y', 'dtype': 'float32', 'shape': [3]}], 'call': ['n', 'a', 'x', 'y'], 'initial': {'y': 'y_in'}, **extra}


# ---------------------------------------------------------------- call and initial
def test_call_and_initial_survive_a_round_trip_and_set_the_arguments():
    case = parse_case(saxpy_case())
    assert case.arguments == ('n', 'a', 'x', 'y') and case.initial == {'y': 'y_in'}
    assert parse_case(case.to_dict()).to_dict() == case.to_dict()
    plain = KernelCase('p', 'k', [TensorSpec('x', 'float32', (3,))], [TensorSpec('y', 'float32', (3,))])
    assert plain.arguments == ('x', 'y') and 'call' not in plain.to_dict() and 'initial' not in plain.to_dict()


@pytest.mark.parametrize('change, message', [
    ({'call': ['n', 'a', 'x']}, 'call must name'),
    ({'call': ['n', 'a', 'x', 'y', 'y']}, 'call must name'),
    ({'call': ['n', 'a', 'x', 'y_in', 'y']}, 'call must name'),
    ({'initial': {'y': 'nothing'}}, 'initial maps'),
    ({'initial': {'nothing': 'y_in'}}, 'initial maps'),
    ({'initial': {'y': 'x'}, 'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': [4]}], 'call': ['x', 'y']}, 'initial:'),
])
def test_a_wrong_call_or_initial_is_refused(change, message):
    with pytest.raises(CumlxError, match=message):
        parse_case(saxpy_case(**change))


def test_harness_passes_arguments_in_the_kernel_order_and_refills_an_inout_buffer():
    case = parse_case(saxpy_case())
    harness = generate_harness(case, 'template <typename T> __global__ void saxpy(int n, float a, const T* x, T* y) {}', '')
    assert 'saxpy<float><<<dim3(1, 1, 1), dim3(256, 1, 1)>>>(dev_n, dev_a, dev_x, dev_y);' in harness
    assert 'cudaMemcpy(dev_y, host_y_in.data(), 12, cudaMemcpyHostToDevice)' in harness and 'cudaMemset(dev_y' not in harness
    assert 'dev_y_in' not in harness  # it only fills y
    plain = KernelCase('p', 'k', [TensorSpec('x', 'float32', (3,))], [TensorSpec('y', 'float32', (3,))])
    harness = generate_harness(plain, 'y[0] = x[0];', '')
    assert 'k<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(dev_x, dev_y);' in harness and 'cudaMemset(dev_y' in harness


def test_a_body_is_wrapped_as_before_when_no_call_is_set():
    plain = KernelCase('p', 'k', [TensorSpec('s', 'int32', ()), TensorSpec('x', 'float32', (3,))], [TensorSpec('y', 'float32', (3,))])
    assert wrap_body(plain, 'B') == '__global__ void k(const int s, const float* x, float* y) {\nB\n}\n'


def test_zero_dimensional_inputs_survive_the_golden(tmp_path):
    case = parse_case(saxpy_case())
    arrays = {'n': np.array(3, np.int32), 'a': np.array(2.5, np.float32), 'x': np.ones(3, np.float32), 'y_in': np.ones(3, np.float32)}
    write_golden(tmp_path, kind='fixture', source={}, kernels={'saxpy': {}}, floating_point={},
                 cases=[{'case': case, 'inputs': arrays, 'outputs': {'y': np.ones(3, np.float32)}}])
    golden = OpGolden(tmp_path)
    inputs, _ = golden.load(golden.cases[0])
    assert inputs['n'].shape == () and int(inputs['n']) == 3 and golden.cases[0].call == ('n', 'a', 'x', 'y')


def test_the_committed_cases_read_back_and_cover_every_kernel(driver):
    cases = read_cases(CASES / 'cases.json')
    doc = driver.sources()
    assert {c.kernel for c, _ in cases} == {*doc['zoo'], driver.SAM2} and len(doc['zoo']) == 12
    assert all(c.entry and c.call for c, _ in cases)
    for case, arrays in cases:
        assert all(a.dtype != np.float64 for a in arrays.values())
    entries = {c.entry for c, _ in cases if c.kernel == driver.SAM2}
    assert entries == {'init_labeling', 'merge', 'compression', 'final_labeling', 'init_counting', 'final_counting'}


def test_no_third_party_kernel_source_is_in_the_repository():
    tracked = subprocess.run(['git', 'ls-files', 'validation/kernel-cases-m3'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert not [p for p in tracked if p.endswith(('.cu', '.cuh', '.inl'))]


# ---------------------------------------------------------------- launch.json
def test_a_launch_expression_is_read_without_eval(driver):
    assert driver.evaluate('min(cdiv(n, 256), 1024)', {'n': 1000}) == 4
    assert driver.evaluate('head_dim // 2', {'head_dim': 8}) == 4 and driver.evaluate('1e-5', {}) == 1e-5 and driver.evaluate(7, {}) == 7
    for bad in ("__import__('os')", 'n.real', 'open(n)', 'n ** 9'):
        with pytest.raises(ValueError):
            driver.evaluate(bad, {'n': 1})


def test_an_inout_parameter_becomes_an_initial_and_a_big_case_stays_out(driver):
    spec = {'entry': 'k<{T}>', 'launch': {'grid': ['cdiv(n, 4)'], 'block': [4]},
            'params': [{'name': 'n', 'kind': 'scalar', 'dtype': 'int32', 'value': 'n'}, {'name': 'x', 'kind': 'input', 'dtype': 'T', 'shape': ['n']},
                       {'name': 'y', 'kind': 'inout', 'dtype': 'T', 'shape': ['n']}, {'name': 'z', 'kind': 'output', 'dtype': 'T', 'shape': ['n']}],
            'init': {'x': {'dist': 'normal', 'edges': True}, 'y': {'dist': 'zeros'}},
            'cases': [{'name': 'small', 'dims': {'n': 9}}, {'name': 'big', 'dims': {'n': driver.MAX_ELEMENTS + 1}}]}
    cases, tensors = driver.zoo_cases('k', spec)
    assert [c['id'] for c in cases] == ['zoo.k.small']
    case = parse_case({**cases[0], 'inputs': [{k: v for k, v in t.items() if k in ('name', 'dtype', 'shape')} for t in cases[0]['inputs']]})
    assert case.entry == 'k<float>' and case.call == ('n', 'x', 'y', 'z') and case.initial == {'y': 'y_in'}
    assert case.grid == (12, 1, 1) and case.threadgroup == (4, 1, 1)
    assert tensors['zoo.k.small/x'][:4].tolist() == [0.0, 0.0, 1.0, -1.0] and tensors['zoo.k.small/x'].dtype == np.float32


# ---------------------------------------------------------------- sam2's inputs
def _components(img):
    """8-connected components by a plain search: a reference that shares no code with the driver."""
    h, w = img.shape
    seen = np.zeros(img.shape, bool)
    out = 0
    for r in range(h):
        for c in range(w):
            if img[r, c] and not seen[r, c]:
                out += 1
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    a, b = stack.pop()
                    for i in range(max(a - 1, 0), min(a + 2, h)):
                        for j in range(max(b - 1, 0), min(b + 2, w)):
                            if img[i, j] and not seen[i, j]:
                                seen[i, j] = True
                                stack.append((i, j))
    return out


def test_the_forest_for_merge_cannot_depend_on_thread_order(driver):
    for h, w in ((32, 32), (48, 64)):
        start = driver.cc_init(h, w)
        forest = driver.cc_merge(driver.cc_sparse_image(h, w), start)
        moved = forest != start
        pairs = int(moved.sum())
        assert pairs == _components(driver.cc_sparse_image(h, w)) and pairs >= 4
        # every changed block points at a root, so no longer chain exists and no union had a choice
        assert all(forest.reshape(-1)[v] == v for v in forest[moved])
    # A wrong image: three blocks in a row join through a chain, which is the case that the test must see.
    img = np.zeros((16, 16), np.uint8)
    img[2, 1:6] = 1
    assert (driver.cc_merge(img, driver.cc_init(16, 16)) != driver.cc_init(16, 16)).sum() == 2


def test_the_serial_pipeline_gives_the_components_of_a_dense_image(driver):
    img = driver.cc_dense_image(32, 32, 7)
    final = driver.cc_final(img, driver.cc_compress(driver.cc_merge(img, driver.cc_init(32, 32))))
    assert len(set(final[final > 0].tolist())) == _components(img)
    assert ((final > 0) == (img > 0)).all()
    assert driver.cc_count(final).sum() == int(img.sum())
    wrong = final.copy()
    wrong[wrong == wrong.max()] = 1
    assert len(set(wrong[wrong > 0].tolist())) != _components(img)


# ---------------------------------------------------------------- the gates
ZOO = ['cumsum', 'embedding_gather', 'layernorm', 'matmul_naive', 'reduce_sum', 'relu', 'rmsnorm', 'rope', 'saxpy', 'softmax', 'transpose', 'vector_add']
LAB = ['fp4_decode', 'fp4_gemv', 'fp4_small_m', 'fp4_tiled', 'fp4_w4a4', 'fp4_quantize', 'rms_norm', 'silu_mul']
GOOD = {'translate': {'stage': 'translated'}, 'compare': {'stage': 'compared', 'status': 'pass', 'grade': 'A'}}
BAD = {'translate': {'stage': 'translated'}, 'compare': {'stage': 'compared', 'status': 'fail', 'grade': None}}


def full_results(group_results=None):
    results = {k: {'group': 'zoo', **GOOD} for k in ZOO}
    results.update({k: {'group': 'lab', **GOOD} for k in LAB})
    results['connected_components'] = {'group': 'sam2', **GOOD}
    results.update(group_results or {})
    return results


def test_nothing_run_meets_no_gate(driver):
    gates = driver.gates({}, 8, [])
    assert not any(g['met'] for g in gates.values())
    assert gates['goldens_20']['count'] == 8 and gates['one_repository']['result'] == 'not_run'


def test_stubs_and_a_missing_golden_meet_no_gate(driver):
    results = {k: {'group': 'zoo', 'translate': {'stage': 'translated'}, 'compare': {'stage': 'no_golden'}} for k in ZOO}
    gates = driver.gates(results, 8, [])
    assert not gates['zoo_12_compared']['met'] and gates['zoo_12_compared']['compared'] == 0
    assert gates['zoo_12_compared']['not_compared']['relu'] == 'no_golden'


def test_every_kernel_grade_a_meets_every_gate(driver):
    gates = driver.gates(full_results(), 8, [*ZOO, 'connected_components'])
    assert all(g['met'] for g in gates.values()) and gates['goldens_20']['count'] == 21


def test_seventy_percent_counts_refused_kernels_in_the_denominator(driver):
    # 14 of 21 is 66.7%, under the gate. The seven others failed for different reasons, and none leaves the denominator.
    results = full_results({'rmsnorm': {'group': 'zoo', 'translate': {'stage': 'refused'}},
                            'layernorm': {'group': 'zoo', 'translate': {'stage': 'translate'}},
                            'rope': {'group': 'zoo', **BAD}, 'cumsum': {'group': 'zoo', **BAD}, 'relu': {'group': 'zoo', **BAD},
                            'saxpy': {'group': 'zoo', **BAD}, 'softmax': {'group': 'zoo', **BAD}})
    gates = driver.gates(results, 8, [*ZOO, 'connected_components'])
    assert gates['grade_a_70_percent']['grade_a'] == 14 and not gates['grade_a_70_percent']['met']
    assert gates['goldens_20']['met'] and gates['one_repository']['met']
    assert gates['grade_a_70_percent']['by_reason']['rmsnorm'] == 'refused'
    # A compared kernel that diverged still counts as compared. A refused one does not.
    assert gates['zoo_12_compared']['compared'] == 10 and not gates['zoo_12_compared']['met']
    # 15 of 21 is 71.4%, over the gate
    results['saxpy'] = {'group': 'zoo', **GOOD}
    assert driver.gates(results, 8, [*ZOO, 'connected_components'])['grade_a_70_percent']['met']


def test_a_diverged_sam2_does_not_meet_the_repository_gate(driver):
    gates = driver.gates(full_results({'connected_components': {'group': 'sam2', **BAD}}), 8, [*ZOO, 'connected_components'])
    assert not gates['one_repository']['met'] and gates['one_repository']['result'] == 'diverged'


def test_lab_kernel_passes_only_if_every_case_passes(driver):
    good = {'status': 'pass', 'passed': 1, 'cases': 1, 'worst_rel_l2': 1e-7, 'grade': 'A', 'stage': 'compared'}
    assert driver.merge_lab([good, good])['grade'] == 'A'
    merged = driver.merge_lab([good, {**good, 'status': 'fail', 'grade': None}])
    assert merged['status'] == 'fail' and merged['cases_passed'] == 1 and merged['cases_tried'] == 2


# ---------------------------------------------------------------- warp-level built-ins
SHUFFLE = '''__global__ void k(const float* x, float* y) {
    float v = x[threadIdx.x];
    y[threadIdx.x] = v + __shfl_xor_sync(0xffffffff, v, 1);
}
'''


def test_warp_level_is_refused_by_default_and_passed_only_on_request(tmp_path):
    path = tmp_path / 'k.cu'
    path.write_text(SHUFFLE)
    with pytest.raises(CumlxError) as caught:
        preprocess_file(path)
    assert caught.value.code == 'E_KERNEL_UNPORTABLE' and caught.value.details['refused'][0]['feature'] == 'warp-level'
    pre = preprocess_file(path, allow_warp_level=True)
    assert '__shfl_xor_sync(0xffffffff, v, 1)' in pre.source
    assert [w['tokens'] for w in pre.warnings if w['code'] == 'W_KERNEL_WARP_LEVEL_PASSED'] == [['__shfl_xor_sync']]


def test_the_flag_does_not_let_other_refusals_through(tmp_path):
    path = tmp_path / 'k.cu'
    path.write_text('__global__ void k(const float* x, float* y) { y[threadIdx.x] = __frcp_rz(x[0]); }\n')
    with pytest.raises(CumlxError) as caught:
        preprocess_file(path, allow_warp_level=True)
    assert caught.value.details['refused'][0]['feature'] == 'rounding-mode'


# ---------------------------------------------------------------- the woof engine, on a stub package that is not woof
STUB_TRANSLATE = '''
from dataclasses import dataclass, field
@dataclass(frozen=True)
class Param:
    name: str
    pointer: bool
    ctype: str
    const: bool = False
    fmt: str = None
    size: int = 0
    offset: int = 0
    global_name: str = None
@dataclass(frozen=True)
class KernelSignature:
    name: str
    params: tuple
    dynamic_shared: bool = False
    pointer_mode: str = 'direct'
    table_slot: int = -1
    scalar_slot: int = -1
    scalar_size: int = 0
    indirect: bool = False
@dataclass
class TranslatedUnit:
    msl: str
    kernels: dict = field(default_factory=dict)
    key: str = ''
    globals: dict = field(default_factory=dict)
def translate(source, unit, include_dirs=()):
    sig = KernelSignature('saxpy', (Param('n', False, 'int', fmt='i', size=4), Param('a', False, 'float', fmt='f', size=4, offset=4),
                                    Param('x', True, 'float', const=True), Param('y', True, 'float')), scalar_slot=2, scalar_size=8)
    return TranslatedUnit('[[kernel]] void saxpy() {}\\n', {'saxpy': sig}, 'k')
'''
STUB_LOADER = '''
import numpy as np
launched = []
class Function:
    def __init__(self, unit):
        self.unit = unit
    def __call__(self, grid, block, args):
        launched.append((grid, block))
        assert [p.name for p in self.unit.kernels['saxpy'].params] == ['n', 'a', 'x', 'y']
        n, a, x, y = args
        sign = -1 if 'WRONG' in self.unit.msl else 1
        if 'SPIN' in self.unit.msl:
            raise RuntimeError('the GPU stopped the command')
        y[:n] = sign * a * x[:n] + y[:n]
class MetalModule:
    def __init__(self, unit, name):
        if 'NOCOMPILE' in unit.msl:
            raise RuntimeError('Metal did not compile the translation')
        self.unit = unit
    def get_function(self, name):
        assert name in self.unit.kernels
        return Function(self.unit)
'''


@pytest.fixture
def stub_woof(tmp_path, monkeypatch):
    base = tmp_path / 'stubs'
    (base / 'woof/backend/metal').mkdir(parents=True)
    (base / 'woof/__init__.py').write_text('')
    (base / 'woof/backend/__init__.py').write_text('')
    (base / 'woof/backend/translate.py').write_text(STUB_TRANSLATE)
    (base / 'woof/backend/metal/__init__.py').write_text('')
    (base / 'woof/backend/metal/loader.py').write_text(STUB_LOADER)
    (base / 'cupy.py').write_text('import numpy as np\nasarray = np.array\nasnumpy = np.array\n')
    monkeypatch.syspath_prepend(str(base))
    for name in [m for m in sys.modules if m == 'woof' or m.startswith('woof.') or m == 'cupy']:
        monkeypatch.delitem(sys.modules, name)

    class FakeBody:
        def describe(self):
            return {'device': 'stub'}

        def run(self, case, source, header, inputs):
            if case.kernel == 'cumlx_probe':
                return {'y': np.array([1e-40, 1e-40, 1e-40, 7e-40, 6e-8], np.float32)}
            return {'y': np.array([32], np.float32)}
    monkeypatch.setattr(kernel_verify, 'MLXEngine', FakeBody)
    yield base
    for name in [m for m in sys.modules if m == 'woof' or m.startswith('woof.') or m == 'cupy']:
        del sys.modules[name]


def saxpy_golden(path, kind='fixture'):
    case = parse_case(saxpy_case())
    x, y0 = np.array([1, 2, 3], np.float32), np.array([10, 20, 30], np.float32)
    arrays = {'n': np.array(3, np.int32), 'a': np.array(2.0, np.float32), 'x': x, 'y_in': y0}
    source = {'framework': 'test'}
    if kind == 'cuda':
        source.update(gpu='Test GPU', driver='0', compiler={'name': 'nvcc', 'version': '0'})
    write_golden(path, kind=kind, source=source, kernels={'saxpy': {'source_sha256': 'aaa', 'features': {}}}, floating_point={},
                 cases=[{'case': case, 'inputs': arrays, 'outputs': {'y': 2 * x + y0}, 'repeat_identical': True}])
    return path


def translated(folder, text='[[kernel]] void saxpy() {}\n'):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'saxpy.metal').write_text(text)
    translate = importlib.import_module('woof.backend.translate')
    sig = {k: dataclasses.asdict(v) for k, v in translate.translate('', 'saxpy').kernels.items()}
    (folder / 'saxpy.port.json').write_text(json.dumps({'metal_signatures': sig}))
    return folder / 'saxpy.metal'


def test_woof_engine_passes_a_right_kernel_with_scalars_and_an_inout_buffer(stub_woof, tmp_path):
    golden = saxpy_golden(tmp_path / 'g', kind='cuda')
    rep = kernel_verify.verify_kernels(translated(tmp_path / 'm'), golden, engine_name='woof', repeats=2)
    assert rep['verdict'] == 'ok' and rep['kernels']['saxpy']['grade'] == 'A' and rep['execution']['engine'] == 'woof'
    assert sys.modules['woof.backend.metal.loader'].launched[0] == ((1, 1, 1), (256, 1, 1))


def test_woof_engine_fails_a_wrong_kernel(stub_woof, tmp_path):
    golden = saxpy_golden(tmp_path / 'g', kind='cuda')
    rep = kernel_verify.verify_kernels(translated(tmp_path / 'm', '[[kernel]] void saxpy() {} // WRONG\n'), golden, engine_name='woof')
    assert rep['verdict'] == 'diverged' and rep['kernels']['saxpy']['grade'] is None


@pytest.mark.parametrize('text, stage', [('[[kernel]] void saxpy() {} // NOCOMPILE\n', 'compile'), ('[[kernel]] void saxpy() {} // SPIN\n', 'run')])
def test_woof_engine_failure_is_incomplete_and_not_a_pass(stub_woof, tmp_path, text, stage):
    golden = saxpy_golden(tmp_path / 'g', kind='cuda')
    rep = kernel_verify.verify_kernels(translated(tmp_path / 'm', text), golden, engine_name='woof')
    assert rep['verdict'] == 'incomplete' and rep['records'][0]['stage'] == stage and rep['kernels']['saxpy']['grade'] is None


def test_woof_engine_needs_the_signature_from_the_port_record(stub_woof, tmp_path):
    golden = saxpy_golden(tmp_path / 'g', kind='cuda')
    metal = translated(tmp_path / 'm')
    (tmp_path / 'm/saxpy.port.json').unlink()
    rep = kernel_verify.verify_kernels(metal, golden, engine_name='woof')
    assert rep['verdict'] == 'incomplete' and rep['records'][0]['stage'] == 'bind'


def test_engine_woof_without_the_package_is_exit_3(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(kernel_verify.importlib.util, 'find_spec', lambda name: None)
    golden = saxpy_golden(tmp_path / 'g')
    metal = tmp_path / 'saxpy.metal'
    metal.write_text('[[kernel]] void saxpy() {}\n')
    code = main(['kernel', 'verify', str(metal), str(golden), '--engine', 'woof', '--format', 'json'])
    doc = json.loads(capsys.readouterr().out)
    assert code == ExitCode.ENVIRONMENT and doc['errors'][0]['code'] == 'E_KERNEL_TRANSLATOR_MISSING' and doc['ok'] is False


@pytest.mark.parametrize('text, complete', [('[[kernel]] void saxpy() {}', True), ('static x;\n[[ kernel ]] void f() {}', True), ('kernel void f() {}', True),
                                            ('y[0] = x[0];', False), ('// [[kernel]] void f()\ny[0] = 1;', False), ('mykernel void f;', False)])
def test_a_library_with_the_attribute_form_is_a_complete_library(text, complete):
    assert is_complete_metal(text) is complete


def test_mlx_engine_refuses_a_case_that_it_cannot_bind():
    engine = object.__new__(kernel_verify.MLXEngine)
    with pytest.raises(kernel_verify.KernelRunError, match='call or initial'):
        engine.run(parse_case(saxpy_case()), 'y[0] = 1;', '', {})


def test_the_adapter_keeps_the_signature_for_the_engine(stub_woof, tmp_path, monkeypatch):
    monkeypatch.setenv('WOOF_BACKEND', 'mlx')
    path = tmp_path / 'saxpy.cu'
    path.write_text('__global__ void saxpy(int n, float a, const float* x, float* y) { y[threadIdx.x] = a * x[threadIdx.x] + y[threadIdx.x]; }\n')
    pre = preprocess_file(path)
    result = kernel_port.translate(pre, kernel_port.load_translator('woof'))
    assert [p['name'] for p in result.signatures['saxpy']['params']] == ['n', 'a', 'x', 'y'] and result.signatures['saxpy']['scalar_size'] == 8
    assert json.dumps(result.signatures)  # it goes in a JSON file


# ---------------------------------------------------------------- the launcher, in a copy that has only fakes around it
def fake(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\n' + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def sandbox(tmp_path):
    """A repository copy with no real driver and no real Python. `tools` proves that the fake is what would run."""
    repo = tmp_path / 'repo'
    (repo / 'spikes').mkdir(parents=True)
    shutil.copy(LAUNCHER, repo / 'spikes/m3_kernels.sh')
    log = tmp_path / 'fake.log'
    fake(repo / '.venv/bin/python', f'echo "FAKE_REPO_PY $*" >> "{log}"\n')
    uv = fake(tmp_path / 'bin/fake-uv', f'''echo "FAKE_UV $*" >> "{log}"
if [ "$1" = venv ]; then
  mkdir -p "$5/bin"
  printf '#!/bin/sh\\necho "FAKE_VENV_PY $*" >> "{log}"\\n' > "$5/bin/python"
  chmod +x "$5/bin/python"
fi
''')
    capture = fake(tmp_path / 'bin/fake-capture', f'echo "FAKE_CAPTURE $*" >> "{log}"\nexit "${{FAKE_CAPTURE_EXIT:-0}}"\n')
    root = tmp_path / 'root'
    env = {**os.environ, 'CUMLX_M3_ROOT': str(root), 'CUMLX_M3_UV': str(uv), 'CUMLX_M3_CAPTURE': str(capture), 'CUMLX_GPU_HOST': 'fake-host'}
    return {'repo': repo, 'log': log, 'env': env, 'root': root, 'uv': uv, 'capture': capture}


def launch(sandbox, *args, env=None):
    return subprocess.run(['sh', str(sandbox['repo'] / 'spikes/m3_kernels.sh'), *args], env={**sandbox['env'], **(env or {})},
                          capture_output=True, text=True, timeout=30)


def wait_for(sandbox, marker):
    deadline = time.time() + 20
    while time.time() < deadline:
        text = (sandbox['root'] / 'run.log').read_text() if (sandbox['root'] / 'run.log').exists() else ''
        if marker in text:
            return text
        time.sleep(0.05)
    raise AssertionError(f'{marker!r} not in the log: {text}')


def test_tools_names_the_fakes_before_anything_runs(sandbox):
    done = launch(sandbox, 'tools')
    assert done.returncode == 0
    assert f"uv {sandbox['uv']}" in done.stdout and f"capture {sandbox['capture']}" in done.stdout
    assert f"python {sandbox['root']}/woof-venv/bin/python" in done.stdout
    assert not sandbox['log'].exists()


def test_start_returns_at_once_and_runs_the_stages_in_order_in_the_new_environment(sandbox):
    assert f"uv {sandbox['uv']}" in launch(sandbox, 'tools').stdout  # the stub is the program, or nothing below proves anything
    done = launch(sandbox)
    assert done.returncode == 0 and 'Starting the M3 translation' in done.stdout and 'pid ' in done.stdout
    wait_for(sandbox, 'finished, exit code 0')
    lines = sandbox['log'].read_text().splitlines()
    assert lines[0].startswith('FAKE_UV venv') and lines[1].startswith('FAKE_UV pip install') and 'metal-woof==1.1.5.2' in lines[1]
    assert [line.split()[2] for line in lines[2:]] == ['fetch', 'cases', 'run'] and all(line.startswith('FAKE_VENV_PY') for line in lines[2:])
    assert 'FAKE_REPO_PY' not in sandbox['log'].read_text()
    assert 'finished' in launch(sandbox, 'status').stdout


def test_capture_uses_the_repository_python_and_stops_when_the_remote_script_stops(sandbox):
    done = launch(sandbox, 'capture', env={'FAKE_CAPTURE_EXIT': '3'})
    assert done.returncode == 0 and 'stops if the GPU has compute processes' in done.stdout
    assert 'capture exit code: 3' in wait_for(sandbox, 'capture exit code')
    text = sandbox['log'].read_text()
    assert 'FAKE_UV' not in text and 'FAKE_VENV_PY' not in text and 'FAKE_REPO_PY' in text
    line = next(x for x in text.splitlines() if x.startswith('FAKE_CAPTURE'))
    assert line.split()[1] == 'fake-host' and line.endswith('validation/kernel-cases-m3/cases.json')


def test_capture_does_not_overwrite_a_golden(sandbox):
    (sandbox['repo'] / 'validation/kernel-golden-v100-m3').mkdir(parents=True)
    launch(sandbox, 'capture')
    wait_for(sandbox, 'exists. Delete it first')
    assert 'FAKE_CAPTURE' not in sandbox['log'].read_text()


def test_an_unknown_mode_is_a_usage_error(sandbox):
    done = launch(sandbox, 'nonsense')
    assert done.returncode == 2 and 'usage' in done.stderr and not sandbox['log'].exists()
