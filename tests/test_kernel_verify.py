"""`cumlx kernel verify` and `capture`: exit codes and reports, with a fake engine. No GPU, and no golden from a GPU is faked."""
import json
from pathlib import Path

import numpy as np
import pytest

from cumlx.cli.main import COMMANDS, build_parser, main
from cumlx.errors import ERROR_CODES, CumlxError, ExitCode
from cumlx.verify import kernel_capture, kernel_verify
from cumlx.verify.kernel_capture import capture_kernels, generate_harness, plan_capture, read_probe
from cumlx.verify.kernel_verify import KernelRunError, _complete_signature, verify_kernels, wrap_metal_body
from cumlx.verify.opgolden import KernelCase, TensorSpec, write_golden

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / 'src/cumlx/kernels'
CASES = ROOT / 'validation/kernel-cases/cases.json'
X = np.array([1, 2, 3], np.float32)


def run(capsys, *argv):
    code = main([*argv, '--format', 'json'])
    return code, json.loads(capsys.readouterr().out)


class FakeEngine:
    """Returns what the test tells it to. The golden in these tests is kind 'fixture', and it never earns a grade."""
    name = 'fake'

    def __init__(self, compute=None):
        self.compute = compute or (lambda case, source, inputs: {t.name: inputs['x'] * 2 for t in case.outputs})
        self.calls = 0

    def describe(self):
        return {'engine': 'fake'}

    def run(self, case, source, header, inputs):
        if case.kernel.startswith('cumlx_probe'):
            return {'y': np.array([1e-40, 1e-40, 1e-40, 7e-40, 6e-8], np.float32)} if case.kernel == 'cumlx_probe' else {'y': np.array([32.0], np.float32)}
        self.calls += 1
        return self.compute(case, source, inputs)


def golden(tmp, kernels=('foo',), *, kind='fixture', dtype='float32', floating_point=None, features=None, per_kernel=1, repeat_identical=True):
    cases = []
    for kernel in kernels:
        for i in range(per_kernel):
            case = KernelCase(f'{kernel}.c{i}', kernel, [TensorSpec('x', dtype, (3,))], [TensorSpec('y', dtype, (3,))], {}, (3, 1, 1), (3, 1, 1))
            cases.append({'case': case, 'inputs': {'x': X.astype(dtype)}, 'outputs': {'y': (X * 2).astype(dtype)}, 'repeat_identical': repeat_identical})
    info = {k: {'source_sha256': 'aaa', 'features': features or {}} for k in kernels}
    fp = floating_point if floating_point is not None else {
        'ftz': {'flag': 'false', 'load_flushes': False, 'input_flushes': False, 'output_flushes': False},
        'fmad': {'flag': 'true', 'contracts': True}, 'warp_size': 32}
    source = {'framework': 'test'}
    if kind == 'cuda':
        source.update(gpu='Test GPU', driver='0', compiler={'name': 'nvcc', 'version': '0'})
    write_golden(tmp, kind=kind, source=source, kernels=info, floating_point=fp, cases=cases)
    return tmp


def metal(tmp, name='foo', text='y[0] = x[0];\n'):
    p = tmp / f'{name}.metal'
    p.write_text(text)
    return p


# ---------------------------------------------------------------- verdicts
def test_right_kernel_passes_and_only_a_cuda_golden_earns_a_grade(tmp_path):
    g = golden(tmp_path / 'g')
    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine())
    assert rep['verdict'] == 'ok' and rep['first_divergence'] is None
    assert rep['kernels']['foo']['status'] == 'pass' and rep['kernels']['foo']['grade'] is None
    assert 'W_KERNEL_GOLDEN_NOT_CUDA' in [w['code'] for w in rep['warnings']]
    g2 = golden(tmp_path / 'g2', kind='cuda')
    rep = verify_kernels(metal(tmp_path), g2, engine=FakeEngine())
    assert rep['kernels']['foo']['grade'] == 'A' and rep['summary']['grade_a_kernels'] == ['foo']
    assert 'W_KERNEL_GOLDEN_NOT_CUDA' not in [w['code'] for w in rep['warnings']]


def test_wrong_kernel_diverges_and_is_located(tmp_path):
    g = golden(tmp_path / 'g', kind='cuda')
    engine = FakeEngine(lambda case, source, inputs: {'y': inputs['x'] * 2.001})
    rep = verify_kernels(metal(tmp_path), g, engine=engine)
    assert rep['verdict'] == 'diverged' and rep['kernels']['foo']['grade'] is None and rep['kernels']['foo']['status'] == 'fail'
    fd = rep['first_divergence']
    assert fd['kernel'] == 'foo' and fd['case'] == 'foo.c0' and fd['output'] == 'y' and fd['reason'] == 'rel_l2_above_limit'


def test_one_wrong_repeat_is_a_failure(tmp_path):
    g = golden(tmp_path / 'g')
    state = {'n': 0}

    def flaky(case, source, inputs):
        state['n'] += 1
        return {'y': inputs['x'] * (2.5 if state['n'] == 3 else 2)}

    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine(flaky))
    assert rep['verdict'] == 'diverged' and state['n'] == 3 and rep['records'][0]['port_repeat_identical'] is False
    assert 'W_KERNEL_EXEC_ORDER' in [w['code'] for w in rep['warnings']]


@pytest.mark.parametrize('stage', ['compile', 'bind', 'run'])
def test_a_kernel_that_produced_no_output_is_never_a_pass(tmp_path, stage):
    g = golden(tmp_path / 'g', kind='cuda')

    def fail(case, source, inputs):
        raise KernelRunError(stage, 'program_source:1:1: error')

    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine(fail))
    assert rep['verdict'] == 'incomplete' and rep['kernels']['foo']['status'] == 'unverified' and rep['kernels']['foo']['grade'] is None
    assert rep['records'][0]['stage'] == stage and 'W_KERNEL_UNVERIFIED' in [w['code'] for w in rep['warnings']]


def test_a_failure_outranks_an_unverified_kernel(tmp_path):
    g = golden(tmp_path / 'g', kernels=('foo', 'bar'))

    def compute(case, source, inputs):
        if case.kernel == 'bar':
            raise KernelRunError('compile', 'bad')
        return {'y': inputs['x'] * 3}

    d = tmp_path / 'm'
    d.mkdir()
    metal(d, 'foo'), metal(d, 'bar')
    assert verify_kernels(d, g, engine=FakeEngine(compute))['verdict'] == 'diverged'


def test_unknown_kernel_is_not_a_pass_and_needs_no_engine(capsys, tmp_path, monkeypatch):
    g = golden(tmp_path / 'g', kind='cuda')
    monkeypatch.setattr(kernel_verify, 'select_engine', lambda *a, **k: pytest.fail('an unmapped kernel needs no engine'))
    code, doc = run(capsys, 'kernel', 'verify', str(metal(tmp_path, 'other')), str(g))
    assert code == ExitCode.INCOMPLETE and doc['status'] == 'incomplete' and not doc['ok']
    assert doc['data']['summary']['unmapped'] == ['other'] and 'W_KERNEL_UNMAPPED' in [w['code'] for w in doc['warnings']]
    # Nothing ran, so nothing is known about this device: no difference may be claimed.
    assert not {'W_KERNEL_FTZ_DIFFERS', 'W_KERNEL_FMAD_DIFFERS'} & {w['code'] for w in doc['warnings']}
    assert next(d for d in doc['data']['execution']['differences'] if d['topic'] == 'ftz')['differs'] is None
    assert any('--kernel' in step['command'] for step in doc['next_steps'])


def test_kernel_flag_maps_a_file_to_a_golden_name(tmp_path):
    g = golden(tmp_path / 'g')
    assert verify_kernels(metal(tmp_path, 'other'), g, kernel='foo', engine=FakeEngine())['verdict'] == 'ok'
    assert verify_kernels(metal(tmp_path, 'other'), g, kernel='nope', engine=FakeEngine())['verdict'] == 'incomplete'


def test_folder_with_a_gap_is_incomplete_and_extra_files_too(tmp_path):
    g = golden(tmp_path / 'g', kernels=('foo', 'bar'))
    d = tmp_path / 'm'
    d.mkdir()
    metal(d, 'foo')
    rep = verify_kernels(d, g, engine=FakeEngine())
    assert rep['verdict'] == 'incomplete' and rep['summary']['not_covered'] == ['bar'] and rep['kernels']['foo']['status'] == 'pass'
    metal(d, 'bar'), metal(d, 'extra')
    rep = verify_kernels(d, g, engine=FakeEngine())
    assert rep['verdict'] == 'incomplete' and rep['summary']['unmapped'] == ['extra']
    (d / 'extra.metal').unlink()
    assert verify_kernels(d, g, engine=FakeEngine())['verdict'] == 'ok'
    (d / 'foo.header.metal').write_text('')  # a header is not a kernel
    assert verify_kernels(d, g, engine=FakeEngine())['verdict'] == 'ok'


def test_one_file_of_a_larger_golden_is_a_partial_pass_and_says_so(tmp_path):
    g = golden(tmp_path / 'g', kernels=('foo', 'bar'))
    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine())
    assert rep['verdict'] == 'ok' and next(w for w in rep['warnings'] if w['code'] == 'W_KERNEL_PARTIAL')['others'] == ['bar']


def test_case_filter_says_what_it_left_out(tmp_path):
    g = golden(tmp_path / 'g', per_kernel=3)
    rep = verify_kernels(metal(tmp_path), g, cases=['foo.c1'], engine=FakeEngine())
    assert rep['summary']['cases']['pass'] == 1 and 'W_KERNEL_PARTIAL' in [w['code'] for w in rep['warnings']]
    with pytest.raises(CumlxError, match='does not have'):
        verify_kernels(metal(tmp_path), g, cases=['nope'], engine=FakeEngine())


def test_float64_golden_is_unverified_on_metal(tmp_path):
    g = golden(tmp_path / 'g', dtype='float64')
    engine = FakeEngine()
    rep = verify_kernels(metal(tmp_path), g, engine=engine)
    assert rep['verdict'] == 'incomplete' and rep['records'][0]['stage'] == 'fp64' and engine.calls == 0


# ---------------------------------------------------------------- differences stated in the report
def test_report_always_states_fp64_ftz_fmad_and_order(tmp_path):
    g = golden(tmp_path / 'g')
    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine())
    topics = {d['topic']: d for d in rep['execution']['differences']}
    assert set(topics) == {'fp64', 'ftz', 'fmad', 'execution_order'}
    assert topics['fp64']['differs'] is False and topics['ftz']['differs'] is False and topics['fmad']['differs'] is False
    assert topics['execution_order']['differs'] is False and 'SIMD groups of 32' in topics['execution_order']['metal']


def test_fp64_feature_is_a_warning(tmp_path):
    g = golden(tmp_path / 'g', features={'fp64': True, 'fp64_literals': 0})
    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine())
    assert 'W_KERNEL_FP64' in [w['code'] for w in rep['warnings']]
    assert next(d for d in rep['execution']['differences'] if d['topic'] == 'fp64')['kernels'] == ['foo']


def test_ftz_and_fmad_differences_are_warnings(tmp_path):
    fp = {'ftz': {'flag': 'false', 'load_flushes': False, 'input_flushes': False, 'output_flushes': False},
          'fmad': {'flag': 'true', 'contracts': True}, 'warp_size': 32}
    g = golden(tmp_path / 'g', floating_point=fp)

    class Flushes(FakeEngine):
        def run(self, case, source, header, inputs):
            if case.kernel == 'cumlx_probe':
                return {'y': np.array([1e-40, 0, 0, 0, 0], np.float32)}
            return super().run(case, source, header, inputs)

    rep = verify_kernels(metal(tmp_path), g, engine=Flushes())
    codes = [w['code'] for w in rep['warnings']]
    assert 'W_KERNEL_FTZ_DIFFERS' in codes and 'W_KERNEL_FMAD_DIFFERS' in codes and rep['verdict'] == 'ok'
    assert rep['execution']['measured']['input_flushes'] is True


def test_golden_that_does_not_repeat_or_uses_atomics_is_stated(tmp_path):
    g = golden(tmp_path / 'g', repeat_identical=False)
    assert 'W_KERNEL_EXEC_ORDER' in [w['code'] for w in verify_kernels(metal(tmp_path), g, engine=FakeEngine())['warnings']]
    g = golden(tmp_path / 'g2', features={'atomics': True})
    rep = verify_kernels(metal(tmp_path), g, engine=FakeEngine())
    assert next(d for d in rep['execution']['differences'] if d['topic'] == 'execution_order')['kernels_with_atomics'] == ['foo']


def test_a_probe_that_fails_is_stated_and_does_not_decide_the_verdict(tmp_path):
    g = golden(tmp_path / 'g')

    class NoProbe(FakeEngine):
        def run(self, case, source, header, inputs):
            if case.kernel.startswith('cumlx_probe'):
                raise KernelRunError('compile', 'no')
            return super().run(case, source, header, inputs)

    rep = verify_kernels(metal(tmp_path), g, engine=NoProbe())
    assert 'W_KERNEL_PROBE_FAILED' in [w['code'] for w in rep['warnings']]
    assert next(d for d in rep['execution']['differences'] if d['topic'] == 'ftz')['differs'] is None


def test_translation_of_another_source_is_stated(tmp_path):
    g = golden(tmp_path / 'g')
    m = metal(tmp_path)
    (tmp_path / 'foo.port.json').write_text(json.dumps({'source_sha256': 'bbb', 'prepared_sha256': 'ccc', 'translator': {'name': 't'}}))
    rep = verify_kernels(m, g, engine=FakeEngine())
    assert 'W_KERNEL_SOURCE_MISMATCH' in [w['code'] for w in rep['warnings']] and rep['translations']['foo']['translator'] == {'name': 't'}
    (tmp_path / 'foo.port.json').write_text(json.dumps({'source_sha256': 'bbb', 'prepared_sha256': 'aaa'}))
    assert 'W_KERNEL_SOURCE_MISMATCH' not in [w['code'] for w in verify_kernels(m, g, engine=FakeEngine())['warnings']]


# ---------------------------------------------------------------- command line
def test_cli_writes_a_report_and_keeps_the_stdout_contract(capsys, tmp_path, monkeypatch):
    g = golden(tmp_path / 'g', kind='cuda')
    monkeypatch.setattr(kernel_verify, 'select_engine', lambda *a, **k: FakeEngine())
    out = tmp_path / 'report'
    code, doc = run(capsys, 'kernel', 'verify', str(metal(tmp_path)), str(g), '-o', str(out))
    assert code == 0 and doc['schema'] == 'cumlx.result.v1' and doc['command'] == 'kernel verify' and doc['exit_code'] == 0
    rep = json.loads((out / 'report.json').read_text())
    assert rep['schema'] == kernel_verify.REPORT_SCHEMA and 'fp64' in (out / 'report.md').read_text()
    assert {a['kind'] for a in doc['artifacts']} == {'report', 'report_md'}
    code, doc = run(capsys, 'kernel', 'verify', str(metal(tmp_path)), str(g), '-o', str(out))
    assert code == ExitCode.USAGE and doc['errors'][0]['code'] == 'E_OUTPUT_EXISTS'


def test_cli_diverged_is_exit_1_with_the_divergence(capsys, tmp_path, monkeypatch):
    g = golden(tmp_path / 'g', kind='cuda')
    monkeypatch.setattr(kernel_verify, 'select_engine', lambda *a, **k: FakeEngine(lambda c, s, i: {'y': i['x'] * 3}))
    code, doc = run(capsys, 'kernel', 'verify', str(metal(tmp_path)), str(g))
    assert code == ExitCode.DIVERGED and doc['status'] == 'diverged' and doc['data']['first_divergence']['kernel'] == 'foo'
    assert doc['data']['failures']['total'] == 1


def test_cli_missing_inputs(capsys, tmp_path):
    g = golden(tmp_path / 'g')
    code, doc = run(capsys, 'kernel', 'verify', str(tmp_path / 'none.metal'), str(g))
    assert code == ExitCode.INPUT and doc['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'
    code, doc = run(capsys, 'kernel', 'verify', str(metal(tmp_path)), str(tmp_path / 'no-golden'))
    assert code == ExitCode.INPUT and doc['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'
    code, doc = run(capsys, 'kernel', 'nonsense')
    assert code == ExitCode.USAGE
    code, doc = run(capsys, 'kernel')
    assert code == ExitCode.USAGE


def test_missing_mlx_is_an_environment_error(capsys, tmp_path, monkeypatch):
    g = golden(tmp_path / 'g')
    monkeypatch.setitem(__import__('sys').modules, 'mlx.core', None)
    monkeypatch.setattr(kernel_verify.importlib.util, 'find_spec', lambda _n: object())
    code, doc = run(capsys, 'kernel', 'verify', str(metal(tmp_path)), str(g), '--engine', 'mlx')
    assert code == ExitCode.ENVIRONMENT and doc['errors'][0]['code'] == 'E_DEP_MISSING'


def test_complete_library_refuses_the_mlx_engine(capsys, tmp_path):
    g = golden(tmp_path / 'g')
    m = metal(tmp_path, 'foo', '#include <metal_stdlib>\nkernel void foo(const constant float* x [[buffer(0)]], device float* y [[buffer(1)]]) {}\n')
    code, doc = run(capsys, 'kernel', 'verify', str(m), str(g), '--engine', 'mlx')
    assert code == ExitCode.USAGE and '--engine torch' in doc['errors'][0]['hint']


def test_describe_lists_the_kernel_commands_and_their_error_codes(capsys):
    code, doc = run(capsys, 'describe')
    commands = doc['data']['commands']
    assert code == 0 and {'kernel capture', 'kernel verify', 'kernel port'} <= set(commands)
    assert 'kernel' not in commands
    assert {a['name'] for a in commands['kernel verify']['arguments']} >= {'metal', 'golden', 'engine', 'kernel', 'case', 'repeats'}
    code, doc = run(capsys, 'describe', '--command', 'kernel capture')
    assert code == 0 and list(doc['data']['commands']) == ['kernel capture']
    for name in ('E_KERNEL_NO_CUDA', 'E_KERNEL_GPU_BUSY', 'E_KERNEL_COMPILE', 'E_KERNEL_TRANSLATOR_MISSING', 'E_KERNEL_TRANSLATE_FAILED', 'E_KERNEL_UNPORTABLE'):
        assert name in ERROR_CODES
    assert build_parser() is not None and all(COMMANDS[c]['examples'] for c in COMMANDS)


def test_doctor_reports_the_programs_that_capture_needs(capsys, monkeypatch):
    from cumlx.verify import env
    monkeypatch.setattr(env.shutil, 'which', lambda _n: None)
    code, doc = run(capsys, 'doctor', '--require', 'kernel capture')
    assert code == ExitCode.ENVIRONMENT and doc['data']['missing_by_command']['kernel capture'] == ['nvcc', 'nvidia-smi']
    assert 'CUDA toolkit' in doc['errors'][0]['hint']
    monkeypatch.setattr(env.shutil, 'which', lambda n: f'/usr/bin/{n}')
    code, doc = run(capsys, 'doctor', '--require', 'kernel capture')
    assert code == 0 and doc['data']['commands_ready']['kernel capture'] is True


# ---------------------------------------------------------------- capture, without a GPU
def test_capture_without_nvidia_tools_is_an_environment_error_and_writes_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_capture.shutil, 'which', lambda _n: None)
    out = tmp_path / 'g'
    code, doc = run(capsys, 'kernel', 'capture', str(LAB), '--cases', str(CASES), '-o', str(out))
    assert code == ExitCode.ENVIRONMENT and doc['errors'][0]['code'] == 'E_KERNEL_NO_CUDA' and not out.exists()


def test_capture_dry_run_checks_everything_and_runs_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_capture, 'query_environment', lambda: pytest.fail('a dry run needs no GPU'))
    out = tmp_path / 'g'
    code, doc = run(capsys, 'kernel', 'capture', str(LAB), '--cases', str(CASES), '-o', str(out), '--dry-run', '--max-items', '-1')
    assert code == 0 and doc['data']['dry_run'] is True and doc['data']['cases']['total'] == 29 and len(doc['data']['kernels']) == 8
    assert not out.exists()


def test_capture_refuses_a_case_for_an_unknown_kernel_and_a_source_with_main(capsys, tmp_path):
    (tmp_path / 'k.cu').write_text('int main() { return 0; }\n')
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps({'schema': 'cumlx.kernel-cases.v1', 'cases': [{
        'id': 'a', 'kernel': 'other', 'grid': [1, 1, 1], 'threadgroup': [1, 1, 1],
        'inputs': [], 'outputs': [{'name': 'y', 'dtype': 'float32', 'shape': [1]}]}]}))
    code, doc = run(capsys, 'kernel', 'capture', str(tmp_path), '--cases', str(cases), '-o', str(tmp_path / 'g'))
    assert code == ExitCode.INPUT and 'no source' in doc['errors'][0]['message']
    cases.write_text(cases.read_text().replace('"other"', '"k"'))
    code, doc = run(capsys, 'kernel', 'capture', str(tmp_path), '--cases', str(cases), '-o', str(tmp_path / 'g'))
    assert code == ExitCode.INPUT and '--preprocess-only' in doc['errors'][0]['hint']


def _fake_tools(monkeypatch, *, apps=('',)):
    calls = {'apps': list(apps)}

    def tool(argv, *, timeout):
        if argv[0] == 'nvcc' and argv[1] == '--version':
            return 0, 'Cuda compilation tools, release 12.0, V12.0.140'
        if argv[0] == 'nvidia-smi' and 'name,driver_version' in argv[1]:
            return 0, 'Test GPU, 580.0'
        if argv[0] == 'nvidia-smi':
            return 0, calls['apps'].pop(0) if len(calls['apps']) > 1 else calls['apps'][0]
        raise AssertionError(argv)

    monkeypatch.setattr(kernel_capture.shutil, 'which', lambda n: f'/usr/bin/{n}')
    monkeypatch.setattr(kernel_capture, '_run_tool', tool)
    return calls


def test_busy_gpu_stops_the_capture_before_anything_runs(tmp_path, monkeypatch):
    _fake_tools(monkeypatch, apps=('4242',))
    monkeypatch.setattr(kernel_capture, '_compile', lambda *a: pytest.fail('a busy GPU must stop the capture first'))
    plan = plan_capture(LAB, CASES, None)
    with pytest.raises(CumlxError) as exc:
        capture_kernels(plan, tmp_path / 'g', arch=None, extra_flags=[], repeats=3, timeout=1, jobs=1, require_idle=True, log=lambda _m: None)
    assert exc.value.code == 'E_KERNEL_GPU_BUSY' and exc.value.details['pids'] == ['4242'] and not (tmp_path / 'g').exists()


def test_process_that_starts_during_the_capture_stops_it_and_no_golden_is_written(tmp_path, monkeypatch):
    _fake_tools(monkeypatch, apps=('', '777'))
    monkeypatch.setattr(kernel_capture, '_compile', lambda *a: (0, ''))
    probe = {'outputs': {'y': np.zeros(5, np.float32)}, 'repeat_identical': True, 'device': {'gpu': 'g', 'arch': '7.0'}}
    monkeypatch.setattr(kernel_capture, '_probe', lambda *a: probe)
    monkeypatch.setattr(kernel_capture, '_execute_case', lambda *a: pytest.fail('it must stop before running'))
    with pytest.raises(CumlxError) as exc:
        capture_kernels(plan_capture(LAB, CASES, None), tmp_path / 'g', arch=None, extra_flags=[], repeats=3, timeout=1, jobs=1, require_idle=True, log=lambda _m: None)
    assert exc.value.code == 'E_KERNEL_GPU_BUSY' and not (tmp_path / 'g').exists()


def test_case_that_does_not_compile_is_left_out_and_makes_the_capture_incomplete(capsys, tmp_path, monkeypatch):
    _fake_tools(monkeypatch)
    probe = {'outputs': {'y': np.zeros(5, np.float32)}, 'repeat_identical': True, 'device': {'gpu': 'g', 'arch': '7.0'}}
    monkeypatch.setattr(kernel_capture, '_probe', lambda *a: probe)
    monkeypatch.setattr(kernel_capture, '_compile', lambda *a: (1, 'error: nope'))
    out = tmp_path / 'g'
    code, doc = run(capsys, 'kernel', 'capture', str(LAB), '--cases', str(CASES), '-o', str(out))
    assert code == ExitCode.INCOMPLETE and doc['data']['cases_recorded'] == 0 and doc['data']['failed']['total'] == 29
    assert not out.exists()  # no output is ever made up


def test_repeats_below_two_are_refused(tmp_path, monkeypatch):
    with pytest.raises(CumlxError, match='2 or more'):
        capture_kernels(plan_capture(LAB, CASES, None), tmp_path / 'g', arch=None, extra_flags=[], repeats=1, timeout=1, jobs=1, require_idle=False, log=lambda _m: None)


def test_harness_launches_blocks_and_fills_outputs_differently():
    case = KernelCase('c', 'k', [TensorSpec('x', 'float32', (4, 2)), TensorSpec('s', 'int32', ())], [TensorSpec('y', 'uint8', (3,))],
                      {'N': 3, 'FLAG': True}, (256, 1, 1), (128, 1, 1))
    text = generate_harness(case, 'y[0] = 1;\n', '// header\n')
    assert 'constexpr int N = 3;' in text and 'constexpr bool FLAG = true;' in text
    assert '__global__ void k(const float* x, const int s, unsigned char* y)' in text
    assert 'k<<<dim3(2, 1, 1), dim3(128, 1, 1)>>>(dev_x, dev_s, dev_y);' in text
    assert 'cudaMemset(dev_y, fills[rep % 2], 3)' in text and '{0, 255}' in text
    assert text.index('// header') < text.index('constexpr int N')
    complete = generate_harness(KernelCase('c', 'k', [], [TensorSpec('y', 'float32', (1,))], entry='k<float>'),
                                'template <typename T> __global__ void k(T* y) { y[0] = 1; }\n', '')
    assert 'k<float><<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(dev_y);' in complete
    assert '__global__ void k(float* y)' not in complete


def test_probe_reading():
    assert read_probe(np.array([1e-40, 1e-40, 1e-40, 7e-40, 6e-8], np.float32)) == {
        'load_flushes': False, 'input_flushes': False, 'output_flushes': False, 'contracts': True}
    assert read_probe(np.array([1e-40, 0, 0, 0, 0], np.float32)) == {
        'load_flushes': False, 'input_flushes': True, 'output_flushes': True, 'contracts': False}


# ---------------------------------------------------------------- library text
def test_wrapped_metal_library_has_the_mlx_signature():
    case = KernelCase('c', 'k', [TensorSpec('x', 'float16', (4,)), TensorSpec('s', 'float32', ())], [TensorSpec('y', 'uint8', (3,))], {'N': 3, 'B': False})
    text = wrap_metal_body(case, 'y[thread_position_in_grid.x] = 1;\n', 'inline int h() { return 1; }\n')
    assert 'const constant half* x [[buffer(0)]]' in text and 'const constant float& s [[buffer(1)]]' in text
    assert 'device uint8_t* y [[buffer(2)]]' in text and 'uint3 thread_position_in_grid [[thread_position_in_grid]]' in text
    assert 'constant int N = 3;' in text and 'constant bool B = false;' in text and 'threadgroup_position_in_grid' not in text


def test_complete_library_binding():
    text = 'kernel void foo(device float* y [[buffer(1)]], constant float* x [[buffer(0)]], uint3 t [[thread_position_in_grid]]) { }\n'
    assert _complete_signature(text, ['foo']) == ('foo', ['x', 'y'])
    assert _complete_signature(text, ['other']) == ('foo', ['x', 'y'])  # one kernel in the library
    with pytest.raises(KernelRunError, match='no `kernel void`'):
        _complete_signature('void nothing() {}', ['foo'])
    two = text + text.replace('foo', 'bar')
    with pytest.raises(KernelRunError, match='none is named'):
        _complete_signature(two, ['other'])
    with pytest.raises(KernelRunError, match='numbered from 0'):
        _complete_signature('kernel void foo(device float* y [[buffer(2)]]) { }', ['foo'])
