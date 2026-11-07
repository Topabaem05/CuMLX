"""`cumlx run`: CUDA calls reach the target with a record, and a call with no equivalent is an error."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.e2e


@pytest.fixture(scope='module')
def target(e2e):
    import torch
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def _run(tmp_path, source, target, *options, script_args=(), env=None):
    script = tmp_path / 'script.py'
    script.write_text(textwrap.dedent(source))
    report = tmp_path / 'run-report.json'
    argv = [sys.executable, '-m', 'cumlx', 'run', str(script), '--target', target, '--report', str(report), '--overwrite',
            '--format', 'json', '--max-items', '-1', *options]
    if script_args:
        argv += ['--', *script_args]
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), **(env or {})}
    done = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path, env=environment, check=False)
    doc = json.loads(done.stdout)  # The script's own stdout must not break the one JSON document.
    assert doc['exit_code'] == done.returncode
    written = json.loads(report.read_text()) if report.exists() else None
    return doc, written, done.stderr


def _calls(report, kind=None):
    return {c['name']: c for c in report['calls'] if kind is None or c['kind'] == kind}


def test_an_unmodified_cuda_training_step_runs_on_the_target(tmp_path, target):
    doc, report, stderr = _run(tmp_path, """
        import json, torch, torch.nn.functional as F
        assert torch.cuda.is_available()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert isinstance(device, torch.device)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = torch.nn.Linear(4, 2).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler('cuda')
        x, y = torch.randn(8, 4).cuda(non_blocking=True), torch.randint(0, 2, (8,)).to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tensors = [torch.zeros(3, device='cuda'), torch.ones(3, device='cuda:0'), torch.tensor([1.0], device=0),
                   torch.rand(2).to('cuda'), torch.rand(2).cuda(0), x, next(model.parameters())]
        print('RESULT', json.dumps({'types': sorted({t.device.type for t in tensors}), 'name': torch.cuda.get_device_name(0),
                                    'count': torch.cuda.device_count(), 'cuda_flag': tensors[0].is_cuda}))
    """, target)
    assert doc['exit_code'] == 0, stderr
    result = json.loads(stderr.split('RESULT ', 1)[1].splitlines()[0])
    assert result['types'] == [target]
    assert 'cumlx run' in result['name'] and 'NVIDIA' not in result['name']
    assert result['count'] == 1 and result['cuda_flag'] is True
    assert report['policy']['torch.cuda.is_available'] is True and report['verified'] is False
    calls = _calls(report)
    for name in ('torch.device', 'Module.cuda', 'Tensor.cuda', 'torch.autocast', 'torch.amp.GradScaler', 'torch.cuda.is_available'):
        assert name in calls, name
    assert calls['Module.cuda']['first_site'].endswith('script.py:7')
    assert {w['code'] for w in doc['warnings']} >= {'W_RUN_NOT_VERIFIED'}


def test_a_call_with_no_equivalent_is_an_error_and_never_a_silent_noop(tmp_path, target):
    doc, report, _ = _run(tmp_path, """
        import torch
        torch.cuda.utilization()
        print('this line must not run')
    """, target)
    assert doc['exit_code'] == 3 and doc['errors'][0]['code'] == 'E_RUN_UNSUPPORTED'
    assert 'torch.cuda.utilization' in _calls(report, 'refused')


def test_patterns_from_real_training_scripts(tmp_path, target):
    """The calls that nanoGPT and minGPT make, in a script written here."""
    doc, report, stderr = _run(tmp_path, """
        import inspect, torch
        from contextlib import nullcontext
        device = 'cuda'
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        model = torch.nn.Linear(4, 2)
        model.to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, **(dict(fused=True) if fused and %r == 'mps' else {}))
        x = torch.randn(8, 4).pin_memory().to(device, non_blocking=True)
        with ctx:
            loss = model(x).pow(2).mean()
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.save({'model': model.state_dict()}, 'ckpt.pt')
        state = torch.load('ckpt.pt', map_location=device)
        model.load_state_dict(state['model'])
        print('PATTERN', x.device.type, next(model.parameters()).device.type, dtype)
    """ % target, target)
    assert doc['exit_code'] == 0, stderr
    assert f'PATTERN {target} {target} bfloat16' in stderr


def test_legacy_cuda_tensor_types_reach_the_target(tmp_path, target):
    doc, report, stderr = _run(tmp_path, """
        import torch
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        made = Tensor(2, 1).fill_(1.0)
        cast = torch.ones(2).type(Tensor)
        named = torch.ones(2).type('torch.cuda.LongTensor')
        print('LEGACY', made.device.type, cast.device.type, named.dtype, isinstance(made, torch.cuda.FloatTensor), isinstance(named, torch.cuda.FloatTensor))
    """, target)
    assert doc['exit_code'] == 0, stderr
    assert f'LEGACY {target} {target} torch.int64 True False' in stderr
    assert {'torch.cuda.FloatTensor', 'Tensor.type'} <= set(_calls(report, 'mapped'))


def test_an_unknown_legacy_type_string_is_refused(tmp_path, target):
    doc, _, _ = _run(tmp_path, """
        import torch
        torch.ones(2).type('torch.cuda.NotAType')
    """, target)
    assert doc['errors'][0]['code'] == 'E_RUN_UNSUPPORTED'


def test_a_refusal_that_the_script_catches_is_still_reported(tmp_path, target):
    doc, report, _ = _run(tmp_path, """
        import torch
        try:
            torch.cuda.memory_snapshot()
        except RuntimeError:
            pass
    """, target)
    assert doc['exit_code'] == 0
    assert 'torch.cuda.memory_snapshot' in _calls(report, 'refused')
    assert 'W_RUN_REFUSED' in {w['code'] for w in doc['warnings']}


def test_a_second_gpu_is_refused(tmp_path, target):
    doc, report, _ = _run(tmp_path, """
        import torch
        torch.zeros(2, device='cuda:1')
    """, target)
    assert doc['errors'][0]['code'] == 'E_RUN_UNSUPPORTED' and report['script_exit_code'] != 0


def test_a_missing_mps_op_is_an_error_unless_the_fallback_is_allowed_by_name(tmp_path, e2e):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip('MPS only')
    source = """
        import torch
        print(torch.linalg.eig(torch.eye(2).cuda())[0].device)
    """
    # The variable in the caller's environment must not switch the fallback on behind the user's back.
    doc, report, _ = _run(tmp_path, source, 'mps', env={'PYTORCH_ENABLE_MPS_FALLBACK': '1'})
    assert doc['exit_code'] == 3 and doc['errors'][0]['code'] == 'E_RUN_MPS_OP_MISSING'
    assert 'removed' in report['environment_changes']['PYTORCH_ENABLE_MPS_FALLBACK']
    assert report['policy']['cpu_fallback_allowed'] is False

    doc, report, _ = _run(tmp_path, source, 'mps', '--allow-cpu-fallback')
    assert doc['exit_code'] == 0
    assert report['policy']['cpu_fallback_allowed'] is True
    assert 'W_RUN_CPU_FALLBACK_ALLOWED' in {w['code'] for w in doc['warnings']}


def test_float64_on_mps_is_not_changed_to_float32_for_the_user(tmp_path, e2e):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip('MPS only')
    doc, _, _ = _run(tmp_path, """
        import torch
        torch.zeros(2, dtype=torch.float64).cuda()
    """, 'mps')
    assert doc['exit_code'] == 3 and doc['errors'][0]['code'] == 'E_RUN_MPS_DTYPE'


def test_a_failing_script_reports_its_error_and_its_output_stays_out_of_stdout(tmp_path, target):
    doc, report, stderr = _run(tmp_path, """
        import sys
        print('script stdout line')
        print(sys.argv[1:])
        raise ValueError('from the script')
    """, target, script_args=['--lr', '0.1'])
    assert doc['exit_code'] == 4 and doc['errors'][0]['code'] == 'E_RUN_SCRIPT_FAILED'
    assert 'from the script' in doc['errors'][0]['message']
    assert 'script stdout line' in stderr and "['--lr', '0.1']" in stderr
    assert report['script_exit_code'] == 1


def test_a_missing_package_points_at_the_recipe(tmp_path, target):
    doc, _, _ = _run(tmp_path, 'import flash_attn\n', target)
    assert doc['errors'][0]['code'] == 'E_RUN_MISSING_PACKAGE'
    assert 'cumlx recipe flash_attn' in doc['errors'][0]['hint']


def test_an_explicit_substitute_serves_the_module_and_is_reported(tmp_path, target):
    (tmp_path / 'stand_in.py').write_text('VALUE = 7\n')
    (tmp_path / 'stand_in_pkg').mkdir()
    (tmp_path / 'stand_in_pkg' / '__init__.py').write_text('')
    (tmp_path / 'stand_in_pkg' / 'layers.py').write_text('VALUE = 8\n')
    doc, report, stderr = _run(tmp_path, """
        import sys
        sys.path.insert(0, '.')
        import not_installed, pkg_not_installed.layers as layers
        print('VALUES', not_installed.VALUE, layers.VALUE, not_installed.__name__)
    """, target, '--substitute', 'not_installed=stand_in', '--substitute', 'pkg_not_installed=stand_in_pkg')
    assert doc['exit_code'] == 0, stderr
    assert 'VALUES 7 8 stand_in' in stderr
    assert set(_calls(report, 'substituted')) >= {'not_installed', 'pkg_not_installed.layers'}
    assert 'W_RUN_SUBSTITUTED' in {w['code'] for w in doc['warnings']}


def test_a_dotted_substitute_name_is_a_usage_error(tmp_path, target):
    doc, _, _ = _run(tmp_path, 'print(1)\n', target, '--substitute', 'flash_attn.bert_padding=stand_in')
    assert doc['exit_code'] == 2 and doc['errors'][0]['code'] == 'E_USAGE'
    assert '--substitute flash_attn=MODULE' in doc['errors'][0]['hint']


def test_the_report_folder_is_made_before_the_script_runs(tmp_path, target):
    script = tmp_path / 'script.py'
    script.write_text('print(1)\n')
    report = tmp_path / 'golden' / 'seed0.json'
    argv = [sys.executable, '-m', 'cumlx', 'run', str(script), '--target', target, '--report', str(report), '--format', 'json']
    done = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, check=False)
    assert json.loads(done.stdout)['exit_code'] == 0, done.stderr
    assert json.loads(report.read_text())['script_exit_code'] == 0


def test_a_checkpoint_saved_on_cuda_loads_on_the_target(tmp_path, target):
    doc, report, stderr = _run(tmp_path, """
        import torch
        # Tag a tensor as saved on cuda:0, as a file from an NVIDIA machine is.
        torch.serialization.register_package(1, lambda obj: 'cuda:0', lambda obj, location: None)
        torch.save({'w': torch.ones(2)}, 'on_cuda.pt')
        torch.serialization._package_registry.pop(0)
        print('LOADED', torch.load('on_cuda.pt', weights_only=True)['w'].device.type)
        print('LOADED', torch.load('on_cuda.pt', map_location='cuda', weights_only=True)['w'].device.type)
    """, target)
    assert doc['exit_code'] == 0, stderr
    assert stderr.count(f'LOADED {target}') == 2
    assert 'torch.load' in _calls(report)


def test_dataloader_workers_and_threads_are_covered(tmp_path, target):
    doc, report, stderr = _run(tmp_path, """
        import threading, torch
        class Data(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, i):
                assert torch.cuda.is_available()
                return torch.tensor([float(i)])
        if __name__ == '__main__':
            for batch in torch.utils.data.DataLoader(Data(), batch_size=2, num_workers=2, pin_memory=True):
                assert batch.cuda().device.type != 'cuda'
            seen = []
            thread = threading.Thread(target=lambda: seen.append(torch.zeros(1, device='cuda').device.type))
            thread.start(); thread.join()
            print('THREAD', seen)
    """, target)
    assert doc['exit_code'] == 0, stderr
    assert f"THREAD ['{target}']" in stderr
    assert report['processes'] >= 3  # The script and its two workers.


def test_module_form_and_nested_python_are_covered(tmp_path, target):
    (tmp_path / 'inner.py').write_text('import torch\nprint("INNER", torch.cuda.is_available())\n')
    doc, _, stderr = _run(tmp_path, """
        import subprocess, sys
        print(subprocess.run([sys.executable, 'inner.py'], capture_output=True, text=True).stdout)
    """, target)
    assert doc['exit_code'] == 0 and 'INNER True' in stderr


def test_without_cumlx_run_nothing_changes(tmp_path, e2e):
    """Importing cumlx leaves torch as it is, and the same script fails on a machine with no CUDA."""
    script = tmp_path / 'plain.py'
    script.write_text(textwrap.dedent("""
        import torch
        import cumlx
        before = torch.cuda.is_available()
        assert not hasattr(torch.device, 'is_cumlx')
        assert type(torch.device).__name__ == 'type'
        assert torch.load.__module__ == 'torch.serialization'
        assert torch.cuda.utilization.__module__.startswith('torch.cuda')
        print('AVAILABLE', before)
        try:
            torch.zeros(1, device='cuda')
        except (RuntimeError, AssertionError) as error:
            print('CUDA ERROR')
    """))
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False,
                          env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})
    assert done.returncode == 0, done.stderr
    assert 'AVAILABLE False' in done.stdout and 'CUDA ERROR' in done.stdout


def test_uninstall_restores_every_attribute(e2e):
    import importlib.util

    import torch
    spec = importlib.util.spec_from_file_location('_layer_for_test', ROOT / 'src' / 'cumlx' / 'compat' / 'torch_cuda.py')
    layer = importlib.util.module_from_spec(spec)
    sys.modules['_layer_for_test'] = layer
    spec.loader.exec_module(layer)
    owners = (torch, torch.cuda, torch.Tensor, torch.nn.Module, torch.amp.autocast_mode.autocast, torch.amp.grad_scaler.GradScaler)
    before = [dict(vars(owner)) for owner in owners]
    layer.install(torch, 'cpu', layer._Recorder(None))
    try:
        assert torch.cuda.is_available() is True
        assert torch.zeros(1, device='cuda').device.type == 'cpu'
    finally:
        layer.uninstall()
    for owner, old in zip(owners, before):
        assert dict(vars(owner)) == old, owner
    if not torch.cuda.is_available():
        with pytest.raises((RuntimeError, AssertionError, NotImplementedError)):
            torch.zeros(1, device='cuda')


def _cli(*argv):
    return subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json'], capture_output=True, text=True, check=False,
                          env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')})


def test_usage_errors(tmp_path, e2e):
    missing = _cli('run', str(tmp_path / 'absent.py'), '--report', str(tmp_path / 'r.json'))
    assert missing.returncode == 4 and json.loads(missing.stdout)['errors'][0]['code'] == 'E_INPUT_NOT_FOUND'
    neither = _cli('run', '--report', str(tmp_path / 'r.json'))
    assert neither.returncode == 2
    script = tmp_path / 'ok.py'
    script.write_text('')
    report = tmp_path / 'exists.json'
    report.write_text('{}')
    refused = _cli('run', str(script), '--target', 'cpu', '--report', str(report))
    assert refused.returncode == 2 and json.loads(refused.stdout)['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    assert report.read_text() == '{}'
    bad = _cli('run', str(script), '--target', 'cpu', '--report', str(tmp_path / 'r2.json'), '--substitute', 'nonsense')
    assert bad.returncode == 2


def test_mcp_hides_run_unless_code_execution_is_allowed():
    from cumlx.mcp_server import _argv, build_tools
    assert 'cumlx_run' not in build_tools()
    tool = build_tools(allow_code=True)['cumlx_run']
    assert _argv(tool, {'script': 'a.py', 'script_args': ['--lr', '1'], 'target': 'cpu'}, ('--format', 'json')) == \
        ['run', 'a.py', '--target', 'cpu', '--format', 'json', '--', '--lr', '1']


def test_arguments_after_the_separator_belong_to_the_script(tmp_path, target):
    """Python 3.12.3 and 3.10 refused this mix of positionals and options. The V100 hit it. --help and --format stay the script's."""
    doc, report, stderr = _run(tmp_path, """
        import sys
        print('ARGS', sys.argv[1:], file=sys.stderr)
    """, target, script_args=('--help', '--format', 'text', '--', 'x'))
    assert doc['exit_code'] == 0
    assert "ARGS ['--help', '--format', 'text', '--', 'x']" in stderr
    assert report['script_args'] == ['--help', '--format', 'text', '--', 'x']


def test_a_module_takes_its_arguments_after_the_separator(tmp_path, target):
    (tmp_path / 'mymod.py').write_text('import sys\nprint("ARGS", sys.argv[1:], file=sys.stderr)\n')
    report_path = tmp_path / 'r.json'
    done = subprocess.run([sys.executable, '-m', 'cumlx', 'run', '--module', 'mymod', '--target', target, '--report', str(report_path), '--format', 'json', '--', '--a=1', 'b'],
                          capture_output=True, text=True, cwd=tmp_path, env={**os.environ, 'PYTHONPATH': f"{ROOT / 'src'}{os.pathsep}{tmp_path}"}, check=False)
    assert json.loads(done.stdout)['exit_code'] == 0, done.stdout
    assert "ARGS ['--a=1', 'b']" in done.stderr


def test_the_m2_install_builds_this_package_without_the_pytorch_index():
    """With that index, uv builds this package against an old `packaging`, and setuptools 77 fails. It happened on the V100."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / 'spikes' / 'm2_scripts.py'
    spec = importlib.util.spec_from_file_location('m2_scripts_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch, package = module.install_commands(Path('/venv/bin/python'), 'cuda')
    assert '--index' in torch and 'torch==2.14.1' in torch and '-e' not in torch
    assert '-e' in package and '--index' not in package
    torch_here, _ = module.install_commands(Path('/venv/bin/python'), 'mps')
    assert '--index' not in torch_here
