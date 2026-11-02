"""HTML report, MCP server, version policy, Action body, partial traces, check_op and NVFP4 checkpoint comparison."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENS = [1, 5, 9, 23, 7, 99, 3, 64, 12, 8]


# --------------------------------------------------------------- version policy
def test_mlx_version_policy():
    from cumlx.mlx_backend import check_mlx_version
    assert check_mlx_version('0.32.2') is True and check_mlx_version('0.32.4') is True
    with pytest.warns(UserWarning, match='have not been run'):
        assert check_mlx_version('0.32.9') is False
    for bad in ('0.32.1', '0.31.9', '0.33.0', '1.0.0'):
        with pytest.raises(RuntimeError, match='revalidate'):
            check_mlx_version(bad)


# --------------------------------------------------------------- HTML
def _report(verdict='diverged'):
    fail = {'rel_l2': 0.5, 'verdict': 'fail'}
    good = {'rel_l2': 1e-6, 'verdict': 'pass'}
    return {'verdict': verdict, 'mode': 'isolated', 'summary': {'records': 2, 'pass': 1, 'fail': 1, 'unverified': 0, 'skipped': 0,
            'ignored': 0, 'chain_drift': 0, 'decode_records': 0, 'parametrized_coverage': 1.0},
            'threshold': {'rel_l2': 1e-4, 'leaf_rel_l2': 1e-4}, 'port': {'path': '<port&"x>', 'profile': 'float32'},
            'trace': {'source': {'model': 'm', 'revision': 'r'}, 'golden': {'kind': 'reference'}},
            'execution': {'device': 'gpu', 'device_name': 'X', 'tf32': 'off', 'gemm_noise_rel_l2': 1e-7},
            'first_divergence': {'module': 'a.b', 'basis': 'isolated', 'confidence': 'high', 'phase': 'decode',
                                 'metrics': fail, 'explanation': 'because <reason>'},
            'warnings': [{'code': 'W_X', 'message': 'careful', 'hint': 'do this'}],
            'records': [{'sample': 's0', 'name': 'a', 'type': 'Linear', 'status': 'pass', 'chain': good, 'isolated': good},
                        {'sample': 's0', 'name': 'a.b', 'type': 'Linear', 'status': 'fail', 'phase': 'decode', 'chain': fail, 'isolated': fail}]}


def test_html_report_is_self_contained_and_escaped():
    from cumlx.verify.report_html import to_html
    page = to_html(_report())
    assert page.startswith('<!doctype html>') and '<svg' in page and 'prefers-color-scheme:dark' in page
    assert 'a.b' in page and 'decode' in page and 'W_X' in page
    assert '<port&' not in page and '&lt;port&amp;' in page and 'because &lt;reason&gt;' in page
    assert 'http://' not in page.replace('http://www.w3.org', '') and 'https://' not in page and '<script src' not in page
    assert page.index('a.b</code></td>') < page.index('<code>a</code></td>')  # the failing record comes first


# --------------------------------------------------------------- MCP
def test_mcp_tools_come_from_the_parser_and_hide_code_execution():
    from cumlx.cli.main import COMMANDS
    from cumlx.mcp_server import _argv, build_tools
    tools = build_tools()
    assert set(tools) == {f'cumlx_{c}' for c in COMMANDS if c != 'run'}  # run executes a file
    assert 'cumlx_run' in build_tools(allow_code=True)
    diff = tools['cumlx_diff']
    assert diff['schema']['required'] == ['trace', 'port'] and 'loader' not in diff['schema']['properties']
    assert 'trust_remote_code' not in tools['cumlx_capture']['schema']['properties']
    assert diff['schema']['properties']['device']['enum'] == ['auto', 'cpu', 'gpu']
    assert 'loader' in build_tools(allow_code=True)['cumlx_diff']['schema']['properties']
    assert _argv(diff, {'trace': 'g', 'port': 'p; rm -rf /', 'mode': 'chain', 'overwrite': True}) == \
        ['diff', 'g', 'p; rm -rf /', '--mode', 'chain', '--overwrite']
    with pytest.raises(ValueError, match='unknown or disabled'):
        _argv(diff, {'trace': 'g', 'port': 'p', 'loader': 'python:evil:run'})
    assert _argv(tools['cumlx_capture'], {'model': 'm', 'output': 'o', 'token_ids': ['1,2', '3']})[-4:] == \
        ['--token-ids', '1,2', '--token-ids', '3']


def test_mcp_protocol_over_stdio():
    from cumlx.mcp_server import Server, serve
    lines = [{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
             {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
             {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
             {'jsonrpc': '2.0', 'id': 3, 'method': 'ping'},
             {'jsonrpc': '2.0', 'id': 4, 'method': 'no/such'},
             {'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call', 'params': {'name': 'cumlx_nope', 'arguments': {}}}]
    stdin = io.StringIO('\n'.join(json.dumps(x) for x in lines) + '\nnot json\n')
    stdout = io.StringIO()
    serve(Server(), stdin, stdout)
    replies = [json.loads(x) for x in stdout.getvalue().splitlines()]
    assert [r.get('id') for r in replies] == [1, 2, 3, 4, 5, None]  # the notification gets no reply
    assert replies[0]['result']['serverInfo']['name'] == 'cumlx' and replies[0]['result']['protocolVersion'] == '2025-06-18'
    assert any(t['name'] == 'cumlx_diff' for t in replies[1]['result']['tools'])
    assert replies[3]['error']['code'] == -32601 and replies[4]['error']['code'] == -32602 and replies[5]['error']['code'] == -32700


def _rpc(proc, msg_id, method, params=None):
    proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': msg_id, 'method': method, 'params': params or {}}) + '\n')
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


@pytest.mark.e2e
def test_mcp_agent_workflow_capture_diff_interpret(tmp_path, e2e):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    proc = subprocess.Popen([sys.executable, '-m', 'cumlx.mcp_server'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        assert _rpc(proc, 1, 'initialize')['result']['capabilities'] == {'tools': {}}

        def tool(msg_id, name, **arguments):
            return _rpc(proc, msg_id, 'tools/call', {'name': name, 'arguments': arguments})['result']
        tiny, golden, report = tmp_path / 'tiny', tmp_path / 'g', tmp_path / 'r'
        assert tool(2, 'cumlx_fixture', kind='tiny-llama', output=str(tiny))['structuredContent']['ok']
        cap = tool(3, 'cumlx_capture', model=str(tiny), token_ids=['1,5,9,23'], output=str(golden))
        assert cap['structuredContent']['data']['records'] > 0
        diff = tool(4, 'cumlx_diff', trace=str(golden), port=str(tiny), output=str(report))
        assert diff['isError'] is False and diff['structuredContent']['data']['verdict'] == 'ok'
        again = tool(5, 'cumlx_diff', trace=str(golden), port=str(tiny), output=str(report))
        assert again['isError'] is True and again['structuredContent']['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
        assert (report / 'report.html').is_file()
        assert tool(6, 'cumlx_report', path=str(report))['structuredContent']['data']['verdict'] == 'ok'
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


# --------------------------------------------------------------- partial traces
@pytest.mark.e2e
def test_layers_record_a_part_and_the_diff_says_so(tmp_path, e2e):
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.diff import run_diff
    from cumlx.verify.fixtures import make_tiny_llama
    from cumlx.verify.trace import Trace, TraceWriter, prepare_output_dir
    tiny = tmp_path / 'tiny'
    make_tiny_llama(tiny)
    out = prepare_output_dir(tmp_path / 'g', overwrite=True)
    w = TraceWriter(out, source={}, golden={'kind': 'reference', 'dtype': 'float32'}, max_bytes=1 << 30)
    capture_model(load_hf_model(str(tiny)), [np.asarray(TOKENS)], w, layers=[1], decode_steps=1)
    w.finish()
    t = Trace(out)
    names = {r.name for r in t.manifest.samples[0].records}
    assert t.manifest.source['partial']['layers'] == [1]
    assert 'model.layers.1.self_attn' in names and not any(n.startswith('model.layers.0') for n in names)
    assert {'', 'model', 'model.embed_tokens', 'lm_head'} <= names
    rep = run_diff(str(out), str(tiny))
    assert rep['verdict'] == 'ok' and any(w['code'] == 'W_PARTIAL_TRACE' for w in rep['warnings'])
    # A fault in a recorded layer is located; one in an unrecorded layer shows at the composite that contains it.
    model = mlx_port.load_port(str(tiny))
    faults.inject(model, 'F7', 'model.layers.1.self_attn')
    assert run_diff(str(out), 'x', model=model)['first_divergence']['module'] == 'model.layers.1.self_attn'
    model = mlx_port.load_port(str(tiny))
    faults.inject(model, 'F1', 'model.layers.0.self_attn.q_proj')
    assert run_diff(str(out), 'x', model=model)['first_divergence']['module'] in ('model', '')


def test_layers_argument_parsing():
    from cumlx.cli.commands import _parse_layers
    from cumlx.errors import CumlxError
    assert _parse_layers('0,11,20-22') == [0, 11, 20, 21, 22] and _parse_layers(None) is None
    for bad in ('a', '', '-1', '1,,2'):
        with pytest.raises(CumlxError):
            _parse_layers(bad)


# --------------------------------------------------------------- check_op
@pytest.mark.e2e
def test_check_op_passes_equal_ops_and_fails_wrong_ones(e2e):
    import mlx.core as mx
    import mlx.nn as nn
    import torch

    from cumlx import check_op
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    y = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)
    assert check_op(torch.nn.functional.silu, nn.silu, [x, x[:2]], device='cpu')['verdict'] == 'ok'
    report = check_op(torch.matmul, mx.matmul, [(x, y)])
    assert report['verdict'] == 'ok' and report['worst_rel_l2'] < 1e-5 and report['execution']['reduced_precision_matmul'] is False
    bad = check_op(torch.nn.functional.silu, nn.gelu, [x], name='wrong')
    assert bad['verdict'] == 'diverged' and bad['failed'] == 1 and bad['name'] == 'wrong'
    assert check_op(torch.nn.functional.silu, nn.silu, [])['verdict'] == 'incomplete'


@pytest.mark.gpu
def test_check_op_metal_kernels_match_pytorch(gpu):
    import mlx.core as mx
    import torch

    from cumlx import check_op, quantize
    from cumlx.quant import dequantize
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 65)).astype(np.float32)
    u = rng.normal(size=(3, 65)).astype(np.float32)
    w = rng.normal(size=65).astype(np.float32)

    def rms_torch(a, b):
        return a * torch.sqrt(1 / (a.pow(2).mean(-1, keepdim=True) + 1e-5)) * b

    rep = check_op(rms_torch, lambda a, b: gpu.rms_norm(a, b, eps=1e-5), [(x, w)], name='rms_norm', rel_l2=1e-5)
    assert rep['verdict'] == 'ok', rep
    rep = check_op(lambda a, b: torch.nn.functional.silu(a) * b, gpu.silu_mul, [(x, u)], name='silu_mul', rel_l2=1e-5)
    assert rep['verdict'] == 'ok', rep
    q = quantize(rng.normal(size=(35, 65)).astype(np.float32), 'nvfp4')
    weight = torch.from_numpy(dequantize(q))
    prepared = gpu.prepare(q)
    rep = check_op(lambda a: a @ weight.T, lambda a: gpu.linear(a, prepared), [x], name='fp4_linear', rel_l2=1e-4)
    assert rep['verdict'] == 'ok', rep
    assert mx.metal.is_available()


# --------------------------------------------------------------- NVFP4 checkpoint
def _modelopt_pair(tmp_path, *, mutate=None, global_scale=lambda m: m * 2688.0):
    from cumlx import quantize
    from cumlx.verify.safetensors_io import save_file
    rng = np.random.default_rng(5)
    original, repack = {'embed.weight': rng.normal(size=(8, 4)).astype(np.float32)}, {}
    repack['embed.weight'] = original['embed.weight'].copy()
    for name, shape in (('layers.0.mixer.q_proj', (16, 64)), ('layers.0.mixer.o_proj', (32, 64))):
        q = quantize(rng.normal(size=shape).astype(np.float32) * 0.3, 'nvfp4')
        original[f'{name}.weight'] = q.payload
        original[f'{name}.weight_scale'] = q.scales
        original[f'{name}.weight_scale_2'] = np.array(q.global_scale, np.float32)
        repack[f'{name}.weight'] = np.ascontiguousarray(q.payload).view(np.uint32).copy()
        repack[f'{name}.scales'] = q.scales.copy()
        gs = global_scale(q.global_scale)
        if gs is not None:
            repack[f'{name}.global_scale'] = np.array(gs, np.float32)
    if mutate:
        mutate(repack)
    save_file(original, tmp_path / 'orig.safetensors')
    save_file(repack, tmp_path / 'repack.safetensors')
    return str(tmp_path / 'orig.safetensors'), str(tmp_path / 'repack.safetensors')


@pytest.mark.e2e
def test_checkpoint_comparison_passes_a_correct_repack(tmp_path, e2e):
    from cumlx.verify.checkpoint import run_checkpoint
    rep = run_checkpoint(*_modelopt_pair(tmp_path))
    assert rep['verdict'] == 'ok', rep['first_failure']
    assert rep['summary']['pass'] == 2 and rep['summary']['plain_identical'] == 1
    assert all(r['payload_identical'] and r['scales_identical'] and r['rel_l2'] < 1e-4 for r in rep['parameters'])


@pytest.mark.e2e
@pytest.mark.parametrize('kwargs,reason,diagnosis', [
    (dict(global_scale=lambda m: m), 'decoded_values_differ', 'MLX divides it by 2688'),
    (dict(global_scale=lambda m: None), 'decoded_values_differ', 'was dropped'),
    (dict(mutate=lambda r: r.update({'layers.0.mixer.o_proj.scales': r['layers.0.mixer.o_proj.scales'] + 1})),
     'block_scales_differ', None),
    (dict(mutate=lambda r: r.update({'layers.0.mixer.o_proj.weight': ((r['layers.0.mixer.o_proj.weight'] & 0x0F0F0F0F) << 4)
                                     | ((r['layers.0.mixer.o_proj.weight'] >> 4) & 0x0F0F0F0F)})),
     'payload_bytes_differ', None),
])
def test_checkpoint_comparison_names_each_defect(tmp_path, e2e, kwargs, reason, diagnosis):
    from cumlx.verify.checkpoint import run_checkpoint
    rep = run_checkpoint(*_modelopt_pair(tmp_path, **kwargs))
    first = rep['first_failure']
    assert rep['verdict'] == 'diverged' and first['reason'] == reason, first
    if diagnosis:
        assert diagnosis in first['diagnosis']


@pytest.mark.e2e
def test_checkpoint_missing_tensor_is_not_a_pass(tmp_path, e2e):
    from cumlx.verify.checkpoint import run_checkpoint
    def drop(r):
        del r['layers.0.mixer.q_proj.weight'], r['layers.0.mixer.q_proj.scales']
    rep = run_checkpoint(*_modelopt_pair(tmp_path, mutate=drop))
    assert rep['verdict'] == 'incomplete' and rep['summary']['unverified'] == 1


# --------------------------------------------------------------- Action body
@pytest.mark.e2e
def test_action_fails_on_a_faulty_port_and_passes_a_clean_one(tmp_path, e2e):
    from cumlx.verify.fixtures import make_tiny_llama
    make_tiny_llama(tmp_path / 'original')
    (tmp_path / 'port_clean.py').write_text('from cumlx.verify.mlx_port import load_port\n'
                                            'def load(path):\n    return load_port(path)\n')
    (tmp_path / 'port_faulty.py').write_text('from cumlx.verify import faults\nfrom cumlx.verify.mlx_port import load_port\n'
                                             'def load(path):\n    m = load_port(path)\n'
                                             "    faults.inject(m, 'F7', 'model.layers.1.self_attn')\n    return m\n")

    def action(module):
        summary, output = tmp_path / f'{module}.md', tmp_path / f'{module}.out'
        env = {**os.environ, 'ORIGINAL': 'original', 'PORT': 'original', 'TOKEN_IDS': '1,5,9,23,7', 'CHECK_WEIGHTS': 'true',
               'LOADER': f'python:{module}:load', 'PYTHONPATH': f"{ROOT / 'src'}{os.pathsep}{tmp_path}",
               'GITHUB_STEP_SUMMARY': str(summary), 'GITHUB_OUTPUT': str(output)}
        p = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'action_verify.py')], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        return p.returncode, output.read_text(), summary.read_text()

    code, out, summary = action('port_clean')
    assert code == 0 and 'verdict=ok' in out, (out, summary)
    code, out, summary = action('port_faulty')
    assert code == 1 and 'verdict=diverged' in out and 'first-divergence=model.layers.1.self_attn' in out
    assert 'First divergence' in summary


# --------------------------------------------------------------- trace comparison
@pytest.mark.e2e
def test_tracediff_passes_equal_traces_and_reports_a_moved_golden(tmp_path, e2e):
    import torch

    from cumlx.errors import CumlxError
    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.fixtures import make_tiny_llama
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    from cumlx.verify.trace_compare import compare_traces
    tiny = tmp_path / 'tiny'
    make_tiny_llama(tiny)

    def capture(name, tokens=TOKENS, scale=None):
        model = load_hf_model(str(tiny))
        if scale:
            with torch.no_grad():
                model.lm_head.weight.mul_(scale)
        out = prepare_output_dir(tmp_path / name, overwrite=True)
        w = TraceWriter(out, source={}, golden={'kind': 'reference', 'dtype': 'float32'}, max_bytes=1 << 30)
        capture_model(model, [np.asarray(tokens)], w, decode_steps=1)
        w.finish()
        return str(out)
    a = capture('a')
    same = compare_traces(a, capture('b'))
    assert same['verdict'] == 'ok' and same['summary']['worst_rel_l2'] == 0.0 and same['summary']['compared'] > 0
    moved = compare_traces(a, capture('c', scale=1.01))
    assert moved['verdict'] == 'diverged' and moved['worst'][0]['key'].startswith(('lm_head', '#'))
    with pytest.raises(CumlxError):
        compare_traces(a, capture('d', tokens=TOKENS[:4]))


# --------------------------------------------------------------- scan and port
def _write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.mark.parametrize('files,tier,finding', [
    ({'a.py': 'model = model.cuda()\n'}, 'device', 'torch-cuda-device'),
    ({'a.py': 'import torch\nx = torch.ones(1)\n'}, 'device', None),
    ({'a.py': 'from flash_attn import flash_attn_func\n', 'b.py': 'x.cuda()\n'}, 'library', 'flash-attn-import'),
    ({'requirements.txt': 'bitsandbytes>=0.4\n'}, 'library', 'bitsandbytes'),
    ({'k.cu': '__global__ void f() {}\n', 'a.py': 'x.cuda()\n'}, 'kernel', 'cuda-kernel-source'),
    ({'setup.py': 'from torch.utils.cpp_extension import CUDAExtension\n'}, 'kernel', 'cuda-build'),
    ({'sim.py': 'import warp as wp\n'}, 'kernel', 'warp-cuda'),
    ({'app.py': 'import omni.kit.app\n', 'k.cu': '__global__ void f() {}\n'}, 'platform', 'omniverse-kit'),
    ({'r.cpp': '#include <optix.h>\noptixInit();\n'}, 'platform', 'optix-rtx'),
])
def test_scan_ranks_the_hardest_dependency(tmp_path, files, tier, finding):
    from cumlx.verify.portability import scan
    _write(tmp_path, files)
    rep = scan(str(tmp_path))
    assert rep['tier'] == tier, rep['findings']
    if finding:
        assert finding in {f['id'] for f in rep['findings']}
    else:
        assert rep['findings'] == [] and 'No CUDA dependency' in rep['recommendation']


def test_scan_reads_text_and_never_executes_the_target(tmp_path):
    from cumlx.verify.portability import scan
    marker = tmp_path / 'ran'
    _write(tmp_path, {'evil.py': f'open({str(marker)!r}, "w").close()\nimport omni.kit\n', 'setup.py': f'open({str(marker)!r}, "w").close()\n'})
    assert scan(str(tmp_path))['tier'] == 'platform'
    assert not marker.exists()


def test_scan_skips_docs_tools_and_tests_by_default(tmp_path):
    from cumlx.verify.portability import scan
    _write(tmp_path, {'docs/a.py': 'import omni.kit\n', 'tools/r.py': 'import omni.kit\n', 'tests/t.py': 'import omni.kit\n', 'a.py': 'x = 1\n'})
    assert scan(str(tmp_path))['tier'] == 'device'
    assert scan(str(tmp_path), include_tests=True)['tier'] == 'platform'


@pytest.mark.e2e
def test_scan_recognizes_a_model_and_remote_code(tmp_path, e2e):
    from cumlx.verify.portability import scan
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'llama', 'architectures': ['LlamaForCausalLM']}))
    rep = scan(str(tmp_path))
    assert rep['tier'] == 'model' and rep['model']['mlx_lm_supports'] is True
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'llama', 'auto_map': {'AutoModel': 'x.Y'}}))
    assert scan(str(tmp_path))['tier'] == 'device'
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'no_such_arch'}))
    rep = scan(str(tmp_path))
    assert rep['model']['mlx_lm_supports'] is False and rep['tier'] != 'model'


def _cli(*argv):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUMLX_FORMAT': 'json'}
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv], capture_output=True, text=True, env=env)
    doc = json.loads(p.stdout)
    assert doc['exit_code'] == p.returncode
    return doc


@pytest.mark.e2e
def test_port_converts_and_proves_a_model(tmp_path, e2e):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from cumlx.verify.fixtures import make_tiny_llama
    make_tiny_llama(tmp_path / 'tiny')
    # mlx_lm convert copies a tokenizer, and the fixture has none.
    vocab = {f't{i}': i for i in range(128)}
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='t0')), unk_token='t0').save_pretrained(tmp_path / 'tiny')
    plan = _cli('port', str(tmp_path / 'tiny'), '-o', str(tmp_path / 'p'), '--dry-run')
    assert plan['ok'] and plan['data']['plan']['model_type'] == 'llama' and not (tmp_path / 'p').exists()
    doc = _cli('port', str(tmp_path / 'tiny'), '-o', str(tmp_path / 'p'), '--token-ids', '1,5,9,23,7', '--dtype', 'bfloat16')
    assert doc['ok'] and doc['data']['verdict'] == 'ok', doc
    assert set(doc['data']['steps']) == {'convert', 'weights', 'capture', 'diff'}
    assert (tmp_path / 'p' / 'mlx' / 'config.json').is_file() and (tmp_path / 'p' / 'report' / 'report.html').is_file()
    assert _cli('port', str(tmp_path / 'tiny'), '-o', str(tmp_path / 'p'))['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    q = _cli('port', str(tmp_path / 'tiny'), '-o', str(tmp_path / 'q'), '--token-ids', '1,5,9,23,7', '--q-bits', '4')
    assert q['ok'] and q['data']['plan']['golden'] == 'port-weights', q


@pytest.mark.e2e
def test_port_refuses_when_no_converter_exists(tmp_path, e2e):
    (tmp_path / 'm').mkdir()
    (tmp_path / 'm' / 'config.json').write_text(json.dumps({'model_type': 'no_such_arch'}))
    doc = _cli('port', str(tmp_path / 'm'), '-o', str(tmp_path / 'p'))
    assert doc['exit_code'] == 4 and doc['errors'][0]['code'] == 'E_MODEL_LOAD' and 'no_such_arch' in doc['errors'][0]['message']
    assert not (tmp_path / 'p').exists()
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'a.py').write_text('x = 1\n')
    doc = _cli('port', str(tmp_path / 'src'), '-o', str(tmp_path / 'p'))
    assert doc['errors'][0]['code'] == 'E_MODEL_LOAD' and 'cumlx scan' in doc['errors'][0]['hint']


# --------------------------------------------------------------- findings from recent real models
@pytest.mark.e2e
def test_replay_uses_the_mask_the_port_really_passed(e2e):
    """LFM2 convolutions take a padding mask or none. A forced causal mask broke them."""
    import mlx.core as mx
    import mlx.nn as nn

    from cumlx.verify import mlx_port

    class Block(nn.Module):
        def __call__(self, x, mask=None, cache=None):
            if isinstance(mask, str):
                raise TypeError('this block has no causal mask')
            return x + 1 if mask is None else x * mask[..., None]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def __call__(self, x):
            return self.block(x, None)

    net = Net()
    x = mx.ones((1, 3, 2))
    with mlx_port.probe(net, {'block'}) as rec:
        net(x)
    assert rec.masks == {'block#0': None}
    assert np.array_equal(np.array(mlx_port.replay(net.block, x, 'hidden', rec.masks['block#0'])), np.full((1, 3, 2), 2.0))


@pytest.mark.e2e
def test_constant_factor_moved_to_the_parent_is_not_a_defect_but_a_real_scale_error_is(tmp_path, e2e):
    from cumlx.verify.diff import _accept_scale_moved_to_parent
    fail = {'sample': 's', 'phase': 'prefill', 'name': 'model.embed', 'status': 'fail', 'fail_cause': 'self',
            'isolated': {'reason': 'rel_l2_above_threshold', 'cosine': 1.0, 'scale': 32.0, 'verdict': 'fail'}}
    good_parent = {'sample': 's', 'phase': 'prefill', 'name': 'model', 'status': 'pass', 'isolated': {'verdict': 'pass'}}
    bad_parent = {**good_parent, 'status': 'fail', 'isolated': {'verdict': 'fail'}}
    rows = [dict(fail), good_parent]
    assert _accept_scale_moved_to_parent(rows) == 1 and rows[0]['status'] == 'pass' and rows[0]['boundary_scale'] == 32.0
    rows = [dict(fail), bad_parent]  # the parent also differs: a real scale error
    assert _accept_scale_moved_to_parent(rows) == 0 and rows[0]['status'] == 'fail'
    rows = [{**fail, 'isolated': {**fail['isolated'], 'cosine': 0.9}}, good_parent]  # not a constant factor
    assert _accept_scale_moved_to_parent(rows) == 0
    rows = [dict(fail)]  # no parent was compared: nothing shows the factor comes back
    assert _accept_scale_moved_to_parent(rows) == 0


@pytest.mark.e2e
def test_partial_port_weights_never_make_a_golden(tmp_path, e2e):
    """Regression: Mixtral expert weights were silently left out, so the golden held the original experts."""
    from cumlx.errors import CumlxError
    from cumlx.verify import mlx_port
    from cumlx.verify.capture_torch import load_hf_model, load_mlx_weights_into
    from cumlx.verify.fixtures import make_tiny_llama
    tiny = tmp_path / 'tiny'
    make_tiny_llama(tiny)
    weights = mlx_port.dequantized_weights(mlx_port.load_port(str(tiny)))
    assert load_mlx_weights_into(load_hf_model(str(tiny)), dict(weights))['loaded'] > 0
    renamed = dict(weights)
    renamed['model.layers.0.mlp.renamed.weight'] = renamed.pop('model.layers.0.mlp.up_proj.weight')
    with pytest.raises(CumlxError) as err:
        load_mlx_weights_into(load_hf_model(str(tiny)), renamed)
    assert err.value.code == 'E_INPUT_INVALID' and err.value.details['unused_port_weights'] and err.value.details['missing_in_port']


@pytest.mark.e2e
def test_conv_weights_are_compared_in_torch_layout(tmp_path, e2e):
    import mlx.core as mx
    import mlx.nn as nn
    import torch

    from cumlx.verify.capture_torch import _conv_to_torch_layout
    from cumlx.verify.safetensors_io import save_file
    from cumlx.verify.weights import run_weights

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(4, 8, 3)

    net = Net()
    original = np.transpose(np.array(net.conv.weight), (0, 2, 1))
    save_file({'conv.weight': original, 'conv.bias': np.array(net.conv.bias)}, tmp_path / 'm.safetensors')
    rep = run_weights(str(tmp_path / 'm.safetensors'), 'x', model=net)
    assert rep['verdict'] == 'ok', rep['first_failure']
    assert next(r for r in rep['parameters'] if r['name'] == 'conv.weight')['layout']
    net.conv.weight = mx.transpose(mx.array(original[::-1].copy()), (0, 2, 1))  # the same shape with other values must fail
    assert run_weights(str(tmp_path / 'm.safetensors'), 'x', model=net)['first_failure']['name'] == 'conv.weight'
    torch_model = torch.nn.Sequential(torch.nn.Conv1d(4, 8, 3))
    arr = np.ones((8, 3, 4), np.float32)
    assert _conv_to_torch_layout(torch_model, '0.weight', arr, torch_model[0].weight).shape == (8, 4, 3)


@pytest.mark.e2e
def test_mixtral_moe_port_is_fully_compared(tmp_path, e2e):
    """A trained Mixtral MoE stayed `incomplete`: PyTorch `mlp` is `block_sparse_moe` in mlx-lm."""
    import torch
    from transformers import MixtralConfig, MixtralForCausalLM

    from cumlx.verify import faults, mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.diff import run_diff
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    from cumlx.verify.weights import run_weights
    torch.manual_seed(0)
    config = MixtralConfig(vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, tie_word_embeddings=False,
                           initializer_range=0.25)
    path = tmp_path / 'mixtral'
    MixtralForCausalLM(config).float().eval().save_pretrained(path)
    golden = prepare_output_dir(tmp_path / 'g', overwrite=True)
    w = TraceWriter(golden, source={'framework': 'torch'}, max_bytes=1 << 30, golden={'kind': 'reference', 'dtype': 'float32'})
    capture_model(load_hf_model(str(path)), [np.asarray(TOKENS)], w, decode_steps=1)
    w.finish()
    rep = run_diff(str(golden), str(path))
    assert rep['verdict'] == 'ok' and rep['summary']['parametrized_coverage'] == 1.0, rep['first_divergence']
    assert run_weights(str(path), str(path))['verdict'] == 'ok'
    model = mlx_port.load_port(str(path))
    faults.inject(model, 'F9', 'model.layers.1.block_sparse_moe')
    fd = run_diff(str(golden), 'x', model=model)['first_divergence']
    assert fd['module'] == 'model.layers.1.mlp.experts' and fd['basis'] == 'isolated'


@pytest.mark.e2e
def test_a_checkpoint_with_missing_weights_never_makes_a_golden(tmp_path, e2e):
    """transformers logs a missing weight and fills it with random values."""
    from cumlx.errors import CumlxError
    from cumlx.verify.capture_torch import load_hf_model
    from cumlx.verify.fixtures import make_tiny_llama
    from cumlx.verify.safetensors_io import load_file, save_file
    tiny = tmp_path / 'tiny'
    make_tiny_llama(tiny)
    assert load_hf_model(str(tiny)) is not None
    tensors = load_file(tiny / 'model.safetensors')
    del tensors['model.layers.0.self_attn.q_proj.weight']
    save_file(tensors, tiny / 'model.safetensors')
    with pytest.raises(CumlxError) as err:
        load_hf_model(str(tiny))
    assert err.value.code == 'E_MODEL_LOAD' and 'model.layers.0.self_attn.q_proj.weight' in err.value.details['missing']


@pytest.mark.e2e
def test_a_float_input_on_the_cache_is_a_hidden_state_not_tokens(e2e):
    import torch

    from cumlx.verify.capture_torch import _replay_inputs
    inputs, kind, _ = _replay_inputs((), {'hidden_states': torch.ones(1, 2, 4), 'past_key_values': object()})
    assert kind == 'single' and inputs[0][1].is_floating_point()  # capture_model then labels it hidden_cached in decode


@pytest.mark.e2e
def test_quantized_mixtral_golden_holds_the_quantized_experts(tmp_path, e2e):
    """Regression: block_sparse_moe names did not reach the PyTorch experts."""
    import mlx.nn as nn
    import torch
    from transformers import MixtralConfig, MixtralForCausalLM

    from cumlx.verify import mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from cumlx.verify.diff import run_diff
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    torch.manual_seed(0)
    config = MixtralConfig(vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, tie_word_embeddings=False,
                           initializer_range=0.25)
    path = tmp_path / 'mixtral'
    MixtralForCausalLM(config).float().eval().save_pretrained(path)
    port = mlx_port.load_port(str(path))
    nn.quantize(port, group_size=32, bits=4, class_predicate=lambda p, m: hasattr(m, 'to_quantized') and 'lm_head' not in p and 'embed' not in p)
    original = load_hf_model(str(path))
    before = original.model.layers[0].mlp.experts.down_proj.detach().clone()
    info = load_mlx_weights_into(original, mlx_port.dequantized_weights(port))
    assert not info['unused_port_weights'] and not torch.equal(before, original.model.layers[0].mlp.experts.down_proj)
    golden = prepare_output_dir(tmp_path / 'g', overwrite=True)
    w = TraceWriter(golden, source={'framework': 'torch'}, max_bytes=1 << 30, golden={'kind': 'quantization-aware', 'dtype': 'float32'})
    capture_model(original, [np.asarray(TOKENS)], w)
    w.finish()
    rep = run_diff(str(golden), 'q', model=port)
    assert rep['verdict'] == 'ok', rep['first_divergence']


# --------------------------------------------------------------- robot policy
@pytest.mark.e2e
def test_robot_policy_example_passes_and_a_wrong_policy_fails(e2e):
    import importlib.util

    import mlx.nn as nn

    from cumlx import check_op
    spec = importlib.util.spec_from_file_location('verify_policy', ROOT / 'examples' / 'robot-policy' / 'verify_policy.py')
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    layers = policy.actor_layers(policy.random_state())
    assert [w.shape for w, _ in layers][0] == (512, 48) and layers[-1][0].shape == (12, 128)
    x = [np.random.default_rng(0).normal(size=(16, 48)).astype(np.float32)]
    good = check_op(policy.torch_policy(layers), policy.mlx_policy(layers), x)
    assert good['verdict'] == 'ok' and good['worst_rel_l2'] < 1e-4
    bad = check_op(policy.torch_policy(layers), policy.mlx_policy(layers, activation=nn.relu), x)
    assert bad['verdict'] == 'diverged'
    with pytest.raises(SystemExit):
        policy.actor_layers({'critic.0.weight': None})
