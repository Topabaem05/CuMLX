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
    assert set(tools) == {f'cumlx_{c}' for c in COMMANDS}
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
