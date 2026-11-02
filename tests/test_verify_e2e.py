"""End-to-end verifier tests: PyTorch golden -> mlx-lm port -> diff, fault injection, conform.

Skipped without torch/transformers/mlx/mlx-lm; `pytest --require-e2e` turns the skip into a failure.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[1]
TOKENS = [1, 5, 9, 23, 7, 99, 3, 64, 12, 8]


@pytest.fixture(scope='module')
def tiny(tmp_path_factory, e2e):
    from cumlx.verify.fixtures import make_tiny_llama
    path = tmp_path_factory.mktemp('fx') / 'tiny'
    make_tiny_llama(path)
    return path


def _capture(tiny, dst, weights=None, samples=None, kind=None):
    from cumlx.verify.capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    model = load_hf_model(str(tiny))
    if weights is not None:
        info = load_mlx_weights_into(model, weights)
        assert info['loaded'] > 0
    out = prepare_output_dir(dst, overwrite=True)
    w = TraceWriter(out, source={'framework': 'torch'}, max_bytes=1 << 30,
                    golden={'kind': kind or ('quantization-aware' if weights is not None else 'reference'),
                            'dtype': 'float32'})
    stats = capture_model(model, samples or [np.asarray(TOKENS)], w)
    w.finish()
    return out, stats


@pytest.fixture(scope='module')
def golden(tiny, tmp_path_factory):
    out, stats = _capture(tiny, tmp_path_factory.mktemp('g') / 'golden',
                          samples=[np.asarray(TOKENS), np.asarray(TOKENS[:4])])
    assert stats['samples'] == 2 and stats['replayable'] > 0
    return out


def test_trace_records_hf_structure(golden):
    from cumlx.verify.trace import Trace
    t = Trace(golden)
    recs = {r.name: r for r in t.manifest.samples[0].records}
    assert recs['model.layers.0.self_attn.q_proj'].replayable
    assert recs['model.layers.0.self_attn.q_proj'].replay_kind == 'single'
    assert recs['model.layers.0.self_attn'].replay_kind == 'hidden'  # hidden state + mask + position embeddings
    assert recs['model.rotary_emb'].replay_kind is None
    assert recs[''].output.source == 'ModelOutput[0]'  # root logits
    assert t.token_ids(t.manifest.samples[1]).shape == (1, 4)


@pytest.mark.parametrize('mode', ['isolated', 'chain'])
def test_clean_port_passes(golden, tiny, mode):
    from cumlx.verify.diff import run_diff
    rep = run_diff(str(golden), str(tiny), mode=mode)
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['summary']['fail'] == 0 and rep['summary']['unverified'] == 0
    assert rep['summary']['parametrized_coverage'] == 1.0
    assert (rep['summary']['isolated_compared'] > 0) == (mode == 'isolated')
    worst = max(e['chain']['rel_l2'] for e in rep['records'] if e.get('chain'))
    assert worst < 1e-5


@pytest.mark.parametrize('fid', ['F1', 'F2', 'F3', 'F6', 'F7', 'F8'])
def test_injected_fault_is_located(golden, tiny, fid):
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    model = mlx_port.load_port(str(tiny))
    target = faults.TINY_LLAMA_TARGETS[fid]
    faults.inject(model, fid, target)
    rep = run_diff(str(golden), str(tiny), model=model, mode='isolated')
    fd = rep['first_divergence']
    assert rep['verdict'] == 'diverged' and fd['module'] == target, fd
    if fid in ('F1', 'F2', 'F3', 'F6'):
        assert fd['basis'] == 'isolated' and fd['confidence'] == 'high'
    # Records before the fault, in completion order, must still pass.
    first_fail = next(e['order'] for e in rep['records'] if e['sample'] == 's0000' and e['status'] == 'fail')
    assert all(e['status'] in ('pass', 'skipped') for e in rep['records']
               if e['sample'] == 's0000' and e['order'] < first_fail)


def test_chain_mode_points_to_rerun_isolated(golden, tiny):
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    model = mlx_port.load_port(str(tiny))
    faults.inject(model, 'F7', faults.TINY_LLAMA_TARGETS['F7'])
    fd = run_diff(str(golden), str(tiny), model=model, mode='chain')['first_divergence']
    assert fd['basis'] == 'chain' and '--mode isolated' in fd['explanation']


@pytest.mark.parametrize('fid,mode,gs', [('F4', 'mxfp4', 32), ('F5', 'nvfp4', 16), ('F10', 'nvfp4', 16)])
def test_quantized_port_with_quantization_aware_golden(tiny, tmp_path, fid, mode, gs):
    import mlx.nn as nn
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff

    def port():
        m = mlx_port.load_port(str(tiny))
        nn.quantize(m, group_size=gs, bits=4, mode=mode,
                    class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and 'lm_head' not in p)
        return m

    clean = port()
    assert mlx_port.port_profile(clean)['profile'] == 'quantized'
    ref_golden, _ = _capture(tiny, tmp_path / 'ref')
    rep = run_diff(str(ref_golden), 'q', model=clean)
    assert rep['verdict'] == 'diverged'  # quantization loss vs a reference golden ...
    assert any(w['code'] == 'W_QUANTIZED_PORT_REFERENCE_GOLDEN' for w in rep['warnings'])  # ... is flagged
    q_golden, _ = _capture(tiny, tmp_path / 'q', weights=mlx_port.dequantized_weights(clean))
    assert run_diff(str(q_golden), 'q', model=clean)['verdict'] == 'ok'
    faulty = port()
    faults.inject(faulty, fid, faults.TINY_LLAMA_TARGETS[fid])
    fd = run_diff(str(q_golden), 'q', model=faulty)['first_divergence']
    assert fd['module'] == faults.TINY_LLAMA_TARGETS[fid] and fd['basis'] == 'isolated'


def test_unmapped_module_is_unverified_not_pass(golden, tiny, tmp_path):
    from cumlx.verify.diff import run_diff
    rules = tmp_path / 'rules.json'
    rules.write_text(json.dumps({'rules': [{'match': r'mlp\.up_proj$', 'replace': 'mlp.missing'}]}))
    rep = run_diff(str(golden), str(tiny), mapping=str(rules))
    assert rep['verdict'] == 'incomplete' and rep['summary']['unverified'] >= 2
    assert rep['summary']['parametrized_coverage'] < 1.0
    rules.write_text(json.dumps({'ignore': [r'mlp\.up_proj$']}))
    rep = run_diff(str(golden), str(tiny), mapping=str(rules))
    assert rep['verdict'] == 'ok' and rep['summary']['ignored'] >= 2


def test_dry_run_reports_mapping(golden, tiny):
    from cumlx.verify.diff import run_diff
    rep = run_diff(str(golden), str(tiny), dry_run=True)
    assert rep['dry_run'] and not rep['mapping']['unmapped_with_params']
    assert rep['port']['profile'] == 'float32'


def test_conform_against_mlx(e2e):
    from cumlx.verify.conform import run_conform
    rep = run_conform()
    by_id = {c['id']: c for c in rep['checks']}
    assert by_id['decode_bitwise.nvfp4']['compared'] > 0 and by_id['decode_bitwise.mxfp4']['compared'] > 0
    for c in rep['checks']:
        assert c['verdict'] in ('pass', 'fail', 'info')
        if c['verdict'] == 'fail':
            assert c['kind'] == 'convention' or c['mismatches'] > 0


def test_cli_end_to_end(tmp_path, e2e):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUMLX_FORMAT': 'json'}

    def cli(*argv):
        p = subprocess.run([sys.executable, '-m', 'cumlx', *argv], capture_output=True, text=True, env=env)
        doc = json.loads(p.stdout)
        assert doc['exit_code'] == p.returncode
        return doc

    assert cli('fixture', 'tiny-llama', '-o', str(tmp_path / 'tiny'))['ok']
    cap = cli('capture', str(tmp_path / 'tiny'), '--token-ids', '1,5,9,23', '-o', str(tmp_path / 'g'))
    assert cap['ok'] and cap['data']['records'] > 0 and cap['data']['samples'] == 1
    assert cap['next_steps'][0]['command'].startswith('cumlx diff')
    d = cli('diff', str(tmp_path / 'g'), str(tmp_path / 'tiny'), '-o', str(tmp_path / 'r'))
    assert d['ok'] and d['data']['verdict'] == 'ok'
    assert (tmp_path / 'r' / 'report.json').is_file() and (tmp_path / 'r' / 'report.md').is_file()
    again = cli('diff', str(tmp_path / 'g'), str(tmp_path / 'tiny'), '-o', str(tmp_path / 'r'))
    assert again['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    assert cli('report', str(tmp_path / 'r'))['data']['verdict'] == 'ok'


def faults_target(fid):
    from cumlx.verify import faults
    return faults.TINY_LLAMA_TARGETS[fid]


def _bf16_port(path, fault=None, target=None):
    import mlx.core as mx
    from cumlx.verify import faults, mlx_port
    model = mlx_port.load_port(str(path))
    model.set_dtype(mx.bfloat16)
    if fault:
        faults.inject(model, fault, target)
    return model


def _rope_factor(model, name, factor):
    import mlx.nn as nn
    from cumlx.verify import mlx_port
    attn = mlx_port.named_modules(model)[name]
    old = attn.rope
    attn.rope = nn.RoPE(old.dims, traditional=old.traditional, base=old.base * factor, scale=old.scale)


def test_multi_input_modules_are_replayed_in_isolation(golden, tiny):
    from cumlx.verify.diff import run_diff
    rep = run_diff(str(golden), str(tiny))
    by_name = {e['name']: e for e in rep['records'] if e['sample'] == 's0000'}
    for name in ('model.layers.0.self_attn', 'model.layers.1'):
        assert by_name[name]['isolated']['verdict'] == 'pass', by_name[name]
    assert rep['execution']['reduced_precision_matmul'] is False
    assert rep['execution']['gemm_noise_rel_l2'] < 1e-5


def test_trace_without_hidden_inputs_falls_back_to_chain_localization(tiny, tmp_path, golden):
    """A trace from before hidden-state replay: attention is not replayable and is judged by chain, not skipped."""
    import shutil
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    old = tmp_path / 'old'
    shutil.copytree(golden, old)
    m = json.loads((old / 'manifest.json').read_text())
    for s in m['samples']:
        for r in s['records']:
            if r.get('replay_kind') == 'hidden':
                r.update(replay_kind=None, replayable=False, inputs=[])
    (old / 'manifest.json').write_text(json.dumps(m))
    assert run_diff(str(old), str(tiny))['verdict'] == 'ok'  # correct port
    model = mlx_port.load_port(str(tiny))
    target = faults.TINY_LLAMA_TARGETS['F7']
    faults.inject(model, 'F7', target)
    fd = run_diff(str(old), str(tiny), model=model)['first_divergence']  # faulty port
    assert (fd['module'], fd['basis'], fd['confidence']) == (target, 'chain+isolated', 'medium')


def test_low_precision_port_is_judged_on_rounding_noise_not_chain_drift(golden, tiny):
    """float32-origin weights rounded to bf16: a clean port passes and a clear fault is located; the limit is stated."""
    from cumlx.verify import faults
    from cumlx.verify.diff import run_diff
    rep = run_diff(str(golden), 'bf16', model=_bf16_port(tiny))
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['port']['profile'] == 'bfloat16' and rep['summary']['native_output']['positions'] == 14
    assert 1e-4 < rep['threshold']['composite_rel_l2'] <= rep['threshold']['composite_rel_l2_max']
    assert any(w['code'] == 'W_WEIGHT_ROUNDING_NOISE' for w in rep['warnings'])
    for fid in ('F1', 'F2', 'F8'):  # F3, F6 and F7 stay under the rounding noise here; see the port-weights test
        target = faults.TINY_LLAMA_TARGETS[fid]
        fd = run_diff(str(golden), 'bf16', model=_bf16_port(tiny, fid, target))['first_divergence']
        assert fd['module'] == target and fd['basis'] == 'isolated', (fid, fd)


def test_one_defective_leaf_cannot_loosen_its_own_threshold(tiny, tmp_path):
    """Regression: with the maximum as the noise estimate, a 1% weight error raised every threshold and passed."""
    import mlx.core as mx
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    lossless = _bf16_port(tiny)
    gdir, _ = _capture(tiny, tmp_path / 'r', weights=mlx_port.dequantized_weights(lossless), kind='reference')
    rep = run_diff(str(gdir), 'bf16', model=lossless)
    assert rep['verdict'] == 'ok' and rep['threshold']['composite_rel_l2'] == 1e-4
    assert not any(w['code'] == 'W_WEIGHT_ROUNDING_NOISE' for w in rep['warnings'])
    faulty = _bf16_port(tiny)
    target = 'model.layers.1.mlp.gate_proj'
    mod = mlx_port.named_modules(faulty)[target]
    mod.weight = (mod.weight.astype(mx.float32) * 1.01).astype(mx.bfloat16)
    rep = run_diff(str(gdir), 'bf16', model=faulty)
    assert rep['verdict'] == 'diverged' and rep['first_divergence']['module'] == target
    assert rep['threshold']['composite_rel_l2'] == 1e-4


def test_promotion_restores_low_precision_parameters(golden, tiny):
    import mlx.core as mx
    from cumlx.verify.diff import run_diff
    model = _bf16_port(tiny)
    run_diff(str(golden), 'bf16', model=model)
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16


def test_non_finite_native_output_is_a_failure_not_drift(golden, tiny):
    """Correct in float32 replay but NaN in native bf16 arithmetic: chain drift must not hide this."""
    import mlx.core as mx
    from cumlx.verify.diff import run_diff
    model = _bf16_port(tiny)
    norm = model.model.layers[1].input_layernorm
    cls = type(norm)

    def call(self, x):
        y = cls.__call__(self, x)
        return y * mx.array(float('nan'), y.dtype) if y.dtype == mx.bfloat16 else y
    norm.__class__ = type(cls.__name__, (cls,), {'__call__': call})
    rep = run_diff(str(golden), 'bf16', model=model)
    fd = rep['first_divergence']
    assert rep['verdict'] == 'diverged' and fd['module'] == 'model.layers.1.input_layernorm'
    assert fd['basis'] == 'chain' and fd['metrics']['reason'] == 'nonfinite_output'


def test_port_weights_golden_is_strict_for_any_dtype(tiny, tmp_path):
    """The same weights in the golden remove rounding noise: a small composite defect fails only with this golden."""
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    clean = _bf16_port(tiny)
    gdir, _ = _capture(tiny, tmp_path / 'p', weights=mlx_port.dequantized_weights(clean))
    rep = run_diff(str(gdir), 'bf16', model=clean)
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['threshold']['isolated_rel_l2'] == rep['threshold']['composite_rel_l2_max'] == 1e-4
    assert rep['summary']['worst_isolated_rel_l2'] < 1e-4
    faulty = _bf16_port(tiny)
    _rope_factor(faulty, 'model.layers.1.self_attn', 1.01)
    fd = run_diff(str(gdir), 'bf16', model=faulty)['first_divergence']
    assert fd is not None and fd['module'] == 'model.layers.1.self_attn' and fd['basis'] == 'isolated'
    # These stay under the rounding noise of a reference golden on this fixture, and are located with this one.
    for fid in ('F3', 'F6', 'F7'):
        target = faults_target(fid)
        fd = run_diff(str(gdir), 'bf16', model=_bf16_port(tiny, fid, target))['first_divergence']
        assert fd is not None and fd['module'] == target and fd['basis'] == 'isolated', (fid, fd)


def test_quantized_port_with_bf16_scales_uses_exact_golden_weights(tiny, tmp_path):
    """Regression: bf16 scales made mx.dequantize round in bf16, so the golden held the wrong weights."""
    import mlx.core as mx
    import mlx.nn as nn
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    port = mlx_port.load_port(str(tiny))
    port.set_dtype(mx.bfloat16)
    nn.quantize(port, group_size=32, bits=4, class_predicate=lambda p, m: isinstance(m, nn.Linear) and 'lm_head' not in p)
    q = port.model.layers[0].self_attn.q_proj
    assert q['scales'].dtype == mx.bfloat16
    gdir, _ = _capture(tiny, tmp_path / 'q', weights=mlx_port.dequantized_weights(port))
    rep = run_diff(str(gdir), 'q', model=port)
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['summary']['worst_isolated_rel_l2'] < 1e-4


def test_tied_lm_head_is_verified_through_as_linear(tmp_path, e2e):
    import mlx.core as mx
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    from cumlx.verify.fixtures import make_tiny_llama
    tied = tmp_path / 'tied'
    make_tiny_llama(tied, tie_word_embeddings=True)
    golden_dir, _ = _capture(tied, tmp_path / 'g')
    from cumlx.verify.trace import Trace
    assert Trace(golden_dir).manifest.source['tied_modules'] == {'lm_head': 'model.embed_tokens'}
    rep = run_diff(str(golden_dir), str(tied))
    assert rep['verdict'] == 'ok' and rep['summary']['unverified'] == 0
    head = next(e for e in rep['records'] if e['name'] == 'lm_head')
    assert head['tied_to'] == 'model.embed_tokens' and head['isolated']['verdict'] == 'pass'
    assert head['chain']['verdict'] == 'pass'
    # Incorrect case: the shared weight is applied wrongly only on the output side.
    model = mlx_port.load_port(str(tied))
    embed = model.model.embed_tokens
    cls = type(embed)
    embed.__class__ = type(cls.__name__, (cls,), {'as_linear': lambda self, x: cls.as_linear(self, x) * mx.array(1.5)})
    fd = run_diff(str(golden_dir), str(tied), model=model)['first_divergence']
    assert fd['module'] == 'lm_head' and fd['basis'] == 'isolated'


def test_gpu_request_never_falls_back_to_cpu(golden, tiny, monkeypatch):
    from cumlx.errors import CumlxError
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    monkeypatch.setattr(mlx_port, 'gpu_available', lambda mx: False)
    with pytest.raises(CumlxError) as err:
        run_diff(str(golden), str(tiny), device='gpu')
    assert err.value.code == 'E_BACKEND_UNAVAILABLE'
    assert run_diff(str(golden), str(tiny), device='cpu')['execution']['device'] == 'cpu'


def _reduced_precision_hardware() -> bool:
    code = ('import os; os.environ["MLX_ENABLE_TF32"]="1"; import mlx.core as mx; '
            'from cumlx.verify.mlx_port import measure_gemm_noise as m; print(m(mx, mx.default_device()))')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, check=True)
    return float(out.stdout.strip().splitlines()[-1]) > 1e-5


def test_tf32_on_is_measured_and_raises_thresholds_with_a_warning(golden, tiny):
    """Needs a separate process: MLX reads the setting once. Without TF32 hardware, nothing can be proved."""
    if not _reduced_precision_hardware():
        pytest.skip('this MLX device has no reduced-precision float32 matmul')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUMLX_FORMAT': 'json'}
    p = subprocess.run([sys.executable, '-m', 'cumlx', 'diff', str(golden), str(tiny), '--tf32', 'on'],
                       capture_output=True, text=True, env=env)
    doc = json.loads(p.stdout)
    assert doc['data']['execution']['reduced_precision_matmul'] is True
    assert doc['data']['threshold']['raised_for_tf32'] and doc['data']['threshold']['isolated_rel_l2'] > 1e-2
    assert any(w['code'] == 'W_TF32_ACTIVE' for w in doc['warnings']) and doc['data']['verdict'] == 'ok'


def test_full_precision_request_is_refused_if_process_already_runs_tf32(tiny, golden):
    if not _reduced_precision_hardware():
        pytest.skip('this MLX device has no reduced-precision float32 matmul')
    code = ('import os; os.environ["MLX_ENABLE_TF32"]="1"; import json, sys, mlx.core as mx\n'
            'from cumlx.errors import CumlxError\nfrom cumlx.verify.diff import run_diff\n'
            'mx.eval(mx.matmul(mx.ones((4,8)), mx.ones((8,8)), stream=mx.gpu))\n'
            'try:\n    run_diff(sys.argv[1], sys.argv[2])\nexcept CumlxError as e:\n    print("CODE", e.code)\n')
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    out = subprocess.run([sys.executable, '-c', code, str(golden), str(tiny)], capture_output=True, text=True, env=env)
    if hasattr(__import__('mlx.core', fromlist=['x']), 'config'):
        pytest.skip('this MLX can change the setting inside a running process')
    assert 'CODE E_BACKEND_UNAVAILABLE' in out.stdout, out.stdout + out.stderr


def test_static_weights_pass_clean_ports_and_name_defects(tiny):
    import mlx.core as mx
    import mlx.nn as nn
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.weights import run_weights
    assert run_weights(str(tiny), str(tiny))['verdict'] == 'ok'
    bf16 = mlx_port.load_port(str(tiny))
    bf16.set_dtype(mx.bfloat16)
    assert run_weights(str(tiny), 'bf16', model=bf16)['verdict'] == 'ok'
    q = mlx_port.load_port(str(tiny))
    nn.quantize(q, group_size=32, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear))
    rep = run_weights(str(tiny), 'q', model=q)
    assert rep['verdict'] == 'ok', rep['first_failure']

    target = 'model.layers.0.self_attn.q_proj'
    transposed = mlx_port.load_port(str(tiny))
    faults.inject(transposed, 'F1', target)
    first = run_weights(str(tiny), 't', model=transposed)['first_failure']
    assert first['name'] == target + '.weight' and first['diagnosis'] == 'transposed'

    for mutate, expected in ((lambda w: w * 2, 'scaled_by_2'), (lambda w: -w, 'sign_flipped')):
        m = mlx_port.load_port(str(tiny))
        mod = mlx_port.named_modules(m)[target]
        mod.weight = mutate(mod.weight)
        first = run_weights(str(tiny), 'm', model=m)['first_failure']
        assert first['name'] == target + '.weight' and first['diagnosis'] == expected

    for mutate in (lambda w: w * 2, lambda w: -w):  # a quantized port hides nothing behind its large rel_l2
        m = mlx_port.load_port(str(tiny))
        mod = mlx_port.named_modules(m)[target]
        mod.weight = mutate(mod.weight)
        nn.quantize(m, group_size=32, bits=8, class_predicate=lambda p, x: isinstance(x, nn.Linear))
        assert run_weights(str(tiny), 'qm', model=m)['first_failure']['name'] == target + '.weight'

    missing = mlx_port.load_port(str(tiny))
    del mlx_port.named_modules(missing)[target]['bias']
    rep = run_weights(str(tiny), 'missing', model=missing)
    assert rep['verdict'] == 'incomplete' and rep['summary']['unverified'] == 1


def test_static_weights_cli(tiny, tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'CUMLX_FORMAT': 'json'}
    out = tmp_path / 'w.json'
    p = subprocess.run([sys.executable, '-m', 'cumlx', 'weights', str(tiny), str(tiny), '-o', str(out)],
                       capture_output=True, text=True, env=env)
    doc = json.loads(p.stdout)
    assert p.returncode == 0 and doc['data']['verdict'] == 'ok' and json.loads(out.read_text())['parameters']


@pytest.fixture(scope='module')
def golden_decode(tiny, tmp_path_factory):
    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    out = prepare_output_dir(tmp_path_factory.mktemp('gd') / 'golden', overwrite=True)
    w = TraceWriter(out, source={'framework': 'torch'}, max_bytes=1 << 30, golden={'kind': 'reference', 'dtype': 'float32'})
    capture_model(load_hf_model(str(tiny)), [np.asarray(TOKENS), np.asarray(TOKENS[:1])], w, decode_steps=2)
    w.finish()
    return out


def test_decode_records_are_replayed_on_the_ports_own_cache(golden_decode, tiny):
    from cumlx.verify.diff import run_diff
    from cumlx.verify.trace import Trace
    trace = Trace(golden_decode)
    sample = trace.manifest.samples[0]
    assert sample.decode_steps == 2 and trace.tensor(sample, 'decode_token_ids').shape == (1, 2)
    kinds = {r.replay_kind for r in sample.records if r.phase == 'decode'}
    assert {'hidden_cached', 'tokens_cached', 'single'} <= kinds
    rep = run_diff(str(golden_decode), str(tiny))
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['summary']['decode_records'] > 0 and rep['summary']['unverified'] == 0
    decode = [e for e in rep['records'] if e['phase'] == 'decode' and e['type'] == 'LlamaAttention']
    assert decode and all(e['isolated']['verdict'] == 'pass' and e['chain']['verdict'] == 'pass' for e in decode)
    assert run_diff(str(golden_decode), str(tiny), mode='chain')['verdict'] == 'ok'


def test_cache_fault_is_invisible_in_prefill_and_located_in_decode(golden, golden_decode, tiny):
    """Incorrect case: F11 passes a prefill-only golden. That is why the decode phase exists."""
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    target = faults.TINY_LLAMA_TARGETS['F11']
    def port():
        m = mlx_port.load_port(str(tiny))
        faults.inject(m, 'F11', target)
        return m
    assert run_diff(str(golden), 'x', model=port())['verdict'] == 'ok'
    rep = run_diff(str(golden_decode), 'x', model=port())
    fd = rep['first_divergence']
    assert rep['verdict'] == 'diverged', fd
    assert (fd['module'], fd['basis'], fd['confidence'], fd['phase']) == (target, 'isolated', 'high', 'decode')
    assert all(e['status'] != 'fail' for e in rep['records'] if e['phase'] == 'prefill')


def test_port_without_kv_cache_leaves_decode_unverified_not_passed(golden_decode, tiny, monkeypatch):
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    def no_cache(model):
        raise TypeError('this port has no KV cache')
    monkeypatch.setattr(mlx_port, 'make_caches', no_cache)
    rep = run_diff(str(golden_decode), str(tiny), mode='chain')
    assert rep['verdict'] == 'incomplete' and any(w['code'] == 'W_DECODE_NOT_RUN' for w in rep['warnings'])
    assert all(e['status'] in ('unverified', 'skipped') for e in rep['records'] if e['phase'] == 'decode')
    rep = run_diff(str(golden_decode), str(tiny))  # isolated: only modules that need no cache can still pass
    cached = [e for e in rep['records'] if e['phase'] == 'decode' and e['type'] in ('LlamaAttention', 'LlamaDecoderLayer')]
    assert cached and all(e['status'] == 'unverified' for e in cached) and rep['verdict'] == 'incomplete'


def test_trace_without_decode_stays_readable(golden, tiny):
    from cumlx.verify.diff import run_diff
    from cumlx.verify.trace import Trace
    assert Trace(golden).manifest.samples[0].decode_steps == 0
    assert run_diff(str(golden), str(tiny))['summary']['decode_records'] == 0


@pytest.fixture(scope='module')
def moe(tmp_path_factory, e2e):
    from cumlx.verify.capture_torch import capture_model, load_hf_model
    from cumlx.verify.fixtures import make_tiny_moe
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    root = tmp_path_factory.mktemp('moe')
    path = root / 'tiny'
    make_tiny_moe(path)
    out = prepare_output_dir(root / 'golden', overwrite=True)
    w = TraceWriter(out, source={'framework': 'torch'}, max_bytes=1 << 30, golden={'kind': 'reference', 'dtype': 'float32'})
    capture_model(load_hf_model(str(path)), [np.asarray(TOKENS), np.asarray(TOKENS[:4])], w, decode_steps=1)
    w.finish()
    return path, out


def test_clean_moe_port_passes_with_experts_replayed_on_golden_routing(moe):
    from cumlx.verify.diff import run_diff
    from cumlx.verify.weights import run_weights
    path, golden = moe
    rep = run_diff(str(golden), str(path))
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['summary']['unverified'] == 0 and rep['summary']['parametrized_coverage'] == 1.0
    by_name = {e['name']: e for e in rep['records'] if e['sample'] == 's0000' and e['phase'] == 'prefill'}
    experts = by_name['model.layers.0.mlp.experts']
    assert experts['mlx_name'] == 'model.layers.0.mlp.switch_mlp' and experts['chain'] is None
    assert experts['isolated']['verdict'] == 'pass' and experts['isolated']['rel_l2'] < 1e-4
    assert by_name['model.layers.0.mlp.gate']['isolated']['verdict'] == 'pass'
    assert rep['mapping']['source'] == 'mlx-lm MoE convention'
    weights = run_weights(str(path), str(path))
    assert weights['verdict'] == 'ok' and weights['summary']['extra_in_port'] == 0
    assert any(r['name'].endswith('experts.3.up_proj.weight') and r['port_name'].endswith('up_proj.weight[3]')
               for r in weights['parameters'])


def test_moe_experts_are_unverified_in_chain_mode_not_passed(moe):
    from cumlx.verify.diff import run_diff
    path, golden = moe
    rep = run_diff(str(golden), str(path), mode='chain')
    assert rep['verdict'] == 'incomplete'
    rows = [e for e in rep['records'] if e['name'].endswith('mlp.experts')]
    assert rows and all(e['status'] == 'unverified' and '--mode isolated' in e['note'] for e in rows)


def test_rotated_experts_and_router_faults_are_told_apart(moe):
    import mlx.core as mx
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    from cumlx.verify.weights import run_weights
    path, golden = moe
    inject_at, expected = faults.TINY_MOE_TARGETS['F9']
    model = mlx_port.load_port(str(path))
    faults.inject(model, 'F9', inject_at)
    fd = run_diff(str(golden), 'x', model=model)['first_divergence']
    assert (fd['module'], fd['basis'], fd['confidence']) == (expected, 'isolated', 'high')
    first = run_weights(str(path), 'x', model=model)['first_failure']
    assert 'expert_order' in first['diagnosis'] and '.layers.1.mlp.experts.' in first['name']
    # A fault in the router alone: the experts stay correct on the golden routing.
    router = mlx_port.load_port(str(path))
    gate = mlx_port.named_modules(router)['model.layers.0.mlp.gate']
    gate.weight = mx.concatenate([gate.weight[1:], gate.weight[:1]], axis=0)
    rep = run_diff(str(golden), 'x', model=router)
    fd = rep['first_divergence']
    assert fd['module'] == 'model.layers.0.mlp.gate' and fd['basis'] == 'isolated'
    experts = [e for e in rep['records'] if e['name'] == 'model.layers.0.mlp.experts']
    assert experts and all(e['status'] == 'pass' for e in experts)


@pytest.mark.parametrize('mode,kwargs', [('affine', dict(bits=4, group_size=32)), ('nvfp4', dict(bits=4, group_size=16)),
                                         ('mxfp4', dict(bits=4, group_size=32))])
def test_quantized_moe_port_and_per_expert_scale_fault(moe, tmp_path, mode, kwargs):
    import mlx.core as mx
    import mlx.nn as nn
    from cumlx.verify import mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from cumlx.verify.diff import run_diff
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    from cumlx.verify.weights import run_weights
    path, _ = moe

    def port():
        m = mlx_port.load_port(str(path))
        nn.quantize(m, mode=mode, class_predicate=lambda p, x: hasattr(x, 'to_quantized') and 'switch_mlp' in p, **kwargs)
        return m
    clean = port()
    assert mlx_port.port_profile(clean)['profile'] == 'quantized'
    torch_model = load_hf_model(str(path))
    info = load_mlx_weights_into(torch_model, mlx_port.dequantized_weights(clean))
    assert not info['missing_in_port'] and not info['unused_port_weights']
    g = prepare_output_dir(tmp_path / 'g', overwrite=True)
    w = TraceWriter(g, source={'framework': 'torch'}, max_bytes=1 << 30, golden={'kind': 'quantization-aware', 'dtype': 'float32'})
    capture_model(torch_model, [np.asarray(TOKENS)], w)
    w.finish()
    rep = run_diff(str(g), 'q', model=clean)
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert run_weights(str(path), 'q', model=clean)['verdict'] == 'ok'
    # Incorrect case: the block scale of one expert only is doubled.
    faulty = port()
    proj = mlx_port.named_modules(faulty)['model.layers.0.mlp.switch_mlp.up_proj']
    s = proj['scales']
    bump = 8 if mode == 'nvfp4' else 1
    new = s.astype(mx.int32) + bump if mode != 'affine' else s * 2
    mask = mx.zeros_like(s).astype(mx.bool_)
    mask = mx.concatenate([mx.ones_like(s[:1]).astype(mx.bool_), mask[1:]], axis=0)
    proj['scales'] = mx.where(mask, new, s.astype(new.dtype)).astype(s.dtype)
    fd = run_diff(str(g), 'q', model=faulty)['first_divergence']
    assert fd is not None and fd['module'] == 'model.layers.0.mlp.experts' and fd['basis'] == 'isolated', fd
    first = run_weights(str(path), 'q', model=faulty)['first_failure']
    assert first is not None and first['port_name'].endswith('up_proj.weight[0]') and first['status'] == 'fail', first


def test_nvfp4_moe_with_one_global_scale_per_expert(moe, tmp_path):
    """Each expert keeps its own tensor scale. Golden weights must use it, and swapping two scales must be located."""
    import mlx.core as mx
    from cumlx.errors import CumlxError
    from cumlx.verify import mlx_port
    from cumlx.verify.capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from cumlx.verify.diff import run_diff
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    from cumlx.verify.weights import run_weights
    path, _ = moe

    def port():
        m = mlx_port.load_port(str(path))
        for layer in m.model.layers:
            for name in ('gate_proj', 'up_proj', 'down_proj'):
                proj = getattr(layer.mlp.switch_mlp, name)
                setattr(layer.mlp.switch_mlp, name, proj.to_quantized(group_size=16, bits=4, mode='nvfp4', global_scale=True))
        return m
    clean = port()
    q = clean.model.layers[0].mlp.switch_mlp.up_proj
    assert q.global_scale is not None and q.global_scale.shape == (4,)
    assert float(mx.max(q.global_scale) - mx.min(q.global_scale)) > 0  # the experts really differ
    weights = mlx_port.dequantized_weights(clean)
    original = np.array(mlx_port.load_port(str(path)).model.layers[0].mlp.switch_mlp.up_proj.weight)
    key = 'model.layers.0.mlp.switch_mlp.up_proj.weight'
    assert 'model.layers.0.mlp.switch_mlp.up_proj.global_scale' not in weights
    assert np.linalg.norm(weights[key] - original) / np.linalg.norm(original) < 0.2  # scale is applied per expert
    torch_model = load_hf_model(str(path))
    load_mlx_weights_into(torch_model, weights)
    g = prepare_output_dir(tmp_path / 'g', overwrite=True)
    w = TraceWriter(g, source={'framework': 'torch'}, max_bytes=1 << 30, golden={'kind': 'quantization-aware', 'dtype': 'float32'})
    capture_model(torch_model, [np.asarray(TOKENS)], w)
    w.finish()
    assert run_weights(str(path), 'q', model=clean)['verdict'] == 'ok'
    if not mlx_port.gpu_available(mx):  # MLX gather_qmm has no global scale on the CPU: reported, not passed
        with pytest.raises(CumlxError) as err:
            run_diff(str(g), 'q', model=clean)
        assert err.value.code == 'E_MODEL_RUN'
        return
    rep = run_diff(str(g), 'q', model=clean)
    assert rep['verdict'] == 'ok', rep['first_divergence']
    faulty = port()
    bad = faulty.model.layers[0].mlp.switch_mlp.up_proj
    gs = bad.global_scale
    bad.global_scale = mx.concatenate([gs[1:2], gs[:1], gs[2:]])
    fd = run_diff(str(g), 'q', model=faulty)['first_divergence']
    assert fd['module'] == 'model.layers.0.mlp.experts' and fd['basis'] == 'isolated'


def test_op_missing_on_this_device_is_a_model_error_not_an_internal_bug(moe):
    """MLX gather_mm supports only float32 on the CPU, so a bf16 MoE port cannot run there."""
    import mlx.core as mx
    from cumlx.errors import CumlxError
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    path, golden = moe
    model = mlx_port.load_port(str(path))
    model.set_dtype(mx.bfloat16)
    with pytest.raises(CumlxError) as err:
        run_diff(str(golden), 'bf16', model=model, device='cpu')
    assert err.value.code == 'E_MODEL_RUN' and '--device' in err.value.hint and 'cpu' in err.value.message


def test_rope_base_lost_from_transformers5_config_is_located(tmp_path, e2e):
    """Real defect in released mlx-lm: a config written by transformers 5 keeps rope_theta in rope_parameters.

    The port's RoPE base decides the expected result, so this test stays correct after mlx-lm fixes it.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, tie_word_embeddings=False, initializer_range=0.25)
    if not isinstance(getattr(config, 'rope_parameters', None), dict):
        pytest.skip('this transformers version writes a flat rope_theta')
    config.rope_parameters['rope_theta'] = 500000.0
    torch.manual_seed(0)
    original = LlamaForCausalLM(config).float().eval()
    path = tmp_path / 'tf5'
    original.save_pretrained(path)
    golden_dir, _ = _capture(path, tmp_path / 'g')
    port = mlx_port.load_port(str(path))
    rep = run_diff(str(golden_dir), 'tf5', model=port)
    if port.model.layers[0].self_attn.rope.base == 500000.0:
        assert rep['verdict'] == 'ok', rep['first_divergence']
    else:
        fd = rep['first_divergence']
        assert rep['verdict'] == 'diverged' and fd['module'] == 'model.layers.0.self_attn' and fd['basis'] == 'isolated'


def test_static_weights_compare_a_stored_head_with_a_tied_embedding(tmp_path, e2e):
    """Checkpoints such as Qwen3-0.6B store lm_head.weight, and the port ties it to the embedding."""
    from cumlx.verify import mlx_port
    from cumlx.verify.fixtures import make_tiny_llama
    from cumlx.verify.safetensors_io import load_file, save_file
    from cumlx.verify.weights import run_weights
    tied = tmp_path / 'tied'
    make_tiny_llama(tied, tie_word_embeddings=True)
    port = mlx_port.load_port(str(tied))
    tensors = load_file(tied / 'model.safetensors')
    original = tmp_path / 'original'
    original.mkdir()
    save_file({**tensors, 'lm_head.weight': tensors['model.embed_tokens.weight']}, original / 'model.safetensors')
    rep = run_weights(str(original), 'port', model=port)
    assert rep['verdict'] == 'ok' and rep['summary']['unverified'] == 0
    # Incorrect case: the original head differs, so tying the two is wrong.
    save_file({**tensors, 'lm_head.weight': tensors['model.embed_tokens.weight'] * 2}, original / 'model.safetensors')
    first = run_weights(str(original), 'port', model=port)['first_failure']
    assert first['name'] == 'lm_head.weight' and first['port_name'] == 'model.embed_tokens.weight'
