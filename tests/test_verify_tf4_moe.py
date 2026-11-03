"""MoE goldens captured with transformers 4, which keeps one module per expert (``experts.3.up_proj``).

The port stacks every expert in one ``switch_mlp``. Each expert must be replayed alone on the tokens that the
original routed to it, so a defect in one expert of one layer is located there.

The capture runs in a subprocess with ``CUMLX_TF4_SITE`` first on PYTHONPATH; this process keeps transformers 5
for nothing more than imports, and the diff itself does not use transformers. Skipped without that directory;
``pytest --require-tf4`` turns the skip into a failure (the CI verifier job does that).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).resolve().parents[1]
TOKENS = '1,5,9,23,7,99,3,64,12,8'

MIXTRAL = '''
import sys, torch
from transformers import MixtralConfig, MixtralForCausalLM
torch.manual_seed(0)
cfg = MixtralConfig(vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
                    rms_norm_eps=1e-5, tie_word_embeddings=False, initializer_range=0.25, rope_theta=10000.0,
                    sliding_window=None)
model = MixtralForCausalLM(cfg).float().eval()
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith('norm.weight'):
            p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
        elif name.endswith('.gate.weight'):  # distinct routing per token, so an expert fault shows
            p.copy_(torch.randn(p.shape, generator=gen))
model.save_pretrained(sys.argv[1], safe_serialization=True)
'''


def _run(args, site=None, check=True):
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(x for x in (site, str(ROOT / 'src')) if x),
           'CUMLX_FORMAT': 'json'}
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        raise AssertionError(f'{args} exited {p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}')
    return p


def _capture(site, model, out, *extra):
    doc = json.loads(_run(['-m', 'cumlx', 'capture', str(model), '--token-ids', TOKENS, '-o', str(out), *extra],
                          site).stdout)
    assert doc['ok'], doc
    return doc


def _quantized_port(src, dst):
    from mlx_lm.utils import load_model, save_config, save_model
    import mlx.nn as nn
    model, config = load_model(Path(src))
    nn.quantize(model, group_size=32, bits=4, mode='mxfp4',
                class_predicate=lambda p, m: hasattr(m, 'to_quantized') and 'switch_mlp' in p)
    config['quantization'] = {'group_size': 32, 'bits': 4, 'mode': 'mxfp4'}
    save_model(Path(dst), model)
    save_config(config, Path(dst) / 'config.json')


@pytest.fixture(scope='module')
def qwen3(tf4_site, tmp_path_factory):
    root = tmp_path_factory.mktemp('tf4-qwen3')
    _run(['-m', 'cumlx', 'fixture', 'tiny-moe', '-o', str(root / 'model')], tf4_site)
    _capture(tf4_site, root / 'model', root / 'golden')
    return root / 'model', root / 'golden'


@pytest.fixture(scope='module')
def mixtral(tf4_site, tmp_path_factory):
    root = tmp_path_factory.mktemp('tf4-mixtral')
    _run(['-c', MIXTRAL, str(root / 'model')], tf4_site)
    _capture(tf4_site, root / 'model', root / 'golden')
    return root / 'model', root / 'golden'


def _expert_records(golden):
    from cumlx.verify.trace import Trace
    return [r for s in Trace(golden).manifest.samples for r in s.records if '.experts.' in r.name]


@pytest.mark.parametrize('which', ['qwen3', 'mixtral'])
def test_clean_port_verifies_every_expert_module(which, request):
    from cumlx.verify.diff import run_diff
    path, golden = request.getfixturevalue(which)
    assert _expert_records(golden), 'transformers 4 records one module per expert'
    rep = run_diff(str(golden), str(path), device='cpu')
    assert rep['verdict'] == 'ok', rep['first_divergence']
    assert rep['summary']['unverified'] == 0 and rep['summary']['parametrized_coverage'] == 1.0
    experts = [e for e in rep['records'] if '.experts.' in e['name'] and e['has_params']]
    assert experts and all(e['status'] == 'pass' and e['isolated']['rel_l2'] < 1e-4 for e in experts)
    assert all('switch_mlp[' in e['mlx_name'] for e in experts)
    assert {e['phase'] for e in experts} == {'prefill', 'decode'}
    if which == 'mixtral':  # w1 / w3 / w2 are gate / up / down
        names = {e['name'].rsplit('.', 1)[-1]: e['mlx_name'].rsplit('.', 1)[-1] for e in experts}
        assert names['w1'] == 'gate_proj' and names['w3'] == 'up_proj' and names['w2'] == 'down_proj'


def test_one_wrong_expert_is_located_in_its_layer(qwen3):
    import mlx.core as mx
    from cumlx.verify import mlx_port
    from cumlx.verify.diff import run_diff
    path, golden = qwen3
    model = mlx_port.load_port(str(path))
    up = mlx_port.named_modules(model)['model.layers.0.mlp.switch_mlp.up_proj']
    up.weight = mx.concatenate([up.weight[:2], up.weight[2:3] * 1.05, up.weight[3:]], axis=0)
    rep = run_diff(str(golden), 'x', model=model, device='cpu')
    fd = rep['first_divergence']
    assert (fd['module'], fd['basis'], fd['confidence']) == ('model.layers.0.mlp.experts.2.up_proj', 'isolated', 'high')
    own = {e['name'] for e in rep['records'] if e.get('fail_cause') == 'self' and '.experts.' in e['name']}
    assert own == {'model.layers.0.mlp.experts.2.up_proj', 'model.layers.0.mlp.experts.2'}


def test_rotated_experts_fail_and_the_router_stays_correct(qwen3):
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    path, golden = qwen3
    model = mlx_port.load_port(str(path))
    faults.inject(model, 'F9', 'model.layers.1.mlp')
    rep = run_diff(str(golden), 'x', model=model, device='cpu')
    fd = rep['first_divergence']
    assert fd['module'].startswith('model.layers.1.mlp.experts.') and fd['basis'] == 'isolated'
    gate = [e for e in rep['records'] if e['name'] == 'model.layers.1.mlp.gate']
    assert gate and all(e['status'] == 'pass' for e in gate)


def test_chain_mode_leaves_experts_unverified_not_passed(qwen3):
    from cumlx.verify.diff import run_diff
    path, golden = qwen3
    rep = run_diff(str(golden), str(path), mode='chain', device='cpu')
    rows = [e for e in rep['records'] if '.experts.' in e['name'] and e['has_params']]
    assert rep['verdict'] == 'incomplete'
    assert rows and all(e['status'] == 'unverified' and '--mode isolated' in e['note'] for e in rows)


def test_quantized_mixtral_with_port_weights_golden_and_scale_fault(tf4_site, mixtral, tmp_path):
    from cumlx.verify import faults, mlx_port
    from cumlx.verify.diff import run_diff
    path, _ = mixtral
    _quantized_port(path, tmp_path / 'q')
    doc = _capture(tf4_site, path, tmp_path / 'golden-q', '--weights-from-mlx', str(tmp_path / 'q'))
    assert doc['data']['weights_from_mlx']['missing_in_port']['total'] == 0
    assert run_diff(str(tmp_path / 'golden-q'), str(tmp_path / 'q'), device='cpu')['verdict'] == 'ok'
    model = mlx_port.load_port(str(tmp_path / 'q'))
    faults.inject(model, 'F4', 'model.layers.0.block_sparse_moe.switch_mlp.up_proj')  # first scale = expert 0
    fd = run_diff(str(tmp_path / 'golden-q'), 'q', model=model, device='cpu')['first_divergence']
    assert (fd['module'], fd['basis']) == ('model.layers.0.block_sparse_moe.experts.0.w3', 'isolated')
