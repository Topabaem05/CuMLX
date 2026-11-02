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


def _capture(tiny, dst, weights=None, samples=None):
    from cumlx.verify.capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from cumlx.verify.trace import TraceWriter, prepare_output_dir
    model = load_hf_model(str(tiny))
    if weights is not None:
        info = load_mlx_weights_into(model, weights)
        assert info['loaded'] > 0
    out = prepare_output_dir(dst, overwrite=True)
    w = TraceWriter(out, source={'framework': 'torch'}, max_bytes=1 << 30,
                    golden={'kind': 'quantization-aware' if weights is not None else 'reference', 'dtype': 'float32'})
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
    assert not recs['model.layers.0.self_attn'].replayable  # hidden states + mask + position embeddings
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


@pytest.mark.parametrize('fid,mode,gs', [('F4', 'mxfp4', 32), ('F5', 'nvfp4', 16)])
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
    assert cap['ok'] and cap['data']['records'] > 0
    assert cap['next_steps'][0]['command'].startswith('cumlx diff')
    d = cli('diff', str(tmp_path / 'g'), str(tmp_path / 'tiny'), '-o', str(tmp_path / 'r'))
    assert d['ok'] and d['data']['verdict'] == 'ok'
    assert (tmp_path / 'r' / 'report.json').is_file() and (tmp_path / 'r' / 'report.md').is_file()
    again = cli('diff', str(tmp_path / 'g'), str(tmp_path / 'tiny'), '-o', str(tmp_path / 'r'))
    assert again['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    assert cli('report', str(tmp_path / 'r'))['data']['verdict'] == 'ok'
