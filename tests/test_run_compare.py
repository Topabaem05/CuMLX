"""`cumlx rundiff`: right traces get a grade, and wrong, noisy or unrelated traces do not."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import pytest

from cumlx.errors import CumlxError
from cumlx.verify.run_compare import compare_runs

ROOT = Path(__file__).resolve().parent.parent


def _curve(seed, scale=1.0, steps=400, noise=0.002, start=1.0):
    rng = random.Random(seed)
    return [start * scale / (1 + i / 50) + rng.gauss(0, noise) for i in range(steps)]


def _run(seed, values, target='mps', **extra):
    return {'schema': 'cumlx.run-report.v1', 'file': f'{target}{seed}', 'target': target, 'machine': {'torch': '2.14.1'}, 'seed': seed, 'module': None,
            'script_sha256': 'abc', 'script_args': [], 'script_exit_code': 0, 'timed_out': False,
            'loss_trace': {'values': values, 'processes_with_losses': 1}, **extra}


def _side(target, offset=0, **curve):
    return [_run(s, _curve(s + offset, **curve), target) for s in range(3)]


def test_same_values_at_the_first_steps_is_grade_a():
    rep = compare_runs(_side('cuda'), _side('mps'))
    assert (rep['verdict'], rep['grade']) == ('ok', 'A')


def test_same_curve_with_other_random_numbers_is_grade_b():
    rep = compare_runs(_side('cuda'), _side('mps', offset=100))
    assert (rep['verdict'], rep['grade']) == ('ok', 'B')
    assert not rep['numeric']['pass']


def test_a_wrong_first_step_is_not_grade_a_when_the_values_only_look_close():
    golden = _side('cuda')
    candidate = _side('mps')
    for run in candidate:
        run['loss_trace']['values'][1] *= 1 + 5e-4  # An optimizer that is slightly wrong moves step 2 first.
    assert compare_runs(golden, candidate)['numeric']['pass'] is False


def test_a_wrong_curve_diverges_even_when_the_first_steps_agree():
    golden = _side('cuda')
    candidate = _side('mps', scale=1.2)
    for run, reference in zip(candidate, golden):
        run['loss_trace']['values'][:2] = reference['loss_trace']['values'][:2]
    rep = compare_runs(golden, candidate)
    assert (rep['verdict'], rep['grade']) == ('diverged', None)
    assert rep['numeric']['pass'] is True


def test_a_wrong_start_cannot_hide_behind_a_right_end():
    candidate = _side('mps', offset=100)
    for run in candidate:
        run['loss_trace']['values'][:100] = [v * 1.5 for v in run['loss_trace']['values'][:100]]
    rep = compare_runs(_side('cuda'), candidate)
    assert rep['verdict'] == 'diverged'
    assert [w['result'] for w in rep['statistical']['windows']][0] == 'fail'


def test_noise_does_not_pass():
    rep = compare_runs(_side('cuda', noise=0.5, start=1.0), _side('mps', offset=100, noise=0.5, start=1.0))
    assert rep['statistical']['state'] in ('inconclusive', 'fail')
    assert rep['grade'] is None
    assert rep['verdict'] != 'ok'


def test_too_few_seeds_is_incomplete():
    rep = compare_runs(_side('cuda')[:2], _side('mps', offset=100)[:2])
    assert (rep['verdict'], rep['grade']) == ('incomplete', None)
    assert '3 seeds' in rep['reason']


def test_a_seed_that_did_not_reach_the_script_is_one_sample():
    same = _curve(0)
    golden = [_run(s, list(same), 'cuda') for s in range(3)]
    candidate = [_run(s, _curve(s + 100), 'mps') for s in range(3)]
    rep = compare_runs(golden, candidate)
    assert rep['verdict'] == 'incomplete'
    assert 'one sample' in rep['reason']


@pytest.mark.parametrize('change, reason', [
    (lambda r: r['loss_trace']['values'].pop(), 'different numbers'),
    (lambda r: r['loss_trace'].update(processes_with_losses=2), 'more than one process'),
    (lambda r: r['loss_trace'].update(values=[]), 'different numbers'),
])
def test_traces_that_do_not_line_up_are_incomplete(change, reason):
    candidate = _side('mps')
    change(candidate[0])
    rep = compare_runs(_side('cuda'), candidate)
    assert rep['verdict'] == 'incomplete' and reason in rep['reason']


def test_no_losses_at_all_is_incomplete():
    rep = compare_runs([_run(0, [], 'cuda')], [_run(0, [], 'mps')])
    assert rep['verdict'] == 'incomplete' and 'nothing to compare' in rep['reason']


def test_a_non_finite_loss_fails_the_candidate_and_voids_the_golden():
    candidate = _side('mps')
    candidate[1]['loss_trace']['values'][50] = None
    assert compare_runs(_side('cuda'), candidate)['verdict'] == 'diverged'
    assert compare_runs(candidate, _side('cuda'))['verdict'] == 'incomplete'


@pytest.mark.parametrize('change', [
    lambda r: r.update(script_sha256='other'),
    lambda r: r.update(script_args=['--lr', '1']),
    lambda r: r.update(script_exit_code=1),
    lambda r: r.update(timed_out=True),
    lambda r: r.pop('loss_trace'),
    lambda r: r.update(seed=1),
])
def test_reports_that_are_not_comparable_are_refused(change):
    candidate = _side('mps')
    change(candidate[0])
    with pytest.raises(CumlxError) as caught:
        compare_runs(_side('cuda'), candidate)
    assert caught.value.code == 'E_RUN_NOT_COMPARABLE'


# ------------------------------------------------------------------ real runs
TRAINING = """
    import os, torch, torch.nn.functional as F
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    x = torch.randn(512, 8); y = (x @ torch.randn(8, 3)).argmax(1)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3 * float(os.environ.get('LR_FACTOR', 1)))
    for step in range(40):
        i = torch.randint(0, 512, (64,))
        loss = F.cross_entropy(model(x[i].to(device)), y[i].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
"""


def _cumlx(tmp_path, *argv, env=None):
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), **(env or {})}
    done = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '-1'], capture_output=True, text=True,
                          cwd=tmp_path, env=environment, check=False)
    doc = json.loads(done.stdout)
    assert doc['exit_code'] == done.returncode, doc
    return doc


def _train(tmp_path, source, folder, target, seed, env=None, script='train.py'):
    (tmp_path / script).write_text(textwrap.dedent(source))
    (tmp_path / folder).mkdir(exist_ok=True)
    report = tmp_path / folder / f'seed{seed}.json'
    doc = _cumlx(tmp_path, 'run', script, '--target', target, '--trace-losses', '--seed', str(seed), '--report', str(report), '--overwrite', env=env)
    assert doc['exit_code'] == 0, doc
    return json.loads(report.read_text())


@pytest.fixture(scope='module')
def device(e2e):
    import torch
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


@pytest.mark.e2e
def test_each_backward_call_is_recorded_once_on_a_redirected_target(tmp_path, device):
    report = _train(tmp_path, TRAINING, 'a', device, 0)
    assert len(report['loss_trace']['values']) == 40
    assert report['loss_trace']['values'][0] != report['loss_trace']['values'][1]
    assert report['script_sha256'] and report['seed'] == 0


@pytest.mark.e2e
def test_the_seed_changes_a_script_that_sets_its_own_seed(tmp_path, device):
    a = _train(tmp_path, TRAINING, 'a', device, 0)['loss_trace']['values']
    again = _train(tmp_path, TRAINING, 'b', device, 0)['loss_trace']['values']
    other = _train(tmp_path, TRAINING, 'c', device, 1)['loss_trace']['values']
    assert a[0] == pytest.approx(again[0], rel=1e-6)
    assert a[0] != pytest.approx(other[0], rel=1e-3)


@pytest.mark.e2e
def test_the_loss_before_scaling_is_recorded(tmp_path, device):
    report = _train(tmp_path, """
        import torch
        t = torch.ones(1, requires_grad=True, device='cuda')
        scaler = torch.amp.GradScaler('cuda', init_scale=1024.0, enabled=True)
        scaler.scale((t * 2).sum()).backward()
    """, 'a', device, 0)
    assert report['loss_trace']['values'] == [2.0]


@pytest.mark.e2e
def test_rundiff_passes_a_right_run_and_rejects_a_wrong_learning_rate(tmp_path, device):
    for seed in range(3):
        _train(tmp_path, TRAINING, 'golden', 'cpu', seed)
        _train(tmp_path, TRAINING, 'right', device, seed)
        _train(tmp_path, TRAINING, 'wrong', device, seed, env={'LR_FACTOR': '0.3'})
    right = _cumlx(tmp_path, 'rundiff', 'golden', 'right')
    assert right['exit_code'] == 0 and right['data']['grade'] in ('A', 'B')
    assert 'W_RUNDIFF_GOLDEN_NOT_CUDA' in {w['code'] for w in right['warnings']}
    wrong = _cumlx(tmp_path, 'rundiff', 'golden', 'wrong')
    assert wrong['exit_code'] == 1 and wrong['data']['grade'] is None
    assert wrong['data']['numeric']['pass'] is False


@pytest.mark.e2e
def test_target_cuda_redirects_nothing(tmp_path, e2e):
    """The golden path. `cumlx run --target cuda` refuses a Mac, so the child is started as it would be on a CUDA machine."""
    import torch
    from cumlx.compat.runner import child_environment, read_events
    script = tmp_path / 'train.py'
    script.write_text(textwrap.dedent("""
        import torch
        assert torch.cuda.is_available() is False, 'nothing may be redirected'
        assert torch.device('cuda').type == 'cuda'
        t = torch.ones(1, requires_grad=True)
        (t * 2).sum().backward()
    """))
    events = str(tmp_path / 'events.jsonl')
    env, _ = child_environment('cuda', events, {}, False, trace_losses=True, seed=3)
    env['PYTHONPATH'] = f"{env['PYTHONPATH']}{os.pathsep}{ROOT / 'src'}"
    done = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    seen = read_events(events)
    assert seen['loss_trace']['values'] == [2.0]
    assert seen['calls'] == []
    if not torch.cuda.is_available():
        refused = _cumlx(tmp_path, 'run', str(script), '--target', 'cuda', '--report', str(tmp_path / 'r.json'))
        assert refused['exit_code'] == 3
        assert refused['errors'][0]['code'] == 'E_BACKEND_UNAVAILABLE'


@pytest.mark.e2e
def test_rundiff_names_a_difference_of_torch_version(tmp_path, device):
    report = _train(tmp_path, TRAINING, 'a', device, 0)
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'seed0.json').write_text(json.dumps({**report, 'machine': {**report['machine'], 'torch': '2.0.0+cu126'}}))
    doc = _cumlx(tmp_path, 'rundiff', 'a', 'b')
    assert 'W_RUNDIFF_TORCH_DIFFERS' in {w['code'] for w in doc['warnings']}
    same = _cumlx(tmp_path, 'rundiff', 'a', 'a')
    assert 'W_RUNDIFF_TORCH_DIFFERS' not in {w['code'] for w in same['warnings']}
