"""Compare `cumlx run` loss traces from a golden device and a target, and give a proof grade.

Two questions have two methods:

* Grade A. Given the same seed, do the first steps give the same loss? Weights and batches come from the CPU
  generator in most scripts, so they can match across machines. Step 1 tests the forward path. Step 2 also
  tests the gradient and the optimizer. Later steps are not tested: rounding moves them apart.
* Grade B. Over several seeds, does training follow the same curve? This needs no shared random numbers.

The limits below were set on 2026-11-03 before any open-source script was measured, and they must not change
after a result is seen (docs/ROADMAP-PLATFORM.md, section 2). validation/run-compare-limits.json has the fixture
measurements: a right port on CPU and MPS differs by 1.7e-7 at step 2, and a learning rate wrong by 20% by 3.1e-3.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from ..errors import CumlxError

SCHEMA = 'cumlx.run-report.v1'
# Grade A. Largest relative difference of the loss at step 1 and at step 2. The float32 limit of `cumlx diff`.
STEP_REL_LIMITS = (1e-4, 1e-4)
# Grade B.
MIN_SEEDS = 3
WINDOWS = 4  # The curve is compared on four equal parts of training, so a wrong start cannot hide behind a right end.
STANDARD_ERRORS = 3.0  # The allowed difference: this many standard errors of the difference in means ...
REL_FLOOR = 0.01  # ... and never less than this part of the golden mean.
INCONCLUSIVE_REL = 0.10  # A window that allows more than this part of the golden mean cannot pass: it is too noisy to prove much.


def load_runs(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise CumlxError('E_INPUT_NOT_FOUND', f'{path} does not exist')
    files = sorted(p.glob('*.json')) if p.is_dir() else [p]
    runs = []
    for file in files:
        try:
            doc = json.loads(file.read_text())
        except json.JSONDecodeError as exc:
            raise CumlxError('E_INPUT_INVALID', f'{file} is not JSON: {exc}') from None
        if isinstance(doc, dict) and doc.get('schema') == SCHEMA:
            runs.append({**doc, 'file': str(file)})
    if not runs:
        raise CumlxError('E_INPUT_INVALID', f'{path} holds no `cumlx run` report')
    return runs


def _check(runs: list[dict[str, Any]], side: str) -> None:
    for run in runs:
        if run['script_exit_code'] != 0 or run['timed_out']:
            raise CumlxError('E_RUN_NOT_COMPARABLE', f"{run['file']} did not run to its end (exit {run['script_exit_code']})", hint='fix or rerun that script first')
        if 'loss_trace' not in run:
            raise CumlxError('E_RUN_NOT_COMPARABLE', f"{run['file']} has no loss trace", hint='run it with --trace-losses')
    seeds = [run['seed'] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise CumlxError('E_RUN_NOT_COMPARABLE', f'the {side} runs repeat a seed: {seeds}', hint='keep one report per seed')


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), 1e-12)


def _numeric(golden: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {run['seed']: run['loss_trace']['values'] for run in candidate}
    pairs = []
    for run in golden:
        seed, a = run['seed'], run['loss_trace']['values']
        b = by_seed.get(seed)
        if seed is None or b is None:
            continue
        steps = []
        for x, y in zip(a[:len(STEP_REL_LIMITS)], b[:len(STEP_REL_LIMITS)]):
            steps.append(None if x is None or y is None else _rel(x, y))
        ok = len(steps) == len(STEP_REL_LIMITS) and all(s is not None and s <= limit for s, limit in zip(steps, STEP_REL_LIMITS))
        pairs.append({'seed': seed, 'step_rel': steps, 'pass': ok})
    return {'limits': list(STEP_REL_LIMITS), 'pairs': pairs, 'pass': bool(pairs) and all(p['pass'] for p in pairs)}


def _window_means(values: list[float]) -> list[float]:
    size = len(values)
    return [fmean(values[i * size // WINDOWS:(i + 1) * size // WINDOWS]) for i in range(WINDOWS)]


def _statistical(golden: list[list[float]], candidate: list[list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {'limits': {'min_seeds': MIN_SEEDS, 'windows': WINDOWS, 'standard_errors': STANDARD_ERRORS, 'rel_floor': REL_FLOOR, 'inconclusive_rel': INCONCLUSIVE_REL},
                           'seeds': [len(golden), len(candidate)], 'compares_mixed_streams_as_one': True}
    for side, runs in (('golden', golden), ('candidate', candidate)):
        if len({tuple(run) for run in runs}) < len(runs):
            out.update(state='unavailable', why=f'two {side} runs gave the same trace, so the seed did not reach the script: this is one sample')
            return out
    if min(len(golden), len(candidate)) < MIN_SEEDS:
        out.update(state='unavailable', why=f'{MIN_SEEDS} seeds per side are needed')
        return out
    g = [_window_means(run) for run in golden]
    c = [_window_means(run) for run in candidate]
    windows, state = [], 'pass'
    for i in range(WINDOWS):
        gs, cs = [run[i] for run in g], [run[i] for run in c]
        mean_g, mean_c = fmean(gs), fmean(cs)
        error = math.sqrt(stdev(gs) ** 2 / len(gs) + stdev(cs) ** 2 / len(cs))
        allowed = max(STANDARD_ERRORS * error, REL_FLOOR * abs(mean_g))
        difference = abs(mean_c - mean_g)
        result = 'fail' if difference > allowed else 'inconclusive' if allowed > INCONCLUSIVE_REL * abs(mean_g) else 'pass'
        windows.append({'window': i + 1, 'golden_mean': mean_g, 'candidate_mean': mean_c, 'difference': difference, 'allowed': allowed, 'result': result})
        if result == 'fail':
            state = 'fail'
        elif result == 'inconclusive' and state == 'pass':
            state = 'inconclusive'
    out.update(state=state, windows=windows)
    return out


def compare_runs(golden: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    _check(golden, 'golden')
    _check(candidate, 'candidate')
    identity = {(run['script_sha256'], run['module'], tuple(run['script_args'])) for run in golden + candidate}
    if len(identity) != 1:
        raise CumlxError('E_RUN_NOT_COMPARABLE', 'these reports are not runs of one script with one set of arguments', hint='compare runs of the same file, with the same arguments')
    report: dict[str, Any] = {
        'schema': 'cumlx.rundiff.v1',
        'golden': {'target': golden[0]['target'], 'machine': golden[0]['machine'], 'seeds': [run['seed'] for run in golden]},
        'candidate': {'target': candidate[0]['target'], 'machine': candidate[0]['machine'], 'seeds': [run['seed'] for run in candidate]},
        'numeric': {}, 'statistical': {}, 'grade': None,
    }
    def finish(verdict: str, reason: str, grade: str | None = None) -> dict[str, Any]:
        report.update(verdict=verdict, reason=reason, grade=grade)
        return report

    traces = [run['loss_trace'] for run in golden + candidate]
    if any(t['processes_with_losses'] > 1 for t in traces):
        return finish('incomplete', 'more than one process called backward(), and traces of separate processes are not compared')
    lengths = {len(t['values']) for t in traces}
    if lengths == {0}:
        return finish('incomplete', 'no scalar reached Tensor.backward(), so there is nothing to compare')
    if len(lengths) != 1:
        return finish('incomplete', f'the runs have different numbers of backward() calls: {sorted(lengths)}')
    if min(lengths) < WINDOWS:
        return finish('incomplete', f'{min(lengths)} backward() calls are too few to compare')
    if any(v is None for run in golden for v in run['loss_trace']['values']):
        return finish('incomplete', 'the golden holds a non-finite loss, so it cannot be a reference')
    if any(v is None for run in candidate for v in run['loss_trace']['values']):
        report['numeric'] = _numeric(golden, candidate)
        return finish('diverged', 'the candidate holds a non-finite loss and the golden does not')

    report['numeric'] = numeric = _numeric(golden, candidate)
    report['statistical'] = stats = _statistical([r['loss_trace']['values'] for r in golden], [r['loss_trace']['values'] for r in candidate])
    if stats['state'] == 'fail':
        failed = next(w for w in stats['windows'] if w['result'] == 'fail')
        return finish('diverged', f"window {failed['window']} of {WINDOWS}: golden mean {failed['golden_mean']:.5g}, candidate mean {failed['candidate_mean']:.5g}, allowed difference {failed['allowed']:.3g}")
    if numeric['pass']:
        return finish('ok', 'the first two steps gave the same loss at each seed', 'A')
    if stats['state'] == 'pass':
        return finish('ok', f"the training curves agree in all {WINDOWS} windows over {min(stats['seeds'])} seeds", 'B')
    return finish('incomplete', stats.get('why') or 'the training losses are too noisy to prove agreement at this number of seeds: add seeds')
