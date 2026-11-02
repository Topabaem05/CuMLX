"""Compare an MLX port against a golden trace and locate the first divergence.

Two comparisons per recorded module call:

* chain    - run the whole port in its native dtype on the trace's token ids; compare every
             module output. Errors accumulate, so everything after a fault also fails.
* isolated - feed the module its *golden* input (attention and decoder layers also get a causal
             mask) in float32 arithmetic and compare its output. A failure here means the module
             itself computes something different.

In isolated mode a module that passes on its golden input is not at fault even if its chain output
exceeds the threshold. That is reported as ``chain_drift`` (accumulated rounding, or a fault elsewhere
that the failing module's own replay reports). A non-finite or wrongly shaped chain output still fails.

Record status: pass | fail | unverified (has parameters but could not be
compared) | skipped (parameter-free helper without a counterpart, e.g. an
activation module) | ignored (excluded by the user or the mapping file).
"""
from __future__ import annotations

from collections.abc import Callable
import contextlib
import re
from typing import Any

import numpy as np

from ..errors import CumlxError
from . import mlx_port
from .mapping import NameMapper
from .metrics import compare, noise_thresholds, resolve_threshold
from .trace import ModuleRecord, Trace

REPORT_SCHEMA = 'cumlx.diff-report.v1'
DRIFT_REASONS = ('rel_l2_above_threshold', 'golden_is_zero_but_output_is_not')
# With TF32 on, thresholds become at least this many times the measured matmul error.
# Measured on Qwen2.5-0.5B, M5 Pro: worst 1.95e-2 at a matmul error of 7.5e-4 (26x).
TF32_NOISE_FACTOR = 60.0


def _parent(name: str) -> str | None:
    if name == '':
        return None
    return name.rsplit('.', 1)[0] if '.' in name else ''


def _is_descendant(name: str, ancestor: str) -> bool:
    return name != ancestor and (ancestor == '' or name.startswith(ancestor + '.'))


def _localize(entries: list[dict], mode: str) -> dict | None:
    """Locate the culprit from the earliest failing record (completion order: children before parents)."""
    e = next((x for x in entries if x['status'] == 'fail'), None)
    if e is None:
        return None
    base = {'sample': e['sample'], 'first_failing_output': e['name'], 'module': e['name'], 'key': e['key'],
            'phase': e.get('phase', 'prefill')}
    label = e['name'] or '(root)'
    iso = (e.get('isolated') or {}).get('verdict')
    if iso == 'fail':
        return {**base, 'basis': 'isolated', 'confidence': 'high', 'metrics': e['isolated'],
                'explanation': f'`{label}` gives a different output from its own golden input. The defect is '
                               'inside this module (weights, bias, normalization, activation, RoPE, mask handling, '
                               'how submodule outputs are combined, or a kernel).'}
    if iso == 'pass':
        return {**base, 'basis': 'chain', 'confidence': 'medium', 'metrics': e['chain'],
                'explanation': f"`{label}` is correct in float32 on its golden input, but its native-dtype chain "
                               f"output is unusable ({e['chain'].get('reason')}). Suspect overflow or a dtype "
                               'limit in the native precision, or an earlier module.'}
    if iso is None and mode == 'isolated' and any(
            _is_descendant(x['name'], e['name']) and (x.get('isolated') or {}).get('verdict') == 'pass'
            for x in entries):
        return {**base, 'basis': 'chain+isolated', 'confidence': 'medium', 'metrics': e['chain'],
                'explanation': f'The submodules of `{label}` are correct on their golden inputs, but `{label}` itself '
                               'could not be replayed and its chain output diverges. The defect is in its own code '
                               'between the submodules (for example RoPE, attention math, masking or a residual).'}
    hint = '' if mode == 'isolated' else ' Run with --mode isolated to narrow it down.'
    return {**base, 'basis': 'chain', 'confidence': 'medium', 'metrics': e['chain'],
            'explanation': f'`{label}` is the first recorded output that diverges. The defect is in this module '
                           'or in unrecorded code between the last matching output and this one.' + hint}


def _native_output(golden: np.ndarray, actual: np.ndarray) -> dict | None:
    """How much of the port's native-dtype prediction agrees with the golden: top-1 per token position."""
    if golden.ndim != 3 or actual.shape != golden.shape:
        return None
    return {'positions': int(golden.shape[0] * golden.shape[1]),
            'top1_matches': int((golden.argmax(-1) == actual.argmax(-1)).sum())}


def run_diff(trace_dir: str, port: str, *, mode: str = 'isolated', loader: str = 'mlx_lm',
             mapping: str | None = None, profile: str = 'auto', rel_l2: float | None = None,
             include: str | None = None, exclude: str | None = None, dry_run: bool = False,
             device: str = 'auto', tf32: str = 'off', model=None,
             log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    mx, _ = mlx_port.mlx()
    trace = Trace(trace_dir)
    mapper = NameMapper.from_file(mapping)
    execution = mlx_port.configure_execution(device, tf32)
    handle = execution.pop('handle')
    log(f"executing on {execution['device']}, TF32 {tf32}, measured GEMM error {execution['gemm_noise_rel_l2']:.2e}")
    with mx.stream(handle):
        return _run(trace, port, mapper, execution, mode=mode, loader=loader, profile=profile, rel_l2=rel_l2,
                    include=include, exclude=exclude, dry_run=dry_run, model=model, log=log, mx=mx)


def _run(trace: Trace, port: str, mapper: NameMapper, execution: dict, *, mode, loader, profile, rel_l2,
         include, exclude, dry_run, model, log, mx) -> dict[str, Any]:
    if model is None:
        log(f'loading MLX port {port}')
        model = mlx_port.load_port(port, loader)
    mods = mlx_port.named_modules(model)
    mapper = mapper.with_port_conventions(mods)
    pinfo = mlx_port.port_profile(model)
    chosen = pinfo['profile'] if profile == 'auto' else profile
    golden_kind = trace.manifest.golden.get('kind')
    threshold = resolve_threshold(chosen, rel_l2, port_weights_golden=golden_kind == 'quantization-aware')
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    warnings: list[dict] = []
    ties = trace.manifest.source.get('tied_modules') or {}
    if pinfo['quantized_modules'] and golden_kind != 'quantization-aware':
        warnings.append({'code': 'W_QUANTIZED_PORT_REFERENCE_GOLDEN',
                         'message': 'the port is quantized but the golden is not quantization-aware; '
                                    'quantization loss will appear as divergence',
                         'hint': f'cumlx capture <original> --weights-from-mlx {port} -o golden-q/'})
    partial = trace.manifest.source.get('partial')
    if partial:
        warnings.append({'code': 'W_PARTIAL_TRACE',
                         'message': f'this trace records only part of the model ({partial}); coverage and the verdict '
                                    'describe the recorded modules, not the whole model',
                         'hint': 'capture without --layers, --include and --exclude to compare every module'})
    if execution['reduced_precision_matmul']:
        floor = TF32_NOISE_FACTOR * execution['gemm_noise_rel_l2']
        if rel_l2 is None:
            for key in ('rel_l2', 'isolated_rel_l2', 'composite_rel_l2_max', 'noise_floor'):
                threshold[key] = max(threshold[key], floor)
            threshold['raised_for_tf32'] = True
        warnings.append({'code': 'W_TF32_ACTIVE',
                         'message': f"matmul runs in TF32 (measured error {execution['gemm_noise_rel_l2']:.1e}); "
                                    f"thresholds are now at least {floor:.1e}, so small defects can pass. This "
                                    'measures the default MLX behavior, not full float32 parity.',
                         'hint': 'omit --tf32 on to compare in full float32'})
    if mode == 'chain' and pinfo['float_param_dtypes'] and chosen in ('float16', 'bfloat16'):
        warnings.append({'code': 'W_CHAIN_NATIVE_ROUNDING',
                         'message': f'chain mode compares native {chosen} arithmetic with a float32 golden; accumulated '
                                    'rounding can exceed the threshold on a correct port',
                         'hint': 'use --mode isolated (the default) to judge weights and logic'})

    def wanted(name: str) -> bool:
        if name == '':
            return True
        if mapper.is_ignored(name):
            return False
        return (inc is None or bool(inc.search(name))) and not (exc and exc.search(name))

    mapped_names = {mapper.map(r.name) for s in trace.manifest.samples for r in s.records}
    if dry_run:
        rows = []
        for r in trace.manifest.samples[0].records if trace.manifest.samples else []:
            target = mapper.map(r.name)
            found = target in mods or _tied_linear(r.name, ties, mapper, mods) is not None
            rows.append({'name': r.name, 'mlx_name': target, 'found': found,
                         'has_params': r.has_params, 'replayable': r.replayable})
        missing = [x for x in rows if not x['found'] and x['has_params'] and wanted(x['name'])]
        return {'schema': REPORT_SCHEMA, 'dry_run': True, 'port': {'path': port, 'loader': loader, **pinfo},
                'threshold': threshold, 'execution': execution,
                'mapping': {'source': mapper.source, 'rows': rows, 'unmapped_with_params': missing},
                'warnings': warnings}

    entries: list[dict] = []
    top1 = {'positions': 0, 'top1_matches': 0}
    for sample in trace.manifest.samples:
        ids = trace.token_ids(sample)
        log(f'{sample.id}: running port on {ids.shape[1]} tokens ({mode})')
        with mlx_port.probe(model, mapped_names & set(mods)) as rec:
            try:
                model(mx.array(ids.astype(np.int32)))
                if rec.outputs:  # MLX is lazy: an op that this device cannot run fails here
                    mx.eval(*rec.outputs.values())
            except Exception as e:  # noqa: BLE001
                raise CumlxError('E_MODEL_RUN', f"MLX port failed on {sample.id} on the {execution['device']}: "
                                                f'{type(e).__name__}: {e}',
                                 hint='check that the port matches the trace (same vocab and architecture); '
                                      'an op may be missing on this device, so also try the other --device') from None
        cached, cached_error = None, None
        if sample.decode_steps:
            cached, cached_error = _run_with_cache(model, ids, trace.tensor(sample, 'decode_token_ids'),
                                                   mapped_names & set(mods), mx)
            if cached_error:
                warnings.append({'code': 'W_DECODE_NOT_RUN',
                                 'message': f'the port could not run the decode phase: {cached_error}. '
                                            'Its decode records are unverified',
                                 'hint': 'capture with --decode-steps 0 if this port has no KV cache'})
        root = next((r for r in sample.records if r.name == '' and r.phase == 'prefill'), None)
        if root is not None and root.output and '#0' in rec.outputs:
            n = _native_output(trace.tensor(sample, root.output.tensor), mlx_port.to_numpy(rec.outputs['#0']))
            if n:
                top1 = {k: top1[k] + n[k] for k in top1}
        with mlx_port.promoted_to_float32(model) if mode == 'isolated' else contextlib.nullcontext():
            for r in sample.records:
                entries.append(_compare_record(trace, sample, r, mapper, mods, cached if r.phase == 'decode' else rec,
                                               mode, threshold, wanted, mx, ties, model))
    _judge_by_noise(entries, threshold, mode)
    if threshold.get('composite_rel_l2', 0) > threshold['noise_floor'] and not threshold.get('fixed_by_user') \
            and golden_kind != 'quantization-aware' and not pinfo['quantized_modules']:
        warnings.append({'code': 'W_WEIGHT_ROUNDING_NOISE',
                         'message': f"the weights of this port differ from the golden by rounding (median leaf rel_l2 "
                                    f"{threshold['median_leaf_error']:.1e}), so thresholds are leaf "
                                    f"{threshold['leaf_rel_l2']:.1e}, composite {threshold['composite_rel_l2']:.1e}; "
                                    'smaller defects can pass. Check the weights with `cumlx weights`',
                         'hint': f'for full float32 sensitivity: cumlx capture <original> --weights-from-mlx {port} -o golden-p/'})
    for e in entries:
        if e.get('chain') or e.get('isolated'):
            _settle(e)
    moved = _accept_scale_moved_to_parent(entries)
    if moved:
        warnings.append({'code': 'W_MODULE_BOUNDARY_DIFFERS',
                         'message': f'{moved} output(s) differ only by a constant factor that the parent module applies '
                                    'in the other framework. The parents match, so these are not counted as failures',
                         'hint': 'see records with boundary_scale'})
    counts = {k: sum(1 for e in entries if e['status'] == k) for k in
              ('pass', 'fail', 'unverified', 'skipped', 'ignored')}
    first = None
    for sid in [s.id for s in trace.manifest.samples]:
        first = _localize([e for e in entries if e['sample'] == sid], mode)
        if first:
            break
    param_rows = [e for e in entries if e['has_params'] and e['status'] != 'ignored']
    verified = sum(1 for e in param_rows if e['status'] in ('pass', 'fail'))
    verdict = 'diverged' if counts['fail'] else ('incomplete' if counts['unverified'] else 'ok')
    chain_values = [e['chain']['rel_l2'] for e in entries if e.get('chain') and e['chain'].get('rel_l2') is not None]
    isolated_values = [e['isolated']['rel_l2'] for e in entries
                       if e.get('isolated') and e['isolated'].get('rel_l2') is not None]
    return {
        'schema': REPORT_SCHEMA, 'verdict': verdict, 'mode': mode,
        'trace': {'path': str(trace.dir), 'source': trace.manifest.source, 'golden': trace.manifest.golden,
                  'samples': len(trace.manifest.samples)},
        'port': {'path': port, 'loader': loader, **pinfo},
        'threshold': threshold, 'mapping': {'source': mapper.source}, 'execution': execution,
        'summary': {**counts, 'records': len(entries),
                    'isolated_compared': sum(1 for e in entries if e.get('isolated')),
                    'chain_drift': sum(1 for e in entries if e.get('chain_drift')),
                    'boundary_differences': sum(1 for e in entries if e.get('boundary_scale') is not None),
                    'decode_records': sum(1 for e in entries if e.get('phase') == 'decode'),
                    'worst_isolated_rel_l2': max(isolated_values, default=None),
                    'worst_chain_rel_l2': max(chain_values, default=None),
                    'native_output': top1 if top1['positions'] else None,
                    'parametrized_coverage': (verified / len(param_rows)) if param_rows else 1.0},
        'first_divergence': first, 'warnings': warnings, 'records': entries,
    }


def _judge_by_noise(entries: list[dict], threshold: dict, mode: str) -> None:
    """Set isolated verdicts from this run's median leaf error (see ``metrics``)."""
    if mode != 'isolated':
        return
    parents: set[str] = set()
    for e in entries:
        if e['has_params']:
            anc = _parent(e['name'])
            while anc is not None:
                parents.add(anc)
                anc = _parent(anc)
    def judged(e: dict) -> bool:
        iso = e.get('isolated')
        return bool(iso) and iso.get('basis') == 'rel_l2' and iso.get('rel_l2') is not None
    errors = [e['isolated']['rel_l2'] for e in entries if e['has_params'] and e['name'] not in parents and judged(e)]
    median = float(np.median(errors)) if errors else 0.0
    leaf_limit, composite_limit = noise_thresholds(threshold, median)
    threshold.update(median_leaf_error=median, leaf_rel_l2=leaf_limit, composite_rel_l2=composite_limit)
    for e in entries:
        if judged(e):
            iso = e['isolated']
            ok = iso['rel_l2'] <= (composite_limit if e['name'] in parents else leaf_limit)
            iso['verdict'] = 'pass' if ok else 'fail'
            if ok:
                iso.pop('reason', None)
            else:
                iso['reason'] = 'rel_l2_above_threshold'


def _accept_scale_moved_to_parent(entries: list[dict]) -> int:
    """An output that is a constant multiple of the golden, under a parent that matches, is a different module boundary.

    Example: transformers scales Gemma embeddings inside the embedding, and mlx-lm scales them in the model body.
    A real scale defect also changes the parent, so it stays a failure.
    """
    by_key = {(e['sample'], e['phase'], e['name']): e for e in entries}
    moved = 0
    for e in entries:
        iso = e.get('isolated') or {}
        if e.get('status') != 'fail' or iso.get('reason') != 'rel_l2_above_threshold' or (iso.get('cosine') or 0) < 1 - 1e-6:
            continue
        anc = _parent(e['name'])
        while anc is not None and (e['sample'], e['phase'], anc) not in by_key:
            anc = _parent(anc)
        parent = by_key.get((e['sample'], e['phase'], anc)) if anc is not None else None
        if parent is not None and (parent.get('isolated') or {}).get('verdict') == 'pass':
            e.update(status='pass', boundary_scale=iso.get('scale'))
            e.pop('fail_cause', None)
            moved += 1
    return moved


def _settle(e: dict) -> None:
    chain_v = (e['chain'] or {}).get('verdict')
    iso_v = (e['isolated'] or {}).get('verdict')
    if iso_v == 'fail':
        e.update(status='fail', fail_cause='self')
    elif iso_v == 'pass' and chain_v == 'fail' and e['chain'].get('reason') in DRIFT_REASONS:
        e.update(status='pass', chain_drift=True)
    elif chain_v == 'fail':
        e.update(status='fail', fail_cause='native' if iso_v == 'pass' else 'unknown')
    elif 'pass' in (chain_v, iso_v):
        e['status'] = 'pass'
    else:
        e.update(status='unverified', note='module was not called by the port or could not be replayed')


def _tied_linear(name: str, ties: dict, mapper: NameMapper, mods: dict):
    """``as_linear`` of the port module that owns the weight a tied PyTorch module shares, if any."""
    owner = ties.get(name)
    return getattr(mods.get(mapper.map(owner)), 'as_linear', None) if owner is not None else None


def _run_with_cache(model, ids: np.ndarray, decode_ids: np.ndarray, names: set[str], mx):
    """Run prefill and then each decode token on the port's own KV cache. Returns (recorder, error)."""
    try:
        caches = mlx_port.make_caches(model)
        with mlx_port.probe(model, names) as rec:
            model(mx.array(ids.astype(np.int32)), cache=caches)
            for i in range(decode_ids.shape[1]):
                model(mx.array(decode_ids[:, i:i + 1].astype(np.int32)), cache=caches)
            if rec.outputs:
                mx.eval(*rec.outputs.values())
        return rec, None
    except Exception as exc:  # noqa: BLE001 - a port without a KV cache is reported, not fatal
        return None, f'{type(exc).__name__}: {exc}'


def _golden_array(arr: np.ndarray, mx):
    return mx.array(arr.astype(np.int32) if np.issubdtype(arr.dtype, np.integer) else arr.astype(np.float32))


def _replay_record(trace, sample, r: ModuleRecord, module, x, model, mx, rec, target: str):
    if r.replay_kind == 'experts':
        indices, scores = (_golden_array(trace.tensor(sample, t.tensor), mx) for t in r.inputs[1:3])
        return mlx_port.replay_experts(module, x, indices, scores)
    if r.replay_kind in ('hidden_cached', 'tokens_cached'):
        earlier = [p for p in sample.records if p.name == r.name and p.order < r.order]
        if not all(p.inputs for p in earlier):
            raise ValueError('an earlier step has no recorded input')
        steps = [_golden_array(trace.tensor(sample, p.inputs[0].tensor), mx) for p in earlier]
        keys = [f'{target}#{p.call}' for p in earlier] + [f'{target}#{r.call}']
        masks = [rec.masks[k] for k in keys] if rec is not None and all(k in rec.masks for k in keys) else None
        return mlx_port.replay_cached(model, module, r.name, steps, x, r.replay_kind, masks)
    key = f'{target}#{r.call}'
    if r.replay_kind == 'hidden' and rec is not None and key in rec.masks:
        return mlx_port.replay(module, x, r.replay_kind, rec.masks[key])
    return mlx_port.replay(module, x, r.replay_kind)


def _compare_record(trace: Trace, sample, r: ModuleRecord, mapper: NameMapper, mods: dict, rec,
                    mode: str, threshold: dict, wanted, mx, ties: dict, model) -> dict:
    target = mapper.map(r.name)
    e: dict[str, Any] = {'sample': sample.id, 'key': r.key, 'name': r.name, 'type': r.type, 'order': r.order,
                         'phase': r.phase, 'has_params': r.has_params, 'replayable': r.replayable,
                         'mlx_name': target, 'chain': None, 'isolated': None}
    if not wanted(r.name):
        e.update(status='ignored', note='excluded by --include/--exclude or mapping ignore')
        return e
    tied = _tied_linear(r.name, ties, mapper, mods) if target not in mods else None
    if target not in mods and tied is None:
        if r.has_params:
            e.update(status='unverified', note='no module with this name in the MLX port; add a mapping rule')
        else:
            e.update(status='skipped', note='parameter-free helper without an MLX counterpart')
        return e
    golden = trace.tensor(sample, r.output.tensor) if r.output else None
    out = rec.outputs.get(f'{target}#{r.call}') if rec is not None else None
    if r.replay_kind == 'experts':
        out = None  # the port returns each expert separately, so only the golden-routing replay is comparable
    if tied is not None:
        e['tied_to'] = ties[r.name]
        parent = next((x for x in sample.records if x.name == _parent(r.name) and x.call == r.call and x.output), None)
        if rec is not None and parent is not None and golden is not None \
                and np.array_equal(trace.tensor(sample, parent.output.tensor), golden):
            out = rec.outputs.get(f'{mapper.map(parent.name)}#{parent.call}')  # the parent returns this output as is
    if out is not None and golden is not None:
        e['chain'] = compare(mlx_port.to_numpy(out), golden, threshold)
    if mode == 'isolated' and r.replayable and golden is not None:
        x = _golden_array(trace.tensor(sample, r.inputs[0].tensor), mx)
        try:
            y = mlx_port.first_array(tied(x)) if tied is not None else _replay_record(trace, sample, r, mods[target], x, model, mx, rec, target)
            if y is None:
                raise TypeError('module returned no array')
            mx.eval(y)
            e['isolated'] = compare(mlx_port.to_numpy(y), golden, {**threshold, 'rel_l2': threshold['isolated_rel_l2']})
        except Exception as ex:  # noqa: BLE001
            e['isolated'] = {'verdict': 'error', 'reason': f'replay raised {type(ex).__name__}: {ex}'}
    if not e['chain'] and not e['isolated']:
        note = 'MoE experts are compared only with --mode isolated' if r.replay_kind == 'experts' \
            else 'module was not called by the port or could not be replayed'
        e.update(status='unverified', note=note)
    return e
