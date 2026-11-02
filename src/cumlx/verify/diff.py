"""Compare an MLX port against a golden trace and locate the first divergence.

Two comparisons per recorded module call:

* chain    - run the whole port on the trace's token ids; compare every module output.
             Errors accumulate, so everything after a fault also fails.
* isolated - feed the module its *golden* input and compare its output. A failure
             here means the module itself computes something different.

Record status: pass | fail | unverified (has parameters but could not be
compared) | skipped (parameter-free helper without a counterpart, e.g. an
activation module) | ignored (excluded by the user or the mapping file).
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from ..errors import CumlxError
from .mapping import NameMapper
from .metrics import compare, resolve_threshold
from . import mlx_port
from .trace import ModuleRecord, Trace

REPORT_SCHEMA = 'cumlx.diff-report.v1'


def _parent(name: str) -> str | None:
    if name == '':
        return None
    return name.rsplit('.', 1)[0] if '.' in name else ''


def _localize(entries: list[dict], by_name: dict[str, list[dict]], mode: str) -> dict | None:
    """Locate the culprit from the earliest failing record (completion order: children before parents)."""
    def fails(e, basis):
        return bool(e.get(basis)) and e[basis].get('verdict') == 'fail'

    e = next((x for x in entries if x['status'] == 'fail'), None)
    if e is None:
        return None
    base = {'sample': e['sample'], 'first_failing_output': e['name']}
    if fails(e, 'isolated'):
        return {**base, 'module': e['name'], 'key': e['key'], 'basis': 'isolated', 'confidence': 'high',
                'metrics': e['isolated'],
                'explanation': f"`{e['name'] or '(root)'}` gives a different output from its own golden input. "
                               'The defect is inside this module (weights, bias, normalization, activation or kernel).'}
    if e.get('isolated') and e['isolated'].get('verdict') == 'pass':
        anc = _parent(e['name'])
        while anc is not None and anc not in by_name:
            anc = _parent(anc)
        target = anc if anc is not None else e['name']
        same = [x for x in by_name.get(target, []) if x['sample'] == e['sample']]
        if same and fails(same[0], 'isolated'):
            return {**base, 'module': target, 'key': same[0]['key'], 'basis': 'isolated', 'confidence': 'high',
                    'metrics': same[0]['isolated'],
                    'explanation': f"`{e['name']}` is correct on its golden input, and its parent `{target or '(root)'}` "
                                   'fails on its own golden input. The defect is in the parent\'s own code between '
                                   'its submodules (for example the activation or how submodule outputs are combined).'}
        return {**base, 'module': target, 'key': f'{target}#0', 'basis': 'chain+isolated', 'confidence': 'medium',
                'metrics': e['chain'],
                'explanation': f"`{e['name']}` is correct on its golden input but receives a diverged input. "
                               f"The divergence starts inside `{target or '(root)'}` in code that is not a recorded "
                               'submodule (for example RoPE, attention math, masking, residual or reshape).'}
    hint = '' if mode == 'isolated' else ' Run with --mode isolated to narrow it down.'
    return {**base, 'module': e['name'], 'key': e['key'], 'basis': 'chain', 'confidence': 'medium',
            'metrics': e['chain'],
            'explanation': f"`{e['name'] or '(root)'}` is the first recorded output that diverges. The defect is in "
                           'this module or in unrecorded code between the last matching output and this one.' + hint}


def run_diff(trace_dir: str, port: str, *, mode: str = 'isolated', loader: str = 'mlx_lm',
             mapping: str | None = None, profile: str = 'auto', rel_l2: float | None = None,
             include: str | None = None, exclude: str | None = None, dry_run: bool = False,
             model=None, log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    mx, _ = mlx_port.mlx()
    trace = Trace(trace_dir)
    mapper = NameMapper.from_file(mapping)
    if model is None:
        log(f'loading MLX port {port}')
        model = mlx_port.load_port(port, loader)
    mods = mlx_port.named_modules(model)
    pinfo = mlx_port.port_profile(model)
    chosen = pinfo['profile'] if profile == 'auto' else profile
    threshold = resolve_threshold(chosen, rel_l2)
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    warnings: list[dict] = []
    golden_kind = trace.manifest.golden.get('kind')
    if pinfo['quantized_modules'] and golden_kind != 'quantization-aware':
        warnings.append({'code': 'W_QUANTIZED_PORT_REFERENCE_GOLDEN',
                         'message': 'the port is quantized but the golden is not quantization-aware; '
                                    'quantization loss will appear as divergence',
                         'hint': f'cumlx capture <original> --weights-from-mlx {port} -o golden-q/'})

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
            rows.append({'name': r.name, 'mlx_name': target, 'found': target in mods,
                         'has_params': r.has_params, 'replayable': r.replayable})
        missing = [x for x in rows if not x['found'] and x['has_params'] and wanted(x['name'])]
        return {'schema': REPORT_SCHEMA, 'dry_run': True, 'port': {'path': port, 'loader': loader, **pinfo},
                'threshold': threshold, 'mapping': {'source': mapper.source, 'rows': rows,
                                                    'unmapped_with_params': missing},
                'warnings': warnings}

    entries: list[dict] = []
    for sample in trace.manifest.samples:
        ids = trace.token_ids(sample)
        log(f'{sample.id}: running port on {ids.shape[1]} tokens ({mode})')
        with mlx_port.probe(model, mapped_names & set(mods)) as rec:
            try:
                model(mx.array(ids.astype(np.int32)))
            except Exception as e:  # noqa: BLE001
                raise CumlxError('E_MODEL_RUN', f'MLX port failed on {sample.id}: {type(e).__name__}: {e}',
                                 hint='check that the port matches the trace (same vocab and architecture)') from None
            if rec.outputs:
                mx.eval(*rec.outputs.values())
        for r in sample.records:
            entries.append(_compare_record(trace, sample, r, mapper, mods, rec, mode, threshold, wanted, mx))
    by_name: dict[str, list[dict]] = {}
    for e in entries:
        by_name.setdefault(e['name'], []).append(e)
    counts = {k: sum(1 for e in entries if e['status'] == k) for k in
              ('pass', 'fail', 'unverified', 'skipped', 'ignored')}
    first = None
    for sid in [s.id for s in trace.manifest.samples]:
        first = _localize([e for e in entries if e['sample'] == sid], by_name, mode)
        if first:
            break
    param_rows = [e for e in entries if e['has_params'] and e['status'] != 'ignored']
    verified = sum(1 for e in param_rows if e['status'] in ('pass', 'fail'))
    verdict = 'diverged' if counts['fail'] else ('incomplete' if counts['unverified'] else 'ok')
    return {
        'schema': REPORT_SCHEMA, 'verdict': verdict, 'mode': mode,
        'trace': {'path': str(trace.dir), 'source': trace.manifest.source, 'golden': trace.manifest.golden,
                  'samples': len(trace.manifest.samples)},
        'port': {'path': port, 'loader': loader, **pinfo},
        'threshold': threshold, 'mapping': {'source': mapper.source},
        'summary': {**counts, 'records': len(entries),
                    'isolated_compared': sum(1 for e in entries if e.get('isolated')),
                    'parametrized_coverage': (verified / len(param_rows)) if param_rows else 1.0},
        'first_divergence': first, 'warnings': warnings, 'records': entries,
    }


def _compare_record(trace: Trace, sample, r: ModuleRecord, mapper: NameMapper, mods: dict, rec,
                    mode: str, threshold: dict, wanted, mx) -> dict:
    target = mapper.map(r.name)
    e: dict[str, Any] = {'sample': sample.id, 'key': r.key, 'name': r.name, 'type': r.type, 'order': r.order,
                         'has_params': r.has_params, 'replayable': r.replayable, 'mlx_name': target,
                         'chain': None, 'isolated': None}
    if not wanted(r.name):
        e.update(status='ignored', note='excluded by --include/--exclude or mapping ignore')
        return e
    if target not in mods:
        if r.has_params:
            e.update(status='unverified', note='no module with this name in the MLX port; add a mapping rule')
        else:
            e.update(status='skipped', note='parameter-free helper without an MLX counterpart')
        return e
    golden = trace.tensor(sample, r.output.tensor) if r.output else None
    out = rec.outputs.get(f'{target}#{r.call}')
    if out is not None and golden is not None:
        e['chain'] = compare(mlx_port.to_numpy(out), golden, threshold)
    if mode == 'isolated' and r.replayable and golden is not None:
        inp = trace.tensor(sample, r.inputs[0].tensor)
        x = mx.array(inp.astype(np.int32) if np.issubdtype(inp.dtype, np.integer) else inp.astype(np.float32))
        try:
            y = mlx_port.first_array(mods[target](x))
            if y is None:
                raise TypeError('module returned no array')
            mx.eval(y)
            e['isolated'] = compare(mlx_port.to_numpy(y), golden, threshold)
        except Exception as ex:  # noqa: BLE001
            e['isolated'] = {'verdict': 'error', 'reason': f'replay raised {type(ex).__name__}: {ex}'}
    verdicts = [c['verdict'] for c in (e['chain'], e['isolated']) if c]
    if 'fail' in verdicts:
        e['status'] = 'fail'
        iso = (e['isolated'] or {}).get('verdict')
        e['fail_cause'] = 'self' if iso == 'fail' else ('upstream' if iso == 'pass' else 'unknown')
    elif 'pass' in verdicts:
        e['status'] = 'pass'
    else:
        e.update(status='unverified', note='module was not called by the port or could not be replayed')
    return e
