"""Command implementations. Each returns a ``Result``; heavy imports stay inside functions."""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
from typing import Any

import numpy as np

from ..errors import CumlxError, ExitCode
from .output import Result, cap

VERDICT_EXIT = {'ok': ExitCode.OK, 'diverged': ExitCode.DIVERGED, 'incomplete': ExitCode.INCOMPLETE}


def _logger(args) -> Callable[[str], None]:
    if getattr(args, 'verbose', False):
        return lambda m: print(f'[cumlx] {m}', file=sys.stderr, flush=True)
    return lambda _m: None


def _regex(value: str | None, flag: str) -> str | None:
    if value is None:
        return None
    try:
        re.compile(value)
    except re.error as exc:
        raise CumlxError('E_USAGE', f'{flag} is not a valid regex: {exc}') from None
    return value


def _q(path: Any) -> str:
    return shlex.quote(str(path))


# --------------------------------------------------------------------------- doctor
def run_doctor(args) -> Result:
    from ..verify.env import REQUIREMENTS, inspect_environment, install_hint
    env = inspect_environment(probe=args.probe)
    required = [c.strip() for c in args.require.split(',') if c.strip()]
    unknown = [c for c in required if c not in REQUIREMENTS]
    if unknown:
        raise CumlxError('E_USAGE', f'unknown command(s) in --require: {unknown}',
                         details={'known': sorted(REQUIREMENTS)})
    ready = sorted(c for c, ok in env['commands_ready'].items() if ok)
    not_ready = sorted(c for c, ok in env['commands_ready'].items() if not ok)
    r = Result('doctor', data=env, summary=f"ready: {', '.join(ready)}" +
               (f"; not ready: {', '.join(not_ready)}" if not_ready else ''))
    for c in not_ready:
        r.suggest(install_hint(env['missing_by_command'].get(c, [])), f'enable `cumlx {c}`')
    failing = [c for c in required if not env['commands_ready'][c]]
    if failing:
        r.exit_code = ExitCode.ENVIRONMENT
        for c in failing:
            mods = env['missing_by_command'].get(c, [])
            r.errors.append({'code': 'E_DEP_MISSING', 'message': f'`cumlx {c}` is not ready: missing {mods}',
                             'hint': install_hint(mods), 'details': {'command': c, 'missing': mods}})
    if not not_ready:
        r.suggest('cumlx selftest', 'prove that capture/diff/conform work in this environment')
    return r


# --------------------------------------------------------------------------- fixture
def run_fixture(args) -> Result:
    from ..verify.fixtures import make_tiny_llama, make_tiny_moe
    from ..verify.trace import prepare_output_dir
    out = prepare_output_dir(args.output, overwrite=args.overwrite, marker='.cumlx-fixture')
    info = {'tiny-llama': make_tiny_llama, 'tiny-moe': make_tiny_moe}[args.kind](out, seed=args.seed)
    (out / '.cumlx-fixture').write_text(json.dumps({'kind': args.kind, 'seed': args.seed}) + '\n')
    r = Result('fixture', summary=f'created {args.kind} at {out}', data=info)
    r.artifact('model', str(out))
    r.suggest(f'cumlx capture {_q(out)} --token-ids 1,5,9,23,7,99,3 -o {_q(str(out) + "-golden")}',
              'record a golden trace from the PyTorch model')
    return r


# --------------------------------------------------------------------------- capture
def _parse_token_ids(values: list[str]) -> list[np.ndarray]:
    out = []
    for v in values:
        try:
            ids = [int(x) for x in v.replace(' ', '').split(',') if x != '']
        except ValueError:
            raise CumlxError('E_USAGE', f'--token-ids must be comma-separated integers, got {v!r}') from None
        if not ids or min(ids) < 0:
            raise CumlxError('E_USAGE', f'--token-ids needs at least one non-negative id, got {v!r}')
        out.append(np.asarray(ids, dtype=np.int64))
    return out


def _parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    out: set[int] = set()
    try:
        for part in value.replace(' ', '').split(','):
            first, _, last = part.partition('-')
            out.update(range(int(first), int(last or first) + 1))
    except ValueError:
        raise CumlxError('E_USAGE', f'--layers needs indices such as 0,11,20-23, got {value!r}') from None
    if not out or min(out) < 0:
        raise CumlxError('E_USAGE', f'--layers needs indices such as 0,11,20-23, got {value!r}')
    return sorted(out)


def run_capture(args) -> Result:
    from ..verify.trace import TraceWriter, check_output_dir, prepare_output_dir
    log = _logger(args)
    if args.max_tokens <= 0:
        raise CumlxError('E_USAGE', '--max-tokens must be positive')
    if args.decode_steps < 0:
        raise CumlxError('E_USAGE', '--decode-steps cannot be negative')
    prompts = list(args.prompt)
    if args.prompts_file:
        pf = Path(args.prompts_file)
        if not pf.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'--prompts-file not found: {pf}')
        prompts += [line for line in pf.read_text(encoding='utf-8').splitlines() if line.strip()]
    token_samples = [ids[:args.max_tokens] for ids in _parse_token_ids(args.token_ids)]
    layers = _parse_layers(args.layers)
    if not prompts and not token_samples:
        raise CumlxError('E_USAGE', 'no inputs: pass --prompt, --prompts-file or --token-ids',
                         hint=f'cumlx capture {_q(args.model)} --token-ids 1,2,3 -o {_q(args.output)}')
    include, exclude = _regex(args.include, '--include'), _regex(args.exclude, '--exclude')
    check_output_dir(args.output, overwrite=args.overwrite)
    golden_kind = 'quantization-aware' if args.weights_from_mlx else 'reference'
    plan = {'model': args.model, 'output': args.output, 'samples': len(prompts) + len(token_samples),
            'prompt_samples': len(prompts), 'token_id_samples': len(token_samples), 'golden_kind': golden_kind,
            'device': args.device, 'store_dtype': args.store_dtype, 'max_tokens': args.max_tokens,
            'decode_steps': args.decode_steps, 'layers': layers,
            'include': include, 'exclude': exclude, 'weights_from_mlx': args.weights_from_mlx}
    if args.dry_run:
        r = Result('capture', summary=f"dry run: would record {plan['samples']} sample(s) from {args.model}",
                   data={'dry_run': True, 'plan': plan})
        argv = getattr(args, 'argv', None) or []
        r.suggest('cumlx ' + ' '.join(shlex.quote(a) for a in argv if a != '--dry-run'), 'run the capture for real')
        return r
    from ..verify.capture_torch import (
        capture_model,
        load_hf_model,
        load_mlx_weights_into,
        resolve_revision,
        tokenize,
    )
    log(f'loading PyTorch model {args.model} on {args.device}')
    model = load_hf_model(args.model, device=args.device, revision=args.revision,
                          trust_remote_code=args.trust_remote_code)
    weights_info = None
    if args.weights_from_mlx:
        from ..verify import mlx_port
        log(f'loading dequantized weights from MLX port {args.weights_from_mlx}')
        port = mlx_port.load_port(args.weights_from_mlx)
        weights_info = load_mlx_weights_into(model, mlx_port.dequantized_weights(port))
        weights_info['port'] = args.weights_from_mlx
        del port
    samples = tokenize(args.model, prompts, revision=args.revision, trust_remote_code=args.trust_remote_code,
                       max_tokens=args.max_tokens) if prompts else []
    samples += token_samples
    import torch
    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:  # pragma: no cover - load_hf_model already required it
        tf_version = None
    source = {'framework': 'torch', 'framework_version': torch.__version__, 'transformers_version': tf_version,
              'model': args.model, 'revision': resolve_revision(args.model, args.revision, model),
              'architecture': type(model).__name__, 'device': args.device,
              'inputs': [{'kind': 'prompt', 'text': p} for p in prompts] +
                        [{'kind': 'token_ids', 'count': int(t.size)} for t in token_samples]}
    golden = {'kind': golden_kind, 'dtype': 'float32', 'store_dtype': args.store_dtype,
              'weights_from_mlx': weights_info}
    out = prepare_output_dir(args.output, overwrite=args.overwrite)
    writer = TraceWriter(out, source=source, golden=golden, max_bytes=args.max_bytes)
    stats = capture_model(model, samples, writer, include=include, exclude=exclude,
                          store_dtype=args.store_dtype, device=args.device, decode_steps=args.decode_steps,
                          layers=layers, log=log)
    manifest = writer.finish()
    r = Result('capture', summary=f"recorded {stats['records']} module calls from {stats['samples']} sample(s) "
                                  f"({stats['replayable']} replayable) into {out}",
               data={'trace': str(out), 'golden_kind': golden_kind, **stats, 'source': source})
    if weights_info:
        r.data['weights_from_mlx'] = {k: (cap(v, args.max_items, r) if isinstance(v, list) else v)
                                      for k, v in weights_info.items()}
    r.artifact('trace', str(out))
    r.artifact('manifest', str(manifest))
    port_hint = args.weights_from_mlx or '<mlx-port>'
    r.suggest(f'cumlx diff {_q(out)} {_q(port_hint)} -o {_q(str(out) + "-report")}',
              'compare the MLX port against this golden trace')
    return r


# --------------------------------------------------------------------------- diff
def run_diff(args) -> Result:
    from ..verify.diff import run_diff as diff
    from ..verify.report import compact_records, to_markdown
    from ..verify.report_html import to_html
    from ..verify.trace import check_output_dir, prepare_output_dir
    include, exclude = _regex(args.include, '--include'), _regex(args.exclude, '--exclude')
    if args.output:
        check_output_dir(args.output, overwrite=args.overwrite, marker='report.json')
    rep = diff(args.trace, args.port, mode=args.mode, loader=args.loader, mapping=args.mapping,
               profile=args.profile, rel_l2=args.rel_l2, include=include, exclude=exclude,
               dry_run=args.dry_run, device=args.device, tf32=args.tf32, log=_logger(args))
    r = Result('diff')
    for w in rep.get('warnings', []):
        r.warnings.append(w)
        if w.get('hint'):
            r.suggest(w['hint'], w['message'])
    if args.dry_run:
        rows = rep['mapping']['rows']
        missing = rep['mapping']['unmapped_with_params']
        r.summary = (f"dry run: {sum(x['found'] for x in rows)}/{len(rows)} recorded modules found in the port; "
                     f"{len(missing)} parametrized module(s) unmapped")
        r.data = {'dry_run': True, 'port': rep['port'], 'threshold': rep['threshold'], 'execution': rep['execution'],
                  'unmapped_with_params': cap(missing, args.max_items, r)}
        if missing:
            r.suggest('cumlx diff ... --mapping rules.json', 'add rename rules for the unmapped modules')
        return r
    r.exit_code = VERDICT_EXIT[rep['verdict']]
    s = rep['summary']
    fd = rep['first_divergence']
    r.summary = (f"{rep['verdict']}: {s['pass']} pass, {s['fail']} fail, {s['unverified']} unverified, "
                 f"{s['skipped']} skipped" +
                 (f", {s['chain_drift']} chain drift" if s.get('chain_drift') else '') +
                 (f"; first divergence at `{fd['module'] or '(root)'}` ({fd['basis']}, {fd['confidence']})" if fd else ''))
    r.data = {'verdict': rep['verdict'], 'first_divergence': fd, 'summary': s, 'mode': rep['mode'],
              'threshold': rep['threshold'], 'port': rep['port'], 'execution': rep['execution'],
              'golden_kind': rep['trace']['golden'].get('kind'),
              'failures': cap(compact_records(rep, ('fail',)), args.max_items, r),
              'unverified': cap(compact_records(rep, ('unverified',)), args.max_items, r)}
    if args.output:
        out = prepare_output_dir(args.output, overwrite=args.overwrite, marker='report.json')
        (out / 'report.json').write_text(json.dumps(_finite(rep), indent=1, ensure_ascii=False) + '\n')
        (out / 'report.md').write_text(to_markdown(rep), encoding='utf-8')
        (out / 'report.html').write_text(to_html(_finite(rep)), encoding='utf-8')
        r.artifact('report_json', str(out / 'report.json'))
        r.artifact('report_markdown', str(out / 'report.md'))
        r.artifact('report_html', str(out / 'report.html'))
    if rep['verdict'] == 'diverged':
        if args.output:
            r.suggest(f"cumlx report {_q(Path(args.output) / 'report.json')} --max-items -1",
                      'list every failing record')
        if args.mode == 'chain':
            r.suggest('cumlx diff ' + f'{_q(args.trace)} {_q(args.port)} --mode isolated',
                      'replay modules on golden inputs to locate the defect precisely')
    elif rep['verdict'] == 'incomplete':
        r.suggest('cumlx diff ... --mapping rules.json',
                  'map the unverified modules (see data.unverified) or exclude them explicitly')
    elif not args.output:
        r.suggest('cumlx diff ... -o report/', 'write report.json and report.md for the record')
    return r


def _finite(obj):
    from .output import _sanitize
    return _sanitize(obj)


# --------------------------------------------------------------------------- conform
def _run_checkpoint(args) -> Result:
    from ..verify.checkpoint import run_checkpoint, write_report
    if not args.against:
        raise CumlxError('E_USAGE', '--checkpoint needs --against REPACK',
                         hint=f'cumlx conform --checkpoint {_q(args.checkpoint)} --against <mlx-repack>')
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise CumlxError('E_OUTPUT_EXISTS', f'{args.output} exists', hint='pass --overwrite or choose a new path')
    rep = run_checkpoint(args.checkpoint, args.against, revision=args.revision)
    s = rep['summary']
    first = rep['first_failure']
    r = Result('conform', exit_code=VERDICT_EXIT[rep['verdict']])
    r.summary = (f"{rep['verdict']}: {s['pass']} of {s['quantized']} quantized tensors match the original "
                 f"({s['bit_identical']} bit-identical), {s['plain_identical']} other tensors identical" +
                 (f"; first failure `{first['name']}` ({first.get('reason')})" if first else ''))
    r.data = {'verdict': rep['verdict'], 'summary': s, 'first_failure': first, 'plain': rep['plain'],
              'failures': cap([x for x in rep['parameters'] if x['status'] == 'fail'], args.max_items, r)}
    if args.output:
        write_report(_finite(rep), args.output)
        r.artifact('checkpoint_report', args.output)
    return r


def run_conform(args) -> Result:
    if args.checkpoint:
        return _run_checkpoint(args)
    from ..verify.conform import run_conform as conform
    fmts = tuple(f.strip().lower() for f in args.formats.split(',') if f.strip())
    bad = [f for f in fmts if f not in ('nvfp4', 'mxfp4')]
    if bad or not fmts:
        raise CumlxError('E_USAGE', f'--formats accepts nvfp4,mxfp4; got {args.formats!r}')
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise CumlxError('E_OUTPUT_EXISTS', f'{args.output} exists', hint='pass --overwrite or choose a new path')
    rep = conform(args.target, fmts)
    r = Result('conform', exit_code=VERDICT_EXIT[rep['verdict']])
    failed = [c for c in rep['checks'] if c['verdict'] == 'fail']
    r.summary = (f"{rep['target']} {rep['target_version']}: {len(rep['checks'])} checks, {len(failed)} failed"
                 + (f" ({', '.join(c['id'] for c in failed)})" if failed else ''))
    brief = [{k: c.get(k) for k in ('id', 'kind', 'verdict', 'mismatches', 'finding', 'note') if c.get(k) is not None}
             for c in rep['checks']]
    r.data = {'verdict': rep['verdict'], 'target': rep['target'], 'target_version': rep['target_version'],
              'checks': cap(brief, args.max_items, r)}
    if args.output:
        Path(args.output).write_text(json.dumps(_finite(rep), indent=1) + '\n')
        r.artifact('conform_report', args.output)
    else:
        r.suggest('cumlx conform -o conform.json', 'save every case and mismatch example')
    return r


# --------------------------------------------------------------------------- weights
def run_weights(args) -> Result:
    from ..verify.weights import run_weights as compare_weights
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise CumlxError('E_OUTPUT_EXISTS', f'{args.output} exists', hint='pass --overwrite or choose a new path')
    rep = compare_weights(args.original, args.port, loader=args.loader, mapping=args.mapping, revision=args.revision)
    r = Result('weights', exit_code=VERDICT_EXIT[rep['verdict']])
    s = rep['summary']
    first = rep['first_failure']
    r.summary = (f"{rep['verdict']}: {s['pass']} pass, {s['fail']} fail, {s['unverified']} unverified of "
                 f"{s['parameters']} parameters" +
                 (f"; first failure `{first['name']}`" + (f" ({first['diagnosis']})" if first.get('diagnosis') else '')
                  if first else ''))
    r.data = {'verdict': rep['verdict'], 'summary': s, 'first_failure': first,
              'failures': cap([x for x in rep['parameters'] if x['status'] == 'fail'], args.max_items, r),
              'unverified': cap([x for x in rep['parameters'] if x['status'] == 'unverified'], args.max_items, r),
              'extra_in_port': cap(rep['extra_in_port'], args.max_items, r)}
    if args.output:
        Path(args.output).write_text(json.dumps(_finite(rep), indent=1) + '\n')
        r.artifact('weights_report', args.output)
    if rep['verdict'] == 'ok':
        r.suggest(f'cumlx capture {_q(args.original)} --prompt Hello -o golden/ && cumlx diff golden/ {_q(args.port)}',
                  'the weights are right; now check the code of the port')
    return r


# --------------------------------------------------------------------------- tracediff
def run_tracediff(args) -> Result:
    from ..verify.trace_compare import compare_traces
    if not (args.rel_l2 > 0):
        raise CumlxError('E_USAGE', '--rel-l2 must be positive')
    rep = compare_traces(args.a, args.b, rel_l2=args.rel_l2)
    s = rep['summary']
    r = Result('tracediff', exit_code=VERDICT_EXIT[rep['verdict']])
    r.summary = (f"{rep['verdict']}: worst rel_l2 {s['worst_rel_l2']}, median {s['median_rel_l2']}, "
                 f"{s['above_threshold']} of {s['compared']} outputs above {rep['threshold_rel_l2']:g}")
    r.data = {'verdict': rep['verdict'], 'summary': s, 'threshold_rel_l2': rep['threshold_rel_l2'],
              'a': rep['a']['source'], 'b': rep['b']['source'], 'worst': cap(rep['worst'], args.max_items, r)}
    return r


# --------------------------------------------------------------------------- report
def run_report(args) -> Result:
    from ..verify.report import compact_records
    p = Path(args.path)
    if p.is_dir():
        p = p / 'report.json'
    if not p.is_file():
        raise CumlxError('E_INPUT_NOT_FOUND', f'report not found: {p}', hint='pass the -o directory of `cumlx diff`')
    try:
        rep = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise CumlxError('E_INPUT_INVALID', f'{p} is not JSON: {exc}') from None
    if rep.get('schema') != 'cumlx.diff-report.v1' or 'records' not in rep:
        raise CumlxError('E_INPUT_INVALID', f'{p} is not a cumlx diff report')
    r = Result('report', exit_code=VERDICT_EXIT.get(rep['verdict'], ExitCode.INPUT))
    s = rep['summary']
    r.summary = f"{rep['verdict']}: {s['pass']} pass, {s['fail']} fail, {s['unverified']} unverified"
    r.data = {'verdict': rep['verdict'], 'first_divergence': rep.get('first_divergence'), 'summary': s,
              'failures': cap(compact_records(rep, ('fail',)), args.max_items, r),
              'unverified': cap(compact_records(rep, ('unverified',)), args.max_items, r)}
    return r


# --------------------------------------------------------------------------- selftest
def run_selftest(args) -> Result:
    from ..verify import selftest
    if args.keep and Path(args.keep).exists():
        raise CumlxError('E_OUTPUT_EXISTS', f'--keep directory exists: {args.keep}', hint='choose a new directory')
    log = _logger(args)
    if args.keep:
        Path(args.keep).mkdir(parents=True)
        steps = selftest.run(Path(args.keep), core_only=args.core_only, device=args.device, log=log)
    else:
        with tempfile.TemporaryDirectory(prefix='cumlx-selftest-') as tmp:
            steps = selftest.run(Path(tmp), core_only=args.core_only, device=args.device, log=log)
    failed = [s for s in steps if s['status'] == 'fail']
    skipped = [s for s in steps if s['status'] == 'skipped']
    r = Result('selftest')
    if failed:
        r.exit_code = ExitCode.DIVERGED
    elif skipped and not args.core_only:
        r.exit_code = ExitCode.INCOMPLETE
    passed = sum(s['status'] == 'pass' for s in steps)
    r.summary = f'{passed} passed, {len(failed)} failed, {len(skipped)} skipped' + (
        ' (skipped checks are not a pass)' if skipped and not args.core_only else '')
    r.data = {'steps': cap(steps, args.max_items, r)}
    if skipped and not args.core_only:
        r.suggest('cumlx doctor', 'install the dependencies needed by the skipped checks')
    if args.keep:
        r.artifact('selftest_dir', args.keep)
    return r
