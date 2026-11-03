"""Command implementations. Each returns a ``Result``; heavy imports stay inside functions."""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from ..errors import ERROR_CODES, CumlxError, ExitCode, missing_dependency
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


# --------------------------------------------------------------------------- scan
def run_scan(args) -> Result:
    from ..verify.portability import scan
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise CumlxError('E_OUTPUT_EXISTS', f'{args.output} exists', hint='pass --overwrite or choose a new path')
    rep = scan(args.target, revision=args.revision, include_tests=args.include_tests)
    r = Result('scan', summary=f"tier {rep['tier']}: {rep['recommendation']}")
    r.data = {'tier': rep['tier'], 'recommendation': rep['recommendation'], 'model': rep['model'], 'tiers': rep['tiers'],
              'files_scanned': rep['files_scanned'], 'cuda_source_files': rep['cuda_source_files'], 'limits': rep['limits'],
              'dependencies': rep['dependencies'], 'uncovered': rep['uncovered'],
              'findings': cap(rep['findings'], args.max_items, r)}
    if args.output:
        Path(args.output).write_text(json.dumps(rep, indent=1) + '\n')
        r.artifact('scan_report', args.output)
    if rep['tier'] == 'model':
        r.suggest(f"cumlx port {_q(args.target)} -o port/", 'convert this model with mlx-lm and verify the result')
    elif rep['findings']:
        r.suggest(f'cumlx recipe {_q(args.target)}', 'show what to use on Apple Silicon for each finding, and how far each was checked')
    return r


# --------------------------------------------------------------------------- recipe
def run_recipe(args) -> Result:
    from ..verify.recipes import lookup
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise CumlxError('E_OUTPUT_EXISTS', f'{args.output} exists', hint='pass --overwrite or choose a new path')
    rep = lookup(args.target, verified_only=args.verified_only, revision=args.revision)
    counts = rep['counts']
    r = Result('recipe', summary=(f"{counts['recipes']} recipe(s): {counts['with_run_evidence']} with an option run on Apple Silicon, "
                                  f"{counts['docs_only']} from documents only, {counts['no_substitute']} with no substitute."))
    r.data = {'query': rep['query'], 'kind': rep['kind'], 'counts': counts, 'tier': rep['tier'], 'recommendation': rep['recommendation'],
              'uncovered': rep['uncovered'], 'limits': rep['limits'], 'recipes': cap(rep['recipes'], args.max_items, r)}
    if counts['docs_only']:
        names = [x['id'] for x in rep['recipes'] if x['best_evidence'] == 'docs']
        r.warn('W_RECIPE_NOT_RUN', f'{len(names)} recipe(s) rest on documents only. CUMLX did not run them.', recipes=names)
    if counts['no_substitute']:
        names = [x['id'] for x in rep['recipes'] if x['outcome'] == 'none']
        r.warn('W_NO_SUBSTITUTE', f'{", ".join(names)} has no substitute on Apple Silicon. The options are workarounds.', recipes=names)
    if rep['uncovered']:
        r.warn('W_NO_RECIPE', f"The catalog has no recipe for: {', '.join(rep['uncovered'])}.", findings=rep['uncovered'])
    if args.output:
        Path(args.output).write_text(json.dumps(rep, indent=1) + '\n')
        r.artifact('recipe_report', args.output)
    return r


# --------------------------------------------------------------------------- run
def run_run(args) -> Result:
    from ..compat import runner
    if bool(args.script) == bool(args.module):
        raise CumlxError('E_USAGE', 'give a script or --module, not both and not neither', hint='cumlx run train.py -- --lr 1e-3')
    if args.script and not Path(args.script).is_file():
        raise CumlxError('E_INPUT_NOT_FOUND', f'{args.script} does not exist')
    report_path = Path(args.report)
    if report_path.exists() and not args.overwrite:
        raise CumlxError('E_OUTPUT_EXISTS', f'{args.report} exists', hint='pass --overwrite or choose a new path with --report')
    substitutes = {}
    for item in args.substitute:
        name, sep, module = item.partition('=')
        if not sep or not name or not module:
            raise CumlxError('E_USAGE', f'--substitute needs NAME=MODULE, got {item!r}')
        substitutes[name] = module
    try:
        import torch
    except ImportError:
        raise missing_dependency('torch', 'capture', purpose='cumlx run') from None
    if args.target == 'mps' and not torch.backends.mps.is_available():
        raise CumlxError('E_BACKEND_UNAVAILABLE', 'MPS is not available on this machine', hint='use --target cpu')
    device_name = None
    if args.target == 'cuda':
        if not torch.cuda.is_available():
            raise CumlxError('E_BACKEND_UNAVAILABLE', 'CUDA is not available on this machine', hint='--target cuda records a golden on a real CUDA device. Use --target mps here')
        device_name = torch.cuda.get_device_name(0)
    if args.seed is not None and args.seed < 0:
        raise CumlxError('E_USAGE', '--seed must not be negative')

    # The script's stdout must not enter the one JSON document on stdout.
    stdout = sys.stderr if args.format == 'json' else None
    rep = runner.run_script(args.script, args.module, list(args.script_args), target=args.target, substitutes=substitutes,
                            allow_cpu_fallback=args.allow_cpu_fallback, timeout=args.timeout, stdout=stdout,
                            trace_losses=args.trace_losses, seed=args.seed, device_name=device_name, torch_version=torch.__version__)
    report_path.write_text(json.dumps(rep, indent=1) + '\n')
    calls = rep['calls']
    by_kind: dict[str, int] = {}
    for call in calls:
        by_kind[call['kind']] = by_kind.get(call['kind'], 0) + 1
    failure = runner.classify_failure(rep)
    r = Result('run')
    r.data = {'target': args.target, 'script_exit_code': rep['script_exit_code'], 'seconds': rep['seconds'], 'processes': rep['processes'],
              'kinds': by_kind, 'seed': args.seed, 'losses_recorded': len(rep['loss_trace']['values']) if args.trace_losses else None, 'policy': rep['policy'], 'environment_changes': rep['environment_changes'], 'verified': False,
              'calls': cap(calls, args.max_items, r)}
    r.artifact('run_report', str(report_path))
    if failure:
        r.exit_code = ERROR_CODES[failure['code']][0]
        r.errors.append(failure)
        r.summary = f"The script failed on {args.target}: {failure['message'][:160]}"
        return r
    r.summary = f"The script ran to its end on {args.target}, with {len(calls)} kind(s) of redirected call. Running is not proof that its results are right."
    r.warn('W_RUN_NOT_VERIFIED', 'The script ran. Its results were not compared with anything.')
    for kind, code, text in (('approximate', 'W_RUN_APPROXIMATE', 'returned a value that only resembles the CUDA value'),
                             ('refused', 'W_RUN_REFUSED', 'were refused, and the script caught the error')):
        names = [c['name'] for c in calls if c['kind'] == kind]
        if names:
            r.warn(code, f"{', '.join(names)} {text}.", calls=names)
    if args.allow_cpu_fallback:
        r.warn('W_RUN_CPU_FALLBACK_ALLOWED', 'Operations that MPS lacks may have run on the CPU. PyTorch does not list them in this report.')
    if any(c['kind'] == 'substituted' for c in calls):
        r.warn('W_RUN_SUBSTITUTED', 'A package was replaced by the --substitute option. Its results can differ from the original.')
    return r


# --------------------------------------------------------------------------- port
def _self(*argv: str) -> dict:
    p = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '5'], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, text=True, check=False)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise CumlxError('E_INTERNAL', f'`cumlx {argv[0]}` gave no JSON result (exit {p.returncode})') from None


def _step(doc: dict) -> dict:
    return {'exit_code': doc.get('exit_code'), 'status': doc.get('status'), 'summary': doc.get('summary'),
            'errors': doc.get('errors', []), 'warnings': [w.get('code') for w in doc.get('warnings', [])]}


def run_port(args) -> Result:
    from ..verify.portability import scan
    from ..verify.trace import check_output_dir, prepare_output_dir
    log = _logger(args)
    check_output_dir(args.output, overwrite=args.overwrite, marker='port.json')
    triage = scan(args.model, revision=args.revision)
    model = triage['model']
    if not model:
        raise CumlxError('E_MODEL_LOAD', f'{args.model} has no model config.json',
                         hint=f'for a source repository, run `cumlx scan {_q(args.model)}`')
    if model.get('mlx_lm_supports') is None:
        raise missing_dependency('mlx_lm', 'diff', purpose='cumlx port')
    if not model['mlx_lm_supports'] or model['needs_remote_code']:
        family = model.get('family')
        if model['needs_remote_code']:
            why = 'needs remote code'
        elif family == 'diffusers':
            why = 'is a diffusers pipeline, which mlx-lm does not implement'
        else:
            why = f"has model type {model['model_type']!r}, which mlx-lm does not implement"
        recipe = {'diffusers': 'diffusers-model', 'vision': 'vision-model'}.get(family)
        hint = (f'`cumlx recipe {recipe}` names a converter for this family; verify its result with `cumlx diff` where the module names can be mapped' if recipe
                else f'cumlx scan {_q(args.model)} ranks the work; write the port by hand, then verify it with `cumlx diff`')
        raise CumlxError('E_MODEL_LOAD', f'{args.model} {why}, so there is no converter to run', hint=hint,
                         details={'tier': triage['tier'], 'model': model})
    inputs = [x for p in args.prompt for x in ('--prompt', p)] + [x for t in args.token_ids for x in ('--token-ids', t)]
    if not inputs:
        inputs = ['--prompt', 'The capital of France is']
    plan = {'model': args.model, 'model_type': model['model_type'], 'dtype': args.dtype, 'q_bits': args.q_bits, 'output': args.output,
            'golden': 'port-weights' if args.q_bits else 'reference'}
    if args.dry_run:
        r = Result('port', summary=f"dry run: mlx-lm implements {model['model_type']}; would convert, compare weights, capture and diff",
                   data={'dry_run': True, 'plan': plan})
        r.suggest(f"cumlx port {_q(args.model)} -o {_q(args.output)}", 'run it')
        return r
    out = prepare_output_dir(args.output, overwrite=args.overwrite, marker='port.json')
    mlx_dir = out / 'mlx'
    convert = [sys.executable, '-m', 'mlx_lm', 'convert', '--hf-path', args.model, '--mlx-path', str(mlx_dir)]
    if args.dtype:
        convert += ['--dtype', args.dtype]
    if args.q_bits:
        convert += ['-q', '--q-bits', str(args.q_bits)]
    if args.revision:
        convert += ['--revision', args.revision]
    log('converting with mlx-lm')
    done = subprocess.run(convert, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise CumlxError('E_MODEL_LOAD', f'mlx_lm convert failed: {done.stderr.strip()[-500:]}',
                         hint='this is the converter, before any verification; check the message and the mlx-lm version')
    steps = {'convert': {'exit_code': 0, 'status': 'ok', 'summary': f'mlx-lm wrote {mlx_dir}'}}
    log('comparing weights')
    weights = _self('weights', args.model, str(mlx_dir), '-o', str(out / 'weights.json'))
    steps['weights'] = _step(weights)
    golden = ['capture', args.model, *inputs, '-o', str(out / 'golden')]
    if args.q_bits:
        golden += ['--weights-from-mlx', str(mlx_dir)]
    if args.revision:
        golden += ['--revision', args.revision]
    log('recording the golden')
    captured = _self(*golden)
    steps['capture'] = _step(captured)
    r = Result('port')
    if captured.get('exit_code') != 0:
        r.exit_code = ExitCode(captured.get('exit_code') or ExitCode.INPUT)
        r.errors.extend(captured.get('errors', []))
        r.summary = f"the port was written, but the golden failed: {captured.get('summary')}"
        r.data = {'verdict': 'error', 'steps': steps, 'plan': plan, 'port': str(mlx_dir)}
        return r
    log('comparing the port')
    compared = _self('diff', str(out / 'golden'), str(mlx_dir), '-o', str(out / 'report'), '--device', args.device)
    steps['diff'] = _step(compared)
    data = compared.get('data') or {}
    verdict = data.get('verdict') or 'error'
    weights_verdict = (weights.get('data') or {}).get('verdict')
    final = verdict
    if verdict == 'ok' and weights_verdict != 'ok':
        final = weights_verdict or 'incomplete'  # a clean diff does not excuse a failed weight comparison
    r.exit_code = VERDICT_EXIT.get(final, ExitCode.INPUT)
    r.errors.extend(compared.get('errors', []))
    r.warnings.extend(w for w in compared.get('warnings', []))
    fd = data.get('first_divergence')
    r.summary = f"{final}: weights {weights_verdict}; {compared.get('summary')}"
    r.data = {'verdict': final, 'port': str(mlx_dir), 'plan': plan, 'steps': steps, 'first_divergence': fd,
              'execution': data.get('execution'), 'native_output': (data.get('summary') or {}).get('native_output')}
    (out / 'port.json').write_text(json.dumps(_finite(r.data), indent=1, ensure_ascii=False) + '\n')
    r.artifact('port', str(mlx_dir))
    r.artifact('report_html', str(out / 'report' / 'report.html'))
    if final == 'ok':
        r.suggest(f"python -m mlx_lm generate --model {_q(mlx_dir)} --prompt Hello", 'use the verified port')
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


# --------------------------------------------------------------------------- rundiff
def run_rundiff(args) -> Result:
    from ..verify.run_compare import compare_runs, load_runs
    rep = compare_runs(load_runs(args.golden), load_runs(args.candidate))
    r = Result('rundiff', exit_code=VERDICT_EXIT[rep['verdict']])
    grade = f"grade {rep['grade']}" if rep['grade'] else 'no grade'
    r.summary = f"{rep['verdict']}, {grade}: {rep['reason']}"
    r.data = rep
    if rep['golden']['target'] != 'cuda':
        r.warn('W_RUNDIFF_GOLDEN_NOT_CUDA', f"The golden ran on {rep['golden']['target']}, so this result says nothing about CUDA.")
    versions = {side: rep[side]['machine'].get('torch') for side in ('golden', 'candidate')}
    if versions['golden'].partition('+')[0] != versions['candidate'].partition('+')[0]:
        r.warn('W_RUNDIFF_TORCH_DIFFERS', f"The sides ran different PyTorch versions: {versions['golden']} and {versions['candidate']}. A difference can come from that.")
    if rep['grade'] == 'A':
        r.warn('W_RUNDIFF_FIRST_STEPS_ONLY', 'Grade A here covers the first steps of training only. Later steps can move apart through rounding.')
    if rep['statistical']:
        r.warn('W_RUNDIFF_ONE_STREAM', 'All losses of a run form one stream. A script with two optimizers, such as a GAN, is compared on their average.')
    return r


# --------------------------------------------------------------------------- kernel
def run_kernel_capture(args) -> Result:
    from ..verify.kernel_capture import capture_kernels, plan_capture
    from ..verify.opgolden import MANIFEST
    from ..verify.trace import check_output_dir, prepare_output_dir
    plan = plan_capture(args.kernels, args.cases, args.header)
    check_output_dir(args.output, overwrite=args.overwrite, marker=MANIFEST)
    case_ids = [c.id for c, _ in plan['cases']]
    r = Result('kernel capture')
    if plan['not_covered']:
        r.warn('W_KERNEL_NOT_COVERED', f"These sources have no cases, so they are not in the golden: {plan['not_covered']}", kernels=plan['not_covered'])
    if args.dry_run:
        r.summary = f"valid: {len(case_ids)} case(s) of {len(plan['kernels'])} kernel(s). Nothing was compiled or run"
        r.data = {'dry_run': True, 'cases': cap(case_ids, args.max_items, r), 'kernels': sorted(plan['kernels'])}
        return r
    out = Path(args.output)
    with tempfile.TemporaryDirectory(prefix='cumlx-kernel-golden-') as tmp:
        staged = Path(tmp) / 'golden'
        result = capture_kernels(plan, staged, arch=args.arch, extra_flags=args.nvcc_flag, repeats=args.repeats, timeout=args.timeout,
                                 jobs=args.jobs, require_idle=args.require_idle_gpu, log=_logger(args))
        if result['manifest'] is not None:
            prepare_output_dir(out, overwrite=args.overwrite, marker=MANIFEST)
            shutil.copytree(staged, out, dirs_exist_ok=True)
    not_repeating = [c['id'] for c in (result['manifest'] or {}).get('cases', []) if not c.get('repeat_identical')]
    r.data = {'kernels_recorded': result['kernels_recorded'], 'cases_recorded': result['recorded'], 'cases_total': len(case_ids),
              'source': result['source'], 'floating_point': result['floating_point'],
              'failed': cap(result['failures'], args.max_items, r), 'not_repeating': not_repeating}
    if not_repeating:
        r.warn('W_KERNEL_NOT_REPEATABLE', f'These cases gave different bytes across repeats, so the golden holds one of several results: {not_repeating}')
    r.summary = (f"recorded {result['recorded']} of {len(case_ids)} case(s), {len(result['kernels_recorded'])} kernel(s), "
                 f"on {result['source']['gpu']} (driver {result['source']['driver']}, nvcc {result['source']['compiler']['version']})")
    if result['failures']:
        r.exit_code = ExitCode.INCOMPLETE
        r.summary += f"; {len(result['failures'])} case(s) did not compile or run and are not in the golden"
        for failure in result['failures'][:args.max_items if args.max_items >= 0 else None]:
            r.warn('W_KERNEL_CASE_FAILED', f"{failure['case']} ({failure['stage']}): {failure['message'][:400]}", case=failure['case'], stage=failure['stage'])
    if result['manifest'] is not None:
        r.artifact('golden', str(out))
        r.suggest(f'cumlx kernel verify foo.metal {_q(out)}', 'on a Mac: compare a Metal kernel with this golden')
    return r


def run_kernel_verify(args) -> Result:
    from ..verify.kernel_verify import report_markdown, verify_kernels
    from ..verify.trace import check_output_dir, prepare_output_dir
    if args.output:
        check_output_dir(args.output, overwrite=args.overwrite, marker='report.json')
    report = verify_kernels(args.metal, args.golden, kernel=args.kernel, cases=args.case or None, header=args.header,
                            engine_name=args.engine, repeats=args.repeats, log=_logger(args))
    r = Result('kernel verify', exit_code=VERDICT_EXIT[report['verdict']])
    s = report['summary']
    r.summary = (f"{report['verdict']}: {s['kernels']['pass']} kernel(s) pass, {s['kernels']['fail']} fail, {s['kernels']['unverified']} unverified; "
                 f"{s['cases']['pass']} of {sum(s['cases'].values())} case(s) pass; {len(s['unmapped'])} unmapped, {len(s['not_covered'])} not covered")
    failures = [{'case': x['case'], 'kernel': x['kernel'], 'outputs': x['outputs']} for x in report['records'] if x['status'] == 'fail']
    unverified = [{'case': x['case'], 'kernel': x['kernel'], 'stage': x.get('stage'), 'reason': x.get('reason')}
                  for x in report['records'] if x['status'] == 'unverified']
    r.data = {'verdict': report['verdict'], 'first_divergence': report['first_divergence'], 'summary': s, 'kernels': report['kernels'],
              'golden_kind': report['golden_kind'], 'thresholds': report['thresholds'], 'execution': report['execution'],
              'translations': report['translations'], 'failures': cap(failures, args.max_items, r), 'unverified': cap(unverified, args.max_items, r)}
    r.warnings.extend(report['warnings'])
    if args.output:
        from .output import _sanitize
        out = prepare_output_dir(args.output, overwrite=args.overwrite, marker='report.json')
        (out / 'report.json').write_text(json.dumps(_sanitize(report), indent=1, allow_nan=False) + '\n')
        (out / 'report.md').write_text(report_markdown(report))
        r.artifact('report', str(out / 'report.json'))
        r.artifact('report_md', str(out / 'report.md'))
    if s['unmapped']:
        r.suggest(f'cumlx kernel verify {_q(args.metal)} {_q(args.golden)} --kernel NAME', 'name the golden kernel that this file translates')
    return r


def run_kernel_port(args) -> Result:
    from ..verify import kernel_port
    out = Path(args.output)
    sidecar = out.with_name(out.name.split('.')[0] + '.port.json')
    header_out = out.with_name(out.name.split('.')[0] + '.header.metal')
    for path in (out, sidecar):
        if path.exists() and not args.overwrite:
            raise CumlxError('E_OUTPUT_EXISTS', f'{path} exists', hint='pass --overwrite or choose a new path with -o')
    adapter = None if args.preprocess_only else kernel_port.load_translator(args.translator)
    pre = kernel_port.preprocess_file(args.source, instantiate=args.instantiate, include_dirs=args.include_dir, header=args.header,
                                      demote_fp64=args.demote_fp64, cases_file=args.cases, case_id=args.case)
    r = Result('kernel port')
    r.warnings.extend(pre.warnings)
    side: dict[str, Any] = {'schema': kernel_port.PORT_SCHEMA, 'source': str(args.source), 'source_sha256': pre.source_sha256,
                            'prepared_sha256': kernel_port.sha256_text(pre.source), 'kernel': pre.name,
                            'form': pre.form, 'entry_points': pre.entry_points, 'preprocess': pre.record}
    if adapter is None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text((pre.header + '\n' if pre.header else '') + pre.source)
        side['translator'] = None
        sidecar.write_text(json.dumps(side, indent=1) + '\n')
        r.summary = f'prepared {pre.name}: nothing was translated'
        r.data = {**side}
        r.artifact('prepared_source', str(out))
        r.artifact('port_record', str(sidecar))
        r.warn('W_KERNEL_NOT_TRANSLATED', '--preprocess-only stops before translation. This is not a Metal kernel.')
        return r
    result = kernel_port.translate(pre, adapter, args.include_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.msl)
    if result.header.strip():
        header_out.write_text(result.header)
        r.artifact('metal_header', str(header_out))
    side.update(translator={'name': result.translator, 'version': result.version, 'notes': result.notes}, metal_form=result.form,
                metal_entry_points=result.entry_points)
    sidecar.write_text(json.dumps(side, indent=1) + '\n')
    r.summary = f"translated {pre.name} with {result.translator}. It is unverified until `cumlx kernel verify` compares it with a CUDA golden"
    r.data = {**side, 'grade': 'unverified'}
    r.artifact('metal', str(out))
    r.artifact('port_record', str(sidecar))
    r.warn('W_KERNEL_NOT_VERIFIED', 'A kernel that translates is not a kernel that is right. Compare it with a CUDA golden.')
    r.suggest(f'cumlx kernel verify {_q(out)} kgolden/', 'compare the translation with a golden recorded on an NVIDIA machine')
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
