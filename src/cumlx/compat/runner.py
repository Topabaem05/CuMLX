"""Start a script with the compatibility layer, and turn what it recorded into a report."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

REPORT_SCHEMA = 'cumlx.run-report.v1'
FALLBACK_ENV = 'PYTORCH_ENABLE_MPS_FALLBACK'
SITE = Path(__file__).resolve().parent / 'site'


def child_environment(target: str, events: str, substitutes: dict[str, str], allow_cpu_fallback: bool,
                      trace_losses: bool = False, seed: int | None = None) -> tuple[dict[str, str], dict[str, str]]:
    env = dict(os.environ)
    changes: dict[str, str] = {}
    env['PYTHONPATH'] = os.pathsep.join(p for p in (str(SITE), env.get('PYTHONPATH')) if p)
    env['CUMLX_RUN_TARGET'] = target
    env['CUMLX_RUN_EVENTS'] = events
    env['CUMLX_RUN_SUBSTITUTE'] = json.dumps(substitutes)
    env['CUMLX_RUN_TRACE_LOSSES'] = '1' if trace_losses else '0'
    env['CUMLX_RUN_SEED'] = '' if seed is None else str(seed)
    if allow_cpu_fallback:
        env[FALLBACK_ENV] = '1'
        changes[FALLBACK_ENV] = 'set to 1 by --allow-cpu-fallback'
    elif FALLBACK_ENV in env:
        del env[FALLBACK_ENV]
        changes[FALLBACK_ENV] = 'removed: pass --allow-cpu-fallback to allow it'
    return env, changes


def read_events(path: str) -> dict[str, Any]:
    calls: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    uncaught: list[dict[str, Any]] = []
    processes: set[int] = set()
    losses: dict[int, list[float | None]] = {}
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # A process that was killed can leave half a line.
        processes.add(event.get('pid'))
        if event['event'] == 'loss':
            losses.setdefault(event['pid'], []).append(event['value'])
            continue
        if event['event'] == 'uncaught':
            uncaught.append({k: event.get(k) for k in ('type', 'message', 'where', 'module_name', 'pid')})
            continue
        key = (event['kind'], event['name'], event.get('to'))
        entry = calls.setdefault(key, {'kind': event['kind'], 'name': event['name'], 'to': event.get('to'), 'count': 0, 'first_site': None, 'first_calls': 0})
        if event['event'] == 'call':
            entry['first_calls'] += 1
            entry['first_site'] = entry['first_site'] or event.get('site')
        else:
            entry['count'] += event['count']
    out = []
    for entry in calls.values():
        # A process that did not reach its exit handler wrote no count, so at least its first call is counted.
        entry['count'] = max(entry['count'], entry.pop('first_calls'))
        out.append(entry)
    out.sort(key=lambda e: (e['kind'], e['name']))
    # The process with the most backward() calls is the trainer. Others are listed so that a comparison can refuse them.
    main_trace = max(losses.values(), key=len, default=[])
    return {'calls': out, 'uncaught': uncaught, 'processes': len(processes),
            'loss_trace': {'values': main_trace, 'processes_with_losses': len(losses)}}


def _policy(target: str, allow_cpu_fallback: bool) -> dict[str, Any]:
    if target == 'cuda':
        return {'torch.cuda.is_available': True, 'device_name': 'as reported by the driver', 'cpu_fallback_allowed': False,
                'meaning': 'Nothing was redirected. This is a run on a real CUDA device.'}
    return {'torch.cuda.is_available': True, 'device_name': f'{target.upper()} (cumlx run)', 'cpu_fallback_allowed': allow_cpu_fallback,
            'meaning': 'The script believes that CUDA exists. This report lists each call that was redirected.'}


def _sha256(path: str | None) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def run_script(script: str | None, module: str | None, script_args: list[str], *, target: str,
               substitutes: dict[str, str], allow_cpu_fallback: bool, timeout: float | None, stdout: Any,
               trace_losses: bool = False, seed: int | None = None, device_name: str | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='cumlx-run-') as scratch:
        events = str(Path(scratch) / 'events.jsonl')
        env, env_changes = child_environment(target, events, substitutes, allow_cpu_fallback, trace_losses, seed)
        argv = [sys.executable, *(['-m', module] if module else [script]), *script_args]
        start = time.monotonic()
        timed_out = False
        try:
            code = subprocess.run(argv, env=env, stdout=stdout, timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
        elapsed = round(time.monotonic() - start, 3)
        seen = read_events(events)
    report = {
        'schema': REPORT_SCHEMA, 'target': target, 'script': script, 'module': module, 'script_args': script_args,
        'script_sha256': _sha256(script), 'seed': seed, 'script_exit_code': code, 'timed_out': timed_out, 'seconds': elapsed,
        'machine': {'platform': platform.platform(), 'arch': platform.machine(), 'device_name': device_name},
        'policy': _policy(target, allow_cpu_fallback),
        'environment_changes': env_changes, 'substitutes': substitutes, 'processes': seen['processes'],
        'calls': seen['calls'], 'uncaught': seen['uncaught'],
        'verified': False, 'limits': 'A script that runs is grade C at most. Compare its outputs with a CUDA golden for more.',
    }
    if trace_losses:
        report['loss_trace'] = seen['loss_trace']
    return report


def classify_failure(report: dict[str, Any]) -> dict[str, Any] | None:
    """The error for a script that failed, or None when it ran to its end."""
    code = report['script_exit_code']
    if report['timed_out']:
        return {'code': 'E_RUN_SCRIPT_FAILED', 'message': 'The script ran past --timeout and was stopped.', 'hint': 'raise --timeout'}
    if code == 0:
        return None
    first = report['uncaught'][0] if report['uncaught'] else None
    if first is None:
        return {'code': 'E_RUN_SCRIPT_FAILED', 'message': f'The script ended with exit code {code}.', 'hint': 'read its output on stderr'}
    kind, message = first['type'], first['message']
    details = {'exception': kind, 'where': first['where'], 'script_exit_code': code}
    if kind == 'UnsupportedCudaCall':
        return {'code': 'E_RUN_UNSUPPORTED', 'message': message, 'hint': 'remove this call from the script, or guard it', 'details': details}
    lowered = message.lower()
    if kind == 'NotImplementedError' and 'mps' in lowered:
        return {'code': 'E_RUN_MPS_OP_MISSING', 'message': message, 'hint': 'pass --allow-cpu-fallback to run this op on the CPU, and expect it in the report', 'details': details}
    if 'mps' in lowered and ('float64' in lowered or 'double' in lowered):
        return {'code': 'E_RUN_MPS_DTYPE', 'message': message, 'hint': 'MPS has no float64. Use float32 where the script creates this tensor. CUMLX does not change precision for you', 'details': details}
    if kind == 'ModuleNotFoundError':
        name = first.get('module_name') or message
        return {'code': 'E_RUN_MISSING_PACKAGE', 'message': message, 'hint': f'run `cumlx recipe {name}` for a replacement, then pass --substitute {name}=<module> or install it', 'details': details}
    return {'code': 'E_RUN_SCRIPT_FAILED', 'message': f'{kind}: {message}', 'hint': 'read the traceback on stderr', 'details': details}
