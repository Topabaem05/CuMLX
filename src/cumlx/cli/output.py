"""Result envelope shared by every command, and its JSON/text renderers.

Contract (``cumlx.result.v1``), stable for agents:

* In JSON mode stdout carries exactly ONE JSON document and nothing else.
  Library chatter is redirected to stderr while a command runs.
* ``exit_code`` in the document equals the process exit status.
* ``status`` is one of ``ok | diverged | incomplete | error``.
* Long lists are capped by ``--max-items``; ``truncated`` tells you when, and
  the complete data is in the file listed under ``artifacts``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

from ..errors import ExitCode

RESULT_SCHEMA = 'cumlx.result.v1'
STATUS_BY_EXIT = {
    ExitCode.OK: 'ok', ExitCode.DIVERGED: 'diverged', ExitCode.INCOMPLETE: 'incomplete',
}


@dataclass
class Result:
    command: str
    exit_code: ExitCode = ExitCode.OK
    summary: str = ''
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    next_steps: list[dict[str, str]] = field(default_factory=list)
    truncated: bool = False

    def warn(self, code: str, message: str, **extra: Any) -> None:
        self.warnings.append({'code': code, 'message': message, **extra})

    def suggest(self, command: str, why: str) -> None:
        self.next_steps.append({'command': command, 'why': why})

    def artifact(self, kind: str, path: str) -> None:
        self.artifacts.append({'kind': kind, 'path': path})

    @property
    def status(self) -> str:
        return STATUS_BY_EXIT.get(self.exit_code, 'error')

    def to_dict(self, *, version: str, elapsed_ms: int) -> dict[str, Any]:
        return {
            'schema': RESULT_SCHEMA,
            'command': self.command,
            'ok': self.exit_code == ExitCode.OK,
            'status': self.status,
            'exit_code': int(self.exit_code),
            'summary': self.summary,
            'data': self.data,
            'artifacts': self.artifacts,
            'warnings': self.warnings,
            'errors': self.errors,
            'next_steps': self.next_steps,
            'truncated': self.truncated,
            'cumlx_version': version,
            'elapsed_ms': elapsed_ms,
        }


def cap(items: list, max_items: int, result: Result) -> dict[str, Any]:
    """Cap a list for stdout. Returns {'items', 'total', 'omitted'}."""
    total = len(items)
    if max_items >= 0 and total > max_items:
        result.truncated = True
        return {'items': items[:max_items], 'total': total, 'omitted': total - max_items}
    return {'items': items, 'total': total, 'omitted': 0}


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, 'item') and callable(obj.item):  # NumPy scalars
        try:
            return _sanitize(obj.item())
        except (TypeError, ValueError):
            pass
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def to_json(doc: dict[str, Any]) -> str:
    return json.dumps(_sanitize(doc), indent=2, ensure_ascii=False, allow_nan=False)


def _text_value(v: Any) -> str:
    if isinstance(v, float):
        return f'{v:.4g}'
    if isinstance(v, (dict, list)):
        s = json.dumps(_sanitize(v), ensure_ascii=False)
        return s if len(s) <= 160 else s[:157] + '...'
    return str(v)


def to_text(doc: dict[str, Any]) -> str:
    lines = [f"[{doc['status'].upper()}] cumlx {doc['command']}: {doc['summary']}"]
    for k, v in doc['data'].items():
        lines.append(f'  {k}: {_text_value(v)}')
    for a in doc['artifacts']:
        lines.append(f"  artifact ({a['kind']}): {a['path']}")
    for w in doc['warnings']:
        lines.append(f"  warning {w['code']}: {w['message']}")
    for e in doc['errors']:
        lines.append(f"  error {e['code']}: {e['message']}")
        if e.get('hint'):
            lines.append(f"    hint: {e['hint']}")
    for s in doc['next_steps']:
        lines.append(f"  next: {s['command']}    # {s['why']}")
    if doc['truncated']:
        lines.append('  (lists truncated by --max-items; see artifacts for full data)')
    lines.append(f"  exit_code={doc['exit_code']}")
    return '\n'.join(lines)
