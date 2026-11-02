"""Human-readable Markdown rendering of a diff report (the JSON stays the source of truth)."""
from __future__ import annotations

from typing import Any


def _f(v: Any) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.3g}'
    return str(v)


def to_markdown(rep: dict[str, Any], *, max_rows: int = 200) -> str:
    s = rep['summary']
    fd = rep.get('first_divergence')
    lines = [
        '# CUMLX diff report', '',
        f"- verdict: **{rep['verdict']}**  (mode: {rep['mode']})",
        f"- trace: `{rep['trace']['path']}` (golden: {rep['trace']['golden'].get('kind')}, "
        f"samples: {rep['trace']['samples']})",
        f"- port: `{rep['port']['path']}` (profile: {rep['port']['profile']}, threshold rel_l2 <= "
        f"{rep['threshold']['rel_l2']:g})",
        f"- records: {s['records']} | pass {s['pass']} | fail {s['fail']} | unverified {s['unverified']} | "
        f"skipped {s['skipped']} | ignored {s['ignored']} | parametrized coverage {s['parametrized_coverage']:.1%}",
        '',
    ]
    if fd:
        lines += ['## First divergence', '',
                  f"- module: `{fd['module'] or '(root)'}` (sample {fd['sample']})",
                  f"- basis: {fd['basis']}, confidence: {fd['confidence']}",
                  f"- first failing output: `{fd.get('first_failing_output', fd['module'])}`",
                  f"- {fd['explanation']}", '']
    for w in rep.get('warnings', []):
        lines.append(f"> **{w['code']}**: {w['message']}" + (f" — `{w['hint']}`" if w.get('hint') else ''))
    if rep.get('warnings'):
        lines.append('')
    lines += ['## Records (completion order)', '',
              '| sample | module | type | status | cause | chain rel_l2 | isolated rel_l2 |',
              '|---|---|---|---|---|---|---|']
    for e in rep['records'][:max_rows]:
        c = (e.get('chain') or {}).get('rel_l2')
        i = (e.get('isolated') or {}).get('rel_l2')
        mark = {'fail': '❌', 'pass': '✅', 'unverified': '⚠️'}.get(e['status'], '·')
        lines.append(f"| {e['sample']} | `{e['name'] or '(root)'}` | {e['type']} | {mark} {e['status']} | "
                     f"{e.get('fail_cause', '')} | {_f(c)} | {_f(i)} |")
    if len(rep['records']) > max_rows:
        lines.append(f"\n({len(rep['records']) - max_rows} more rows in report.json)")
    return '\n'.join(lines) + '\n'


def compact_records(rep: dict[str, Any], statuses: tuple[str, ...]) -> list[dict]:
    out = []
    for e in rep['records']:
        if e['status'] in statuses:
            out.append({k: e.get(k) for k in ('sample', 'name', 'type', 'status', 'fail_cause', 'note')} |
                       {'chain_rel_l2': (e.get('chain') or {}).get('rel_l2'),
                        'isolated_rel_l2': (e.get('isolated') or {}).get('rel_l2')})
    return out
