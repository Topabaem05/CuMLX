"""Human-readable Markdown rendering of a diff report (the JSON stays the source of truth)."""
from __future__ import annotations

from typing import Any


def _f(v: Any) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.3g}'
    return str(v)


def _thresholds(t: dict[str, Any]) -> str:
    text = f"chain {t['rel_l2']:g}"
    if t.get('composite_rel_l2') is not None:
        text += f", leaf {t['leaf_rel_l2']:g}, composite {t['composite_rel_l2']:g}"
    return text


def _execution_lines(rep: dict[str, Any]) -> list[str]:
    s = rep['summary']
    lines = []
    ex = rep.get('execution')
    if ex:
        lines.append(f"- execution: {ex['device']}" + (f" ({ex['device_name']})" if ex.get('device_name') else '') +
                     f", TF32 {ex['tf32']}, measured float32 GEMM error {ex['gemm_noise_rel_l2']:.2e}")
    if s.get('worst_isolated_rel_l2') is not None or s.get('worst_chain_rel_l2') is not None:
        lines.append(f"- worst rel_l2: isolated {_f(s.get('worst_isolated_rel_l2'))}, chain {_f(s.get('worst_chain_rel_l2'))}"
                     f" | chain drift: {s.get('chain_drift', 0)} record(s)")
    n = s.get('native_output')
    if n:
        lines.append(f"- native-dtype prediction: top-1 matches {n['top1_matches']}/{n['positions']} token positions")
    return lines


def to_markdown(rep: dict[str, Any], *, max_rows: int = 200) -> str:
    s = rep['summary']
    fd = rep.get('first_divergence')
    lines = [
        '# CUMLX diff report', '',
        f"- verdict: **{rep['verdict']}**  (mode: {rep['mode']})",
        f"- trace: `{rep['trace']['path']}` (golden: {rep['trace']['golden'].get('kind')}, "
        f"samples: {rep['trace']['samples']})",
        f"- port: `{rep['port']['path']}` (profile: {rep['port']['profile']}, thresholds rel_l2: "
        f"{_thresholds(rep['threshold'])})",
        f"- records: {s['records']} | pass {s['pass']} | fail {s['fail']} | unverified {s['unverified']} | "
        f"skipped {s['skipped']} | ignored {s['ignored']} | parametrized coverage {s['parametrized_coverage']:.1%}",
        *_execution_lines(rep),
        '',
    ]
    if fd:
        lines += ['## First divergence', '',
                  f"- module: `{fd['module'] or '(root)'}` (sample {fd['sample']})",
                  f"- phase: {fd.get('phase', 'prefill')}, basis: {fd['basis']}, confidence: {fd['confidence']}",
                  f"- first failing output: `{fd.get('first_failing_output', fd['module'])}`",
                  f"- {fd['explanation']}", '']
    for w in rep.get('warnings', []):
        lines.append(f"> **{w['code']}**: {w['message']}" + (f" — `{w['hint']}`" if w.get('hint') else ''))
    if rep.get('warnings'):
        lines.append('')
    lines += ['## Records (completion order)', '',
              '| sample | phase | module | type | status | cause | chain rel_l2 | isolated rel_l2 |',
              '|---|---|---|---|---|---|---|---|']
    for e in rep['records'][:max_rows]:
        c = (e.get('chain') or {}).get('rel_l2')
        i = (e.get('isolated') or {}).get('rel_l2')
        mark = {'fail': '❌', 'pass': '✅', 'unverified': '⚠️'}.get(e['status'], '·')
        lines.append(f"| {e['sample']} | {e.get('phase', 'prefill')} | `{e['name'] or '(root)'}` | {e['type']} | {mark} {e['status']} | "
                     f"{e.get('fail_cause', '')} | {_f(c)} | {_f(i)} |")
    if len(rep['records']) > max_rows:
        lines.append(f"\n({len(rep['records']) - max_rows} more rows in report.json)")
    return '\n'.join(lines) + '\n'


def compact_records(rep: dict[str, Any], statuses: tuple[str, ...]) -> list[dict]:
    out = []
    for e in rep['records']:
        if e['status'] in statuses:
            out.append({k: e.get(k) for k in ('sample', 'name', 'type', 'status', 'fail_cause', 'note', 'tied_to')} |
                       {'chain_rel_l2': (e.get('chain') or {}).get('rel_l2'),
                        'isolated_rel_l2': (e.get('isolated') or {}).get('rel_l2')})
    return out
