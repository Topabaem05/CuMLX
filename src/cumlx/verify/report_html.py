"""Self-contained HTML rendering of a diff report: one file, no network, works offline and in CI artifacts.

The JSON stays the source of truth. The chart plots the error of every comparison against execution order on
a log scale, so a fault shows as a step in the chain series and a single spike in the isolated series.
"""
from __future__ import annotations

import html
import json
import math
from typing import Any

WIDTH, HEIGHT, LEFT, RIGHT, TOP, BOTTOM = 960, 300, 56, 16, 16, 34
Y_MIN, Y_MAX = -9.0, 1.0  # log10(rel_l2)

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--grid:#e6e6e6;--chain:#b8860b;--iso:#2257a7;--fail:#c62828;--ok:#2e7d32;--card:#f6f6f6}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#aaa;--grid:#333;--chain:#e0b040;--iso:#7fa8e8;--fail:#ef6f6c;--ok:#7bc77f;--card:#1f1f1f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1040px}
h1{font-size:22px;margin:.2em 0}h2{font-size:17px;margin:1.3em 0 .4em}code{font-size:.92em}
.verdict{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:600;background:var(--card)}
.ok{color:var(--ok)}.diverged{color:var(--fail)}.incomplete{color:var(--chain)}
.card{background:var(--card);padding:10px 14px;border-radius:6px;margin:8px 0;overflow-wrap:anywhere}
svg{width:100%;height:auto;display:block}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:3px 8px;border-bottom:1px solid var(--grid);text-align:left}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.fail td{color:var(--fail)}.wrap{overflow-x:auto}.legend span{margin-right:16px}.muted{color:var(--muted)}
"""


def _esc(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _num(v: Any) -> str:
    return f'{v:.3g}' if isinstance(v, (int, float)) and not isinstance(v, bool) else '-'


def _y(v: float) -> float:
    clamped = min(max(math.log10(max(v, 1e-12)), Y_MIN), Y_MAX)
    return TOP + (Y_MAX - clamped) / (Y_MAX - Y_MIN) * (HEIGHT - TOP - BOTTOM)


def _chart(rows: list[dict], threshold: float | None) -> str:
    n = max(len(rows) - 1, 1)
    x = lambda i: LEFT + i / n * (WIDTH - LEFT - RIGHT)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="rel_l2 per recorded module, in execution order">']
    for tick in range(int(Y_MIN), int(Y_MAX) + 1, 2):
        y = _y(10.0 ** tick)
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>'
                     f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">1e{tick}</text>')
    if threshold:
        y = _y(threshold)
        parts.append(f'<line x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--fail)" stroke-dasharray="4 3"/>')
    for key, color in (('chain', 'var(--chain)'), ('isolated', 'var(--iso)')):
        for i, e in enumerate(rows):
            metric = e.get(key) or {}
            v = metric.get('rel_l2')
            if v is None:
                continue
            bad = metric.get('verdict') == 'fail' and e['status'] == 'fail'
            fill = 'var(--fail)' if bad else color
            label = _esc(f"{e['name'] or '(root)'} [{e.get('phase', 'prefill')}] {key} {v:.3g}")
            parts.append(f'<circle cx="{x(i):.1f}" cy="{_y(v):.1f}" r="{3 if bad else 1.6}" fill="{fill}"><title>{label}</title></circle>')
    parts.append(f'<text x="{(LEFT + WIDTH) / 2}" y="{HEIGHT - 8}" text-anchor="middle" font-size="12" fill="var(--muted)">'
                 'recorded module, in execution order</text></svg>')
    return ''.join(parts)


def to_html(rep: dict[str, Any], *, max_rows: int = 400) -> str:
    s, t = rep['summary'], rep['threshold']
    fd = rep.get('first_divergence')
    ex = rep.get('execution') or {}
    sample = rep['records'][0]['sample'] if rep['records'] else None
    rows = [e for e in rep['records'] if e['sample'] == sample]
    verdict = rep['verdict']
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1"><title>CUMLX diff report</title>'
           f'<style>{CSS}</style></head><body><h1>CUMLX diff report</h1>'
           f'<p><span class="verdict {_esc(verdict)}">{_esc(verdict)}</span> '
           f'<span class="muted">mode {_esc(rep["mode"])}, {s["records"]} records</span></p>'
           f'<div class="card"><div>Port: <code>{_esc(rep["port"]["path"])}</code> (profile {_esc(rep["port"]["profile"])})</div>'
           f'<div>Golden: <code>{_esc(rep["trace"]["source"].get("model", ""))}</code> '
           f'revision <code>{_esc(rep["trace"]["source"].get("revision"))}</code>, kind {_esc(rep["trace"]["golden"].get("kind"))}</div>'
           f'<div>Execution: {_esc(ex.get("device"))} {_esc(ex.get("device_name") or "")}, TF32 {_esc(ex.get("tf32"))}, '
           f'measured float32 matmul error {_num(ex.get("gemm_noise_rel_l2"))}</div>'
           f'<div>pass {s["pass"]}, fail {s["fail"]}, unverified {s["unverified"]}, skipped {s["skipped"]}, ignored {s["ignored"]}, '
           f'chain drift {s.get("chain_drift", 0)}, decode records {s.get("decode_records", 0)}, '
           f'parametrized coverage {s["parametrized_coverage"]:.1%}</div></div>']
    if fd:
        out.append(f'<h2>First divergence</h2><div class="card"><code>{_esc(fd["module"] or "(root)")}</code> '
                   f'({_esc(fd.get("phase", "prefill"))} phase, basis {_esc(fd["basis"])}, confidence {_esc(fd["confidence"])}, '
                   f'rel_l2 {_num((fd.get("metrics") or {}).get("rel_l2"))})<br>{_esc(fd["explanation"])}</div>')
    for w in rep.get('warnings', []):
        out.append(f'<div class="card"><b>{_esc(w["code"])}</b>: {_esc(w["message"])}'
                   + (f'<br><code>{_esc(w["hint"])}</code>' if w.get('hint') else '') + '</div>')
    out.append(f'<h2>Error per module (sample {_esc(sample)})</h2><div class="legend">'
               '<span style="color:var(--iso)">● isolated</span><span style="color:var(--chain)">● chain</span>'
               '<span style="color:var(--fail)">● failing</span>'
               f'<span class="muted">dashed: leaf threshold {_num(t.get("leaf_rel_l2", t.get("rel_l2")))}</span></div>')
    out.append(_chart(rows, t.get('leaf_rel_l2', t.get('rel_l2'))))
    ordered = sorted(rep['records'], key=lambda e: (e['status'] != 'fail', e['status'] != 'unverified'))
    out.append('<h2>Records</h2><p class="muted">Failing and unverified records come first.</p><div class="wrap"><table><thead><tr>'
               '<th>sample</th><th>phase</th><th>module</th><th>type</th><th>status</th>'
               '<th class="n">isolated rel_l2</th><th class="n">chain rel_l2</th><th>note</th></tr></thead><tbody>')
    for e in ordered[:max_rows]:
        note = e.get('note') or ('chain drift' if e.get('chain_drift') else (f"tied to {e['tied_to']}" if e.get('tied_to') else ''))
        out.append(f'<tr class="{_esc(e["status"])}"><td>{_esc(e["sample"])}</td><td>{_esc(e.get("phase", "prefill"))}</td>'
                   f'<td><code>{_esc(e["name"] or "(root)")}</code></td><td>{_esc(e["type"])}</td><td>{_esc(e["status"])}</td>'
                   f'<td class="n">{_num((e.get("isolated") or {}).get("rel_l2"))}</td>'
                   f'<td class="n">{_num((e.get("chain") or {}).get("rel_l2"))}</td><td>{_esc(note)}</td></tr>')
    out.append('</tbody></table></div>')
    if len(ordered) > max_rows:
        out.append(f'<p class="muted">{len(ordered) - max_rows} more records are in report.json.</p>')
    out.append(f'<script type="application/json" id="summary">{_esc(json.dumps(s))}</script></body></html>')
    return ''.join(out)
