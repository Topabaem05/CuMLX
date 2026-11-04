"""docs/PLATFORM-REPORT.md must agree with the gates in docs/ROADMAP-PLATFORM.md, so that a tick cannot move without it."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / 'docs'
GRADES = re.compile(r'^(A|B|C|미검증)\b')
FLAGSHIPS = 5


def roadmap_gates(text: str) -> dict[str, bool]:
    """{'M2-1': True, ...}: each completion gate of each milestone, in order, and whether it is ticked."""
    gates: dict[str, bool] = {}
    for match in re.finditer(r'^### (M\d)\..*?(?=^### |^## )', text, re.M | re.S):
        for n, tick in enumerate(re.findall(r'^- \[([ x])\]', match.group(0), re.M), 1):
            gates[f'{match.group(1)}-{n}'] = tick == 'x'
    return gates


def report_rows(text: str) -> dict[str, tuple[str, str]]:
    """{'M2-1': (state, grade), ...}"""
    rows: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells and re.fullmatch(r'M\d-\d+', cells[0]) and len(cells) >= 6:
            rows[cells[0]] = (cells[2], cells[3])
    return rows


def disagreements(roadmap: str, report: str) -> list[str]:
    gates, rows = roadmap_gates(roadmap), report_rows(report)
    problems = [f'{g}: no row' for g in gates if g not in rows] + [f'{r}: no gate' for r in rows if r not in gates]
    for gate, ticked in gates.items():
        if gate in rows:
            state, grade = rows[gate]
            if (state == '달성') != ticked:
                problems.append(f'{gate}: roadmap {ticked}, report {state}')
            if not GRADES.match(grade):
                problems.append(f'{gate}: grade {grade!r}')
    total = re.search(r'\*\*합계\*\* \| \*\*(\d+)\*\* \| \*\*(\d+)\*\*', report)
    if not total or (int(total.group(1)), int(total.group(2))) != (sum(gates.values()), len(gates)):
        problems.append(f'the total does not say {sum(gates.values())} of {len(gates)}')
    return problems


@pytest.fixture
def texts():
    return (DOCS / 'ROADMAP-PLATFORM.md').read_text(encoding='utf-8'), (DOCS / 'PLATFORM-REPORT.md').read_text(encoding='utf-8')


def test_every_gate_has_a_row_with_the_same_state_and_a_grade(texts):
    roadmap, report = texts
    assert len(roadmap_gates(roadmap)) >= 25  # The parser found the gates, so an empty comparison cannot pass.
    assert disagreements(roadmap, report) == []


def test_a_row_that_disagrees_is_found(texts):
    roadmap, report = texts
    assert any(p.startswith('M6-1') for p in disagreements(roadmap, re.sub(r'(\| M6-1 \|[^|]*\| )미달', r'\1달성', report)))
    assert any(p.startswith('M3-5') for p in disagreements(roadmap, re.sub(r'(\| M3-5 \|[^|]*\| 달성 \| )A', r'\1좋음', report)))
    assert any('no row' in p for p in disagreements(roadmap, re.sub(r'^\| M5-3 .*$', '', report, flags=re.M)))
    ticked = roadmap.replace('- [ ] 기함 작업을 새 Mac에서', '- [x] 기함 작업을 새 Mac에서')
    assert ticked != roadmap and any(p.startswith('M6-1') for p in disagreements(ticked, report))


def test_no_flagship_task_is_called_reproduced_and_the_failures_stay(texts):
    report = texts[1]
    rows = [line for line in report.splitlines() if re.match(r'^\| F\d \|', line)]
    assert len(rows) == FLAGSHIPS
    for row in rows:
        state = row.split('|')[4].strip()
        assert state.startswith(('재현 안 됨', '일부 달성')), row
    for failure in ('Go2 시드 1', 'Isaac Lab 처리량', 'MuJoCo Warp 일치, 첫 시험', 'MLX PPO 업데이트, 첫 시험', 'OMNI_KIT_ACCEPT_EULA'):
        assert failure in report


@pytest.mark.parametrize('name', ['PLATFORM-REPORT.md', 'UNVERIFIED.md'])
def test_the_documents_hold_no_local_paths_or_addresses(name):
    text = (DOCS / name).read_text(encoding='utf-8')
    assert not re.search(r'/Users/|/home/|[\w.+-]+@[\w-]+\.\w+', text)
