"""The compatibility catalog: for a CUDA dependency, what to use on Apple Silicon and how far that claim was checked.

Every option carries its evidence. ``run`` means that it was executed on Apple Silicon and the result is in the named
file under ``validation/``. ``docs`` means that only its documentation was read. Grades are the proof grades of
docs/ROADMAP-PLATFORM.md: A numeric parity, B statistical parity, C it runs.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
import re
from typing import Any

from ..errors import CumlxError

SCHEMA = 'cumlx.recipes.v1'
CATALOG = Path(__file__).with_name('recipes.json')
TIERS = ('model', 'device', 'library', 'kernel', 'platform')
OUTCOMES = ('substitute', 'remove', 'none')
EVIDENCE = ('run', 'docs')
GRADES = ('A', 'B', 'C')


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or CATALOG).read_text(encoding='utf-8'))


def validate_catalog(catalog: dict[str, Any], repo_root: Path | None = None) -> list[str]:
    """Return every problem in a catalog. With repo_root, `run` evidence must name a file that exists."""
    problems = []
    if catalog.get('schema') != SCHEMA:
        problems.append(f'schema is not {SCHEMA}')
    seen: set[str] = set()
    for recipe in catalog.get('recipes', []):
        rid = recipe.get('id', '<no id>')
        names = {_key(n) for n in (rid, *recipe.get('aliases', []), *recipe.get('packages', []))}
        problems.extend(f'{rid}: {name!r} is claimed twice' for name in sorted(names & seen))
        seen |= names
        if recipe.get('tier') not in TIERS:
            problems.append(f'{rid}: bad tier {recipe.get("tier")!r}')
        if recipe.get('outcome') not in OUTCOMES:
            problems.append(f'{rid}: bad outcome {recipe.get("outcome")!r}')
        for field in ('replaces', 'verify', 'checked'):
            if not recipe.get(field):
                problems.append(f'{rid}: {field} is empty')
        if not recipe.get('options'):
            problems.append(f'{rid}: no options')
        for option in recipe.get('options', []):
            where = f'{rid}/{option.get("name")}'
            evidence = option.get('evidence')
            if evidence not in EVIDENCE:
                problems.append(f'{where}: bad evidence {evidence!r}')
            elif evidence == 'run':
                if option.get('grade') not in GRADES:
                    problems.append(f'{where}: run evidence needs a measured grade')
                source = str(option.get('source', ''))
                if not source.startswith('validation/'):
                    problems.append(f'{where}: run evidence must name a file in validation/')
                elif repo_root is not None and not (repo_root / source).is_file():
                    problems.append(f'{where}: {source} does not exist')
            else:
                if option.get('grade') is not None:
                    problems.append(f'{where}: an option that was not run cannot have a measured grade')
                if not str(option.get('source', '')).startswith('https://'):
                    problems.append(f'{where}: docs evidence needs a URL')
    return problems


def _key(name: str) -> str:
    return re.sub(r'[-_\s.]+', '-', name.strip().lower())


def find(name: str, catalog: dict[str, Any] | None = None) -> dict[str, Any] | None:
    key = _key(name)
    for recipe in (catalog or load_catalog())['recipes']:
        if key in {_key(n) for n in (recipe['id'], *recipe['aliases'], *recipe['packages'])}:
            return recipe
    return None


def for_finding(finding: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Recipes for one scan finding. A rule that covers several libraries is narrowed by the text that it matched."""
    candidates = [r for r in catalog['recipes'] if finding['id'] in r['scan_rules'] or finding.get('recipe') == r['id']]
    matched = ' '.join(finding.get('matched', [])).lower()
    if len(candidates) > 1 and matched:
        named = [r for r in candidates if any(n.lower() in matched for n in (r['id'], *r['aliases'], *r['packages']))]
        return named or candidates
    return candidates


def declared_dependencies(root: Path) -> list[str]:
    """Direct dependency names from requirements files, pyproject.toml and environment.yml. Files are read as text."""
    names: set[str] = set()
    requirement = re.compile(r'^\s*(?:-\s*)?([A-Za-z0-9][A-Za-z0-9._-]*)')
    for path in sorted(root.glob('requirements*.txt')) + sorted(root.glob('environment*.y*ml')):
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.split('#')[0]
            if path.suffix != '.txt' and not line.lstrip().startswith('-'):
                continue
            found = requirement.match(line)
            if found and not line.lstrip().startswith(('--', '-r', '-e', '-c')):
                names.add(found.group(1).lower())
    pyproject = root / 'pyproject.toml'
    if pyproject.is_file():
        with contextlib.suppress(OSError):
            names.update(_pyproject_dependencies(pyproject.read_text(encoding='utf-8', errors='replace')))
    return sorted(names)


def _pyproject_dependencies(text: str) -> set[str]:
    """Names in [project] dependencies and [project.optional-dependencies]. No tomllib: CUMLX supports Python 3.10."""
    names: set[str] = set()
    section = ''
    pending = False  # inside a dependency array
    for line in text.splitlines():
        stripped = line.strip()
        if not pending and stripped.startswith('['):
            section = stripped
            continue
        if not pending:
            wanted = section == '[project.optional-dependencies]' or (section == '[project]' and re.match(r'dependencies\s*=', stripped))
            if not wanted or '=' not in stripped:
                continue
            stripped = stripped.split('=', 1)[1]
            pending = stripped.lstrip().startswith('[')
            if not pending:
                continue
        names.update(item.lower() for item in re.findall(r"""["']([A-Za-z0-9][A-Za-z0-9._-]*)""", stripped))
        if re.sub(r"""(["']).*?\1""", '', stripped).rstrip().endswith(']'):
            pending = False
    return names


def annotate_scan(report: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Link each finding of a scan to its recipes, and list what has no recipe."""
    catalog = catalog or load_catalog()
    uncovered = []
    for finding in report['findings']:
        ids = [r['id'] for r in for_finding(finding, catalog)]
        finding['recipes'] = ids
        if not ids:
            uncovered.append(finding['id'])
    report['uncovered'] = uncovered
    return report


def _view(recipe: dict[str, Any], verified_only: bool) -> dict[str, Any] | None:
    options = [o for o in recipe['options'] if o['evidence'] == 'run' or not verified_only]
    if verified_only and not options:
        return None
    return {**recipe, 'options': options, 'best_evidence': 'run' if any(o['evidence'] == 'run' for o in recipe['options']) else 'docs'}


def _counts(recipes: list[dict[str, Any]]) -> dict[str, int]:
    return {'recipes': len(recipes), 'with_run_evidence': sum(r['best_evidence'] == 'run' for r in recipes),
            'docs_only': sum(r['best_evidence'] == 'docs' for r in recipes), 'no_substitute': sum(r['outcome'] == 'none' for r in recipes)}


def _read_scan_file(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CumlxError('E_INPUT_INVALID', f'{path} is not a readable JSON file: {exc}', hint='pass a report from `cumlx scan -o`') from None
    if not isinstance(report, dict) or report.get('schema') != 'cumlx.scan.v1':
        raise CumlxError('E_INPUT_INVALID', f'{path} is not a cumlx scan report', hint='pass a report from `cumlx scan -o`')
    return report


def lookup(target: str | None, *, verified_only: bool = False, revision: str | None = None) -> dict[str, Any]:
    """Resolve a name, a directory, an HF id, a scan report or nothing (the whole catalog) into recipes."""
    catalog = load_catalog()
    if not target:
        chosen, kind, report = list(catalog['recipes']), 'catalog', None
    else:
        recipe = find(target, catalog)
        path = Path(target)
        if recipe and not path.exists():
            chosen, kind, report = [recipe], 'name', None
        elif path.is_file():
            chosen, kind, report = [], 'scan_report', _read_scan_file(path)
        elif path.is_dir() or '/' in target:
            from .portability import scan
            chosen, kind, report = [], 'scan', scan(target, revision=revision)
        else:
            known = sorted(r['id'] for r in catalog['recipes'])
            raise CumlxError('E_INPUT_NOT_FOUND', f'no recipe is named {target!r}, and it is not a path',
                             hint=f'run `cumlx recipe` to list all {len(known)} recipes, or pass a directory to scan', details={'known': known})
    uncovered: list[str] = []
    if report is not None:
        annotate_scan(report, catalog)
        uncovered = report['uncovered']
        wanted = [rid for finding in report['findings'] for rid in finding['recipes']]
        chosen = [r for r in catalog['recipes'] if r['id'] in wanted]
    views = [v for v in (_view(r, verified_only) for r in chosen) if v is not None]
    return {'schema': SCHEMA, 'query': target, 'kind': kind, 'counts': _counts(views), 'recipes': views, 'uncovered': uncovered,
            'tier': report['tier'] if report else None, 'recommendation': report['recommendation'] if report else None,
            'limits': 'evidence "run" covers the stated scope only. evidence "docs" was not executed by CUMLX.'}
