"""Op-level golden format ``cumlx.opgolden.v1``: what one kernel computed on one set of inputs.

The specification is in docs/TRACE-FORMAT.md. This module reads and writes the format and has no dependency on a GPU.
Only ``golden.kind == 'cuda'`` is evidence about CUDA. ``'fixture'`` exists for tests and never earns a proof grade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..errors import CumlxError
from .safetensors_io import load_file, save_file

SCHEMA = 'cumlx.opgolden.v1'
CASES_SCHEMA = 'cumlx.kernel-cases.v1'
GOLDEN_KINDS = ('cuda', 'fixture')
MANIFEST = 'manifest.json'
DTYPES: dict[str, Any] = {
    'float32': np.float32, 'float16': np.float16, 'float64': np.float64, 'uint8': np.uint8, 'int8': np.int8,
    'uint16': np.uint16, 'int16': np.int16, 'uint32': np.uint32, 'int32': np.int32, 'uint64': np.uint64,
    'int64': np.int64}
_NAME = re.compile(r'^[A-Za-z_]\w*$')
_ENTRY = re.compile(r'^[A-Za-z_]\w*(<[\w\s,.]*>)?$')
_ID = re.compile(r'^[\w.+-]+$')


def _invalid(where: str, message: str) -> CumlxError:
    return CumlxError('E_INPUT_INVALID', f'{where}: {message}')


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'dtype': self.dtype, 'shape': list(self.shape)}


@dataclass
class KernelCase:
    """One launch of one kernel. ``grid`` and ``threadgroup`` count logical threads, as MLX does."""
    id: str
    kernel: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    constants: dict[str, int | bool] = field(default_factory=dict)
    grid: tuple[int, int, int] = (1, 1, 1)
    threadgroup: tuple[int, int, int] = (1, 1, 1)
    entry: str | None = None

    @property
    def blocks(self) -> tuple[int, int, int]:
        return (self.grid[0] // self.threadgroup[0], self.grid[1] // self.threadgroup[1],
                self.grid[2] // self.threadgroup[2])

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            'id': self.id, 'kernel': self.kernel, 'inputs': [t.to_dict() for t in self.inputs],
            'outputs': [t.to_dict() for t in self.outputs], 'constants': dict(self.constants),
            'grid': list(self.grid), 'threadgroup': list(self.threadgroup)}
        if self.entry:
            out['entry'] = self.entry
        return out


def _tensor_spec(raw: Any, where: str) -> TensorSpec:
    if not isinstance(raw, dict):
        raise _invalid(where, 'a tensor must be an object')
    name, dtype, shape = raw.get('name'), raw.get('dtype'), raw.get('shape')
    if not isinstance(name, str) or not _NAME.match(name):
        raise _invalid(where, f'bad tensor name {name!r}')
    if dtype not in DTYPES:
        raise _invalid(where, f'{name}: unsupported dtype {dtype!r}; supported: {sorted(DTYPES)}')
    if not isinstance(shape, list) or any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in shape):
        raise _invalid(where, f'{name}: shape must be a list of positive integers, got {shape!r}')
    return TensorSpec(name, str(dtype), tuple(shape))


def _triple(raw: Any, where: str, field_name: str) -> tuple[int, int, int]:
    if (not isinstance(raw, list) or len(raw) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in raw)):
        raise _invalid(where, f'{field_name} must be three positive integers, got {raw!r}')
    return raw[0], raw[1], raw[2]


def parse_case(raw: Any) -> KernelCase:
    if not isinstance(raw, dict):
        raise _invalid('case', 'a case must be an object')
    case_id = raw.get('id')
    if not isinstance(case_id, str) or not _ID.match(case_id):
        raise _invalid('case', f'bad case id {case_id!r}')
    where = f'case {case_id}'
    kernel = raw.get('kernel')
    if not isinstance(kernel, str) or not _NAME.match(kernel):
        raise _invalid(where, f'bad kernel name {kernel!r}')
    entry = raw.get('entry')
    if entry is not None and (not isinstance(entry, str) or not _ENTRY.match(entry)):
        raise _invalid(where, f'bad entry {entry!r}')
    inputs = [_tensor_spec(t, where) for t in raw.get('inputs') or []]
    outputs = [_tensor_spec(t, where) for t in raw.get('outputs') or []]
    if not outputs:
        raise _invalid(where, 'a case needs at least one output')
    names = [t.name for t in inputs + outputs]
    if len(set(names)) != len(names):
        raise _invalid(where, f'tensor names repeat: {names}')
    constants = raw.get('constants') or {}
    if not isinstance(constants, dict):
        raise _invalid(where, 'constants must be an object')
    for key, value in constants.items():
        if not _NAME.match(str(key)) or not isinstance(value, (int, bool)):
            raise _invalid(where, f'constant {key!r} must be an integer or a boolean, got {value!r}')
        if key in names:
            raise _invalid(where, f'{key} is both a constant and a tensor')
    grid = _triple(raw.get('grid'), where, 'grid')
    threadgroup = _triple(raw.get('threadgroup'), where, 'threadgroup')
    if any(g % t for g, t in zip(grid, threadgroup)):
        raise _invalid(where, f'grid {grid} is not a whole number of threadgroups {threadgroup}')
    return KernelCase(case_id, kernel, inputs, outputs, dict(constants), grid, threadgroup, entry)


def as_dtype(array: np.ndarray, spec: TensorSpec, where: str) -> np.ndarray:
    a = np.asarray(array)
    if a.dtype != np.dtype(DTYPES[spec.dtype]) or a.shape != spec.shape:
        raise _invalid(where, f'{spec.name} is {a.dtype}{list(a.shape)}, but the case says {spec.dtype}{list(spec.shape)}')
    return a


def read_cases(path: str | Path) -> list[tuple[KernelCase, dict[str, np.ndarray]]]:
    """Read a ``cumlx.kernel-cases.v1`` file: the launches to record, and their inputs."""
    p = Path(path)
    if not p.is_file():
        raise CumlxError('E_INPUT_NOT_FOUND', f'cases file not found: {p}',
                         hint='scripts/make_kernel_cases.py writes the cases of the FP4 kernels')
    try:
        doc = json.loads(p.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CumlxError('E_INPUT_INVALID', f'{p} is not JSON: {exc}') from None
    if not isinstance(doc, dict) or doc.get('schema') != CASES_SCHEMA:
        raise _invalid(str(p), f'schema must be {CASES_SCHEMA}')
    if not isinstance(doc.get('cases'), list) or not doc['cases']:
        raise _invalid(str(p), 'cases must be a non-empty list')
    out: list[tuple[KernelCase, dict[str, np.ndarray]]] = []
    files: dict[str, dict[str, np.ndarray]] = {}
    for raw in doc['cases']:
        case = parse_case(raw)
        sources = {t['name']: t for t in raw['inputs']} if case.inputs else {}
        arrays: dict[str, np.ndarray] = {}
        for spec in case.inputs:
            source = sources[spec.name]
            if 'values' in source:
                arrays[spec.name] = np.asarray(source['values'], dtype=DTYPES[spec.dtype]).reshape(spec.shape)
            elif 'file' in source:
                file_name = str(source['file'])
                if Path(file_name).is_absolute() or '..' in Path(file_name).parts:
                    raise _invalid(f'case {case.id}', f'{spec.name}: file must stay beside the cases file')
                if file_name not in files:
                    files[file_name] = load_file(p.parent / file_name)
                tensor = str(source.get('tensor', spec.name))
                if tensor not in files[file_name]:
                    raise _invalid(f'case {case.id}', f'{file_name} has no tensor {tensor!r}')
                arrays[spec.name] = as_dtype(files[file_name][tensor], spec, f'case {case.id}')
            else:
                raise _invalid(f'case {case.id}', f'input {spec.name} needs "values" or "file"')
        out.append((case, arrays))
    ids = [c.id for c, _ in out]
    if len(set(ids)) != len(ids):
        raise _invalid(str(p), 'case ids repeat')
    return out


def write_golden(out_dir: Path, *, kind: str, source: dict[str, Any], kernels: dict[str, Any],
                 floating_point: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Write a golden. Each item of ``cases`` holds ``case``, ``inputs``, ``outputs`` and any extra manifest fields."""
    from .. import __version__
    if kind not in GOLDEN_KINDS:
        raise ValueError(f'unknown golden kind {kind!r}')
    (out_dir / 'cases').mkdir(parents=True, exist_ok=True)
    entries = []
    for i, item in enumerate(cases):
        case: KernelCase = item['case']
        rel = f'cases/c{i:04d}.safetensors'
        tensors = {f'in.{k}': v for k, v in item['inputs'].items()}
        tensors.update({f'out.{k}': v for k, v in item['outputs'].items()})
        save_file(tensors, out_dir / rel)
        extra = {k: v for k, v in item.items() if k not in ('case', 'inputs', 'outputs')}
        entries.append({**case.to_dict(), 'file': rel, 'sha256': sha256_file(out_dir / rel), **extra})
    manifest = {
        'schema': SCHEMA, 'cumlx_version': __version__,
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'golden': {'kind': kind}, 'source': source, 'floating_point': floating_point,
        'kernels': kernels, 'cases': entries}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


class OpGolden:
    """Reader. Rejects an unknown schema or kind, a file outside the golden, and a checksum mismatch."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        manifest_path = self.path / MANIFEST
        if not manifest_path.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'no golden at {self.path}: {MANIFEST} is missing',
                             hint='record one on an NVIDIA machine with `cumlx kernel capture`')
        try:
            manifest = json.loads(manifest_path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CumlxError('E_INPUT_INVALID', f'{manifest_path} is not JSON: {exc}') from None
        if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
            found = manifest.get('schema') if isinstance(manifest, dict) else None
            raise _invalid(str(manifest_path), f'schema is {found!r}, expected {SCHEMA}')
        self.manifest: dict[str, Any] = manifest
        self.kind = (manifest.get('golden') or {}).get('kind')
        if self.kind not in GOLDEN_KINDS:
            raise _invalid(str(manifest_path), f'unknown golden.kind {self.kind!r}')
        self.source: dict[str, Any] = manifest.get('source') or {}
        if self.kind == 'cuda':
            for name in ('compiler', 'gpu', 'driver'):
                if not self.source.get(name):
                    raise _invalid(str(manifest_path), f'a CUDA golden must record source.{name}')
        self.floating_point: dict[str, Any] = manifest.get('floating_point') or {}
        self.kernels: dict[str, Any] = manifest.get('kernels') or {}
        raw_cases = manifest.get('cases')
        if not isinstance(raw_cases, list) or not raw_cases:
            raise _invalid(str(manifest_path), 'a golden needs at least one case')
        self.cases = [parse_case(raw) for raw in raw_cases]
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise _invalid(str(manifest_path), 'case ids repeat')
        self._raw_by_id = {c['id']: c for c in raw_cases}

    def info(self, case_id: str) -> dict[str, Any]:
        return self._raw_by_id[case_id]

    def load(self, case: KernelCase) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        raw = self._raw_by_id[case.id]
        rel = raw.get('file')
        if not isinstance(rel, str) or Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise _invalid(f'case {case.id}', f'bad file {rel!r}')
        path = self.path / rel
        if not path.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'golden file is missing: {path}')
        if sha256_file(path) != raw.get('sha256'):
            raise _invalid(f'case {case.id}', f'{path} does not match its SHA-256; the golden was changed or is damaged')
        tensors = load_file(path)
        inputs: dict[str, np.ndarray] = {}
        outputs: dict[str, np.ndarray] = {}
        for group, prefix, specs in ((inputs, 'in.', case.inputs), (outputs, 'out.', case.outputs)):
            for spec in specs:
                name = prefix + spec.name
                if name not in tensors:
                    raise _invalid(f'case {case.id}', f'{path} has no tensor {name!r}')
                group[spec.name] = as_dtype(tensors[name], spec, f'case {case.id}')
        return inputs, outputs
