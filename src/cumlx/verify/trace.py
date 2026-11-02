"""Golden trace container: ``manifest.json`` + one safetensors file per sample.

Layout::

    <trace>/manifest.json
    <trace>/samples/s0000.safetensors   token_ids, r00000.in0, r00000.out, ...

Records are kept in hook completion order: a child module completes before
its parent. ``key`` = ``<module name>#<call index>`` distinguishes repeated
calls of one module. The root module has the empty name ``""``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ..errors import CumlxError
from .safetensors_io import SafetensorsReader, save_file

TRACE_SCHEMA = 'cumlx.trace.v1'
MANIFEST = 'manifest.json'
GOLDEN_KINDS = ('reference', 'quantization-aware')


@dataclass
class TensorRef:
    tensor: str
    shape: list[int]
    dtype: str
    source: str  # where it came from: 'arg0', 'kw:hidden_states', 'tensor', 'tuple[0]', 'ModelOutput[0]'


@dataclass
class ModuleRecord:
    key: str
    name: str
    call: int
    order: int
    type: str
    has_params: bool
    replayable: bool
    replay_note: str | None = None
    inputs: list[TensorRef] = field(default_factory=list)
    output: TensorRef | None = None
    # 'single' = the only input is one tensor. 'hidden' = a hidden state plus causal context
    # (mask, position embeddings) that the port recomputes. None = not replayable.
    replay_kind: str | None = None

    @staticmethod
    def from_dict(d: dict) -> 'ModuleRecord':
        d = dict(d)
        d['inputs'] = [TensorRef(**t) for t in d.get('inputs', [])]
        d['output'] = TensorRef(**d['output']) if d.get('output') else None
        if d.get('replay_kind') is None and d.get('replayable'):  # traces written before replay_kind existed
            d['replay_kind'] = 'single'
        return ModuleRecord(**d)


@dataclass
class Sample:
    id: str
    file: str
    sha256: str
    num_tokens: int
    records: list[ModuleRecord]

    @staticmethod
    def from_dict(d: dict) -> 'Sample':
        d = dict(d)
        d['records'] = [ModuleRecord.from_dict(r) for r in d['records']]
        return Sample(**d)


@dataclass
class Manifest:
    source: dict[str, Any]
    golden: dict[str, Any]
    samples: list[Sample]
    schema: str = TRACE_SCHEMA
    cumlx_version: str = ''
    created_utc: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'Manifest':
        if not isinstance(d, dict) or d.get('schema') != TRACE_SCHEMA:
            raise CumlxError('E_INPUT_INVALID', f'trace manifest schema is not {TRACE_SCHEMA}',
                             details={'found': d.get('schema') if isinstance(d, dict) else None},
                             hint='re-capture the trace with this CUMLX version')
        try:
            m = Manifest(source=d['source'], golden=d['golden'],
                         samples=[Sample.from_dict(s) for s in d['samples']],
                         schema=d['schema'], cumlx_version=d.get('cumlx_version', ''),
                         created_utc=d.get('created_utc', ''))
        except (KeyError, TypeError) as exc:
            raise CumlxError('E_INPUT_INVALID', f'trace manifest is malformed: {exc}') from None
        if m.golden.get('kind') not in GOLDEN_KINDS:
            raise CumlxError('E_INPUT_INVALID', 'trace manifest has an unknown golden.kind')
        return m


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_output_dir(path: str | Path, *, overwrite: bool, marker: str = MANIFEST) -> Path:
    """Validate that ``path`` may be (re)created. ``marker`` proves a directory is ours to replace."""
    out = Path(path)
    if out.exists():
        if not overwrite:
            raise CumlxError('E_OUTPUT_EXISTS', f'output path already exists: {out}',
                             hint='choose a new -o/--output path or pass --overwrite',
                             details={'path': str(out)})
        if not out.is_dir():
            raise CumlxError('E_OUTPUT_EXISTS', f'{out} exists and is not a directory')
        if any(out.iterdir()) and not (out / marker).exists():
            raise CumlxError('E_OUTPUT_EXISTS', f'{out} is a non-empty directory not created by this command',
                             hint=f'refusing to delete it (no {marker}); choose an empty or new directory',
                             details={'path': str(out), 'marker': marker})
    return out


def prepare_output_dir(path: str | Path, *, overwrite: bool, marker: str = MANIFEST) -> Path:
    out = check_output_dir(path, overwrite=overwrite, marker=marker)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    return out


class TraceWriter:
    def __init__(self, out_dir: Path, *, source: dict, golden: dict, max_bytes: int):
        from .. import __version__
        self.out_dir = out_dir
        (out_dir / 'samples').mkdir(exist_ok=True)
        self.manifest = Manifest(source=source, golden=golden, samples=[], cumlx_version=__version__,
                                 created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
        self.max_bytes = max_bytes
        self.total_bytes = 0

    def add_sample(self, tensors: dict[str, np.ndarray], records: list[ModuleRecord], num_tokens: int) -> Sample:
        nbytes = sum(int(a.nbytes) for a in tensors.values())
        if self.total_bytes + nbytes > self.max_bytes:
            raise CumlxError('E_SIZE_LIMIT', f'trace would exceed --max-bytes={self.max_bytes}',
                             hint='capture fewer modules (--include/--exclude), shorter prompts, '
                                  '--store-dtype float16, or raise --max-bytes',
                             details={'needed_bytes': self.total_bytes + nbytes})
        self.total_bytes += nbytes
        sid = f's{len(self.manifest.samples):04d}'
        rel = f'samples/{sid}.safetensors'
        path = self.out_dir / rel
        save_file(tensors, path, metadata={'schema': TRACE_SCHEMA, 'sample': sid})
        sample = Sample(id=sid, file=rel, sha256=_file_sha256(path), num_tokens=num_tokens, records=records)
        self.manifest.samples.append(sample)
        return sample

    def finish(self) -> Path:
        path = self.out_dir / MANIFEST
        path.write_text(json.dumps(self.manifest.to_dict(), indent=1, ensure_ascii=False) + '\n')
        return path


class Trace:
    def __init__(self, path: str | Path, *, verify: bool = True):
        self.dir = Path(path)
        mpath = self.dir / MANIFEST
        if not mpath.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'no {MANIFEST} in {self.dir}',
                             hint='pass the directory written by `cumlx capture -o DIR`')
        try:
            raw = json.loads(mpath.read_text())
        except json.JSONDecodeError as exc:
            raise CumlxError('E_INPUT_INVALID', f'{mpath} is not valid JSON: {exc}') from None
        self.manifest = Manifest.from_dict(raw)
        self._readers: dict[str, SafetensorsReader] = {}
        for s in self.manifest.samples:
            p = self.dir / s.file
            if not p.is_file():
                raise CumlxError('E_INPUT_NOT_FOUND', f'trace sample file missing: {p}')
            if verify and _file_sha256(p) != s.sha256:
                raise CumlxError('E_INPUT_INVALID', f'checksum mismatch for {p}; the trace was modified or truncated')
            self._readers[s.id] = SafetensorsReader(p)

    def tensor(self, sample: Sample, name: str) -> np.ndarray:
        return self._readers[sample.id].get(name)

    def token_ids(self, sample: Sample) -> np.ndarray:
        return self.tensor(sample, 'token_ids')
