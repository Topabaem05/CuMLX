"""Minimal, dependency-free safetensors reader/writer (NumPy only).

Format: 8-byte little-endian header length, a JSON header, then one byte
buffer. The header maps tensor names to dtype, shape and [begin, end) offsets
into that buffer, plus an optional ``__metadata__`` str->str map. Files
written here load with the official ``safetensors`` package and vice versa.

BF16 tensors are read as float32 (exact widening). This module never writes BF16.
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Iterator, Mapping

import numpy as np

from ..errors import CumlxError

_TO_NUMPY = {
    'F64': np.float64, 'F32': np.float32, 'F16': np.float16,
    'I64': np.int64, 'I32': np.int32, 'I16': np.int16, 'I8': np.int8,
    'U64': np.uint64, 'U32': np.uint32, 'U16': np.uint16, 'U8': np.uint8,
    'BOOL': np.bool_,
}
_FROM_NUMPY = {np.dtype(v).str.lstrip('<>|='): k for k, v in _TO_NUMPY.items()}
_ITEMSIZE = {**{k: np.dtype(v).itemsize for k, v in _TO_NUMPY.items()}, 'BF16': 2}
MAX_HEADER_BYTES = 100 * 1024 * 1024


def _dtype_tag(a: np.ndarray) -> str:
    key = a.dtype.str.lstrip('<>|=')
    if key not in _FROM_NUMPY:
        raise CumlxError('E_INPUT_INVALID', f'dtype {a.dtype} cannot be stored in a trace',
                         hint='convert the tensor to float32, float16, an integer type or bool first')
    return _FROM_NUMPY[key]


def save_file(tensors: Mapping[str, np.ndarray], path: str | Path,
              metadata: Mapping[str, str] | None = None) -> None:
    header: dict = {}
    if metadata:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise ValueError('safetensors metadata must map str to str')
        header['__metadata__'] = dict(metadata)
    arrays = []
    offset = 0
    for name in sorted(tensors):
        if name == '__metadata__':
            raise ValueError('reserved tensor name')
        a = np.asarray(tensors[name])
        if not a.flags.c_contiguous:  # np.ascontiguousarray would turn 0-d into 1-d
            a = a.copy(order='C')
        if a.dtype.byteorder == '>':
            a = a.astype(a.dtype.newbyteorder('<'))
        header[name] = {'dtype': _dtype_tag(a), 'shape': list(a.shape),
                        'data_offsets': [offset, offset + a.nbytes]}
        arrays.append(a)
        offset += a.nbytes
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * ((8 - len(raw) % 8) % 8)
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as f:
        f.write(struct.pack('<Q', len(raw)))
        f.write(raw)
        for a in arrays:
            f.write(a.tobytes(order='C'))
    tmp.replace(path)


class SafetensorsReader:
    """Lazy reader. Each tensor is read from disk only when requested."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'safetensors file not found: {self.path}')
        size = self.path.stat().st_size
        with self.path.open('rb') as f:
            head = f.read(8)
            if len(head) != 8:
                raise CumlxError('E_INPUT_INVALID', f'{self.path} is too short to be safetensors')
            (n,) = struct.unpack('<Q', head)
            if n > MAX_HEADER_BYTES or 8 + n > size:
                raise CumlxError('E_INPUT_INVALID', f'{self.path} has an invalid header length')
            try:
                header = json.loads(f.read(n))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CumlxError('E_INPUT_INVALID', f'{self.path} header is not JSON: {exc}') from None
        if not isinstance(header, dict):
            raise CumlxError('E_INPUT_INVALID', f'{self.path} header is not an object')
        self.metadata: dict[str, str] = header.pop('__metadata__', None) or {}
        self._base = 8 + n
        self._entries: dict[str, tuple[str, tuple[int, ...], int, int]] = {}
        buffer_size = size - self._base
        spans = []
        for name, info in header.items():
            try:
                tag, shape, (begin, end) = info['dtype'], tuple(int(s) for s in info['shape']), info['data_offsets']
            except (KeyError, TypeError, ValueError):
                raise CumlxError('E_INPUT_INVALID', f'{self.path}: malformed entry {name!r}') from None
            if tag not in _ITEMSIZE or any(s < 0 for s in shape):
                raise CumlxError('E_INPUT_INVALID', f'{self.path}: unsupported dtype/shape for {name!r}')
            expected = int(np.prod(shape, dtype=np.int64)) * _ITEMSIZE[tag]
            if not (0 <= begin <= end <= buffer_size) or end - begin != expected:
                raise CumlxError('E_INPUT_INVALID', f'{self.path}: offsets of {name!r} do not match its shape')
            self._entries[name] = (tag, shape, begin, end)
            spans.append((begin, end))
        spans.sort()
        for (_, e0), (b1, _) in zip(spans, spans[1:]):
            if b1 < e0:
                raise CumlxError('E_INPUT_INVALID', f'{self.path}: tensor byte ranges overlap')

    def keys(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def shape(self, name: str) -> tuple[int, ...]:
        return self._entries[name][1]

    def dtype_tag(self, name: str) -> str:
        return self._entries[name][0]

    def get(self, name: str) -> np.ndarray:
        if name not in self._entries:
            raise CumlxError('E_INPUT_INVALID', f'{self.path}: tensor {name!r} is missing')
        tag, shape, begin, end = self._entries[name]
        if end == begin:
            dt = np.float32 if tag == 'BF16' else _TO_NUMPY[tag]
            return np.zeros(shape, dt)
        with self.path.open('rb') as f:
            f.seek(self._base + begin)
            raw = f.read(end - begin)
        if len(raw) != end - begin:
            raise CumlxError('E_INPUT_INVALID', f'{self.path}: truncated data for {name!r}')
        if tag == 'BF16':
            bits = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
            return bits.view(np.float32).reshape(shape).copy()
        return np.frombuffer(raw, dtype=np.dtype(_TO_NUMPY[tag]).newbyteorder('<')).reshape(shape).copy()


def load_file(path: str | Path) -> dict[str, np.ndarray]:
    reader = SafetensorsReader(path)
    return {k: reader.get(k) for k in reader.keys()}
