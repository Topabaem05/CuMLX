"""PyTorch -> MLX module name mapping.

Default: identical dotted paths (mlx-lm keeps HF paths such as
``model.layers.0.self_attn.q_proj``). A JSON rules file can rewrite names::

    {"rules": [{"match": "^transformer\\.h\\.(\\d+)", "replace": "model.layers.\\1"}],
     "ignore": ["rotary_emb$"]}

Rules are applied in order with ``re.sub``. ``ignore`` patterns mark PyTorch
modules as intentionally not compared (reported as ``ignored``, not as passed).
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..errors import CumlxError


class NameMapper:
    def __init__(self, rules: list[tuple[re.Pattern, str]] | None = None, ignore: list[re.Pattern] | None = None,
                 source: str = 'identity'):
        self.rules = rules or []
        self.ignore = ignore or []
        self.source = source

    @staticmethod
    def from_file(path: str | None) -> 'NameMapper':
        if not path:
            return NameMapper()
        p = Path(path)
        if not p.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'mapping file not found: {p}')
        try:
            raw = json.loads(p.read_text())
            rules = [(re.compile(r['match']), r['replace']) for r in raw.get('rules', [])]
            ignore = [re.compile(x) for x in raw.get('ignore', [])]
        except (json.JSONDecodeError, KeyError, TypeError, re.error) as exc:
            raise CumlxError('E_INPUT_INVALID', f'invalid mapping file {p}: {exc}',
                             hint='expected {"rules": [{"match": REGEX, "replace": STR}], "ignore": [REGEX]}') from None
        return NameMapper(rules, ignore, source=str(p))

    def is_ignored(self, torch_name: str) -> bool:
        return any(p.search(torch_name) for p in self.ignore)

    def map(self, torch_name: str) -> str:
        name = torch_name
        for pattern, repl in self.rules:
            name = pattern.sub(repl, name)
        return name
