"""bitsandbytes.optim under cumlx run: every optimizer is refused, because this adapter covers LLM.int8 inference only.

A program that imports an 8-bit optimizer gets NotImplementedError at the import or at the attribute, not a plain torch optimizer in
its place. A name that starts with an underscore is an ordinary missing attribute, so that hasattr and import machinery work.
"""
from __future__ import annotations

from ._matmul import _refuse


def __getattr__(name: str) -> object:
    if name.startswith('_'):
        raise AttributeError(name)
    raise _refuse(f'bitsandbytes.optim.{name}: the optimizers of bitsandbytes')
