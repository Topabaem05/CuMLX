"""Text-level facts about a CUDA or Metal kernel source. Nothing here compiles or runs anything."""
from __future__ import annotations

import re
from typing import Any

IDENT = re.compile(r'[A-Za-z_]\w*')
# An unsuffixed floating literal is a double in CUDA C++, and a float in Metal.
FP64_LITERAL = re.compile(r'(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+)(?:[eE][+-]?\d+)?(?![\w.])')
ATOMICS = frozenset({'atomicAdd', 'atomicSub', 'atomicExch', 'atomicMin', 'atomicMax', 'atomicInc', 'atomicDec',
                     'atomicCAS', 'atomicAnd', 'atomicOr', 'atomicXor', 'atomic_fetch_add_explicit', 'atomic_store_explicit'})
WARP_LEVEL = frozenset({'__match_any_sync', '__match_all_sync', '__activemask', '__ballot_sync', '__all_sync', '__any_sync',
                        '__shfl_sync', '__shfl_up_sync', '__shfl_down_sync', '__shfl_xor_sync', '__syncwarp', '__shfl',
                        '__shfl_up', '__shfl_down', '__shfl_xor', '__ballot', '__all', '__any'})


def mask_source(text: str, *, keep_strings: bool = False) -> str:
    """Blank comments and string contents, keeping every offset and newline, so that a scan cannot match inside them."""
    out = list(text)
    i, n = 0, len(text)

    def blank(begin: int, end: int) -> None:
        for j in range(begin, end):
            if out[j] != '\n':
                out[j] = ' '

    while i < n:
        c = text[i]
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            j = i
            while j < n and text[j] != '\n':
                j += 2 if text[j] == '\\' and j + 1 < n else 1
            blank(i, min(j, n))
            i = j
        elif c == '/' and i + 1 < n and text[i + 1] == '*':
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            blank(i, end)
            i = end
        elif c in '"\'':
            if c == "'" and i > 0 and text[i - 1].isalnum():  # a digit separator such as 1'000
                i += 1
                continue
            j = i + 1
            while j < n and text[j] != c and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            if not keep_strings:
                blank(i + 1, min(j, n))
            i = j + 1
        else:
            i += 1
    return ''.join(out)


def strip_comments(text: str) -> str:
    """Remove comments and keep strings. Offsets and line numbers do not change."""
    return mask_source(text, keep_strings=True)


def line_of(text: str, offset: int) -> int:
    return text.count('\n', 0, offset) + 1


def scan_features(text: str) -> dict[str, Any]:
    """What a kernel source does that matters for CUDA against Metal. A text scan: it can be fooled by a macro."""
    masked = mask_source(text)
    tokens = set(IDENT.findall(masked))
    return {'fp64': 'double' in tokens,
            'fp64_literals': len(FP64_LITERAL.findall(masked)),
            'atomics': bool(tokens & ATOMICS),
            'warp_level': sorted(tokens & WARP_LEVEL)}


def is_complete_cuda(text: str) -> bool:
    """A complete source has a ``__global__`` function. Otherwise it is a function body, as ``cuda_kernel`` takes."""
    return bool(re.search(r'\b__global__\b', mask_source(text)))


def is_complete_metal(text: str) -> bool:
    """A complete library has a ``kernel`` function. Otherwise it is a function body, as ``metal_kernel`` takes."""
    return bool(re.search(r'(?:\[\[\s*kernel\s*\]\]|\bkernel)\s+void\b', mask_source(text)))
