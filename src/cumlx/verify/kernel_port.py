"""`cumlx kernel port`: the pre-translation layer, and the adapter to a CUDA-to-Metal translator.

M0 showed where a translator stops on real kernels: standard headers, templates with no instance, host code in the same
file, and PTX built-ins. This layer handles those four and refuses, with a reason, what has no Metal equivalent. It is a
source-to-source step, not a compiler front end. It works on tokens and declarations, so a macro can hide things from it.

It does not translate to Metal. A translator does, through ``TranslatorAdapter``. A missing translator is an environment
error, and a translated kernel is unverified until `cumlx kernel verify` compares it with a CUDA golden.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
import importlib
import importlib.util
from pathlib import Path
import re
from typing import Any, Protocol

from ..errors import CumlxError
from .kernel_capture import wrap_body
from .kernel_source import (
    FP64_LITERAL,
    IDENT,
    WARP_LEVEL,
    line_of,
    mask_source,
    strip_comments,
)
from .opgolden import KernelCase, read_cases, sha256_text

PORT_SCHEMA = 'cumlx.kernel-port.v1'

# ---------------------------------------------------------------------------- standard headers
# Header -> what stays in the output. None: nothing stays, because the prelude or the translator covers it.
STANDARD_HEADERS: dict[str, str | None] = {
    'math.h': '<math.h>', 'cmath': '<math.h>', 'stdint.h': '<stdint.h>', 'cstdint': '<stdint.h>',
    'stddef.h': None, 'cstddef': None, 'float.h': None, 'cfloat': None, 'limits.h': None, 'climits': None, 'limits': None,
    'stdbool.h': None, 'algorithm': None, 'type_traits': None, 'cuda_runtime.h': None, 'cuda_runtime_api.h': None,
    'cuda_device_runtime_api.h': None, 'cuda_fp16.h': None, 'cuda_bf16.h': None, 'cuda.h': None}
# Host-only libraries. Each is dropped. A device function that needs one still ends in an error from the translator.
HOST_HEADERS = frozenset({
    'iostream', 'ostream', 'istream', 'fstream', 'sstream', 'iomanip', 'vector', 'string', 'map', 'unordered_map', 'set',
    'unordered_set', 'list', 'deque', 'queue', 'stack', 'memory', 'functional', 'tuple', 'array', 'utility', 'numeric',
    'random', 'chrono', 'thread', 'mutex', 'atomic', 'future', 'exception', 'stdexcept', 'cassert', 'assert.h', 'cstdio',
    'stdio.h', 'cstdlib', 'stdlib.h', 'cstring', 'string.h', 'ctime', 'time.h', 'unistd.h', 'Python.h', 'nvml.h'})
HOST_HEADER_PREFIXES = ('ATen/', 'torch/', 'c10/', 'pybind11/', 'opencv', 'GL/', 'GLES', 'EGL/', 'boost/', 'Eigen/', 'nanobind/')
# Macro and type values for what the dropped headers define. Emitted only when the kernel uses the name.
HEADER_DEFINES = {
    'FLT_MAX': '3.40282347e+38f', 'FLT_MIN': '1.17549435e-38f', 'FLT_EPSILON': '1.19209290e-07f', 'INT_MAX': '2147483647',
    'INT_MIN': '(-2147483647 - 1)', 'UINT_MAX': '4294967295u', 'M_PI': '3.14159265358979323846f', 'M_PI_2': '1.57079632679489661923f',
    'M_E': '2.71828182845904523536f'}
NUMERIC_LIMITS = {('float', 'max'): 'FLT_MAX', ('float', 'min'): 'FLT_MIN', ('float', 'lowest'): '(-FLT_MAX)',
                  ('float', 'epsilon'): 'FLT_EPSILON', ('float', 'infinity'): 'INFINITY', ('float', 'quiet_NaN'): 'NAN',
                  ('int', 'max'): 'INT_MAX', ('int', 'min'): 'INT_MIN', ('unsigned int', 'max'): 'UINT_MAX'}
STD_ALLOWED = frozenset({'min', 'max', 'abs', 'fabs', 'sqrt', 'exp', 'log', 'pow', 'sin', 'cos', 'tan', 'floor', 'ceil', 'fmod',
                         'isnan', 'isinf', 'isfinite', 'signbit', 'copysign', 'ldexp', 'fma', 'round', 'trunc', 'tanh', 'atan2',
                         'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t', 'size_t'})

# ---------------------------------------------------------------------------- PTX built-ins
_ROUNDING = ('The PTX rounding modifier has no Metal equivalent: Metal cannot select a rounding mode, so this value could not '
             'be reproduced. Rewrite the kernel to use round-to-nearest if its result does not depend on the direction.')
_WARP = ('Warp-level behaviour has no Metal equivalent. Metal SIMD groups have no guaranteed width of 32 and no mask of '
         'active threads, so the result would depend on the group size. Rewrite the kernel with threadgroup memory.')
_NOLIB = 'The Metal standard library has no such function. Provide your own implementation and verify it.'
_TEX = 'CUDA texture and surface objects have no direct mapping to Metal texture arguments. Pass the data in a buffer.'
_FENCE = 'A memory fence with no barrier has no Metal equivalent: threadgroup_barrier also synchronizes the threads.'


@dataclass(frozen=True)
class BuiltIn:
    """``direct``: the translator must map it to ``metal``. ``rewrite``: this layer replaces it. ``refuse``: no equivalent."""
    status: str
    metal: str = ''
    helper: str = ''
    reason: str = ''


def _helper(signature_name: str, ret: str, params: str, body: str) -> str:
    return f'__device__ __forceinline__ {ret} {signature_name}({params}) {{ {body} }}\n'


_HELPERS: dict[str, str] = {
    'cumlx_fadd_rn': _helper('cumlx_fadd_rn', 'float', 'float a, float b', 'return a + b;'),
    'cumlx_fsub_rn': _helper('cumlx_fsub_rn', 'float', 'float a, float b', 'return a - b;'),
    'cumlx_fmul_rn': _helper('cumlx_fmul_rn', 'float', 'float a, float b', 'return a * b;'),
    'cumlx_fdiv_rn': _helper('cumlx_fdiv_rn', 'float', 'float a, float b', 'return a / b;'),
    'cumlx_fma_rn': _helper('cumlx_fma_rn', 'float', 'float a, float b, float c', 'return fmaf(a, b, c);'),
    'cumlx_frcp_rn': _helper('cumlx_frcp_rn', 'float', 'float x', 'return 1.0f / x;'),
    'cumlx_fsqrt_rn': _helper('cumlx_fsqrt_rn', 'float', 'float x', 'return sqrtf(x);'),
    'cumlx_rsqrtf': _helper('cumlx_rsqrtf', 'float', 'float x', 'return 1.0f / sqrtf(x);'),
    'cumlx_saturatef': _helper('cumlx_saturatef', 'float', 'float x', 'return x > 0.0f ? (x < 1.0f ? x : 1.0f) : 0.0f;'),
    'cumlx_fdividef': _helper('cumlx_fdividef', 'float', 'float a, float b', 'return a / b;'),
    'cumlx_exp10f': _helper('cumlx_exp10f', 'float', 'float x', 'return powf(10.0f, x);'),
    'cumlx_half2float': _helper('cumlx_half2float', 'float', '__half x', 'return (float)x;'),
    'cumlx_float2half': _helper('cumlx_float2half', '__half', 'float x', 'return (__half)x;'),
    'cumlx_clz': _helper('cumlx_clz', 'int', 'unsigned int x',
                         'int n = 0; for (unsigned int m = 0x80000000u; m != 0u && (x & m) == 0u; m >>= 1u) ++n; return n;'),
    'cumlx_popc': _helper('cumlx_popc', 'int', 'unsigned int x', 'int n = 0; for (; x != 0u; x &= x - 1u) ++n; return n;'),
    'cumlx_brev': _helper('cumlx_brev', 'unsigned int', 'unsigned int x',
                          'unsigned int r = 0u; for (int i = 0; i < 32; ++i) { r = (r << 1u) | (x & 1u); x >>= 1u; } return r;'),
    'cumlx_ffs': _helper('cumlx_ffs', 'int', 'int v',
                         'unsigned int x = (unsigned int)v; if (x == 0u) return 0; int n = 1; while ((x & 1u) == 0u) { x >>= 1u; ++n; } return n;'),
    'cumlx_hadd': _helper('cumlx_hadd', 'int', 'int a, int b', 'return (a & b) + ((a ^ b) >> 1);'),
    'cumlx_rhadd': _helper('cumlx_rhadd', 'int', 'int a, int b', 'return (a | b) - ((a ^ b) >> 1);'),
    # 32-bit limbs: Metal has no `long long`, and these two helpers must not depend on a 64-bit integer type.
    'cumlx_umulhi': _helper('cumlx_umulhi', 'unsigned int', 'unsigned int a, unsigned int b',
                            'unsigned int a0 = a & 0xffffu, a1 = a >> 16u, b0 = b & 0xffffu, b1 = b >> 16u; '
                            'unsigned int t = a1 * b0 + ((a0 * b0) >> 16u); unsigned int w1 = (t & 0xffffu) + a0 * b1; '
                            'return a1 * b1 + (t >> 16u) + (w1 >> 16u);'),
    'cumlx_mulhi': _helper('cumlx_mulhi', 'int', 'int a, int b',
                           'unsigned int r = cumlx_umulhi((unsigned int)a, (unsigned int)b); '
                           'if (a < 0) r -= (unsigned int)b; if (b < 0) r -= (unsigned int)a; return (int)r;'),
}
_HELPER_NEEDS = {'cumlx_mulhi': ('cumlx_umulhi',)}


def _build_builtins() -> dict[str, BuiltIn]:
    table: dict[str, BuiltIn] = {
        'threadIdx': BuiltIn('direct', 'thread_position_in_threadgroup'), 'blockIdx': BuiltIn('direct', 'threadgroup_position_in_grid'),
        'blockDim': BuiltIn('direct', 'threads_per_threadgroup'), 'gridDim': BuiltIn('direct', 'threadgroups_per_grid'),
        '__syncthreads': BuiltIn('direct', 'threadgroup_barrier(mem_flags::mem_threadgroup)')}
    for name, metal in {'atomicAdd': 'atomic_fetch_add_explicit', 'atomicSub': 'atomic_fetch_sub_explicit',
                        'atomicExch': 'atomic_exchange_explicit', 'atomicMin': 'atomic_fetch_min_explicit',
                        'atomicMax': 'atomic_fetch_max_explicit', 'atomicAnd': 'atomic_fetch_and_explicit',
                        'atomicOr': 'atomic_fetch_or_explicit', 'atomicXor': 'atomic_fetch_xor_explicit',
                        'atomicCAS': 'atomic_compare_exchange_weak_explicit'}.items():
        table[name] = BuiltIn('direct', metal)
    for name in ('atomicInc', 'atomicDec'):
        table[name] = BuiltIn('refuse', reason=f'{name} wraps at a limit that the caller passes, and Metal has no such atomic. Use atomicAdd or atomicCAS.')
    for name, helper in {'__fadd_rn': 'cumlx_fadd_rn', '__fsub_rn': 'cumlx_fsub_rn', '__fmul_rn': 'cumlx_fmul_rn', '__fdiv_rn': 'cumlx_fdiv_rn',
                         '__fmaf_rn': 'cumlx_fma_rn', '__fmaf_ieee_rn': 'cumlx_fma_rn', '__frcp_rn': 'cumlx_frcp_rn',
                         '__fsqrt_rn': 'cumlx_fsqrt_rn', '__frsqrt_rn': 'cumlx_rsqrtf', 'rsqrtf': 'cumlx_rsqrtf',
                         '__saturatef': 'cumlx_saturatef', '__fdividef': 'cumlx_fdividef', '__expf': 'exp', '__exp10f': 'cumlx_exp10f',
                         '__half2float': 'cumlx_half2float', '__float2half': 'cumlx_float2half', '__clz': 'cumlx_clz', '__popc': 'cumlx_popc',
                         '__brev': 'cumlx_brev', '__ffs': 'cumlx_ffs', '__hadd': 'cumlx_hadd', '__rhadd': 'cumlx_rhadd',
                         '__mulhi': 'cumlx_mulhi', '__umulhi': 'cumlx_umulhi'}.items():
        if helper == 'exp':
            table[name] = BuiltIn('rewrite', 'exp', 'expf')
        else:
            table[name] = BuiltIn('rewrite', helper=helper)
    for name, target in {'__sinf': 'sinf', '__cosf': 'cosf', '__tanf': 'tanf', '__logf': 'logf', '__log2f': 'log2f', '__log10f': 'log10f', '__powf': 'powf'}.items():
        table[name] = BuiltIn('rewrite', helper=target)
    for op in ('fadd', 'fsub', 'fmul', 'fdiv', 'fmaf', 'frcp', 'fsqrt', 'dadd', 'dsub', 'dmul', 'ddiv', 'fma', 'drcp', 'dsqrt'):
        for mode in ('rz', 'ru', 'rd'):
            table[f'__{op}_{mode}'] = BuiltIn('refuse', reason=_ROUNDING)
    for op in ('dadd', 'dsub', 'dmul', 'ddiv', 'fma', 'drcp', 'dsqrt'):
        table[f'__{op}_rn'] = BuiltIn('refuse', reason='fp64: Metal has no double.')
    for name in ('__match_any_sync', '__match_all_sync', '__activemask', '__ballot_sync', '__ballot', '__all_sync', '__all', '__any_sync', '__any',
                 '__shfl_sync', '__shfl', '__shfl_up_sync', '__shfl_up', '__shfl_down_sync', '__shfl_down', '__shfl_xor_sync', '__shfl_xor', '__syncwarp'):
        table[name] = BuiltIn('refuse', reason=_WARP)
    for name in ('__threadfence', '__threadfence_block', '__threadfence_system'):
        table[name] = BuiltIn('refuse', reason=_FENCE)
    for name in ('tex1D', 'tex2D', 'tex3D', 'texCubemap', 'tex1Dfetch', 'tex2DLod', 'tex3DLod', 'tex2DLayered', 'tex2DGather',
                 'cudaTextureObject_t', 'cudaSurfaceObject_t', 'surf1Dread', 'surf2Dread', 'surf3Dread', 'surf1Dwrite', 'surf2Dwrite', 'surf3Dwrite'):
        table[name] = BuiltIn('refuse', reason=_TEX)
    for name in ('erff', 'erfcf', 'erfinvf', 'erfcinvf', 'erfcxf', 'lgammaf', 'tgammaf', 'j0f', 'j1f', 'jnf', 'y0f', 'y1f', 'ynf',
                 'cyl_bessel_i0f', 'cyl_bessel_i1f', 'normcdff', 'normcdfinvf'):
        table[name] = BuiltIn('refuse', reason=_NOLIB)
    table.update({
        'printf': BuiltIn('refuse', reason='A device printf has no Metal equivalent. Remove it from the kernel.'),
        'assert': BuiltIn('refuse', reason='A device assert has no Metal equivalent. Remove it from the kernel.'),
        '__assert_fail': BuiltIn('refuse', reason='A device assert has no Metal equivalent. Remove it from the kernel.'),
        '__trap': BuiltIn('refuse', reason='__trap has no Metal equivalent.'), '__brkpt': BuiltIn('refuse', reason='__brkpt has no Metal equivalent.'),
        'asm': BuiltIn('refuse', reason='Inline PTX has no Metal equivalent.'), '__asm': BuiltIn('refuse', reason='Inline PTX has no Metal equivalent.'),
        '__asm__': BuiltIn('refuse', reason='Inline PTX has no Metal equivalent.'),
        '__CUDA_ARCH__': BuiltIn('refuse', reason='Code that branches on the GPU architecture cannot be resolved here. Keep only the branch you want.'),
        'wmma': BuiltIn('refuse', reason='Tensor Core instructions (wmma) have no Metal equivalent that this layer can map. Rewrite with plain FMA loops.'),
        'cooperative_groups': BuiltIn('refuse', reason='CUDA cooperative groups have no Metal equivalent.'),
        'curand_init': BuiltIn('refuse', reason='CURAND has no Metal equivalent. Its sequence could not be reproduced.'),
        'thrust': BuiltIn('refuse', reason='Thrust is a host and device library with no Metal equivalent here.'),
        '__nv_bfloat16': BuiltIn('refuse', reason='bfloat16 is not available on all Metal GPUs. Convert to float at the kernel boundary.'),
        '__float128': BuiltIn('refuse', reason='Metal has no 128-bit float.'), '__int128': BuiltIn('refuse', reason='Metal has no 128-bit integer.')})
    return table


PTX_BUILTINS = _build_builtins()
# Qualifiers and compiler words that may appear in device code, and what happens to them.
ALLOWED_DUNDER = frozenset({'__global__', '__device__', '__host__', '__forceinline__', '__inline__', '__shared__', '__constant__', '__restrict__',
                            '__half', '__FILE__', '__LINE__', '__func__', '__attribute__', '__builtin_expect', '__noinline__',
                            '__launch_bounds__', '__grid_constant__', '__maxnreg__'})
OPERATOR_NAME = re.compile(r'\boperator\s*(?:\(\s*\)|\[\s*\]|[-+*/%^&|~!=<>,]+)(?=\s*\()')
STRIPPED_MARKERS = re.compile(r'\b(?:__launch_bounds__|__maxnreg__)\s*\([^()]*\)[ \t]*|\b(?:__noinline__|__grid_constant__)\b[ \t]*')
_SIMPLE_TYPES = frozenset({'float', 'int', 'unsigned', 'unsigned int', 'char', 'unsigned char', 'short', 'unsigned short', 'long', 'long long',
                           'unsigned long long', 'bool', 'double', 'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'int8_t', 'int16_t', 'int32_t',
                           'int64_t', 'size_t', '__half'})


@dataclass
class Refusal:
    feature: str
    token: str
    reason: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unportable(name: str, refusals: Iterable[Refusal]) -> CumlxError:
    items = [r.to_dict() for r in refusals]
    text = '; '.join(f"{r['token']}: {r['feature']}" for r in items[:6])
    more = f' and {len(items) - 6} more' if len(items) > 6 else ''
    return CumlxError('E_KERNEL_UNPORTABLE', f'{name} cannot be ported: {text}{more}.',
                      hint=items[0]['reason'] if items else None, details={'refused': items})


# ---------------------------------------------------------------------------- includes
@dataclass
class HeaderRecord:
    replaced: dict[str, str | None] = field(default_factory=dict)
    host_dropped: list[str] = field(default_factory=list)
    inlined: list[str] = field(default_factory=list)
    unknown_dropped: list[str] = field(default_factory=list)


_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"\n]+)[>"][^\n]*$', re.M)


def flatten_includes(text: str, base: Path, include_dirs: list[Path], record: HeaderRecord, refusals: list[Refusal], *,
                     seen: set[Path] | None = None) -> str:
    """Remove comments and resolve each #include: replace it, drop it, inline it, or refuse it."""
    seen = set() if seen is None else seen
    clean = strip_comments(text)

    def resolve(match: re.Match[str]) -> str:
        kind, name = match.group(1), match.group(2).strip()
        leaf = name.split('/')[-1]
        if kind == '"':  # a quoted name means the project's own file first, even if a standard header has that name
            for folder in [base, *include_dirs]:
                candidate = (folder / name).resolve()
                if candidate.is_file():
                    if candidate in seen:
                        return ''
                    seen.add(candidate)
                    record.inlined.append(name)
                    return flatten_includes(candidate.read_text(errors='replace'), candidate.parent, include_dirs, record, refusals, seen=seen)
        if name in STANDARD_HEADERS or leaf in STANDARD_HEADERS:
            value = STANDARD_HEADERS[name if name in STANDARD_HEADERS else leaf]
            record.replaced[name] = value
            return ''
        if name in HOST_HEADERS or name.startswith(HOST_HEADER_PREFIXES):
            record.host_dropped.append(name)
            return ''
        if kind == '"':
            refusals.append(Refusal('unresolved-include', f'"{name}"', f'{name} is not in the folder of the source or in --include-dir. Pass --include-dir.',
                                    line_of(clean, match.start())))
            return ''
        record.unknown_dropped.append(name)
        return ''

    return _INCLUDE.sub(resolve, clean)


# ---------------------------------------------------------------------------- declarations
@dataclass
class Chunk:
    kind: str  # directive, function, prototype, type, variable, other
    text: str
    start: int
    name: str | None = None
    device: bool = False
    is_global: bool = False
    template: str | None = None  # the text inside template<...>
    keep_reason: str | None = None


def _match_close(masked: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for i in range(start, len(masked)):
        if masked[i] == opening:
            depth += 1
        elif masked[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    raise CumlxError('E_INPUT_INVALID', f'unbalanced {opening}{closing} near line {line_of(masked, start)}')


def _template_head(head: str) -> tuple[str | None, str]:
    stripped = head.lstrip()
    if not re.match(r'template\b', stripped):
        return None, head
    open_at = stripped.find('<')
    if open_at < 0:
        return None, head
    depth = 0
    for i in range(open_at, len(stripped)):
        if stripped[i] == '<':
            depth += 1
        elif stripped[i] == '>':
            depth -= 1
            if depth == 0:
                return stripped[open_at + 1:i].strip(), stripped[i + 1:]
    return None, head


def _strip_attributes(head: str) -> str:
    head = STRIPPED_MARKERS.sub(' ', head)
    while True:
        m = re.search(r'\b__attribute__\s*\(', head)
        if not m:
            return head
        depth, i = 0, m.end() - 1
        while i < len(head):
            depth += {'(': 1, ')': -1}.get(head[i], 0)
            i += 1
            if depth == 0:
                break
        head = head[:m.start()] + ' ' + head[i:]


def _function_name(head: str) -> str | None:
    head = _strip_attributes(head)
    close = head.rfind(')')
    if close < 0:
        return None
    depth, i = 0, close
    while i >= 0:
        depth += {')': 1, '(': -1}.get(head[i], 0)
        if depth == 0:
            break
        i -= 1
    match = re.search(r'([A-Za-z_]\w*)\s*$', head[:i])
    return match.group(1) if match else None


def _declaration_shape(rest: str) -> tuple[int, bool]:
    """Where the first parenthesis is, and whether an `=` comes before it. The name of `operator*=` counts as neither."""
    plain = OPERATOR_NAME.sub(lambda m: ' ' * len(m.group(0)), rest)
    first_paren = plain.find('(')
    before = plain[:first_paren] if first_paren >= 0 else plain
    return first_paren, re.search(r'(?<![=!<>])=(?!=)', before) is not None


def _classify(head: str, text: str, start: int, ended_with_body: bool) -> Chunk:
    template, rest = _template_head(head)
    tokens = set(IDENT.findall(rest))
    is_global = '__global__' in tokens
    device = is_global or '__device__' in tokens or '__constant__' in tokens
    if re.match(r'\s*(?:typedef|using)\b', rest):
        names = IDENT.findall(rest.split('=')[0])
        if rest.lstrip().startswith('using namespace'):
            return Chunk('other', text, start, None)
        return Chunk('type', text, start, names[-1] if names else None, template=template)
    first_paren, has_assign = _declaration_shape(rest)
    keyword = re.search(r'\b(struct|class|union|enum)\b', rest[:first_paren] if first_paren >= 0 else rest)
    if keyword and not ended_with_body:
        after = IDENT.findall(rest[keyword.end():].split('{')[0].split(':')[0])
        after = [t for t in after if t not in ('class', 'struct')]
        return Chunk('type', text, start, after[0] if after else None, template=template)
    if keyword and ended_with_body and first_paren < 0:
        after = IDENT.findall(rest[keyword.end():].split(':')[0])
        return Chunk('type', text, start, after[0] if after else None, template=template)
    if first_paren >= 0 and not has_assign:
        name = _function_name(rest)
        kind = 'function' if ended_with_body else 'prototype'
        return Chunk(kind, text, start, name, device=device, is_global=is_global, template=template)
    names = IDENT.findall(rest.split('=')[0].split('[')[0])
    return Chunk('variable', text, start, names[-1] if names else None, device=device, template=template)


def split_declarations(source: str) -> list[Chunk]:
    """Split a source into its top-level directives and declarations. ``namespace`` and ``extern "C"`` blocks are opened."""
    masked = mask_source(source)
    chunks: list[Chunk] = []

    def walk(begin: int, end: int) -> None:
        i = begin
        while i < end:
            while i < end and masked[i].isspace():
                i += 1
            if i >= end:
                return
            if masked[i] == '#':
                j = i
                while True:
                    nl = masked.find('\n', j, end)
                    if nl < 0:
                        j = end
                        break
                    if masked[nl - 1] != '\\':
                        j = nl
                        break
                    j = nl + 1
                text = source[i:j]
                m = re.match(r'#\s*define\s+([A-Za-z_]\w*)', masked[i:j])
                chunks.append(Chunk('directive', text, i, m.group(1) if m else None))
                i = j
                continue
            depth, j, done = 0, i, False
            ended_with_body = False
            while j < end and not done:
                c = masked[j]
                if c in '([':
                    depth += 1
                elif c in ')]':
                    depth -= 1
                elif depth == 0 and c == ';':
                    j += 1
                    done = True
                    continue
                elif depth == 0 and c == '{':
                    head = masked[i:j]
                    close = _match_close(masked, j, '{', '}')
                    if re.match(r'\s*(?:inline\s+)?namespace\b', head) or re.match(r'\s*extern\s*"\s*C\s*"\s*$', source[i:j]):
                        walk(j + 1, close)
                        i = close + 1
                        done = True
                        j = -1
                        continue
                    _, rest = _template_head(head)
                    first_paren, has_assign = _declaration_shape(rest)
                    is_type = re.search(r'\b(struct|class|union|enum)\b', rest[:first_paren] if first_paren >= 0 else rest)
                    if first_paren >= 0 and not is_type and not has_assign:
                        j = close + 1
                        ended_with_body = True
                        done = True
                        continue
                    j = close + 1
                    continue
                j += 1
            if j == -1:
                continue
            j = min(j, end)
            if not done:
                raise CumlxError('E_INPUT_INVALID', f'a declaration at line {line_of(masked, i)} does not end')
            head_end = source.find('{', i, j) if ended_with_body else j
            head_masked = masked[i:head_end if head_end >= 0 else j]
            if ended_with_body:
                head_masked = masked[i:masked.find('{', i, j)]
            chunks.append(_classify(head_masked, source[i:j], i, ended_with_body))
            i = j

    walk(0, len(source))
    return chunks


# ---------------------------------------------------------------------------- device-code checks
def _check_device_code(text: str, *, base_line: int, demote_fp64: bool, allow_warp_level: bool = False) -> tuple[list[Refusal], dict[str, str], dict[str, str]]:
    """Return refusals, the built-ins used that the translator must know, and the built-ins this layer rewrites."""
    masked = mask_source(text)
    refusals: list[Refusal] = []
    direct: dict[str, str] = {}
    rewritten: dict[str, str] = {}
    for match in IDENT.finditer(masked):
        token = match.group(0)
        line = base_line + masked.count('\n', 0, match.start())
        builtin = PTX_BUILTINS.get(token)
        if builtin:
            if builtin.status == 'refuse' and builtin.reason == _WARP and allow_warp_level:
                direct[token] = 'passed to the translator by --allow-warp-level'
            elif builtin.status == 'refuse':
                feature = ('warp-level' if builtin.reason == _WARP else 'rounding-mode' if builtin.reason == _ROUNDING
                           else 'no-metal-equivalent')
                refusals.append(Refusal(feature, token, builtin.reason, line))
            elif builtin.status == 'direct':
                direct[token] = builtin.metal
            else:
                rewritten[token] = builtin.helper
            continue
        if token == 'double' and not demote_fp64:
            refusals.append(Refusal('fp64', token, 'Metal has no double. Pass --demote-fp64 to compute in float32, and let `cumlx kernel verify` judge the result.', line))
        elif token.startswith('__') and token not in ALLOWED_DUNDER:
            refusals.append(Refusal('unknown-builtin', token, 'This layer does not know this name, so it cannot say that it is safe to translate.', line))
    if '<<<' in masked:
        refusals.append(Refusal('dynamic-parallelism', '<<<', 'A kernel launch from device code has no Metal equivalent.',
                                base_line + masked.count('\n', 0, masked.index('<<<'))))
    for match in re.finditer(r'\bstd::(\w+)', masked):
        if match.group(1) not in STD_ALLOWED and match.group(1) != 'numeric_limits':
            refusals.append(Refusal('host-symbol-in-device', f'std::{match.group(1)}', 'Device code uses a host library type or function, which has no Metal equivalent.',
                                    base_line + masked.count('\n', 0, match.start())))
    return refusals, direct, rewritten


def _rewrite_device_code(text: str, *, demote_fp64: bool) -> str:
    masked = mask_source(text)
    pieces, last = [], 0
    for match in IDENT.finditer(masked):
        builtin = PTX_BUILTINS.get(match.group(0))
        replacement = builtin.helper if builtin and builtin.status == 'rewrite' else None
        if demote_fp64 and match.group(0) == 'double':
            replacement = 'float'
        if replacement:
            pieces += [text[last:match.start()], replacement]
            last = match.end()
    pieces.append(text[last:])
    out = ''.join(pieces)
    out = re.sub(r'\bstd::numeric_limits\s*<\s*([\w ]+?)\s*>\s*::\s*(\w+)\s*\(\s*\)',
                 lambda m: NUMERIC_LIMITS.get((m.group(1), m.group(2)), m.group(0)), out)
    out = re.sub(r'\bstd::(?=\w)', '', out)
    if demote_fp64:
        masked = mask_source(out)
        pieces, last = [], 0
        for match in FP64_LITERAL.finditer(masked):
            pieces += [out[last:match.end()], 'f']
            last = match.end()
        pieces.append(out[last:])
        out = ''.join(pieces)
    return STRIPPED_MARKERS.sub('', out)


def _prelude(text: str, includes: Iterable[str]) -> str:
    """The lines that replace the dropped headers. Each is present only if the kernel uses its name."""
    tokens = set(IDENT.findall(mask_source(text)))
    lines = [f'#include {inc}\n' for inc in sorted(set(includes))]
    used = tokens & set(_HELPERS)
    used |= {need for helper in used for need in _HELPER_NEEDS.get(helper, ())}
    helpers = sorted(used, key=lambda h: (h in _HELPER_NEEDS, h))  # a helper comes before the helpers that call it
    if '__half' in tokens or {'cumlx_half2float', 'cumlx_float2half'} & tokens:
        lines.append('#include <cuda_fp16.h>\n')  # __half is a class from this header. A translator must know it
    if helpers and '<math.h>' not in includes:
        lines.append('#include <math.h>\n')
    for name, value in HEADER_DEFINES.items():
        if name in tokens:
            lines.append(f'#ifndef {name}\n#define {name} {value}\n#endif\n')
    if 'size_t' in tokens:
        lines.append('typedef unsigned long size_t;\n')
    return ''.join(lines) + ''.join(_HELPERS[h] for h in helpers)


# ---------------------------------------------------------------------------- templates
def _template_params(head: str) -> list[tuple[str, str | None, bool]]:
    """(name, default, is_type) for each template parameter."""
    params, depth, current = [], 0, ''
    for c in head + ',':
        if c == ',' and depth == 0:
            if current.strip():
                params.append(current.strip())
            current = ''
            continue
        depth += {'<': 1, '>': -1, '(': 1, ')': -1}.get(c, 0)
        current += c
    out = []
    for p in params:
        left, _, default = p.partition('=')
        names = IDENT.findall(left)
        if not names:
            continue
        is_type = names[0] in ('typename', 'class')
        out.append((names[-1], default.strip() or None, is_type))
    return out


def _split_args(text: str) -> list[str]:
    args, depth, current = [], 0, ''
    for c in text + ',':
        if c == ',' and depth == 0:
            args.append(current.strip())
            current = ''
            continue
        depth += {'<': 1, '>': -1, '(': 1, ')': -1}.get(c, 0)
        current += c
    return [a for a in args if a]


def _valid_argument(arg: str) -> bool:
    return arg in _SIMPLE_TYPES or bool(re.fullmatch(r'[-+]?\d+[uUlL]*|true|false', arg))


def _instance_name(name: str, args: list[str]) -> str:
    return name + '__' + '_'.join(re.sub(r'\W+', '_', a) for a in args)


def _instantiate(chunk: Chunk, args: list[str], new_name: str) -> str:
    params = _template_params(chunk.template or '')
    values = list(args)
    for param_name, default, _ in params[len(values):]:
        if default is None:
            raise CumlxError('E_INPUT_INVALID', f'{chunk.name}: template parameter {param_name} has no value and no default')
        values.append(default)
    if len(values) > len(params):
        raise CumlxError('E_USAGE', f'{chunk.name} takes {len(params)} template argument(s), got {len(values)}')
    _, rest = _template_head(chunk.text)
    mapping = {p[0]: v for p, v in zip(params, values)}
    rest = re.sub(r'\b(' + '|'.join(re.escape(k) for k in mapping) + r')\b', lambda m: mapping[m.group(1)], rest) if mapping else rest
    assert chunk.name is not None
    return re.sub(rf'\b{re.escape(chunk.name)}(\s*\()', lambda m: new_name + m.group(1), rest, count=1)


@dataclass
class Preprocessed:
    name: str
    form: str  # complete or body
    source: str
    header: str
    entry_points: list[str]
    source_sha256: str
    record: dict[str, Any]
    warnings: list[dict[str, Any]]


def host_instantiations(chunks: list[Chunk], template_names: set[str]) -> tuple[dict[str, list[list[str]]], list[str]]:
    """Find `name<args>` in host code. Returns the usable instances, and the uses whose arguments are not concrete."""
    found: dict[str, list[list[str]]] = {}
    unresolved: list[str] = []
    if not template_names:
        return found, unresolved
    pattern = re.compile(r'\b(' + '|'.join(re.escape(n) for n in sorted(template_names)) + r')\s*<([^<>;{}]*(?:<[^<>;{}]*>[^<>;{}]*)*)>')
    for chunk in chunks:
        if chunk.device or chunk.kind == 'directive':
            continue
        for match in pattern.finditer(mask_source(chunk.text)):
            args = _split_args(match.group(2))
            if args and all(_valid_argument(a) for a in args):
                if args not in found.setdefault(match.group(1), []):
                    found[match.group(1)].append(args)
            else:
                unresolved.append(match.group(0))
        tail = re.finditer(r'\btemplate\s+__global__[^;{]*?\b(\w+)\s*<([^<>;{}]*)>', mask_source(chunk.text))
        for match in tail:
            if match.group(1) in template_names:
                args = _split_args(match.group(2))
                if args not in found.setdefault(match.group(1), []):
                    found[match.group(1)].append(args)
    return found, unresolved


def parse_instantiate(items: list[str]) -> dict[str, list[list[str]]]:
    out: dict[str, list[list[str]]] = {}
    for item in items:
        match = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*<(.*)>\s*', item)
        if not match:
            raise CumlxError('E_USAGE', f'--instantiate needs NAME<ARGS>, such as vector_add<float>, got {item!r}')
        args = _split_args(match.group(2))
        if not args or not all(_valid_argument(a) for a in args):
            raise CumlxError('E_USAGE', f'--instantiate {item!r}: each argument must be a plain type or an integer',
                             details={'types': sorted(_SIMPLE_TYPES)})
        out.setdefault(match.group(1), []).append(args)
    return out


def _plain_comma_count(text: str) -> int:
    """How many items `text` holds if a `<` in it is a comparison. `_split_args` reads it as a bracket."""
    if not text.strip():
        return 0
    count, depth = 1, 0
    for c in text:
        depth += {'(': 1, '[': 1, '{': 1, ')': -1, ']': -1, '}': -1}.get(c, 0)
        count += c == ',' and depth == 0
    return count


def _parameter_range(masked: str) -> tuple[int, int | None]:
    """The fewest and the most arguments that a call to this definition can have. None means no upper limit."""
    brace = masked.find('{')
    head = _strip_attributes(masked[:brace] if brace >= 0 else masked)
    close = head.rfind(')')
    if close < 0:
        return 0, None
    depth, opening = 0, close
    while opening >= 0:
        depth += {')': 1, '(': -1}.get(head[opening], 0)
        if depth == 0:
            break
        opening -= 1
    inner = head[opening + 1:close].strip()
    if inner == 'void':
        return 0, 0
    params = _split_args(inner)
    required = sum(1 for p in params if '=' not in p and '...' not in p)
    if '...' in inner:
        return required, None
    return required, max(len(params), _plain_comma_count(inner))


def _call_arities(masked: str, opening: int) -> set[int] | None:
    """The argument counts that this call can have, or None if the parentheses cannot be read here."""
    try:
        close = _match_close(masked, opening, '(', ')')
    except CumlxError:
        return None
    inner = masked[opening + 1:close]
    return {len(_split_args(inner)), _plain_comma_count(inner)}


_CallNode = tuple[str, int]  # a function's name and which definition of that name


def _call_graph_cycle(functions: dict[str, list[str]]) -> list[str] | None:
    """Find a call cycle. Each definition is its own node, so a wrapper of a built-in with the same name is not a cycle.

    A call goes to each overload that accepts its number of arguments. This layer does not read types, so overloads
    with the same number of arguments are all candidates: a cycle among them is reported, because it cannot be ruled out.
    """
    bodies: dict[_CallNode, str] = {}
    ranges: dict[_CallNode, tuple[int, int | None]] = {}
    for name, texts in functions.items():
        for index, text in enumerate(texts):
            masked = mask_source(text)
            bodies[name, index] = masked[max(masked.find('{'), 0):]
            ranges[name, index] = _parameter_range(masked)

    def accepts(node: _CallNode, arities: set[int] | None) -> bool:
        if arities is None:
            return True
        low, high = ranges[node]
        return any(low <= n and (high is None or n <= high) for n in arities)

    graph: dict[_CallNode, set[_CallNode]] = {}
    for node, body in bodies.items():
        graph[node] = set()
        for m in re.finditer(r'\b([A-Za-z_]\w*)\s*\(', body):
            if m.group(1) in functions:
                arities = _call_arities(body, m.end() - 1)
                graph[node].update((m.group(1), i) for i in range(len(functions[m.group(1)])) if accepts((m.group(1), i), arities))
    state: dict[_CallNode, int] = {}
    stack: list[_CallNode] = []

    def visit(node: _CallNode) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for callee in sorted(graph[node]):
            if state.get(callee) == 1:
                return [n[0] for n in stack[stack.index(callee):]] + [callee[0]]
            if callee not in state:
                found = visit(callee)
                if found:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if node not in state:
            found = visit(node)
            if found:
                return found
    return None


def preprocess_complete(flat: str, *, name: str, sha: str, record: HeaderRecord, refusals: list[Refusal], instantiate: dict[str, list[list[str]]],
                        demote_fp64: bool, warnings: list[dict[str, Any]], allow_warp_level: bool = False) -> Preprocessed:
    chunks = split_declarations(flat)
    device_functions = [c for c in chunks if c.kind in ('function', 'prototype') and c.device]
    entries = [c for c in device_functions if c.is_global and c.kind == 'function']
    if not entries:
        refusals.append(Refusal('no-entry-point', name, 'The source has no __global__ function, so there is nothing to translate.'))
    for chunk in chunks:
        if chunk.kind == 'variable' and chunk.device and '__constant__' not in chunk.text.split('=')[0]:
            refusals.append(Refusal('device-global', chunk.name or '?', 'A mutable __device__ global has no Metal equivalent: Metal has no program-scope device memory. Pass it in a buffer.',
                                    line_of(flat, chunk.start)))
        if chunk.kind == 'variable' and re.search(r'\b(__managed__|__shared__)\b', chunk.text.split('=')[0]):
            refusals.append(Refusal('device-global', chunk.name or '?', 'A global __managed__ or __shared__ variable has no Metal equivalent. Pass it in a buffer.',
                                    line_of(flat, chunk.start)))
        if chunk.kind in ('function', 'prototype') and chunk.template == '' and chunk.device:
            refusals.append(Refusal('template-specialization', chunk.name or '?', 'This layer does not handle a full template specialization. Write the specialized kernel as a plain function.',
                                    line_of(flat, chunk.start)))

    # Which declarations device code needs: start from the device functions and follow the names.
    by_name: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.name:
            by_name.setdefault(chunk.name, []).append(chunk)
    kept: set[int] = {id(c) for c in device_functions}
    kept.update(id(c) for c in chunks if c.kind == 'variable' and c.device)
    host_functions = {c.name: c for c in chunks if c.kind in ('function', 'prototype') and not c.device and c.name}
    work = [c for c in chunks if id(c) in kept]
    while work:
        current = work.pop()
        masked = mask_source(current.text)
        for token in set(IDENT.findall(masked)):
            for candidate in by_name.get(token, []):
                if id(candidate) in kept or candidate.kind in ('function', 'prototype'):
                    continue
                if candidate.kind == 'variable' and not re.search(r'\b(?:const|constexpr)\b', candidate.text.split('=')[0]):
                    continue
                kept.add(id(candidate))
                work.append(candidate)
        for callee in host_functions:
            if current.kind in ('function', 'variable', 'type') and re.search(rf'(?<![\w.>:]){re.escape(callee)}\s*\(', masked):
                refusals.append(Refusal('host-symbol-in-device', callee, f'Device code calls {callee}, which is a host function. Mark it __device__ if it is safe to run on the GPU.',
                                        line_of(flat, current.start)))

    # Templates: an entry point must have concrete types. A device helper stays a template unless a call names its arguments.
    template_entries = {c.name for c in entries if c.template and c.name}
    template_helpers = {c.name for c in device_functions if c.template and not c.is_global and c.name}
    from_host, unresolved = host_instantiations(chunks, template_entries)
    instances: dict[str, list[list[str]]] = {}
    for tname in sorted(template_entries):
        chunk = next(c for c in entries if c.name == tname)
        params = _template_params(chunk.template or '')
        chosen = instantiate.get(tname) or from_host.get(tname)
        if not chosen and all(p[1] is not None for p in params):
            chosen = [[]]
        if not chosen:
            hint = f'pass --instantiate "{tname}<float>"'
            if unresolved:
                hint += f'; the host code has {unresolved[:3]}, whose arguments are not concrete'
            refusals.append(Refusal('uninstantiated-template', tname, f'{tname} is a template, and the host code names no concrete instance: {hint}.', line_of(flat, chunk.start)))
            continue
        instances[tname] = chosen
    unknown = sorted(set(instantiate) - template_entries)
    if unknown:
        raise CumlxError('E_USAGE', f'--instantiate names what is not a template __global__ function here: {unknown}',
                         details={'template_entries': sorted(template_entries)})

    entry_points: list[str] = []
    entry_map: dict[str, str] = {}
    pieces: list[tuple[int, str]] = []  # (start offset, text); instances keep the position of their template
    helper_instances: dict[str, str] = {}
    used_helpers: set[str] = set()
    checked: list[tuple[str, int]] = []
    function_texts: dict[str, list[str]] = {}

    def expand_helpers(text: str) -> str:
        """Replace `helper<args>(` with a named instance, and record each instance to emit."""
        for hname in sorted(template_helpers):
            pattern = re.compile(rf'\b{re.escape(hname)}\s*<([^<>;{{}}]*)>(?=\s*\()')
            for match in list(pattern.finditer(mask_source(text))):
                args = _split_args(match.group(1))
                if not all(_valid_argument(a) for a in args):
                    continue
                inst = _instance_name(hname, args)
                if inst not in helper_instances:
                    helper = next(c for c in device_functions if c.name == hname)
                    helper_instances[inst] = ''  # stops a helper that calls itself; the recursion check names it
                    helper_instances[inst] = expand_helpers(_instantiate(helper, args, inst))

            def rename(m: re.Match[str], helper_name: str = hname) -> str:
                args = _split_args(m.group(1))
                return _instance_name(helper_name, args) if all(_valid_argument(a) for a in args) else m.group(0)

            text = pattern.sub(rename, text)
        return text

    for chunk in chunks:
        if chunk.kind == 'directive':
            if chunk.name and id(chunk) not in kept:
                continue
            if re.match(r'#\s*pragma\b', chunk.text):
                continue
            pieces.append((chunk.start, chunk.text + '\n'))
            continue
        if id(chunk) not in kept:
            continue
        if chunk.kind == 'prototype' and (chunk.is_global or chunk.template is not None):
            continue  # only host code needs the prototype of a kernel, and a template instance needs none
        if chunk.template is not None and chunk.is_global and chunk.name in instances:
            many = len(instances[chunk.name]) > 1
            for args in instances[chunk.name]:
                new_name = _instance_name(chunk.name, args) if many and args else chunk.name
                text = expand_helpers(_instantiate(chunk, args, new_name))
                entry_points.append(new_name)
                entry_map[f"{chunk.name}<{', '.join(args)}>" if args else chunk.name] = new_name
                function_texts.setdefault(new_name, []).append(text)
                pieces.append((chunk.start, 'extern "C" ' + re.sub(r'^\s*(?:static\s+)?', '', text.lstrip()) + '\n'))
                checked.append((text, line_of(flat, chunk.start)))
            continue
        if chunk.template is not None and chunk.name in template_helpers:
            used_helpers.add(chunk.name)
            checked.append((chunk.text, line_of(flat, chunk.start)))
            function_texts.setdefault(chunk.name, []).append(chunk.text)
            continue  # placed below if a call still deduces its arguments
        text = expand_helpers(chunk.text) if chunk.kind in ('function', 'variable', 'type') else chunk.text
        if chunk.kind == 'function' and chunk.name:
            function_texts.setdefault(chunk.name, []).append(text)
        if chunk.is_global and chunk.kind == 'function' and chunk.name:
            entry_points.append(chunk.name)
            text = 'extern "C" ' + re.sub(r'^\s*(?:static\s+)?', '', text.lstrip())
        pieces.append((chunk.start, text + '\n'))
        checked.append((text, line_of(flat, chunk.start)))
    cycle = _call_graph_cycle(function_texts)
    if cycle:
        refusals.append(Refusal('recursion', ' -> '.join(cycle), 'Metal does not allow recursion in a kernel. Rewrite it as a loop. If these are overloads with the same number of arguments, this layer cannot tell them apart by type: rename one.'))

    # A helper called with deduced arguments stays a template, and the translator must deduce them. State this.
    kept_templates = []
    rendered = ''.join(t for _, t in pieces) + ''.join(helper_instances.values())
    for hname in sorted(used_helpers):
        if re.search(rf'\b{re.escape(hname)}\s*\(', mask_source(rendered)):
            chunk = next(c for c in device_functions if c.name == hname)
            kept_templates.append(hname)
            pieces.append((chunk.start, chunk.text + '\n'))
    if kept_templates:
        warnings.append({'code': 'W_KERNEL_TEMPLATE_KEPT', 'message': f'{kept_templates} stay templates because a call deduces their arguments. The translator must handle this.', 'templates': kept_templates})
    # Instances go right before the earliest entry that uses them. Helpers are declared before they are used in valid CUDA.
    first_entry = min((c.start for c in entries), default=0)
    pieces.extend((first_entry - 1, t + '\n') for t in helper_instances.values())
    pieces.sort(key=lambda item: item[0])
    body = ''.join(t for _, t in pieces)

    direct_all: dict[str, str] = {}
    rewritten_all: dict[str, str] = {}
    for text, line in checked + [(t, line_of(flat, first_entry)) for t in helper_instances.values()]:
        found, direct, rewritten = _check_device_code(text, base_line=line, demote_fp64=demote_fp64, allow_warp_level=allow_warp_level)
        refusals.extend(found)
        direct_all.update(direct)
        rewritten_all.update(rewritten)
    for chunk in chunks:  # a type or macro that device code needs is checked with it
        if id(chunk) in kept and chunk.kind in ('type', 'directive', 'variable'):
            found, direct, rewritten = _check_device_code(chunk.text, base_line=line_of(flat, chunk.start), demote_fp64=demote_fp64, allow_warp_level=allow_warp_level)
            refusals.extend(found)
            direct_all.update(direct)
            rewritten_all.update(rewritten)
    if refusals:
        raise unportable(name, _unique(refusals))
    body = _rewrite_device_code(body, demote_fp64=demote_fp64)
    includes = [v for v in record.replaced.values() if v]
    source = _prelude(body, includes) + body
    host_removed = sorted({c.name for c in chunks if c.kind in ('function', 'prototype') and not c.device and c.name})
    rec = {'host_removed': host_removed, 'templates': entry_map, 'builtins_direct': direct_all, 'builtins_rewritten': rewritten_all}
    return Preprocessed(name, 'complete', source, '', entry_points, sha, rec, warnings)


def _unique(refusals: list[Refusal]) -> list[Refusal]:
    seen, out = set(), []
    for r in refusals:
        key = (r.feature, r.token, r.line)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _preprocess_body(text: str, header_text: str, *, name: str, sha: str, record: HeaderRecord, refusals: list[Refusal],
                     demote_fp64: bool, wrap: KernelCase | None, warnings: list[dict[str, Any]], allow_warp_level: bool = False) -> Preprocessed:
    direct_all: dict[str, str] = {}
    rewritten_all: dict[str, str] = {}
    for label, part in (('header', header_text), ('body', text)):
        found, direct, rewritten = _check_device_code(part, base_line=1, demote_fp64=demote_fp64, allow_warp_level=allow_warp_level)
        for item in found:
            item.reason = f'{label}: {item.reason}'
        refusals.extend(found)
        direct_all.update(direct)
        rewritten_all.update(rewritten)
    if refusals:
        raise unportable(name, _unique(refusals))
    body = _rewrite_device_code(text.strip() + '\n', demote_fp64=demote_fp64)
    head = _rewrite_device_code(header_text, demote_fp64=demote_fp64)
    includes = [v for v in record.replaced.values() if v]
    head = _prelude(head + body, includes) + head.strip() + '\n'
    rec = {'host_removed': [], 'templates': {}, 'builtins_direct': direct_all, 'builtins_rewritten': rewritten_all}
    if wrap is None:
        return Preprocessed(name, 'body', body, head, [name], sha, rec, warnings)
    macros = ''.join(f'#define {k} {int(v)}\n' for k, v in wrap.constants.items())
    complete = head + macros + wrap_body(wrap, body, name=name, extern_c=True)
    rec['case'] = wrap.id
    return Preprocessed(name, 'complete', complete, '', [name], sha, rec, warnings)


def preprocess_file(path: str | Path, *, instantiate: list[str] | None = None, include_dirs: list[str] | None = None, header: str | None = None,
                    demote_fp64: bool = False, cases_file: str | None = None, case_id: str | None = None,
                    allow_warp_level: bool = False) -> Preprocessed:
    p = Path(path)
    if not p.is_file():
        raise CumlxError('E_INPUT_NOT_FOUND', f'{p} does not exist')
    raw = p.read_text(errors='replace')
    name = p.name.split('.')[0]
    if not re.fullmatch(r'[A-Za-z_]\w*', name):
        raise CumlxError('E_USAGE', f'the file name {p.name} is not a valid kernel name')
    sha = sha256_text(raw)
    record = HeaderRecord()
    refusals: list[Refusal] = []
    warnings: list[dict[str, Any]] = []
    dirs = [Path(d) for d in include_dirs or []]
    for d in dirs:
        if not d.is_dir():
            raise CumlxError('E_INPUT_NOT_FOUND', f'--include-dir {d} does not exist')
    flat = flatten_includes(raw, p.parent, dirs, record, refusals)
    wrap: KernelCase | None = None
    if bool(cases_file) != bool(case_id):
        raise CumlxError('E_USAGE', '--cases and --case go together')
    if cases_file:
        matches = [c for c, _ in read_cases(cases_file) if c.id == case_id]
        if not matches:
            raise CumlxError('E_USAGE', f'{cases_file} has no case {case_id!r}')
        wrap = matches[0]
    if re.search(r'\b(__global__|__device__)\b', mask_source(flat)):
        if wrap is not None:
            raise CumlxError('E_USAGE', '--case wraps a function body, and this source is already a complete kernel')
        out = preprocess_complete(flat, name=name, sha=sha, record=record, refusals=refusals, instantiate=parse_instantiate(instantiate or []),
                                  demote_fp64=demote_fp64, warnings=warnings, allow_warp_level=allow_warp_level)
    else:
        if instantiate:
            raise CumlxError('E_USAGE', '--instantiate needs a complete kernel, and this source is a function body')
        if not re.search(r'\b(threadIdx|blockIdx)\b', mask_source(flat)):
            refusals.append(Refusal('no-entry-point', name, 'The source has no __global__ function and does not read threadIdx or blockIdx, so it is not a kernel.'))
        header_path = Path(header) if header else p.parent / 'common_cuda.h'
        if header and not header_path.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'--header {header} does not exist')
        header_text = flatten_includes(header_path.read_text(errors='replace'), header_path.parent, dirs, record, refusals) if header_path.is_file() else ''
        out = _preprocess_body(flat, header_text, name=name, sha=sha, record=record, refusals=refusals, demote_fp64=demote_fp64, wrap=wrap, warnings=warnings, allow_warp_level=allow_warp_level)
    out.record['headers'] = asdict(record)
    if record.unknown_dropped:
        out.warnings.append({'code': 'W_KERNEL_HEADER_DROPPED', 'message': f'These unknown headers were dropped: {record.unknown_dropped}. If a kernel needs one, the translator ends with an error.',
                             'headers': record.unknown_dropped})
    literals = len(FP64_LITERAL.findall(mask_source(out.source + out.header)))
    if literals:
        out.warnings.append({'code': 'W_KERNEL_FP64_LITERAL', 'message': f'{literals} unsuffixed floating literal(s) in device code: double on CUDA and float on Metal. Values can differ.', 'count': literals})
    passed = sorted(token for token in set(IDENT.findall(mask_source(out.source + out.header))) if token in WARP_LEVEL)
    if allow_warp_level and passed:
        out.warnings.append({'code': 'W_KERNEL_WARP_LEVEL_PASSED', 'message': f'--allow-warp-level passed {passed} to the translator. Metal does not promise a SIMD group of 32 or a lane equal to threadIdx.x % 32, so only `cumlx kernel verify` can say whether it works, on this device only.', 'tokens': passed})
    if demote_fp64:
        out.warnings.append({'code': 'W_KERNEL_FP64_DEMOTED', 'message': '--demote-fp64 turned double into float. The port no longer computes what the original computes in fp64.'})
    return out


# ---------------------------------------------------------------------------- translators
@dataclass(frozen=True)
class TranslationRequest:
    name: str
    source: str
    form: str
    header: str
    entry_points: tuple[str, ...]
    include_dirs: tuple[str, ...] = ()


@dataclass
class TranslationResult:
    """``msl`` is a complete Metal library, or a function body when ``form`` is ``body``. ``header`` goes with a body."""
    msl: str
    form: str
    translator: str
    version: str | None = None
    header: str = ''
    entry_points: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    signatures: dict[str, Any] = field(default_factory=dict)  # for an engine that binds the kernel with the translator's own launcher


class TranslatorAdapter(Protocol):
    """What CUMLX needs from a translator. ``unavailable`` returns None when it can run, or else the reason it cannot."""
    name: str
    accepts_body: bool

    def unavailable(self) -> str | None: ...

    def translate(self, request: TranslationRequest) -> TranslationResult: ...


class WoofAdapter:
    """The woof translator (`pip install metal-woof`), which the user installs. CUMLX neither ships nor installs it.

    The calls follow spikes/probes/probe_woof.py, which ran against metal-woof 1.1.5.2. This adapter itself has been tested
    only with a stub package that has the same names, not with woof.
    """
    name = 'woof'
    accepts_body = False

    def unavailable(self) -> str | None:
        if importlib.util.find_spec('woof') is None:
            return 'the Python package woof is not installed'
        return None

    def translate(self, request: TranslationRequest) -> TranslationResult:
        import os
        os.environ.setdefault('WOOF_BACKEND', 'mlx')
        importlib.import_module('woof')
        translator = importlib.import_module('woof.backend.translate')
        result = translator.translate(request.source, unit=request.name, include_dirs=request.include_dirs)
        try:
            from importlib import metadata
            version = metadata.version('metal-woof')
        except metadata.PackageNotFoundError:
            version = None
        signatures = {name: asdict(s) for name, s in result.kernels.items() if is_dataclass(s) and not isinstance(s, type)}
        notes = ['the unit has module globals, which `--engine woof` does not bind'] if getattr(result, 'globals', None) else []
        return TranslationResult(result.msl, 'complete', self.name, version, entry_points=sorted(result.kernels), notes=notes, signatures=signatures)


def load_translator(spec: str) -> TranslatorAdapter:
    if spec == 'woof':
        return WoofAdapter()
    if spec.startswith('python:'):
        parts = spec[len('python:'):].split(':')
        if len(parts) != 2:
            raise CumlxError('E_USAGE', f'--translator python: needs python:MODULE:ATTRIBUTE, got {spec!r}')
        try:
            adapter = getattr(importlib.import_module(parts[0]), parts[1])
        except (ImportError, AttributeError) as exc:
            raise CumlxError('E_KERNEL_TRANSLATOR_MISSING', f'could not load translator {spec}: {exc}') from None
        adapter = adapter() if callable(adapter) else adapter
        for attribute in ('name', 'accepts_body', 'unavailable', 'translate'):
            if not hasattr(adapter, attribute):
                raise CumlxError('E_USAGE', f'{spec} is not a translator adapter: it has no {attribute}')
        return adapter
    raise CumlxError('E_USAGE', f'unknown translator {spec!r}', hint='use woof or python:MODULE:ATTRIBUTE')


def translate(pre: Preprocessed, adapter: TranslatorAdapter, include_dirs: list[str] | None = None) -> TranslationResult:
    reason = adapter.unavailable()
    if reason:
        raise CumlxError('E_KERNEL_TRANSLATOR_MISSING', f'translator {adapter.name} cannot run: {reason}. Nothing was translated, and this is not a pass.',
                         hint="install it yourself if you accept its terms (woof: python -m pip install metal-woof), or use --preprocess-only",
                         details={'translator': adapter.name})
    if pre.form == 'body' and not adapter.accepts_body:
        raise CumlxError('E_KERNEL_TRANSLATE_FAILED', f'translator {adapter.name} needs a complete kernel, and {pre.name} is a function body.',
                         hint='give launch constants with --cases cases.json --case ID, which wraps the body in a complete kernel')
    request = TranslationRequest(pre.name, pre.source, pre.form, pre.header, tuple(pre.entry_points), tuple(include_dirs or ()))
    try:
        result = adapter.translate(request)
    except CumlxError:
        raise
    except Exception as exc:  # noqa: BLE001 - a translator failure is a result, and its message is the reason
        raise CumlxError('E_KERNEL_TRANSLATE_FAILED', f'translator {adapter.name} failed on {pre.name}: {type(exc).__name__}: {exc}'[:1500],
                         details={'translator': adapter.name}) from None
    if not isinstance(result, TranslationResult) or not (result.msl or '').strip():
        raise CumlxError('E_KERNEL_TRANSLATE_FAILED', f'translator {adapter.name} returned no Metal source for {pre.name}.')
    if result.form not in ('complete', 'body'):
        raise CumlxError('E_KERNEL_TRANSLATE_FAILED', f'translator {adapter.name} returned an unknown form {result.form!r}.')
    if result.form == 'complete':
        missing = sorted(set(pre.entry_points) - set(result.entry_points))
        if missing:
            raise CumlxError('E_KERNEL_TRANSLATE_FAILED', f'translator {adapter.name} left out {missing}, which the source defines.',
                             details={'translated': result.entry_points})
    return result


def builtin_table() -> dict[str, dict[str, str]]:
    """The PTX built-in map, for `describe`-like tools and documentation."""
    return {k: {f: v for f, v in asdict(b).items() if v} for k, b in sorted(PTX_BUILTINS.items())}
