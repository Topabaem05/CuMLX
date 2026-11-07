"""CUDA ``__nv_bfloat16`` on Metal, bit for bit, and a NumPy model of what CUDA computes.

Every Apple Silicon Mac has the MSL type ``bfloat`` (Apple6 and later, MSL 3.1, macOS 14). It is not used here: the MSL
specification allows round-toward-zero and flush-to-zero for it, and on an M5 Pro its conversions flush subnormals. Metal
float32 arithmetic also flushes subnormal inputs and results (measured 2026-11-07 in math modes safe and fast), so an
emulation through float arithmetic differs from CUDA on subnormal values too.

So this header stores a value in 16 bits and does every operation in 32-bit integer arithmetic. What it computes:

- The conversions from float are those of ``cuda_bf16.hpp`` (CUDA 12.0) for sm70: round to nearest even, toward zero,
  down and up, with subnormals kept, and every NaN becomes 0x7FFF.
- ``__hadd``, ``__hsub``, ``__hmul`` and ``__hfma`` are those of sm80 and later, where CUDA 12.0 computes them with
  ``fma.rn.bf16``: the exact result, rounded once to nearest even, with subnormals kept (PTX ISA, half precision fma).
  CUDA 12.0 has no bf16 arithmetic on sm70. A NaN result is 0x7FFF here; PTX does not specify the NaN that the GPU gives,
  so `cumlx kernel verify` compares NaN by class.
- ``__hneg`` and ``__habs`` change the sign bit. PTX says that a NaN input gives an unspecified NaN.
- Division is not mapped: ``operator/`` is deleted, so a kernel that divides two bfloat16 values does not compile.

The conversions were compared with ``cuda_bf16.hpp`` on every float32 bit pattern, and the arithmetic with the exact model
below on every pair of bfloat16 values (scripts/bf16_exhaustive.py). The model is not a CUDA run.
"""
from __future__ import annotations

import re

import numpy as np

HEADER_NAME = 'cumlx_bf16.h'
TYPE = 'cumlx_bf16'
CANONICAL_NAN = 0x7FFF
# CUDA name -> the name in the prepared source and in the Metal header.
INTRINSICS = {
    '__float2bfloat16': 'cumlx_float2bfloat16_rn', '__float2bfloat16_rn': 'cumlx_float2bfloat16_rn',
    '__float2bfloat16_rz': 'cumlx_float2bfloat16_rz', '__float2bfloat16_rd': 'cumlx_float2bfloat16_rd',
    '__float2bfloat16_ru': 'cumlx_float2bfloat16_ru', '__bfloat162float': 'cumlx_bfloat162float',
    '__bfloat16_as_ushort': 'cumlx_bfloat16_as_ushort', '__ushort_as_bfloat16': 'cumlx_ushort_as_bfloat16',
    '__hsub': 'cumlx_hsub', '__hmul': 'cumlx_hmul', '__hfma': 'cumlx_hfma', '__hneg': 'cumlx_hneg', '__habs': 'cumlx_habs'}
# __hadd keeps its existing name: cumlx_hadd(int, int) is the integer halving add, and this header adds the bf16 overload.
NAMES = frozenset({TYPE, 'cumlx_hadd', *INTRINSICS.values()})

_METAL = r'''
// The bits of a float, rounded to bf16 as cuda_bf16.hpp (CUDA 12.0) does on sm70. Every NaN becomes 0x7FFF.
inline ushort cumlx_bf16_rn_bits(uint x) {
    if ((x & 0x7fffffffu) > 0x7f800000u) return ushort(0x7fffu);
    uint r = x << 16u; uint h = x >> 16u;
    if (r > 0x80000000u || (r == 0x80000000u && (h & 1u) != 0u)) h += 1u;
    return ushort(h);
}
inline ushort cumlx_bf16_rz_bits(uint x) { return (x & 0x7fffffffu) > 0x7f800000u ? ushort(0x7fffu) : ushort(x >> 16u); }
inline ushort cumlx_bf16_rd_bits(uint x) {
    if ((x & 0x7fffffffu) > 0x7f800000u) return ushort(0x7fffu);
    uint h = x >> 16u;
    if ((x << 16u) != 0u && (x >> 31u) != 0u) h += 1u;
    return ushort(h);
}
inline ushort cumlx_bf16_ru_bits(uint x) {
    if ((x & 0x7fffffffu) > 0x7f800000u) return ushort(0x7fffu);
    uint h = x >> 16u;
    if ((x << 16u) != 0u && (x >> 31u) == 0u) h += 1u;
    return ushort(h);
}
// (-1)^s * m * 2^e with m < 2^31, rounded once to the nearest even bf16. Subnormals are kept; an overflow is infinity.
inline ushort cumlx_bf16_round(uint s, uint m, int e) {
    if (m == 0u) return ushort(s << 15u);
    int lead = 31 - int(clz(m));
    int g = max(e + lead - 7, -133);
    int k = g - e;
    uint q;
    if (k <= 0) {
        q = m << uint(-k);
    } else if (k >= 32) {
        q = 0u;
    } else {
        q = m >> uint(k);
        uint r = m & ((1u << uint(k)) - 1u);
        uint half_ = 1u << uint(k - 1);
        if (r > half_ || (r == half_ && (q & 1u) != 0u)) q += 1u;
    }
    if (q == 0u) return ushort(s << 15u);
    int l2 = 31 - int(clz(q));
    int ex = g + l2;
    if (ex > 127) return ushort((s << 15u) | 0x7f80u);
    if (ex < -126) return ushort((s << 15u) | (q << uint(g + 133)));
    uint mant = l2 >= 7 ? (q >> uint(l2 - 7)) : (q << uint(7 - l2));
    return ushort((s << 15u) | (uint(ex + 127) << 7u) | (mant & 0x7fu));
}
inline void cumlx_bf16_decode(uint b, thread uint& s, thread uint& m, thread int& e) {
    s = (b >> 15u) & 1u;
    uint ex = (b >> 7u) & 0xffu;
    uint f = b & 0x7fu;
    if (ex == 0u) { m = f; e = -133; } else { m = f | 0x80u; e = int(ex) - 134; }
}
// The exact sum of two finite values, rounded once. Each m has at most 16 significant bits.
inline ushort cumlx_bf16_sum(uint sx, uint mx, int ex, uint sy, uint my, int ey) {
    if (mx == 0u && my == 0u) return ushort((sx & sy) << 15u);
    if (mx == 0u) return cumlx_bf16_round(sy, my, ey);
    if (my == 0u) return cumlx_bf16_round(sx, mx, ex);
    int lx = 31 - int(clz(mx)), ly = 31 - int(clz(my));
    uint X = mx << uint(29 - lx); int Ex = ex - (29 - lx);
    uint Y = my << uint(29 - ly); int Ey = ey - (29 - ly);
    if (Ey > Ex || (Ey == Ex && Y > X)) {
        uint ts = sx; sx = sy; sy = ts;
        uint tm = X; X = Y; Y = tm;
        int te = Ex; Ex = Ey; Ey = te;
    }
    int d = Ex - Ey;
    // Bits 0..13 of Y are zero, so a shift of up to 14 is exact. A longer one keeps a sticky bit in bit 0, far below
    // the rounding position, which is then at bit 21 or higher.
    uint A = d == 0 ? Y : d < 30 ? ((Y >> uint(d)) | ((Y & ((1u << uint(d)) - 1u)) != 0u ? 1u : 0u)) : 1u;
    uint S = sx == sy ? X + A : X - A;
    if (S == 0u) return ushort(0u);
    return cumlx_bf16_round(sx, S, Ex);
}
inline bool cumlx_bf16_isnan_bits(uint b) { return (b & 0x7fffu) > 0x7f80u; }
// a*b + c with no intermediate rounding, rounded once: fma.rn.bf16.
inline ushort cumlx_bf16_fma_bits(uint a, uint b, uint c) {
    if (cumlx_bf16_isnan_bits(a) || cumlx_bf16_isnan_bits(b) || cumlx_bf16_isnan_bits(c)) return ushort(0x7fffu);
    uint sp = ((a ^ b) >> 15u) & 1u;
    bool ai = (a & 0x7fffu) == 0x7f80u, bi = (b & 0x7fffu) == 0x7f80u, ci = (c & 0x7fffu) == 0x7f80u;
    if (ai || bi) {
        if ((a & 0x7fffu) == 0u || (b & 0x7fffu) == 0u) return ushort(0x7fffu);
        if (ci && ((c >> 15u) & 1u) != sp) return ushort(0x7fffu);
        return ushort((sp << 15u) | 0x7f80u);
    }
    if (ci) return ushort(c);
    uint s1, m1, s2, m2, s3, m3; int e1, e2, e3;
    cumlx_bf16_decode(a, s1, m1, e1);
    cumlx_bf16_decode(b, s2, m2, e2);
    cumlx_bf16_decode(c, s3, m3, e3);
    return cumlx_bf16_sum(sp, m1 * m2, e1 + e2, s3, m3, e3);
}
inline int cumlx_bf16_order(uint b) { return (b & 0x8000u) != 0u ? -int(b & 0x7fffu) : int(b & 0x7fffu); }

struct cumlx_bf16 {
    ushort x;
    cumlx_bf16() thread = default;
    cumlx_bf16() threadgroup = default;
    cumlx_bf16() device = default;
    cumlx_bf16(float f) thread : x(cumlx_bf16_rn_bits(as_type<uint>(f))) {}
    cumlx_bf16(float f) threadgroup : x(cumlx_bf16_rn_bits(as_type<uint>(f))) {}
    cumlx_bf16(float f) device : x(cumlx_bf16_rn_bits(as_type<uint>(f))) {}
    operator float() const thread { return as_type<float>(uint(x) << 16u); }
    operator float() const threadgroup { return as_type<float>(uint(x) << 16u); }
    operator float() const device { return as_type<float>(uint(x) << 16u); }
    operator float() const constant { return as_type<float>(uint(x) << 16u); }
};
inline cumlx_bf16 cumlx_ushort_as_bfloat16(ushort b) { cumlx_bf16 r; r.x = b; return r; }
inline ushort cumlx_bfloat16_as_ushort(cumlx_bf16 h) { return h.x; }
inline cumlx_bf16 cumlx_float2bfloat16_rn(float f) { return cumlx_ushort_as_bfloat16(cumlx_bf16_rn_bits(as_type<uint>(f))); }
inline cumlx_bf16 cumlx_float2bfloat16_rz(float f) { return cumlx_ushort_as_bfloat16(cumlx_bf16_rz_bits(as_type<uint>(f))); }
inline cumlx_bf16 cumlx_float2bfloat16_rd(float f) { return cumlx_ushort_as_bfloat16(cumlx_bf16_rd_bits(as_type<uint>(f))); }
inline cumlx_bf16 cumlx_float2bfloat16_ru(float f) { return cumlx_ushort_as_bfloat16(cumlx_bf16_ru_bits(as_type<uint>(f))); }
inline float cumlx_bfloat162float(cumlx_bf16 h) { return as_type<float>(uint(h.x) << 16u); }
inline cumlx_bf16 cumlx_hfma(cumlx_bf16 a, cumlx_bf16 b, cumlx_bf16 c) { return cumlx_ushort_as_bfloat16(cumlx_bf16_fma_bits(a.x, b.x, c.x)); }
inline cumlx_bf16 cumlx_hadd(cumlx_bf16 a, cumlx_bf16 b) { return cumlx_ushort_as_bfloat16(cumlx_bf16_fma_bits(a.x, 0x3f80u, b.x)); }
inline cumlx_bf16 cumlx_hsub(cumlx_bf16 a, cumlx_bf16 b) { return cumlx_ushort_as_bfloat16(cumlx_bf16_fma_bits(b.x, 0xbf80u, a.x)); }
inline cumlx_bf16 cumlx_hmul(cumlx_bf16 a, cumlx_bf16 b) { return cumlx_ushort_as_bfloat16(cumlx_bf16_fma_bits(a.x, b.x, 0x8000u)); }
inline cumlx_bf16 cumlx_hneg(cumlx_bf16 a) { return cumlx_ushort_as_bfloat16(ushort(a.x ^ 0x8000u)); }
inline cumlx_bf16 cumlx_habs(cumlx_bf16 a) { return cumlx_ushort_as_bfloat16(ushort(a.x & 0x7fffu)); }
inline cumlx_bf16 operator+(cumlx_bf16 a, cumlx_bf16 b) { return cumlx_hadd(a, b); }
inline cumlx_bf16 operator-(cumlx_bf16 a, cumlx_bf16 b) { return cumlx_hsub(a, b); }
inline cumlx_bf16 operator*(cumlx_bf16 a, cumlx_bf16 b) { return cumlx_hmul(a, b); }
cumlx_bf16 operator/(cumlx_bf16 a, cumlx_bf16 b) = delete;  // CUDA's __hdiv is not mapped
inline cumlx_bf16 operator-(cumlx_bf16 a) { return cumlx_hneg(a); }
inline cumlx_bf16 operator+(cumlx_bf16 a) { return a; }
#define CUMLX_BF16_ASSIGN(SPACE) \
    inline SPACE cumlx_bf16& operator+=(SPACE cumlx_bf16& a, cumlx_bf16 b) { a.x = cumlx_hadd(a, b).x; return a; } \
    inline SPACE cumlx_bf16& operator-=(SPACE cumlx_bf16& a, cumlx_bf16 b) { a.x = cumlx_hsub(a, b).x; return a; } \
    inline SPACE cumlx_bf16& operator*=(SPACE cumlx_bf16& a, cumlx_bf16 b) { a.x = cumlx_hmul(a, b).x; return a; }
CUMLX_BF16_ASSIGN(thread)
CUMLX_BF16_ASSIGN(threadgroup)
CUMLX_BF16_ASSIGN(device)
#undef CUMLX_BF16_ASSIGN
inline bool operator==(cumlx_bf16 a, cumlx_bf16 b) { return !cumlx_bf16_isnan_bits(a.x) && !cumlx_bf16_isnan_bits(b.x) && cumlx_bf16_order(a.x) == cumlx_bf16_order(b.x); }
inline bool operator!=(cumlx_bf16 a, cumlx_bf16 b) { return !(a == b); }
inline bool operator<(cumlx_bf16 a, cumlx_bf16 b) { return !cumlx_bf16_isnan_bits(a.x) && !cumlx_bf16_isnan_bits(b.x) && cumlx_bf16_order(a.x) < cumlx_bf16_order(b.x); }
inline bool operator>(cumlx_bf16 a, cumlx_bf16 b) { return b < a; }
inline bool operator<=(cumlx_bf16 a, cumlx_bf16 b) { return a < b || a == b; }
inline bool operator>=(cumlx_bf16 a, cumlx_bf16 b) { return b < a || a == b; }
'''

_CUDA = '''
#include <cuda_bf16.h>
typedef __nv_bfloat16 cumlx_bf16;
__device__ __forceinline__ cumlx_bf16 cumlx_float2bfloat16_rn(float f) { return __float2bfloat16_rn(f); }
__device__ __forceinline__ cumlx_bf16 cumlx_float2bfloat16_rz(float f) { return __float2bfloat16_rz(f); }
__device__ __forceinline__ cumlx_bf16 cumlx_float2bfloat16_rd(float f) { return __float2bfloat16_rd(f); }
__device__ __forceinline__ cumlx_bf16 cumlx_float2bfloat16_ru(float f) { return __float2bfloat16_ru(f); }
__device__ __forceinline__ float cumlx_bfloat162float(cumlx_bf16 h) { return __bfloat162float(h); }
__device__ __forceinline__ unsigned short cumlx_bfloat16_as_ushort(cumlx_bf16 h) { return __bfloat16_as_ushort(h); }
__device__ __forceinline__ cumlx_bf16 cumlx_ushort_as_bfloat16(unsigned short b) { return __ushort_as_bfloat16(b); }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ cumlx_bf16 cumlx_hadd(cumlx_bf16 a, cumlx_bf16 b) { return __hadd(a, b); }
__device__ __forceinline__ cumlx_bf16 cumlx_hsub(cumlx_bf16 a, cumlx_bf16 b) { return __hsub(a, b); }
__device__ __forceinline__ cumlx_bf16 cumlx_hmul(cumlx_bf16 a, cumlx_bf16 b) { return __hmul(a, b); }
__device__ __forceinline__ cumlx_bf16 cumlx_hfma(cumlx_bf16 a, cumlx_bf16 b, cumlx_bf16 c) { return __hfma(a, b, c); }
__device__ __forceinline__ cumlx_bf16 cumlx_hneg(cumlx_bf16 a) { return __hneg(a); }
__device__ __forceinline__ cumlx_bf16 cumlx_habs(cumlx_bf16 a) { return __habs(a); }
#endif
'''

METAL_HEADER = f'#ifndef CUMLX_BF16_H\n#define CUMLX_BF16_H\n{_METAL}#endif\n'
HEADER = (f'// {HEADER_NAME}: CUDA __nv_bfloat16 for the source that `cumlx kernel port` prepared. On Metal it computes what CUDA\n'
          '// computes, bit for bit, in integer arithmetic. With nvcc it forwards to cuda_bf16.h. Written by CUMLX.\n'
          f'#ifndef CUMLX_BF16_H\n#define CUMLX_BF16_H\n#ifdef __METAL_VERSION__\n{_METAL}#else\n{_CUDA}#endif\n#endif\n')
_INCLUDE_LINE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*["<]cumlx_bf16\.h[">][^\n]*\n?', re.M)
_OWN_DEFINITION = re.compile(r'\bstruct\s+cumlx_bf16\b')


def uses_bf16(tokens: set[str]) -> bool:
    return TYPE in tokens


def with_metal_header(source: str, header: str, tokens: set[str]) -> tuple[str, str, bool]:
    """Put the Metal header in front of ``header`` when the kernel names ``cumlx_bf16`` and does not define it itself."""
    if not uses_bf16(tokens) or _OWN_DEFINITION.search(source + header):
        return source, header, False
    return _INCLUDE_LINE.sub('', source), METAL_HEADER + _INCLUDE_LINE.sub('', header), True


# ---------------------------------------------------------------------------- the model, in NumPy
def bits_to_float(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _wide(bits: np.ndarray) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        return bits_to_float(bits).astype(np.float64)


def is_nan(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, np.uint16) & np.uint16(0x7FFF)) > np.uint16(0x7F80)


def from_float_bits(x: np.ndarray, mode: str = 'rn') -> np.ndarray:
    """``__float2bfloat16_{mode}`` on the bits of float32 values, as cuda_bf16.hpp (CUDA 12.0) computes it on sm70."""
    x = np.asarray(x, np.uint32)
    h = (x >> np.uint32(16)).astype(np.uint32)
    r = x << np.uint32(16)
    sign = (x >> np.uint32(31)) != 0
    if mode == 'rn':
        h = h + ((r > 0x80000000) | ((r == 0x80000000) & ((h & 1) != 0)))
    elif mode == 'rd':
        h = h + ((r != 0) & sign)
    elif mode == 'ru':
        h = h + ((r != 0) & ~sign)
    elif mode != 'rz':
        raise ValueError(mode)
    return np.where((x & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000), np.uint32(CANONICAL_NAN), h).astype(np.uint16)


def round_float64(v: np.ndarray) -> np.ndarray:
    """Round float64 values to the nearest even bf16, with subnormals kept and NaN as 0x7FFF.

    This equals one rounding of the exact result when ``v`` is the float64 result of one add, subtract or multiply of
    two bf16 values: 53 >= 2 * 8 + 2, so the float64 step does not change the second rounding (Figueroa, 1995). A float64
    multiply of two bf16 values is exact, and so is the float64 sum of two values on the grid of 2^-133 that are below
    2^-80. The algorithm is independent of the Metal header: it works on float64 values with frexp and rint.
    """
    v = np.asarray(v, np.float64)
    out = np.zeros(v.shape, np.uint32)
    sign = np.signbit(v).astype(np.uint32) << np.uint32(15)
    a = np.abs(v)
    nan = np.isnan(v)
    with np.errstate(invalid='ignore', over='ignore'):
        m, e = np.frexp(a)  # a = m * 2^e, 0.5 <= m < 1
        small = a < 2.0 ** -126
        n_sub = np.rint(a * 2.0 ** 133)  # the bits of a subnormal, and 128 = the smallest normal
        n = np.rint(m * 256.0)  # 8 significant bits, from 128 to 256
        carry = n == 256
        n = np.where(carry, 128, n)
        e = np.where(carry, e + 1, e)
        exponent = (e - 1) + 127  # m * 256 = n means a = n * 2^(e - 8) = (n / 128) * 2^(e - 1)
        normal_bits = (exponent.astype(np.int64) << 7) | (n.astype(np.int64) & 0x7F)
        bits = np.where(small, n_sub, normal_bits)
        bits = np.where(~small & (exponent >= 255), 0x7F80, bits)
        bits = np.where(np.isinf(a), 0x7F80, bits)
    out = (np.nan_to_num(bits).astype(np.uint32) | sign).astype(np.uint32)
    out = np.where(nan, np.uint32(CANONICAL_NAN), out)
    return out.astype(np.uint16)


def model_op(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``__hadd``, ``__hsub`` or ``__hmul`` of sm80 and later on bf16 bit patterns: the exact result, rounded once."""
    fa, fb = _wide(a), _wide(b)
    with np.errstate(invalid='ignore', over='ignore'):
        v = {'add': fa + fb, 'sub': fa - fb, 'mul': fa * fb}[op]
    return round_float64(v)


def model_fma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``__hfma``: a*b + c, rounded once. Exact integer arithmetic in Python, so it is slow: use it on samples."""
    from fractions import Fraction
    out = np.empty(np.shape(a), np.uint16)
    fa, fb, fc = (_wide(x) for x in (a, b, c))
    for i, (x, y, z) in enumerate(zip(fa.ravel(), fb.ravel(), fc.ravel())):
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            out.flat[i] = CANONICAL_NAN
            continue
        if np.isinf(x) or np.isinf(y) or np.isinf(z):
            with np.errstate(invalid='ignore'):
                out.flat[i] = round_float64(np.array([x * y + z]))[0]
            continue
        exact = Fraction(x) * Fraction(y) + Fraction(z)
        if exact == 0:
            negative = (np.signbit(x) != np.signbit(y)) and np.signbit(z) and (x * y == 0)
            out.flat[i] = 0x8000 if negative else 0
            continue
        out.flat[i] = _round_fraction(exact)
    return out


def _round_fraction(v) -> int:
    from fractions import Fraction
    sign = 0x8000 if v < 0 else 0
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    g = max(e - 7, -133)  # the spacing of bf16 at this size
    q = a / Fraction(2) ** g
    n = q.numerator // q.denominator
    rest = q - n
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and n % 2 == 1):
        n += 1
    if n == 0:
        return sign
    value = Fraction(n) * Fraction(2) ** g
    if value >= Fraction(2) ** 128:
        return sign | 0x7F80
    if value < Fraction(2) ** -126:
        return sign | n << (g + 133)
    e2 = n.bit_length() - 1 + g
    mant = (n << (7 - (n.bit_length() - 1))) if n.bit_length() <= 8 else n >> (n.bit_length() - 8)
    return sign | (e2 + 127) << 7 | (mant & 0x7F)


def compare_bits(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Compare a bfloat16 output with its golden, as bit patterns. Every element that is not NaN in the golden must have
    the same 16 bits. A NaN must be a NaN; its payload may differ, because PTX does not specify the NaN of an operation."""
    a, e = np.asarray(actual), np.asarray(expected)
    out: dict = {'shape': list(a.shape), 'expected_shape': list(e.shape), 'dtype': 'bfloat16', 'basis': 'bits',
                 'rel_l2': None, 'max_rel': None, 'max_abs': None, 'max_ulp': None}
    if a.shape != e.shape:
        return {**out, 'verdict': 'fail', 'reason': 'shape_mismatch'}
    if a.dtype != np.uint16 or e.dtype != np.uint16:
        return {**out, 'verdict': 'fail', 'reason': 'dtype_mismatch', 'actual_dtype': str(a.dtype)}
    a_nan, e_nan = is_nan(a), is_nan(e)
    out['nan_in_golden'] = int(e_nan.sum())
    out['nan_payload_differs'] = int((a_nan & e_nan & (a != e)).sum())
    wrong_nan = a_nan != e_nan
    if wrong_nan.any():
        at = tuple(int(i) for i in np.argwhere(wrong_nan)[0])
        return {**out, 'verdict': 'fail', 'reason': 'nan_mismatch', 'mismatched': int(wrong_nan.sum()), 'index': list(at),
                'golden': f'0x{int(e[at]):04x}', 'port': f'0x{int(a[at]):04x}'}
    wrong = (a != e) & ~e_nan
    out['mismatched'] = int(wrong.sum())
    finite = ~e_nan
    wide_a, wide_e = _wide(a), _wide(e)
    with np.errstate(invalid='ignore'):
        both = finite & np.isfinite(wide_a) & np.isfinite(wide_e)
        diff = np.where(both, wide_a - wide_e, 0.0)
        norm = float(np.linalg.norm(np.where(both, wide_e, 0.0)))
    out['max_abs'] = float(np.max(np.abs(diff))) if diff.size else 0.0
    out['rel_l2'] = float(np.linalg.norm(diff) / norm) if norm else None
    order = lambda x: np.where(x & 0x8000, -(x & 0x7FFF).astype(np.int32), (x & 0x7FFF).astype(np.int32))  # noqa: E731
    out['max_ulp'] = int(np.max(np.abs(order(a) - order(e))[finite])) if finite.any() else 0
    if out['mismatched']:
        at = tuple(int(i) for i in np.argwhere(wrong)[0])
        return {**out, 'verdict': 'fail', 'reason': 'bits_differ', 'index': list(at), 'golden': float(wide_e[at]), 'port': float(wide_a[at]),
                'golden_bits': f'0x{int(e[at]):04x}', 'port_bits': f'0x{int(a[at]):04x}'}
    return {**out, 'verdict': 'pass', 'basis': 'bit_exact'}
