"""Write the test vectors that prove the Rust metrics equal the Python metrics.

``rust/cumlx/tests/data/python-vectors.json`` holds inputs and what ``src/cumlx/verify/metrics.py`` and
``src/cumlx/verify/weights.py`` return for them. ``cargo test`` reads the file and requires the Rust functions
to give the same keys, the same strings and integers, and floats within 1e-12 relative.
``tests/test_rust_parity.py`` regenerates the file and fails when the committed copy is stale.

The inputs come from integer arithmetic, not from a random generator, so they are the same on every NumPy.

    python scripts/rust_vectors.py            # rewrite the file
    python scripts/rust_vectors.py --check    # exit 1 when the file is stale
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from cumlx.cli.output import _sanitize  # noqa: E402
from cumlx.errors import ERROR_CODES, EXIT_CODE_DOCS  # noqa: E402
from cumlx.verify import metrics, weights  # noqa: E402

OUT = ROOT / 'rust' / 'cumlx' / 'tests' / 'data' / 'python-vectors.json'
SCHEMA = 'cumlx.rust-vectors.v1'
REL_TOL = 1e-12
TAGS = {'float64': 'F64', 'float32': 'F32', 'float16': 'F16', 'int64': 'I64', 'int32': 'I32', 'int8': 'I8',
        'uint8': 'U8', 'bool': 'BOOL'}


def values(n: int, seed: int, dtype=np.float32, scale: float = 4.0) -> np.ndarray:
    """n values in [-scale/2, scale/2): the top 32 bits of SplitMix64 of (seed, index). Integer arithmetic only."""
    golden = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over='ignore'):
        z = np.arange(n, dtype=np.uint64) + np.uint64(seed) * golden + golden
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    k = z >> np.uint64(32)
    return ((k.astype(np.float64) / 2.0**32 - 0.5) * scale).astype(dtype)


def enc(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a)
    if not a.flags.c_contiguous:  # np.ascontiguousarray would turn 0-d into 1-d
        a = a.copy(order='C')
    return {'dtype': TAGS[str(a.dtype)], 'shape': list(a.shape), 'hex': a.astype(a.dtype.newbyteorder('<')).tobytes().hex()}


def clean(obj: Any) -> Any:
    return _sanitize(obj)


def compare_cases() -> list[dict]:
    x = values(48, 1).reshape(4, 12)
    f32 = metrics.resolve_threshold('float32')
    bf = metrics.resolve_threshold('bfloat16')
    limits = {'rel_l2': f32['rel_l2'], 'zero_atol': f32['zero_atol']}
    loose = {'rel_l2': bf['rel_l2'], 'zero_atol': bf['zero_atol']}
    big = values(4096, 2).reshape(64, 64)
    nan_out = x.copy()
    nan_out[1, 3] = np.nan
    nan_out[2, 0] = np.inf
    inf_golden = x.copy()
    inf_golden[0, 0] = -np.inf
    cases = [
        ('identical', x, x, limits),
        ('one_percent', (x * np.float32(1.01)).astype(np.float32), x, limits),
        ('noise_below', (x + values(48, 3, scale=1e-5).reshape(4, 12)).astype(np.float32), x, limits),
        ('noise_above', (x + values(48, 3, scale=1e-3).reshape(4, 12)).astype(np.float32), x, limits),
        ('noise_bf16_limit', (x + values(48, 3, scale=1e-2).reshape(4, 12)).astype(np.float32), x, loose),
        ('reshaped', x.reshape(48), x, limits),
        ('reshaped_wrong', x[::-1].reshape(2, 24), x, limits),
        ('shape_mismatch', x[:, :6], x, limits),
        ('nonfinite_output', nan_out, x, limits),
        ('nonfinite_golden', x, inf_golden, limits),
        ('zero_golden_pass', np.full((3, 2), 1e-7, np.float32), np.zeros((3, 2), np.float32), limits),
        ('zero_golden_fail', np.full((3, 2), 1e-3, np.float32), np.zeros((3, 2), np.float32), limits),
        ('both_zero', np.zeros((5,), np.float32), np.zeros((5,), np.float32), limits),
        ('empty', np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32), limits),
        ('empty_vs_other', np.zeros((0,), np.float32), np.zeros((0, 4), np.float32), limits),
        ('scalar', np.array(2.5, np.float32), np.array(2.5, np.float32), limits),
        ('float16', x.astype(np.float16), x, loose),
        ('float64', x.astype(np.float64) * 1.00001, x.astype(np.float64), limits),
        ('int64', np.arange(12, dtype=np.int64), np.arange(12, dtype=np.int64)[::-1].copy(), limits),
        ('bool', np.array([True, False, True]), np.array([True, True, True]), limits),
        ('negated', -x, x, limits),
        ('large', (big + values(4096, 4, scale=1e-6).reshape(64, 64)).astype(np.float32), big, limits),
    ]
    out = []
    for name, a, e, lim in cases:
        out.append({'name': name, 'actual': enc(a), 'expected': enc(e), 'limits': lim,
                    'result': clean(metrics.compare(a, e, lim))})
    return out


def kernel_cases() -> list[dict]:
    g = values(30, 5).reshape(5, 6)
    near = g.copy()
    near[2, 3] = np.nextafter(near[2, 3], np.float32(np.inf))
    special = g.copy()
    special[0, 0] = np.nan
    special[1, 1] = np.inf
    special[4, 5] = -np.inf
    nan_moved = special.copy()
    nan_moved[0, 0] = 1.0
    inf_sign = special.copy()
    inf_sign[1, 1] = -np.inf
    sub = g.copy()
    sub[3, 2] = np.float32(1e-40)
    flushed = sub.copy()
    flushed[3, 2] = 0.0
    one_bad = g.copy()
    one_bad[4, 4] += np.float32(4e-5)  # rel_l2 under 1e-5, max_rel over it
    spread = (g * np.float32(1 + 3e-5)).astype(np.float32)
    ints = np.arange(12, dtype=np.int32).reshape(3, 4)
    ints_bad = ints.copy()
    ints_bad[2, 1] = 99
    cases = [
        ('identical', g, g, None),
        ('one_ulp', near, g, None),
        ('special_identical', special, special, None),
        ('nan_mismatch', nan_moved, special, None),
        ('infinity_sign', inf_sign, special, None),
        ('subnormal_flushed', flushed, sub, None),
        ('max_rel', one_bad, g, None),
        ('rel_l2', spread, g, None),
        ('rel_l2_custom_limits', spread, g, {'rel_l2': 1e-4, 'max_rel': 1e-4, 'zero_atol': 1e-6}),
        ('all_special', np.array([np.nan, np.inf], np.float32), np.array([np.nan, np.inf], np.float32), None),
        ('zero_golden', np.array([0.0, 1e-7], np.float32), np.zeros(2, np.float32), None),
        ('zero_golden_fail', np.array([0.0, 1e-3], np.float32), np.zeros(2, np.float32), None),
        ('float16', g.astype(np.float16), g.astype(np.float16), None),
        ('float16_off', (g * 1.01).astype(np.float16), g.astype(np.float16), None),
        ('int_equal', ints, ints, None),
        ('int_mismatch', ints_bad, ints, None),
        ('bool_mismatch', np.array([True, False]), np.array([True, True]), None),
        ('dtype_mismatch', g.astype(np.float16), g, None),
        ('shape_mismatch', g[:4], g, None),
    ]
    return [{'name': n, 'actual': enc(a), 'expected': enc(e), 'limits': lim,
             'result': clean(metrics.compare_kernel_output(a, e, lim))} for n, a, e, lim in cases]


def threshold_cases() -> dict:
    resolved = []
    for profile in metrics.THRESHOLD_PROFILES:
        for rel in (None, 1e-3):
            for pwg in (False, True):
                t = metrics.resolve_threshold(profile, rel, port_weights_golden=pwg)
                noise = [{'median': m, 'result': list(metrics.noise_thresholds(t, m))}
                         for m in (0.0, 1e-7, 1e-5, 3e-5, 1e-4, 2.5e-4, 1e-3, 1e-2, 1.0, 1e9)]
                resolved.append({'profile': profile, 'rel_l2': rel, 'port_weights_golden': pwg, 'result': t,
                                 'noise': noise})
    errors = []
    for profile, rel in (('int8', None), ('float32', 0.0), ('float32', -1.0), ('float32', math.inf)):
        try:
            metrics.resolve_threshold(profile, rel)
        except ValueError as exc:
            errors.append({'profile': profile, 'rel_l2': None if rel is None else clean(rel),
                           'infinite': rel == math.inf, 'error': str(exc)})
    return {'resolved': clean(resolved), 'errors': errors}


def weight_cases() -> list[dict]:
    o = values(24, 7).reshape(4, 6)
    sq = values(36, 8).reshape(6, 6)
    stack = values(3 * 8, 9).reshape(3, 2, 4)
    cases = [
        ('equal', o, o, 'float32'),
        ('float16_rounding', o.astype(np.float16).astype(np.float32), o, 'float16'),
        ('float16_rounding_judged_float32', o.astype(np.float16).astype(np.float32), o, 'float32'),
        ('bfloat16_limit', (o * np.float32(1.004)).astype(np.float32), o, 'bfloat16'),
        ('unknown_kind_is_float32', (o * np.float32(1.004)).astype(np.float32), o, 'float64'),
        ('transposed_square', np.ascontiguousarray(sq.T), sq, 'float32'),
        ('transposed_shape', np.ascontiguousarray(o.T), o, 'float32'),
        ('shape_is_transposed', values(6, 13).reshape(1, 6), values(6, 14).reshape(6, 1), 'float32'),
        ('scaled_by_2', o * np.float32(2), o, 'float32'),
        ('scaled_by_1_5', o * np.float32(1.5), o, 'float32'),
        ('scaled_by_0_3', o * np.float32(0.3), o, 'float32'),
        ('sign_flipped', -o, o, 'float32'),
        ('unrelated', values(24, 11).reshape(4, 6), o, 'float32'),
        ('nonfinite', np.where(o > 1.5, np.float32(np.inf), o), o, 'float32'),
        ('zero_original', np.zeros_like(o), np.zeros_like(o), 'float32'),
        ('zero_original_port_not', o, np.zeros_like(o), 'float32'),
        ('quantized_ok', (o + values(24, 12, scale=0.3).reshape(4, 6)).astype(np.float32), o, 'quantized'),
        ('quantized_scaled', o * np.float32(1.2), o, 'quantized'),
        ('three_d_shape', stack.reshape(3, 4, 2), stack, 'float32'),
        ('original_float16', o, o.astype(np.float16), 'float32'),
        ('original_float64', o.astype(np.float64), o.astype(np.float64) * 1.0000001, 'float32'),
    ]
    out = []
    for name, port, orig, kind in cases:
        out.append({'name': name, 'port': enc(port), 'original': enc(orig), 'kind': kind,
                    'result': clean(weights._compare(port, orig, kind))})
    return out


def build() -> dict:
    return {
        'schema': SCHEMA,
        'generator': 'scripts/rust_vectors.py',
        'constants': {
            'threshold_profiles': metrics.THRESHOLD_PROFILES,
            'port_weights_rel_l2': metrics.PORT_WEIGHTS_RELL2,
            'leaf_noise_factor': metrics.LEAF_NOISE_FACTOR,
            'composite_noise_factor': metrics.COMPOSITE_NOISE_FACTOR,
            'kernel_thresholds': metrics.KERNEL_THRESHOLDS,
            'float32_tiny': metrics._FLOAT32_TINY,
            'weights_rel_l2_limits': weights.REL_L2_LIMITS,
            'quantized_min_cosine': weights.QUANTIZED_MIN_COSINE,
            'quantized_max_scale_error': weights.QUANTIZED_MAX_SCALE_ERROR,
            'tied_embedding': weights.TIED_EMBEDDING,
            'error_codes': [[k, int(v[0]), v[1]] for k, v in ERROR_CODES.items()],
            'exit_codes': [[int(k), k.name, v] for k, v in EXIT_CODE_DOCS.items()],
        },
        'thresholds': threshold_cases(),
        'compare': compare_cases(),
        'compare_kernel_output': kernel_cases(),
        'weights_compare': weight_cases(),
    }


def same(a: Any, b: Any, path: str = '$', tol: float = REL_TOL) -> list[str]:
    """Differences between two JSON values: keys in order, exact strings and integers, floats within tol."""
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        return [] if a == b and type(a) is type(b) else [f'{path}: {a!r} != {b!r}']
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return [] if a == b else [f'{path}: {a} != {b}']
        if a == b or abs(a - b) <= tol * max(abs(a), abs(b)):
            return []
        return [f'{path}: {a!r} != {b!r}']
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f'{path}: keys {list(a)} != {list(b)}']
        return [d for k in a for d in same(a[k], b[k], f'{path}.{k}', tol)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f'{path}[{i}]', tol)]
    return [f'{path}: {type(a).__name__} {a!r} != {type(b).__name__} {b!r}']


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--check', action='store_true', help='exit 1 when the committed file differs from a fresh one')
    args = p.parse_args(argv)
    fresh = json.loads(json.dumps(build(), allow_nan=False))
    if args.check:
        if not OUT.is_file():
            print(f'{OUT} is missing; run python scripts/rust_vectors.py', file=sys.stderr)
            return 1
        diffs = same(fresh, json.loads(OUT.read_text()))
        for d in diffs[:20]:
            print(d, file=sys.stderr)
        return 1 if diffs else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fresh, indent=1, allow_nan=False) + '\n')
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
