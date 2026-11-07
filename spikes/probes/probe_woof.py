"""Can woof's translator run CUDA kernels it was not written for? Usage: probe_woof.py <cumlx kernels dir> [foreign .cu ...]

CUMLX's kernels run on Metal and are compared with a NumPy float64 reference. A wrong kernel must fail the same comparison.
Foreign files are only translated and compiled, and they are not checked for correct values. They see empty cuda_fp16.h and
cuda_bf16.h files, because a missing header says nothing about the translator.
"""
import atexit
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ['WOOF_BACKEND'] = 'mlx'
import woof  # noqa: E402,F401  - installs the cupy facade, so it comes first

# isort: split
import cupy  # noqa: E402
import numpy as np  # noqa: E402
from woof.backend.metal.loader import load_unit  # noqa: E402
from woof.backend.translate import translate  # noqa: E402

KERNELS = Path(sys.argv[1])
COMMON = (KERNELS / 'common_cuda.h').read_text()
rng = np.random.default_rng(0)


def report(**fields):
    print(json.dumps({'probe': 'woof', **fields}), flush=True)


def unit(defines, name, params, body=None):
    text = body if body is not None else (KERNELS / f'{name}.cu').read_text()
    macros = ''.join(f'#define {key} {value}\n' for key, value in defines.items())
    return f'{COMMON}\n{macros}extern "C" __global__ void {name}({params}) {{\n{text}\n}}\n'


def rel_l2(actual, expected):
    return float(np.linalg.norm(actual.astype(np.float64) - expected) / np.linalg.norm(expected))


def run(label, source, name, grid, block, args, expected):
    try:
        kernel = load_unit(label, source).get_function(name)
        arrays = [cupy.asarray(a) for a in args]
        kernel(grid, block, tuple(arrays))
        actual = cupy.asnumpy(arrays[-1])
        report(case=label, rel_l2=rel_l2(actual, expected))
    except Exception as exc:  # noqa: BLE001 - a failure is the result
        report(case=label, error=f'{type(exc).__name__}: {exc}'[:400])


size = 4096
gate = rng.normal(0, 2, size).astype(np.float32)
up = rng.normal(0, 2, size).astype(np.float32)
g = gate.astype(np.float64)
e = np.exp(-np.abs(g))
expected = g * np.where(g >= 0, 1 / (1 + e), e / (1 + e)) * up
output = np.zeros(size, np.float32)
params = 'const float* gate, const float* up, float* y'
blocks = (size // 256,)
run('silu_mul', unit({'SIZE': f'{size}u'}, 'silu_mul', params), 'silu_mul', blocks, (256,), (gate, up, output), expected)
wrong = 'unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);\nif (i >= SIZE) return;\ny[i] = cumlx_silu(float(gate[i])) + float(up[i]);\n'
run('silu_mul_wrong_control', unit({'SIZE': f'{size}u'}, 'silu_mul', params, wrong), 'silu_mul', blocks, (256,), (gate, up, output.copy()), expected)

rows, k = 8, 1024
x = rng.normal(0, 1, (rows, k)).astype(np.float32)
weight = rng.normal(1, 0.1, k).astype(np.float32)
epsilon = np.array([1e-5], np.float32)
xd = x.astype(np.float64)
expected = xd / np.sqrt(np.mean(xd * xd, axis=1, keepdims=True) + float(epsilon[0])) * weight
run('rms_norm', unit({'K': f'{k}u', 'HAS_RESIDUAL': '0'}, 'rms_norm', 'const float* x, const float* residual, const float* weight, const float* epsilon, float* y'),
    'rms_norm', (rows,), (128,), (x.ravel(), np.zeros(1, np.float32), weight, epsilon, np.zeros(rows * k, np.float32)), expected.ravel())

n, k, wg = 4, 128, 16
codes = rng.integers(0, 16, (n, k)).astype(np.uint8)
packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
scales = rng.integers(40, 80, (n, k // wg)).astype(np.uint8)
glob = np.array([0.5], np.float32)
mags = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
values = mags[codes & 7] * np.where(codes & 8, -1.0, 1.0)
exponent, mantissa = scales >> 3, scales & 7
scale_values = np.where(exponent == 0, mantissa * 0.001953125, np.ldexp(1 + mantissa * 0.125, exponent.astype(int) - 7))
expected = values * np.repeat(scale_values, wg, axis=1) * float(glob[0])
defines = {'N': f'{n}u', 'K': f'{k}u', 'WP': f'{k // 2}u', 'WSG': f'{k // wg}u', 'WG': f'{wg}u', 'WF': '0'}
run('fp4_decode', unit(defines, 'fp4_decode', 'const unsigned char* wq, const unsigned char* ws, const float* wglobal, float* y'),
    'fp4_decode', ((n * k + 255) // 256,), (256,), (packed.ravel(), scales.ravel(), glob, np.zeros(n * k, np.float32)), expected.ravel())

shim = tempfile.mkdtemp()
atexit.register(shutil.rmtree, shim, ignore_errors=True)
for header in ('cuda_fp16.h', 'cuda_bf16.h'):
    Path(shim, header).touch()
for path in sys.argv[2:]:
    source = Path(path)
    label = '/'.join(source.parts[-2:])
    try:
        translated = translate(source.read_text(errors='replace'), unit=label, include_dirs=(shim, str(source.parent)))
        fields = {'kernels': sorted(translated.kernels), 'msl_lines': translated.msl.count('\n')}
        try:
            load_unit(label, source.read_text(errors='replace'), options=(f'-I{shim}', f'-I{source.parent}')).build_all()
            fields['metal_compile'] = 'ok'
        except Exception as exc:  # noqa: BLE001 - a failure is the result
            fields['metal_compile'] = f'{type(exc).__name__}: {exc}'[:300]
        report(foreign=label, translated=True, **fields)
    except Exception as exc:  # noqa: BLE001 - a failure is the result
        report(foreign=label, translated=False, reason=f'{type(exc).__name__}: {exc}'[:300])
