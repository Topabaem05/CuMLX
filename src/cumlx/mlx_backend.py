"""MLX launch bindings for unvalidated GPU candidates. No CUDA facade or CPU fallback.

Dependencies are lazy: importing the package and inspecting source works without MLX.
Execution is restricted to the documented 0.32.2 API profile and real GPU backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import platform
import warnings
from typing import Any
import numpy as np

from .quant import FP4Matrix, group_size, normalize_format
from .planning import plan_linear

TARGET_MLX = '0.32.2'  # the lowest MLX API these kernels were written for
# All 35 GPU tests passed on these versions (Apple M5 Pro, 2026-11-02). Other 0.32.x releases run with a warning.
VALIDATED_MLX = ('0.32.2', '0.32.4')
KERNELS = {
    'fp4_decode': (('wq', 'ws', 'wglobal'), ('y',)),
    'fp4_gemv': (('x', 'wq', 'ws', 'wglobal', 'bias'), ('y',)),
    'fp4_small_m': (('x', 'wq', 'ws', 'wglobal', 'bias'), ('y',)),
    'fp4_tiled': (('x', 'wq', 'ws', 'wglobal', 'bias'), ('y',)),
    'fp4_w4a4': (('aq', 'ascales', 'aglobal', 'wq', 'ws', 'wglobal', 'bias'), ('y',)),
    'fp4_quantize': (('x', 'global'), ('outq', 'outs')),
    'rms_norm': (('x', 'residual', 'weight', 'epsilon'), ('y',)),
    'silu_mul': (('gate', 'up'), ('y',)),
}


def kernel_source(name: str, backend: str) -> tuple[str, str, str]:
    if name not in KERNELS or backend not in ('metal', 'cuda'):
        raise ValueError('unknown kernel name or physical backend')
    base = Path(__file__).parent / 'kernels'
    ext = 'metal' if backend == 'metal' else 'cu'
    header = (base / f'common_{backend}.h').read_text()
    source = (base / f'{name}.{ext}').read_text()
    digest = sha256((header + '\n' + source).encode()).hexdigest()
    return header, source, digest


def environment_report(*, probe: bool = False) -> dict:
    try:
        version = metadata.version('mlx')
    except metadata.PackageNotFoundError:
        version = None
    report = dict(platform=platform.platform(), machine=platform.machine(),
                  python=platform.python_version(), mlx_installed=version,
                  mlx_target=TARGET_MLX, mlx_validated=list(VALIDATED_MLX), device_validation='NOT_RUN',
                  cuda_facing_pytorch=False, native_fp4_matrix_instruction=False,
                  automatic_cpu_fallback=False,
                  metal_available=None, cuda_available=None)
    if probe:
        try:
            import mlx.core as mx
            for backend in ('metal', 'cuda'):
                api = getattr(mx, backend, None)
                report[f'{backend}_available'] = bool(api and api.is_available())
            if report['metal_available'] or report['cuda_available']:
                report['gpu_info'] = mx.device_info(mx.gpu)
        except (ImportError, OSError, RuntimeError) as exc:
            report['probe_error'] = f'{type(exc).__name__}: {exc}'
    return report


@dataclass(frozen=True, eq=False)
class DeviceFP4:
    """Owner-tagged buffers. Do not mutate/rebind contained MLX array payloads.

    This is not a Torch tensor, an external pointer, or a zero-copy promise.
    Only MLXBackend.prepare/quantize are supported construction paths.
    """
    format: str
    shape: tuple[int, int]
    payload: Any = field(repr=False)
    scales: Any = field(repr=False)
    global_scale: Any = field(repr=False)
    backend: str
    _owner: Any = field(repr=False)
    source_checksum: str | None = None

    @property
    def group_size(self) -> int:
        return group_size(self.format)


def check_mlx_version(version: str) -> bool:
    """Allow the 0.32 API line from TARGET_MLX up. Returns True when this exact version was validated."""
    def parts(v: str) -> tuple[int, ...]:
        numbers = []
        for piece in v.split('.')[:3]:
            digits = ''.join(ch for ch in piece if ch.isdigit())
            numbers.append(int(digits or 0))
        return tuple(numbers)
    current, lowest = parts(version), parts(TARGET_MLX)
    if current[:2] != lowest[:2] or current < lowest:
        raise RuntimeError(f'expected MLX {lowest[0]}.{lowest[1]}.x at or above {TARGET_MLX}, got {version}; '
                           'revalidate the kernel APIs and run the GPU tests before you allow this version')
    if version not in VALIDATED_MLX:
        warnings.warn(f'MLX {version} is in the allowed range, but the GPU tests have not been run on it '
                      f'(validated: {", ".join(VALIDATED_MLX)})', stacklevel=2)
        return False
    return True


class MLXBackend:
    def __init__(self, backend: str = 'metal', *, allow_unvalidated: bool = False):
        if backend not in ('metal', 'cuda'):
            raise ValueError('backend must be metal or native cuda; no implicit CPU backend')
        if not allow_unvalidated:
            raise RuntimeError('GPU kernels are unvalidated on hardware; explicit allow_unvalidated=True is required')
        import mlx.core as mx
        check_mlx_version(metadata.version('mlx'))
        api = getattr(mx, backend, None)
        if api is None or not api.is_available():
            raise RuntimeError(f'physical {backend} GPU unavailable; no fallback was selected')
        if not hasattr(mx.fast, f'{backend}_kernel'):
            raise RuntimeError(f'MLX lacks {backend}_kernel')
        self.mx, self.kind = mx, backend
        self.stream = mx.default_stream(mx.gpu)
        self._owner = object()
        self._cache: dict[str, Any] = {}
        with mx.stream(self.stream):
            self._zero = mx.zeros((1,), dtype=mx.float32)
        mx.eval(self._zero)

    def _floating(self, x: Any, name: str, ndim: int | None = None) -> None:
        mx = self.mx
        if not isinstance(x, mx.array):
            raise TypeError(f'{name} must be an MLX array; use mx.array explicitly for a host upload')
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise TypeError(f'{name} must be float16, bfloat16 or float32')
        if (ndim is not None and x.ndim != ndim) or x.size <= 0 or x.size > 2**31 - 1:
            raise ValueError(f'{name} has unsupported rank/size')

    def _q(self, q: DeviceFP4) -> None:
        if not isinstance(q, DeviceFP4) or q._owner is not self._owner or q.backend != self.kind:
            raise ValueError('FP4 buffers belong to a different backend/session; prepare explicitly')
        n, k = q.shape
        g = q.group_size
        if (q.payload.shape != (n, (k + 1)//2) or q.scales.shape != (n, (k + g - 1)//g)
                or q.payload.dtype != self.mx.uint8 or q.scales.dtype != self.mx.uint8
                or q.global_scale.shape != (1,) or q.global_scale.dtype != self.mx.float32):
            raise ValueError('FP4 buffer metadata was modified')

    def _params(self, q: DeviceFP4, prefix: str = 'W') -> dict:
        _, k = q.shape
        return {prefix + 'P': (k + 1)//2, prefix + 'SG': (k + q.group_size - 1)//q.group_size,
                prefix + 'G': q.group_size, prefix + 'F': 0 if q.format == 'nvfp4' else 1}

    def _bias(self, bias: Any, n: int, epilogue: str) -> tuple[Any, dict]:
        if epilogue not in ('none', 'silu'):
            raise ValueError('epilogue must be none or silu')
        if bias is not None:
            self._floating(bias, 'bias', 1)
            if bias.shape != (n,):
                raise ValueError('bias must have shape (N,)')
            # Explicit conversion allows a single FP32 epilogue contract.
            bias = bias.astype(self.mx.float32, stream=self.stream)
        return bias if bias is not None else self._zero, {
            'HAS_BIAS': bias is not None, 'APPLY_SILU': epilogue == 'silu'}

    @staticmethod
    def _grid(elements: int) -> tuple[int, int, int]:
        threads = ((elements + 127)//128)*128
        if elements <= 0 or threads > 2**31 - 1:
            raise ValueError('launch grid exceeds the int32 profile')
        return threads, 1, 1

    def _launch(self, name: str, inputs: list, params: dict, grid: tuple,
                shapes: list, dtypes: list | None = None) -> list:
        mx = self.mx
        if name not in self._cache:
            header, source, digest = kernel_source(name, self.kind)
            input_names, output_names = KERNELS[name]
            kwargs = dict(name=f'cumlx_{name}_{digest[:12]}', input_names=list(input_names),
                          output_names=list(output_names), source=source, header=header,
                          ensure_row_contiguous=True)
            if self.kind == 'metal':
                kwargs['compile_options'] = {'math_mode': 'safe'}
            self._cache[name] = getattr(mx.fast, f'{self.kind}_kernel')(**kwargs)
        with mx.stream(self.stream):
            return self._cache[name](inputs=inputs, template=list(params.items()),
                                     grid=grid, threadgroup=(128, 1, 1),
                                     output_shapes=shapes,
                                     output_dtypes=dtypes or [mx.float32]*len(shapes),
                                     stream=self.stream)

    def prepare(self, q: FP4Matrix) -> DeviceFP4:
        if not isinstance(q, FP4Matrix):
            raise TypeError('prepare requires a validated CPU FP4Matrix')
        mx = self.mx
        with mx.stream(self.stream):
            p = mx.array(q.payload)
            s = mx.array(q.scales)
            global_scale = mx.array([q.global_scale], dtype=mx.float32)
        # Explicit upload/preparation boundary; excluded from warm kernel benchmarks.
        mx.eval(p, s, global_scale)
        return DeviceFP4(q.format, q.shape, p, s, global_scale, self.kind, self._owner, q.checksum)

    def decode(self, q: DeviceFP4) -> Any:
        """Diagnostic full FP32 allocation. The linear methods never call this."""
        self._q(q)
        n, k = q.shape
        return self._launch('fp4_decode', [q.payload, q.scales, q.global_scale],
                            dict(N=n, K=k, **self._params(q)), self._grid(n*k), [(n, k)])[0]

    def linear(self, x: Any, w: DeviceFP4, *, bias: Any = None,
               algorithm: str = 'auto', epilogue: str = 'none') -> Any:
        self._q(w)
        self._floating(x, 'x', 2)
        m, k = x.shape
        n, wk = w.shape
        if k != wk:
            raise ValueError('x.shape[-1] and weight K differ')
        plan = plan_linear(m, n, k, algorithm=algorithm)
        bias, ep = self._bias(bias, n, epilogue)
        params = dict(M=m, N=n, K=k, **self._params(w), **ep)
        if plan.algorithm in ('gemv', 'small_m'):
            params['BM'] = plan.tile_m
        return self._launch('fp4_' + plan.algorithm,
                            [x, w.payload, w.scales, w.global_scale, bias], params,
                            plan.grid, [(m, n)])[0]

    def linear_fp4(self, a: DeviceFP4, w: DeviceFP4, *, bias: Any = None,
                   epilogue: str = 'none') -> Any:
        self._q(a)
        self._q(w)
        m, k = a.shape
        n, wk = w.shape
        if k != wk:
            raise ValueError('activation and weight K differ')
        plan = plan_linear(m, n, k, algorithm='tiled')
        bias, ep = self._bias(bias, n, epilogue)
        return self._launch('fp4_w4a4',
                            [a.payload, a.scales, a.global_scale, w.payload, w.scales, w.global_scale, bias],
                            dict(M=m, N=n, K=k, **self._params(a, 'A'), **self._params(w), **ep),
                            plan.grid, [(m, n)])[0]

    def quantize(self, x: Any, fmt: str = 'nvfp4', *, global_scale: float | None = None) -> DeviceFP4:
        """Experimental device encoder; finite/normalization checks synchronize.

        Uses the CUMLX rowwise FP32 recipe, not an arbitrary producer/training recipe.
        No validate=False escape: nonfinite inputs must not be turned into fake codes.
        """
        fmt = normalize_format(fmt)
        self._floating(x, 'x', 2)
        mx = self.mx
        if not bool(mx.all(mx.isfinite(x)).item()):
            raise ValueError('quantize input contains NaN or infinity')
        n, k = x.shape
        plan_linear(n, 1, k, algorithm='tiled')
        g = group_size(fmt)
        with mx.stream(self.stream):
            amax = mx.max(mx.abs(x.astype(mx.float32)))
            if global_scale is None:
                gs = (mx.where(amax == 0, mx.array(1., mx.float32),
                               mx.maximum(amax / 2688., np.finfo(np.float32).tiny))
                      if fmt == 'nvfp4' else mx.array(1., mx.float32))
                gs = gs.reshape(1)
            else:
                value = float(global_scale)
                if (not np.isfinite(value) or value <= 0 or value > np.finfo(np.float32).max
                        or np.float32(value) <= 0 or (fmt == 'mxfp4' and value != 1)):
                    raise ValueError('invalid global_scale for format')
                gs = mx.array([value], dtype=mx.float32)
            sg, p = (k + g - 1)//g, (k + 1)//2
            outq, outs = self._launch('fp4_quantize', [x, gs],
                dict(N=n, K=k, SG=sg, G=g, F=0 if fmt == 'nvfp4' else 1, P=p),
                self._grid(n*sg), [(n, p), (n, sg)], [mx.uint8, mx.uint8])
        return DeviceFP4(fmt, (n, k), outq, outs, gs, self.kind, self._owner)

    def silu_mul(self, gate: Any, up: Any) -> Any:
        self._floating(gate, 'gate')
        self._floating(up, 'up')
        if gate.shape != up.shape:
            raise ValueError('gate/up shapes must match; broadcasting is not implicit')
        return self._launch('silu_mul', [gate, up], dict(SIZE=gate.size),
                            self._grid(gate.size), [gate.shape])[0]

    def rms_norm(self, x: Any, weight: Any, *, eps: float = 1e-5,
                 residual: Any = None) -> Any:
        self._floating(x, 'x', 2)
        self._floating(weight, 'weight', 1)
        m, k = x.shape
        if weight.shape != (k,):
            raise ValueError('RMSNorm weight must have shape (K,)')
        if not np.isfinite(eps) or eps <= 0 or eps > np.finfo(np.float32).max or np.float32(eps) <= 0:
            raise ValueError('epsilon must be positive finite float32')
        has_residual = residual is not None
        if has_residual:
            self._floating(residual, 'residual', 2)
            if residual.shape != x.shape:
                raise ValueError('residual shape must equal x shape')
        else:
            residual = self._zero
        with self.mx.stream(self.stream):
            epsilon = self.mx.array([eps], dtype=self.mx.float32)
        return self._launch('rms_norm', [x, residual, weight, epsilon],
                            dict(K=k, HAS_RESIDUAL=has_residual), self._grid(m*128), [x.shape])[0]

    def native_linear(self, x: Any, w: DeviceFP4) -> Any:
        """Optional upstream comparator, NOT a fallback or guaranteed bitwise equivalent.

        Native QMM applies block scales internally; the tensor multiplier is applied
        to the output because MLX 0.32.2 QMM has no global_scale parameter.
        Its rounding can differ. Gate accuracy before comparing timings.
        """
        self._q(w)
        self._floating(x, 'x', 2)
        _, k = w.shape
        if x.shape[1] != k or k % w.group_size or k % 8:
            raise ValueError('native comparator requires matching K aligned to the FP4 group size')
        mx = self.mx
        with mx.stream(self.stream):
            packed = w.payload.view(mx.uint32)
            y = mx.quantized_matmul(x, packed, w.scales, transpose=True,
                                   group_size=w.group_size, bits=4, mode=w.format, stream=self.stream)
            return y.astype(mx.float32) * w.global_scale
