"""CUMLX kernel laboratory. CPU reference is usable without installing MLX."""
from .quant import FP4Matrix, quantize, dequantize, load_fp4, save_fp4
from .planning import KernelPlan, plan_linear
from . import reference

__version__ = '0.4.0'
__all__ = ['FP4Matrix', 'quantize', 'dequantize', 'load_fp4', 'save_fp4',
           'KernelPlan', 'plan_linear', 'reference', 'MLXBackend', 'check_op']


def __getattr__(name):
    if name == 'MLXBackend':
        from .mlx_backend import MLXBackend
        return MLXBackend
    if name == 'check_op':
        from .verify.checkop import check_op
        return check_op
    raise AttributeError(name)
