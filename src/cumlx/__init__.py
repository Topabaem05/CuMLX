"""CUMLX kernel laboratory. CPU reference is usable without installing MLX."""
from .quant import FP4Matrix, quantize, dequantize, load_fp4, save_fp4
from .planning import KernelPlan, plan_linear
from . import reference

__version__ = '0.3.0.dev0'
__all__ = ['FP4Matrix', 'quantize', 'dequantize', 'load_fp4', 'save_fp4',
           'KernelPlan', 'plan_linear', 'reference', 'MLXBackend']


def __getattr__(name):
    if name == 'MLXBackend':
        from .mlx_backend import MLXBackend
        return MLXBackend
    raise AttributeError(name)
