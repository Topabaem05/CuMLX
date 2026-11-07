"""bitsandbytes' LLM.int8 inference in plain torch, for an unmodified program on a Mac.

    cumlx run infer.py --substitute bitsandbytes=cumlx.compat.bitsandbytes_int8

It covers what a Transformers program with BitsAndBytesConfig(load_in_8bit=True) uses: Int8Params, Linear8bitLt, matmul and the
int8 functions, with the arithmetic of bitsandbytes 0.50.2 on CUDA (see functional.py for the two known rounding differences).
4-bit quantization, the training mode (has_fp16_weights=True), gradients, checkpoints that bitsandbytes already quantized and every
name in bitsandbytes.optim raise NotImplementedError, because a program that lost one in silence would get a wrong result.

Under cumlx run, importlib.metadata reports the distribution bitsandbytes at version 0.50.2+cumlx, so that Transformers finds it.
The local label says that it is not bitsandbytes. The rules and results are in validation/m6-f1-limits.json.
"""
from __future__ import annotations

from . import functional, nn, optim
from ._matmul import MatmulLtState, matmul

__version__ = '0.50.2+cumlx'
DISTRIBUTION = ('bitsandbytes', __version__)
# The devices whose tensors these functions take. Plain torch runs on each, and cumlx run maps cuda to the Mac's device.
supported_torch_devices = frozenset({'cpu', 'cuda', 'mps'})

__all__ = ['MatmulLtState', 'functional', 'matmul', 'nn', 'optim', 'supported_torch_devices']
