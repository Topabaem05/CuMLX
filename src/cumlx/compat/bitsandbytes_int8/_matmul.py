"""bitsandbytes.matmul and MatmulLtState for LLM.int8 inference, as MatMul8bitLt.forward does it in bitsandbytes 0.50.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from . import functional


def _refuse(what: str) -> NotImplementedError:
    return NotImplementedError(f'cumlx.compat.bitsandbytes_int8 does not implement {what}, so it stops here and does not ignore it')


@dataclass
class MatmulLtState:
    force_no_igemmlt: bool = False
    CB: torch.Tensor | None = None
    SB: Any = None
    SCB: torch.Tensor | None = None
    CBt: torch.Tensor | None = None
    SBt: Any = None
    subB: torch.Tensor | None = None
    outlier_pool: Any = None
    threshold: float = 0.0
    idx: torch.Tensor | None = None
    is_training: bool = True
    has_fp16_weights: bool = True
    use_pool: bool = False

    def reset_grads(self) -> None:
        self.CB = self.CBt = self.SCB = self.SBt = None
        self.SB = None


def matmul(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor | None = None, state: MatmulLtState | None = None, threshold: float = 0.0,
           bias: torch.Tensor | None = None) -> torch.Tensor:
    """A @ B.T for an int8 B held in state, for inference only. Training through this raises."""
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    if state.has_fp16_weights:
        raise _refuse('has_fp16_weights=True, the training mode of LLM.int8')
    if torch.is_grad_enabled() and (A.requires_grad or (bias is not None and bias.requires_grad)):
        raise _refuse('gradients through LLM.int8')
    if state.CB is None:
        raise _refuse('a weight that was not quantized: move the layer to its device first')
    if A.numel() == 0:
        return torch.empty(*A.shape[:-1], state.CB.shape[0], dtype=A.dtype, device=A.device)
    shape = A.shape
    flat = A.reshape(-1, shape[-1]) if A.dim() == 3 else A
    CA, SCA, outlier_cols = functional.int8_vectorwise_quant(flat.to(torch.float16), threshold=state.threshold)
    if state.threshold > 0.0:
        state.idx = outlier_cols
        output, _ = functional.int8_mixed_scaled_mm(flat, CA, state.CB, SCA, state.SCB, outlier_cols, bias)
    else:
        output = functional.int8_scaled_mm(CA, state.CB, SCA, state.SCB, bias=bias, dtype=flat.dtype)
    if A.dim() == 3:
        output = output.reshape(*shape[:-1], state.CB.shape[0])
    return output
