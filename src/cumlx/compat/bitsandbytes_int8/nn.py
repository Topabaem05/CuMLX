"""bitsandbytes.nn: Int8Params and Linear8bitLt for LLM.int8 inference, with the behavior of bitsandbytes 0.50.2. 4-bit classes raise."""
from __future__ import annotations

import copy
from typing import Any

import torch

from . import functional
from ._matmul import MatmulLtState, _refuse, matmul


class Int8Params(torch.nn.Parameter):
    """A weight that becomes int8 rows and their float32 absmax (SCB) when it moves from the CPU to a device."""

    def __new__(cls, data: torch.Tensor | None = None, requires_grad: bool = True, has_fp16_weights: bool = False, CB: torch.Tensor | None = None,
                SCB: torch.Tensor | None = None, **kwargs: Any) -> Int8Params:
        if data is None:
            data = torch.empty(0)
        obj = torch.Tensor._make_subclass(cls, data, requires_grad)
        obj.CB = CB
        obj.SCB = SCB
        obj.has_fp16_weights = has_fp16_weights
        return obj

    def _quantize(self, device: Any) -> Int8Params:
        if self.has_fp16_weights:
            raise _refuse('has_fp16_weights=True, the training mode of LLM.int8')
        weights = self.data.contiguous().to(device=device, dtype=torch.float16)
        CB, SCB, _ = functional.int8_vectorwise_quant(weights)
        self.data = CB
        self.CB = CB
        self.SCB = SCB
        return self

    def cpu(self) -> Int8Params:
        return self.to(device='cpu')

    def cuda(self, device: Any = None, non_blocking: bool = False) -> Int8Params:
        return self.to(device='cuda' if device is None else device, non_blocking=non_blocking)

    def to(self, *args: Any, **kwargs: Any) -> Int8Params:
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type != 'meta' and self.data.device.type == 'cpu' and self.data.dtype != torch.int8:
            return self._quantize(device)
        if self.data.dtype == torch.int8:
            dtype = None  # int8 rows stay int8, as in bitsandbytes
        moved = Int8Params(super().to(device=device, dtype=dtype, non_blocking=non_blocking), requires_grad=self.requires_grad, has_fp16_weights=self.has_fp16_weights)
        if self.data.dtype == torch.int8:
            moved.CB = moved.data
            moved.SCB = self.SCB.to(device) if self.SCB is not None and device is not None and self.SCB.device.type != 'meta' else self.SCB
        return moved

    def __deepcopy__(self, memo: dict) -> Int8Params:
        new = type(self).__new__(type(self), data=copy.deepcopy(self.data, memo), requires_grad=self.requires_grad, has_fp16_weights=self.has_fp16_weights,
                                 CB=copy.deepcopy(self.CB, memo), SCB=copy.deepcopy(self.SCB, memo))
        return new


def maybe_rearrange_weight(state_dict: dict, prefix: str, local_metadata: Any, strict: bool, missing_keys: list, unexpected_keys: list, error_msgs: list) -> None:
    weight_format = state_dict.pop(f'{prefix}weight_format', 'row')
    if isinstance(weight_format, torch.Tensor):
        weight_format = int(weight_format.item())
    if weight_format not in ('row', 0):
        raise ValueError(f'only the row-major format is supported, got {weight_format!r}')


class Linear8bitLt(torch.nn.Linear):
    """nn.Linear with LLM.int8 weights. Inference only: the training mode and gradients raise."""

    def __init__(self, input_features: int, output_features: int, bias: bool = True, has_fp16_weights: bool = True, threshold: float = 0.0,
                 index: Any = None, device: Any = None) -> None:
        super().__init__(input_features, output_features, bias, device)
        self.state = MatmulLtState()
        self.index = index
        self.state.threshold = threshold
        self.state.has_fp16_weights = has_fp16_weights
        if threshold > 0.0 and not has_fp16_weights:
            self.state.use_pool = True
        self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)
        self._register_load_state_dict_pre_hook(maybe_rearrange_weight)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, local_metadata: Any, strict: bool, missing_keys: list, unexpected_keys: list, error_msgs: list) -> None:
        if f'{prefix}SCB' in state_dict:
            raise _refuse('loading a checkpoint that bitsandbytes already quantized (an SCB key)')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def init_8bit_state(self) -> None:
        self.state.CB = self.weight.CB
        self.state.SCB = self.weight.SCB
        self.weight.CB = None
        self.weight.SCB = None

    def to(self, *args: Any, **kwargs: Any) -> Linear8bitLt:
        result = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            if result.state.CB is not None:
                result.state.CB = result.state.CB.to(device)
            if result.state.SCB is not None:
                result.state.SCB = result.state.SCB.to(device)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state.is_training = self.training
        if self.weight.CB is not None:
            self.init_8bit_state()
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        out = matmul(x, self.weight, bias=self.bias, state=self.state)
        if not self.state.has_fp16_weights and self.state.CB is not None:
            self.weight.data = self.state.CB
        return out


class _FourBit:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _refuse('4-bit quantization (Linear4bit, Params4bit)')


class Linear4bit(_FourBit, torch.nn.Module):
    pass


class Params4bit(_FourBit):
    @classmethod
    def from_prequantized(cls, *args: Any, **kwargs: Any) -> Params4bit:
        raise _refuse('4-bit quantization (Linear4bit, Params4bit)')


__all__ = ['Int8Params', 'Linear8bitLt', 'Linear4bit', 'Params4bit', 'matmul']
