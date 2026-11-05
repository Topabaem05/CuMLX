"""flash_attn.bert_padding: move between padded [batch, seqlen, ...] tensors and packed [tokens, ...] tensors, as flash-attn 2.8 does."""
from __future__ import annotations

import torch


def index_first_axis(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values[indices]


def index_put_first_axis(values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
    out = torch.zeros((first_axis_dim, *values.shape[1:]), device=values.device, dtype=values.dtype)
    out[indices] = values
    return out


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor, unused_mask: torch.Tensor | None = None):
    """Returns (packed tokens, indices, cumulative lengths int32, longest length, used lengths), the five values of flash-attn 2.8."""
    all_masks = attention_mask + unused_mask if unused_mask is not None else attention_mask
    seqlens = all_masks.sum(dim=-1, dtype=torch.int32)
    used = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    packed = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return index_first_axis(packed, indices), indices, cu_seqlens, int(seqlens.max().item()), used


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    out = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return out.view(batch, seqlen, *hidden_states.shape[1:])
