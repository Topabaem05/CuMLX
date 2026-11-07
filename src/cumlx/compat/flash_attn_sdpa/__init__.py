"""flash_attn's FlashAttention-2 API as exact attention in float32, for an unmodified program on a Mac or on a GPU that FlashAttention-2 does not support.

    cumlx run infer.py --substitute flash_attn=cumlx.compat.flash_attn_sdpa

FlashAttention-2 computes exact attention, so this computes the same thing without its tiling: QK^T, the softmax and PV in float32 on the
tensors' own device, and the result in the input's dtype. It is slower and uses O(S^2) memory. It covers flash_attn_func and
flash_attn_varlen_func with causal or full attention, grouped-query heads and the softmax scale. Dropout, a sliding window, softcap,
ALiBi, returned probabilities, paged caches, flash_attn_with_kvcache and the four packed functions raise NotImplementedError, because a
program that lost one in silence would get a wrong result.

The causal mask is aligned to the bottom right, as in flash-attn 2.1 and later: query i of Sq attends key j of Sk when j <= i + Sk - Sq.
A query row that attends no key gives zeros, as flash-attn does.

Under cumlx run, importlib.metadata reports the distribution flash-attn at version 2.8.3+cumlx, so that a library that checks the
version finds it. The local label says that it is not flash-attn. The rules and results are in validation/m6-f1-limits.json.
"""
from __future__ import annotations

from typing import Any

import torch

from . import (
    bert_padding,  # noqa: F401  # flash_attn.bert_padding is a submodule that programs import
)

__version__ = '2.8.3+cumlx'
DISTRIBUTION = ('flash-attn', __version__)


def _refuse(what: str) -> NotImplementedError:
    return NotImplementedError(f'cumlx.compat.flash_attn_sdpa does not implement {what}, so it stops here and does not ignore it')


def _check(dropout_p: float, window_size: Any, softcap: float, alibi_slopes: Any, return_attn_probs: bool, block_table: Any = None) -> None:
    if dropout_p:
        raise _refuse(f'dropout_p={dropout_p}')
    if tuple(window_size) != (-1, -1):
        raise _refuse(f'window_size={tuple(window_size)}')
    if softcap:
        raise _refuse(f'softcap={softcap}')
    if alibi_slopes is not None:
        raise _refuse('alibi_slopes')
    if return_attn_probs:
        raise _refuse('return_attn_probs=True')
    if block_table is not None:
        raise _refuse('block_table')


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float | None, causal: bool) -> torch.Tensor:
    """q [B, Sq, Hq, D], k and v [B, Sk, Hk, D] -> [B, Sq, Hq, D] in q's dtype."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise _refuse(f'tensors of rank {q.dim()}, {k.dim()}, {v.dim()}: flash_attn_func takes [batch, seqlen, heads, headdim]')
    batch, sq, hq, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    if k.shape != v.shape or k.shape[0] != batch or k.shape[3] != d:
        raise _refuse(f'shapes q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}')
    if hq % hk:
        raise _refuse(f'{hq} query heads over {hk} key heads')
    scale = d ** -0.5 if softmax_scale is None else float(softmax_scale)
    qf, kf, vf = (t.to(torch.float32).transpose(1, 2) for t in (q, k, v))
    if hk != hq:
        kf = kf.repeat_interleave(hq // hk, dim=1)
        vf = vf.repeat_interleave(hq // hk, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    if causal:
        rows = torch.arange(sq, device=q.device).view(-1, 1)
        cols = torch.arange(sk, device=q.device).view(1, -1)
        blocked = cols > rows + (sk - sq)
        scores = scores.masked_fill(blocked, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(blocked.all(dim=-1, keepdim=True), 0.0)
    else:
        weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, vf)
    return out.transpose(1, 2).to(q.dtype)


def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float = 0.0, softmax_scale: float | None = None, causal: bool = False,
                    window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0, alibi_slopes: Any = None, deterministic: bool = False, return_attn_probs: bool = False) -> torch.Tensor:
    _check(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs)
    return _attend(q, k, v, softmax_scale, causal)


def flash_attn_varlen_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int,
                           dropout_p: float = 0.0, softmax_scale: float | None = None, causal: bool = False, window_size: tuple[int, int] = (-1, -1), softcap: float = 0.0,
                           alibi_slopes: Any = None, deterministic: bool = False, return_attn_probs: bool = False, block_table: Any = None) -> torch.Tensor:
    """q [total_q, Hq, D], k and v [total_k, Hk, D], sequences given by the cumulative lengths. Each sequence attends only to itself."""
    _check(dropout_p, window_size, softcap, alibi_slopes, return_attn_probs, block_table)
    starts_q, starts_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    if len(starts_q) != len(starts_k):
        raise _refuse(f'{len(starts_q) - 1} query sequences with {len(starts_k) - 1} key sequences')
    if starts_q[-1] != q.shape[0] or starts_k[-1] != k.shape[0]:
        raise _refuse(f'cumulative lengths that end at {starts_q[-1]} and {starts_k[-1]} for {q.shape[0]} and {k.shape[0]} tokens')
    parts = []
    for index in range(len(starts_q) - 1):
        a, b = starts_q[index], starts_q[index + 1]
        c, e = starts_k[index], starts_k[index + 1]
        if b - a > max_seqlen_q or e - c > max_seqlen_k:
            raise _refuse(f'a sequence longer than max_seqlen ({b - a} > {max_seqlen_q} or {e - c} > {max_seqlen_k})')
        parts.append(_attend(q[a:b][None], k[c:e][None], v[c:e][None], softmax_scale, causal)[0])
    return torch.cat(parts, dim=0) if parts else q.new_empty(q.shape)


def flash_attn_with_kvcache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor | None = None, v: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> torch.Tensor:
    raise _refuse('flash_attn_with_kvcache')


def flash_attn_qkvpacked_func(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise _refuse('flash_attn_qkvpacked_func')


def flash_attn_kvpacked_func(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise _refuse('flash_attn_kvpacked_func')


def flash_attn_varlen_qkvpacked_func(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise _refuse('flash_attn_varlen_qkvpacked_func')


def flash_attn_varlen_kvpacked_func(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise _refuse('flash_attn_varlen_kvpacked_func')
