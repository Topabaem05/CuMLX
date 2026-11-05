"""LLM.int8 as bitsandbytes 0.50.2 computes it on CUDA, in plain torch on any device.

Each function names the bitsandbytes step that it follows. The int8 product is exact: it runs as float32 products in blocks of
1024 along K, where no partial sum can pass 2^24, and the blocks are summed in int32. bitsandbytes on CUDA uses cuBLASLt there.

Known differences from CUDA, each one rounding at most:

* CUDA takes 127 / absmax with __fdividef, which can be 2 ulp from the division here, so an activation or weight can round to the
  next int8 step where val * 127 / absmax lies within that much of a half.
* The CUDA dequant kernel uses one fused multiply-add for x * 6.200012e-05 + bias. Here it is a multiply and an add.
"""
from __future__ import annotations

import torch

MM_DEQUANT_CONST = 6.200012e-05  # csrc/kernels.cu, 1 / (127 * 127)
DEQUANT_RECIPROCAL = 7.874015718698502e-3  # _ops.py int8_vectorwise_dequant, 1 / 127
EXACT_BLOCK = 1024  # 1024 * 127 * 127 = 16,516,096 < 2^24 = 16,777,216


def int8_vectorwise_quant(A: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """bitsandbytes::int8_vectorwise_quant: per row, round(val * 127 / absmax) with ties to even.

    With a threshold above 0, a value with |val| >= threshold is left out of the row's absmax and becomes 0, and when there is more
    than one row, each column that holds such a value becomes 0 in every row. Returns (int8 values, float32 absmax per row, outlier
    columns or None when the threshold is 0).
    """
    if A.dtype != torch.float16:
        raise ValueError(f'int8_vectorwise_quant takes float16, got {A.dtype}')
    if threshold < 0:
        raise ValueError('threshold must be at least 0')
    rows = A.reshape(-1, A.shape[-1])
    outlier_cols = None
    kept = rows
    if threshold > 0.0:
        outliers = rows.abs() >= threshold
        outlier_cols = torch.argwhere(outliers.any(dim=0)).view(-1)
        kept = rows.masked_fill(outliers, 0.0)
    row_stats = kept.abs().amax(dim=1).to(torch.float32)
    scale = 127.0 / row_stats
    # A row of zeros gives 0 * inf. CUDA's __float2int_rn turns that NaN into 0, and torch leaves NaN to int8 undefined.
    product = torch.nan_to_num(kept.to(torch.float32) * scale.unsqueeze(-1), nan=0.0)
    out = torch.round(product).to(torch.int8)
    if outlier_cols is not None and rows.shape[0] > 1 and outlier_cols.numel():
        out[:, outlier_cols] = 0
    return out.view(A.shape), row_stats, outlier_cols


def int8_vectorwise_dequant(A: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
    """bitsandbytes::int8_vectorwise_dequant: A * stats * (1 / 127), in float32."""
    return A * stats.view(-1, 1) * DEQUANT_RECIPROCAL


def int8_linear_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """bitsandbytes::int8_linear_matmul: A [..., K] int8 times B [N, K] int8, transposed, as exact int32."""
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise ValueError(f'int8_linear_matmul takes int8, got {A.dtype} and {B.dtype}')
    k = A.shape[-1]
    if B.shape[-1] != k:
        raise ValueError(f'inner sizes differ: {k} and {B.shape[-1]}')
    flat = A.reshape(-1, k)
    total = None
    for start in range(0, k, EXACT_BLOCK):
        part = torch.matmul(flat[:, start:start + EXACT_BLOCK].to(torch.float32), B[:, start:start + EXACT_BLOCK].to(torch.float32).t()).to(torch.int32)
        total = part if total is None else total + part
    if total is None:
        total = torch.zeros(flat.shape[0], B.shape[0], dtype=torch.int32, device=A.device)
    return total.view(*A.shape[:-1], B.shape[0])


def int8_mm_dequant(A: torch.Tensor, row_stats: torch.Tensor, col_stats: torch.Tensor, out: torch.Tensor | None = None, bias: torch.Tensor | None = None,
                    dtype: torch.dtype | None = None) -> torch.Tensor:
    """bitsandbytes::int8_mm_dequant on CUDA: float16 of A * row * col * 6.200012e-05 + bias.

    The kernel takes a float16 bias inside. A bias of another dtype is added after it, to the float16 result, and then the result
    becomes dtype.
    """
    if A.dtype != torch.int32 or row_stats.dtype != torch.float32 or col_stats.dtype != torch.float32:
        raise ValueError('int8_mm_dequant takes int32 values and float32 stats')
    flat = A.reshape(-1, A.shape[-1])
    value = flat.to(torch.float32) * row_stats.reshape(-1, 1) * col_stats.reshape(1, -1) * MM_DEQUANT_CONST
    inside = bias is not None and bias.dtype == torch.float16
    if inside:
        value = value + bias.to(torch.float32)
    result = value.to(torch.float16)
    if bias is not None and not inside:
        result.add_(bias)
    result = result.view(A.shape).to(dtype or torch.float16)
    if out is not None:
        return out.copy_(result)
    return result


def int8_scaled_mm(A: torch.Tensor, B: torch.Tensor, row_stats: torch.Tensor, col_stats: torch.Tensor, bias: torch.Tensor | None = None,
                   dtype: torch.dtype | None = None) -> torch.Tensor:
    """bitsandbytes::int8_scaled_mm: the exact int8 product, then the dequant."""
    return int8_mm_dequant(int8_linear_matmul(A, B), row_stats, col_stats, dtype=dtype or torch.float16, bias=bias)


def int8_mixed_scaled_mm(A: torch.Tensor, CA: torch.Tensor, CB: torch.Tensor, SCA: torch.Tensor, SCB: torch.Tensor, outlier_cols: torch.Tensor | None = None,
                         bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """bitsandbytes::int8_mixed_scaled_mm: the int8 product of the other columns, plus the outlier columns in A's dtype.

    The outlier part is A[:, cols] @ dequant(CB[:, cols]) in A's dtype, added to the output as torch.addmm does on CUDA: float32 sums
    and one rounding to A's dtype.
    """
    sub_b = None
    sub_a = None
    if outlier_cols is not None and outlier_cols.numel():
        sub_a = A[:, outlier_cols].contiguous()
        sub_b = int8_vectorwise_dequant(CB[:, outlier_cols].contiguous(), SCB).to(A.dtype).t()
    output = int8_scaled_mm(CA, CB, SCA, SCB, bias=bias, dtype=A.dtype)
    if sub_b is not None:
        output = (output.to(torch.float32) + torch.matmul(sub_a.to(torch.float32), sub_b.to(torch.float32))).to(A.dtype)
    return output, sub_a
