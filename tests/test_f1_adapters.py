"""cumlx.compat.flash_attn_sdpa and cumlx.compat.bitsandbytes_int8 against hand computations, and the distributions that cumlx run reports for them."""

import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cumlx.compat import bitsandbytes_int8 as bnb  # noqa: E402
from cumlx.compat import flash_attn_sdpa as fa  # noqa: E402
from cumlx.compat import torch_cuda  # noqa: E402
from cumlx.compat.bitsandbytes_int8 import functional as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spikes"))
from m6_f1_record import exact_attention  # noqa: E402


def rand(*shape, seed=0, dtype=torch.float32):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


# ----------------------------------------------------------------------------------------------- flash_attn


@pytest.mark.parametrize("sq, sk, hq, hk, causal", [(5, 5, 4, 4, True), (5, 5, 4, 2, True), (1, 7, 4, 2, True), (3, 7, 2, 1, True), (4, 6, 2, 2, False)])
def test_attention_is_exact_attention_with_a_bottom_right_mask(sq, sk, hq, hk, causal):
    q, k, v = rand(2, sq, hq, 8, seed=1), rand(2, sk, hk, 8, seed=2), rand(2, sk, hk, 8, seed=3)
    out = fa.flash_attn_func(q, k, v, causal=causal, softmax_scale=0.3)
    want = exact_attention(q.numpy(), k.numpy(), v.numpy(), causal, 0.3)
    assert np.allclose(out.numpy(), want, atol=1e-5)


def test_a_decode_query_sees_every_key_and_the_default_scale_is_one_over_root_d():
    q, k, v = rand(1, 1, 2, 16, seed=4), rand(1, 9, 2, 16, seed=5), rand(1, 9, 2, 16, seed=6)
    out = fa.flash_attn_func(q, k, v, causal=True)
    assert np.allclose(out.numpy(), exact_attention(q.numpy(), k.numpy(), v.numpy(), False, 16 ** -0.5), atol=1e-5)


def test_a_query_that_sees_no_key_gives_zeros():
    # Sq > Sk with a bottom-right mask: the first query rows attend no key, and flash-attn returns zeros for them.
    q, k, v = rand(1, 4, 1, 8, seed=7), rand(1, 2, 1, 8, seed=8), rand(1, 2, 1, 8, seed=9)
    out = fa.flash_attn_func(q, k, v, causal=True)
    assert torch.all(out[0, :2] == 0) and torch.all(torch.isfinite(out))


def test_the_output_keeps_the_input_dtype_and_computes_in_float32():
    q, k, v = (rand(1, 6, 2, 8, seed=s, dtype=torch.bfloat16) for s in (10, 11, 12))
    out = fa.flash_attn_func(q, k, v, causal=True)
    assert out.dtype == torch.bfloat16
    want = exact_attention(q.float().numpy(), k.float().numpy(), v.float().numpy(), True, None)
    assert np.allclose(out.float().numpy(), torch.tensor(want, dtype=torch.float32).to(torch.bfloat16).float().numpy(), atol=1e-2)


def test_varlen_attends_within_each_sequence_only():
    lengths = [3, 5, 2]
    q, k, v = rand(sum(lengths), 2, 8, seed=13), rand(sum(lengths), 2, 8, seed=14), rand(sum(lengths), 2, 8, seed=15)
    cu = torch.tensor([0, 3, 8, 10], dtype=torch.int32)
    out = fa.flash_attn_varlen_func(q, k, v, cu, cu, 5, 5, causal=True)
    for a, b in ((0, 3), (3, 8), (8, 10)):
        alone = fa.flash_attn_func(q[a:b][None], k[a:b][None], v[a:b][None], causal=True)[0]
        assert torch.allclose(out[a:b], alone)


@pytest.mark.parametrize("change", [{"dropout_p": 0.1}, {"window_size": (4, 0)}, {"softcap": 30.0}, {"alibi_slopes": torch.ones(2)}, {"return_attn_probs": True}])
def test_what_it_lacks_raises(change):
    q = rand(1, 3, 2, 8)
    with pytest.raises(NotImplementedError):
        fa.flash_attn_func(q, q, q, **change)
    with pytest.raises(NotImplementedError):
        fa.flash_attn_with_kvcache(q, q, q)


def test_the_signatures_keep_flash_attn_names_that_transformers_reads():
    import inspect
    names = set(inspect.signature(fa.flash_attn_func).parameters)
    assert {"dropout_p", "softmax_scale", "causal", "window_size", "softcap", "alibi_slopes", "deterministic", "return_attn_probs"} <= names
    assert {"cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "block_table"} <= set(inspect.signature(fa.flash_attn_varlen_func).parameters)


def test_unpad_and_pad_are_inverse():
    hidden = rand(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    packed, indices, cu, longest, used = fa.bert_padding.unpad_input(hidden, mask)
    assert packed.shape == (5, 3) and cu.tolist() == [0, 3, 5] and longest == 3 and used.tolist() == [3, 2]
    back = fa.bert_padding.pad_input(packed, indices, 2, 4)
    assert torch.equal(back * mask[..., None], hidden * mask[..., None])


# ----------------------------------------------------------------------------------------------- bitsandbytes


def test_rows_quantize_to_127_over_their_absmax_with_ties_to_even():
    A = torch.tensor([[1.0, -2.0, 0.5, 4.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float16)
    out, stats, cols = F.int8_vectorwise_quant(A)
    assert stats.tolist() == [4.0, 0.0] and cols is None
    assert out[0].tolist() == [32, -64, 16, 127]  # 31.75 -> 32, 15.875 -> 16
    assert out[1].tolist() == [0, 0, 0, 0]
    ties = torch.tensor([[2.5 * 4 / 127, 4.0], [3.5 * 4 / 127, 4.0]], dtype=torch.float16)
    q, _, _ = F.int8_vectorwise_quant(ties)
    expect = [round(float(ties[0, 0]) * 127 / 4), round(float(ties[1, 0]) * 127 / 4)]
    assert q[:, 0].tolist() == expect


def test_outliers_leave_the_absmax_and_their_columns_become_zero():
    A = torch.tensor([[1.0, 8.0, 2.0], [0.5, 1.0, -7.0]], dtype=torch.float16)
    out, stats, cols = F.int8_vectorwise_quant(A, threshold=6.0)
    assert cols.tolist() == [1, 2]
    assert stats.tolist() == [2.0, 1.0]
    assert out[:, 1:].abs().sum() == 0 and out[0, 0] == 64 and out[1, 0] == 64


def test_one_row_keeps_its_other_values_in_outlier_columns():
    A = torch.tensor([[1.0, 8.0, 2.0]], dtype=torch.float16)
    out, _, cols = F.int8_vectorwise_quant(A, threshold=6.0)
    assert cols.tolist() == [1] and out[0].tolist() == [64, 0, 127]


def test_the_int8_product_is_exact_where_float32_alone_is_not():
    A = torch.full((2, 4864), 127, dtype=torch.int8)
    B = torch.full((3, 4864), 127, dtype=torch.int8)
    B[1] = -127
    got = F.int8_linear_matmul(A, B)
    assert got.dtype == torch.int32 and got[0].tolist() == [4864 * 16129, -4864 * 16129, 4864 * 16129]
    rng = np.random.default_rng(0)
    a, b = rng.integers(-127, 128, (5, 3000)), rng.integers(-127, 128, (7, 3000))
    exact = a.astype(np.int64) @ b.astype(np.int64).T
    assert np.array_equal(F.int8_linear_matmul(torch.tensor(a, dtype=torch.int8), torch.tensor(b, dtype=torch.int8)).numpy(), exact)


def test_the_dequant_follows_the_cuda_kernel():
    A = torch.tensor([[1000, -3]], dtype=torch.int32)
    rows, cols = torch.tensor([2.5]), torch.tensor([0.75, 4.0])
    want = (np.float32(1000) * np.float32(2.5) * np.float32(0.75) * np.float32(F.MM_DEQUANT_CONST))
    out = F.int8_mm_dequant(A, rows, cols)
    assert out.dtype == torch.float16 and float(out[0, 0]) == float(np.float16(want))
    fp16_bias = torch.tensor([1.0, 2.0], dtype=torch.float16)
    bf16_bias = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    inside = F.int8_mm_dequant(A, rows, cols, bias=fp16_bias)
    after = F.int8_mm_dequant(A, rows, cols, bias=bf16_bias, dtype=torch.bfloat16)
    assert float(inside[0, 0]) == float(np.float16(want + np.float32(1.0)))
    assert after.dtype == torch.bfloat16 and float(after[0, 0]) == float(torch.tensor(float(np.float16(np.float16(want) + np.float16(1.0)))).to(torch.bfloat16))


def reference_linear(x, weight, bias, threshold):
    """LLM.int8 written out by hand in float64 from the steps of bitsandbytes, with each rounding where the CUDA path has it."""
    W = weight.to(torch.float16).float().numpy().astype(np.float64)
    w_stats = np.abs(W).max(axis=1)
    CB = np.round(W * (127.0 / w_stats)[:, None])
    A = x.reshape(-1, x.shape[-1]).to(torch.float16).float().numpy().astype(np.float64)
    out_cols = np.where((np.abs(A) >= threshold).any(axis=0))[0] if threshold > 0 else np.array([], dtype=int)
    kept = np.where(np.abs(A) >= threshold, 0.0, A) if threshold > 0 else A
    a_stats = np.abs(kept).max(axis=1)
    CA = np.round(kept * (127.0 / a_stats)[:, None])
    if len(A) > 1:
        CA[:, out_cols] = 0
    y = (CA @ CB.T) * a_stats[:, None] * w_stats[None, :] / (127 * 127)
    y = y + bias.float().numpy()
    x_full = x.reshape(-1, x.shape[-1]).float().numpy().astype(np.float64)
    y = y + x_full[:, out_cols] @ (CB[:, out_cols] * w_stats[:, None] / 127).T
    return y


def test_a_linear_layer_matches_llm_int8_by_hand():
    torch.manual_seed(0)
    layer = bnb.nn.Linear8bitLt(64, 32, bias=True, has_fp16_weights=False, threshold=6.0)
    weight = torch.randn(32, 64) * 0.1
    layer.weight = bnb.nn.Int8Params(weight.clone(), requires_grad=False, has_fp16_weights=False)
    layer.bias = torch.nn.Parameter(torch.randn(32).to(torch.bfloat16), requires_grad=False)
    layer = layer.to("cpu")
    assert layer.weight.dtype == torch.int8 and layer.weight.SCB is not None
    x = torch.randn(1, 5, 64).clamp(-4, 4).to(torch.bfloat16)
    x[0, 2, 7] = 9.0  # an outlier column
    with torch.no_grad():
        out = layer(x)
    want = reference_linear(x, weight, layer.bias, 6.0)
    assert out.dtype == torch.bfloat16 and out.shape == (1, 5, 32)
    assert np.linalg.norm(out.float().numpy().reshape(5, 32) - want) / np.linalg.norm(want) < 6e-3
    assert layer.state.idx.tolist() == [7]


def test_the_weight_quantizes_on_a_move_from_the_cpu_and_keeps_its_scales():
    # As in bitsandbytes: data on the CPU that is not int8 quantizes on any move that is not to meta, the CPU itself included.
    weight = torch.randn(8, 16)
    param = bnb.nn.Int8Params(weight, requires_grad=False, has_fp16_weights=False)
    assert param.to("meta").dtype == torch.float32
    quantized = param.to("cpu", dtype=torch.float16)
    assert quantized.dtype == torch.int8 and quantized.SCB.shape == (8,) and torch.equal(quantized.CB, quantized.data)
    assert torch.equal(quantized.SCB, weight.to(torch.float16).abs().amax(dim=1).float())
    again = quantized.to("cpu", dtype=torch.float16)
    assert again.dtype == torch.int8 and torch.equal(again.SCB, quantized.SCB) and torch.equal(again.CB, quantized.CB)


def test_training_and_four_bit_raise():
    layer = bnb.nn.Linear8bitLt(4, 2, has_fp16_weights=True)
    with pytest.raises(NotImplementedError):
        layer(torch.randn(1, 4))
    with pytest.raises(NotImplementedError):
        bnb.nn.Linear4bit(4, 2)
    with pytest.raises(NotImplementedError):
        bnb.nn.Params4bit.from_prequantized()


def test_the_optimizers_and_the_packed_functions_raise():
    for name in ("Adam8bit", "PagedAdamW8bit", "GlobalOptimManager"):
        with pytest.raises(NotImplementedError):
            getattr(bnb.optim, name)
    with pytest.raises(NotImplementedError):
        from cumlx.compat.bitsandbytes_int8.optim import AdamW8bit  # noqa: F401
    assert not hasattr(bnb.optim, "__wrapped__")  # import machinery and hasattr still see an ordinary module
    q = rand(1, 3, 2, 8)
    for name in ("flash_attn_qkvpacked_func", "flash_attn_kvpacked_func", "flash_attn_varlen_qkvpacked_func", "flash_attn_varlen_kvpacked_func"):
        with pytest.raises(NotImplementedError):
            getattr(fa, name)(q)


@pytest.mark.gpu
def test_the_int8_product_is_exact_on_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    rng = np.random.default_rng(1)
    for k in (896, 4864):
        a, b = rng.integers(-127, 128, (9, k)), rng.integers(-127, 128, (300, k))
        a[0] = 127
        b[0] = 127
        got = F.int8_linear_matmul(torch.tensor(a, dtype=torch.int8, device="mps"), torch.tensor(b, dtype=torch.int8, device="mps")).cpu().numpy()
        assert np.array_equal(got, a.astype(np.int64) @ b.astype(np.int64).T)


# ----------------------------------------------------------------------------------------------- distributions


def test_each_adapter_declares_the_distribution_that_cumlx_run_reports():
    for module in (fa, bnb):
        name, version, top = torch_cuda.ADAPTER_DISTRIBUTIONS[module.__name__]
        assert (name, version) == module.DISTRIBUTION and version.endswith("+cumlx") and version == module.__version__


class _Recorder:
    def __init__(self):
        self.calls = []

    def call(self, *args):
        self.calls.append(args)


def test_a_known_adapter_gets_a_distribution_and_others_get_none(monkeypatch):
    recorder = _Recorder()
    hooks = torch_cuda._Hooks("mps", recorder, {"flash_attn": "cumlx.compat.flash_attn_sdpa", "bitsandbytes": "cumlx.compat.bitsandbytes_int8", "gsplat": "cumlx.compat.gsplat_metalsplat"})
    found = list(hooks.find_distributions(importlib.metadata.DistributionFinder.Context(name="flash_attn")))
    assert [d.metadata["Name"] for d in found] == ["flash-attn"] and found[0].version == "2.8.3+cumlx"
    assert "It is not flash-attn" in found[0].metadata["Summary"]
    assert found[0].read_text("top_level.txt").split() == ["flash_attn"]
    names = sorted(d.metadata["Name"] for d in hooks.find_distributions(importlib.metadata.DistributionFinder.Context()))
    assert names == ["bitsandbytes", "flash-attn"]
    assert list(hooks.find_distributions(importlib.metadata.DistributionFinder.Context(name="gsplat"))) == []
    assert ("distribution", "flash-attn", "cumlx.compat.flash_attn_sdpa 2.8.3+cumlx") in recorder.calls


def test_a_known_adapter_under_another_import_name_gets_none():
    hooks = torch_cuda._Hooks("mps", _Recorder(), {"bnb": "cumlx.compat.bitsandbytes_int8"})
    assert list(hooks.find_distributions(importlib.metadata.DistributionFinder.Context())) == []


def test_the_rules_name_the_versions_that_the_adapters_report():
    limits = json.loads((REPO / "validation" / "m6-f1-limits.json").read_text())
    assert "flash-attn 2.8.3+cumlx" in limits["parts"]["program_mac"] and "bitsandbytes 0.50.2+cumlx" in limits["parts"]["program_mac"]


def test_the_scale_table_is_the_measured_file_and_stays_within_one_ulp_of_division():
    import hashlib
    record = json.loads((REPO / "validation" / "bnb-fdividef-sm70.json").read_text())
    assert hashlib.sha256(F.SCALE_TABLE.read_bytes()).hexdigest() == record["sha256"]
    bits = np.arange(65536, dtype=np.uint16)
    h = np.abs(bits.view(np.float16).astype(np.float32))
    table = np.frombuffer(F.SCALE_TABLE.read_bytes(), dtype="<f4")
    ok = np.isfinite(h) & (h > 0)
    with np.errstate(divide="ignore"):
        exact = (np.float32(127.0) / h[ok]).astype(np.float32)
    assert np.abs(table[ok].view(np.int32).astype(np.int64) - exact.view(np.int32).astype(np.int64)).max() <= 1
    assert np.isinf(table[0])


def test_the_activation_of_f1_attempt_2_rounds_as_on_cuda():
    # Row 2 of model.layers.14.self_attn.o_proj: absmax 2.34375, and -1.171875 * __fdividef(127, 2.34375) = -63.499996, which rounds to -63.
    A = torch.tensor([[2.34375, -1.171875, 0.5]], dtype=torch.float16)
    out, stats, _ = F.int8_vectorwise_quant(A)
    assert out[0].tolist()[:2] == [127, -63]
    assert float(F.cuda_row_scale(stats)[0]) == float(np.float32(54.186665))
