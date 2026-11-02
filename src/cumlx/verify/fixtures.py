"""Tiny random models for offline end-to-end tests (requires torch + transformers)."""
from __future__ import annotations

import json
from pathlib import Path

from ..errors import missing_dependency

TINY_LLAMA = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    rms_norm_eps=1e-5, rope_theta=10000.0, attention_bias=True, mlp_bias=False,
    tie_word_embeddings=False,
    # Larger than the HF default (0.02) so attention is not near-uniform; otherwise
    # position faults (RoPE, mask) barely change the output and cannot be tested.
    initializer_range=0.25,
)


def make_tiny_llama(out_dir: str | Path, *, seed: int = 0, tie_word_embeddings: bool = False) -> dict:
    """Write a 2-layer random Llama (float32 safetensors + config.json) loadable by HF and mlx-lm.

    Norm weights and biases are randomized so that faults such as a dropped
    norm weight or bias change the output (default init would hide them).
    """
    try:
        import torch
    except ImportError:
        raise missing_dependency('torch', 'capture', purpose='fixture') from None
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        raise missing_dependency('transformers', 'capture', purpose='fixture') from None
    torch.manual_seed(seed)
    settings = {**TINY_LLAMA, 'tie_word_embeddings': tie_word_embeddings}
    config = LlamaConfig(**settings)
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('norm.weight'):
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
            elif name.endswith('.bias'):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
    out = Path(out_dir)
    model.save_pretrained(out, safe_serialization=True)
    n_params = sum(p.numel() for p in model.parameters())
    return {'path': str(out), 'architecture': 'LlamaForCausalLM', 'parameters': int(n_params),
            'config': settings, 'seed': seed, 'dtype': 'float32'}


TINY_MOE = dict(
    vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, num_experts=4, num_experts_per_tok=2, head_dim=16,
    max_position_embeddings=128, rms_norm_eps=1e-5, tie_word_embeddings=False, sliding_window=None,
    use_sliding_window=False, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
    initializer_range=0.25,
)
TINY_MOE_ROPE_THETA = 10000.0


def make_tiny_moe(out_dir: str | Path, *, seed: int = 0) -> dict:
    """Write a 2-layer random Qwen3 MoE model (float32 safetensors + config.json) loadable by HF and mlx-lm."""
    try:
        import torch
    except ImportError:
        raise missing_dependency('torch', 'capture', purpose='fixture') from None
    try:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError:
        raise missing_dependency('transformers', 'capture', purpose='fixture') from None
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_MOE)).to(torch.float32).eval()
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith('norm.weight'):
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=gen))
            elif name.endswith('.gate.weight'):  # distinct routing per token, so an expert fault shows
                p.copy_(1.0 * torch.randn(p.shape, generator=gen))
    out = Path(out_dir)
    model.save_pretrained(out, safe_serialization=True)
    # transformers 5 writes rope_theta under rope_parameters; mlx-lm 0.32 reads the top-level keys.
    path = out / 'config.json'
    config = json.loads(path.read_text())
    config.update({k: TINY_MOE[k] for k in ('num_experts', 'mlp_only_layers', 'decoder_sparse_step', 'norm_topk_prob')})
    config['rope_theta'] = TINY_MOE_ROPE_THETA
    path.write_text(json.dumps(config, indent=1) + '\n')
    n_params = sum(p.numel() for p in model.parameters())
    return {'path': str(out), 'architecture': 'Qwen3MoeForCausalLM', 'parameters': int(n_params),
            'config': dict(TINY_MOE), 'seed': seed, 'dtype': 'float32'}
