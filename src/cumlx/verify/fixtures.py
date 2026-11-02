"""Tiny random models for offline end-to-end tests (requires torch + transformers)."""
from __future__ import annotations

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
