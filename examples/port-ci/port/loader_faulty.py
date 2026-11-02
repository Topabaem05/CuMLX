"""A port with a defect that raises no error: the RoPE base is wrong. The action must catch it."""
import mlx.nn as nn

from cumlx.verify.mlx_port import load_port


def load(path: str):
    model = load_port(path)
    attention = model.model.layers[1].self_attn
    rope = attention.rope
    attention.rope = nn.RoPE(rope.dims, traditional=rope.traditional, base=rope.base * 10, scale=rope.scale)
    return model
