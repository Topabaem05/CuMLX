"""Deliberate defects for testing the verifier itself (fault injection).

Each fault mutates a loaded MLX port in place and returns the module name
where the verifier must report the first divergence. A verifier that cannot
locate these is not trustworthy, so they are part of ``cumlx selftest``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..errors import CumlxError
from . import mlx_port


@dataclass(frozen=True)
class Fault:
    id: str
    title: str
    target_kind: str          # what kind of module the target name must be
    expect: str               # 'self' = located at the target; 'parent' = at the target's parent
    apply: Callable[[object, str], None]


def _mod(model, name: str):
    mods = mlx_port.named_modules(model)
    if name not in mods:
        raise CumlxError('E_USAGE', f'no module {name!r} in the port')
    return mods[name]


def _transpose(model, name):
    m = _mod(model, name)
    w = m['weight']
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise CumlxError('E_USAGE', f'F1 needs a square Linear weight; {name} has {tuple(w.shape)}')
    m['weight'] = w.T


def _drop_bias(model, name):
    m = _mod(model, name)
    if 'bias' not in m:
        raise CumlxError('E_USAGE', f'F2 needs a Linear with bias; {name} has none')
    del m['bias']


def _norm_eps(model, name):
    m = _mod(model, name)
    if not hasattr(m, 'eps'):
        raise CumlxError('E_USAGE', f'F3 needs a norm module with eps; {name} has none')
    m.eps = 0.5


def _scale_bump(model, name, *, mode: str, delta: int):
    mx, _ = mlx_port.mlx()
    m = _mod(model, name)
    if getattr(m, 'mode', None) != mode:
        raise CumlxError('E_USAGE', f'this fault needs a {mode} QuantizedLinear; {name} is {getattr(m, "mode", None)}')
    s = m['scales']
    flat = s.reshape(-1)
    bumped = mx.minimum(flat[:1].astype(mx.int32) + delta, 126 if mode == 'nvfp4' else 254).astype(mx.uint8)
    m['scales'] = mx.concatenate([bumped, flat[1:]]).reshape(s.shape)


def _swap_nibbles(model, name):
    mx, _ = mlx_port.mlx()
    m = _mod(model, name)
    if getattr(m, 'mode', None) not in ('nvfp4', 'mxfp4'):
        raise CumlxError('E_USAGE', f'F10 needs an FP4 QuantizedLinear; {name} is {getattr(m, "mode", None)}')
    w = m['weight']
    mask = mx.array(0x0F0F0F0F, dtype=mx.uint32)
    m['weight'] = mx.left_shift(mx.bitwise_and(w, mask), 4) | mx.bitwise_and(mx.right_shift(w, 4), mask)


def _shuffle_experts(model, name):
    mx, _ = mlx_port.mlx()
    m = _mod(model, name)
    switch = getattr(m, 'switch_mlp', None)
    if switch is None:
        raise CumlxError('E_USAGE', f'F9 needs an MoE block with switch_mlp; {name} has none')
    for proj in (switch.gate_proj, switch.up_proj, switch.down_proj):
        for key in ('weight', 'scales', 'biases', 'global_scale'):
            v = proj.get(key)
            if v is not None:
                proj[key] = mx.concatenate([v[1:], v[:1]], axis=0)


def _swap_activation(model, name):
    _, nn = mlx_port.mlx()
    m = _mod(model, name)
    if not all(hasattr(m, a) for a in ('gate_proj', 'up_proj', 'down_proj')):
        raise CumlxError('E_USAGE', f'F6 needs a gated MLP (gate/up/down_proj); {name} is {type(m).__name__}')
    cls = type(m)
    m.__class__ = type(cls.__name__, (cls,), {
        '__call__': lambda self, x: self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))})


def _rope_base(model, name):
    _, nn = mlx_port.mlx()
    m = _mod(model, name)
    rope = getattr(m, 'rope', None)
    if rope is None or not hasattr(rope, 'base'):
        raise CumlxError('E_USAGE', f'F7 needs an attention module with a standard nn.RoPE; {name} has none')
    m.rope = nn.RoPE(rope.dims, traditional=rope.traditional, base=rope.base * 10, scale=rope.scale)


def _rope_ignores_cache_offset(model, name):
    m = _mod(model, name)
    rope = getattr(m, 'rope', None)
    if rope is None:
        raise CumlxError('E_USAGE', f'F11 needs an attention module with a RoPE; {name} has none')
    cls = type(rope)
    base_call = cls.__call__
    rope.__class__ = type(cls.__name__, (cls,), {'__call__': lambda self, x, offset=0: base_call(self, x, 0)})


def _drop_mask(model, name):
    m = _mod(model, name)
    cls = type(m)
    base_call = cls.__call__
    m.__class__ = type(cls.__name__, (cls,), {
        '__call__': lambda self, x, mask=None, cache=None: base_call(self, x, None, cache)})


FAULTS: dict[str, Fault] = {f.id: f for f in [
    Fault('F1', 'Linear weight transposed', 'square Linear', 'self', _transpose),
    Fault('F2', 'Linear bias dropped', 'Linear with bias', 'self', _drop_bias),
    Fault('F3', 'norm epsilon changed to 0.5', 'RMSNorm/LayerNorm', 'self', _norm_eps),
    Fault('F4', 'MXFP4 E8M0 block scale exponent +1', 'mxfp4 QuantizedLinear', 'self',
          lambda mdl, n: _scale_bump(mdl, n, mode='mxfp4', delta=1)),
    Fault('F5', 'NVFP4 E4M3 block scale code +8 (x2)', 'nvfp4 QuantizedLinear', 'self',
          lambda mdl, n: _scale_bump(mdl, n, mode='nvfp4', delta=8)),
    Fault('F6', 'MLP activation SiLU -> GELU', 'gated MLP', 'self', _swap_activation),
    Fault('F7', 'RoPE base x10', 'attention with nn.RoPE', 'self', _rope_base),
    Fault('F8', 'causal attention mask dropped', 'attention(x, mask, cache)', 'self', _drop_mask),
    Fault('F9', 'MoE experts rotated by one index', 'MoE block with switch_mlp', 'experts', _shuffle_experts),
    Fault('F10', 'FP4 nibble order reversed (low/high)', 'FP4 QuantizedLinear', 'self', _swap_nibbles),
    # Visible only in the decode phase: prefill starts at position 0, so its outputs stay correct.
    Fault('F11', 'RoPE ignores the KV cache offset', 'attention with RoPE', 'self', _rope_ignores_cache_offset),
]}


def inject(model, fault_id: str, name: str) -> Fault:
    if fault_id not in FAULTS:
        raise CumlxError('E_USAGE', f'unknown fault {fault_id!r}', details={'known': sorted(FAULTS)})
    f = FAULTS[fault_id]
    f.apply(model, name)
    return f


# Default targets on the `cumlx fixture tiny-llama` model (2 layers, attention bias on).
TINY_LLAMA_TARGETS = {
    'F1': 'model.layers.0.self_attn.q_proj',
    'F2': 'model.layers.1.self_attn.k_proj',
    'F3': 'model.layers.1.input_layernorm',
    'F4': 'model.layers.0.mlp.up_proj',
    'F5': 'model.layers.1.mlp.down_proj',
    'F6': 'model.layers.0.mlp',
    'F7': 'model.layers.1.self_attn',
    'F8': 'model.layers.0.self_attn',
    'F10': 'model.layers.0.self_attn.v_proj',
    'F11': 'model.layers.1.self_attn',
}
# On the `cumlx fixture tiny-moe` model: (module to inject into, PyTorch module the verifier must name).
TINY_MOE_TARGETS = {'F9': ('model.layers.1.mlp', 'model.layers.1.mlp.experts')}
