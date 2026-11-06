#!/usr/bin/env python3
"""F1 module by module: record a golden on one machine, and replay its inputs on another. The rules are in validation/m6-f1-limits.json.

The model loads as spikes/m6_f1_program.py loads it: flash_attention_2, BitsAndBytesConfig(load_in_8bit=True), device_map {'': 0}.
It has no Mac-specific line. Run it through cumlx run:

    # the V100, with bitsandbytes itself. FlashAttention-2 does not run on its sm70, so attention is the adapter there too
    python -m cumlx run --target cuda spikes/m6_f1_record.py --substitute flash_attn=cumlx.compat.flash_attn_sdpa -- capture --output golden/
    # the Mac, with both adapters
    cumlx run spikes/m6_f1_record.py --substitute flash_attn=cumlx.compat.flash_attn_sdpa --substitute bitsandbytes=cumlx.compat.bitsandbytes_int8 -- replay --golden golden/ --output replay.json
    ... -- replay --golden golden/ --output fault.json --fault no_outlier_split

capture writes golden/tensors.npz (inputs and outputs of each Linear8bitLt, decoder layer, the final norm, lm_head and each attention
call of the prefill and of three decode steps), golden/cb.npz (each layer's int8 weights and their scales) and golden/summary.json.
replay feeds each recorded input to this machine's module and writes the measured differences. spikes/m6_f1_judge.py applies the rules.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

MODEL = 'Qwen/Qwen2.5-0.5B'
REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'
PROMPT = 'The capital of France is'
DECODE_STEPS_RECORDED = 3
NEW_TOKENS = 20
FAULTS = ('no_outlier_split', 'activation_scale_128', 'bias_dropped', 'causal_top_left', 'no_softmax_scale')


def to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().to('cpu').float().numpy()


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, attn_implementation='flash_attention_2',
                                                 quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map={'': 0})
    return tokenizer, model


def int8_layers(model: Any) -> dict[str, Any]:
    return {name: module for name, module in model.named_modules() if type(module).__name__ == 'Linear8bitLt'}


def int8_weights(module: Any) -> tuple[Any, Any]:
    state = module.state
    return (state.CB if state.CB is not None else module.weight.CB), (state.SCB if state.SCB is not None else module.weight.SCB)


class AttentionRecorder:
    """Wraps flash_attn.flash_attn_func before Transformers imports it. The wrapper keeps the signature, which Transformers reads."""

    def __init__(self) -> None:
        import flash_attn
        self.module = flash_attn
        self.real = flash_attn.flash_attn_func
        self.signature = inspect.signature(self.real)
        self.calls: list[dict[str, Any]] = []
        self.varlen_calls = 0
        self.active = False
        self.decode_steps_seen = 0
        real_varlen = flash_attn.flash_attn_varlen_func

        @functools.wraps(self.real)
        def recording(*args: Any, **kwargs: Any) -> Any:
            out = self.real(*args, **kwargs)
            if self.active:
                bound = self.signature.bind(*args, **kwargs)
                bound.apply_defaults()
                q, k, v = bound.arguments['q'], bound.arguments['k'], bound.arguments['v']
                self.calls.append({'q': q, 'k': k, 'v': v, 'out': out, 'causal': bool(bound.arguments['causal']), 'softmax_scale': bound.arguments['softmax_scale']})
            return out

        @functools.wraps(real_varlen)
        def counting(*args: Any, **kwargs: Any) -> Any:
            self.varlen_calls += 1
            return real_varlen(*args, **kwargs)

        flash_attn.flash_attn_func = recording
        flash_attn.flash_attn_varlen_func = counting


def recording_hook(save: Any, in_key: str, out_key: str) -> Any:
    """A forward hook that saves the module's first input and its output, and returns None.

    torch replaces a module's output with whatever a forward hook returns when that is not None. The first form of this script used
    lambdas that returned a tuple, which turned each layer's output into (None, None) (found by review before any run, 2026-11-06).
    """
    def hook(module: Any, args: Any, out: Any) -> None:
        save(in_key, args[0])
        save(out_key, out)
    return hook


def capture(output: Path) -> int:
    import torch
    output.mkdir(parents=True, exist_ok=False)
    recorder = AttentionRecorder()
    tokenizer, model = load()
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {'leaves': [], 'layers': [], 'layer_kwargs': {}, 'attention': [], 'dtypes': {}}
    hooks = []

    def save(key: str, tensor: Any) -> None:
        arrays[key] = to_numpy(tensor)
        meta['dtypes'][key] = str(tensor.dtype).replace('torch.', '')

    for name, module in int8_layers(model).items():
        meta['leaves'].append(name)
        hooks.append(module.register_forward_hook(recording_hook(save, f'leaf_in/{name}', f'leaf_out/{name}')))
    for index, layer in enumerate(model.model.layers):
        name = f'model.layers.{index}'
        meta['layers'].append(name)

        def layer_hook(mod: Any, args: Any, kwargs: dict[str, Any], out: Any, name: str = name) -> None:
            hidden = args[0] if args else kwargs['hidden_states']
            save(f'layer_in/{name}', hidden)
            save(f'layer_out/{name}', out[0] if isinstance(out, tuple) else out)
            plain = {}
            for key, value in kwargs.items():
                if key == 'hidden_states':
                    continue
                if isinstance(value, torch.Tensor):
                    save(f'layer_kw/{name}/{key}', value)
                    plain[key] = 'tensor'
                elif isinstance(value, tuple) and all(isinstance(item, torch.Tensor) for item in value):
                    for position, item in enumerate(value):
                        save(f'layer_kw/{name}/{key}/{position}', item)
                    plain[key] = f'tuple:{len(value)}'
                elif value is None or isinstance(value, (bool, int, float, str)):
                    plain[key] = value
                else:
                    plain[key] = f'unsupported:{type(value).__name__}'
            meta['layer_kwargs'][name] = plain
        hooks.append(layer.register_forward_hook(layer_hook, with_kwargs=True))
    for key, module in (('norm', model.model.norm), ('lm_head', model.lm_head)):
        hooks.append(module.register_forward_hook(recording_hook(save, f'{key}_in', f'{key}_out')))

    inputs = tokenizer(PROMPT, return_tensors='pt').to('cuda')
    recorder.active = True
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits
    recorder.active = False
    save('prefill_logits', logits[0])
    for handle in hooks:
        handle.remove()
    prefill_calls = len(recorder.calls)

    # Decode: record the attention calls of the first steps of generation, whose queries are one token long.
    layers = len(model.model.layers)
    original_forward = model.forward
    step = {'n': 0}

    def counted_forward(*args: Any, **kwargs: Any) -> Any:
        recorder.active = 1 <= step['n'] <= DECODE_STEPS_RECORDED
        try:
            return original_forward(*args, **kwargs)
        finally:
            recorder.active = False
            step['n'] += 1
    model.forward = counted_forward
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=NEW_TOKENS, do_sample=False)
    model.forward = original_forward
    tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
    for index, call in enumerate(recorder.calls):
        phase = 'prefill' if index < prefill_calls else 'decode'
        key = f'attn/{index}'
        for part in ('q', 'k', 'v', 'out'):
            save(f'{key}/{part}', call[part])
        meta['attention'].append({'key': key, 'phase': phase, 'layer': index % layers, 'causal': call['causal'], 'softmax_scale': call['softmax_scale'],
                                  'q_shape': list(call['q'].shape), 'k_shape': list(call['k'].shape)})

    weights = {}
    hashes = {}
    for name, module in int8_layers(model).items():
        CB, SCB = int8_weights(module)
        cb = CB.detach().to('cpu').numpy()
        weights[f'CB/{name}'] = cb
        weights[f'SCB/{name}'] = SCB.detach().to('cpu').float().numpy()
        hashes[name] = hashlib.sha256(np.ascontiguousarray(cb).tobytes()).hexdigest()
    np.savez(output / 'tensors.npz', **arrays)
    np.savez_compressed(output / 'cb.npz', **weights)
    import bitsandbytes
    import flash_attn
    import transformers
    summary = {'schema': 'cumlx.m6_f1_golden.v1', 'model': MODEL, 'revision': REVISION, 'prompt': PROMPT, 'input_ids': inputs['input_ids'][0].tolist(), 'tokens': tokens,
               'text': tokenizer.decode(tokens), 'device': str(logits.device), 'dtype': str(model.dtype), 'prefill_attention_calls': prefill_calls,
               'decode_attention_calls': len(recorder.calls) - prefill_calls, 'varlen_calls': recorder.varlen_calls, 'cb_sha256': hashes,
               'bitsandbytes': {'module': bitsandbytes.__name__, 'version': getattr(bitsandbytes, '__version__', None)},
               'flash_attn': {'module': flash_attn.__name__, 'version': getattr(flash_attn, '__version__', None)},
               'torch': torch.__version__, 'transformers': transformers.__version__, **meta}
    (output / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'tokens': len(tokens), 'text': summary['text'], 'leaves': len(meta['leaves']), 'attention_calls': len(recorder.calls), 'device': summary['device']}))
    return 0


def exact_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool, softmax_scale: float | None) -> np.ndarray:
    """Attention in float64 with flash-attn 2.1's bottom-right causal mask. q [B, Sq, Hq, D], k and v [B, Sk, Hk, D]."""
    q, k, v = (np.asarray(t, dtype=np.float64) for t in (q, k, v))
    hq, hk, d = q.shape[2], k.shape[2], q.shape[3]
    scale = d ** -0.5 if softmax_scale is None else float(softmax_scale)
    k = np.repeat(k, hq // hk, axis=2)
    v = np.repeat(v, hq // hk, axis=2)
    scores = np.einsum('bqhd,bkhd->bhqk', q, k) * scale
    sq, sk = q.shape[1], k.shape[1]
    if causal:
        blocked = np.arange(sk)[None, :] > np.arange(sq)[:, None] + (sk - sq)
        scores = np.where(blocked, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum('bhqk,bkhd->bqhd', weights, v)


def difference(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    norm = np.linalg.norm(expected)
    return {'rel_l2': float(np.linalg.norm(actual - expected) / norm) if norm > 0 else float(np.linalg.norm(actual - expected)),
            'max_abs': float(np.abs(actual - expected).max()) if actual.size else 0.0, 'equal_fraction': float(np.mean(actual == expected)) if actual.size else 1.0,
            'finite': bool(np.isfinite(actual).all())}


def apply_fault(name: str) -> None:
    """Patch one adapter function in this process, so that a control can show that the comparison fails."""
    import torch

    from cumlx.compat import flash_attn_sdpa
    from cumlx.compat.bitsandbytes_int8 import functional
    if name == 'no_outlier_split':
        real = functional.int8_vectorwise_quant
        functional.int8_vectorwise_quant = lambda A, threshold=0.0: real(A, 0.0)
    elif name == 'activation_scale_128':
        real = functional.int8_vectorwise_quant

        def scaled(A: Any, threshold: float = 0.0) -> Any:
            out, stats, cols = real(A, threshold)
            if threshold > 0.0:  # activations only: the weights are quantized with threshold 0
                rows = A.reshape(-1, A.shape[-1]).to(torch.float32)
                kept = rows.masked_fill(rows.abs() >= threshold, 0.0)
                wrong = torch.round(kept * (128.0 / stats).unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
                if cols is not None and rows.shape[0] > 1 and cols.numel():
                    wrong[:, cols] = 0
                out = wrong.view(A.shape)
            return out, stats, cols
        functional.int8_vectorwise_quant = scaled
    elif name == 'bias_dropped':
        real = functional.int8_mm_dequant
        functional.int8_mm_dequant = lambda A, row_stats, col_stats, out=None, bias=None, dtype=None: real(A, row_stats, col_stats, out=out, bias=None, dtype=dtype)
    elif name == 'causal_top_left':
        real = flash_attn_sdpa._attend

        def top_left(q: Any, k: Any, v: Any, softmax_scale: Any, causal: bool) -> Any:
            if not causal or q.shape[1] == k.shape[1]:
                return real(q, k, v, softmax_scale, causal)
            return real(q, k[:, :q.shape[1]], v[:, :q.shape[1]], softmax_scale, causal)  # key j <= query i, counted from the top left
        flash_attn_sdpa._attend = top_left
    elif name == 'no_softmax_scale':
        real = flash_attn_sdpa._attend
        flash_attn_sdpa._attend = lambda q, k, v, softmax_scale, causal: real(q, k, v, 1.0, causal)
    else:
        raise SystemExit(f'unknown fault {name!r}: one of {FAULTS}')


def replay(golden: Path, output: Path, fault: str | None) -> int:
    import torch
    if fault:
        apply_fault(fault)
    summary = json.loads((golden / 'summary.json').read_text())
    tensors = np.load(golden / 'tensors.npz')
    dtypes = summary['dtypes']
    tokenizer, model = load()
    device = next(model.parameters()).device

    def tensor(key: str) -> Any:
        return torch.tensor(tensors[key], device=device).to(getattr(torch, dtypes[key]))

    # One forward, so that each Linear8bitLt moves its int8 weights into its state, as in the golden run.
    inputs = tokenizer(PROMPT, return_tensors='pt').to('cuda')
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits
    result: dict[str, Any] = {'schema': 'cumlx.m6_f1_replay.v1', 'fault': fault, 'device': str(device), 'dtype': str(model.dtype),
                              'input_ids_equal': inputs['input_ids'][0].tolist() == summary['input_ids'], 'leaves': {}, 'layers': {}, 'attention': {}, 'weights': {}}
    result['prefill_chain'] = {**difference(to_numpy(logits[0]), tensors['prefill_logits']),
                               'top1_equal_positions': int((to_numpy(logits[0]).argmax(-1) == tensors['prefill_logits'].argmax(-1)).sum()), 'positions': int(logits.shape[1])}
    modules = dict(model.named_modules())
    cb = np.load(golden / 'cb.npz')
    with torch.no_grad():
        for name in summary['leaves']:
            out = modules[name](tensor(f'leaf_in/{name}'))
            result['leaves'][name] = difference(to_numpy(out), tensors[f'leaf_out/{name}'])
            CB, SCB = int8_weights(modules[name])
            mine, theirs = CB.detach().to('cpu').numpy().astype(np.int16), cb[f'CB/{name}'].astype(np.int16)
            result['weights'][name] = {'shape_equal': mine.shape == theirs.shape, 'cb_differ_fraction': float(np.mean(mine != theirs)) if mine.shape == theirs.shape else 1.0,
                                       'cb_max_step': int(np.abs(mine - theirs).max()) if mine.shape == theirs.shape else None,
                                       'scb_equal': bool(np.array_equal(SCB.detach().to('cpu').float().numpy(), cb[f'SCB/{name}']))}
        for name in summary['layers']:
            kwargs: dict[str, Any] = {}
            for key, kind in summary['layer_kwargs'][name].items():
                if kind == 'tensor':
                    kwargs[key] = tensor(f'layer_kw/{name}/{key}')
                elif isinstance(kind, str) and kind.startswith('tuple:'):
                    kwargs[key] = tuple(tensor(f'layer_kw/{name}/{key}/{i}') for i in range(int(kind.split(':')[1])))
                elif isinstance(kind, str) and kind.startswith('unsupported:'):
                    raise SystemExit(f'{name} got {key} of a type that this script cannot replay: {kind}')
                else:
                    kwargs[key] = kind
            out = modules[name](tensor(f'layer_in/{name}'), **kwargs)
            out = out[0] if isinstance(out, tuple) else out
            result['layers'][name] = difference(to_numpy(out), tensors[f'layer_out/{name}'])
        for key, module in (('norm', model.model.norm), ('lm_head', model.lm_head)):
            result['layers'][key] = difference(to_numpy(module(tensor(f'{key}_in'))), tensors[f'{key}_out'])
        import flash_attn
        for call in summary['attention']:
            key = call['key']
            out = flash_attn.flash_attn_func(tensor(f'{key}/q'), tensor(f'{key}/k'), tensor(f'{key}/v'), causal=call['causal'], softmax_scale=call['softmax_scale'])
            exact = exact_attention(tensors[f'{key}/q'], tensors[f'{key}/k'], tensors[f'{key}/v'], call['causal'], call['softmax_scale'])
            result['attention'][key] = {'phase': call['phase'], 'layer': call['layer'], 'against_exact': difference(to_numpy(out), exact),
                                        'golden_against_exact': difference(tensors[f'{key}/out'], exact), 'against_golden': difference(to_numpy(out), tensors[f'{key}/out'])}
    output.write_text(json.dumps(result, indent=1))
    worst_leaf = max((item['rel_l2'] for item in result['leaves'].values()), default=None)
    print(json.dumps({'fault': fault, 'leaves': len(result['leaves']), 'worst_leaf_rel_l2': worst_leaf, 'layers': len(result['layers']), 'attention': len(result['attention'])}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--output', required=True, type=Path)
    r = sub.add_parser('replay')
    r.add_argument('--golden', required=True, type=Path)
    r.add_argument('--output', required=True, type=Path)
    r.add_argument('--fault', choices=FAULTS)
    args = parser.parse_args()
    return capture(args.output) if args.mode == 'capture' else replay(args.golden, args.output, args.fault)


if __name__ == '__main__':
    sys.exit(main())
