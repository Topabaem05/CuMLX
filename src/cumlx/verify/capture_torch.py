"""Golden trace capture from a PyTorch model with forward hooks.

Only the PyTorch side is touched here. The model must already be loaded; use
``load_hf_model`` for HF Hub ids or local directories.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..errors import CumlxError, missing_dependency
from .trace import ModuleRecord, TensorRef, TraceWriter


def _torch():
    try:
        import torch
    except ImportError:
        raise missing_dependency('torch', 'capture', purpose='capture') from None
    return torch


def load_hf_model(model_id: str, *, device: str = 'cpu', revision: str | None = None,
                  trust_remote_code: bool = False):
    torch = _torch()
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise missing_dependency('transformers', 'capture', purpose='capture') from None
    kwargs: dict[str, Any] = dict(revision=revision, trust_remote_code=trust_remote_code, output_loading_info=True)
    try:
        try:
            model, info = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kwargs)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            model, info = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, **kwargs)
    except (OSError, ValueError) as exc:
        raise CumlxError('E_MODEL_LOAD', f'could not load PyTorch model {model_id!r}: {exc}',
                         hint='check the HF id/path; private or gated repos need `huggingface-cli login`') from None
    # transformers only logs a missing weight and fills it with random values. A golden from that is meaningless.
    tied = {f'{name}.weight' for name in tied_modules(model)}
    missing = sorted(set(info.get('missing_keys') or []) - tied)
    if missing:
        raise CumlxError('E_MODEL_LOAD', f'{model_id!r} does not hold these weights for {type(model).__name__}, so they '
                                         f'would be random: {missing[:5]}',
                         hint='the checkpoint and its config describe different models; this repository may need its own code',
                         details={'missing': missing[:20], 'count': len(missing)})
    try:
        model = model.to(device)
    except (RuntimeError, AssertionError) as exc:
        raise CumlxError('E_BACKEND_UNAVAILABLE', f'torch device {device!r} is not usable: {exc}',
                         hint='use --device cpu') from None
    return model.eval()


def tokenize(model_id: str, prompts: Sequence[str], *, revision: str | None, trust_remote_code: bool,
             max_tokens: int) -> list[np.ndarray]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise missing_dependency('transformers', 'capture', purpose='capture') from None
    try:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)
    except (OSError, ValueError) as exc:
        raise CumlxError('E_MODEL_LOAD', f'no usable tokenizer for {model_id!r}: {exc}',
                         hint='pass --token-ids instead of --prompt') from None
    out = []
    for p in prompts:
        ids = np.asarray(tok(p)['input_ids'], dtype=np.int64)[:max_tokens]
        if ids.size == 0:
            raise CumlxError('E_INPUT_INVALID', f'prompt {p!r} produced no tokens', hint='this tokenizer is empty; pass --token-ids instead')
        out.append(ids)
    return out


def _to_numpy(t, store_dtype: str) -> np.ndarray:
    torch = _torch()
    t = t.detach().to('cpu')
    if t.is_floating_point():
        t = t.to(torch.float32)
        a = t.numpy()
        return a.astype(np.float16) if store_dtype == 'float16' else a
    if t.dtype == torch.bool:
        return t.numpy()
    return t.to(torch.int64).numpy()


def _first_tensor(obj) -> tuple[Any, str] | tuple[None, None]:
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        return obj, 'tensor'
    if hasattr(obj, 'to_tuple') and callable(obj.to_tuple):  # transformers ModelOutput
        items = obj.to_tuple()
        if items and isinstance(items[0], torch.Tensor):
            return items[0], 'ModelOutput[0]'
    if isinstance(obj, (tuple, list)) and obj and isinstance(obj[0], torch.Tensor):
        return obj[0], 'tuple[0]'
    return None, None


CACHE_KWARGS = ('use_cache', 'past_key_values', 'cache_position')
LAYER_INDEX = re.compile(r'(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)')


def _replay_inputs(args: tuple, kwargs: dict) -> tuple[list[tuple[str, Any]], str | None, str | None]:
    """Return ([(source, tensor)], replay_kind, note).

    ``single``: exactly one tensor and nothing else. ``hidden``: a float [batch, time, dim] hidden
    state plus context such as a causal mask or position embeddings, which the port recomputes.
    Capture never pads its inputs, so that context is always plain causal.
    """
    torch = _torch()
    tensors, others = [], []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            tensors.append((f'arg{i}', a))
        elif a is not None:
            others.append(f'arg{i}:{type(a).__name__}')
    for k, v in kwargs.items():
        if k in CACHE_KWARGS:  # the cache is replayed by filling the port's own cache
            continue
        if isinstance(v, torch.Tensor):
            tensors.append((f'kw:{k}', v))
        elif v is not None and v is not False:
            others.append(f'kw:{k}:{type(v).__name__}')
    if len(tensors) == 1 and not others:
        return tensors, 'single', None
    context = ', '.join([name for name, _ in tensors] + others or ['none'])
    primary = next((x for x in tensors if x[0] in ('arg0', 'kw:hidden_states')), None)
    if primary is not None and primary[1].is_floating_point() and primary[1].ndim == 3:
        return [primary], 'hidden', f'replayed with the hidden state and a causal mask; recorded inputs = {context}'
    return [], None, f'not replayable in isolation: inputs = {context}'


def tied_modules(model) -> dict[str, str]:
    """Modules whose parameter is the same object as an earlier module's, e.g. lm_head -> embed_tokens."""
    first_owner: dict[int, str] = {}
    ties: dict[str, str] = {}
    for mod_name, mod in model.named_modules():
        for _, param in mod.named_parameters(recurse=False):
            owner = first_owner.setdefault(id(param), mod_name)
            if owner != mod_name:
                ties[mod_name] = owner
    return ties


def resolve_revision(model_id: str, revision: str | None, model=None) -> str | None:
    """The commit that was actually loaded. A local directory has none."""
    found = getattr(getattr(model, 'config', None), '_commit_hash', None)
    if found:
        return found
    if Path(model_id).exists():
        return revision
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_id, 'config.json', revision=revision)
    except Exception:  # noqa: BLE001 - provenance is best effort; the caller records what it has
        return revision
    return Path(cached).parent.name if isinstance(cached, str) else revision  # .../snapshots/<commit>/config.json


def capture_model(model, samples: Sequence[np.ndarray], writer: TraceWriter, *,
                  include: str | None = None, exclude: str | None = None, store_dtype: str = 'float32',
                  device: str = 'cpu', call: Callable[[Any, Any], Any] | None = None, decode_steps: int = 0,
                  layers: Sequence[int] | None = None, log: Callable[[str], None] = lambda _m: None) -> dict:
    """Run ``model`` once per sample and record every module call into ``writer``.

    ``decode_steps`` > 0 also records that many greedy steps on top of the KV cache. The next token
    comes from the golden logits, so the port receives the same tokens as the original.
    ``layers`` records only those decoder layers, plus every module outside the layer stack. It keeps
    the trace small on a large model; the diff then covers only what was recorded, and says so.
    """
    torch = _torch()
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    names = {mod: name for name, mod in model.named_modules()}
    has_params = {mod: any(True for _ in mod.parameters()) for mod in names}
    keep = set(layers) if layers is not None else None

    def in_layers(n: str) -> bool:
        match = LAYER_INDEX.search(n)
        return keep is None or match is None or int(match.group(1)) in keep

    selected = {mod for mod, n in names.items()
                if n == '' or (in_layers(n) and (inc is None or inc.search(n)) and not (exc and exc.search(n)))}
    if keep is not None or inc or exc:
        writer.manifest.source['partial'] = {'layers': sorted(keep) if keep is not None else None,
                                             'include': include, 'exclude': exclude}
    call = call or (lambda m, ids: m(input_ids=ids, use_cache=False))
    stats = defaultdict(int)
    writer.manifest.source.setdefault('tied_modules', tied_modules(model))
    for idx, ids in enumerate(samples):
        tensors: dict[str, np.ndarray] = {'token_ids': np.asarray(ids, dtype=np.int64)[None, :]}
        records: list[ModuleRecord] = []
        calls: dict[str, int] = defaultdict(int)
        phase = 'prefill'

        def hook(mod, args, kwargs, output):
            name = names[mod]
            n_call = calls[name]
            calls[name] += 1
            out_t, out_src = _first_tensor(output)
            if out_t is None:
                stats['no_tensor_output'] += 1
                return
            order = len(records)
            prefix = f'r{order:05d}'
            out_np = _to_numpy(out_t, store_dtype)
            tensors[prefix + '.out'] = out_np
            inputs, replay_kind, note = _replay_inputs(args, kwargs)
            if name.rsplit('.', 1)[-1] == 'experts' and len(args) == 3 and all(isinstance(a, torch.Tensor) for a in args):
                # (hidden, top-k indices, top-k weights): the golden routing tests the experts alone
                inputs = [(f'arg{i}', a) for i, a in enumerate(args)]
                replay_kind, note = 'experts', 'replayed with the golden routing'
            if phase == 'decode':
                if replay_kind == 'hidden':
                    replay_kind = 'hidden_cached'
                elif replay_kind == 'single' and kwargs.get('past_key_values') is not None:
                    # A float input is a hidden state, as in the LFM2 convolution: only token ids reach all layers.
                    replay_kind = 'hidden_cached' if inputs[0][1].is_floating_point() else 'tokens_cached'
            if replay_kind == 'hidden' and not has_params[mod]:  # e.g. a rotary embedding: nothing to verify
                inputs, replay_kind = [], None
            in_refs = []
            for j, (src, t) in enumerate(inputs):
                arr = _to_numpy(t, store_dtype)
                tensors[f'{prefix}.in{j}'] = arr
                in_refs.append(TensorRef(f'{prefix}.in{j}', list(arr.shape), str(arr.dtype), src))
            records.append(ModuleRecord(
                key=f'{name}#{n_call}', name=name, call=n_call, order=order, type=type(mod).__name__,
                has_params=has_params[mod], replayable=bool(in_refs), replay_note=note, inputs=in_refs,
                output=TensorRef(prefix + '.out', list(out_np.shape), str(out_np.dtype), out_src),
                replay_kind=replay_kind, phase=phase))

        handles = [m.register_forward_hook(hook, with_kwargs=True) for m in selected]
        steps_done = 0
        try:
            with torch.no_grad():
                ids_t = torch.as_tensor(tensors['token_ids'], device=device)
                if decode_steps:
                    out = model(input_ids=ids_t, use_cache=True)
                    chosen = []
                    phase = 'decode'
                    for _ in range(decode_steps):
                        if getattr(out, 'past_key_values', None) is None:
                            stats['no_kv_cache'] += 1
                            break
                        next_id = out.logits[:, -1].argmax(-1, keepdim=True)
                        chosen.append(next_id)
                        out = model(input_ids=next_id, past_key_values=out.past_key_values, use_cache=True)
                        steps_done += 1
                    if chosen:
                        tensors['decode_token_ids'] = torch.cat(chosen, dim=1).to('cpu').to(torch.int64).numpy()
                else:
                    call(model, ids_t)
        except Exception as e:  # noqa: BLE001 - surfaced as an actionable input error
            raise CumlxError('E_MODEL_RUN', f'PyTorch model failed on sample {idx}: {type(e).__name__}: {e}') from None
        finally:
            for h in handles:
                h.remove()
        writer.add_sample(tensors, records, num_tokens=int(tensors['token_ids'].shape[1]), decode_steps=steps_done)
        stats['records'] += len(records)
        stats['replayable'] += sum(r.replayable for r in records)
        log(f'sample {idx}: {len(records)} module calls recorded')
    stats['samples'] = len(samples)
    stats['bytes'] = writer.total_bytes
    return dict(stats)


def _moe_names_to_torch(port_weights: dict[str, np.ndarray], state: dict) -> dict[str, np.ndarray]:
    """Rewrite stacked ``switch_mlp`` expert weights to the names and layout of this PyTorch model."""
    out = {}
    stacked: dict[str, dict[str, np.ndarray]] = {}
    for name, arr in port_weights.items():
        mixtral = re.match(r'(.+)\.block_sparse_moe\.(.+)$', name)
        if mixtral and f'{mixtral.group(1)}.mlp.gate.weight' in state:
            name = f'{mixtral.group(1)}.mlp.{mixtral.group(2)}'  # transformers 5 calls this block `mlp`
        match = re.match(r'(.+)\.switch_mlp\.(gate_proj|up_proj|down_proj)\.weight$', name)
        if match:
            stacked.setdefault(match.group(1), {})[match.group(2)] = arr
        else:
            out[name] = arr
    for block, parts in stacked.items():
        if len(parts) != 3:
            out.update({f'{block}.switch_mlp.{k}.weight': v for k, v in parts.items()})
        elif f'{block}.experts.gate_up_proj' in state:
            out[f'{block}.experts.gate_up_proj'] = np.concatenate([parts['gate_proj'], parts['up_proj']], axis=1)
            out[f'{block}.experts.down_proj'] = parts['down_proj']
        else:
            for proj, arr in parts.items():
                out.update({f'{block}.experts.{e}.{proj}.weight': arr[e] for e in range(arr.shape[0])})
    return out


def _conv_to_torch_layout(model, name: str, arr: np.ndarray, target) -> np.ndarray:
    """MLX convolutions keep channels last: [out, kernel, in] against PyTorch [out, in, kernel]."""
    torch = _torch()
    if tuple(target.shape) == arr.shape or arr.ndim not in (3, 4) or '.' not in name:
        return arr
    try:
        owner = model.get_submodule(name.rsplit('.', 1)[0])
    except AttributeError:
        return arr
    if not isinstance(owner, torch.nn.modules.conv._ConvNd):
        return arr
    return np.transpose(arr, (0, 2, 1) if arr.ndim == 3 else (0, 3, 1, 2))


def load_mlx_weights_into(model, port_weights: dict[str, np.ndarray]) -> dict:
    """Copy (dequantized) MLX port weights into a PyTorch model by parameter name."""
    torch = _torch()
    state = model.state_dict()
    port_weights = _moe_names_to_torch(port_weights, state)
    loaded, shape_mismatch = [], []
    for name, arr in port_weights.items():
        if name not in state:
            continue
        arr = _conv_to_torch_layout(model, name, arr, state[name])
        if tuple(state[name].shape) != arr.shape:
            shape_mismatch.append({'name': name, 'torch': list(state[name].shape), 'port': list(arr.shape)})
            continue
        with torch.no_grad():
            state[name].copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(state[name].dtype))
        loaded.append(name)
    tied = {f'{name}.weight' for name in tied_modules(model)}
    missing = sorted({k for k in state if not k.endswith('inv_freq')} - set(loaded) - tied)
    if shape_mismatch:
        raise CumlxError('E_INPUT_INVALID', 'MLX port weights do not match the PyTorch model shapes',
                         details={'mismatches': shape_mismatch[:20]})
    unused = sorted(set(port_weights) - set(loaded))
    if missing or unused:
        # A golden that holds only some of the port's weights compares different models. Never write one.
        raise CumlxError('E_INPUT_INVALID', 'the MLX port weights do not cover the PyTorch model, so a golden with the '
                                            "port's weights would be wrong",
                         hint='check the architecture; a parameter name differs between the two',
                         details={'missing_in_port': missing[:20], 'unused_port_weights': unused[:20]})
    return {'loaded': len(loaded), 'missing_in_port': missing, 'unused_port_weights': unused}
