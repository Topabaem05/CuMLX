"""Golden trace capture from a PyTorch model with forward hooks.

Only the PyTorch side is touched here. The model must already be loaded; use
``load_hf_model`` for HF Hub ids or local directories.
"""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Callable, Sequence

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
    kwargs: dict[str, Any] = dict(revision=revision, trust_remote_code=trust_remote_code)
    try:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, **kwargs)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, **kwargs)
    except (OSError, ValueError) as exc:
        raise CumlxError('E_MODEL_LOAD', f'could not load PyTorch model {model_id!r}: {exc}',
                         hint='check the HF id/path; private or gated repos need `huggingface-cli login`') from None
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
            raise CumlxError('E_INPUT_INVALID', f'prompt {p!r} produced no tokens')
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


def _replay_inputs(args: tuple, kwargs: dict) -> tuple[list[tuple[str, Any]], str | None]:
    """Return ([(source, tensor)], note). Replayable only with exactly one tensor and no other inputs."""
    torch = _torch()
    tensors, others = [], []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor):
            tensors.append((f'arg{i}', a))
        elif a is not None:
            others.append(f'arg{i}:{type(a).__name__}')
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            tensors.append((f'kw:{k}', v))
        elif v is not None and v is not False:
            others.append(f'kw:{k}:{type(v).__name__}')
    if len(tensors) == 1 and not others:
        return tensors, None
    names = [s for s, _ in tensors] + others
    return [], 'not replayable in isolation (beta): inputs = ' + ', '.join(names or ['none'])


def capture_model(model, samples: Sequence[np.ndarray], writer: TraceWriter, *,
                  include: str | None = None, exclude: str | None = None, store_dtype: str = 'float32',
                  device: str = 'cpu', call: Callable[[Any, Any], Any] | None = None,
                  log: Callable[[str], None] = lambda _m: None) -> dict:
    """Run ``model`` once per sample and record every module call into ``writer``."""
    torch = _torch()
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    names = {mod: name for name, mod in model.named_modules()}
    has_params = {mod: any(True for _ in mod.parameters()) for mod in names}
    selected = {mod for mod, n in names.items()
                if n == '' or ((inc is None or inc.search(n)) and not (exc and exc.search(n)))}
    call = call or (lambda m, ids: m(input_ids=ids, use_cache=False))
    stats = defaultdict(int)
    for idx, ids in enumerate(samples):
        tensors: dict[str, np.ndarray] = {'token_ids': np.asarray(ids, dtype=np.int64)[None, :]}
        records: list[ModuleRecord] = []
        calls: dict[str, int] = defaultdict(int)

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
            inputs, note = _replay_inputs(args, kwargs)
            in_refs = []
            for j, (src, t) in enumerate(inputs):
                arr = _to_numpy(t, store_dtype)
                tensors[f'{prefix}.in{j}'] = arr
                in_refs.append(TensorRef(f'{prefix}.in{j}', list(arr.shape), str(arr.dtype), src))
            records.append(ModuleRecord(
                key=f'{name}#{n_call}', name=name, call=n_call, order=order, type=type(mod).__name__,
                has_params=has_params[mod], replayable=bool(in_refs), replay_note=note, inputs=in_refs,
                output=TensorRef(prefix + '.out', list(out_np.shape), str(out_np.dtype), out_src)))

        handles = [m.register_forward_hook(hook, with_kwargs=True) for m in selected]
        try:
            with torch.no_grad():
                call(model, torch.as_tensor(tensors['token_ids'], device=device))
        except Exception as e:  # noqa: BLE001 - surfaced as an actionable input error
            raise CumlxError('E_MODEL_RUN', f'PyTorch model failed on sample {idx}: {type(e).__name__}: {e}') from None
        finally:
            for h in handles:
                h.remove()
        writer.add_sample(tensors, records, num_tokens=int(tensors['token_ids'].shape[1]))
        stats['records'] += len(records)
        stats['replayable'] += sum(r.replayable for r in records)
        log(f'sample {idx}: {len(records)} module calls recorded')
    stats['samples'] = len(samples)
    stats['bytes'] = writer.total_bytes
    return dict(stats)


def load_mlx_weights_into(model, port_weights: dict[str, np.ndarray]) -> dict:
    """Copy (dequantized) MLX port weights into a PyTorch model by parameter name."""
    torch = _torch()
    state = model.state_dict()
    loaded, shape_mismatch = [], []
    for name, arr in port_weights.items():
        if name not in state:
            continue
        if tuple(state[name].shape) != arr.shape:
            shape_mismatch.append({'name': name, 'torch': list(state[name].shape), 'port': list(arr.shape)})
            continue
        with torch.no_grad():
            state[name].copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(state[name].dtype))
        loaded.append(name)
    missing = sorted(set(k for k in state if not k.endswith('inv_freq')) - set(loaded))
    if shape_mismatch:
        raise CumlxError('E_INPUT_INVALID', 'MLX port weights do not match the PyTorch model shapes',
                         details={'mismatches': shape_mismatch[:20]})
    return {'loaded': len(loaded), 'missing_in_port': missing,
            'unused_port_weights': sorted(set(port_weights) - set(loaded))}
