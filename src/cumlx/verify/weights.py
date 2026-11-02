"""Static weight comparison: the original checkpoint against an MLX port, without running either model.

This finds transposed, rescaled, sign-flipped, missing or wrongly shaped weights quickly, and it names
the parameter. It proves nothing about the port's code; ``cumlx diff`` does that.

Each parameter is judged by what its port dtype can lose:

* float32, float16, bfloat16: rounding only, so ``rel_l2`` has a small bound.
* quantized: a large ``rel_l2`` is normal, so the verdict uses the direction (``cosine``) and the
  least-squares scale (``scale``), which a real defect moves and quantization does not.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, cast

import numpy as np

from ..errors import CumlxError, missing_dependency
from . import mlx_port
from .mapping import NameMapper
from .safetensors_io import SafetensorsReader

WEIGHTS_SCHEMA = 'cumlx.weights-report.v1'
# Unit roundoff: float16 4.9e-4, bfloat16 2.0e-3. float32 must be exact up to the source dtype.
REL_L2_LIMITS = {'float32': 1e-6, 'float16': 1e-3, 'bfloat16': 5e-3}
# Measured on Qwen2.5-0.5B and SmolLM2-135M at 4 and 8 bits; see docs/CLI.md.
QUANTIZED_MIN_COSINE = 0.9
QUANTIZED_MAX_SCALE_ERROR = 0.05


def checkpoint_files(original: str, revision: str | None = None) -> list[Path]:
    root = Path(original)
    if not root.exists():
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise missing_dependency('huggingface_hub', 'diff', purpose='reading a checkpoint by HF id') from None
        try:
            root = Path(snapshot_download(original, revision=revision,
                                          allow_patterns=['*.safetensors', '*.safetensors.index.json']))
        except Exception as exc:  # noqa: BLE001 - the hub raises many types
            raise CumlxError('E_MODEL_LOAD', f'could not fetch {original!r}: {type(exc).__name__}: {exc}',
                             hint='check the HF id; private or gated repos need `huggingface-cli login`') from None
    if root.is_file():
        return [root]
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        return [root / f for f in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
    files = sorted(root.glob('*.safetensors'))
    if not files:
        raise CumlxError('E_INPUT_NOT_FOUND', f'no *.safetensors in {root}',
                         hint='only safetensors checkpoints can be read without torch')
    return files


def _diagnose(port: np.ndarray, orig: np.ndarray) -> str | None:
    """Name the defect when the two arrays are an exact simple transform of each other."""
    def close(a, b) -> bool:
        return a.shape == b.shape and float(np.linalg.norm(a - b)) <= 1e-2 * max(float(np.linalg.norm(b)), 1e-30)
    if port.ndim == 2 and close(port.T, orig):
        return 'transposed'
    if close(-port, orig):
        return 'sign_flipped'
    dot = float(np.sum(orig * orig))
    if dot > 0:
        scale = float(np.sum(port * orig)) / dot
        if abs(scale - 1) > 1e-2 and close(port, scale * orig):
            return f'scaled_by_{scale:.6g}'
    return None


TIED_EMBEDDING = 'model.embed_tokens.weight'
EXPERT_PROJ = {'w1': 'gate_proj', 'w2': 'down_proj', 'w3': 'up_proj',
               'gate_proj': 'gate_proj', 'up_proj': 'up_proj', 'down_proj': 'down_proj'}


def _expert_parts(name: str, orig: np.ndarray, weights: dict[str, np.ndarray]) -> list[tuple[str, str, int, np.ndarray]] | None:
    """Original MoE expert tensors as (row name, stacked port name, expert index, original array).

    Handles one tensor per expert (``experts.3.up_proj.weight``, ``experts.3.w3.weight``) and the
    fused transformers 5 layout (``experts.gate_up_proj`` = gate and up, concatenated on axis 1).
    """
    def stacked(block: str, proj: str) -> str | None:
        for candidate in (block, block.replace('.mlp', '.block_sparse_moe')):
            key = f'{candidate}.switch_mlp.{proj}.weight'
            if key in weights:
                return key
        return None
    single = re.match(r'(.+)\.(?:experts)\.(\d+)\.(\w+)\.weight$', name)
    if single and single.group(3) in EXPERT_PROJ:
        key = stacked(single.group(1).replace('.block_sparse_moe', '.mlp'), EXPERT_PROJ[single.group(3)])
        return [(name, key, int(single.group(2)), orig)] if key else None
    fused = re.match(r'(.+)\.experts\.(gate_up_proj|down_proj)$', name)
    if fused is None:
        return None
    block, tail = fused.groups()
    rows = []
    for e in range(orig.shape[0]):
        if tail == 'down_proj':
            parts = [('down_proj', orig[e])]
        else:
            gate, up = np.split(orig[e], 2, axis=0)
            parts = [('gate_proj', gate), ('up_proj', up)]
        for proj, arr in parts:
            key = stacked(block, proj)
            if key is None:
                return None
            rows.append((f'{block}.experts.{e}.{proj}.weight', key, e, arr))
    return rows


def _other_expert(port: np.ndarray, orig: np.ndarray, expert: int) -> int | None:
    """The expert index whose port weights hold this original expert, when it is not its own."""
    for e in range(port.shape[0]):
        if e != expert and _diagnose_same(port[e], orig):
            return e
    return None


def _diagnose_same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.linalg.norm(a.astype(np.float64) - b)) <= 0.5 * float(np.linalg.norm(b))


def _compare(port: np.ndarray, orig: np.ndarray, kind: str) -> dict[str, Any]:
    if port.shape != orig.shape:
        mismatch: dict[str, Any] = {'verdict': 'fail', 'reason': 'shape_mismatch', 'port_shape': list(port.shape),
                                    'original_shape': list(orig.shape)}
        if port.ndim == 2 and port.T.shape == orig.shape:
            mismatch['diagnosis'] = 'transposed' if _diagnose(port, orig) == 'transposed' else 'shape_is_transposed'
        return mismatch
    o, p = orig.astype(np.float64).ravel(), port.astype(np.float64).ravel()
    if not np.isfinite(p).all():
        return {'verdict': 'fail', 'reason': 'nonfinite_port_weight'}
    on, pn, dn = float(np.linalg.norm(o)), float(np.linalg.norm(p)), float(np.linalg.norm(p - o))
    rel = dn / on if on > 0 else (0.0 if dn == 0 else float('inf'))
    cosine = float(np.dot(p, o) / (pn * on)) if pn > 0 and on > 0 else (1.0 if pn == on else 0.0)
    scale = float(np.dot(p, o) / (on * on)) if on > 0 else 1.0
    out: dict[str, Any] = {'rel_l2': rel, 'cosine': cosine, 'scale': scale}
    if kind == 'quantized':
        ok = cosine >= QUANTIZED_MIN_COSINE and abs(scale - 1) <= QUANTIZED_MAX_SCALE_ERROR
    else:
        ok = rel <= REL_L2_LIMITS.get(kind, REL_L2_LIMITS['float32'])
    out['verdict'] = 'pass' if ok else 'fail'
    if not ok:
        out['reason'] = 'quantized_direction_or_scale' if kind == 'quantized' else 'rel_l2_above_limit'
        d = _diagnose(port.astype(np.float64), orig.astype(np.float64))
        if d:
            out['diagnosis'] = d
    return out


def run_weights(original: str, port: str, *, loader: str = 'mlx_lm', mapping: str | None = None,
                revision: str | None = None, model=None) -> dict[str, Any]:
    mx, _ = mlx_port.mlx()
    from mlx.utils import tree_flatten
    mapper = NameMapper.from_file(mapping)
    readers = {}
    for f in checkpoint_files(original, revision):
        reader = SafetensorsReader(f)
        for name in reader:
            readers[name] = reader
    if not readers:
        raise CumlxError('E_INPUT_INVALID', f'no tensors in the checkpoint {original!r}')
    if model is None:
        model = mlx_port.load_port(port, loader)
    weights = mlx_port.dequantized_weights(model)
    modules = mlx_port.named_modules(model)
    kinds: dict[str, str] = {}
    flat = cast('list[tuple[str, Any]]', tree_flatten(model.parameters()))
    for param, value in flat:
        if mx.issubdtype(value.dtype, mx.floating):
            kinds[param] = str(value.dtype).replace('mlx.core.', '')
    for prefix, m in modules.items():
        if hasattr(m, 'scales') and hasattr(m, 'bits'):
            kinds[(prefix + '.' if prefix else '') + 'weight'] = 'quantized'
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in sorted(readers):
        target = mapper.map(name)
        if mapper.is_ignored(name):
            rows.append({'name': name, 'status': 'ignored'})
            continue
        orig_dtype_is_float = readers[name].dtype_tag(name) in ('F64', 'F32', 'F16', 'BF16')
        parts = _expert_parts(target, readers[name].get(name), weights) if orig_dtype_is_float and target not in weights else None
        if parts:
            for row_name, key, expert, arr in parts:
                seen.add(key)
                c = _compare(weights[key][expert], arr, kinds.get(key, 'float32'))
                if c['verdict'] == 'fail':
                    other = _other_expert(weights[key], arr, expert)
                    if other is not None:
                        c['diagnosis'] = f'expert_order: this expert is at index {other} in the port'
                rows.append({'name': row_name, 'port_name': f'{key}[{expert}]', 'kind': kinds.get(key, 'float32'),
                             'status': c.pop('verdict'), **c})
            continue
        if target not in weights and target == 'lm_head.weight' and TIED_EMBEDDING in weights:
            target = TIED_EMBEDDING  # the port shares the embedding, so the original head must equal it
        if target not in weights:
            rows.append({'name': name, 'port_name': target, 'status': 'unverified',
                         'reason': 'no such parameter in the port'})
            continue
        seen.add(target)
        orig = readers[name].get(name)
        if not np.issubdtype(orig.dtype, np.floating):
            rows.append({'name': name, 'port_name': target, 'status': 'ignored', 'reason': f'non-float {orig.dtype}'})
            continue
        c = _compare(weights[target], orig, kinds.get(target, 'float32'))
        rows.append({'name': name, 'port_name': target, 'kind': kinds.get(target, 'float32'),
                     'status': c.pop('verdict'), **c})
    extra = sorted(set(weights) - seen)
    counts = {k: sum(r['status'] == k for r in rows) for k in ('pass', 'fail', 'unverified', 'ignored')}
    verdict = 'diverged' if counts['fail'] else ('incomplete' if counts['unverified'] else 'ok')
    if not counts['pass'] and verdict == 'ok':
        verdict = 'incomplete'
    worst = next((r for r in rows if r['status'] == 'fail'), None)
    return {'schema': WEIGHTS_SCHEMA, 'verdict': verdict, 'original': original, 'port': port,
            'summary': {**counts, 'parameters': len(rows), 'extra_in_port': len(extra)},
            'first_failure': worst, 'extra_in_port': extra, 'parameters': rows}
