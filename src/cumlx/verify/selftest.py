"""``cumlx selftest``: prove that the verifier itself works in this environment.

Core checks need only NumPy. End-to-end checks build a tiny random Llama,
capture it with PyTorch, load it with mlx-lm and require that:

* the clean port passes in both modes (no false alarm), and
* every injected fault is located at the expected module (isolated mode),
  including quantized-weight faults against a quantization-aware golden, and
* the static weight diff passes the clean port and names a transposed weight.

A check that cannot run because a dependency is missing is ``skipped``,
never ``pass``.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import util
from pathlib import Path
import time
import traceback

import numpy as np

TOKENS = [1, 5, 9, 23, 7, 99, 3, 64, 12, 8]


def _step(steps: list, sid: str, title: str, fn: Callable[[], dict | None], log) -> bool:
    t0 = time.perf_counter()
    try:
        details = fn() or {}
        status = details.pop('_status', 'pass')
    except AssertionError as exc:
        status, details = 'fail', {'reason': str(exc)}
    except Exception as exc:  # noqa: BLE001
        status, details = 'fail', {'reason': f'{type(exc).__name__}: {exc}',
                                   'traceback': traceback.format_exc(limit=4).splitlines()[-6:]}
    steps.append({'id': sid, 'title': title, 'status': status, 'ms': int((time.perf_counter() - t0) * 1000),
                  **details})
    log(f'{sid} {status}: {title}')
    return status == 'pass'


def _skip(steps: list, sid: str, title: str, missing: list[str]) -> None:
    steps.append({'id': sid, 'title': title, 'status': 'skipped', 'missing': missing,
                  'hint': "python -m pip install 'cumlx-lab[capture,diff]'"})


def run(work: Path, *, core_only: bool = False, device: str = 'auto', log=lambda _m: None) -> list[dict]:
    steps: list[dict] = []
    _core(steps, work, log)
    if core_only:
        return steps
    missing = [m for m in ('torch', 'transformers', 'mlx', 'mlx_lm') if util.find_spec(m) is None]
    e2e = ['E1 fixture', 'E2 capture', 'E3 clean port passes', 'E4 faults located', 'E5 quantized faults located',
           'E7 static weight diff', 'E8 MoE']
    if missing:
        for title in e2e:
            _skip(steps, title.split()[0], title, missing)
    else:
        _end_to_end(steps, work, device, log)
    if util.find_spec('mlx') is None:
        _skip(steps, 'E6', 'conform runs against MLX', ['mlx'])
    else:
        def conform():
            from .conform import run_conform
            rep = run_conform()
            bitwise = [c for c in rep['checks'] if c['kind'] == 'bitwise']
            return {'target_version': rep['target_version'], 'target_verdict': rep['verdict'],
                    'target_findings': rep['summary']['failed_ids'],
                    'note': 'target findings describe MLX, not a selftest failure',
                    'bitwise_checks': len(bitwise)}
        _step(steps, 'E6', 'conform runs against MLX', conform, log)
    return steps


def _core(steps: list, work: Path, log) -> None:
    def safetensors_roundtrip():
        from .safetensors_io import load_file, save_file
        rng = np.random.default_rng(0)
        data = {'a': rng.normal(size=(3, 5)).astype(np.float32), 'b': np.arange(7, dtype=np.int64),
                'c': rng.normal(size=(2, 2)).astype(np.float16), 'z': np.zeros((0, 4), np.float32)}
        p = work / 'roundtrip.safetensors'
        save_file(data, p, {'k': 'v'})
        back = load_file(p)
        assert set(back) == set(data), 'tensor names changed'
        for k in data:
            assert back[k].dtype == data[k].dtype and np.array_equal(back[k], data[k]), f'{k} changed'
        return {'tensors': len(data)}

    def metrics_sanity():
        from .metrics import compare, resolve_threshold
        t = resolve_threshold('float32')
        x = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
        assert compare(x, x, t)['verdict'] == 'pass', 'identical arrays must pass'
        assert compare(x * 1.01, x, t)['verdict'] == 'fail', '1% error must fail at float32'
        assert compare(x[:, :4], x, t)['reason'] == 'shape_mismatch', 'shape mismatch must fail'
        y = x.copy()
        y[0, 0] = np.nan
        assert compare(y, x, t)['reason'] == 'nonfinite_output', 'NaN must fail'
        return {}

    def codec_sanity():
        from ..quant import dequantize, quantize
        w = np.random.default_rng(2).normal(size=(8, 64)).astype(np.float32)
        out = {}
        for fmt in ('nvfp4', 'mxfp4'):
            rel = float(np.linalg.norm(dequantize(quantize(w, fmt)) - w) / np.linalg.norm(w))
            assert rel < 0.2, f'{fmt} round-trip error {rel:.3f} is implausible'
            out[f'{fmt}_roundtrip_rel_l2'] = rel
        return out

    _step(steps, 'C1', 'safetensors write/read round trip', safetensors_roundtrip, log)
    _step(steps, 'C2', 'metrics accept identical and reject perturbed outputs', metrics_sanity, log)
    _step(steps, 'C3', 'FP4 codec round trip', codec_sanity, log)


def _end_to_end(steps: list, work: Path, device: str, log) -> None:
    import mlx.nn as nn

    from . import diff, faults, mlx_port
    from .capture_torch import capture_model, load_hf_model, load_mlx_weights_into
    from .fixtures import make_tiny_llama
    from .trace import TraceWriter, prepare_output_dir

    tiny = work / 'tiny-llama'
    state: dict = {}

    def capture(dst: Path, weights=None) -> dict:
        model = load_hf_model(str(tiny))
        if weights is not None:
            load_mlx_weights_into(model, weights)
        out = prepare_output_dir(dst, overwrite=True)
        writer = TraceWriter(out, source={'framework': 'torch', 'model': str(tiny)},
                             golden={'kind': 'reference' if weights is None else 'quantization-aware',
                                     'dtype': 'float32'}, max_bytes=1 << 30)
        stats = capture_model(model, [np.asarray(TOKENS)], writer, decode_steps=1)
        writer.finish()
        return stats

    def located(rep, expected: str) -> dict:
        fd = rep['first_divergence']
        assert rep['verdict'] == 'diverged', f"fault not detected (verdict {rep['verdict']})"
        assert fd and fd['module'] == expected, f"located {fd and fd['module']!r}, expected {expected!r}"
        return {'located': fd['module'], 'basis': fd['basis'], 'confidence': fd['confidence']}

    if not _step(steps, 'E1', 'fixture: tiny random Llama', lambda: make_tiny_llama(tiny), log):
        return
    if not _step(steps, 'E2', 'capture golden trace with PyTorch',
                 lambda: capture(work / 'golden'), log):
        return

    def clean():
        out = {}
        for mode in ('isolated', 'chain'):
            rep = diff.run_diff(str(work / 'golden'), str(tiny), mode=mode, device=device)
            assert rep['verdict'] == 'ok', f"clean port gave {rep['verdict']} in {mode} mode: {rep['first_divergence']}"
            out[mode] = rep['summary']
        worst = max(e['chain']['rel_l2'] for e in rep['records'] if e.get('chain'))
        out['worst_chain_rel_l2'] = worst
        out['execution'] = rep['execution']
        state['clean'] = True
        return out

    _step(steps, 'E3', 'clean port passes in isolated and chain modes (no false alarm)', clean, log)

    def fault_suite():
        results, missed = {}, []
        for fid in ('F1', 'F2', 'F3', 'F6', 'F7', 'F8', 'F11'):
            model = mlx_port.load_port(str(tiny))
            target = faults.TINY_LLAMA_TARGETS[fid]
            faults.inject(model, fid, target)
            rep = diff.run_diff(str(work / 'golden'), str(tiny), model=model, mode='isolated', device=device)
            try:
                results[fid] = {'title': faults.FAULTS[fid].title, **located(rep, target)}
            except AssertionError as exc:
                missed.append(fid)
                results[fid] = {'title': faults.FAULTS[fid].title, 'error': str(exc)}
        assert not missed, f'faults not located: {missed}: {results}'
        return {'faults': results}

    _step(steps, 'E4', 'injected faults F1-F3, F6-F8, F11 are located (isolated mode)', fault_suite, log)

    def quantized_suite():
        results = {}
        for fid, mode, gs in (('F4', 'mxfp4', 32), ('F5', 'nvfp4', 16), ('F10', 'nvfp4', 16)):
            def port(gs=gs, mode=mode):
                m = mlx_port.load_port(str(tiny))
                nn.quantize(m, group_size=gs, bits=4, mode=mode,
                            class_predicate=lambda p, mod: isinstance(mod, nn.Linear) and 'lm_head' not in p)
                return m
            clean_port = port()
            gdir = work / f'golden-{mode}'
            capture(gdir, mlx_port.dequantized_weights(clean_port))
            rep = diff.run_diff(str(gdir), f'tiny-{mode}', model=clean_port, device=device)
            assert rep['verdict'] == 'ok', f'clean {mode} port vs quantization-aware golden: {rep["verdict"]}'
            faulty = port()
            target = faults.TINY_LLAMA_TARGETS[fid]
            faults.inject(faulty, fid, target)
            rep = diff.run_diff(str(gdir), f'tiny-{mode}', model=faulty, device=device)
            results[fid] = {'title': faults.FAULTS[fid].title, **located(rep, target)}
        return {'faults': results}

    _step(steps, 'E5', 'quantized-weight faults F4 (MXFP4), F5 and F10 (NVFP4) are located', quantized_suite, log)

    def static_weights():
        from .weights import run_weights
        rep = run_weights(str(tiny), str(tiny))
        assert rep['verdict'] == 'ok', f"clean weights gave {rep['verdict']}: {rep['first_failure']}"
        model = mlx_port.load_port(str(tiny))
        target = faults.TINY_LLAMA_TARGETS['F1']
        faults.inject(model, 'F1', target)
        rep = run_weights(str(tiny), 'faulty', model=model)
        first = rep['first_failure']
        assert first and first['name'] == target + '.weight' and first.get('diagnosis') == 'transposed', str(first)
        return {'parameters': rep['summary']['parameters'], 'located': first['name'], 'diagnosis': first['diagnosis']}

    _step(steps, 'E7', 'static weight diff passes a clean port and names a transposed weight', static_weights, log)

    def moe():
        from .fixtures import make_tiny_moe
        from .weights import run_weights
        path = work / 'tiny-moe'
        make_tiny_moe(path)
        model = load_hf_model(str(path))
        out = prepare_output_dir(work / 'golden-moe', overwrite=True)
        writer = TraceWriter(out, source={'framework': 'torch', 'model': str(path)},
                             golden={'kind': 'reference', 'dtype': 'float32'}, max_bytes=1 << 30)
        capture_model(model, [np.asarray(TOKENS)], writer, decode_steps=1)
        writer.finish()
        rep = diff.run_diff(str(out), str(path), device=device)
        assert rep['verdict'] == 'ok', f"clean MoE port gave {rep['verdict']}: {rep['first_divergence']}"
        assert run_weights(str(path), str(path))['verdict'] == 'ok', 'clean MoE weights failed'
        inject_at, expected = faults.TINY_MOE_TARGETS['F9']
        faulty = mlx_port.load_port(str(path))
        faults.inject(faulty, 'F9', inject_at)
        rep = diff.run_diff(str(out), str(path), model=faulty, device=device)
        result = {'F9': {'title': faults.FAULTS['F9'].title, **located(rep, expected)}}
        first = run_weights(str(path), 'faulty', model=faulty)['first_failure']
        assert first and 'expert_order' in first.get('diagnosis', ''), f'weights did not name the expert order: {first}'
        return {'faults': result, 'weights_diagnosis': first['diagnosis']}

    _step(steps, 'E8', 'MoE: clean port passes, and rotated experts (F9) are located', moe, log)
