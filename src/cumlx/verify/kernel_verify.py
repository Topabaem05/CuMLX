"""`cumlx kernel verify`: run a Metal kernel on the inputs of an op-level golden and compare the outputs.

An unknown kernel, a kernel that does not compile, an unsupported dtype and a golden kernel with no port are never a
pass: each ends as ``incomplete``. fp64, FTZ, FMA contraction and execution order are stated in every report.
"""
from __future__ import annotations

from collections.abc import Callable
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np

from ..errors import CumlxError, missing_dependency
from . import bf16
from .kernel_capture import PROBE_BODY, PROBE_INPUT, PROBE_OUTPUTS, read_probe
from .kernel_source import IDENT, is_complete_metal, mask_source
from .metrics import KERNEL_THRESHOLDS, compare_kernel_output
from .opgolden import DTYPES, KernelCase, OpGolden, TensorSpec, sha256_text

REPORT_SCHEMA = 'cumlx.kernel-report.v1'
HEADER_SUFFIX = '.header.metal'
METAL_TYPES = {'float32': 'float', 'float16': 'half', 'uint8': 'uint8_t', 'int8': 'int8_t', 'uint16': 'uint16_t',
               'int16': 'int16_t', 'uint32': 'uint32_t', 'int32': 'int32_t', 'uint64': 'uint64_t', 'int64': 'int64_t',
               'bfloat16': bf16.TYPE}
ATTRIBUTES = {'thread_position_in_grid': 'uint3', 'threadgroup_position_in_grid': 'uint3',
              'thread_position_in_threadgroup': 'uint3', 'threads_per_threadgroup': 'uint3',
              'threadgroups_per_grid': 'uint3', 'threads_per_grid': 'uint3', 'grid_size': 'uint3', 'grid_origin': 'uint3',
              'thread_index_in_threadgroup': 'uint', 'thread_index_in_simdgroup': 'uint', 'simdgroup_index_in_threadgroup': 'uint',
              'threads_per_simdgroup': 'uint', 'thread_execution_width': 'uint', 'simdgroups_per_threadgroup': 'uint',
              'thread_index_in_quadgroup': 'uint', 'quadgroup_index_in_threadgroup': 'uint', 'quadgroups_per_threadgroup': 'uint'}
SIMD_BODY = 'y[0] = float(threads_per_simdgroup);\n'


class KernelRunError(Exception):
    """The kernel did not produce outputs. ``stage`` is compile, bind or run."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class Engine(Protocol):
    name: str

    def describe(self) -> dict[str, Any]: ...

    def run(self, case: KernelCase, source: str, header: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


def _poison(spec: TensorSpec) -> np.ndarray:
    dtype = np.dtype(DTYPES[spec.dtype])
    return np.full(spec.shape, np.nan if dtype.kind == 'f' else 0xAB, dtype)


def _classify_error(exc: Exception) -> KernelRunError:
    text = str(exc)
    stage = 'compile' if re.search(r'program_source|Unable to (?:build|load)|[Cc]ompil', text) else 'run'
    return KernelRunError(stage, f'{type(exc).__name__}: {text}'[:1500])


class MLXEngine:
    """Runs a function body through ``mx.fast.metal_kernel`` in math mode ``safe``, as ``MLXBackend`` does."""
    name = 'mlx'

    def __init__(self) -> None:
        try:
            import mlx.core as mx
        except ImportError:
            raise missing_dependency('mlx', 'metal', purpose='`cumlx kernel verify --engine mlx`') from None
        metal = getattr(mx, 'metal', None)
        if metal is None or not metal.is_available() or not hasattr(mx.fast, 'metal_kernel'):
            raise CumlxError('E_BACKEND_UNAVAILABLE', 'MLX found no Metal GPU, so the kernel cannot run.',
                             hint='run this on Apple Silicon. A kernel that did not run is not a pass')
        self.mx = mx
        self._cache: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        from importlib import metadata
        info = self.mx.device_info(self.mx.gpu) if hasattr(self.mx, 'device_info') else {}
        return {'engine': self.name, 'api': 'mx.fast.metal_kernel', 'mlx': metadata.version('mlx'), 'math_mode': 'safe',
                'device': info.get('device_name') or info.get('architecture') or str(self.mx.gpu)}

    def run(self, case: KernelCase, source: str, header: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if case.initial or case.call is not None:
            raise KernelRunError('bind', f'{case.id} sets call or initial, and mx.fast.metal_kernel takes only inputs and new outputs. Use --engine torch')
        if any(t.dtype == 'bfloat16' for t in case.inputs + case.outputs):
            raise KernelRunError('bind', f'{case.id} has bfloat16 buffers. mx.fast.metal_kernel would declare them uint16_t, where an '
                                 'integer converts to float without a word. Use --engine torch, which declares them cumlx_bf16')
        mx = self.mx
        input_names = [t.name for t in case.inputs]
        output_names = [t.name for t in case.outputs]
        digest = hashlib.sha256('\0'.join([source, header, *input_names, '|', *output_names]).encode()).hexdigest()[:20]
        try:
            if digest not in self._cache:
                self._cache[digest] = mx.fast.metal_kernel(
                    name=f'cumlx_kv_{digest}', input_names=input_names, output_names=output_names, source=source, header=header,
                    ensure_row_contiguous=True, compile_options={'math_mode': 'safe'})
            arrays = [mx.array(inputs[name]) for name in input_names]
            mx.eval(*arrays)
            # MLX does not clear an output buffer, and it reuses freed ones, so an element that the kernel never writes holds
            # whatever was there. This step makes that element NaN, or 0xAB in an integer output, so that it reads as
            # nan_mismatch. MLX does not promise this reuse: it is a measured behaviour, not a guarantee. Measured 2026-11-03,
            # a kernel that writes nothing diverged in 5 of 5 runs with this step (nan_mismatch) and in 5 of 5 without it
            # (rel_l2), so the step makes the failure clear. It was not shown to be needed for the failure itself.
            mx.clear_cache()
            poison = [mx.array(_poison(t)) for t in case.outputs]
            mx.eval(*poison)
            del poison
            gc.collect()
            template = [(k, v) for k, v in case.constants.items()]
            outputs = self._cache[digest](
                inputs=arrays, template=template, grid=case.grid, threadgroup=case.threadgroup,
                output_shapes=[t.shape for t in case.outputs],
                output_dtypes=[getattr(mx, t.dtype) for t in case.outputs], stream=mx.gpu)
            mx.eval(*outputs)
            result = {t.name: np.array(o) for t, o in zip(case.outputs, outputs)}
            del outputs
            return result
        except Exception as exc:  # noqa: BLE001 - any failure here means that the kernel did not produce outputs
            raise _classify_error(exc) from None


def metal_constants(case: KernelCase, used: set[str] | None = None) -> str:
    lines = []
    for key, value in case.constants.items():
        if used is not None and key not in used:
            continue
        if isinstance(value, bool):
            lines.append(f'constant bool {key} = {str(value).lower()};\n')
        else:
            lines.append(f'constant int {key} = {value};\n')
    return ''.join(lines)


def wrap_metal_body(case: KernelCase, source: str, header: str, *, entry: str = 'cumlx_kernel') -> str:
    """A complete Metal library around a body, with the signature that ``mx.fast.metal_kernel`` would generate."""
    params = []
    for i, t in enumerate(case.inputs):
        kind = METAL_TYPES[t.dtype]
        params.append(f'const constant {kind}* {t.name} [[buffer({i})]]' if t.shape else f'const constant {kind}& {t.name} [[buffer({i})]]')
    for j, t in enumerate(case.outputs):
        params.append(f'device {METAL_TYPES[t.dtype]}* {t.name} [[buffer({len(case.inputs) + j})]]')
    tokens = set(IDENT.findall(mask_source(source)))
    params += [f'{kind} {name} [[{name}]]' for name, kind in ATTRIBUTES.items() if name in tokens]
    prelude = '#include <metal_stdlib>\nusing namespace metal;\n'
    return f"{prelude}{header}\n{metal_constants(case)}kernel void {entry}({', '.join(params)}) {{\n{source}\n}}\n"


def _complete_signature(source: str, wanted: list[str]) -> tuple[str, list[str]]:
    """Find the entry point of a complete library and the parameter name of each buffer, in buffer order."""
    masked = mask_source(source)
    found = {m.group(1): m.end() for m in re.finditer(r'(?:\[\[\s*kernel\s*\]\]|\bkernel)\s+void\s+([A-Za-z_]\w*)\s*\(', masked)}
    if not found:
        raise KernelRunError('bind', 'no `kernel void` function in this library')
    entry = next((name for name in wanted if name in found), None)
    if entry is None and len(found) == 1:
        entry = next(iter(found))
    if entry is None:
        raise KernelRunError('bind', f'the library has kernels {sorted(found)}, and none is named {wanted}')
    start = found[entry]
    depth, i = 1, start
    while depth and i < len(masked):
        depth += {'(': 1, ')': -1}.get(masked[i], 0)
        i += 1
    params: dict[int, str] = {}
    for m in re.finditer(r'([A-Za-z_]\w*)\s*\[\[\s*buffer\(\s*(\d+)\s*\)\s*\]\]', masked[start:i]):
        params[int(m.group(2))] = m.group(1)
    if sorted(params) != list(range(len(params))):
        raise KernelRunError('bind', f'the buffers of {entry} are not numbered from 0: {params}')
    return entry, [params[k] for k in sorted(params)]


class TorchEngine:
    """Runs a complete library through ``torch.mps.compile_shader``. It also accepts a body, and wraps it first."""
    name = 'torch'

    def __init__(self) -> None:
        try:
            import torch
        except ImportError:
            raise missing_dependency('torch', 'capture', purpose='`cumlx kernel verify --engine torch`') from None
        if not torch.backends.mps.is_available() or not hasattr(torch.mps, 'compile_shader'):
            raise CumlxError('E_BACKEND_UNAVAILABLE', 'PyTorch has no MPS device or no compile_shader, so the kernel cannot run.',
                             hint='use PyTorch 2.8 or later on Apple Silicon. A kernel that did not run is not a pass')
        self.torch = torch
        self._cache: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        return {'engine': self.name, 'api': 'torch.mps.compile_shader', 'torch': self.torch.__version__,
                'math_mode': 'library default (PyTorch sets no option)', 'device': 'mps'}

    def run(self, case: KernelCase, source: str, header: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        if is_complete_metal(source):
            used = set(IDENT.findall(mask_source(source)))
            library = metal_constants(case, used) + source
            prelude = '#include <metal_stdlib>\nusing namespace metal;\n'
            library = prelude + header + '\n' + library
            base = (case.entry or case.kernel).split('<')[0]
            entry, order = _complete_signature(source, [base, case.kernel])
        else:
            library = wrap_metal_body(case, source, header)
            entry, order = 'cumlx_kernel', [t.name for t in case.inputs + case.outputs]
        outputs = {t.name: torch.from_numpy(np.array(inputs[case.initial[t.name]]) if t.name in case.initial else _poison(t).copy()).to('mps')
                   for t in case.outputs}
        missing = sorted(set(outputs) - set(order))
        if missing:
            raise KernelRunError('bind', f'{entry} has no buffer for output(s) {missing}; its buffers are {order}')
        unknown = [name for name in order if name not in outputs and name not in inputs]
        if unknown:
            raise KernelRunError('bind', f'{entry} has buffers that the golden does not have: {unknown}')
        try:
            key = sha256_text(library)
            if key not in self._cache:
                self._cache[key] = torch.mps.compile_shader(library)
            args = [outputs[name] if name in outputs else torch.from_numpy(np.array(inputs[name])).to('mps') for name in order]
            getattr(self._cache[key], entry)(*args, threads=case.grid, group_size=case.threadgroup)
            torch.mps.synchronize()
            return {name: tensor.cpu().numpy().copy() for name, tensor in outputs.items()}
        except Exception as exc:  # noqa: BLE001 - any failure here means that the kernel did not produce outputs
            raise _classify_error(exc) from None


class WoofEngine:
    """Runs a library that woof translated through woof's own launcher, which binds its scalar struct and threadgroup memory.

    ``torch.mps.compile_shader`` sets neither. The signature of each kernel comes from NAME.port.json. A function body goes to
    MLX: only the floating-point probe is one here. It has been tested only with a stub package that has the same names, not
    with woof, until validation/kernel-m3.json says otherwise.
    """
    name = 'woof'

    def __init__(self) -> None:
        if importlib.util.find_spec('woof') is None:
            raise CumlxError('E_KERNEL_TRANSLATOR_MISSING', '`--engine woof` needs the Python package woof, which is not installed. Nothing ran, and this is not a pass.',
                             hint='install it yourself in its own environment if you accept its terms: spikes/m3_kernels.sh')
        os.environ.setdefault('WOOF_BACKEND', 'mlx')
        self._woof = importlib.import_module('woof')  # it installs the cupy facade, so it comes first
        self._facade = importlib.import_module('cupy')
        self._translate = importlib.import_module('woof.backend.translate')
        self._loader = importlib.import_module('woof.backend.metal.loader')
        self._body = MLXEngine()
        self._signatures: dict[str, dict[str, Any]] = {}
        self._modules: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        from importlib import metadata
        try:
            version = metadata.version('metal-woof')
        except metadata.PackageNotFoundError:
            version = None
        return {'engine': self.name, 'api': 'woof.backend.metal.loader', 'woof': version, 'math_mode': 'safe',
                'device': self._body.describe()['device'], 'probe': 'mx.fast.metal_kernel, not the translator: woof sets its own FMA contraction'}

    def use_translations(self, translations: dict[str, Any]) -> None:
        self._signatures = {k: v.get('metal_signatures') or {} for k, v in translations.items()}

    def _unit(self, kernel: str, source: str) -> Any:
        signatures = self._signatures.get(kernel)
        if not signatures:
            raise KernelRunError('bind', f'{kernel}.port.json has no metal_signatures. Translate it with `cumlx kernel port --translator woof`')
        t = self._translate
        kernels = {name: t.KernelSignature(**{**s, 'params': tuple(t.Param(**p) for p in s['params'])}) for name, s in signatures.items()}
        return t.TranslatedUnit(source, kernels, sha256_text(source))

    def run(self, case: KernelCase, source: str, header: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not is_complete_metal(source):
            return self._body.run(case, source, header, inputs)
        key = sha256_text(source)
        entry = (case.entry or case.kernel).split('<')[0]
        by_input = {t.name: t for t in case.inputs}
        made: dict[str, Any] = {}
        args: list[Any] = []
        for name in case.arguments:
            if name in by_input:
                value = inputs[name]
                args.append(self._facade.asarray(np.array(value)) if by_input[name].shape else value[()])
            else:
                spec = next(t for t in case.outputs if t.name == name)
                start = inputs[case.initial[name]] if name in case.initial else _poison(spec)
                made[name] = self._facade.asarray(np.array(start))
                args.append(made[name])
        try:
            if key not in self._modules:
                self._modules[key] = self._loader.MetalModule(self._unit(case.kernel, source), case.kernel)
            self._modules[key].get_function(entry)(case.blocks, case.threadgroup, tuple(args))
            return {name: np.array(self._facade.asnumpy(array)) for name, array in made.items()}
        except KernelRunError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure here means that the kernel did not produce outputs
            raise _classify_error(exc) from None


def select_engine(name: str, *, needs_complete: bool, needs_bf16: bool = False) -> Engine:
    if name == 'woof':
        return WoofEngine()
    if name == 'mlx':
        if needs_complete:
            raise CumlxError('E_USAGE', '--engine mlx takes a function body, and this file is a complete library (`kernel void`).',
                             hint='use --engine torch, or leave out --engine')
        return MLXEngine()
    if name == 'torch':
        return TorchEngine()
    if needs_complete or needs_bf16 or importlib.util.find_spec('mlx') is None:
        return TorchEngine()
    return MLXEngine()


def probe_device(engine: Engine) -> dict[str, Any]:
    """Measure on this device what the golden measured on CUDA, through the same engine as the kernel."""
    case = KernelCase('probe', 'cumlx_probe', [TensorSpec('a', 'float32', (len(PROBE_INPUT),))],
                      [TensorSpec('y', 'float32', (PROBE_OUTPUTS,))])
    simd = KernelCase('probe_simd', 'cumlx_probe_simd', [], [TensorSpec('y', 'float32', (1,))], {}, (32, 1, 1), (32, 1, 1))
    try:
        values = engine.run(case, PROBE_BODY, '', {'a': PROBE_INPUT})['y']
        facts: dict[str, Any] = read_probe(values)
        facts['probe_outputs'] = [float(v) for v in values]
        facts['simd_width'] = int(engine.run(simd, SIMD_BODY, '', {})['y'][0])
    except KernelRunError as exc:
        return {'error': str(exc)}
    return facts


def _differences(golden: OpGolden, measured: dict[str, Any], kernels: dict[str, Any], *,
                 port_repeats: bool, golden_repeats: bool) -> list[dict[str, Any]]:
    fp = golden.floating_point
    ftz, fmad = fp.get('ftz') or {}, fp.get('fmad') or {}
    using_fp64 = sorted(k for k, v in kernels.items() if (v.get('features') or {}).get('fp64') or (v.get('features') or {}).get('fp64_literals'))
    atomics = sorted(k for k, v in kernels.items() if (v.get('features') or {}).get('atomics'))
    warp = sorted(k for k, v in kernels.items() if (v.get('features') or {}).get('warp_level'))
    flush_keys = ('load_flushes', 'input_flushes', 'output_flushes')
    has_probe = bool(measured) and 'error' not in measured and bool(ftz)  # nothing ran: nothing is known about this device
    return [
        {'topic': 'fp64', 'cuda': 'double runs in 64 bits, and an unsuffixed literal such as 1.0 is a double',
         'metal': 'no double; an unsuffixed literal is a float', 'differs': bool(using_fp64), 'kernels': using_fp64,
         'effect': 'These kernels compute part of their work in 64 bits on CUDA and in 32 bits on Metal.' if using_fp64
         else 'No kernel here uses double or an unsuffixed floating literal (text scan; a macro can hide one).'},
        {'topic': 'ftz', 'cuda': {k: ftz.get(k) for k in flush_keys} | {'flag': ftz.get('flag')},
         'metal': {k: measured.get(k) for k in flush_keys},
         'differs': (any(ftz.get(k) != measured.get(k) for k in flush_keys) if has_probe else None),
         'effect': 'Read `subnormal_flushed` in each output. A flushed element changes the result only as much as its size.'},
        {'topic': 'fmad', 'cuda': {'flag': fmad.get('flag'), 'contracts': fmad.get('contracts')},
         'metal': {'contracts': measured.get('contracts')},
         'differs': (fmad.get('contracts') != measured.get('contracts') if has_probe else None),
         'effect': 'a*b+c rounds once when fused and twice when not. Explicit fma calls are the same on both.'},
        {'topic': 'execution_order', 'cuda': f"threads in a block run in no set order, in warps of {fp.get('warp_size', 'unknown')}",
         'metal': f"threads in a threadgroup run in no set order, in SIMD groups of {measured.get('simd_width', 'unknown')}",
         'differs': bool(atomics or warp or not port_repeats or not golden_repeats),
         'kernels_with_atomics': atomics, 'kernels_with_warp_level': warp,
         'golden_repeats_bitwise': golden_repeats, 'port_repeats_bitwise': port_repeats,
         'effect': 'A kernel that sums by lane in a fixed tree gives the same result in any order. '
                   'An atomic float sum does not, and warp-level code depends on the group size.'},
    ]


def resolve_targets(metal: str | Path, golden: OpGolden, kernel: str | None) -> tuple[dict[str, Path], list[str], list[str]]:
    """(golden kernel -> file, files with no golden, golden kernels that nothing covers)."""
    path = Path(metal)
    if not path.exists():
        raise CumlxError('E_INPUT_NOT_FOUND', f'{path} does not exist', hint='pass a .metal file or a folder of them')
    known = {c.kernel for c in golden.cases}
    if path.is_dir():
        if kernel:
            raise CumlxError('E_USAGE', '--kernel maps one file; this is a folder')
        files = {p.name[:-len('.metal')]: p for p in sorted(path.glob('*.metal')) if not p.name.endswith(HEADER_SUFFIX)}
        if not files:
            raise CumlxError('E_INPUT_NOT_FOUND', f'{path} has no .metal files')
        return ({k: p for k, p in files.items() if k in known}, sorted(set(files) - known), sorted(known - set(files)))
    name = kernel or path.name.split('.')[0]
    if name not in known:
        return {}, [name], []
    return {name: path}, [], []


def _header_for(path: Path, explicit: str | None) -> str:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise CumlxError('E_INPUT_NOT_FOUND', f'--header {explicit} does not exist')
        return p.read_text()
    own = path.with_name(path.name[:-len('.metal')] + HEADER_SUFFIX)
    if own.is_file():
        return own.read_text()
    sibling = path.parent / 'common_metal.h'
    return sibling.read_text() if sibling.is_file() else ''


def verify_kernels(metal: str | Path, golden_path: str | Path, *, kernel: str | None = None, cases: list[str] | None = None,
                   header: str | None = None, engine_name: str = 'auto', repeats: int = 3,
                   engine: Engine | None = None, log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    if repeats < 1:
        raise CumlxError('E_USAGE', '--repeats must be 1 or more')
    golden = OpGolden(golden_path)
    targets, unmapped, uncovered = resolve_targets(metal, golden, kernel)
    if cases:
        unknown_cases = sorted(set(cases) - {c.id for c in golden.cases})
        if unknown_cases:
            raise CumlxError('E_USAGE', f'--case names cases that the golden does not have: {unknown_cases}')
    records: list[dict[str, Any]] = []
    translations: dict[str, Any] = {}
    sources: dict[str, tuple[str, str]] = {}
    bf16_kernels: list[str] = []
    for name, path in targets.items():
        text, head = path.read_text(), _header_for(path, header)
        text, head, added = bf16.with_metal_header(text, head, set(IDENT.findall(mask_source(text + head))))
        if added:
            bf16_kernels.append(name)
        sources[name] = (text, head)
        sidecar = path.with_name(path.name[:-len('.metal')] + '.port.json')
        if sidecar.is_file():
            try:
                translations[name] = json.loads(sidecar.read_text())
            except json.JSONDecodeError as exc:
                raise CumlxError('E_INPUT_INVALID', f'{sidecar} is not JSON: {exc}') from None
    needs_complete = any(is_complete_metal(s) for s, _ in sources.values())
    execution: dict[str, Any] = {}
    measured: dict[str, Any] = {}
    if sources and engine is None:
        needs_bf16 = any(t.dtype == 'bfloat16' for c in golden.cases if c.kernel in sources for t in c.inputs + c.outputs)
        engine = select_engine(engine_name, needs_complete=needs_complete, needs_bf16=needs_bf16)
    if engine is not None and hasattr(engine, 'use_translations'):
        engine.use_translations(translations)
    if engine is not None and sources:
        execution = engine.describe()
        measured = probe_device(engine)
    port_repeats, golden_repeats = True, True
    for case in golden.cases:
        if case.kernel not in sources or (cases and case.id not in cases):
            continue
        record: dict[str, Any] = {'case': case.id, 'kernel': case.kernel, 'outputs': {}}
        records.append(record)
        info = golden.info(case.id)
        record['golden_repeat_identical'] = info.get('repeat_identical')
        golden_repeats = golden_repeats and info.get('repeat_identical') is not False
        wide = [t.name for t in case.inputs + case.outputs if t.dtype == 'float64']
        if wide:
            record.update(status='unverified', stage='fp64', reason=f'Metal has no float64, and {wide} are float64 in the golden')
            continue
        inputs, expected = golden.load(case)
        source, head = sources[case.kernel]
        log(f'running {case.id}')
        assert engine is not None
        try:
            runs = [engine.run(case, source, head, inputs) for _ in range(repeats)]
        except KernelRunError as exc:
            record.update(status='unverified', stage=exc.stage, reason=str(exc))
            continue
        identical = all(all(run[t.name].tobytes() == runs[0][t.name].tobytes() for t in case.outputs) for run in runs[1:])
        record['port_repeat_identical'] = identical
        port_repeats = port_repeats and identical
        verdict = 'pass'
        for spec in case.outputs:
            compare = bf16.compare_bits if spec.dtype == 'bfloat16' else compare_kernel_output
            per_run = [compare(run[spec.name], expected[spec.name]) for run in runs]
            worst = next((m for m in per_run if m['verdict'] == 'fail'), per_run[0])
            record['outputs'][spec.name] = worst
            if worst['verdict'] == 'fail':
                verdict = 'fail'
        record['status'] = verdict

    per_kernel: dict[str, dict[str, Any]] = {}
    for name in sorted(sources):
        mine = [r for r in records if r['kernel'] == name]
        statuses = [r['status'] for r in mine]
        status = 'fail' if 'fail' in statuses else 'unverified' if 'unverified' in statuses or not statuses else 'pass'
        rels = [m['rel_l2'] for r in mine for m in r['outputs'].values() if m.get('rel_l2') is not None]
        ulps = [m['max_ulp'] for r in mine for m in r['outputs'].values() if m.get('max_ulp') is not None]
        per_kernel[name] = {'status': status, 'cases': len(mine), 'passed': statuses.count('pass'),
                            'worst_rel_l2': max(rels) if rels else None, 'worst_ulp': max(ulps) if ulps else None,
                            'grade': 'A' if status == 'pass' and golden.kind == 'cuda' else None}
    counts = {s: sum(1 for v in per_kernel.values() if v['status'] == s) for s in ('pass', 'fail', 'unverified')}
    case_counts = {s: sum(1 for r in records if r['status'] == s) for s in ('pass', 'fail', 'unverified')}
    any_fail = counts['fail'] > 0
    any_open = bool(counts['unverified'] or unmapped or uncovered or not records)
    verdict = 'diverged' if any_fail else 'incomplete' if any_open else 'ok'

    first_divergence = None
    for record in records:
        if record['status'] != 'fail':
            continue
        name, metric = next((k, m) for k, m in record['outputs'].items() if m['verdict'] == 'fail')
        first_divergence = {
            'kernel': record['kernel'], 'case': record['case'], 'output': name, 'reason': metric.get('reason'),
            'rel_l2': metric.get('rel_l2'), 'max_rel': metric.get('max_rel'), 'index': metric.get('index'),
            'golden': metric.get('golden'), 'port': metric.get('port'), 'mismatched': metric.get('mismatched'),
            'explanation': f"{record['kernel']}: output {name} of case {record['case']} differs from the CUDA golden "
                           f"({metric.get('reason')}). The translation computes something else on these inputs."}
        break

    kernels_info = {k: golden.kernels.get(k, {}) for k in sources}
    differences = _differences(golden, measured, kernels_info, port_repeats=port_repeats, golden_repeats=golden_repeats)
    execution.update(measured=measured, golden=golden.floating_point, golden_source=golden.source, differences=differences)
    warnings: list[dict[str, Any]] = []

    def warn(code: str, message: str, **extra: Any) -> None:
        warnings.append({'code': code, 'message': message, **extra})

    if golden.kind != 'cuda':
        warn('W_KERNEL_GOLDEN_NOT_CUDA', f'This golden is kind {golden.kind!r}, not a CUDA run, so the result says nothing about CUDA and earns no grade.')
    for name in unmapped:
        warn('W_KERNEL_UNMAPPED', f'{name} has no cases in this golden, so it was not compared. Pass --kernel if its golden has another name.')
    for name in uncovered:
        warn('W_KERNEL_NOT_COVERED', f'The golden has {name}, and no .metal file here covers it.')
    if not unmapped and not uncovered:
        others = sorted({c.kernel for c in golden.cases} - set(sources))
        if others:
            warn('W_KERNEL_PARTIAL', f'The verdict covers only {sorted(sources)}. This golden also has {others}.', others=others)
    if cases:
        warn('W_KERNEL_PARTIAL', f'The verdict covers only the {len(cases)} case(s) named by --case.')
    for record in records:
        if record['status'] == 'unverified':
            warn('W_KERNEL_UNVERIFIED', f"{record['case']}: not compared ({record.get('stage')}): {record.get('reason', '')[:300]}",
                 case=record['case'], stage=record.get('stage'))
    if 'error' in measured:
        warn('W_KERNEL_PROBE_FAILED', f"The floating-point probe did not run, so FTZ and FMA on this device are unknown: {measured['error'][:300]}")
    by_topic = {d['topic']: d for d in differences}
    if by_topic['fp64']['differs']:
        warn('W_KERNEL_FP64', f"{by_topic['fp64']['kernels']} use fp64 on CUDA. Metal has none, so a pass here means only that the float32 result is close.")
    flushed = sum(m.get('subnormal_flushed', 0) for r in records for m in r['outputs'].values())
    if by_topic['ftz']['differs'] or flushed:
        warn('W_KERNEL_FTZ_DIFFERS', f'CUDA and Metal treat subnormal floats differently. {flushed} golden subnormal output element(s) were flushed to zero here.',
             subnormal_flushed=flushed)
    if by_topic['fmad']['differs']:
        warn('W_KERNEL_FMAD_DIFFERS', 'One side fuses a*b+c and the other does not, so such expressions round differently.')
    if by_topic['execution_order']['differs']:
        warn('W_KERNEL_EXEC_ORDER', 'A kernel here uses atomics or warp-level code, or a repeat gave different bytes. Its result can depend on execution order.')
    for name, side in translations.items():
        # A golden may be recorded from the original, or from the prepared source when the original also holds host code.
        ours = {side.get('source_sha256'), side.get('prepared_sha256')} - {None}
        golden_sha = (golden.kernels.get(name) or {}).get('source_sha256')
        if ours and golden_sha and golden_sha not in ours:
            warn('W_KERNEL_SOURCE_MISMATCH', f'{name} was translated from a source that differs from the one in the golden.', kernel=name)

    if bf16_kernels:
        warn('W_KERNEL_BF16_EMULATED', f'{bf16_kernels} name cumlx_bf16, so the integer bfloat16 header of this version was added in front. '
             'Its arithmetic follows sm80 and later; CUDA 12.0 has none on sm70.', kernels=bf16_kernels)
        execution['bf16_header_sha256'] = sha256_text(bf16.METAL_HEADER)
    return {'schema': REPORT_SCHEMA, 'verdict': verdict, 'golden': str(golden.path), 'golden_kind': golden.kind,
            'thresholds': dict(KERNEL_THRESHOLDS), 'first_divergence': first_divergence,
            'summary': {'kernels': counts, 'cases': case_counts, 'unmapped': unmapped, 'not_covered': uncovered,
                        'repeats': repeats, 'grade_a_kernels': sorted(k for k, v in per_kernel.items() if v['grade'] == 'A')},
            'kernels': per_kernel, 'translations': translations, 'execution': execution, 'records': records, 'warnings': warnings}


def report_markdown(report: dict[str, Any]) -> str:
    lines = ['# Kernel verification', '', f"Verdict: **{report['verdict']}**. Golden: `{report['golden']}`, kind `{report['golden_kind']}`.", '',
             '| Kernel | Status | Grade | Cases passed | Worst rel_l2 | Worst ULP |', '|---|---|---|---|---|---|']
    for name, k in report['kernels'].items():
        lines.append(f"| `{name}` | {k['status']} | {k['grade'] or '-'} | {k['passed']}/{k['cases']} | {k['worst_rel_l2']} | {k['worst_ulp']} |")
    summary = report['summary']
    if summary['unmapped'] or summary['not_covered']:
        lines += ['', f"Not compared: unmapped {summary['unmapped']}, not covered {summary['not_covered']}."]
    if report['first_divergence']:
        lines += ['', f"First divergence: {report['first_divergence']['explanation']}"]
    lines += ['', '## CUDA and Metal differences', '', '| Topic | Differs | Effect |', '|---|---|---|']
    for d in report['execution'].get('differences', []):
        lines.append(f"| {d['topic']} | {d['differs']} | {d['effect']} |")
    for w in report['warnings']:
        lines.append(f"\n- `{w['code']}`: {w['message']}")
    return '\n'.join(lines) + '\n'
