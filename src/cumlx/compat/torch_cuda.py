"""Redirect torch's CUDA API to another device, and record each redirected call.

Rules:

* ``torch.cuda.is_available()`` returns True. ``get_device_name()`` returns "MPS (cumlx run)", never an
  NVIDIA name. This policy is written to the report.
* A ``torch.cuda`` function with no equivalent raises UnsupportedCudaCall. It is never a silent no-op.
* An operation that the target lacks stays an error. PYTORCH_ENABLE_MPS_FALLBACK is removed unless the
  user allows it by name.
* A function whose result is only close to the original is recorded as ``approximate``.

This file uses the standard library and torch only. ``sitecustomize`` loads it by path.
"""
from __future__ import annotations

import atexit
from contextlib import nullcontext
import functools
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import os
import sys
import threading
import traceback
from types import SimpleNamespace
from typing import Any

ENV_TARGET = 'CUMLX_RUN_TARGET'
ENV_EVENTS = 'CUMLX_RUN_EVENTS'
ENV_SUBSTITUTE = 'CUMLX_RUN_SUBSTITUTE'
DEVICE_NAMES = {'mps': 'MPS (cumlx run)', 'cpu': 'CPU (cumlx run)'}
# Exceptions and helpers that stay as they are, so that `except torch.cuda.OutOfMemoryError` keeps working.
KEEP = {'CudaError', 'DeferredCudaCallError', 'OutOfMemoryError'}


class UnsupportedCudaCall(RuntimeError):
    """A CUDA API with no equivalent on the target. CUMLX refuses it instead of doing nothing."""


class _Recorder:
    """Writes each new call at once, so that a crash keeps it. Counts are written at exit."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.counts: dict[tuple[str, str, str | None], int] = {}
        self.lock = threading.Lock()
        if path:
            atexit.register(self.flush)

    def _write(self, event: dict[str, Any]) -> None:
        if not self.path:
            return
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps({**event, 'pid': os.getpid()}) + '\n')

    def call(self, kind: str, name: str, to: str | None = None) -> None:
        key = (kind, name, to)
        with self.lock:
            first = key not in self.counts
            self.counts[key] = self.counts.get(key, 0) + 1
        if first:
            self._write({'event': 'call', 'kind': kind, 'name': name, 'to': to, 'site': _call_site()})

    def flush(self) -> None:
        for (kind, name, to), count in self.counts.items():
            self._write({'event': 'count', 'kind': kind, 'name': name, 'to': to, 'count': count})
        self.counts = {}

    def uncaught(self, kind: type, value: BaseException, tb: Any) -> None:
        frames = traceback.extract_tb(tb)
        where = f'{frames[-1].filename}:{frames[-1].lineno}' if frames else None
        self._write({'event': 'uncaught', 'type': kind.__name__, 'message': str(value)[:800], 'where': where,
                     'module_name': getattr(value, 'name', None)})


def _call_site() -> str | None:
    """The first frame outside torch and this file: the line in the script or its library."""
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if name != __file__ and f'{os.sep}torch{os.sep}' not in name and not name.startswith('<') and not name.endswith('warnings.py'):
            return f'{name}:{frame.f_lineno}'
        frame = frame.f_back
    return None


_state: dict[str, Any] | None = None


def install(torch: Any, target: str, recorder: _Recorder) -> None:
    """Patch torch in this process. uninstall() restores every attribute."""
    global _state
    if _state is not None:
        return
    if target not in DEVICE_NAMES:
        raise ValueError(f'unknown target {target!r}')
    saved: list[tuple[Any, str, Any, bool]] = []
    real_device = torch.device

    def put(owner: Any, name: str, value: Any) -> None:
        present = name in vars(owner)
        saved.append((owner, name, vars(owner)[name] if present else None, present))
        setattr(owner, name, value)

    def refuse(name: str, why: str, original: Any) -> Any:
        # torch._dynamo reads attributes of these functions at import, so they keep the original's.
        @functools.wraps(original)
        def refused(*args: Any, **kwargs: Any) -> Any:
            recorder.call('refused', name)
            raise UnsupportedCudaCall(f'{name} has no equivalent on {target}: {why}. `cumlx run` refuses it '
                                      'instead of doing nothing.')
        return refused

    def mapped_index(text: str, name: str) -> str:
        index = text.partition(':')[2]
        if index not in ('', '0'):
            recorder.call('refused', f'{name}({text})')
            raise UnsupportedCudaCall(f'{text} is a second GPU, and {target} has one device.')
        return target

    def is_cuda_like(value: Any) -> bool:
        if isinstance(value, str):
            return value == 'cuda' or value.startswith('cuda:')
        return isinstance(value, real_device) and value.type == 'cuda'

    def mapped(value: Any, name: str) -> Any:
        """The target for a device given as a string, a device object or an index. Other values stay."""
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            recorder.call('mapped', name, target)
            return mapped_index(f'cuda:{value}', name)
        if is_cuda_like(value):
            recorder.call('mapped', name, target)
            return mapped_index(str(value), name)
        return value

    # -------------------------------------------------------------------- torch.device
    class _DeviceMeta(type):
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            if len(args) == 2 and args[0] == 'cuda':
                args = (f'cuda:{args[1]}',)
            if len(args) == 1 and not isinstance(args[0], int) and is_cuda_like(args[0]):
                recorder.call('mapped', 'torch.device', target)
                return real_device(mapped_index(str(args[0]), 'torch.device'))
            if args and isinstance(args[0], int) and not isinstance(args[0], bool):
                recorder.call('mapped', 'torch.device', target)
                return real_device(mapped_index(f'cuda:{args[0]}', 'torch.device'))
            return real_device(*args, **kwargs)

        def __instancecheck__(cls, instance: Any) -> bool:
            return isinstance(instance, real_device)

    class device(metaclass=_DeviceMeta):
        """torch.device, with cuda mapped to the target. isinstance still works on real devices."""
    put(torch, 'device', device)

    # -------------------------------------------------------------------- torch.cuda functions
    is_mps = target == 'mps'
    mps = torch.mps
    def total_memory() -> int:
        return int(mps.recommended_max_memory()) if is_mps else 0

    def allocated(*a: Any, **k: Any) -> int:
        return int(mps.current_allocated_memory()) if is_mps else 0

    def reserved(*a: Any, **k: Any) -> int:
        return int(mps.driver_allocated_memory()) if is_mps else 0

    properties = SimpleNamespace(name=DEVICE_NAMES[target], total_memory=0, major=0, minor=0, multi_processor_count=1, is_cumlx_run=True)

    def get_device_properties(*a: Any, **k: Any) -> Any:
        properties.total_memory = total_memory()
        return properties

    class _NoDeviceSwitch(nullcontext):
        """torch.cuda.device(i): there is one device, so there is nothing to switch."""
        def __init__(self, idx: Any = 0) -> None:
            mapped(idx, 'torch.cuda.device')
            super().__init__()

    mps_or_none = (lambda fn, *a: fn(*a)) if is_mps else (lambda fn, *a: None)
    # name: (kind, implementation). `mapped` has the same meaning, `noop` has nothing to do on this device,
    # `approximate` returns a value that only resembles the original.
    table: dict[str, tuple[str, Any]] = {
        'is_available': ('policy', lambda: True),
        'is_initialized': ('policy', lambda: True),
        'init': ('noop', lambda: None),
        'device_count': ('mapped', lambda: 1),
        'current_device': ('mapped', lambda: 0),
        'get_device_name': ('mapped', lambda *a: DEVICE_NAMES[target]),
        'get_device_properties': ('approximate', get_device_properties),
        'set_device': ('mapped', lambda d: mapped(d, 'torch.cuda.set_device') and None),
        'device': ('mapped', _NoDeviceSwitch),
        'empty_cache': ('mapped', lambda: mps_or_none(mps.empty_cache)),
        'ipc_collect': ('noop', lambda: None),
        'synchronize': ('mapped', lambda *a: mps_or_none(mps.synchronize)),
        'manual_seed': ('mapped', lambda seed: mps_or_none(mps.manual_seed, int(seed))),
        'manual_seed_all': ('mapped', lambda seed: mps_or_none(mps.manual_seed, int(seed))),
        'seed': ('mapped', lambda: mps_or_none(mps.seed)),
        'seed_all': ('mapped', lambda: mps_or_none(mps.seed)),
        'initial_seed': ('mapped', lambda: torch.initial_seed()),
        'get_rng_state': ('mapped', lambda *a: mps.get_rng_state() if is_mps else torch.get_rng_state()),
        'set_rng_state': ('mapped', lambda state, *a: mps.set_rng_state(state) if is_mps else torch.set_rng_state(state)),
        'get_rng_state_all': ('mapped', lambda: [mps.get_rng_state() if is_mps else torch.get_rng_state()]),
        'set_rng_state_all': ('mapped', lambda states: mps.set_rng_state(states[0]) if is_mps else torch.set_rng_state(states[0])),
        'memory_allocated': ('mapped', allocated),
        'memory_reserved': ('mapped', reserved),
        'memory_cached': ('mapped', reserved),
        # MPS keeps no peak counter, so these return the present value.
        'max_memory_allocated': ('approximate', allocated),
        'max_memory_reserved': ('approximate', reserved),
        'max_memory_cached': ('approximate', reserved),
        'reset_peak_memory_stats': ('noop', lambda *a: None),
        'reset_max_memory_allocated': ('noop', lambda *a: None),
        'reset_max_memory_cached': ('noop', lambda *a: None),
        'reset_accumulated_memory_stats': ('noop', lambda *a: None),
        'mem_get_info': ('approximate', lambda *a: (max(total_memory() - reserved(), 0), total_memory())),
        'memory_stats': ('approximate', lambda *a, **k: {'allocated_bytes.all.current': allocated(), 'reserved_bytes.all.current': reserved()}),
        'memory_summary': ('approximate', lambda *a, **k: f'{DEVICE_NAMES[target]}: {allocated()} bytes allocated, {reserved()} bytes reserved\n'),
        'set_per_process_memory_fraction': ('mapped', lambda fraction, *a: mps_or_none(mps.set_per_process_memory_fraction, fraction)),
        'is_bf16_supported': ('mapped', lambda *a, **k: True),
        'is_tf32_supported': ('mapped', lambda *a, **k: False),
        'get_arch_list': ('mapped', lambda: []),
        'get_gencode_flags': ('mapped', lambda: ''),
    }

    def wrap(name: str, kind: str, fn: Any) -> Any:
        full = f'torch.cuda.{name}'

        def call(*args: Any, **kwargs: Any) -> Any:
            recorder.call(kind, full, target if kind != 'noop' else None)
            return fn(*args, **kwargs)
        original = getattr(torch.cuda, name, None)
        if inspect.isroutine(original):
            functools.update_wrapper(call, original)
        call.__name__ = name
        return call

    for name, (kind, fn) in table.items():
        put(torch.cuda, name, fn if name == 'device' else wrap(name, kind, fn))
    for name in torch.cuda.__all__:
        if name in table or name in KEEP:
            continue
        value = getattr(torch.cuda, name, None)
        if inspect.isroutine(value):
            put(torch.cuda, name, refuse(f'torch.cuda.{name}', 'it depends on the NVIDIA driver or on CUDA streams', value))

    # -------------------------------------------------------------------- legacy tensor types
    legacy_dtypes = {'FloatTensor': torch.float32, 'DoubleTensor': torch.float64, 'HalfTensor': torch.float16,
                     'BFloat16Tensor': torch.bfloat16, 'LongTensor': torch.int64, 'IntTensor': torch.int32,
                     'ShortTensor': torch.int16, 'CharTensor': torch.int8, 'ByteTensor': torch.uint8, 'BoolTensor': torch.bool}

    class _LegacyMeta(type):
        def __call__(cls, *args: Any, **kwargs: Any) -> Any:
            recorder.call('mapped', f'torch.cuda.{cls.__name__}', target)
            return getattr(torch, cls.__name__)(*args, **kwargs).to(target)

        def __instancecheck__(cls, instance: Any) -> bool:
            return isinstance(instance, torch.Tensor) and instance.dtype == cls.cumlx_dtype and instance.device.type == target

    for name, dtype in legacy_dtypes.items():
        put(torch.cuda, name, _LegacyMeta(name, (), {'cumlx_dtype': dtype}))

    def legacy_dtype(value: Any) -> Any:
        """The dtype that a legacy CUDA type stands for, given as the class or by its name."""
        if isinstance(value, _LegacyMeta):
            return value.cumlx_dtype
        if isinstance(value, str) and value.startswith('torch.cuda.'):
            return legacy_dtypes.get(value.rpartition('.')[2])
        return None

    # -------------------------------------------------------------------- tensors and modules
    def tensor_cuda(self: Any, device: Any = None, non_blocking: bool = False, memory_format: Any = None) -> Any:
        recorder.call('mapped', 'Tensor.cuda', target)
        mapped(device, 'Tensor.cuda')
        extra = {} if memory_format is None else {'memory_format': memory_format}
        return self.to(target, non_blocking=non_blocking, **extra)

    def module_cuda(self: Any, device: Any = None) -> Any:
        recorder.call('mapped', 'Module.cuda', target)
        mapped(device, 'Module.cuda')
        return self.to(target)

    put(torch.Tensor, 'cuda', tensor_cuda)
    put(torch.Tensor, 'is_cuda', property(lambda self: self.device.type == target))
    put(torch.nn.Module, 'cuda', module_cuda)

    # -------------------------------------------------------------------- device arguments
    class _DeviceMode(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            kwargs = kwargs or {}
            given = kwargs.get('device')
            if given is not None:
                new = mapped(given, 'device=')
                if new is not given:
                    kwargs = {**kwargs, 'device': new}
            name = getattr(func, '__name__', None)
            if name == 'to' and len(args) > 1:
                new = mapped(args[1], 'Tensor.to')
                if new is not args[1]:
                    args = (args[0], new, *args[2:])
            elif name == 'pin_memory':
                recorder.call('noop', 'Tensor.pin_memory')
                return args[0]
            elif name == 'is_pinned':
                return False
            elif name == 'type' and len(args) > 1:
                dtype = legacy_dtype(args[1])
                if dtype is not None:
                    recorder.call('mapped', 'Tensor.type', target)
                    return args[0].to(target, dtype)
                if 'cuda' in str(args[1]):
                    recorder.call('refused', 'Tensor.type')
                    raise UnsupportedCudaCall(f'Tensor.type({args[1]}) is not a known CUDA type.')
            return func(*args, **kwargs)

    mode = _DeviceMode()
    mode.__enter__()
    # The mode stack belongs to one thread.
    thread_run = threading.Thread.run

    def run_in_mode(self: threading.Thread) -> None:
        with _DeviceMode():
            thread_run(self)
    put(threading.Thread, 'run', run_in_mode)

    # -------------------------------------------------------------------- amp, loading, defaults
    def first_arg_is_device(owner: Any, label: str, keyword: str) -> None:
        original = owner.__init__

        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            if args and args[0] == 'cuda':
                recorder.call('mapped', label, target)
                args = (target, *args[1:])
            elif kwargs.get(keyword) == 'cuda':
                recorder.call('mapped', label, target)
                kwargs = {**kwargs, keyword: target}
            original(self, *args, **kwargs)
        put(owner, '__init__', __init__)

    first_arg_is_device(torch.amp.autocast_mode.autocast, 'torch.autocast', 'device_type')
    first_arg_is_device(torch.amp.grad_scaler.GradScaler, 'torch.amp.GradScaler', 'device')

    real_load = torch.load

    def load(f: Any, map_location: Any = None, *args: Any, **kwargs: Any) -> Any:
        restore = torch.serialization.default_restore_location

        def to_target(storage: Any, location: str) -> Any:
            if location.startswith('cuda'):
                recorder.call('mapped', 'torch.load', target)
                return restore(storage, target)
            return restore(storage, location)
        if map_location is None:
            map_location = to_target
        elif is_cuda_like(map_location) or isinstance(map_location, int):
            map_location = mapped(map_location, 'torch.load')
        return real_load(f, map_location, *args, **kwargs)
    put(torch, 'load', load)

    real_default = torch.set_default_device
    put(torch, 'set_default_device', lambda d: real_default(mapped(d, 'torch.set_default_device')))

    _state = {'saved': saved, 'mode': mode, 'target': target}


def uninstall() -> None:
    """Restore every attribute. Used by the tests that show the layer leaves nothing behind."""
    global _state
    if _state is None:
        return
    _state['mode'].__exit__(None, None, None)
    for owner, name, value, present in reversed(_state['saved']):
        if present:
            setattr(owner, name, value)
        else:
            delattr(owner, name)
    _state = None


class _AliasLoader(importlib.abc.Loader):
    """Serve a module under another name, for an explicit --substitute."""

    def __init__(self, real: str) -> None:
        self.real = real
        self.spec: Any = None

    def create_module(self, spec: Any) -> Any:
        module = importlib.import_module(self.real)
        self.spec = module.__spec__
        return module

    def exec_module(self, module: Any) -> None:
        module.__spec__ = self.spec


class _Hooks(importlib.abc.MetaPathFinder):
    """Install the layer when torch has been imported, and resolve substitutions."""

    def __init__(self, target: str, recorder: _Recorder, substitutes: dict[str, str]) -> None:
        self.target = target
        self.recorder = recorder
        self.substitutes = substitutes

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name == 'torch':
            return self._torch_spec()
        root = name.partition('.')[0]
        if root in self.substitutes:
            real = self.substitutes[root] + name[len(root):]
            self.recorder.call('substituted', name, real)
            return importlib.machinery.ModuleSpec(name, _AliasLoader(real))
        return None

    def _torch_spec(self) -> Any:
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec('torch')
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        inner, hooks = spec.loader, self

        class _Loader(importlib.abc.Loader):
            def create_module(self, spec: Any) -> Any:
                return inner.create_module(spec)

            def exec_module(self, module: Any) -> None:
                inner.exec_module(module)
                module.__spec__.loader = inner
                install(module, hooks.target, hooks.recorder)

        spec.loader = _Loader()
        return spec


def activate() -> None:
    """Called by sitecustomize in each process that `cumlx run` starts."""
    target = os.environ[ENV_TARGET]
    events = os.environ.get(ENV_EVENTS)
    recorder = _Recorder(events)
    previous = sys.excepthook

    def hook(kind: type, value: BaseException, tb: Any) -> None:
        recorder.uncaught(kind, value, tb)
        previous(kind, value, tb)
    sys.excepthook = hook
    substitutes = json.loads(os.environ.get(ENV_SUBSTITUTE) or '{}')
    sys.meta_path.insert(0, _Hooks(target, recorder, substitutes))


