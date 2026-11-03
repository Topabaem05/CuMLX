#!/usr/bin/env python3
"""Install and measure the CUDA-package substitutes that M2 of docs/ROADMAP-PLATFORM.md must decide on.

This answers two questions for each candidate. Does it compute the right numbers on this Mac? Can
`cumlx run --substitute ORIGINAL=MODULE` map the original package to it? Each candidate gets its own
venv, at a pinned version or commit. One failure does not stop the others. A failure stays a failure.

    sh spikes/m2_substitutes.sh                                # starts this in the background
    python spikes/m2_substitutes.py --work ~/cumlx-m2-substitutes --only metalsplat,msplat
    python spikes/m2_substitutes.py --work /tmp/w --manifest stub.json     # no download. Used by the tests

It runs third-party code with your permissions, and it downloads about 600 MB. It writes nothing outside
the work folder, except the uv cache and the Python builds that uv manages.

Status rules:

- `pass`: the package installed and imported, every required check ran and met its limit, and at least
  one required check ran. The limits are in the code and were written before any measurement.
- `fail`: the install, the import or a required check failed, or the process crashed or timed out.
- `could_not_test`: the README does not show an API that a required check needs, so nothing was concluded.

A substitute run has its own status, because the two decisions differ: a package can be a good recipe
and still not work as an automatic import mapping. See `final_status` and `substitute_status`.

The first run against the real packages was on 2026-11-03. It found three bugs in these checks, now fixed:
mlx3d's alpha is not coverage, native_modules missed lazy imports, and the GPU-use rule accepted another
app's load. Each failure holds its traceback, so read it before you blame the package.

`--torch` replaces the pinned torch. Use it with another work folder, to test a package on an older torch.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import date
import hashlib
import importlib
from importlib import metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

SCHEMA = 'cumlx.spike.m2_substitutes.v1'
ROOT = Path(__file__).resolve().parents[1]
TORCH = 'torch==2.15.0'
INSTALL_TIMEOUT = 3600
CHECK_TIMEOUT = 1200
SUBSTITUTE_TIMEOUT = 300
CHECKED = '2026-11-03'  # The date of the pins and of the repo_licence notes below.
NO_FILE = 'No licence file in the repository at this commit. PyPI metadata says MIT. The README says MIT.'

# Wheels: mps-bitsandbytes, mps-deform-conv, mps-correlation, mps-carafe and mps-conv3d publish only a
# cp314 macOS wheel, so they use Python 3.14. The mps-bitsandbytes sdist does not list torch as a build
# requirement while its setup.py imports torch, so `--only-binary` keeps every source build out of the result.
# mps-flash-attn 0.6.3 pins `torch<2.14`. Its torch==2.15.0 entry is expected to be refused by the resolver,
# and that result is real. The torch213 entry measures the package on the newest torch that it accepts.
CANDIDATES: list[dict[str, Any]] = [
    {'name': 'mps_flash_attn', 'python': '3.12', 'requirements': ['mps-flash-attn==0.6.3'], 'only_binary': ['mps-flash-attn'],
     'module': 'mps_flash_attn', 'dist': 'mps-flash-attn', 'check': 'flash_attn', 'source': 'https://pypi.org/project/mps-flash-attn/0.6.3/',
     'commit': '41c7e20c86d1b783a4f7bb29c2db7bfe4b758c43', 'repo_licence': 'LICENSE: MIT, Copyright (c) 2025 imperatormk, and it notes MIT code from metal-flash-attention.',
     'substitutes': [{'original': 'flash_attn', 'module': 'mps_flash_attn', 'script': 'flash_attn'}]},
    {'name': 'mps_flash_attn_torch213', 'python': '3.12', 'torch': 'torch==2.13.0', 'requirements': ['mps-flash-attn==0.6.3'], 'only_binary': ['mps-flash-attn'],
     'module': 'mps_flash_attn', 'dist': 'mps-flash-attn', 'check': 'flash_attn', 'source': 'https://pypi.org/project/mps-flash-attn/0.6.3/',
     'commit': '41c7e20c86d1b783a4f7bb29c2db7bfe4b758c43', 'repo_licence': 'LICENSE: MIT, Copyright (c) 2025 imperatormk, and it notes MIT code from metal-flash-attention.',
     'note': 'Not the pinned torch. The package refuses torch 2.15.0, so this entry uses the newest torch that it accepts.',
     'substitutes': [{'original': 'flash_attn', 'module': 'mps_flash_attn', 'script': 'flash_attn'}]},
    {'name': 'mps_bitsandbytes', 'python': '3.14', 'requirements': ['mps-bitsandbytes==0.7.0'], 'only_binary': ['mps-bitsandbytes'],
     'module': 'mps_bitsandbytes', 'dist': 'mps-bitsandbytes', 'check': 'bitsandbytes', 'source': 'https://pypi.org/project/mps-bitsandbytes/0.7.0/',
     'commit': 'a3d731dac2159eaa8f73a0a863a709c96b0b3f46', 'repo_licence': NO_FILE,
     'substitutes': [{'original': 'bitsandbytes', 'module': 'mps_bitsandbytes', 'script': 'bitsandbytes'}]},
    {'name': 'mps_deform_conv', 'python': '3.14', 'requirements': ['mps-deform-conv==0.2.2', 'torchvision==0.29.2'], 'only_binary': ['mps-deform-conv'],
     'module': 'mps_deform_conv', 'dist': 'mps-deform-conv', 'check': 'deform_conv', 'source': 'https://pypi.org/project/mps-deform-conv/0.2.2/',
     'commit': 'ff2b34bd5d37d3e0c540d80a11632f374a410904', 'repo_licence': NO_FILE,
     'substitutes': [{'original': 'mmcv', 'module': 'mps_deform_conv', 'script': 'mmcv_deform'}]},
    {'name': 'mps_correlation', 'python': '3.14', 'requirements': ['mps-correlation==0.2.1'], 'only_binary': ['mps-correlation'],
     'module': 'mps_correlation', 'dist': 'mps-correlation', 'check': 'correlation', 'source': 'https://pypi.org/project/mps-correlation/0.2.1/',
     'commit': '6dbb38c492d28e048f95f945db53695912cd78a0', 'repo_licence': NO_FILE,
     'substitutes': [{'original': 'spatial_correlation_sampler', 'module': 'mps_correlation', 'script': 'correlation'}]},
    {'name': 'mps_carafe', 'python': '3.14', 'requirements': ['mps-carafe==0.2.1'], 'only_binary': ['mps-carafe'],
     'module': 'mps_carafe', 'dist': 'mps-carafe', 'check': 'carafe', 'source': 'https://pypi.org/project/mps-carafe/0.2.1/',
     'commit': '74d32f652699adb7ad927425ce302a287113e9d8', 'repo_licence': NO_FILE,
     'substitutes': [{'original': 'mmcv', 'module': 'mps_carafe', 'script': 'mmcv_carafe'}]},
    {'name': 'mps_conv3d', 'python': '3.14', 'requirements': ['mps-conv3d==0.2.1'], 'only_binary': ['mps-conv3d'],
     'module': 'mps_conv3d', 'dist': 'mps-conv3d', 'check': 'conv3d', 'source': 'https://pypi.org/project/mps-conv3d/0.2.1/',
     'commit': '841115d73b7724b012120691006bcb31864bc954', 'repo_licence': NO_FILE,
     'substitutes': [], 'no_substitute': 'It replaces torch.nn.functional.conv3d, which is part of torch, so there is no package name to map. Its patch_conv3d is measured in the check.'},
    {'name': 'mpsops', 'python': '3.14', 'requirements': ['mpsops==0.1.19'], 'only_binary': ['mpsops', 'mps-flash-attn', 'mps-bitsandbytes', 'mps-deform-conv', 'mps-correlation', 'mps-carafe', 'mps-conv3d'],
     'module': 'mpsops', 'dist': 'mpsops', 'check': 'mpsops', 'source': 'https://pypi.org/project/mpsops/0.1.19/',
     'commit': 'd28d61bdc51f6596254128c368c7b57438aa5032', 'repo_licence': NO_FILE,
     'note': 'The metapackage does not pin mps-flash-attn, so the resolver chooses a version for torch 2.15.0. versions in the result shows which.',
     'substitutes': [], 'no_substitute': 'A metapackage. Its parts are measured in their own entries.'},
    {'name': 'metalsplat', 'python': '3.12', 'requirements': ['metalsplat @ git+https://github.com/tchauffi/metalsplat@84e8c3a433f71c107e9387d3f72a765436c11f6e'], 'only_binary': [],
     'module': 'metalsplat', 'dist': 'metalsplat', 'check': 'metalsplat', 'source': 'https://github.com/tchauffi/metalsplat (tag v0.3.0; not on PyPI)',
     'commit': '84e8c3a433f71c107e9387d3f72a765436c11f6e', 'repo_licence': 'LICENSE: MIT, Copyright (c) 2026 Thibaut CHAUFFIER.',
     'substitutes': [{'original': 'gsplat', 'module': 'metalsplat', 'script': 'gsplat_metalsplat'}]},
    {'name': 'msplat', 'python': '3.12', 'requirements': ['msplat==1.1.4'], 'only_binary': ['msplat'],
     'module': 'msplat', 'dist': 'msplat', 'check': 'msplat', 'source': 'https://pypi.org/project/msplat/1.1.4/',
     'commit': '6b819711fa7c90f054567cb4fb4937743367afdd', 'repo_licence': 'LICENSE: Apache License 2.0.',
     'substitutes': [{'original': 'gsplat', 'module': 'msplat', 'script': 'gsplat_msplat'}]},
    {'name': 'mlx3d', 'python': '3.12', 'requirements': ['mlx3d==0.3.2', 'mlx==0.32.4'], 'only_binary': ['mlx3d'],
     'module': 'mlx3d', 'dist': 'mlx3d', 'check': 'mlx3d', 'source': 'https://pypi.org/project/mlx3d/0.3.2/',
     'commit': '0124b891a1983800e05e3d8fdb3ebdc53b449dd5', 'repo_licence': 'LICENSE: MIT, Copyright (c) 2026 AmirHossein Razlighi.',
     'substitutes': [{'original': 'pytorch3d', 'module': 'mlx3d', 'script': 'pytorch3d'}, {'original': 'nvdiffrast', 'module': 'mlx3d', 'script': 'nvdiffrast'}]},
]


# --------------------------------------------------------------------------- what runs inside each venv
class Recorder:
    """Results of one check process. It saves after each change, so a crash keeps what was measured."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {'import': None, 'import_error': None, 'module_version': None, 'native_modules': [], 'licence': None, 'versions': {},
                                     'device': None, 'numbers': {}, 'checks': [], 'could_not_test': [], 'finished': False}
        self.devices: list[str] = []
        self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(_clean(self.data), indent=1) + '\n')
        temporary.replace(self.path)

    def number(self, name: str, value: Any) -> Any:
        self.data['numbers'][name] = value
        self.save()
        return value

    def check(self, name: str, ok: bool, detail: Any = '', required: bool = True) -> bool:
        self.data['checks'].append({'name': name, 'ok': bool(ok), 'required': required, 'detail': detail})
        self.save()
        return bool(ok)

    def at_most(self, name: str, value: float, limit: float, required: bool = True) -> bool:
        """`value <= limit`. A NaN fails, because the comparison is false."""
        self.number(name, value)
        return self.check(name, value <= limit, {'value': value, 'limit': limit}, required)

    def cannot(self, name: str, reason: str, required: bool = True) -> None:
        self.data['could_not_test'].append({'name': name, 'reason': reason, 'required': required})
        self.save()

    def attempt(self, name: str, fn: Callable[[], Any], api: str = 'README', required: bool = True) -> Any:
        """Run an API call. `api` says where it was documented, and that decides what an error means.

        An error from an API in the README is a failure: the package does not do what it says. An error
        that shows a name or signature is wrong, for an API that the README does not show, says nothing
        about the package, so it is could_not_test. Returns None on any error.
        """
        try:
            return fn()
        except Exception as exc:
            text = f'{type(exc).__name__}: {exc}'[:800]
            if api != 'README' and isinstance(exc, (AttributeError, TypeError, ImportError)):
                self.cannot(name, f'{api} is the only source for this API, and the call failed: {text}', required)
            else:
                self.check(name, False, {'api_source': api, 'error': text, 'traceback': traceback.format_exc()[-1500:]}, required)
            return None

    def expect_mps(self, tensor: Any) -> Any:
        """Record the device from the tensor itself, not from the name of the package."""
        self.devices.append(str(tensor.device.type))
        return tensor

    def set_device(self, used: str, evidence: Any) -> None:
        self.data['device'] = {'used': used, 'evidence': evidence}
        self.save()

    def close_device(self) -> None:
        if self.devices:
            seen = sorted(set(self.devices))
            self.set_device(','.join(seen), f'tensor.device.type of {len(self.devices)} outputs')
            self.check('output_device_is_mps', seen == ['mps'], {'seen': seen})


def _clean(value: Any) -> Any:
    """JSON has no NaN, and a measured NaN must survive as a value that a reader can see."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'item') and getattr(value, 'ndim', 1) == 0:
        return _clean(value.item())
    return str(value)


_GPU_SAMPLER = """
import re, subprocess, sys, time
while True:
    text = subprocess.run(['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'], capture_output=True, text=True, check=False).stdout
    found = re.search(r'"Device Utilization %"=(\\d+)', text)
    if found:
        print(found.group(1), flush=True)
    time.sleep(0.5)
"""


class GpuSampler:
    """System-wide GPU use from ioreg, which needs no privileges. Other apps count too, so it is evidence, not proof.

    It is a separate process, because a native training loop can hold the GIL for its whole run.
    """

    def __init__(self) -> None:
        self.samples: list[int] = []
        self._process: subprocess.Popen | None = None

    @staticmethod
    def read() -> int | None:
        try:
            text = subprocess.run(['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator'], capture_output=True, text=True, timeout=5, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        found = re.search(r'"Device Utilization %"=(\d+)', text)
        return int(found.group(1)) if found else None

    def __enter__(self) -> GpuSampler:
        self._process = subprocess.Popen([sys.executable, '-I', '-c', _GPU_SAMPLER], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def __exit__(self, *_: Any) -> None:
        self._process.terminate()
        output, _unused = self._process.communicate()
        self.samples = [int(line) for line in output.split() if line.isdigit()]

    def summary(self, baseline: int | None) -> dict[str, Any]:
        return {'baseline_percent': baseline, 'peak_percent': max(self.samples) if self.samples else None, 'samples': len(self.samples)}

    def busy(self, baseline: int | None) -> bool:
        """10 points over the reading before the work. A high reading alone proves nothing, because another app can cause it.

        The per-process accumulatedGPUTime in ioreg read 0 after real GPU work on macOS 27.0.1, so it cannot replace this.
        """
        peak = max(self.samples) if self.samples else None
        return peak is not None and baseline is not None and peak >= baseline + 10


def _rel(a: Any, b: Any) -> float:
    """Relative L2 error of `a` against the reference `b`, in float64 on the CPU."""
    import torch
    a, b = a.detach().cpu().to(torch.float64), b.detach().cpu().to(torch.float64)
    denominator = b.norm().item()
    return (a - b).norm().item() / denominator if denominator > 0 else (a - b).norm().item()


# Limits on a float32 result. 1e-4 is the module threshold of `cumlx diff`; float16 and gradients get more room.
FP32 = 1e-4
FP16 = 1e-2
GRAD = 1e-3


def _mps(tensor: Any) -> Any:
    return tensor.to('mps')


def check_flash_attn(rec: Recorder) -> None:
    # README of mps-flash-attn 0.6.3 on PyPI, the API this check relies on:
    #   from mps_flash_attn import flash_attention
    #   # (B, H, N, D) format
    #   out = flash_attention(q, k, v)
    #   out = flash_attention(q, k, v, is_causal=True)
    #   from mps_flash_attn import replace_sdpa
    #   replace_sdpa()  # Patches F.scaled_dot_product_attention
    #   | Forward pass | yes | FP16/BF16/FP32 |
    #   | Backward pass | yes | Full gradient support |
    # The reference is torch's own attention on the CPU in float32, on the same inputs.
    from mps_flash_attn import flash_attention, replace_sdpa
    import torch
    import torch.nn.functional as F
    sdpa = F.scaled_dot_product_attention

    def inputs(n: int, dtype: Any) -> tuple[Any, Any, Any]:
        torch.manual_seed(n)
        return tuple(torch.randn(2, 4, n, 32, device='mps', dtype=dtype) for _ in range(3))

    def forward_case(label: str, n: int, dtype: Any, limit: float) -> None:
        q, k, v = inputs(n, dtype)
        reference = sdpa(q.cpu().float(), k.cpu().float(), v.cpu().float())
        out = rec.attempt(f'flash_attention_{label}', lambda: flash_attention(q, k, v))
        if out is not None:
            rec.expect_mps(out)
            rec.at_most(f'rel_l2_vs_cpu_sdpa_{label}', _rel(out, reference), limit)
            rec.number(f'rel_l2_torch_mps_sdpa_vs_cpu_{label}', _rel(sdpa(q, k, v), reference))

    forward_case('fp32_n32', 32, torch.float32, FP32)
    forward_case('fp32_n1024', 1024, torch.float32, FP32)
    forward_case('fp16_n32', 32, torch.float16, FP16)
    forward_case('fp16_n1024', 1024, torch.float16, FP16)

    q, k, v = inputs(64, torch.float32)
    causal_reference = sdpa(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
    causal = rec.attempt('flash_attention_causal', lambda: flash_attention(q, k, v, is_causal=True))
    if causal is not None:
        rec.expect_mps(causal)
        rec.at_most('rel_l2_vs_cpu_sdpa_causal', _rel(causal, causal_reference), FP32)
        # A control: the comparison must be able to see a wrong mask.
        rec.check('causal_differs_from_uncausal', _rel(causal, sdpa(q.cpu(), k.cpu(), v.cpu())) > 0.01)

    grad_q, grad_k, grad_v = (t.clone().requires_grad_() for t in (q, k, v))
    ref_q, ref_k, ref_v = (t.cpu().clone().requires_grad_() for t in (q, k, v))
    torch.manual_seed(1)
    weight = torch.randn(q.shape)

    def backward() -> bool:
        (flash_attention(grad_q, grad_k, grad_v) * weight.to('mps')).sum().backward()
        return True

    if rec.attempt('flash_attention_backward', backward) is not None:
        (sdpa(ref_q, ref_k, ref_v) * weight).sum().backward()
        for label, got, want in (('q', grad_q, ref_q), ('k', grad_k, ref_k), ('v', grad_v, ref_v)):
            rec.at_most(f'rel_l2_grad_{label}_vs_cpu_sdpa', _rel(got.grad, want.grad), GRAD)

    def timing() -> None:
        q, k, v = inputs(1024, torch.float16)
        start = time.perf_counter()
        flash_attention(q, k, v)
        torch.mps.synchronize()
        middle = time.perf_counter()
        sdpa(q, k, v)
        torch.mps.synchronize()
        rec.number('ms_flash_attention_fp16_n1024', (middle - start) * 1000)
        rec.number('ms_torch_mps_sdpa_fp16_n1024', (time.perf_counter() - middle) * 1000)

    rec.attempt('timing', timing, required=False)

    # Last, because it replaces the torch function for the rest of the process.
    q, k, v = inputs(64, torch.float32)
    reference = sdpa(q.cpu(), k.cpu(), v.cpu())
    if rec.attempt('replace_sdpa', lambda: replace_sdpa() or True) is not None:
        patched = rec.attempt('patched_sdpa', lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
        if patched is not None:
            rec.expect_mps(patched)
            rec.at_most('rel_l2_patched_sdpa_vs_cpu_sdpa', _rel(patched, reference), FP32)


# The NF4 table of the original bitsandbytes (create_normal_map), for a hand-made 4-bit reference.
NF4_TABLE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


def check_bitsandbytes(rec: Recorder) -> None:
    # README of mps-bitsandbytes 0.7.0 on PyPI, the API this check relies on:
    #   linear = torch.nn.Linear(4096, 4096).half().to('mps')
    #   linear_4bit = Linear4bit.from_linear(linear)  # NF4 by default
    #   linear_fp4 = Linear4bit.from_linear(linear, quant_type='fp4')
    #   linear_int8 = Linear8bit.from_linear(linear)
    #   linear_fp8 = LinearFP8.from_linear(linear)
    #   # Drop-in replacement for torch optimizers
    #   optimizer = Adam8bit(model.parameters(), lr=1e-3)
    #   optimizer = AdamW8bit(model.parameters(), lr=1e-3, weight_decay=0.01)
    #   optimizer = Lion8bit(model.parameters(), lr=1e-4)
    #   optimizer = SGD8bit(model.parameters(), lr=0.1, momentum=0.9)
    #   optimizer = PagedAdamW(model.parameters(), lr=1e-3, page_to_cpu=True)
    #   packed, absmax = quantize_nf4(weight, block_size=64)
    #   output = matmul_nf4(input, packed, absmax)
    #   `has_native_kernels()` - Check Metal kernels loaded
    #   | NF4/FP4 | CUDA | Metal |
    # The README's functional lines are tried as written. They are optional, because the modules above
    # are what a bitsandbytes user calls. At the pinned commit the source names the argument `blocksize`.
    import mps_bitsandbytes as bnb
    import torch
    import torch.nn.functional as F

    # The README's table says that the optimizers are pure PyTorch and the layers are Metal. So the optimizers
    # run first: on torch 2.15.0 a native layer ended this process with SIGSEGV, and that lost every later number.
    # Optimizers on one least-squares problem. The reference is the torch optimizer from the same start.
    torch.manual_seed(2)
    a = torch.randn(64, 16, device='mps')
    target = a @ torch.randn(16, 1, device='mps')

    def optimize(make: Callable[[list[Any]], Any], steps: int = 200) -> dict[str, Any]:
        w = torch.nn.Parameter(torch.zeros(16, 1, device='mps'))
        optimizer = make([w])
        losses, first = [], None
        for step in range(steps):
            optimizer.zero_grad()
            loss = (a @ w - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step == 0:
                first = w.detach().clone()
        rec.expect_mps(w)
        return {'first_step': first, 'final': w.detach().clone(), 'loss0': losses[0], 'loss_end': (a @ w - target).pow(2).mean().item(), 'optimizer': optimizer}

    def optimizer_case(label: str, ours: Callable[[list[Any]], Any], theirs: Callable[[list[Any]], Any], final_limit: float = 0.1) -> Any:
        got = rec.attempt(f'{label}_steps', lambda: optimize(ours))
        if got is None:
            return None
        want = optimize(theirs)
        rec.at_most(f'{label}_first_step_rel_l2_vs_torch', _rel(got['first_step'], want['first_step']), 0.01)
        rec.at_most(f'{label}_final_param_rel_l2_vs_torch', _rel(got['final'], want['final']), final_limit)
        rec.number(f'{label}_loss_start', got['loss0'])
        rec.number(f'{label}_loss_end', got['loss_end'])
        rec.number(f'torch_{label}_loss_end', want['loss_end'])
        rec.check(f'{label}_loss_fell_by_100x', got['loss_end'] <= got['loss0'] / 100, {'start': got['loss0'], 'end': got['loss_end']})
        state = [v for values in got['optimizer'].state.values() for v in values.values() if torch.is_tensor(v)]
        rec.number(f'{label}_state_dtypes', sorted({str(v.dtype) for v in state}))
        return got

    adam = optimizer_case('adam8bit', lambda p: bnb.Adam8bit(p, lr=0.05), lambda p: torch.optim.Adam(p, lr=0.05))
    adamw = optimizer_case('adamw8bit', lambda p: bnb.AdamW8bit(p, lr=0.05, weight_decay=0.01), lambda p: torch.optim.AdamW(p, lr=0.05, weight_decay=0.01))
    sgd = optimizer_case('sgd8bit', lambda p: bnb.SGD8bit(p, lr=0.01, momentum=0.9), lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9))
    for label, result in (('adam8bit', adam), ('adamw8bit', adamw), ('sgd8bit', sgd)):
        if result is not None:
            dtypes = rec.data['numbers'][f'{label}_state_dtypes']
            rec.check(f'{label}_state_is_8_bit', any(d in ('torch.int8', 'torch.uint8') for d in dtypes), dtypes)
    optimizer_case('paged_adamw', lambda p: bnb.PagedAdamW(p, lr=0.05, page_to_cpu=True), lambda p: torch.optim.AdamW(p, lr=0.05))

    # Lion has no torch counterpart. The hand check is that the loss falls by half or more.
    lion = rec.attempt('lion8bit_steps', lambda: optimize(lambda p: bnb.Lion8bit(p, lr=0.01)))
    if lion is not None:
        rec.number('lion8bit_loss_start', lion['loss0'])
        rec.number('lion8bit_loss_end', lion['loss_end'])
        rec.check('lion8bit_loss_fell_by_half', lion['loss_end'] <= lion['loss0'] / 2, {'start': lion['loss0'], 'end': lion['loss_end']})
    native = rec.attempt('has_native_kernels', lambda: bool(bnb.has_native_kernels()))
    rec.number('has_native_kernels', native)
    # True proves only that _C imports. On torch 2.15.0 it was True while each kernel refused an MPS tensor.
    rec.check('native_metal_kernels_loaded', native is True, 'The README says NF4, FP4, INT8 and FP8 run in Metal. False means a pure PyTorch path.')

    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 128).half().to('mps')
    torch.nn.init.normal_(linear.weight, std=0.02)
    x = torch.randn(16, 256, device='mps', dtype=torch.float16)
    reference = linear(x)

    def layer_case(label: str, make: Callable[[], Any], limit: float) -> None:
        layer = rec.attempt(f'{label}_from_linear', make)
        if layer is None:
            return
        out = rec.attempt(f'{label}_forward', lambda: layer(x))
        if out is not None:
            rec.expect_mps(out)
            rec.at_most(f'rel_l2_{label}_vs_float_linear', _rel(out, reference), limit)

    # Limits for weights drawn from a normal distribution, set from the bit width and not from a run.
    layer_case('nf4', lambda: bnb.Linear4bit.from_linear(linear), 0.35)
    layer_case('fp4', lambda: bnb.Linear4bit.from_linear(linear, quant_type='fp4'), 0.5)
    layer_case('int8', lambda: bnb.Linear8bit.from_linear(linear), 0.05)
    layer_case('fp8', lambda: bnb.LinearFP8.from_linear(linear), 0.05)

    # A hand reference for NF4: per block of 64 weights, scale by the largest magnitude and take the nearest table value.
    flat = linear.weight.detach().float().cpu().flatten()
    table = torch.tensor(NF4_TABLE)
    blocks = flat.split(64)
    quantized = torch.cat([table[((b / b.abs().max()).unsqueeze(1) - table).abs().argmin(1)] * b.abs().max() for b in blocks]).view_as(linear.weight).to('mps')
    hand_error = _rel(F.linear(x.float(), quantized, linear.bias.float()), reference)
    rec.number('rel_l2_hand_nf4_vs_float_linear', hand_error)
    nf4_error = rec.data['numbers'].get('rel_l2_nf4_vs_float_linear')
    if nf4_error is not None:
        rec.at_most('nf4_error_over_hand_nf4_error', nf4_error / hand_error, 1.5)

    weight = torch.randn(128, 256, device='mps', dtype=torch.float16)

    def readme_functional() -> Any:
        packed, absmax = bnb.quantize_nf4(weight, block_size=64)
        return bnb.matmul_nf4(x, packed, absmax)

    rec.attempt('readme_functional_nf4_as_written', readme_functional, required=False)

    rec.cannot('huggingface_integration', 'The README needs transformers and a model download. This spike has neither.', required=False)


def check_deform_conv(rec: Recorder) -> None:
    # README of mps-deform-conv 0.2.2 on PyPI, the API this check relies on:
    #   from mps_deform_conv import deform_conv2d
    #   # Weight: (out_channels, in_channels, kernel_h, kernel_w)
    #   # Offset: (batch, 2 * kernel_h * kernel_w, out_h, out_w)
    #   output = deform_conv2d(input, offset, weight, padding=(1, 1))
    #   ### `deform_conv2d(input, offset, weight, bias, stride, padding, dilation, mask)`
    #   Functional interface matching `torchvision.ops.deform_conv2d`.
    #   | `offset` | Tensor | Offset tensor (N, 2*K*K*groups, H_out, W_out) |
    #   - Grouped convolutions supported
    #   - Full forward and backward pass (training supported)
    # The reference is torchvision's implementation on the CPU, and F.conv2d when every offset is zero.
    from mps_deform_conv import deform_conv2d
    import torch
    import torch.nn.functional as F
    from torchvision.ops import deform_conv2d as torchvision_deform
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    weight = torch.randn(4, 8, 3, 3)
    bias = torch.randn(4)
    offset = torch.randn(2, 18, 16, 16) * 2.0

    zero = rec.attempt('zero_offset', lambda: deform_conv2d(_mps(x), _mps(torch.zeros_like(offset)), _mps(weight), padding=(1, 1)))
    if zero is not None:
        rec.expect_mps(zero)
        rec.at_most('rel_l2_zero_offset_vs_conv2d', _rel(zero, F.conv2d(x, weight, padding=1)), FP32)

    def native_option(label: str, fn: Callable[[], Any], reference: Any) -> None:
        """The option with no package at all: torch's own path on MPS. It is optional, and only numbers are recorded."""
        out = rec.attempt(label, fn, required=False)
        if out is not None:
            rec.number(f'{label}_device', out.device.type)
            rec.number(f'{label}_rel_l2_vs_cpu', _rel(out, reference))

    def case(label: str, offsets: Any, weights: Any, inputs: Any = x, **kw: Any) -> None:
        reference = torchvision_deform(inputs, offsets, weights, **kw)
        moved = {k: (_mps(v) if torch.is_tensor(v) else v) for k, v in kw.items()}
        native_option(f'torchvision_on_mps_{label}', lambda: torchvision_deform(_mps(inputs), _mps(offsets), _mps(weights), **moved), reference)
        out = rec.attempt(label, lambda: deform_conv2d(_mps(inputs), _mps(offsets), _mps(weights), **moved))
        if out is not None:
            rec.expect_mps(out)
            rec.at_most(f'rel_l2_{label}_vs_torchvision_cpu', _rel(out, reference), FP32)

    case('random_offset', offset, weight, padding=(1, 1))
    case('random_offset_bias', offset, weight, bias=bias, padding=(1, 1))
    out_size = (16 + 2 - 2 * (3 - 1) - 1) // 2 + 1
    case('stride_dilation', torch.randn(2, 18, out_size, out_size), weight, stride=(2, 2), padding=(1, 1), dilation=(2, 2))
    case('dcnv2_mask', offset, weight, padding=(1, 1), mask=torch.rand(2, 9, 16, 16))
    case('groups_2', torch.randn(2, 36, 16, 16), torch.randn(4, 4, 3, 3), padding=(1, 1))

    grad_x = _mps(x).clone().requires_grad_()
    ref_x = x.clone().requires_grad_()
    backward = rec.attempt('backward', lambda: deform_conv2d(grad_x, _mps(offset), _mps(weight), padding=(1, 1)).pow(2).sum().backward() or True)
    if backward is not None:
        torchvision_deform(ref_x, offset, weight, padding=(1, 1)).pow(2).sum().backward()
        rec.at_most('rel_l2_grad_input_vs_torchvision_cpu', _rel(grad_x.grad, ref_x.grad), GRAD)

    # Does torch itself need this package? The README says that torchvision has no MPS path.
    grad_native = _mps(x).clone().requires_grad_()
    grad_cpu = x.clone().requires_grad_()
    if rec.attempt('torchvision_on_mps_backward', lambda: torchvision_deform(grad_native, _mps(offset), _mps(weight), padding=(1, 1)).pow(2).sum().backward() or True, required=False) is not None:
        torchvision_deform(grad_cpu, offset, weight, padding=(1, 1)).pow(2).sum().backward()
        rec.number('torchvision_on_mps_grad_rel_l2_vs_cpu', _rel(grad_native.grad, grad_cpu.grad))


def _correlation_volume(f1: Any, f2: Any, d: int, order: str) -> Any:
    """The README's formula: corr[x, y, dx, dy] = sum(fmap1[x, y, :] * fmap2[x+dx, y+dy, :]), with zeros outside."""
    import torch
    import torch.nn.functional as F
    _, _, h, w = f1.shape
    padded = F.pad(f2, [d] * 4)
    channels = []
    outer = [(dy, dx) for dy in range(-d, d + 1) for dx in range(-d, d + 1)] if order == 'dy_major' else [(dy, dx) for dx in range(-d, d + 1) for dy in range(-d, d + 1)]
    for dy, dx in outer:
        channels.append((f1 * padded[:, :, d + dy:d + dy + h, d + dx:d + dx + w]).sum(1))
    return torch.stack(channels, 1)


def check_correlation(rec: Recorder) -> None:
    # README of mps-correlation 0.2.1 on PyPI, the API this check relies on:
    #   corr = correlation(fmap1, fmap2, kernel_size=1, max_displacement=4, stride1=1, stride2=1, pad_size=4)
    #   # Output: (1, 81, 64, 64) - 81 = (2*4+1)^2 displacement channels
    #   corr[x, y, dx, dy] = sum(fmap1[x, y, :] * fmap2[x+dx, y+dy, :])
    #   - Full forward and backward pass (training supported)
    # The README does not state the order of the displacement channels or a normalization. So the hand
    # volume is compared in both orders, only away from the border, and up to one constant factor. The
    # order and the factor are reported, and a user of the original sampler must check them.
    from mps_correlation import CorrBlock, correlation
    import torch
    torch.manual_seed(0)
    d = 2
    f1, f2 = torch.randn(1, 8, 12, 12), torch.randn(1, 8, 12, 12)
    out = rec.attempt('correlation', lambda: correlation(f1.to('mps'), f2.to('mps'), kernel_size=1, max_displacement=d, stride1=1, stride2=1, pad_size=d))
    if out is None:
        return
    rec.expect_mps(out)
    rec.check('output_shape', list(out.shape) == [1, 25, 12, 12], list(out.shape))
    inner = (slice(None), slice(None), slice(d, -d), slice(d, -d))
    got = out[inner].detach().cpu().double()
    errors, factors = {}, {}
    for order in ('dy_major', 'dx_major'):
        hand = _correlation_volume(f1, f2, d, order)[inner].double()
        factors[order] = ((got * hand).sum() / (hand * hand).sum()).item()
        errors[order] = _rel(got, factors[order] * hand)
    best = min(errors, key=errors.get)
    rec.number('displacement_order', best)
    rec.number('constant_factor', factors[best])
    rec.at_most('rel_l2_vs_hand_volume_best_order', errors[best], 1e-3)
    rec.check('constant_factor_is_not_zero', abs(factors[best]) > 1e-6, factors[best])

    weight = torch.randn(got.shape)
    grad1 = f1.to('mps').clone().requires_grad_()

    def backward() -> bool:
        (correlation(grad1, f2.to('mps'), kernel_size=1, max_displacement=d, stride1=1, stride2=1, pad_size=d)[inner] * weight.to('mps')).sum().backward()
        return True

    if rec.attempt('backward', backward) is not None:
        ref1 = f1.clone().requires_grad_()
        (factors[best] * _correlation_volume(ref1, f2, d, best)[inner] * weight.double()).sum().backward()
        rec.at_most('rel_l2_grad_vs_hand_volume', _rel(grad1.grad, ref1.grad), 1e-2)

    block = rec.attempt('corr_block', lambda: CorrBlock(f1.to('mps'), f2.to('mps'), num_levels=2, radius=2)(torch.zeros(1, 2, 12, 12, device='mps')), required=False)
    if block is not None:
        rec.number('corr_block_shape', list(block.shape))
    rec.cannot('corr_block_values', 'The README shows the call and no values, and no reference is installed.', required=False)


def _carafe_hand(x: Any, masks: Any, k: int, scale: int, order: str) -> Any:
    """The README's three steps, written with unfold: weight the input neighborhood of each output pixel, and sum."""
    import torch.nn.functional as F
    n, c, h, w = x.shape
    r = k // 2
    patches = F.unfold(F.pad(x, [r] * 4), k).view(n, c, k * k, h, w)
    if order == 'column_major':
        patches = patches.view(n, c, k, k, h, w).transpose(2, 3).reshape(n, c, k * k, h, w)
    patches = patches.repeat_interleave(scale, 3).repeat_interleave(scale, 4)
    return (patches * masks.unsqueeze(1)).sum(2)


def check_carafe(rec: Recorder) -> None:
    # README of mps-carafe 0.2.1 on PyPI, the API this check relies on:
    #   # Reassembly masks (N, group_size * k^2, H*scale, W*scale)
    #   output = carafe(features, masks, kernel_size, group_size, scale_factor)
    #   1. For each output pixel, identify the corresponding input neighborhood
    #   2. Use learned reassembly kernels (masks) to weight input pixels
    #   3. Sum the weighted inputs to produce the output
    # The README does not state the order of the kernel in the mask channels or the border handling. So
    # a center-only kernel, which needs neither, is compared everywhere; the rest stay away from the
    # border and accept either order. The order that matched is reported.
    from mps_carafe import carafe
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    k, scale, r = 5, 2, 2
    x = torch.randn(1, 4, 16, 16)

    delta = torch.zeros(1, k * k, 32, 32)
    delta[:, r * k + r] = 1.0
    out = rec.attempt('carafe_center_kernel', lambda: carafe(_mps(x), _mps(delta), k, 1, scale))
    if out is not None:
        rec.expect_mps(out)
        rec.check('output_shape', list(out.shape) == [1, 4, 32, 32], list(out.shape))
        rec.at_most('rel_l2_center_kernel_vs_nearest_upsample', _rel(out, F.interpolate(x, scale_factor=scale, mode='nearest')), FP32)

    inner = (slice(None), slice(None), slice(r * scale, -r * scale), slice(r * scale, -r * scale))
    uniform = torch.full((1, k * k, 32, 32), 1 / (k * k))
    out = rec.attempt('carafe_uniform_kernel', lambda: carafe(_mps(x), _mps(uniform), k, 1, scale))
    if out is not None:
        rec.expect_mps(out)
        rec.at_most('rel_l2_uniform_kernel_vs_hand', _rel(out[inner], _carafe_hand(x, uniform, k, scale, 'row_major')[inner]), FP32)

    masks = torch.softmax(torch.randn(1, k * k, 32, 32) * 2, 1)
    out = rec.attempt('carafe_random_kernel', lambda: carafe(_mps(x), _mps(masks), k, 1, scale))
    if out is not None:
        rec.expect_mps(out)
        errors = {order: _rel(out[inner], _carafe_hand(x, masks, k, scale, order)[inner]) for order in ('row_major', 'column_major')}
        best = min(errors, key=errors.get)
        rec.number('kernel_order', best)
        rec.at_most('rel_l2_random_kernel_vs_hand_best_order', errors[best], FP32)
        rec.check('a_wrong_order_is_visible', max(errors.values()) > 0.01, errors)

        grad_x = _mps(x).clone().requires_grad_()
        ref_x = x.clone().requires_grad_()
        if rec.attempt('backward', lambda: carafe(grad_x, _mps(masks), k, 1, scale)[inner].pow(2).sum().backward() or True) is not None:
            _carafe_hand(ref_x, masks, k, scale, best)[inner].pow(2).sum().backward()
            rec.at_most('rel_l2_grad_vs_hand', _rel(grad_x.grad, ref_x.grad), GRAD)


def check_conv3d(rec: Recorder) -> None:
    # README of mps-conv3d 0.2.1 on PyPI, the API this check relies on:
    #   out = conv3d(x, w, stride=1, padding=(1, 3, 3))
    #   Same signature as `torch.nn.functional.conv3d`.
    #   ### `patch_conv3d()` Monkey-patches `torch.nn.functional.conv3d` to use MPS implementation for MPS tensors.
    #   ### `unpatch_conv3d()` Restores original `torch.nn.functional.conv3d`.
    #   - fp32 and fp16 supported
    #   - Groups and dilation supported
    # The reference is torch.nn.functional.conv3d on the CPU, on the same inputs.
    from mps_conv3d import conv3d, patch_conv3d, unpatch_conv3d
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 16, 16)
    w = torch.randn(6, 3, 3, 5, 5)
    original = F.conv3d

    def case(label: str, inputs: Any, weights: Any, limit: float = FP32, **kw: Any) -> None:
        reference = original(inputs.float(), weights.float(), **kw)
        # Does torch itself still need this package? The README says that MPS has no conv3d. This is optional, and only numbers are recorded.
        native = rec.attempt(f'torch_on_mps_{label}', lambda: original(_mps(inputs), _mps(weights), **kw), required=False)
        if native is not None:
            rec.number(f'torch_on_mps_{label}_device', native.device.type)
            rec.number(f'torch_on_mps_{label}_rel_l2_vs_cpu', _rel(native, reference))
        out = rec.attempt(label, lambda: conv3d(_mps(inputs), _mps(weights), **kw))
        if out is not None:
            rec.expect_mps(out)
            rec.at_most(f'rel_l2_{label}_vs_cpu', _rel(out, reference), limit)

    case('padding', x, w, stride=1, padding=(1, 2, 2))
    case('stride', x, w, stride=(1, 2, 2), padding=(1, 2, 2))
    case('dilation', x, w, padding=(2, 4, 4), dilation=(2, 2, 2))
    case('groups_3', x, torch.randn(6, 1, 3, 5, 5), padding=(1, 2, 2), groups=3)
    case('fp16', x.half(), w.half(), limit=FP16, padding=(1, 2, 2))

    grad_native = _mps(x).clone().requires_grad_()
    grad_cpu = x.clone().requires_grad_()
    if rec.attempt('torch_on_mps_backward', lambda: original(grad_native, _mps(w), padding=(1, 2, 2)).pow(2).sum().backward() or True, required=False) is not None:
        original(grad_cpu, w, padding=(1, 2, 2)).pow(2).sum().backward()
        rec.number('torch_on_mps_grad_rel_l2_vs_cpu', _rel(grad_native.grad, grad_cpu.grad))

    grad_x = _mps(x).clone().requires_grad_()
    ref_x = x.clone().requires_grad_()
    if rec.attempt('backward', lambda: conv3d(grad_x, _mps(w), padding=(1, 2, 2)).pow(2).sum().backward() or True) is not None:
        original(ref_x, w, padding=(1, 2, 2)).pow(2).sum().backward()
        rec.at_most('rel_l2_grad_vs_cpu', _rel(grad_x.grad, ref_x.grad), GRAD)

    if rec.attempt('patch_conv3d', lambda: patch_conv3d() or True) is not None:
        out = rec.attempt('patched_conv3d', lambda: F.conv3d(_mps(x), _mps(w), padding=(1, 2, 2)))
        if out is not None:
            rec.expect_mps(out)
            rec.at_most('rel_l2_patched_vs_cpu', _rel(out, original(x, w, padding=(1, 2, 2))), FP32)
        unpatch_conv3d()
        rec.check('unpatch_restores_torch', F.conv3d is original)


def check_mpsops(rec: Recorder) -> None:
    # README of mpsops 0.1.19 on PyPI, the API this check relies on:
    #   mpsops.print_status()
    #   from mpsops import flash_attn_func, deform_conv2d, correlation, carafe, conv3d, patch_conv3d
    # The README gives no signature for these names, so their numbers are could_not_test here. The
    # entries for each package measure the numbers. This entry proves only which names work on torch
    # 2.15.0. At the pinned commit each name falls back to None when its import fails, so None is a failure.
    import mpsops
    rec.attempt('print_status', mpsops.print_status)
    for dist in ('mps-flash-attn', 'mps-bitsandbytes', 'mps-deform-conv', 'mps-correlation', 'mps-carafe', 'mps-conv3d'):
        try:
            rec.data['versions'][dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            rec.data['versions'][dist] = None
    for name in ('flash_attn_func', 'deform_conv2d', 'correlation', 'carafe', 'conv3d', 'patch_conv3d'):
        rec.check(f'readme_name_{name}_is_usable', getattr(mpsops, name, None) is not None, 'None or missing means that the import failed inside mpsops.')
    rec.set_device('not_measured', 'This entry runs no tensor. The entries for each package measure the device.')
    rec.cannot('values', 'The README shows these names without a signature.', required=False)


def _blob_center(image: Any) -> tuple[float, float, float]:
    """Center of the pixels above half of the peak, in continuous pixel coordinates (pixel i spans i to i+1)."""
    import numpy as np
    luminance = np.asarray(image, dtype=np.float64).sum(-1)
    peak = float(luminance.max())
    ys, xs = np.nonzero(luminance >= 0.5 * peak)
    weights = luminance[ys, xs]
    return float((xs * weights).sum() / weights.sum() + 0.5), float((ys * weights).sum() / weights.sum() + 0.5), peak


def _axis_sigma(values: Any) -> float:
    """Standard deviation of a 1D Gaussian profile: fit a parabola to the logarithm of the pixels above 20% of the peak."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    keep = np.nonzero(values > 0.2 * values.max())[0]
    curvature = np.polyfit(keep, np.log(values[keep]), 2)[0]
    return float(math.sqrt(-1 / (2 * curvature))) if curvature < 0 else float('nan')


def check_metalsplat(rec: Recorder) -> None:
    # README of metalsplat at 84e8c3a, the API this check relies on:
    #   model = GaussianModel.random(n=5000, bound=1.0, device="mps")
    #   camera = Camera.identity(fx=128, fy=128, cx=64, cy=64, img_width=128, img_height=128).to("mps")
    #   image = render(model, camera)  # (H, W, 3), differentiable
    #   loss.backward()  # gradients flow to model.means, .raw_scales, .raw_quats, .raw_opacities, .raw_colors
    #   See `examples/fit_image.py` for a full training loop
    #   `render()` / `render_2dgs()` always use the Metal kernels and need tensors on the `mps` device.
    # The README does not show how to place one known Gaussian. The example it names does:
    #   model = GaussianModel(means, scales=scales, colors=colors).to(DEVICE)
    #   # Place gaussians at depth z = fx so ... world x/y and scale then map 1:1 onto pixel offsets/sizes
    # The README does not state the image axes. The code comment at this commit says
    # `v = fy*y/z + cy, and v grows downward`, and the hand computation uses that.
    from metalsplat import Camera, GaussianModel, render
    import numpy as np
    import torch

    size, focal = 128, 128.0
    camera = rec.attempt('camera_identity', lambda: Camera.identity(fx=focal, fy=focal, cx=size / 2, cy=size / 2, img_width=size, img_height=size).to('mps'))
    if camera is None:
        return

    def readme_usage() -> Any:
        torch.manual_seed(0)
        model = GaussianModel.random(n=5000, bound=1.0, device='mps')
        image = render(model, camera)
        image.pow(2).mean().backward()
        torch.mps.synchronize()
        return model, image

    usage = rec.attempt('readme_usage', readme_usage)
    if usage is not None:
        model, image = usage
        rec.expect_mps(image)
        rec.check('readme_image_shape', list(image.shape) == [size, size, 3], list(image.shape))
        for name in ('means', 'raw_scales', 'raw_quats', 'raw_opacities', 'raw_colors'):
            grad = getattr(model, name).grad
            # GaussianModel.random makes round Gaussians, and rotating one changes nothing, so a zero gradient on the rotation is right.
            nonzero = name == 'raw_quats' or bool(grad is not None and grad.abs().sum() > 0)
            rec.check(f'readme_gradient_{name}', grad is not None and bool(torch.isfinite(grad).all()) and nonzero)

    def single(mean: tuple[float, float, float], scale: Any, color: tuple[float, float, float]) -> Any:
        scales = torch.tensor([list(scale) if isinstance(scale, tuple) else [scale] * 3], dtype=torch.float32, device='mps')
        return GaussianModel(torch.tensor([mean], dtype=torch.float32, device='mps'), scales=scales, colors=torch.tensor([color], dtype=torch.float32, device='mps')).to('mps')

    def draw(model: Any) -> Any:
        with torch.no_grad():
            image = render(model, camera)
            torch.mps.synchronize()
        return rec.expect_mps(image)

    # A known Gaussian at depth = fx, so one world unit is one pixel. By hand: u = fx*x/z + cx = 74, v = fy*y/z + cy = 58.
    mean, sigma, color = (10.0, -6.0, focal), 6.0, (0.9, 0.6, 0.2)
    made = rec.attempt('single_gaussian', lambda: draw(single(mean, sigma, color)), api='examples/fit_image.py')
    if made is None:
        return
    image = made.cpu().numpy()
    u, v, _ = _blob_center(image)
    rec.number('peak_u', u)
    rec.number('peak_v', v)
    rec.at_most('peak_error_px_vs_hand', float(math.hypot(u - 74.0, v - 58.0)), 1.0)
    row, column = image[..., 0][int(v)], image[..., 0][:, int(u)]
    rec.at_most('sigma_x_error_vs_hand', abs(_axis_sigma(row) / sigma - 1), 0.05)
    rec.at_most('sigma_y_error_vs_hand', abs(_axis_sigma(column) / sigma - 1), 0.05)
    peak = image[int(v), int(u)]
    ratio = peak / np.array(color)
    rec.number('peak_over_color', ratio.tolist())
    rec.at_most('color_ratio_spread', float(ratio.max() / ratio.min() - 1), 0.02)
    ys, xs = np.mgrid[0:size, 0:size]
    amplitude = float(ratio.mean())
    analytic = amplitude * np.exp(-0.5 * ((xs + 0.5 - 74.0) ** 2 + (ys + 0.5 - 58.0) ** 2) / sigma**2)[..., None] * np.array(color)
    near = (xs + 0.5 - 74.0) ** 2 + (ys + 0.5 - 58.0) ** 2 <= (3 * sigma) ** 2
    rec.at_most('rel_l2_image_vs_analytic_gaussian', float(np.linalg.norm((image - analytic)[near]) / np.linalg.norm(analytic[near])), 0.1)

    wide = rec.attempt('anisotropic_gaussian', lambda: draw(single((0.0, 0.0, focal), (8.0, 4.0, 1.0), color)), api='examples/fit_image.py')
    if wide is not None:
        wide = wide.cpu().numpy()[..., 0]
        rec.at_most('anisotropic_sigma_x_error', abs(_axis_sigma(wide[size // 2]) / 8.0 - 1), 0.05)
        rec.at_most('anisotropic_sigma_y_error', abs(_axis_sigma(wide[:, size // 2]) / 4.0 - 1), 0.05)

    # Compositing by hand, with the far Gaussian first in the tensor: pixel = a1*c1 + (1 - a1)*a2*c2.
    alpha = amplitude
    def overlap() -> Any:
        means = torch.tensor([[0.0, 0.0, focal], [0.0, 0.0, 100.0]], dtype=torch.float32, device='mps')
        colors = torch.tensor([[0.0, 0.0, 0.9], [0.9, 0.0, 0.0]], dtype=torch.float32, device='mps')
        return draw(GaussianModel(means, scales=torch.full((2, 3), sigma, device='mps'), colors=colors).to('mps'))
    composite = rec.attempt('two_gaussians', overlap, api='examples/fit_image.py')
    if composite is not None:
        pixel = composite.cpu().numpy()[size // 2, size // 2]
        near_alpha = float(pixel[0] / 0.9)
        rec.number('composite_pixel', pixel.tolist())
        rec.at_most('near_gaussian_alpha_error', abs(near_alpha / alpha - 1), 0.03)
        rec.at_most('far_gaussian_through_near_error', abs(float(pixel[2]) / (0.9 * (1 - near_alpha) * alpha) - 1), 0.03)

    # The Metal backward kernel against a finite difference, and then an optimization.
    target = draw(single(mean, sigma, color)).detach()
    model = single((0.0, 0.0, focal), sigma, color)

    def mse(value: float) -> float:
        with torch.no_grad():
            model.means.data[0, 0] = value
            result = (render(model, camera) - target).pow(2).mean().item()
            torch.mps.synchronize()
        return result

    finite = (mse(0.5) - mse(-0.5)) / 1.0
    model.means.data[0, 0] = 0.0
    model.zero_grad()
    loss = (render(model, camera) - target).pow(2).mean()
    loss.backward()
    torch.mps.synchronize()
    analytic_grad = model.means.grad[0, 0].item()
    rec.number('grad_mean_x_kernel', analytic_grad)
    rec.number('grad_mean_x_finite_difference', finite)
    rec.at_most('grad_mean_x_error_vs_finite_difference', abs(analytic_grad / finite - 1) if finite else float('inf'), 0.2)

    optimizer = torch.optim.Adam([model.means], lr=0.25)
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        loss = (render(model, camera) - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        losses.append(loss.item())
    rec.number('fit_loss_start', losses[0])
    rec.number('fit_loss_end', losses[-1])
    rec.check('fit_loss_fell_by_20x', losses[-1] <= losses[0] / 20, {'start': losses[0], 'end': losses[-1]})
    final = model.means.detach().cpu().numpy()[0]
    rec.number('fit_mean', final.tolist())
    rec.at_most('fit_position_error_vs_true', float(math.hypot(final[0] - mean[0], final[1] - mean[1])), 1.0)


def _write_png(path: Path, image: Any) -> None:
    """Write an 8-bit RGB PNG with the standard library, so the fixture needs no imaging package."""
    import struct
    import zlib

    import numpy as np
    pixels = (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
    height, width, _ = pixels.shape
    raw = b''.join(b'\x00' + pixels[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def _gaussian_image(size: int, u: float, v: float, sigma: float, color: Any) -> Any:
    import numpy as np
    ys, xs = np.mgrid[0:size, 0:size]
    return np.exp(-0.5 * ((xs + 0.5 - u) ** 2 + (ys + 0.5 - v) ** 2) / sigma**2)[..., None] * np.asarray(color)


def _gl_camera(eye: Any) -> Any:
    """Camera-to-world, OpenGL convention: +x right, +y up, looks along -z, at the origin."""
    import numpy as np
    eye = np.asarray(eye, dtype=np.float64)
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, np.cross(right, forward), -forward, eye
    return pose


def _project_gl(pose: Any, point: Any, focal: float, center: float) -> tuple[float, float, float]:
    import numpy as np
    camera = pose[:3, :3].T @ (np.asarray(point) - pose[:3, 3])
    depth = -camera[2]
    return center + focal * camera[0] / depth, center - focal * camera[1] / depth, depth


def _ring_eye(azimuth_degrees: float, elevation_degrees: float, radius: float) -> Any:
    import numpy as np
    a, e = math.radians(azimuth_degrees), math.radians(elevation_degrees)
    return np.array([radius * math.cos(e) * math.sin(a), radius * math.sin(e), radius * math.cos(e) * math.cos(a)])


BLOB = {'center': (0.4, 0.2, -0.3), 'sigma': 0.25, 'color': (0.9, 0.6, 0.2)}


def _write_nerfstudio_scene(root: Path, count: int, size: int, focal: float, radius: float) -> list[Any]:
    """Views of one Gaussian blob, drawn by hand, in the Nerfstudio layout. Returns the poses in file-name order."""
    import numpy as np
    (root / 'images').mkdir(parents=True, exist_ok=True)
    poses, frames = [], []
    center = size / 2
    for index in range(count):
        # Alternate elevations around the ring, so the mean camera position is the origin.
        pose = _gl_camera(_ring_eye(index * 360.0 / count, 20.0 if index % 2 else -20.0, radius))
        u, v, depth = _project_gl(pose, BLOB['center'], focal, center)
        name = f'images/frame_{index:03d}.png'
        _write_png(root / name, _gaussian_image(size, u, v, focal * BLOB['sigma'] / depth, BLOB['color']))
        poses.append(pose)
        frames.append({'file_path': name, 'transform_matrix': pose.tolist()})
    generator = np.random.default_rng(0)
    points = np.concatenate([generator.uniform(-1, 1, (1500, 3)), np.asarray(BLOB['center']) + generator.normal(0, BLOB['sigma'], (1500, 3))])
    header = ['ply', 'format ascii 1.0', f'element vertex {len(points)}', 'property float x', 'property float y', 'property float z',
              'property uchar red', 'property uchar green', 'property uchar blue', 'end_header']
    lines = [f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128' for p in points]
    (root / 'points3D.ply').write_text('\n'.join(header + lines) + '\n')
    (root / 'transforms.json').write_text(json.dumps({'w': size, 'h': size, 'fl_x': focal, 'fl_y': focal, 'cx': center, 'cy': center, 'ply_file_path': 'points3D.ply', 'frames': frames}))
    return poses


def _read_splat_ply(path: Path) -> dict[str, Any]:
    """The standard 3DGS PLY: every property is float32. This reads the numbers only, and runs no package code."""
    import numpy as np
    head, body = Path(path).read_bytes().split(b'end_header\n', 1)
    names = [line.split()[-1].decode() for line in head.splitlines() if line.startswith(b'property')]
    count = int(next(line for line in head.splitlines() if line.startswith(b'element vertex')).split()[-1])
    rows = np.frombuffer(body, dtype='<f4', count=count * len(names)).reshape(count, -1).astype(np.float64)
    column = dict(zip(names, rows.T))
    quats = np.stack([column[f'rot_{i}'] for i in range(4)], 1)
    return {'means': np.stack([column['x'], column['y'], column['z']], 1), 'opacities': 1 / (1 + np.exp(-column['opacity'])),
            'scales': np.exp(np.stack([column[f'scale_{i}'] for i in range(3)], 1)), 'quats': quats / np.linalg.norm(quats, axis=1, keepdims=True),
            'colors': 0.5 + 0.28209479 * np.stack([column[f'f_dc_{i}'] for i in range(3)], 1)}


def _render_splats_by_hand(splats: dict[str, Any], pose: Any, focal: float, size: int) -> Any:
    """Standard 3DGS by hand, nearest first, with an OpenGL camera. DC color only: no view-dependent spherical harmonics."""
    import numpy as np
    world_to_camera = pose[:3, :3].T
    centers = (splats['means'] - pose[:3, 3]) @ pose[:3, :3]  # Row vectors: this is world_to_camera @ (point - eye).
    ys, xs = np.mgrid[0:size, 0:size]
    image, through = np.zeros((size, size, 3)), np.ones((size, size))
    for i in np.argsort(-centers[:, 2]):
        w, x, y, z = splats['quats'][i]
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        covariance = world_to_camera @ rotation @ np.diag(splats['scales'][i] ** 2) @ rotation.T @ world_to_camera.T
        cx, cy, cz = centers[i]
        depth = -cz
        jacobian = focal / depth * np.array([[1, 0, cx / depth], [0, -1, -cy / depth]])
        inverse = np.linalg.inv(jacobian @ covariance @ jacobian.T + np.eye(2) * 0.3)
        dx, dy = xs + 0.5 - (size / 2 + focal * cx / depth), ys + 0.5 - (size / 2 - focal * cy / depth)
        alpha = splats['opacities'][i] * np.exp(-0.5 * (inverse[0, 0] * dx * dx + 2 * inverse[0, 1] * dx * dy + inverse[1, 1] * dy * dy))
        image += (through * alpha)[..., None] * splats['colors'][i]
        through = through * (1 - alpha)
    return image


def check_msplat(rec: Recorder) -> None:
    # README of msplat 1.1.4 on PyPI, the API this check relies on:
    #   dataset = msplat.load_dataset("path/to/colmap/", eval_mode=True)
    #   config = msplat.TrainingConfig(iterations=7000, num_downscales=0)
    #   trainer = msplat.GaussianTrainer(dataset, config)
    #   trainer.train(lambda s: print(f"step={s.iteration} splats={s.splat_count:,}"), callback_every=100)
    #   trainer.export_ply("output.ply")
    #   metrics = trainer.evaluate()
    #   pose = dataset.camera_pose(0)   # (4, 4) cam-to-world matrix
    #   img = trainer.render_from_pose(pose)  # numpy (H, W, 3) float32
    #   Supported dataset formats: COLMAP, Nerfstudio, Polycam.
    #   Type stubs (`_core.pyi`) are included for IDE autocompletion.
    # The README does not show enough for the planned test, in three ways:
    # - It has no call that places one known Gaussian. So this trains on hand-drawn views of one blob
    #   and checks where the trained scene draws it. It cannot prove the rasterizer alone.
    # - It gives no loss. So "a loss goes down" is held-out PSNR that rises with training.
    # - It does not describe the file layout. This writes the Nerfstudio layout, and the black background
    #   is `bg_color` from the stubs (the default is magenta). Both are from the stubs and loader at v1.1.4.
    # What replaces the known Gaussian: `trainer.export_ply` writes the trained splats, and a hand render of
    # them must land where msplat's own render does. That tests the rasterizer apart from the training.
    # That limit of 1 px was added after the first run, where the hand render differed by 0.12 px or less in error size.
    # The peak limit of 3 px against the true blob stays as it was. It measures how well the training fitted the blob.
    import msplat
    import numpy as np
    size, focal, radius, count = 128, 150.0, 3.0, 24
    scene = Path('scene').resolve()
    poses = _write_nerfstudio_scene(scene, count, size, focal, radius)
    rec.cannot('single_known_gaussian', 'The README has no call that places a known Gaussian, so the rasterizer was not tested alone.', required=False)

    def trained(iterations: int, eval_mode: bool) -> Any:
        dataset = msplat.load_dataset(str(scene), eval_mode=eval_mode)
        config = msplat.TrainingConfig(iterations=iterations, num_downscales=0, bg_color=[0.0, 0.0, 0.0])
        trainer = msplat.GaussianTrainer(dataset, config)
        trainer.train(lambda s: None, callback_every=100)
        return dataset, trainer

    baseline = GpuSampler.read()
    with GpuSampler() as sampler:
        short = rec.attempt('train_short', lambda: trained(30, True), api='README and _core.pyi')
        long = rec.attempt('train_long', lambda: trained(1500, True), api='README and _core.pyi')
    gpu = sampler.summary(baseline)
    rec.number('gpu_system_wide', gpu)
    busy = sampler.busy(baseline)
    rec.set_device('gpu_busy_during_training' if busy else 'unconfirmed',
                   {'note': 'msplat returns numpy arrays, so no array shows a device. This is system-wide GPU use from ioreg during training, so other apps count.', **gpu})
    if not busy:
        rec.cannot('device', f'The system-wide GPU use did not rise during training, so the GPU is not confirmed: {gpu}')
    if short is None or long is None:
        return
    before = rec.attempt('evaluate_short', short[1].evaluate)
    after = rec.attempt('evaluate_long', long[1].evaluate)
    if before is not None and after is not None:
        rec.number('heldout_psnr_30_iterations', before['psnr'])
        rec.number('heldout_psnr_1500_iterations', after['psnr'])
        rec.number('heldout_ssim_1500_iterations', after['ssim'])
        rec.check('heldout_psnr_rose_by_3_db', after['psnr'] >= before['psnr'] + 3, {'before': before['psnr'], 'after': after['psnr']})
        rec.check('heldout_psnr_at_least_25_db', after['psnr'] >= 25, after['psnr'])
    rec.number('splats_after_1500_iterations', long[1].splat_count)

    # Every camera is in the training set here, so camera_pose(i) is the i-th image in file-name order.
    full = rec.attempt('train_all_views', lambda: trained(1500, False), api='README and _core.pyi')
    if full is None:
        return
    dataset, trainer = full
    center = size / 2
    errors, renders = [], {}
    for index in (0, 7, 13, 19):
        rendered = rec.attempt(f'render_from_pose_{index}', lambda i=index: trainer.render_from_pose(dataset.camera_pose(i)))
        if rendered is None:
            continue
        got_u, got_v, _ = _blob_center(rendered)
        want_u, want_v, _ = _project_gl(poses[index], BLOB['center'], focal, center)
        errors.append(float(math.hypot(got_u - want_u, got_v - want_v)))
        renders[index] = rendered
    if errors:
        rec.number('training_view_peak_errors_px', errors)
        rec.at_most('training_view_peak_error_px', max(errors), 3.0)

    # A pose that no training camera took. msplat rescales the world, so the translation is scaled by the same factor.
    scale = float(np.linalg.norm(dataset.camera_pose(0)[:3, 3]) / np.linalg.norm(poses[0][:3, 3]))
    rec.number('msplat_world_scale', scale)
    novel = _gl_camera(_ring_eye(7.5, 0.0, radius))
    scaled = novel.copy()
    scaled[:3, 3] *= scale
    rendered = rec.attempt('render_novel_pose', lambda: trainer.render_from_pose(scaled.astype(np.float32)))
    if rendered is not None:
        got_u, got_v, _ = _blob_center(rendered)
        want_u, want_v, depth = _project_gl(novel, BLOB['center'], focal, center)
        rec.at_most('novel_view_peak_error_px', float(math.hypot(got_u - want_u, got_v - want_v)), 3.0)
        mse = float(np.mean((rendered - _gaussian_image(size, want_u, want_v, focal * BLOB['sigma'] / depth, BLOB['color'])) ** 2))
        rec.number('novel_view_psnr_vs_hand', -10 * math.log10(max(mse, 1e-12)))
        rec.check('novel_view_psnr_at_least_20_db', mse <= 0.01, mse)
    exported = rec.attempt('export_ply', lambda: trainer.export_ply('output.ply') or Path('output.ply').stat().st_size)
    if exported is not None:
        rec.check('export_ply_wrote_a_file', exported > 0, exported)
        splats = _read_splat_ply(Path('output.ply'))
        rec.number('exported_splats', len(splats['means']))
        mass = splats['opacities'] * splats['scales'].prod(1)
        mean = (splats['means'] * mass[:, None]).sum(0) / mass.sum()
        rec.number('splat_mean_error_vs_true_center_world_units', float(np.linalg.norm(mean / scale - np.asarray(BLOB['center']))))
        differences = []
        for index, rendered in renders.items():
            by_hand = _render_splats_by_hand(splats, dataset.camera_pose(index).astype(np.float64), focal, size)
            hand_u, hand_v, _ = _blob_center(by_hand)
            got_u, got_v, _ = _blob_center(rendered)
            differences.append(float(math.hypot(got_u - hand_u, got_v - hand_v)))
        if differences:
            rec.number('render_vs_hand_render_of_splats_px', differences)
            rec.at_most('render_vs_hand_render_of_splats_max_px', max(differences), 1.0)


def _project_opencv(eye: Any, point: Any, focal: float, center: float, up: Any = (0.0, 1.0, 0.0)) -> tuple[float, float, float]:
    """The README's convention: an OpenCV/COLMAP camera at `eye` that looks at the origin. +x right, +y down, +z forward."""
    import numpy as np
    eye = np.asarray(eye, dtype=np.float64)
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    relative = np.asarray(point, dtype=np.float64) - eye
    depth = float(relative @ forward)
    return center + focal * float(relative @ right) / depth, center + focal * float(relative @ down) / depth, depth


def check_mlx3d(rec: Recorder) -> None:
    # README of mlx3d 0.3.2 on PyPI, the API this check relies on:
    #   model = GaussianModel.from_points(points=mx.random.normal((10_000, 3)) * 0.5, colors=mx.random.uniform(shape=(10_000, 3)))
    #   camera = Camera.look_at(eye=(0, 0, -4), at=(0, 0, 0), width=1280, height=720)
    #   out = model.render(camera)            # differentiable end to end
    #   print(out["image"].shape)             # (720, 1280, 3)
    #   OpenCV/COLMAP-convention pinhole cameras (ray generation, projection, look-at) and batched rotation conversions (quaternion, axis-angle, Euler, 6D).
    #   uv run python examples/render_mesh.py        # soft mesh rasterization
    #   uv run python examples/fit_gaussians.py      # fit 3D Gaussians
    # The README does not show enough for the rest, so these come from elsewhere and each is marked in `api`:
    # - `fov=45.0` for a known focal length, `GaussianTrainer`, `TrainerConfig`, `render_mesh_soft`, `ico_sphere`: examples named above.
    # - `quaternion_to_matrix`, `axis_angle_to_matrix`: docs/api/transforms.md, which the README's documentation link covers.
    # - `Camera.project_points`: source at v0.3.2. The README names only "projection".
    # The README names chamfer distance and k-NN without an API, so those are not run.
    import mlx.core as mx
    from mlx3d.cameras import Camera
    from mlx3d.splatting import GaussianModel
    import numpy as np
    rec.cannot('chamfer_and_knn', 'The README names them without a call.', required=False)
    mx.random.seed(0)
    size = 128
    focal = (size / 2) / math.tan(math.radians(45.0) / 2)
    eye = (0.0, 0.0, -4.0)

    def readme_quick_example() -> Any:
        model = GaussianModel.from_points(points=mx.random.normal((10_000, 3)) * 0.5, colors=mx.random.uniform(shape=(10_000, 3)))
        out = model.render(Camera.look_at(eye=(0, 0, -4), at=(0, 0, 0), width=128, height=72))
        mx.eval(out['image'])
        return out

    quick = rec.attempt('readme_quick_example', readme_quick_example)
    if quick is not None:
        rec.check('readme_image_shape', list(quick['image'].shape) == [72, 128, 3], list(quick['image'].shape))
        rec.check('readme_image_is_finite', bool(np.isfinite(np.array(quick['image'])).all()))

    # A tight cluster stands in for one Gaussian, because the README does not show how to set scales.
    center = (0.5, 0.25, 0.0)

    def cluster() -> tuple[Any, Any]:
        points = mx.random.normal((64, 3)) * 0.02 + mx.array(center)
        return GaussianModel.from_points(points=points, colors=mx.ones((64, 3))), Camera.look_at(eye=eye, at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=size, height=size)

    built = rec.attempt('cluster', cluster, api='examples/fit_gaussians.py')
    if built is not None:
        model, camera = built
        out = rec.attempt('cluster_render', lambda: model.render(camera))
        if out is not None:
            mx.eval(out['image'])
            image = np.array(out['image'])
            got_u, got_v, peak = _blob_center(image)
            want_u, want_v, depth = _project_opencv(eye, center, focal, size / 2)
            rec.number('cluster_peak_luminance', peak)
            rec.number('cluster_center_uv', [got_u, got_v])
            rec.number('cluster_center_uv_by_hand', [want_u, want_v])
            rec.check('cluster_is_visible', peak > 0.1, peak)
            rec.at_most('cluster_peak_error_px_vs_hand', float(math.hypot(got_u - want_u, got_v - want_v)), 1.5)

            projected = rec.attempt('project_points', lambda: camera.project_points(mx.array([center])), api='source at v0.3.2')
            if projected is not None:
                xy = np.array(projected[0])[0]
                rec.at_most('project_points_error_px_vs_hand', float(math.hypot(xy[0] - want_u, xy[1] - want_v)), 1e-2)
                rec.at_most('project_points_depth_error_vs_hand', abs(float(np.array(projected[1])[0]) - depth), 1e-4)

            # Does it depend on the GPU? MLX arrays have no device of their own, so this is the only direct test.
            try:
                with mx.stream(mx.cpu):
                    on_cpu = model.render(camera)
                    mx.eval(on_cpu['image'])
                rec.number('render_on_cpu_stream', {'ran': True, 'rel_l2_vs_default': float(np.linalg.norm(np.array(on_cpu['image']) - image) / max(np.linalg.norm(image), 1e-12))})
            except Exception as exc:
                rec.number('render_on_cpu_stream', {'ran': False, 'error': f'{type(exc).__name__}: {exc}'[:300]})

    def rotation_checks() -> None:
        from mlx3d.transforms import axis_angle_to_matrix, quaternion_to_matrix
        half = math.sqrt(0.5)
        by_hand = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        # A quarter turn about +z, real part first. Real-first is the order in the source at v0.3.2.
        rec.at_most('quaternion_quarter_turn_error', float(np.abs(np.array(quaternion_to_matrix(mx.array([[half, 0.0, 0.0, half]])))[0] - by_hand).max()), 1e-5)
        generator = np.random.default_rng(0)
        q = generator.normal(size=(16, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        closed = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                           2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
        rec.at_most('quaternion_batch_error_vs_closed_form', float(np.abs(np.array(quaternion_to_matrix(mx.array(q.astype(np.float32)))) - closed).max()), 1e-5)
        vectors = generator.normal(size=(16, 3))
        angles = np.linalg.norm(vectors, axis=1)
        k = vectors / angles[:, None]
        skew = np.zeros((16, 3, 3))
        skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 0], skew[:, 1, 2], skew[:, 2, 0], skew[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
        rodrigues = np.eye(3) + np.sin(angles)[:, None, None] * skew + (1 - np.cos(angles))[:, None, None] * skew @ skew
        rec.at_most('axis_angle_error_vs_rodrigues', float(np.abs(np.array(axis_angle_to_matrix(mx.array(vectors.astype(np.float32)))) - rodrigues).max()), 1e-4)

    rec.attempt('rotation_checks', rotation_checks, api='docs/api/transforms.md')

    def mesh_check() -> None:
        from mlx3d.renderer import render_mesh_soft
        from mlx3d.utils import ico_sphere
        mesh = ico_sphere(level=3, radius=1.0)
        camera = Camera.look_at(eye=eye, at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=size, height=size)
        out = render_mesh_soft(camera, mesh, verts_colors=mx.ones_like(mesh.verts_packed()), sigma=3e-3, background=(0.0, 0.0, 0.0))
        mx.eval(out['alpha'])
        alpha = np.array(out['alpha'])
        # The hand value is fx * tan(asin(1/4)), the silhouette of a unit sphere at distance 4.
        by_hand = focal * math.tan(math.asin(1 / 4))
        # This alpha is not coverage. Source at v0.3.2: alpha = 1 - exp(-sum(w)), w = exp(min(T*(1/z - 1/min_z), 0)), T = 25.
        # A pixel has a front and a back face, so by hand alpha is about 0.65 at the center and 0.32 at the rim.
        # The first run measured `alpha > 0.5`, which gave 27.03 px against 39.89 px. That threshold tests the wrong
        # thing: alpha = 0.5 falls at 27.3 px by this formula. The edge is measured at 0.15, under half of the rim
        # value, and the 5% limit is unchanged. Each alpha value below is also compared with the formula.
        measured = math.sqrt(float((alpha > 0.15).sum()) / math.pi)
        rec.number('sphere_silhouette_radius_px', measured)
        rec.number('sphere_silhouette_radius_px_by_hand', by_hand)
        rec.at_most('sphere_silhouette_radius_error', abs(measured / by_hand - 1), 0.05)
        min_z = float(np.array(mesh.verts_packed())[:, 2].min()) + 4.0

        def soft_alpha(radius: float) -> float:
            angle = math.atan(radius / focal)
            root = math.sqrt(max(1 - (4 * math.sin(angle)) ** 2, 0.0))
            depths = [(4 * math.cos(angle) + sign * root) * math.cos(angle) for sign in (-1, 1)]
            return 1 - math.exp(-sum(math.exp(min(25.0 * (1 / z - 1 / min_z), 0.0)) for z in depths))

        center_alpha = float(alpha[size // 2 - 1:size // 2 + 1, size // 2 - 1:size // 2 + 1].mean())
        rec.number('sphere_alpha_center', center_alpha)
        rec.number('sphere_alpha_center_by_formula', soft_alpha(0.0))
        rec.at_most('sphere_alpha_center_error_vs_formula', abs(center_alpha / soft_alpha(0.0) - 1), 0.05)
        half = math.sqrt(float((alpha > 0.5).sum()) / math.pi)
        low, high = 0.0, by_hand
        for _ in range(50):
            middle = (low + high) / 2
            low, high = (middle, high) if soft_alpha(middle) > 0.5 else (low, middle)
        rec.number('sphere_alpha_half_radius_px', half)
        rec.number('sphere_alpha_half_radius_px_by_formula', low)
        rec.at_most('sphere_alpha_half_radius_error_vs_formula', abs(half / low - 1), 0.05)

    def hard_mesh_check() -> None:
        # The closer match to nvdiffrast: a hard z-buffer. Its docstring calls it the O(H*W) hard rasterizer. It is optional, because the README does not name it.
        from mlx3d.renderer.rasterizer import rasterize_meshes
        from mlx3d.utils import ico_sphere
        fragments = rasterize_meshes(Camera.look_at(eye=eye, at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=size, height=size), ico_sphere(level=3, radius=1.0))
        covered = np.array(fragments.valid)
        measured = math.sqrt(float(covered.sum()) / math.pi)
        rec.number('hard_rasterizer_silhouette_radius_px', measured)
        rec.number('hard_rasterizer_silhouette_radius_error', abs(measured / (focal * math.tan(math.asin(1 / 4))) - 1))

    rec.attempt('hard_rasterizer', hard_mesh_check, api='source at v0.3.2', required=False)
    rec.attempt('mesh_rasterizer', mesh_check, api='examples/render_mesh.py')

    def train() -> tuple[Any, list[float]]:
        from mlx3d.splatting import GaussianTrainer, TrainerConfig
        res, views_radius, fit_focal = 64, 3.5, 32 / math.tan(math.radians(45.0) / 2)
        views = []
        for elevation in (-20.0, 20.0):
            for azimuth in range(0, 360, 45):
                position = tuple(float(c) for c in _ring_eye(float(azimuth), elevation, views_radius))
                u, v, d = _project_opencv(position, BLOB['center'], fit_focal, res / 2)
                target = _gaussian_image(res, u, v, fit_focal * BLOB['sigma'] / d, BLOB['color']).astype(np.float32)
                views.append((Camera.look_at(eye=position, at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=res, height=res), mx.array(target)))
        key = mx.random.normal((2000, 3))
        points = key / mx.maximum(mx.linalg.norm(key, axis=-1, keepdims=True), 1e-6) * mx.random.uniform(shape=(2000, 1)) ** (1 / 3) * 0.9
        model = GaussianModel.from_points(points, mx.random.uniform(shape=(2000, 3)), sh_degree=2)
        trainer = GaussianTrainer(model, TrainerConfig(densify_from=100, densify_until=600, densify_every=100, opacity_reset_every=10_000, sh_increase_every=200, white_background=False), scene_extent=1.0)
        losses = []
        for iteration in range(600):
            camera, target = views[iteration % len(views)]
            losses.append(float(trainer.step(camera, target)['loss']))
        return model, losses

    baseline = GpuSampler.read()
    with GpuSampler() as sampler:
        fitted = rec.attempt('train_gaussians', train, api='examples/fit_gaussians.py')
    if fitted is not None:
        model, losses = fitted
        start, end = float(np.mean(losses[:16])), float(np.mean(losses[-16:]))
        rec.number('fit_loss_start', start)
        rec.number('fit_loss_end', end)
        rec.check('fit_loss_fell_by_30_percent', end <= 0.7 * start, {'start': start, 'end': end})
        position = tuple(float(c) for c in _ring_eye(22.5, 0.0, 3.5))
        out = model.render(Camera.look_at(eye=position, at=(0, 0, 0), up=(0, 1, 0), fov=45.0, width=64, height=64))
        mx.eval(out['image'])
        image = np.array(out['image'])
        fit_focal = 32 / math.tan(math.radians(45.0) / 2)
        u, v, d = _project_opencv(position, BLOB['center'], fit_focal, 32)
        got_u, got_v, _ = _blob_center(image)
        rec.at_most('novel_view_peak_error_px', float(math.hypot(got_u - u, got_v - v)), 3.0)
        mse = float(np.mean((image - _gaussian_image(64, u, v, fit_focal * BLOB['sigma'] / d, BLOB['color'])) ** 2))
        rec.number('novel_view_psnr_vs_hand', -10 * math.log10(max(mse, 1e-12)))

    gpu = sampler.summary(baseline)
    rec.number('gpu_system_wide', gpu)
    rec.check('mlx_default_device_is_gpu', 'gpu' in str(mx.default_device()), str(mx.default_device()))
    rec.set_device(str(mx.default_device()), {'note': 'MLX arrays have no device of their own. This is mx.default_device(), with render_on_cpu_stream and system-wide GPU use from ioreg during training, where other apps count.', 'metal_available': bool(mx.metal.is_available()), **gpu})
    if fitted is not None and not sampler.busy(baseline):
        rec.cannot('device', f'The system-wide GPU use did not rise during training, so the GPU is not confirmed: {gpu}')


CHECKS: dict[str, Callable[[Recorder], None]] = {
    'flash_attn': check_flash_attn, 'bitsandbytes': check_bitsandbytes, 'deform_conv': check_deform_conv, 'correlation': check_correlation,
    'carafe': check_carafe, 'conv3d': check_conv3d, 'mpsops': check_mpsops, 'metalsplat': check_metalsplat, 'msplat': check_msplat, 'mlx3d': check_mlx3d,
}


def licence_of(dist_name: str | None) -> dict[str, Any]:
    """What the installed package itself says: its metadata, and the text of any licence file that it ships."""
    out: dict[str, Any] = {'dist': dist_name, 'metadata': None, 'files': [], 'text_found': False}
    if not dist_name:
        return out
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        out['error'] = 'not an installed distribution'
        return out
    out['metadata'] = dist.metadata.get('License-Expression') or dist.metadata.get('License')
    for file in dist.files or []:
        if re.search(r'(^|/)(LICEN[CS]E|COPYING|NOTICE)', str(file), re.IGNORECASE):
            try:
                text = Path(file.locate()).read_text(errors='replace')
            except OSError:
                continue
            first = next((line.strip() for line in text.splitlines() if line.strip()), '')
            out['files'].append({'path': str(file), 'first_line': first, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'text': text[:1500]})
    out['text_found'] = bool(out['files'])
    return out


def child_main(check: str, out: Path, module: str, dist: str | None) -> int:
    """Run one check in the venv of its candidate. This process may crash, and the parent reads what was saved."""
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        sys.path.pop(0)  # A file in spikes/ must not shadow a package that the check imports.
    rec = Recorder(out)
    rec.data['python'] = platform.python_version()
    rec.data['licence'] = licence_of(dist)
    rec.save()
    try:
        imported = importlib.import_module(module)
    except BaseException as exc:
        rec.data['import'] = False
        rec.data['import_error'] = f'{type(exc).__name__}: {exc}'[:800]
        rec.data['import_traceback'] = traceback.format_exc()[-1500:]
        rec.data['finished'] = True
        rec.save()
        return 0
    rec.data['import'] = True
    rec.data['module_version'] = getattr(imported, '__version__', None)
    rec.data['module_file'] = getattr(imported, '__file__', None)
    # A native extension that failed to load can leave a silent pure-Python path, so record which ones loaded.
    rec.data['native_modules'] = sorted(name for name, m in sys.modules.items() if name.startswith(module + '.') and str(getattr(m, '__file__', '')).endswith(('.so', '.dylib')))
    for name in ('torch', 'torchvision', 'mlx', 'numpy', dist):
        if name:
            try:
                rec.data['versions'][name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                rec.data['versions'][name] = None
    rec.save()
    try:
        if check.startswith('file:'):
            spec = importlib.util.spec_from_file_location('m2_check_file', check[len('file:'):])
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
            loaded.check(rec)
        else:
            CHECKS[check](rec)
    except Exception as exc:
        rec.check('check_raised', False, {'error': f'{type(exc).__name__}: {exc}'[:800], 'traceback': traceback.format_exc()[-2000:], 'note': 'Could be a bug in this check. Read the traceback.'})
    rec.close_device()
    # Again here: mps_deform_conv and others load their native library at the first call, not at import.
    rec.data['native_modules'] = sorted(name for name, m in sys.modules.items() if name.startswith(module + '.') and str(getattr(m, '__file__', '')).endswith(('.so', '.dylib')))
    rec.data['finished'] = True
    rec.save()
    return 0


# --------------------------------------------------------------------------- substitute runs under `cumlx run`
SUB_PRELUDE = '''
import json, sys
OUT = sys.argv[1]
result = {"route": {}, "drop_in": {}}
def save():
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=1, default=str)
def attempt(section, label, fn):
    """Each result is a dict with `ok`. An error is a failed item, and the script goes on."""
    try:
        result[section][label] = fn()
    except Exception as exc:
        result[section][label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    save()
def rel(a, b):
    import torch
    a, b = a.detach().cpu().to(torch.float64), b.detach().cpu().to(torch.float64)
    return float((a - b).norm() / b.norm())
def names(module, *wanted):
    """The attributes of the original package that a real script uses."""
    missing = [w for w in wanted if not hasattr(module, w)]
    return {"ok": not missing, "missing": missing, "present": [w for w in wanted if hasattr(module, w)]}
def imported(name, *wanted):
    import importlib
    return names(importlib.import_module(name), *wanted)
save()
'''

# Each script imports the ORIGINAL name, and it asks for `cuda`, which `cumlx run` redirects. `route` proves the
# import path and the tensors with the substitute's own API. `drop_in` uses the original package's real API.
SUB_SCRIPTS = {
    'flash_attn': r'''
import torch, torch.nn.functional as F
import flash_attn
torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 32, 16, device="cuda") for _ in range(3))
reference = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu())
def route():
    out = flash_attn.flash_attention(q, k, v)
    error = rel(out, reference)
    return {"ok": error <= 1e-4 and out.device.type == "mps", "rel_l2_vs_cpu_sdpa": error, "device": out.device.type}
def drop_in():
    # flash_attn_func(q, k, v, dropout_p, softmax_scale, causal), with (B, S, H, D) tensors.
    out = flash_attn.flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, causal=False).transpose(1, 2)
    error = rel(out, reference)
    return {"ok": error <= 1e-4, "rel_l2_vs_cpu_sdpa": error}
attempt("route", "flash_attention", route)
attempt("drop_in", "names", lambda: imported("flash_attn", "flash_attn_func", "flash_attn_varlen_func"))
attempt("drop_in", "flash_attn_func", drop_in)
''',
    'bitsandbytes': r'''
import torch
import bitsandbytes as bnb
torch.manual_seed(0)
a = torch.randn(32, 8).cuda()
target = a @ torch.randn(8, 1).cuda()
def fit(make):
    w = torch.nn.Parameter(torch.zeros(8, 1).cuda())
    optimizer = make([w])
    for _ in range(100):
        optimizer.zero_grad()
        (a @ w - target).pow(2).mean().backward()
        optimizer.step()
    return w.detach(), (a @ w - target).pow(2).mean().item(), w.device.type
def route():
    w, loss, device = fit(lambda p: bnb.Adam8bit(p, lr=0.05))
    return {"ok": loss < 1e-2 and device == "mps", "loss": loss, "device": device}
def drop_in_optimizer():
    import bitsandbytes.optim
    w, loss, _ = fit(lambda p: bitsandbytes.optim.Adam8bit(p, lr=0.05))
    reference, _, _ = fit(lambda p: torch.optim.Adam(p, lr=0.05))
    error = rel(w, reference)
    return {"ok": error <= 0.1, "rel_l2_vs_torch_adam": error, "loss": loss}
attempt("route", "top_level_adam8bit", route)
attempt("drop_in", "optim_names", lambda: imported("bitsandbytes.optim", "Adam8bit", "AdamW8bit", "Lion8bit", "SGD8bit"))
attempt("drop_in", "nn_names", lambda: imported("bitsandbytes.nn", "Linear4bit", "Linear8bitLt", "Params4bit", "Int8Params"))
attempt("drop_in", "functional_names", lambda: imported("bitsandbytes.functional", "dequantize_4bit"))
attempt("drop_in", "optim_adam8bit", drop_in_optimizer)
''',
    'mmcv_deform': r'''
import torch, torch.nn.functional as F
import mmcv
torch.manual_seed(0)
x, w = torch.randn(1, 4, 8, 8).cuda(), torch.randn(2, 4, 3, 3).cuda()
offset = torch.zeros(1, 18, 8, 8).cuda()
reference = F.conv2d(x.cpu(), w.cpu(), padding=1)
def route():
    out = mmcv.deform_conv2d(x, offset, w, padding=(1, 1))
    error = rel(out, reference)
    return {"ok": error <= 1e-4 and out.device.type == "mps", "rel_l2_zero_offset_vs_conv2d": error, "device": out.device.type}
def drop_in():
    from mmcv.ops import DeformConv2d
    layer = DeformConv2d(4, 2, 3, padding=1).cuda()
    return {"ok": list(layer(x, offset).shape) == [1, 2, 8, 8]}
attempt("route", "deform_conv2d", route)
attempt("drop_in", "mmcv_ops_names", lambda: imported("mmcv.ops", "DeformConv2d", "deform_conv2d", "DeformConv2dPack"))
attempt("drop_in", "mmcv_ops_DeformConv2d", drop_in)
''',
    'mmcv_carafe': r'''
import torch, torch.nn.functional as F
import mmcv
torch.manual_seed(0)
x = torch.randn(1, 4, 8, 8).cuda()
masks = torch.zeros(1, 25, 16, 16).cuda()
masks[:, 12] = 1.0
reference = F.interpolate(x.cpu(), scale_factor=2, mode="nearest")
def route():
    out = mmcv.carafe(x, masks, 5, 1, 2)
    error = rel(out, reference)
    return {"ok": error <= 1e-4 and out.device.type == "mps", "rel_l2_vs_nearest": error, "device": out.device.type}
def drop_in():
    from mmcv.ops import carafe
    error = rel(carafe(x, masks, 5, 1, 2), reference)
    return {"ok": error <= 1e-4, "rel_l2_vs_nearest": error}
attempt("route", "carafe", route)
attempt("drop_in", "mmcv_ops_names", lambda: imported("mmcv.ops", "CARAFE", "CARAFENaive", "CARAFEPack", "carafe"))
attempt("drop_in", "mmcv_ops_carafe", drop_in)
''',
    'correlation': r'''
import torch
import spatial_correlation_sampler as sampler
torch.manual_seed(0)
a, b = torch.randn(1, 8, 12, 12).cuda(), torch.randn(1, 8, 12, 12).cuda()
def route():
    out = sampler.correlation(a, b, kernel_size=1, max_displacement=2, stride1=1, stride2=1, pad_size=2)
    return {"ok": list(out.shape) == [1, 25, 12, 12] and out.device.type == "mps", "shape": list(out.shape), "device": out.device.type}
def drop_in():
    # spatial_correlation_sample(input1, input2, kernel_size, patch_size, stride, padding, dilation_patch): the output has shape (N, PatchH, PatchW, H, W).
    out = sampler.spatial_correlation_sample(a, b, kernel_size=1, patch_size=5, stride=1, padding=0, dilation_patch=1)
    return {"ok": list(out.shape) == [1, 5, 5, 12, 12], "shape": list(out.shape)}
attempt("route", "correlation", route)
attempt("drop_in", "names", lambda: imported("spatial_correlation_sampler", "spatial_correlation_sample", "SpatialCorrelationSampler"))
attempt("drop_in", "spatial_correlation_sample", drop_in)
''',
    'gsplat_metalsplat': r'''
import torch
import gsplat
def route():
    camera = gsplat.Camera.identity(fx=128.0, fy=128.0, cx=64.0, cy=64.0, img_width=128, img_height=128).to("cuda")
    means = torch.tensor([[10.0, -6.0, 128.0]], device="cuda")
    model = gsplat.GaussianModel(means, scales=torch.full((1, 3), 6.0, device="cuda"), colors=torch.full((1, 3), 0.8, device="cuda")).to("cuda")
    image = gsplat.render(model, camera)
    torch.mps.synchronize()
    y, x = torch.nonzero(image.sum(-1) == image.sum(-1).max())[0].tolist()
    return {"ok": abs(x + 0.5 - 74.0) <= 1.5 and abs(y + 0.5 - 58.0) <= 1.5 and image.device.type == "mps", "peak_xy": [x, y], "device": image.device.type}
def drop_in():
    # gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height), with a batch of cameras.
    d = {"device": "cuda"}
    colors, alphas, _ = gsplat.rasterization(torch.tensor([[10.0, -6.0, 128.0]], **d), torch.tensor([[1.0, 0.0, 0.0, 0.0]], **d), torch.full((1, 3), 6.0, **d), torch.full((1,), 0.5, **d),
                                             torch.full((1, 3), 0.8, **d), torch.eye(4, **d)[None], torch.tensor([[[128.0, 0.0, 64.0], [0.0, 128.0, 64.0], [0.0, 0.0, 1.0]]], **d), 128, 128)
    y, x = torch.nonzero(colors[0].sum(-1) == colors[0].sum(-1).max())[0].tolist()
    return {"ok": abs(x + 0.5 - 74.0) <= 1.5 and abs(y + 0.5 - 58.0) <= 1.5, "peak_xy": [x, y]}
attempt("route", "own_api_under_original_name", route)
attempt("drop_in", "names", lambda: imported("gsplat", "rasterization"))
attempt("drop_in", "rasterization", drop_in)
''',
    'gsplat_msplat': r'''
import gsplat
attempt("route", "own_api_under_original_name", lambda: imported("gsplat", "GaussianTrainer", "TrainingConfig", "load_dataset"))
attempt("drop_in", "names", lambda: imported("gsplat", "rasterization"))
''',
    'pytorch3d': r'''
import math, torch
import mlx.core as mx
import pytorch3d
half = math.sqrt(0.5)
by_hand = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
def route():
    from pytorch3d.transforms import quaternion_to_matrix
    out = torch.tensor(quaternion_to_matrix(mx.array([[half, 0.0, 0.0, half]])).tolist())[0]
    error = float((out - by_hand).abs().max())
    return {"ok": error <= 1e-5, "error_on_mlx_array": error}
def drop_in():
    # A pytorch3d script passes torch tensors and receives torch tensors.
    from pytorch3d.transforms import quaternion_to_matrix
    out = quaternion_to_matrix(torch.tensor([[half, 0.0, 0.0, half]], device="cuda"))
    error = float((out[0].cpu() - by_hand).abs().max())
    return {"ok": torch.is_tensor(out) and error <= 1e-5, "returned_type": type(out).__name__, "error": error}
attempt("route", "transforms_on_mlx_array", route)
attempt("drop_in", "transforms_on_torch_tensor", drop_in)
attempt("drop_in", "ops_names", lambda: imported("pytorch3d.ops", "knn_points", "sample_points_from_meshes"))
attempt("drop_in", "loss_names", lambda: imported("pytorch3d.loss", "chamfer_distance"))
attempt("drop_in", "structures_names", lambda: imported("pytorch3d.structures", "Meshes", "Pointclouds"))
attempt("drop_in", "renderer_names", lambda: imported("pytorch3d.renderer", "MeshRenderer", "MeshRasterizer", "FoVPerspectiveCameras"))
''',
    'nvdiffrast': r'''
import nvdiffrast
attempt("route", "own_renderer_under_original_name", lambda: imported("nvdiffrast.renderer", "render_mesh_soft"))
attempt("drop_in", "torch_names", lambda: imported("nvdiffrast.torch", "RasterizeCudaContext", "RasterizeGLContext", "rasterize", "interpolate", "texture", "antialias"))
''',
}


def substitute_status(section_route: dict, section_drop_in: dict, recorded: bool) -> tuple[str, str]:
    """`pass` needs both the import path and the original API. A renamed API is not a drop-in, and it must not look like one."""
    if not recorded:
        return 'fail', 'run-report.json has no substituted call for the original name'
    if not section_route:
        return 'fail', 'the script stopped before it measured anything'
    if not all(item.get('ok') for item in section_route.values()):
        return 'fail', 'the import path or the own API failed under the original name'
    if not section_drop_in:
        return 'could_not_test', 'this script has no drop-in measurement'
    bad = [name for name, item in section_drop_in.items() if not item.get('ok')]
    if bad:
        return 'fail', f"the import maps, but the original package's API does not work: {', '.join(bad)}"
    return 'pass', ''


def run_substitute(entry: dict, python: Path, folder: Path, target: str, env: dict[str, str], timeout: float) -> dict:
    original, module = entry['original'], entry['module']
    label = f"sub_{original}_{entry.get('script') or 'file'}"
    script = folder / f'{label}.py'
    body = Path(entry['script_file']).read_text() if entry.get('script_file') else SUB_SCRIPTS[entry['script']]
    script.write_text(SUB_PRELUDE + body)
    out_path, report = folder / f'{label}.json', folder / f'{label}.run-report.json'
    argv = [str(python), '-m', 'cumlx', 'run', str(script), '--target', target, '--report', str(report), '--overwrite', '--format', 'json', '--max-items', '-1',
            '--timeout', str(int(timeout)), '--substitute', f'{original}={module}', '--', str(out_path)]
    start = time.monotonic()
    result: dict[str, Any] = {'original': original, 'module': module, 'command': ' '.join(argv)}
    try:
        with (folder / f'{label}.log').open('w') as log:
            process = subprocess.run(argv, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=log, text=True, timeout=timeout + 60, check=False)
    except subprocess.TimeoutExpired:
        result.update(status='fail', reason=f'timed out after {timeout + 60:.0f} s', seconds=round(time.monotonic() - start, 1))
        return result
    result['seconds'] = round(time.monotonic() - start, 1)
    result['exit_code'] = process.returncode
    try:
        doc = json.loads(process.stdout)
    except json.JSONDecodeError:
        result.update(status='could_not_test', reason=f'cumlx run gave no JSON result, exit {process.returncode}')
        return result
    result['errors'] = doc.get('errors')
    if process.returncode == 3:
        result.update(status='could_not_test', reason=f"the environment stopped cumlx run: {doc.get('summary')}")
        return result
    if not out_path.exists():
        result.update(status='fail', reason=f"the script wrote nothing: {doc.get('summary')}")
        return result
    measured = json.loads(out_path.read_text())
    result['route'], result['drop_in'] = measured.get('route', {}), measured.get('drop_in', {})
    calls = json.loads(report.read_text()).get('calls', []) if report.exists() else []
    result['recorded_calls'] = sorted({c['name'] for c in calls if c.get('kind') == 'substituted'})
    result['status'], result['reason'] = substitute_status(result['route'], result['drop_in'], bool(result['recorded_calls']))
    return result


# --------------------------------------------------------------------------- the parent
def install_plan(candidate: dict, python: Path, root: Path = ROOT) -> list[list[str]]:
    """One command, so that the resolver itself must accept the candidate together with the pinned torch."""
    binary = [flag for package in candidate.get('only_binary', []) for flag in ('--only-binary', package)]
    return [['uv', 'pip', 'install', '--python', str(python), *binary, candidate.get('torch', TORCH), *candidate['requirements'], '-e', str(root)]]


def final_status(result: dict) -> tuple[str, str]:
    """Only the listed path reaches `pass`. An unknown state is never a pass."""
    if not result.get('installed'):
        return 'fail', f"not installed: {result.get('install_error', '')[-400:]}"
    if result.get('timed_out'):
        return 'fail', f"the check timed out after {result.get('timeout')} s"
    if result.get('import') is not True:
        if result.get('import') is False:
            return 'fail', f"import failed: {result.get('import_error')}"
        return 'fail', f"the process ended before the import finished, exit {result.get('child_exit')}"
    if result.get('child_exit') != 0 or not result.get('finished'):
        return 'fail', f"the check process ended early, exit {result.get('child_exit')}"
    required = [c for c in result.get('checks', []) if c.get('required', True)]
    failed = [c['name'] for c in required if not c['ok']]
    if failed:
        return 'fail', f"required checks failed: {', '.join(failed)}"
    missing = [c for c in result.get('could_not_test', []) if c.get('required', True)]
    if missing:
        return 'could_not_test', '; '.join(f"{c['name']}: {c['reason']}" for c in missing)[:800]
    if not required:
        return 'could_not_test', 'no required check ran'
    if not result.get('device'):
        return 'could_not_test', 'the device was not observed'
    return 'pass', ''


def child_environment(entry: dict) -> dict[str, str]:
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    # No silent CPU fallback: an operation that MPS lacks must fail here, not pass on the CPU.
    env.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
    env.pop('PYTHONPATH', None)
    # MLX runs float32 matmul in TF32 by default on M5 (about 1e-3 error). The CUMLX default is off, and it keeps that effect out of these limits.
    env['MLX_ENABLE_TF32'] = '0'
    if entry.get('pythonpath'):
        env['PYTHONPATH'] = str(entry['pythonpath'])
    return env


def run_candidate(entry: dict, work: Path, target: str, manifest: bool, check_timeout: float, substitute_timeout: float) -> dict:
    name = entry['name']
    folder = work / 'candidates' / name
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result: dict[str, Any] = {'name': name, 'requirements': entry.get('requirements', []), 'torch_pin': entry.get('torch', TORCH), 'python_pin': entry.get('python'),
                              'source': entry.get('source'), 'commit': entry.get('commit'), 'repo_licence': entry.get('repo_licence'), 'note': entry.get('note'),
                              'installed': False, 'import': None, 'substitutes': []}
    if manifest:
        python, commands = Path(sys.executable), [entry['install']] if entry.get('install') else []
    else:
        venv = work / 'venvs' / name
        python = venv / 'bin' / 'python'
        commands = [['uv', 'venv', '--clear', '-q', '--python', entry['python'], str(venv)], *install_plan(entry, python)]
    install_log = folder / 'install.log'
    with install_log.open('w') as log:
        for command in commands:
            print(f"[{name}] {' '.join(command)}", flush=True)
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=INSTALL_TIMEOUT, check=False).returncode
            except (OSError, subprocess.SubprocessError) as exc:
                log.write(f'{type(exc).__name__}: {exc}\n')
                code = 1
            if code:
                result['install_error'] = install_log.read_text(errors='replace')[-1500:]
                result['install_exit'] = code
                break
    result['install_seconds'] = round(time.monotonic() - start, 1)
    if 'install_error' not in result:
        result['installed'] = True
        env = child_environment(entry)
        if not manifest:
            with (folder / 'freeze.txt').open('w') as freeze:
                subprocess.run(['uv', 'pip', 'freeze', '--python', str(python)], stdout=freeze, stderr=subprocess.DEVNULL, check=False)
        out_path = folder / 'result.json'
        out_path.unlink(missing_ok=True)
        check = f"file:{entry['check_file']}" if entry.get('check_file') else entry['check']
        argv = [str(python), str(Path(__file__).resolve()), '--child', check, '--child-out', str(out_path), '--child-module', entry['module']]
        if entry.get('dist'):
            argv += ['--child-dist', entry['dist']]
        print(f'[{name}] running the check', flush=True)
        try:
            with (folder / 'check.log').open('w') as log:
                result['child_exit'] = subprocess.run(argv, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=check_timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            result['timed_out'], result['timeout'] = True, check_timeout
        if out_path.exists():
            try:
                result.update(json.loads(out_path.read_text()))
            except json.JSONDecodeError:
                result['result_unreadable'] = True
        if result.get('import') is True:
            for substitute in entry.get('substitutes', []):
                print(f"[{name}] running under cumlx run --substitute {substitute['original']}={substitute['module']}", flush=True)
                result['substitutes'].append(run_substitute(substitute, python, folder, target, env, substitute_timeout))
        else:
            result['substitutes'] = [{'original': s['original'], 'module': s['module'], 'status': 'could_not_test', 'reason': 'the module did not import'} for s in entry.get('substitutes', [])]
    if not entry.get('substitutes') and entry.get('no_substitute'):
        result['substitute_not_applicable'] = entry['no_substitute']
    result['status'], result['reason'] = final_status(result)
    result['seconds'] = round(time.monotonic() - start, 1)
    return result


def licence_line(result: dict) -> str:
    licence = result.get('licence') or {}
    if licence.get('text_found'):
        return f"file found, first line {licence['files'][0]['first_line'][:60]!r}, metadata {licence.get('metadata')}"
    return f"no licence file installed, metadata {licence.get('metadata')}"


def print_result(result: dict) -> None:
    device = (result.get('device') or {}).get('used')
    substitute = ', '.join(f"{s['original']}={s['module']}: {s['status']}" for s in result['substitutes']) or result.get('substitute_not_applicable', 'none')
    print(f"[{result['name']}] status {result['status']}. {result.get('reason') or ''}", flush=True)
    print(f"[{result['name']}]   installed {'yes' if result['installed'] else 'no'}, import {'yes' if result.get('import') else 'no'}, device {device}, {result['seconds']} s", flush=True)
    print(f"[{result['name']}]   licence: {licence_line(result)}. Repository: {result.get('repo_licence')}", flush=True)
    print(f"[{result['name']}]   substitute: {substitute}", flush=True)
    for key, value in (result.get('numbers') or {}).items():
        print(f"[{result['name']}]   {key} = {value}", flush=True)
    for check in result.get('checks', []):
        if not check['ok']:
            print(f"[{result['name']}]   FAILED{'' if check.get('required', True) else ' (optional)'} {check['name']}: {str(check.get('detail'))[:300]}", flush=True)
    for item in result.get('could_not_test', []):
        print(f"[{result['name']}]   could_not_test{'' if item.get('required', True) else ' (optional)'} {item['name']}: {item['reason'][:300]}", flush=True)


def write_summary(path: Path, entries: list[dict], results: list[dict], previous: list[dict]) -> None:
    names = {r['name'] for r in results}
    merged = [r for r in previous if r['name'] not in names] + results
    summary = {'schema': SCHEMA, 'date': date.today().isoformat(), 'pins_checked': CHECKED, 'machine': {'platform': platform.platform(), 'machine': platform.machine()},
               'pins': {e['name']: {'requirements': e.get('requirements', []), 'torch': e.get('torch', TORCH), 'python': e.get('python'), 'source': e.get('source'), 'commit': e.get('commit')} for e in entries},
               'decision_inputs': {r['name']: {'status': r['status'], 'reason': r.get('reason'), 'device': (r.get('device') or {}).get('used'), 'licence_text_found': bool((r.get('licence') or {}).get('text_found')),
                                               'substitutes': {f"{s['original']}={s['module']}": s['status'] for s in r['substitutes']}} for r in merged},
               'results': merged}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(_clean(summary), indent=1) + '\n')
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--only', help='comma-separated candidate names. Earlier results for other candidates stay in summary.json')
    parser.add_argument('--manifest', type=Path, help='JSON list of entries in place of CANDIDATES, with no venv and no download. Used by the tests')
    parser.add_argument('--target', choices=['mps', 'cpu'], default='mps', help='target of cumlx run. The tests use cpu where MPS is missing')
    parser.add_argument('--torch', help='a requirement that replaces the pinned torch, such as torch==2.9.1. Use another work folder')
    parser.add_argument('--check-timeout', type=float, default=CHECK_TIMEOUT)
    parser.add_argument('--substitute-timeout', type=float, default=SUBSTITUTE_TIMEOUT)
    parser.add_argument('--child', help=argparse.SUPPRESS)
    parser.add_argument('--child-out', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--child-module', help=argparse.SUPPRESS)
    parser.add_argument('--child-dist', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child_main(args.child, args.child_out, args.child_module, args.child_dist)
    if not args.work:
        parser.error('--work is required')
    work = args.work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    summary_path = work / 'summary.json'
    if args.manifest:
        entries = json.loads(args.manifest.read_text())
    else:
        entries = CANDIDATES
        if shutil.which('uv') is None:
            print('error: uv not found', file=sys.stderr)
            return 3
    if args.torch:
        entries = [{**e, 'torch': args.torch, 'note': f'Not the pinned torch: --torch {args.torch}.'} for e in entries]
    wanted = entries
    if args.only:
        names = set(args.only.split(','))
        unknown = names - {e['name'] for e in entries}
        if unknown:
            print(f"error: unknown candidates: {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        wanted = [e for e in entries if e['name'] in names]
    previous = json.loads(summary_path.read_text()).get('results', []) if args.only and summary_path.exists() else []
    results: list[dict] = []
    for position, entry in enumerate(wanted, 1):
        print(f"[{position}/{len(wanted)}] {entry['name']}", flush=True)
        try:
            result = run_candidate(entry, work, args.target, bool(args.manifest), args.check_timeout, args.substitute_timeout)
        except Exception as exc:
            # A bug in this launcher must not stop the other candidates.
            result = {'name': entry['name'], 'installed': False, 'substitutes': [], 'seconds': 0, 'status': 'fail', 'reason': f'launcher error: {type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()[-1500:]}
        results.append(result)
        print_result(result)
        write_summary(summary_path, entries, results, previous)
    counts = {status: sum(r['status'] == status for r in results) for status in ('pass', 'fail', 'could_not_test')}
    print(f"[done] {counts['pass']} pass, {counts['fail']} fail, {counts['could_not_test']} could_not_test, of {len(results)}. Result: {summary_path}", flush=True)
    return 0 if counts['pass'] == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
