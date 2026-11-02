"""Static triage: how much work is it to run this CUDA project or model on Apple Silicon?

Nothing here runs or imports the scanned code. Files are read as text. The result is a plan, not proof: a pattern that
is not found may still be needed through a dependency, and only `cumlx diff` shows that a port computes the same.

Tiers, from easy to impossible without the vendor:

* ``model``    an architecture that mlx-lm implements. `cumlx port` converts and verifies it.
* ``device``   CUDA is only the device string or a backend switch. Mechanical to change.
* ``library``  a CUDA-only library with a known replacement on Apple Silicon.
* ``kernel``   hand-written CUDA kernels. Each one needs a Metal or MLX rewrite, checked with `cumlx.check_op`.
* ``platform`` the project needs NVIDIA hardware or a closed runtime (RTX ray tracing, Omniverse, PhysX GPU). There is
               no port to write. Replace the subsystem, or keep it on an NVIDIA machine.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from ..errors import CumlxError, missing_dependency

SCHEMA = 'cumlx.scan.v1'
TIER_ORDER = ('model', 'device', 'library', 'kernel', 'platform')
SOURCE_SUFFIXES = {'.py', '.pyx', '.pyi', '.cu', '.cuh', '.cpp', '.cc', '.c', '.h', '.hpp', '.cmake', '.lua', '.toml', '.cfg', '.txt', '.kit', '.yaml', '.yml'}
BUILD_NAMES = {'CMakeLists.txt', 'setup.py', 'setup.cfg', 'pyproject.toml', 'requirements.txt', 'environment.yml', 'package.toml'}
TEXT_ONLY_NAMES = {'requirements.txt', 'environment.yml'}
SKIP_DIRS = {'.git', '.github', 'node_modules', '__pycache__', '.venv', 'venv', 'build', 'dist', 'third_party', 'thirdparty', 'external', 'vendor',
             'docs', 'doc', 'examples', 'example', 'tests', 'test', 'benchmarks', 'notebooks', 'tools', 'scripts', 'asv', 'demo', 'demos'}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_FILES = 40000


@dataclass(frozen=True)
class Rule:
    id: str
    tier: str
    pattern: re.Pattern
    meaning: str
    action: str
    suffixes: frozenset[str] | None = None


def _r(id_: str, tier: str, pattern: str, meaning: str, action: str, suffixes: tuple[str, ...] | None = None, flags: int = 0) -> Rule:
    return Rule(id_, tier, re.compile(pattern, flags), meaning, action, frozenset(suffixes) if suffixes else None)


PY = ('.py', '.pyi', '.pyx')
BUILD = ('.py', '.txt', '.cmake', '.toml', '.cfg', '.lua', '.yaml', '.yml')
RULES: list[Rule] = [
    _r('torch-cuda-device', 'device', r"""\.cuda\(|torch\.cuda\.|device\s*=\s*['"]cuda|['"]cuda:\d""", 'PyTorch selects a CUDA device',
       'Select the device from a setting. On a Mac, use "mps", or port the model to MLX', PY),
    _r('cuda-amp-backends', 'device', r'torch\.backends\.cudnn|torch\.cuda\.amp|autocast\(\s*["\']cuda', 'CUDA-only PyTorch backend settings',
       'Guard them on the device. MPS and MLX have no CuDNN', PY),
    _r('bitsandbytes', 'library', r'\bbitsandbytes\b', 'CUDA quantization and optimizers', 'Use MLX quantization (`mlx_lm.convert -q`)', BUILD),
    _r('flash-attn-import', 'library', r'\b(?:import|from)\s+flash_attn\b|flash_attention_2|flash-attn', 'FlashAttention CUDA kernels',
       'Use `mx.fast.scaled_dot_product_attention` in MLX, or PyTorch sdpa on MPS', BUILD),
    _r('xformers', 'library', r'\bxformers\b', 'Memory-efficient attention for CUDA', 'Use sdpa attention', BUILD),
    _r('apex', 'library', r'\bfrom apex\b|\bimport apex\b', 'NVIDIA Apex mixed precision', 'Use the framework native mixed precision', PY),
    _r('deepspeed', 'library', r'\bimport deepspeed\b|\bfrom deepspeed\b', 'DeepSpeed distributed training', 'Remove it for single-device inference', PY),
    _r('cupy-rapids', 'library', r'\bimport cupy\b|\bfrom cupy\b|\bimport cudf\b|\bimport cuml\b|\bimport cugraph\b', 'GPU array and data libraries',
       'Use NumPy or `mlx.core`. CuPy array code often maps directly', PY),
    _r('cuda-pip-wheels', 'library', r'onnxruntime-gpu|tensorflow-gpu|nvidia-dali|nvidia-cublas|pynvml|nvidia-ml-py', 'A dependency that needs the NVIDIA stack',
       'Use the CPU, CoreML or Metal build, or remove it', ('.txt', '.toml', '.cfg', '.yaml', '.yml', '.py')),
    _r('cuda-kernel-source', 'kernel', r'\b__global__\b|\bcudaLaunchKernel\b|\bcudaMemcpy\b|\bCUDA_CALLABLE\b|__device__\s', 'Hand-written CUDA kernel code',
       'Rewrite in Metal or `mx.fast.metal_kernel`, then compare with `cumlx.check_op`', ('.cu', '.cuh', '.cpp', '.cc', '.h', '.hpp', '.c')),
    _r('cuda-build', 'kernel', r'\bCUDAExtension\b|\bfind_package\(\s*CUDA\b|\bcuda_add_library\b|\bCUDA_ARCHITECTURES\b|\benable_language\(\s*CUDA\b',
       'The build compiles CUDA code', 'Add a Metal target or a CPU fallback to the build', BUILD),
    _r('nvrtc-ptx', 'kernel', r'\bnvrtc\w*\(|\bcuModuleLoad\w*\(|\bPTX\b', 'Runtime CUDA compilation', 'Replace the JIT path with Metal library compilation',
       ('.cu', '.cuh', '.cpp', '.cc', '.h', '.hpp', '.py')),
    _r('warp-cuda', 'kernel', r'\bimport warp\b|\bfrom warp\b', 'NVIDIA Warp. On macOS it runs on the CPU only, without Metal (checked with Warp 1.19)',
       'Run it on the Mac CPU if that is fast enough, or rewrite its kernels in MLX', PY),
    _r('omniverse-kit', 'platform', r'\bimport omni\.|\bfrom omni\.|\bimport carb\b|\bfrom carb\b|\bomni\.kit\.|\bimport isaacsim\b|\bfrom isaacsim\b',
       'NVIDIA Omniverse Kit or Isaac Sim runtime. It is not available for macOS', 'Keep this part on an NVIDIA machine, or replace the runtime', PY),
    _r('optix-rtx', 'platform', r'\boptix\w*\(|\bOptixDeviceContext\b|<optix\.h>|\bRtxSensor\b|\bRTX Lidar\b|isaacsim\.sensors\.rtx', 'Ray tracing on NVIDIA RT cores',
       'Replace with a renderer that runs on Metal, or keep it on an NVIDIA machine', ('.py', '.cpp', '.cc', '.h', '.hpp', '.cu', '.cuh', '.toml', '.kit')),
    _r('physx-gpu', 'platform', r'\benable_gpu_dynamics\b|\bPxCudaContextManager\b|\bPxgDynamics\b|\bgpu_found_lost_pairs\b|\bphysxScene:enableGPUDynamics\b',
       'PhysX GPU dynamics needs CUDA', 'Use PhysX on the CPU, or MuJoCo for simulation on a Mac', ('.py', '.cpp', '.h', '.hpp', '.toml', '.kit')),
    _r('nvenc-nvdec', 'platform', r'\bNvEncoder\b|\bNvDecoder\b|\bnvEncodeAPI\b|\bcuvid\w*\(', 'NVIDIA video hardware', 'Use VideoToolbox', ('.py', '.cpp', '.h', '.hpp')),
]


def _source_files(root: Path, include_tests: bool):
    skip = SKIP_DIRS - ({'tests', 'test'} if include_tests else set())
    count = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.')]
        for name in files:
            path = Path(base) / name
            if path.suffix in SOURCE_SUFFIXES or name in BUILD_NAMES:
                count += 1
                if count > MAX_FILES:
                    return
                yield path


def scan_files(root: Path, *, include_tests: bool = False, max_examples: int = 5) -> dict[str, Any]:
    hits: dict[str, dict[str, Any]] = defaultdict(lambda: {'files': 0, 'matches': 0, 'examples': []})
    scanned = 0
    cuda_sources = 0
    for path in _source_files(root, include_tests):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        scanned += 1
        cuda_sources += path.suffix in ('.cu', '.cuh')
        for rule in RULES:
            if rule.suffixes is not None and path.suffix not in rule.suffixes and path.name not in BUILD_NAMES:
                continue
            found = rule.pattern.findall(text)
            if found:
                hit = hits[rule.id]
                hit['files'] += 1
                hit['matches'] += len(found)
                if len(hit['examples']) < max_examples:
                    hit['examples'].append(str(path.relative_to(root)))
    by_id = {r.id: r for r in RULES}
    findings = [{'id': rid, 'tier': by_id[rid].tier, 'meaning': by_id[rid].meaning, 'action': by_id[rid].action, **h} for rid, h in hits.items()]
    if cuda_sources:
        findings.append({'id': 'cuda-source-files', 'tier': 'kernel', 'meaning': 'The tree holds .cu or .cuh files', 'files': cuda_sources,
                         'matches': cuda_sources, 'examples': [], 'action': 'Count them first: each kernel needs a rewrite and a check_op comparison'})
    findings.sort(key=lambda f: (-TIER_ORDER.index(f['tier']), -f['files']))
    return {'files_scanned': scanned, 'cuda_source_files': cuda_sources, 'findings': findings}


def mlx_lm_architectures() -> set[str] | None:
    """Model types the installed mlx-lm implements. None if mlx-lm is not installed."""
    try:
        import pkgutil

        import mlx_lm.models
    except ImportError:
        return None
    return {m.name for m in pkgutil.iter_modules(mlx_lm.models.__path__)}


def _resolve(target: str, revision: str | None) -> tuple[Path, str]:
    path = Path(target)
    if path.exists():
        return path, 'path'
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise missing_dependency('huggingface_hub', 'diff', purpose='scanning a model by HF id') from None
    try:
        # Text only. Weights stay out, so this is fast on any model size.
        local = snapshot_download(target, revision=revision, allow_patterns=['*.json', '*.py', '*.cu', '*.cuh', '*.txt', '*.toml'])
    except Exception as exc:  # noqa: BLE001 - the hub raises many types
        raise CumlxError('E_INPUT_NOT_FOUND', f'{target!r} is not a path and could not be fetched: {type(exc).__name__}: {exc}',
                         hint='pass a local directory, or check the HF id; gated repos need `huggingface-cli login`') from None
    return Path(local), 'hf'


def _model_block(root: Path, kind: str) -> dict[str, Any] | None:
    config_path = root / 'config.json'
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict) or 'model_type' not in config:
        return None
    known = mlx_lm_architectures()
    model_type = config.get('model_type')
    return {'source': kind, 'model_type': model_type, 'architectures': config.get('architectures'),
            'needs_remote_code': bool(config.get('auto_map')), 'mlx_lm_installed': known is not None,
            'mlx_lm_supports': None if known is None else model_type in known}


def recommend(model: dict | None, findings: list[dict]) -> tuple[str, str]:
    """Return (tier, one-sentence recommendation)."""
    tiers = {f['tier'] for f in findings}
    if model and model.get('mlx_lm_supports') and not model.get('needs_remote_code'):
        return 'model', 'mlx-lm implements this architecture. Convert and verify it with `cumlx port`.'
    if 'platform' in tiers:
        return 'platform', ('This needs NVIDIA hardware or a closed runtime. No port exists to write. Keep that part on an NVIDIA '
                            'machine, or replace the subsystem and verify the parts you can.')
    if 'kernel' in tiers:
        count = sum(f['files'] for f in findings if f['id'] == 'cuda-source-files')
        size = f'{count} CUDA source file(s) were found. ' if count else ''
        return 'kernel', f'{size}Hand-written CUDA kernels need a Metal or MLX rewrite; check first whether each is optional. Verify each with `cumlx.check_op`.'
    if 'library' in tiers:
        return 'library', 'Replace the CUDA-only libraries listed in findings. The rest is ordinary code.'
    if 'device' in tiers:
        return 'device', 'Only device selection is CUDA-specific. Change it, then verify the result with `cumlx capture` and `cumlx diff`.'
    return 'device', 'No CUDA dependency was found in the source. A dependency can still need CUDA; verify the result.'


def scan(target: str, *, revision: str | None = None, include_tests: bool = False) -> dict[str, Any]:
    root, kind = _resolve(target, revision)
    if root.is_file():
        raise CumlxError('E_USAGE', f'{target} is a file', hint='pass a directory or an HF id')
    files = scan_files(root, include_tests=include_tests)
    model = _model_block(root, kind)
    tier, advice = recommend(model, files['findings'])
    return {'schema': SCHEMA, 'target': target, 'tier': tier, 'recommendation': advice, 'model': model, **files,
            'tiers': {t: sum(f['files'] for f in files['findings'] if f['tier'] == t) for t in TIER_ORDER[1:]},
            'limits': 'A text scan. It does not read dependencies and proves nothing about behavior.'}
