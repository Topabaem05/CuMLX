#!/usr/bin/env python3
"""The M3 gates: third-party CUDA kernels, recorded on a V100, translated by woof, and compared on the Mac.

    python spikes/m3_kernels.py fetch   --root R        # sources at pinned commits, each hash checked. Runs none of them
    python spikes/m3_kernels.py cases   --root R        # R/kernels/*.cu and validation/kernel-cases-m3/ (inputs, launches)
    python spikes/m3_kernels.py prepare --root R        # `cumlx kernel port --preprocess-only` on each file. No translator
    python spikes/m3_kernels.py run     --root R        # in woof's own environment: translate, run, compare, one kernel at a time
    python spikes/m3_kernels.py summary --root R        # the gates, from what ran. What did not run is not a pass

Use spikes/m3_kernels.sh to run it: it makes the environment and keeps the run in the background.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any
import urllib.request
import zlib

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CASES_DIR = REPO / 'validation/kernel-cases-m3'
LAB_CASES = REPO / 'validation/kernel-cases/cases.json'
LAB_KERNELS = REPO / 'src/cumlx/kernels'
LAB_GOLDEN = REPO / 'validation/kernel-golden-v100'
EXTERNAL_GOLDEN = REPO / 'validation/kernel-golden-v100-m3'
SEED = 20261103
MAX_ELEMENTS = 20000  # per tensor, so that the goldens stay small enough for git
STEP_TIMEOUT = 240  # a translated kernel that spins must not hold the GPU
MAX_USED_GB = 43.0  # stop the run above this much system memory in use
LAB = ('fp4_decode', 'fp4_gemv', 'fp4_small_m', 'fp4_tiled', 'fp4_w4a4', 'fp4_quantize', 'rms_norm', 'silu_mul')
SAM2 = 'connected_components'
GATE_GOLDENS, GATE_GRADE_A, GATE_ZOO = 20, 0.70, 12
# launch.json's own cases of these three are too big for git, or too small to cross a block. These fit MAX_ELEMENTS and cross one.
EXTRA_CASES = {'embedding_gather': [{'name': 'm3_64x50x300', 'dims': {'tokens': 64, 'vocab': 50, 'dim': 300}}],
               'rope': [{'name': 'm3_16tok_4h_128d', 'dims': {'tokens': 16, 'heads': 4, 'head_dim': 128, 'max_pos': 32}}],
               'transpose': [{'name': 'm3_33x300', 'dims': {'rows': 33, 'cols': 300}}]}


def sources() -> dict[str, Any]:
    return json.loads((CASES_DIR / 'sources.json').read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check(name: str, data: bytes, expected: str) -> None:
    if _sha256(data) != expected:
        raise SystemExit(f'{name}: SHA-256 is {_sha256(data)}, and sources.json says {expected}')


# ---------------------------------------------------------------- fetch
def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 - pinned https URLs from sources.json
        return response.read()


def _safe_members(archive: tarfile.TarFile, wanted: re.Pattern[str]) -> dict[str, bytes]:
    """Only regular files whose path, without the first folder, matches. Nothing is extracted to disk from the archive itself."""
    found: dict[str, bytes] = {}
    for member in archive.getmembers():
        rel = member.name.split('/', 1)[1] if '/' in member.name else ''
        if member.isfile() and wanted.fullmatch(rel):
            handle = archive.extractfile(member)
            assert handle is not None
            found[rel] = handle.read()
    return found


def fetch(root: Path) -> None:
    doc = sources()
    out = root / 'src'
    out.mkdir(parents=True, exist_ok=True)
    repos = doc['repositories']

    def archive(name: str) -> tarfile.TarFile:
        spec = repos[name]
        slug = spec['url'].split('github.com/')[1]
        data = _download(f"https://codeload.github.com/{slug}/tar.gz/{spec['commit']}")
        _check(f'{name} archive', data, spec['archive_sha256'])
        return tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')

    with archive('kernelport') as tar:
        files = _safe_members(tar, re.compile(r'zoo/\w+/(kernel\.cu|launch\.json)'))
    for name, sha in doc['zoo'].items():
        _check(f'zoo/{name}/kernel.cu', files[f'zoo/{name}/kernel.cu'], sha)
        folder = out / 'zoo' / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'kernel.cu').write_bytes(files[f'zoo/{name}/kernel.cu'])
        (folder / 'launch.json').write_bytes(files[f'zoo/{name}/launch.json'])

    with archive('nvdiffrast') as tar:
        files = _safe_members(tar, re.compile(r'csrc/common/.*\.(cu|h|inl)'))
    for rel, data in files.items():
        target = out / 'nvdiffrast' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name, sha in doc['nvdiffrast_files'].items():
        _check(f'nvdiffrast {name}', files[f'csrc/common/{name}.cu'], sha)

    spec = doc['sam2_file']
    url = f"https://raw.githubusercontent.com/facebookresearch/sam2/{repos['sam2']['commit']}/{spec['path']}"
    data = _download(url)
    _check('sam2 connected_components.cu', data, spec['sha256'])
    (out / 'sam2').mkdir(exist_ok=True)
    (out / 'sam2' / f'{SAM2}.cu').write_bytes(data)
    print(f"fetched {len(doc['zoo'])} zoo kernels, {len(doc['nvdiffrast_files'])} nvdiffrast files and sam2 into {out}", flush=True)


# ---------------------------------------------------------------- inputs of the zoo kernels
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}


def evaluate(expr: Any, dims: dict[str, Any]) -> Any:
    """A launch.json expression such as ``min(cdiv(n, 256), 1024)``, without eval: the file comes from a third party."""
    if isinstance(expr, (int, float)):
        return expr

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return dims[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [walk(a) for a in node.args]
            if node.func.id == 'cdiv' and len(args) == 2:
                return (args[0] + args[1] - 1) // args[1]
            if node.func.id == 'min':
                return min(args)
        raise ValueError(f'launch.json expression not understood: {expr!r}')

    return walk(ast.parse(str(expr), mode='eval'))


def _uniform01(gen: np.random.Generator, n: int) -> np.ndarray:
    return gen.random(n)


def generate(init: dict[str, Any], shape: tuple[int, ...], dtype: str, seed: int, dims: dict[str, Any]) -> np.ndarray:
    """The distributions of the zoo's launch.json, with this repository's own generator. Values differ from kernelport's."""
    n = int(np.prod(shape)) if shape else 1
    gen = np.random.default_rng(seed)
    is_float = dtype.startswith('float')
    dist = init.get('dist', 'normal' if is_float else 'randint')
    if dist == 'normal':
        s = sum(_uniform01(gen, n) for _ in range(4)) - 2.0
        vals = init.get('mean', 0.0) + init.get('std', 1.0) * (s * 1.7320508075688772)
    elif dist == 'uniform':
        vals = init.get('low', -1.0) + (init.get('high', 1.0) - init.get('low', -1.0)) * _uniform01(gen, n)
    elif dist == 'randint':
        lo, hi = int(evaluate(init.get('low', 0), dims)), int(evaluate(init.get('high', 100), dims))
        vals = np.floor(lo + (hi - lo) * _uniform01(gen, n))
    elif dist == 'zeros':
        vals = np.zeros(n)
    else:
        raise ValueError(f'unknown dist {dist!r}')
    if init.get('edges') and is_float:
        big = float(init.get('edge_big', 1000.0))
        edges = [0.0, -0.0, 1.0, -1.0, big, -big, 1.0 / big, -1.0 / big][:n]
        vals[:len(edges)] = edges
    if init.get('step') and is_float:
        vals = np.round(vals / init['step']) * init['step']  # sums of these are exact in float32, in any order
    return vals.reshape(shape).astype(dtype)


def zoo_cases(name: str, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Each case of launch.json that fits MAX_ELEMENTS, in float32. The zoo also lists float16 and bfloat16: not recorded here."""
    cases, tensors = [], {}
    types = {'T': 'float32'}
    for raw in [*spec['cases'], *EXTRA_CASES.get(name, [])]:
        dims = raw['dims']
        shapes = {p['name']: tuple(int(evaluate(s, dims)) for s in p.get('shape', [])) for p in spec['params'] if p['kind'] != 'scalar'}
        if any(int(np.prod(s)) > MAX_ELEMENTS for s in shapes.values()):
            continue
        case_id = f"zoo.{name}.{raw['name']}"
        inputs, outputs, call, initial = [], [], [], {}
        for p in spec['params']:
            dtype = types.get(p['dtype'], p['dtype'])
            pname = p['name']
            if p['kind'] == 'scalar':
                value = evaluate(p['value'], dims)
                inputs.append({'name': pname, 'dtype': dtype, 'shape': [], 'values': value})
                call.append(pname)
                continue
            if p['kind'] == 'output':
                outputs.append({'name': pname, 'dtype': dtype, 'shape': list(shapes[pname])})
                call.append(pname)
                continue
            init = {**(spec.get('init') or {}).get(pname, {}), **(raw.get('init') or {}).get(pname, {})}
            seed = (SEED * 1_000_003 + zlib.crc32(f'{name}|{raw["name"]}|{pname}'.encode())) & 0xFFFFFFFF
            data = generate(init, shapes[pname], dtype, seed, dims)
            source = f'{pname}_in' if p['kind'] == 'inout' else pname
            tensor = f'{case_id}/{source}'
            tensors[tensor] = data
            inputs.append({'name': source, 'dtype': dtype, 'shape': list(shapes[pname]), 'file': 'inputs.safetensors', 'tensor': tensor})
            if p['kind'] == 'inout':
                outputs.append({'name': pname, 'dtype': dtype, 'shape': list(shapes[pname])})
                initial[pname] = source
            call.append(pname)
        launch = spec['launch']
        block = [int(evaluate(b, dims)) for b in launch['block']] + [1, 1]
        grid = [int(evaluate(g, dims)) for g in launch['grid']] + [1, 1]
        case = {'id': case_id, 'kernel': name, 'entry': spec['entry'].replace('{T}', 'float'), 'inputs': inputs, 'outputs': outputs, 'call': call,
                'grid': [grid[i] * block[i] for i in range(3)], 'threadgroup': block[:3]}
        if initial:
            case['initial'] = initial
        cases.append(case)
    return cases, tensors


# ---------------------------------------------------------------- inputs of sam2's connected components
BLOCK = 16


def cc_init(h: int, w: int) -> np.ndarray:
    label = np.zeros((h, w), np.int32)
    for row in range(0, h, 2):
        for col in range(0, w, 2):
            label[row, col] = row * w + col
    return label


def _cc_find(s: np.ndarray, n: int) -> int:
    while s[n] != n:
        n = int(s[n])
    return n


def _cc_union(s: np.ndarray, a: int, b: int) -> None:
    a, b = _cc_find(s, a), _cc_find(s, b)
    if a < b:
        s[b] = a
    elif b < a:
        s[a] = b


def cc_merge(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    """One serial order of sam2's `merge`. It makes a valid forest for the later kernels. It is an input, never a reference."""
    h, w = img.shape
    im, s = img.reshape(-1), label.copy().reshape(-1)
    for row in range(0, h, 2):
        for col in range(0, w, 2):
            idx = row * w + col
            p = 0
            if im[idx]:
                p |= 0x777
            if row + 1 < h and im[idx + w]:
                p |= 0x777 << 4
            if col + 1 < w and im[idx + 1]:
                p |= 0x777 << 1
            if col == 0:
                p &= 0xEEEE
            if col + 1 >= w:
                p &= 0x3333
            elif col + 2 >= w:
                p &= 0x7777
            if row == 0:
                p &= 0xFFF0
            if row + 1 >= h:
                p &= 0xFF
            if p > 0:
                if p >> 0 & 1 and im[idx - w - 1]:
                    _cc_union(s, idx, idx - 2 * w - 2)
                if (p >> 1 & 1 and im[idx - w]) or (p >> 2 & 1 and im[idx - w + 1]):
                    _cc_union(s, idx, idx - 2 * w)
                if p >> 3 & 1 and im[idx + 2 - w]:
                    _cc_union(s, idx, idx - 2 * w + 2)
                if (p >> 4 & 1 and im[idx - 1]) or (p >> 8 & 1 and im[idx + w - 1]):
                    _cc_union(s, idx, idx - 2)
    return s.reshape(h, w)


def cc_compress(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    s = label.copy().reshape(-1)
    for row in range(0, h, 2):
        for col in range(0, w, 2):
            s[row * w + col] = _cc_find(s, row * w + col)
    return s.reshape(h, w)


def cc_final(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    h, w = img.shape
    out = label.copy()
    for row in range(0, h, 2):
        for col in range(0, w, 2):
            y = int(label[row, col]) + 1
            for r in (row, row + 1):
                for c in (col, col + 1):
                    if r < h and c < w:
                        out[r, c] = y if img[r, c] else 0
    return out


def cc_count(label: np.ndarray) -> np.ndarray:
    out = np.zeros(label.shape, np.int32).reshape(-1)
    for y in label.reshape(-1):
        if y > 0:
            out[y - 1] += 1
    return out.reshape(label.shape)


def cc_sparse_image(h: int, w: int) -> np.ndarray:
    """Pairs of 2x2 blocks far apart, in the four directions that `merge` tests. Each union joins two roots, so that the
    forest does not depend on the order in which threads run. A denser image can give another valid forest on each run."""
    img = np.zeros((h, w), np.uint8)
    offsets = [((0, 1), (0, 2)), ((1, 0), (2, 0)), ((1, 1), (2, 2)), ((1, 2), (2, 1))]
    index = 0
    for row in range(2, h - 4, 8):
        for col in range(2, w - 4, 8):
            for dr, dc in offsets[index % 4]:
                img[row + dr, col + dc] = 1
            index += 1
    return img


def cc_dense_image(h: int, w: int, seed: int) -> np.ndarray:
    return (np.random.default_rng(seed).random((h, w)) < 0.55).astype(np.uint8)


def sam2_cases() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """One launch of each of the six kernels, as `get_connected_componnets` does, on inputs that the earlier kernels would make."""
    cases, tensors = [], {}

    def tensor(case_id: str, name: str, data: np.ndarray) -> dict[str, Any]:
        key = f'{case_id}/{name}'
        tensors[key] = data
        return {'name': name, 'dtype': str(data.dtype), 'shape': list(data.shape), 'file': 'inputs.safetensors', 'tensor': key}

    def add(entry: str, size: tuple[int, int], tag: str, in_tensors: dict[str, np.ndarray], out_name: str, start: np.ndarray, *, count_grid: bool, order: list[str], scalar: str) -> None:
        h, w = size
        case_id = f'sam2.{entry}.{tag}.{h}x{w}'
        inputs = [tensor(case_id, k, v) for k, v in in_tensors.items()]
        source = f'{out_name}_in'
        inputs.append(tensor(case_id, source, start))
        inputs += [{'name': 'W', 'dtype': scalar, 'shape': [], 'values': w}, {'name': 'H', 'dtype': scalar, 'shape': [], 'values': h}]
        if count_grid:
            blocks = [(w + BLOCK) // BLOCK, (h + BLOCK) // BLOCK]
        else:
            blocks = [((w + 1) // 2 + BLOCK - 1) // BLOCK, ((h + 1) // 2 + BLOCK - 1) // BLOCK]
        cases.append({'id': case_id, 'kernel': SAM2, 'entry': entry, 'inputs': inputs,
                      'outputs': [{'name': out_name, 'dtype': str(start.dtype), 'shape': list(start.shape)}],
                      'call': order, 'initial': {out_name: source},
                      'grid': [blocks[0] * BLOCK, blocks[1] * BLOCK, 1], 'threadgroup': [BLOCK, BLOCK, 1]})

    for i, size in enumerate(((32, 32), (48, 64))):
        h, w = size
        zero = np.zeros(size, np.int32)
        start = cc_init(h, w)
        add('init_labeling', size, 'zeros', {}, 'label', zero, count_grid=False, order=['label', 'W', 'H'], scalar='uint32')
        add('merge', size, 'sparse', {'img': cc_sparse_image(h, w)}, 'label', start, count_grid=False, order=['img', 'label', 'W', 'H'], scalar='uint32')
        img = cc_dense_image(h, w, SEED + i)
        forest = cc_merge(img, start)
        compressed = cc_compress(forest)
        final = cc_final(img, compressed)
        counts = cc_count(final)
        add('compression', size, 'dense', {}, 'label', forest, count_grid=False, order=['label', 'W', 'H'], scalar='int32')
        add('final_labeling', size, 'dense', {'img': img}, 'label', compressed, count_grid=False, order=['img', 'label', 'W', 'H'], scalar='int32')
        add('init_counting', size, 'dense', {'label': final}, 'count_init', zero, count_grid=True, order=['label', 'count_init', 'W', 'H'], scalar='int32')
        add('final_counting', size, 'dense', {'label': final, 'count_init': counts}, 'count_final', zero, count_grid=True,
            order=['label', 'count_init', 'count_final', 'W', 'H'], scalar='int32')
    return cases, tensors


def write_cases(root: Path, out: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / 'src'))
    from cumlx.verify.kernel_port import preprocess_file
    from cumlx.verify.safetensors_io import save_file

    kernels = root / 'kernels'
    kernels.mkdir(parents=True, exist_ok=True)
    doc = sources()
    cases: list[dict[str, Any]] = []
    tensors: dict[str, np.ndarray] = {}
    for name in doc['zoo']:
        folder = root / 'src/zoo' / name
        shutil.copyfile(folder / 'kernel.cu', kernels / f'{name}.cu')
        new_cases, new_tensors = zoo_cases(name, json.loads((folder / 'launch.json').read_text()))
        cases += new_cases
        tensors.update(new_tensors)
    # The original holds host code and includes ATen, so the golden comes from the prepared source. It has the same device code.
    pre = preprocess_file(root / 'src/sam2' / f'{SAM2}.cu')
    (kernels / f'{SAM2}.cu').write_text(pre.source)
    new_cases, new_tensors = sam2_cases()
    cases += new_cases
    tensors.update(new_tensors)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cases.json').write_text(json.dumps({'schema': 'cumlx.kernel-cases.v1', 'seed': SEED, 'cases': cases}, indent=1) + '\n')
    save_file(tensors, out / 'inputs.safetensors')
    return {'cases': len(cases), 'kernels': sorted({c['kernel'] for c in cases}), 'tensors': len(tensors),
            'inputs_bytes': (out / 'inputs.safetensors').stat().st_size}


# ---------------------------------------------------------------- one step at a time
def used_gb() -> float | None:
    try:
        text = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=20).stdout
    except OSError:
        return None
    size = re.search(r'page size of (\d+)', text)
    fields = {k: re.search(rf'{k}:\s+(\d+)', text) for k in ('Pages active', 'Pages wired down', 'Pages occupied by compressor')}
    if not size or not all(fields.values()):
        return None
    return sum(int(m.group(1)) for m in fields.values() if m) * int(size.group(1)) / 1e9


def cumlx_run(argv: list[str], cwd: Path, timeout: int = STEP_TIMEOUT) -> dict[str, Any]:
    """Run one cumlx command in its own process. A kernel that spins or a translator that crashes stops only this step."""
    used = used_gb()
    if used is not None and used > MAX_USED_GB:
        return {'stage': 'not_run', 'reason': f'system memory in use is {used:.1f} GB, above {MAX_USED_GB}'}
    env = {**os.environ, 'PYTHONPATH': str(REPO / 'src'), 'WOOF_BACKEND': 'mlx'}
    try:
        done = subprocess.run([sys.executable, '-m', 'cumlx', *argv, '--format', 'json', '--max-items', '-1'], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'stage': 'timeout', 'reason': f'no result in {timeout} s'}
    try:
        doc = json.loads(done.stdout)
    except json.JSONDecodeError:
        return {'stage': 'crash', 'reason': f'exit {done.returncode}: {done.stderr[-800:]}'}
    return doc


def _error(doc: dict[str, Any]) -> dict[str, Any]:
    if 'stage' in doc:
        return doc
    first = (doc.get('errors') or [{}])[0]
    code = first.get('code')
    stage = {'E_KERNEL_UNPORTABLE': 'refused', 'E_KERNEL_TRANSLATE_FAILED': 'translate', 'E_KERNEL_TRANSLATOR_MISSING': 'no_translator'}.get(code, 'error')
    return {'stage': stage, 'code': code, 'reason': str(first.get('message'))[:1500], 'refused': (first.get('details') or {}).get('refused')}


def port(source: Path, out: Path, cwd: Path, translator: str, extra: list[str]) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = cumlx_run(['kernel', 'port', str(source), '-o', str(out), '--translator', translator, '--overwrite', *extra], cwd)
    if doc.get('exit_code') == 0:
        return {'stage': 'translated', 'warnings': [w.get('code') for w in doc.get('warnings', [])],
                'entry_points': doc['data'].get('metal_entry_points')}
    return _error(doc)


def compare(metal: Path, golden: Path, kernel: str, case_ids: list[str], cwd: Path, engine: str) -> dict[str, Any]:
    argv = ['kernel', 'verify', str(metal), str(golden), '--kernel', kernel, '--engine', engine]
    for case_id in case_ids:
        argv += ['--case', case_id]
    doc = cumlx_run(argv, cwd)
    if 'data' not in doc or doc.get('exit_code') not in (0, 1, 5):
        return _error(doc)
    data = doc['data']
    entry = (data.get('kernels') or {}).get(kernel) or {}
    return {'stage': 'compared', 'status': entry.get('status'), 'grade': entry.get('grade'), 'passed': entry.get('passed'), 'cases': entry.get('cases'),
            'worst_rel_l2': entry.get('worst_rel_l2'), 'first_divergence': data.get('first_divergence'),
            'unverified': data.get('unverified'), 'warnings': [w.get('code') for w in doc.get('warnings', [])]}


def golden_cases(golden: Path) -> dict[str, list[str]]:
    manifest = golden / 'manifest.json'
    if not manifest.is_file():
        return {}
    out: dict[str, list[str]] = {}
    for case in json.loads(manifest.read_text())['cases']:
        out.setdefault(case['kernel'], []).append(case['id'])
    return out


def merge_lab(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """A lab kernel is a body with constants, so each case is translated alone. The kernel passes only if every case passes."""
    bad = next((p for p in parts if p.get('status') != 'pass'), None)
    if bad is None:
        rels = [p['worst_rel_l2'] for p in parts if p.get('worst_rel_l2') is not None]
        return {'stage': 'compared', 'status': 'pass', 'grade': 'A', 'passed': sum(p['passed'] for p in parts), 'cases': sum(p['cases'] for p in parts),
                'worst_rel_l2': max(rels) if rels else None}
    return {**bad, 'cases_tried': len(parts), 'cases_passed': sum(p.get('status') == 'pass' for p in parts)}


def run_all(root: Path, translator: str, engine: str) -> dict[str, Any]:
    doc = sources()
    work = root / 'run'
    metal = root / 'metal'
    work.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    external = golden_cases(EXTERNAL_GOLDEN)
    lab_golden = golden_cases(LAB_GOLDEN)

    def save() -> None:
        (root / 'results.json').write_text(json.dumps(results, indent=1) + '\n')

    for name in doc['zoo']:
        source = root / 'kernels' / f'{name}.cu'
        flags = ['--instantiate', f'{name}<float>']
        step = port(source, metal / 'zoo' / f'{name}.metal', work, translator, flags)
        record: dict[str, Any] = {'group': 'zoo', 'translate': step}
        if step['stage'] == 'refused' and any(r.get('feature') == 'warp-level' for r in step.get('refused') or []):
            # The default refusal is the first result. Woof has __shfl_xor_sync, so this also measures what it does.
            step = port(source, metal / 'zoo' / f'{name}.metal', work, translator, [*flags, '--allow-warp-level'])
            record['translate_with_allow_warp_level'] = step
        if step['stage'] == 'translated':
            record['compare'] = (compare(metal / 'zoo' / f'{name}.metal', EXTERNAL_GOLDEN, name, [], work, engine) if name in external
                                 else {'stage': 'no_golden'})
        results[name] = record
        save()
        print(f"{name}: {json.dumps(record)[:300]}", flush=True)

    source = root / 'kernels' / f'{SAM2}.cu'
    step = port(source, metal / f'{SAM2}.metal', work, translator, [])
    record = {'group': 'sam2', 'translate': step}
    if step['stage'] == 'translated':
        record['compare'] = (compare(metal / f'{SAM2}.metal', EXTERNAL_GOLDEN, SAM2, [], work, engine) if SAM2 in external else {'stage': 'no_golden'})
    results[SAM2] = record
    save()
    print(f"{SAM2}: {json.dumps(record)[:300]}", flush=True)

    for name in LAB:
        parts, translate = [], None
        for case_id in lab_golden.get(name, []):
            out = metal / 'lab' / case_id / f'{name}.metal'
            step = port(LAB_KERNELS / f'{name}.cu', out, work, translator, ['--cases', str(LAB_CASES), '--case', case_id])
            translate = step if translate is None or step['stage'] != 'translated' else translate
            if step['stage'] != 'translated':
                break
            parts.append(compare(out, LAB_GOLDEN, name, [case_id], work, engine))
        record = {'group': 'lab', 'translate': translate or {'stage': 'no_golden'}}
        if parts:
            record['compare'] = merge_lab(parts)
        results[name] = record
        save()
        print(f"{name}: {json.dumps(record)[:300]}", flush=True)

    for name in doc['nvdiffrast_files']:  # sam2 is the repository for the gate. These say what stops nvdiffrast
        folder = root / 'src/nvdiffrast/csrc/common'
        results[f'nvdiffrast.{name}'] = {'group': 'nvdiffrast', 'translate': port(folder / f'{name}.cu', metal / 'nvdiffrast' / f'{name}.metal', work, translator, ['--include-dir', str(folder)])}
        save()

    # The control: the same pipeline must fail a wrong kernel. The plus sign of vector_add becomes a minus sign.
    if 'vector_add' in external:
        wrong = root / 'control' / 'vector_add.cu'
        wrong.parent.mkdir(exist_ok=True)
        text = (root / 'kernels/vector_add.cu').read_text()
        if 'a[i] + b[i]' not in text:
            raise SystemExit('the control expects `a[i] + b[i]` in vector_add')
        wrong.write_text(text.replace('a[i] + b[i]', 'a[i] - b[i]'))
        step = port(wrong, root / 'control/vector_add.metal', work, translator, ['--instantiate', 'vector_add<float>'])
        results['control.wrong_vector_add'] = {'group': 'control', 'translate': step}
        if step['stage'] == 'translated':
            results['control.wrong_vector_add']['compare'] = compare(root / 'control/vector_add.metal', EXTERNAL_GOLDEN, 'vector_add', [], work, engine)
    save()
    return results


def prepare_all(root: Path) -> dict[str, Any]:
    """What this repository's own layer does with each file. Facts, and no translator runs."""
    doc = sources()
    out = root / 'prepared'
    results: dict[str, Any] = {}
    items: list[tuple[str, Path, list[str]]] = [(k, root / 'src/zoo' / k / 'kernel.cu', ['--instantiate', f'{k}<float>']) for k in doc['zoo']]
    items.append((SAM2, root / 'src/sam2' / f'{SAM2}.cu', []))
    folder = root / 'src/nvdiffrast/csrc/common'
    items += [(f'nvdiffrast.{k}', folder / f'{k}.cu', ['--include-dir', str(folder)]) for k in doc['nvdiffrast_files']]
    for name, path, flags in items:
        report = cumlx_run(['kernel', 'port', str(path), '--preprocess-only', '-o', str(out / f'{name}.pre.cu'), '--overwrite', *flags], root)
        if report.get('exit_code') == 0:
            data = report['data']
            results[name] = {'prepared': True, 'entry_points': data['entry_points'], 'prepared_sha256': data['prepared_sha256'],
                             'host_removed': data['preprocess']['host_removed'], 'builtins_rewritten': data['preprocess']['builtins_rewritten']}
        else:
            err = _error(report)
            refused = err.get('refused') or []
            results[name] = {'prepared': False, 'refused_count': len(refused), 'refused_features': sorted({r['feature'] for r in refused}),
                             'refused_tokens': sorted({r['token'] for r in refused})[:12], 'reason': err.get('reason', '')[:300]}
    return results


# ---------------------------------------------------------------- the gates
def classify(record: dict[str, Any]) -> str:
    """One word for why a kernel is not grade A. `no_golden` and `not_run` are open, not failures of the translation."""
    step = record.get('translate_with_allow_warp_level') or record.get('translate') or {}
    if step.get('stage') != 'translated':
        return str(step.get('stage') or 'not_run')
    compared = record.get('compare') or {}
    if compared.get('stage') != 'compared':
        return str(compared.get('stage') or 'not_run')
    if compared.get('status') == 'pass' and compared.get('grade') == 'A':
        return 'grade_a'
    if compared.get('status') == 'fail':
        return 'diverged'
    return f"unverified:{(compared.get('unverified') or [{}])[0].get('stage', 'unknown')}" if compared.get('unverified') else 'unverified'


def gates(results: dict[str, Any], lab_goldens: int, external_goldens: list[str]) -> dict[str, Any]:
    """The four written gates. Each `met` is True only from a run. Nothing is rounded up."""
    doc = sources()
    zoo = list(doc['zoo'])
    kinds = {name: classify(rec) for name, rec in results.items() if rec.get('group') in ('zoo', 'sam2', 'lab')}
    golden_total = lab_goldens + len(external_goldens)
    graded = [k for k in kinds if k in external_goldens or results[k].get('group') == 'lab']
    grade_a = [k for k in graded if kinds[k] == 'grade_a']
    zoo_a = [k for k in zoo if kinds.get(k) == 'grade_a']
    zoo_compared = [k for k in zoo if kinds.get(k) in ('grade_a', 'diverged') or str(kinds.get(k)).startswith('unverified')]
    sam2 = kinds.get(SAM2)
    return {
        'goldens_20': {'met': golden_total >= GATE_GOLDENS, 'count': golden_total, 'lab': lab_goldens, 'external': external_goldens},
        'grade_a_70_percent': {'met': bool(graded) and golden_total >= GATE_GOLDENS and len(grade_a) / max(golden_total, 1) >= GATE_GRADE_A,
                               'grade_a': len(grade_a), 'of_goldens': golden_total, 'by_reason': {k: kinds[k] for k in sorted(kinds)}},
        'zoo_12_compared': {'met': len(zoo_compared) == GATE_ZOO, 'compared': len(zoo_compared), 'grade_a': len(zoo_a),
                            'not_compared': {k: kinds.get(k, 'not_run') for k in zoo if k not in zoo_compared}},
        'one_repository': {'met': sam2 == 'grade_a', 'repository': 'sam2', 'result': sam2 or 'not_run'},
    }


def summary(root: Path) -> dict[str, Any]:
    path = root / 'results.json'
    results = json.loads(path.read_text()) if path.is_file() else {}
    lab = len(golden_cases(LAB_GOLDEN))
    external = sorted(golden_cases(EXTERNAL_GOLDEN))
    control = results.get('control.wrong_vector_add', {})
    out = {'gates': gates(results, lab, external),
           'control_wrong_kernel': {'ran': control.get('compare', {}).get('stage') == 'compared',
                                    'caught': control.get('compare', {}).get('status') == 'fail', 'detail': control or None}}
    (root / 'summary.json').write_text(json.dumps(out, indent=1) + '\n')
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fetch', 'cases', 'prepare', 'run', 'summary'])
    ap.add_argument('--root', default=os.path.expanduser('~/cumlx-m3'))
    ap.add_argument('--translator', default='woof')
    ap.add_argument('--engine', default='woof')
    args = ap.parse_args(argv)
    root = Path(args.root)
    if args.stage == 'fetch':
        fetch(root)
    elif args.stage == 'cases':
        print(json.dumps(write_cases(root, CASES_DIR)), flush=True)
    elif args.stage == 'prepare':
        print(json.dumps(prepare_all(root), indent=1), flush=True)
    elif args.stage == 'run':
        run_all(root, args.translator, args.engine)
        print(json.dumps(summary(root)['gates'], indent=1), flush=True)
    else:
        print(json.dumps(summary(root), indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
