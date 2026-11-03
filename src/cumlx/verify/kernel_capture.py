"""Record op-level goldens: compile the original CUDA kernels with nvcc, run them on a real GPU, keep the outputs.

This module EXECUTES what it compiles, on the host and on the GPU. MCP hides `kernel capture` for that reason.
It never makes up an output: a case that does not compile or run is reported and stays out of the golden.
"""
from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from ..errors import CumlxError
from .kernel_source import is_complete_cuda, mask_source, scan_features
from .opgolden import (
    DTYPES,
    KernelCase,
    TensorSpec,
    read_cases,
    sha256_text,
    write_golden,
)

COMPILE_TIMEOUT_SECONDS = 300
CUDA_TYPES = {'float32': 'float', 'float16': '__half', 'float64': 'double', 'uint8': 'unsigned char', 'int8': 'signed char',
              'uint16': 'unsigned short', 'int16': 'short', 'uint32': 'unsigned int', 'int32': 'int',
              'uint64': 'unsigned long long', 'int64': 'long long'}
# Bytes that fill the output memory before each repeat. An element that the kernel does not write then differs.
FILL_BYTES = (0x00, 0xFF)

# One thread runs this. `one` and `zero` come from a buffer, so that the compiler cannot fold them.
PROBE_INPUT = np.array([1e-40, 2.0 ** -100, 2.0 ** -30, 1 + 2.0 ** -12, 1 + 2.0 ** -12, -(1 + 2.0 ** -11), 1.0, 0.0], np.float32)
PROBE_BODY = '''
y[0] = a[0];
y[1] = a[0] * a[6];
y[2] = a[0] + a[7];
y[3] = a[1] * a[2];
y[4] = a[3] * a[4] + a[5];
'''
PROBE_OUTPUTS = 5


def read_probe(values: np.ndarray) -> dict[str, bool]:
    """Turn the outputs of the probe into facts. ``contracts`` means that a*b+c ran as one fused operation."""
    return {'load_flushes': bool(values[0] == 0), 'input_flushes': bool(values[1] == 0 or values[2] == 0),
            'output_flushes': bool(values[3] == 0), 'contracts': bool(values[4] != 0)}


def _params(case: KernelCase) -> list[str]:
    params = []
    for t in case.inputs:
        params.append(f'const {CUDA_TYPES[t.dtype]} {t.name}' if not t.shape else f'const {CUDA_TYPES[t.dtype]}* {t.name}')
    params += [f'{CUDA_TYPES[t.dtype]}* {t.name}' for t in case.outputs]
    return params


def constants_block(case: KernelCase) -> str:
    return ''.join(f"constexpr {'bool' if isinstance(v, bool) else 'int'} {k} = {str(v).lower() if isinstance(v, bool) else v};\n"
                   for k, v in case.constants.items())


def wrap_body(case: KernelCase, body: str, *, name: str | None = None, extern_c: bool = False) -> str:
    """A complete ``__global__`` function around a body, with the parameters that MLX would generate."""
    prefix = 'extern "C" ' if extern_c else ''
    return f"{prefix}__global__ void {name or case.kernel}({', '.join(_params(case))}) {{\n{body}\n}}\n"


def generate_harness(case: KernelCase, source: str, header: str) -> str:
    """One C++ program: the kernel, a launch of it, and code that reads inputs and writes outputs as raw files."""
    complete = is_complete_cuda(source)
    entry = case.entry or case.kernel
    kernel = source if complete else wrap_body(case, source)
    uses_half = any(t.dtype == 'float16' for t in case.inputs + case.outputs)
    lines = ['#include <cstdio>', '#include <cstdlib>', '#include <cstring>', '#include <string>', '#include <vector>',
             '#include <cuda_runtime.h>']
    if uses_half:
        lines.append('#include <cuda_fp16.h>')
    lines += [header, constants_block(case), kernel, '''
#define CUMLX_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \\
    fprintf(stderr, "CUDA error %s at %s\\n", cudaGetErrorString(e_), #call); exit(2); } } while (0)
static std::vector<char> read_all(const std::string& path, size_t size) {
    std::vector<char> data(size);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f || fread(data.data(), 1, size, f) != size) { fprintf(stderr, "cannot read %s\\n", path.c_str()); exit(3); }
    fclose(f);
    return data;
}
static void write_all(const std::string& path, const void* data, size_t size) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f || fwrite(data, 1, size, f) != size) { fprintf(stderr, "cannot write %s\\n", path.c_str()); exit(3); }
    fclose(f);
}''']
    body = ['int main(int argc, char** argv) {',
            '    if (argc != 3) return 4;',
            '    std::string dir = argv[1];',
            '    int repeats = atoi(argv[2]);',
            f'    static const int fills[{len(FILL_BYTES)}] = {{{", ".join(str(b) for b in FILL_BYTES)}}};']
    args = []
    for t in case.inputs:
        size = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(DTYPES[t.dtype]).itemsize
        body.append(f'    std::vector<char> host_{t.name} = read_all(dir + "/in.{t.name}.bin", {size});')
        if t.shape:
            body += [f'    {CUDA_TYPES[t.dtype]}* dev_{t.name} = nullptr;',
                     f'    CUMLX_CHECK(cudaMalloc(&dev_{t.name}, {size}));',
                     f'    CUMLX_CHECK(cudaMemcpy(dev_{t.name}, host_{t.name}.data(), {size}, cudaMemcpyHostToDevice));']
            args.append(f'dev_{t.name}')
        else:
            body.append(f'    {CUDA_TYPES[t.dtype]} dev_{t.name}; memcpy(&dev_{t.name}, host_{t.name}.data(), {size});')
            args.append(f'dev_{t.name}')
    for t in case.outputs:
        size = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(DTYPES[t.dtype]).itemsize
        body += [f'    {CUDA_TYPES[t.dtype]}* dev_{t.name} = nullptr;',
                 f'    CUMLX_CHECK(cudaMalloc(&dev_{t.name}, {size}));',
                 f'    std::vector<char> host_{t.name}({size});']
        args.append(f'dev_{t.name}')
    body.append('    for (int rep = 0; rep < repeats; ++rep) {')
    for t in case.outputs:
        size = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(DTYPES[t.dtype]).itemsize
        body.append(f'        CUMLX_CHECK(cudaMemset(dev_{t.name}, fills[rep % {len(FILL_BYTES)}], {size}));')
    bx, by, bz = case.blocks
    tx, ty, tz = case.threadgroup
    body += [f'        {entry}<<<dim3({bx}, {by}, {bz}), dim3({tx}, {ty}, {tz})>>>({", ".join(args)});',
             '        CUMLX_CHECK(cudaGetLastError());',
             '        CUMLX_CHECK(cudaDeviceSynchronize());']
    for t in case.outputs:
        size = int(np.prod(t.shape, dtype=np.int64)) * np.dtype(DTYPES[t.dtype]).itemsize
        body += [f'        CUMLX_CHECK(cudaMemcpy(host_{t.name}.data(), dev_{t.name}, {size}, cudaMemcpyDeviceToHost));',
                 f'        write_all(dir + "/out.{t.name}." + std::to_string(rep) + ".bin", host_{t.name}.data(), {size});']
    body += ['    }',
             '    int runtime = 0, driver = 0, device = 0;',
             '    cudaDeviceProp prop;',
             '    CUMLX_CHECK(cudaRuntimeGetVersion(&runtime));',
             '    CUMLX_CHECK(cudaDriverGetVersion(&driver));',
             '    CUMLX_CHECK(cudaGetDevice(&device));',
             '    CUMLX_CHECK(cudaGetDeviceProperties(&prop, device));',
             '    FILE* info = fopen((dir + "/device.json").c_str(), "w");',
             '    if (!info) return 3;',
             r'    fprintf(info, "{\"gpu\": \"%s\", \"arch\": \"%d.%d\", \"cuda_runtime\": %d, \"cuda_driver_api\": %d}\n",',
             '            prop.name, prop.major, prop.minor, runtime, driver);',
             '    fclose(info);',
             '    return 0;',
             '}']
    lines.append('\n'.join(body))
    return '\n'.join(lines) + '\n'


def _run_tool(argv: list[str], *, timeout: float) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f'{argv[0]} ran past {timeout} s'
    except OSError as exc:
        return 127, f'{argv[0]}: {exc}'
    return p.returncode, (p.stdout + p.stderr).strip()


def query_environment() -> dict[str, Any]:
    """Facts about this machine. Raises when it cannot record a CUDA golden."""
    missing = [tool for tool in ('nvcc', 'nvidia-smi') if not shutil.which(tool)]
    if missing:
        raise CumlxError('E_KERNEL_NO_CUDA', f'`cumlx kernel capture` needs an NVIDIA machine: {", ".join(missing)} is not on PATH.',
                         hint='run this command on a machine with an NVIDIA GPU and the CUDA toolkit, then copy the golden here',
                         details={'missing': missing})
    code, text = _run_tool(['nvcc', '--version'], timeout=60)
    match = re.search(r'release [\d.]+, V([\d.]+)', text)
    if code != 0 or not match:
        raise CumlxError('E_KERNEL_NO_CUDA', f'could not read the nvcc version: {text[:300]}')
    code, gpu_text = _run_tool(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], timeout=60)
    if code != 0 or not gpu_text:
        raise CumlxError('E_KERNEL_NO_CUDA', f'nvidia-smi found no usable GPU: {gpu_text[:300]}')
    name, _, driver = gpu_text.splitlines()[0].partition(',')
    return {'nvcc_version': match.group(1), 'gpu': name.strip(), 'driver': driver.strip(), 'host_machine': platform.machine(),
            'host_system': platform.system()}


def busy_processes() -> list[str]:
    code, text = _run_tool(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], timeout=60)
    if code != 0:
        raise CumlxError('E_KERNEL_NO_CUDA', f'nvidia-smi could not list compute processes: {text[:300]}')
    return [line.strip() for line in text.splitlines() if line.strip()]


def _compile(source_path: Path, binary: Path, flags: list[str], timeout: float) -> tuple[int, str]:
    return _run_tool(['nvcc', *flags, '-o', str(binary), str(source_path)], timeout=timeout)


def _read_output(path: Path, spec: TensorSpec) -> np.ndarray:
    dtype = np.dtype(DTYPES[spec.dtype]).newbyteorder('<')
    return np.frombuffer(path.read_bytes(), dtype=dtype).reshape(spec.shape).copy()


def _execute_case(case: KernelCase, arrays: dict[str, np.ndarray], binary: Path, work: Path, repeats: int,
                  timeout: float) -> dict[str, Any]:
    for spec in case.inputs:
        (work / f'in.{spec.name}.bin').write_bytes(arrays[spec.name].astype(np.dtype(DTYPES[spec.dtype]).newbyteorder('<')).tobytes())
    code, text = _run_tool([str(binary), str(work), str(repeats)], timeout=timeout)
    if code != 0:
        raise RuntimeError(f'exit {code}: {text[-1500:]}')
    outputs: dict[str, np.ndarray] = {}
    identical = True
    for spec in case.outputs:
        first = (work / f'out.{spec.name}.0.bin').read_bytes()
        identical = identical and all((work / f'out.{spec.name}.{r}.bin').read_bytes() == first for r in range(1, repeats))
        outputs[spec.name] = _read_output(work / f'out.{spec.name}.0.bin', spec)
    device = json.loads((work / 'device.json').read_text())
    return {'outputs': outputs, 'repeat_identical': identical, 'device': device}


def _probe(work: Path, flags: list[str], timeout: float) -> dict[str, Any]:
    case = KernelCase('probe', 'cumlx_probe', [TensorSpec('a', 'float32', (len(PROBE_INPUT),))],
                      [TensorSpec('y', 'float32', (PROBE_OUTPUTS,))])
    work.mkdir(parents=True)
    source = work / 'probe.cu'
    source.write_text(generate_harness(case, PROBE_BODY, ''))
    code, text = _compile(source, work / 'probe.bin', flags, timeout)
    if code != 0:
        raise CumlxError('E_KERNEL_COMPILE', f'the floating-point probe did not compile: {text[-1500:]}',
                         hint='nvcc itself does not work with these flags')
    return _execute_case(case, {'a': PROBE_INPUT}, work / 'probe.bin', work, 3, timeout)


def _flag_value(flags: list[str], name: str, default: str) -> str:
    for flag in flags:
        match = re.match(rf'^-{{1,2}}{name}[= ](\S+)$', flag)
        if match:
            return match.group(1)
    return f'{default} (nvcc default)'


def plan_capture(kernels: str | Path, cases_path: str | Path, header: str | None) -> dict[str, Any]:
    """Read and check everything that can be checked without a GPU."""
    root = Path(kernels)
    if not root.exists():
        raise CumlxError('E_INPUT_NOT_FOUND', f'{root} does not exist', hint='pass a folder of .cu files or one .cu file')
    folder = root if root.is_dir() else root.parent
    available = {p.stem: p for p in sorted(folder.glob('*.cu'))} if root.is_dir() else {root.stem: root}
    if not available:
        raise CumlxError('E_INPUT_NOT_FOUND', f'{root} has no .cu files')
    header_path = Path(header) if header else folder / 'common_cuda.h'
    if header and not header_path.is_file():
        raise CumlxError('E_INPUT_NOT_FOUND', f'--header {header} does not exist')
    header_text = header_path.read_text() if header_path.is_file() else ''
    cases = read_cases(cases_path)
    unknown = sorted({c.kernel for c, _ in cases} - set(available))
    if unknown:
        raise CumlxError('E_INPUT_INVALID', f'the cases name kernels with no source in {root}: {unknown}',
                         details={'available': sorted(available)})
    sources: dict[str, str] = {}
    kernels_info: dict[str, Any] = {}
    for name in sorted({c.kernel for c, _ in cases}):
        text = available[name].read_text()
        if re.search(r'\bmain\s*\(', mask_source(text)):
            raise CumlxError('E_INPUT_INVALID', f'{available[name]} has its own main(); capture needs kernel code only',
                             hint=f'separate the host code first: cumlx kernel port {available[name]} --preprocess-only -o {name}.pre.cu')
        sources[name] = text
        kernels_info[name] = {'source_file': available[name].name, 'source_sha256': sha256_text(text),
                              'header_file': header_path.name if header_text else None,
                              'header_sha256': sha256_text(header_text) if header_text else None,
                              'form': 'complete' if is_complete_cuda(text) else 'body',
                              'features': scan_features(header_text + '\n' + text)}
    not_covered = sorted(set(available) - set(sources))
    return {'cases': cases, 'sources': sources, 'header': header_text, 'kernels': kernels_info, 'not_covered': not_covered}


def capture_kernels(plan: dict[str, Any], out_dir: Path, *, arch: str | None, extra_flags: list[str], repeats: int,
                    timeout: float, jobs: int, require_idle: bool,
                    log: Callable[[str], None]) -> dict[str, Any]:
    """Compile and run each case. Returns the manifest and the cases that failed. Writes the golden only for what ran."""
    if repeats < 2:
        raise CumlxError('E_USAGE', '--repeats must be 2 or more: one run cannot show that a kernel repeats')
    env = query_environment()
    if require_idle:
        busy = busy_processes()
        if busy:
            raise CumlxError('E_KERNEL_GPU_BUSY', f'the GPU has compute processes: {", ".join(busy)}',
                             hint='wait until it is idle, or leave out --require-idle-gpu on a machine that is yours alone',
                             details={'pids': busy})
    flags = [f'-arch={arch or "native"}', *extra_flags]
    failures: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix='cumlx-kernel-capture-') as tmp:
        work = Path(tmp)
        log(f'probing floating-point behaviour with {flags}')
        probe = _probe(work / 'probe', flags, timeout)
        items = list(plan['cases'])

        def build(item: tuple[int, tuple[KernelCase, dict[str, np.ndarray]]]) -> tuple[int, int, str]:
            index, (case, _) = item
            case_dir = work / f'c{index:04d}'
            case_dir.mkdir()
            source = case_dir / 'kernel.cu'
            source.write_text(generate_harness(case, plan['sources'][case.kernel], plan['header']))
            code, text = _compile(source, case_dir / 'kernel.bin', flags, COMPILE_TIMEOUT_SECONDS)
            return index, code, text

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            built = {i: (code, text) for i, code, text in pool.map(build, enumerate(items))}
        device: dict[str, Any] = dict(probe['device'])
        for index, (case, arrays) in enumerate(items):
            code, text = built[index]
            if code != 0:
                failures.append({'case': case.id, 'kernel': case.kernel, 'stage': 'compile', 'message': text[-1500:]})
                continue
            if require_idle and busy_processes():
                raise CumlxError('E_KERNEL_GPU_BUSY', 'another compute process started during the capture',
                                 hint='the golden was not written; wait and run again')
            log(f'running {case.id}')
            try:
                result = _execute_case(case, arrays, work / f'c{index:04d}' / 'kernel.bin', work / f'c{index:04d}', repeats, timeout)
            except (RuntimeError, OSError, ValueError) as exc:
                failures.append({'case': case.id, 'kernel': case.kernel, 'stage': 'run', 'message': str(exc)})
                continue
            results.append({'case': case, 'inputs': arrays, 'outputs': result['outputs'],
                            'repeat_identical': result['repeat_identical']})
    source = {'framework': 'cuda', 'gpu': device.get('gpu') or env['gpu'], 'driver': env['driver'], 'arch': device.get('arch'),
              'compiler': {'name': 'nvcc', 'version': env['nvcc_version'], 'flags': flags},
              'cuda_runtime': device.get('cuda_runtime'), 'cuda_driver_api': device.get('cuda_driver_api'),
              'host': {'machine': env['host_machine'], 'system': env['host_system']},
              'repeats': repeats, 'idle_during_capture': True if require_idle else None}
    facts = read_probe(probe['outputs']['y'])
    floating_point = {
        'ftz': {'flag': _flag_value(flags, 'ftz', 'false'), 'load_flushes': facts['load_flushes'],
                'input_flushes': facts['input_flushes'], 'output_flushes': facts['output_flushes']},
        'fmad': {'flag': _flag_value(flags, 'fmad', 'true'), 'contracts': facts['contracts']},
        'fp64': {'available': True}, 'warp_size': 32, 'probe_outputs': [float(v) for v in probe['outputs']['y']]}
    kernels = {k: v for k, v in plan['kernels'].items() if any(r['case'].kernel == k for r in results)}
    manifest = None
    if results:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = write_golden(out_dir, kind='cuda', source=source, kernels=kernels, floating_point=floating_point, cases=results)
    return {'manifest': manifest, 'failures': failures, 'source': source, 'floating_point': floating_point,
            'recorded': len(results), 'kernels_recorded': sorted(kernels)}
