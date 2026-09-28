#!/usr/bin/env python3
"""Physical GPU benchmark. Fresh evaluated graphs, accuracy gates, no silent fallback.

Reported warm values are synchronized Python-to-result wall times, not GPU-only
profiler timings. Device quantization is excluded: the same CPU-prepared packed
weights are used by every candidate. Host/reference allocations are explicit.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import statistics
import time
import traceback
import numpy as np
from cumlx import MLXBackend, quantize, reference, plan_linear
from cumlx.mlx_backend import environment_report, kernel_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['metal','cuda'], default='metal')
    parser.add_argument('--format', choices=['nvfp4','mxfp4'], default='nvfp4')
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--repeat', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--dtype', choices=['float16','bfloat16','float32'], default='float32')
    parser.add_argument('--output', type=Path, default=Path('benchmark.json'))
    args = parser.parse_args()
    if min(args.m,args.n,args.k,args.repeat) <= 0 or args.warmup < 0:
        parser.error('dimensions and repeat must be positive; warmup nonnegative')
    plan = plan_linear(args.m,args.n,args.k)
    report = dict(environment=environment_report(probe=True), arguments=vars(args).copy(),
                  methodology='fresh graph + mx.eval + stream synchronize, wall latency',
                  device_compilation_validated_by_this_run=False, candidates=[], plan=plan.as_dict(),
                  hidden_cpu_fallback=False, quantization_timed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write():
        args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')

    try:
        gpu = MLXBackend(args.backend, allow_unvalidated=True)
        mx = gpu.mx
        rng = np.random.default_rng(28)
        hostx = rng.normal(size=(args.m,args.k)).astype(np.float32)
        hostw = rng.normal(size=(args.n,args.k)).astype(np.float32)
        start = time.perf_counter()
        q = quantize(hostw,args.format)
        report['cpu_pack_ms'] = (time.perf_counter()-start)*1000
        report['packed_storage_bytes'] = q.storage_nbytes
        start = time.perf_counter()
        qw = gpu.prepare(q)
        with mx.stream(gpu.stream):
            x = mx.array(hostx).astype(getattr(mx,args.dtype))
        mx.eval(x)
        mx.synchronize(gpu.stream)
        report['prepare_and_input_upload_ms'] = (time.perf_counter()-start)*1000
        # Oracle uses the values actually representable in the chosen activation dtype.
        actual_hostx = np.array(x.astype(mx.float32))
        expected = reference.linear(actual_hostx,q)
        report['host_reference_materializes_weights'] = True
        report['tolerance'] = dict(atol=2e-4,rtol=3e-4)
        report['kernel_sha256'] = kernel_source('fp4_'+plan.algorithm,args.backend)[2]

        candidates = [('cumlx_auto', lambda: gpu.linear(x,qw), True, False),
                      ('cumlx_tiled',lambda: gpu.linear(x,qw,algorithm='tiled'), True, False)]
        # This candidate can reject shape/dtype/format constraints. Do not disguise it.
        candidates.append(('mlx_native_qmm',lambda:gpu.native_linear(x,qw),False,False))
        compiled_native = mx.compile(lambda value:gpu.native_linear(value,qw))
        candidates.append(('mlx_compiled_qmm',lambda:compiled_native(x),False,False))
        # Diagnostic comparator explicitly materializes an FP32 weight matrix.
        fullw = gpu.decode(qw)
        mx.eval(fullw)
        dense = mx.compile(lambda a,b:mx.matmul(a.astype(mx.float32),b.T,stream=gpu.stream))
        candidates.append(('mlx_compiled_dense_full_fp32_weight',lambda:dense(x,fullw),False,True))
        report['dense_baseline_weight_bytes'] = args.n*args.k*4
        report['peak_memory_scope_note'] = 'Absolute allocator peak includes diagnostic dense baseline, not a packed-path memory proof.'
        required_ok = True
        for name, fn, required, full_dequant in candidates:
            row = dict(name=name, full_weight_dequantization=full_dequant, status='NOT_RUN')
            report['candidates'].append(row)
            try:
                mx.synchronize(gpu.stream)
                start = time.perf_counter()
                y = fn()  # fresh graph; possible first compile
                mx.eval(y)
                mx.synchronize(gpu.stream)
                row['first_call_ms'] = (time.perf_counter()-start)*1000
                got = np.array(y)
                row['max_abs_error'] = float(np.max(np.abs(got-expected)))
                row['rmse'] = float(np.sqrt(np.mean((got.astype(np.float64)-expected)**2)))
                np.testing.assert_allclose(got,expected,atol=2e-4,rtol=3e-4)
                for _ in range(args.warmup):
                    y=fn(); mx.eval(y); mx.synchronize(gpu.stream)
                samples=[]
                for _ in range(args.repeat):
                    start=time.perf_counter()
                    y=fn(); mx.eval(y); mx.synchronize(gpu.stream)
                    samples.append((time.perf_counter()-start)*1000)
                row.update(status='ACCURACY_PASSED_AND_TIMED',warm_wall_ms=samples,
                           median_ms=statistics.median(samples),min_ms=min(samples),
                           p95_ms=float(np.percentile(samples,95)))
            except Exception as exc:
                row.update(status='FAILED' if required else 'UNAVAILABLE_OR_ACCURACY_REJECTED',
                           error=f'{type(exc).__name__}: {exc}')
                if required: required_ok=False
            write()
        report['device_compilation_validated_by_this_run'] = required_ok
        report['scope'] = 'Only these shapes/dtype and this device; does not certify the entire library.'
        if hasattr(mx,'get_peak_memory'):
            report['absolute_peak_device_bytes'] = mx.get_peak_memory()
        write()
        return 0 if required_ok else 1
    except Exception as exc:
        report['fatal_error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
        write()
        print(report['fatal_error'])
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
