#!/usr/bin/env python3
"""Deterministic CPU example: does NOT test Metal or CUDA."""
import json
import numpy as np
from cumlx import quantize, dequantize, reference, plan_linear


def main():
    rng = np.random.default_rng(28)
    x = rng.normal(size=(3, 67)).astype(np.float32)
    weight = rng.normal(size=(35, 67)).astype(np.float32)
    records = []
    for fmt in ('nvfp4', 'mxfp4'):
        q = quantize(weight, fmt)
        y = reference.linear(x, q)
        dense = x.astype(np.float64) @ weight.astype(np.float64).T
        restored = dequantize(q)
        records.append(dict(backend='CPU_REFERENCE_ONLY', format=fmt,
                            input_shape=list(x.shape), output_shape=list(y.shape),
                            storage_bytes=q.storage_nbytes,
                            original_fp16_bytes=weight.size*2,
                            weight_quantization_rmse=float(np.sqrt(np.mean((weight-restored)**2))),
                            output_error_vs_unquantized_rmse=float(np.sqrt(np.mean((y-dense)**2))),
                            quantization_error_is_not_kernel_error=True,
                            plan=plan_linear(*[3,35,67]).as_dict(), checksum=q.checksum))
    print(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
