"""Does a Quadrants kernel run on Metal, and does it agree with the CPU? Gate: rel_l2 under 1e-5."""
import json
import time

import numpy as np
import quadrants as qd


def run(arch, name, n=4_000_000):
    qd.init(arch=arch, default_fp=qd.f32)
    a = qd.field(qd.f32, shape=n)
    b = qd.field(qd.f32, shape=n)

    @qd.kernel
    def fill():
        for i in range(n):
            a[i] = qd.sin(i * 1e-3)

    @qd.kernel
    def work():
        for i in range(n):
            b[i] = a[i] * a[i] + qd.sqrt(qd.abs(a[i]))

    fill()
    work()
    start = time.perf_counter()
    for _ in range(20):
        work()
    out = b.to_numpy()
    print(json.dumps({'probe': 'quadrants', 'arch': name, 'ms_per_iteration': (time.perf_counter() - start) / 20 * 1e3}))
    qd.reset()
    return out


cpu = run(qd.cpu, 'cpu')
metal = run(qd.metal, 'metal')
rel = float(np.linalg.norm(metal.astype(np.float64) - cpu) / np.linalg.norm(cpu))
print(json.dumps({'probe': 'quadrants', 'gate': 'rel_l2 under 1e-5', 'value': rel, 'pass': rel < 1e-5}))
