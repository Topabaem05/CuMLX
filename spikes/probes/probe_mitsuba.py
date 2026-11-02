"""Does Mitsuba render on the Metal backend, and does it agree with the CPU backend?"""
import json
import time

import mitsuba as mi
import numpy as np

print(json.dumps({'probe': 'mitsuba', 'version': mi.__version__, 'variants': mi.variants()}))
means = {}
for variant in ('metal_ad_rgb', 'llvm_ad_rgb'):
    try:
        mi.set_variant(variant)
        scene = mi.load_dict(mi.cornell_box())
        np.array(mi.render(scene, spp=64))  # the first render compiles kernels
        start = time.perf_counter()
        image = np.array(mi.render(scene, spp=64, seed=1))
        means[variant] = float(image.mean())
        print(json.dumps({'probe': 'mitsuba', 'variant': variant, 'shape': image.shape, 'mean': means[variant],
                          'seconds': time.perf_counter() - start}))
    except Exception as exc:  # noqa: BLE001 - a failure is the result
        print(json.dumps({'probe': 'mitsuba', 'variant': variant, 'error': f'{type(exc).__name__}: {exc}'[:300]}))
if len(means) == 2:
    difference = abs(means['metal_ad_rgb'] - means['llvm_ad_rgb']) / means['llvm_ad_rgb']
    print(json.dumps({'probe': 'mitsuba', 'gate': 'mean difference under 5%', 'value': difference, 'pass': difference < 0.05}))
