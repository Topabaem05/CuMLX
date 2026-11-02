"""Does Mitsuba render on the Metal backend, and does it agree with a CPU reference?

scalar_rgb is the reference because llvm_ad_rgb needs a libLLVM that a plain Mac does not have.
Gate: mean difference under 5%. The PSNR is measured and reported, with no gate yet.
"""
import json
import time

import mitsuba as mi
import numpy as np

print(json.dumps({'probe': 'mitsuba', 'version': mi.__version__, 'variants': mi.variants()}))
images = {}
for variant, spp in (('metal_ad_rgb', 1024), ('scalar_rgb', 1024)):
    try:
        mi.set_variant(variant)
        scene = mi.load_dict(mi.cornell_box())
        np.array(mi.render(scene, spp=16))  # the first render compiles kernels
        start = time.perf_counter()
        image = np.array(mi.render(scene, spp=spp, seed=1))
        seconds = time.perf_counter() - start
        images[variant] = image
        print(json.dumps({'probe': 'mitsuba', 'variant': variant, 'shape': image.shape, 'spp': spp, 'mean': float(image.mean()),
                          'seconds': seconds, 'mega_paths_per_s': image.shape[0] * image.shape[1] * spp / seconds / 1e6}))
    except Exception as exc:  # noqa: BLE001 - a failure is the result
        print(json.dumps({'probe': 'mitsuba', 'variant': variant, 'error': f'{type(exc).__name__}: {exc}'[:300]}))
if len(images) == 2:
    metal, cpu = images['metal_ad_rgb'], images['scalar_rgb']
    difference = abs(float(metal.mean()) - float(cpu.mean())) / float(cpu.mean())
    clipped = np.clip(metal, 0, 1) - np.clip(cpu, 0, 1)
    psnr = -10 * np.log10(float(np.mean(clipped**2)))
    print(json.dumps({'probe': 'mitsuba', 'gate': 'mean difference under 5%', 'value': difference, 'psnr_db': psnr, 'pass': difference < 0.05}))
