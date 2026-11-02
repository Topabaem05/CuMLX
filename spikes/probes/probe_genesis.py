"""Does Genesis run parallel rigid bodies on Metal? Gate: 4096 environments, at least 3x the CPU throughput."""
import json
import time

import genesis as gs

rates = {}
for name, backend, sizes in (('metal', getattr(gs, 'metal', None), (1, 512, 4096)), ('cpu', gs.cpu, (1, 512))):
    if backend is None:
        print(json.dumps({'probe': 'genesis', 'backend': name, 'error': 'no such backend'}))
        continue
    for n in sizes:
        try:
            gs.init(backend=backend, logging_level='warning')
            scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
            scene.add_entity(gs.morphs.Plane())
            box = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0, 0, 1.0)))
            scene.build(n_envs=n if n > 1 else 0)
            start = time.perf_counter()
            for _ in range(200):
                scene.step()
            rates[(name, n)] = 200 * n / (time.perf_counter() - start)
            height = float(box.get_pos()[..., 2].mean())
            print(json.dumps({'probe': 'genesis', 'backend': name, 'envs': n, 'env_steps_per_s': rates[(name, n)], 'final_height': height}))
        except Exception as exc:  # noqa: BLE001 - a failure is the result
            print(json.dumps({'probe': 'genesis', 'backend': name, 'envs': n, 'error': f'{type(exc).__name__}: {exc}'[:300]}))
        gs.destroy()
if ('metal', 4096) in rates and ('cpu', 512) in rates:
    ratio = rates[('metal', 4096)] / rates[('cpu', 512)]
    print(json.dumps({'probe': 'genesis', 'gate': 'at least 3x', 'value': ratio, 'pass': ratio >= 3}))
