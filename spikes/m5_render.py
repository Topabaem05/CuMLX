#!/usr/bin/env python3
"""M5: render the same scenes on a device, compare two folders of images, and measure the sensor frame rate.

    python spikes/m5_render.py render  --variant metal_ad_rgb --seeds 11,12,13 --out DIR [--fault light_2pct|direct_only]
    python spikes/m5_render.py compare GOLDEN_DIR CANDIDATE_DIR        # exit 0 grade B, 1 diverged, 5 inconclusive or missing
    python spikes/m5_render.py rate    --variant metal_ad_rgb --out FILE.json

`render` records a failure of the variant in summary.json as its result, with exit 1. The limits are in
src/cumlx/render/compare.py and validation/m5-render-limits.json. They were committed before the first render.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx.render.compare import compare_images  # noqa: E402
from cumlx.render.sensors import (  # noqa: E402
    Camera,
    Sensors,
    box_scene,
    load_mitsuba,
    process_gpu_seconds,
    system_used_gb,
)

SCENES = ('cornell_box', 'cornell_spheres')
SPP = 4096  # 1024 was too noisy for the floor. See validation/m5-render-limits.json.
PASS_SPP = 64  # All samples in passes, so that one wavefront stays near 4 million paths.
RATE_SPP = (1, 4, 16, 64)
MAX_SYSTEM_USED_GB = 43.0


def stage(name: str, stall_seconds: float) -> None:
    """Say what starts now, and end the process with a traceback if it is still in this stage after `stall_seconds`.

    A run on the V100 sat at 100 percent CPU for 30 minutes with an empty log, so the stuck stage could not be named.
    faulthandler's own thread needs no GIL, so a stall inside native code still ends. The exit code is 1, and no summary is written.
    """
    print(json.dumps({'stage': name}), flush=True)
    faulthandler.dump_traceback_later(stall_seconds, exit=True)


def check_memory() -> None:
    used = system_used_gb()
    if used is not None and used > MAX_SYSTEM_USED_GB:
        raise SystemExit(f'stopped: the system uses {used:.1f} GB, over the limit of {MAX_SYSTEM_USED_GB} GB')


def nvidia_compute_apps() -> str | None:
    try:
        return subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                              capture_output=True, text=True, check=False, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def scene_dict(mi: Any, name: str, fault: str | None = None) -> dict[str, Any]:
    scene = mi.cornell_box()
    if name == 'cornell_spheres':
        del scene['small-box'], scene['large-box']
        scene['metal'] = {'type': 'sphere', 'center': [-0.45, -0.3, -0.6], 'radius': 0.4, 'bsdf': {'type': 'roughconductor', 'alpha': 0.05}}
        scene['glass'] = {'type': 'sphere', 'center': [0.45, 0.2, -0.6], 'radius': 0.4, 'bsdf': {'type': 'dielectric'}}
    elif name != 'cornell_box':
        raise ValueError(name)
    if fault == 'light_2pct':
        radiance = scene['light']['emitter']['radiance']
        radiance['value'] = [v * 1.02 for v in radiance['value']]
    elif fault == 'direct_only':
        scene['integrator']['max_depth'] = 2
    elif fault is not None:
        raise ValueError(fault)
    return scene


def render_seed(mi: Any, scene: Any, seed: int) -> np.ndarray:
    total = None
    for index in range(SPP // PASS_SPP):
        image = np.array(mi.render(scene, spp=PASS_SPP, seed=seed * 1000 + index), dtype=np.float64)
        total = image if total is None else total + image
        check_memory()
    assert total is not None
    return (total * PASS_SPP / SPP).astype(np.float32)


def run_render(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {'variant': args.variant, 'spp': SPP, 'pass_spp': PASS_SPP, 'fault': args.fault, 'machine': platform.machine(), 'system': platform.system(), 'renders': []}
    status = 0
    try:
        stage('load mitsuba', args.stall_seconds)
        mi = load_mitsuba(args.variant)
        import drjit as dr
        summary.update(mitsuba=mi.__version__, drjit=dr.__version__, numpy=np.__version__)
        for name in SCENES:
            stage(f'load scene {name}', args.stall_seconds)
            scene = mi.load_dict(scene_dict(mi, name, args.fault))
            stage(f'first render {name}', args.stall_seconds)
            begin = time.perf_counter()
            np.array(mi.render(scene, spp=4, seed=0))  # The first render compiles the kernels.
            compile_seconds = time.perf_counter() - begin
            for seed in (int(s) for s in args.seeds.split(',')):
                stage(f'render {name} seed {seed}', args.stall_seconds)
                begin = time.perf_counter()
                image = render_seed(mi, scene, seed)
                seconds = time.perf_counter() - begin
                np.save(out / f'{name}_{seed}.npy', image)
                summary['renders'].append({'scene': name, 'seed': seed, 'seconds': round(seconds, 3), 'first_render_seconds': round(compile_seconds, 3),
                                           'mean': image.mean(axis=(0, 1)).tolist(), 'shape': list(image.shape),
                                           'mega_paths_per_s': round(image.shape[0] * image.shape[1] * SPP / seconds / 1e6, 2)})
                print(json.dumps(summary['renders'][-1]), flush=True)
    except Exception as exc:  # noqa: BLE001 - a failure of the variant is the result, and it must reach summary.json.
        summary['error'] = f'{type(exc).__name__}: {exc}'[:2000]
        status = 1
    faulthandler.cancel_dump_traceback_later()
    summary['nvidia_compute_apps'] = nvidia_compute_apps()
    summary['process_gpu_seconds'] = process_gpu_seconds(os.getpid())
    summary['status'] = 'failed' if status else 'rendered'
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({key: summary[key] for key in ('status', 'variant', 'process_gpu_seconds', 'nvidia_compute_apps')}), flush=True)
    if status:
        print(summary['error'], file=sys.stderr)
    return status


def run_compare(args: argparse.Namespace) -> int:
    scenes = {}
    result = {'golden': args.golden, 'candidate': args.candidate}
    for side in ('golden', 'candidate'):
        doc = json.loads((Path(getattr(args, side)) / 'summary.json').read_text())
        result[f'{side}_summary'] = {k: doc.get(k) for k in ('variant', 'mitsuba', 'fault', 'status', 'machine', 'process_gpu_seconds', 'nvidia_compute_apps', 'error')}
    exit_code = 0
    for name in SCENES:
        sides = {}
        for side in ('golden', 'candidate'):
            files = sorted(Path(getattr(args, side)).glob(f'{name}_*.npy'))
            sides[side] = {int(f.stem.rsplit('_', 1)[1]): np.load(f) for f in files}
        if not sides['golden'] or not sides['candidate']:
            scenes[name] = {'verdict': 'missing', 'grade': None, 'images': {k: len(v) for k, v in sides.items()}}
            exit_code = max(exit_code, 5)
            continue
        scenes[name] = compare_images(sides['golden'], sides['candidate'])
        exit_code = max(exit_code, {'ok': 0, 'diverged': 1, 'inconclusive': 5}[scenes[name]['verdict']])
    result['scenes'] = scenes
    print(json.dumps(result, indent=1))
    return exit_code


def rate_scenes(mi: Any) -> dict[str, tuple[dict[str, Any], str | None]]:
    camera = Camera(origin=(3.0, -3.0, 2.0), target=(0.0, 0.0, 0.3))
    return {'cornell_box': (scene_dict(mi, 'cornell_box'), None), 'box_on_floor_moving': (box_scene(mi, camera, 0.25), 'box')}


def run_rate(args: argparse.Namespace) -> int:
    mi = load_mitsuba(args.variant)
    import drjit as dr
    record: dict[str, Any] = {'variant': args.variant, 'mitsuba': mi.__version__, 'drjit': dr.__version__, 'resolution': [256, 256], 'repeats': args.repeats, 'frames_per_repeat': args.frames,
                              'note': 'Each frame is a NumPy copy of the finished image, so the time covers the GPU work. The first frame of a repeat is left out and shown apart, because it can compile kernels.', 'runs': []}
    for name, (scene_data, moving) in rate_scenes(mi).items():
        sensors = Sensors(scene_data, args.variant)
        for spp in RATE_SPP:
            for repeat in range(args.repeats):
                check_memory()
                used = system_used_gb()
                begin = time.perf_counter()
                sensors.rgb(spp=spp, seed=repeat * 100000)
                first = time.perf_counter() - begin
                frames, begin = 0, time.perf_counter()
                while frames < args.frames and time.perf_counter() - begin < args.max_seconds:
                    if moving:
                        sensors.set_pose(moving, (0.002 * frames, 0.0, 0.5 + 0.001 * frames))
                    sensors.rgb(spp=spp, seed=repeat * 100000 + frames + 1)
                    frames += 1
                seconds = time.perf_counter() - begin
                record['runs'].append({'scene': name, 'pose_update_each_frame': bool(moving), 'spp': spp, 'repeat': repeat, 'frames': frames, 'seconds': round(seconds, 4),
                                       'frames_per_second': round(frames / seconds, 2), 'first_frame_seconds': round(first, 4), 'system_used_gb_before': used})
                print(json.dumps(record['runs'][-1]), flush=True)
    record['process_gpu_seconds'] = process_gpu_seconds(os.getpid())
    record['uptime'] = subprocess.run(['uptime'], capture_output=True, text=True, check=False).stdout.strip()
    summary = {}
    for run in record['runs']:
        summary.setdefault(f"{run['scene']} spp={run['spp']}", []).append(run['frames_per_second'])
    record['frames_per_second_range'] = {key: {'min': min(values), 'median': sorted(values)[len(values) // 2], 'max': max(values)} for key, values in summary.items()}
    Path(args.out).write_text(json.dumps(record, indent=1))
    print(json.dumps(record['frames_per_second_range'], indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    render = commands.add_parser('render')
    render.add_argument('--variant', required=True)
    render.add_argument('--seeds', required=True)
    render.add_argument('--stall-seconds', type=float, default=900.0, help='end the run with a traceback if one stage takes longer')
    render.add_argument('--out', required=True)
    render.add_argument('--fault', choices=['light_2pct', 'direct_only'])
    compare = commands.add_parser('compare')
    compare.add_argument('golden')
    compare.add_argument('candidate')
    rate = commands.add_parser('rate')
    rate.add_argument('--variant', default='metal_ad_rgb')
    rate.add_argument('--out', required=True)
    rate.add_argument('--repeats', type=int, default=5)
    rate.add_argument('--frames', type=int, default=200)
    rate.add_argument('--max-seconds', type=float, default=10.0)
    args = parser.parse_args()
    return {'render': run_render, 'compare': run_compare, 'rate': run_rate}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
