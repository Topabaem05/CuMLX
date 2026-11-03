#!/usr/bin/env python3
"""Write validation/m5-render.json from the raw results in ~/cumlx-m5. Standard library only.

    python spikes/m5_record.py ~/cumlx-m5 --date 2026-11-03

Run `sh spikes/m5_render_v100.sh fetch` and `python spikes/m5_render.py compare ~/cumlx-m5/cuda ~/cumlx-m5/metal > ~/cumlx-m5/cmp_cuda_metal.json`
first to add the CUDA comparison. Without those files the record says that the gate is open.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


CUDA_ATTEMPT = {'date': '2026-11-04', 'version': 'Mitsuba 3.9.1, Dr.Jit 1.5.0', 'result': 'failed before any render', 'first_run': 'Started 00:25 KST by the user. 70 minutes at full CPU on one thread, 308 MiB on the GPU, 0 percent GPU use, empty log, no OptiX library loaded. The user stopped it. The cause of this stall was not found.', 'second_run': "With stage lines. In 'load mitsuba': 'compute capability of device too low (need >= 7.5), skipping' and 'the CUDA backend is inactive because no compatible CUDA devices were found'. The V100 is compute capability 7.0.", 'next': 'validation/m5-render-limits.json, protocol_change_2026_11_04: Mitsuba 3.8.0 on the V100, with a control of the version difference. It has not run.'}

def load(path: Path) -> Any:
    return json.loads(path.read_text())


def short_compare(doc: dict[str, Any]) -> dict[str, Any]:
    keep = ('verdict', 'grade', 'failures', 'worst_psnr_across_db', 'worst_psnr_within_db', 'mse_ratio', 'channel_mean_difference', 'channel_mean_allowed')
    return {'golden': doc['golden_summary'].get('variant'), 'golden_fault': doc['golden_summary'].get('fault'), 'candidate': doc['candidate_summary'].get('variant'), 'candidate_fault': doc['candidate_summary'].get('fault'),
            'scenes': {name: {key: scene.get(key) for key in keep} for name, scene in doc['scenes'].items()}}


def compares(folder: Path) -> dict[str, Any]:
    return {file.stem.removeprefix('cmp_'): short_compare(load(file)) for file in sorted(folder.glob('cmp_*.json')) if 'cuda' not in file.stem}


def scenario_summary(result: dict[str, Any]) -> dict[str, Any]:
    frames = result['frames']
    errors = [f['floor_mean_rel'] for f in frames if f['floor_mean_rel'] is not None]
    medians = [f['floor_median_rel'] for f in frames if f['floor_median_rel'] is not None]
    first_failure = next((f for f in frames if not f['ok']), None)
    return {**{key: result[key] for key in ('frames_passed', 'frames_total', 'worst_pixel_error', 'final_height', 'height_range', 'centroid_travel_px', 'seconds_per_frame', 'build_seconds')},
            'worst_floor_mean_rel': max(errors, key=abs) if errors else None, 'worst_floor_median_rel': max(medians) if medians else None,
            'first_failed_frame': None if first_failure is None else {'step': first_failure['step'], 'failures': first_failure['failures']},
            'height_every_60_steps': [round(f['position'][2], 4) for f in frames[::12]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw')
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    raw = Path(args.raw).expanduser()

    trace = (raw / 'rt_trace.log').read_text().splitlines()
    pattern = re.compile(r'raytracing::|instance_acceleration_structure|jit_metal_configure_scene')
    rate = load(raw / 'rate.json')
    drop = load(raw / 'drop' / 'result.json')
    first = load(raw / 'first-1024' / 'drop_first_run' / 'result.json')
    metal = load(raw / 'metal' / 'summary.json')
    cuda_summary = raw / 'cuda' / 'summary.json'
    cuda_compare = raw / 'cmp_cuda_metal.json'

    if cuda_compare.exists():
        cuda: dict[str, Any] = {'status': 'compared', **short_compare(load(cuda_compare))}
    elif cuda_summary.exists():
        cuda = {'status': 'rendered_or_failed_not_compared', 'summary': load(cuda_summary)}
    else:
        cuda = {'status': 'not_run', 'why': 'No image from the V100 yet. See cuda_attempt_with_3_9_1.'}

    record = {
        'schema': 'cumlx.m5-render.v1',
        'date': args.date,
        'machine': 'Apple M5 Pro, 48 GB, macOS 27',
        'versions': {key: metal.get(key) for key in ('mitsuba', 'drjit', 'numpy')},
        'limits': 'validation/m5-render-limits.json',
        'gates': {
            'metal_vs_v100_cuda_grade_b': 'open' if cuda['status'] != 'compared' else 'see cuda_comparison',
            'frame_rate_256': 'measured, see frame_rate',
            'physics_and_camera_example': 'ran, see physics_and_camera',
            'rtx_differences_document': 'docs/RENDER-SENSORS.md',
        },
        'cuda_comparison': cuda,
        'cuda_attempt_with_3_9_1': CUDA_ATTEMPT,
        'metal_renders': metal,
        'local_checks_of_the_comparison': {
            'shows': 'The rule on real images, without CUDA: Metal against the CPU reference, and two injected faults. This does not meet the CUDA gate.',
            'spp_4096': compares(raw),
            'spp_1024_first_protocol': compares(raw / 'first-1024'),
        },
        'frame_rate': {key: rate[key] for key in ('variant', 'resolution', 'repeats', 'frames_per_repeat', 'note', 'uptime', 'process_gpu_seconds', 'frames_per_second_range', 'runs')},
        'frame_rate_note': 'Other jobs ran on this Mac during the measurement, so the load average is in `uptime`. Noisy images: spp 1 and 4 are not usable without a denoiser. box_on_floor_moving updates the mesh and rebuilds the acceleration structure each frame.',
        'physics_and_camera': {
            'example': 'examples/physics-camera/drop_box.py',
            'physics': 'Genesis 1.4.5 on Metal, a cube of side 0.5 m. In physics_alone, backend 4 is gs.metal.',
            'camera': 'Mitsuba metal_ad_rgb, 256 x 256, RGB at 16 spp, depth and segmentation at 1 spp, one frame per 5 physics steps',
            'ok': drop['ok'], 'checks': drop['checks'], 'final_speed': drop['final_speed'],
            'gpu_seconds_of_the_process': drop['gpu_seconds_of_this_process'], 'physics_alone': drop['physics_alone'],
            'scenarios': {name: scenario_summary(result) for name, result in drop['results'].items()},
            'first_run_failed': {
                'what': 'All frames of all scenarios failed the floor depth limit, limit 1e-3.',
                'worst_floor_mean_rel': scenario_summary(first['results']['drop'])['worst_floor_mean_rel'],
                'cause': "Mitsuba starts camera rays at near_clip, 0.01 by default, and its depth AOV counts from there. The error was about 1 cm at every distance. Sensors now adds that distance. The limits did not change. See worst_floor_mean_rel here and in scenarios.drop.",
            },
        },
        'hardware_ray_tracing': {
            'measured': {
                'kernel_lines': sorted({line.strip()[:120] for line in trace if pattern.search(line)})[:10],
                'device': 'Apple M5 Pro: supportsRaytracing true, supportsRaytracingFromRender true, MTLGPUFamily.apple9 true (Swift, Metal API)',
                'libraries': 'libdrjit-core.dylib holds raytracing::intersector and instance_acceleration_structure. libmitsuba.dylib holds MTLAccelerationStructure and newAccelerationStructure.',
            },
            'documents': 'Apple: hardware-accelerated ray tracing comes to the Mac with the M3 family, and M5 has a third-generation ray-tracing engine.',
            'not_measured': 'Metal ray tracing also runs on Apple6 GPUs, which have no ray tracing hardware. No counter that shows the dedicated circuit at work was read.',
            'conclusion': 'Mitsuba on Metal uses the Metal ray tracing API, and on this GPU that API is hardware-accelerated according to Apple. The second half is from documents.',
        },
        'not_built': ['LiDAR and other ray cast sensors', 'USD scenes', 'motion vectors', 'lens distortion', 'semantic labels', 'batched environments', 'a cumlx.sim camera sensor'],
    }
    out = ROOT / 'validation' / 'm5-render.json'
    out.write_text(json.dumps(record, indent=1) + '\n')
    print(f'{out}: {out.stat().st_size // 1024} KB')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
