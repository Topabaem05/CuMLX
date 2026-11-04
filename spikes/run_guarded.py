#!/usr/bin/env python3
"""Run one command with this Mac's memory limits: it starts only under a start limit, and it stops when memory passes a stop limit.

    python spikes/run_guarded.py --stop-gb 38 --start-gb 33 --process-gb 16 --output usage.json -- COMMAND ARGS...

Memory in use comes from cumlx.sysmem, on macOS and Linux, and the default stop limit is a fraction of this machine's RAM. It stops only the process group that it started. The exit code is the
command's, 75 when the start limit refused to start it, and 76 when a limit stopped it. No NVIDIA machine is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from cumlx import sysmem  # noqa: E402

REFUSED, STOPPED = 75, 76


def system_used_gb() -> float:
    used = sysmem.used_gb()
    if used is None:
        print('error: this machine gives no reading of memory in use, so the limits cannot hold.', file=sys.stderr)
        raise SystemExit(REFUSED)
    return used


def process_group_gb(group: int) -> float:
    text = subprocess.run(['ps', '-o', 'rss=', '-g', str(group)], capture_output=True, text=True, check=False, timeout=20).stdout
    return sum(int(v) for v in text.split()) * 1024 / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stop-gb', type=float, help='default: the limit of cumlx.sysmem for this machine')
    parser.add_argument('--start-gb', type=float, help='default: 7/8 of the stop limit')
    parser.add_argument('--process-gb', type=float, default=16.0)
    parser.add_argument('--disk-gb', type=float, default=15.0, help='stop when the free disk falls under this')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--output')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('no command')
    if args.stop_gb is None:
        args.stop_gb = sysmem.limit_gb()
    if args.stop_gb is None:
        print('error: no stop limit: this machine does not report its memory. Pass --stop-gb.', file=sys.stderr)
        return REFUSED
    if args.start_gb is None:
        args.start_gb = args.stop_gb * 7 / 8
    used = system_used_gb()
    record = {'command': command, 'start_used_gb': round(used, 2), 'peak_used_gb': round(used, 2), 'peak_process_gb': 0.0, 'stopped_for': None}

    def finish(code: int) -> int:
        record['exit_code'] = code
        if args.output:
            Path(args.output).write_text(json.dumps(record, indent=1))
        print(json.dumps({k: record[k] for k in ('exit_code', 'start_used_gb', 'peak_used_gb', 'peak_process_gb', 'stopped_for')}), file=sys.stderr, flush=True)
        return code

    if used > args.start_gb:
        record['stopped_for'] = f'memory in use is {used:.1f} GB, over the start limit {args.start_gb}'
        return finish(REFUSED)
    process = subprocess.Popen(command, start_new_session=True)
    stopped = False
    while process.poll() is None:
        time.sleep(args.interval)
        used = system_used_gb()
        mine = process_group_gb(process.pid)
        free = shutil.disk_usage(Path.home()).free / 2**30
        record['peak_used_gb'] = round(max(record['peak_used_gb'], used), 2)  # type: ignore[type-var]
        record['peak_process_gb'] = round(max(record['peak_process_gb'], mine), 2)  # type: ignore[type-var]
        reason = (f'memory in use is {used:.1f} GB, over {args.stop_gb}' if used > args.stop_gb else
                  f'the process group uses {mine:.1f} GB, over {args.process_gb}' if mine > args.process_gb else
                  f'{free:.1f} GB of disk is free, under {args.disk_gb}' if free < args.disk_gb else None)
        if reason and not stopped:
            record['stopped_for'] = reason
            stopped = True
            os.killpg(process.pid, signal.SIGTERM)
            for _ in range(20):
                time.sleep(0.5)
                if process.poll() is not None:
                    break
            else:
                os.killpg(process.pid, signal.SIGKILL)
    return finish(STOPPED if stopped else process.returncode)


if __name__ == '__main__':
    raise SystemExit(main())
