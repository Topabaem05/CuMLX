#!/usr/bin/env python3
"""Run one command with this Mac's memory limits: it starts only under a start limit, and it stops when memory passes a stop limit.

    python spikes/run_guarded.py --stop-gb 38 --start-gb 33 --process-gb 16 --output usage.json -- COMMAND ARGS...

Memory in use is active + wired + compressor pages from vm_stat. It stops only the process group that it started. The exit code is the
command's, 75 when the start limit refused to start it, and 76 when a limit stopped it. No NVIDIA machine is involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

REFUSED, STOPPED = 75, 76


def system_used_gb() -> float:
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True, timeout=20).stdout
    size = re.search(r'page size of (\d+)', text)
    values = [re.search(rf'{label}:\s+(\d+)', text) for label in ('Pages active', 'Pages wired down', 'Pages occupied by compressor')]
    if size is None or not all(values):
        raise RuntimeError('vm_stat printed a form that this script cannot read')
    return sum(int(v.group(1)) for v in values if v) * int(size.group(1)) / 1e9


def process_group_gb(group: int) -> float:
    text = subprocess.run(['ps', '-o', 'rss=', '-g', str(group)], capture_output=True, text=True, check=False, timeout=20).stdout
    return sum(int(v) for v in text.split()) * 1024 / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stop-gb', type=float, default=38.0)
    parser.add_argument('--start-gb', type=float, default=33.0)
    parser.add_argument('--process-gb', type=float, default=16.0)
    parser.add_argument('--disk-gb', type=float, default=15.0, help='stop when the free disk falls under this')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--output')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('no command')
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
