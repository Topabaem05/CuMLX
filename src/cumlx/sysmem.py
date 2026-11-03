"""System memory in use, and the limit for long runs. Standard library only, so that a spike in any environment can use it.

Used memory means active + wired + compressed pages on macOS, from `vm_stat` and the page size that it reports. On Linux it
means MemTotal - MemAvailable, from /proc/meminfo. A platform without either gives None: the caller cannot guard memory there,
and it must say so. Sizes are in units of 1e9 bytes, as in the records.

The default limit is a fraction of this machine's RAM, not a fixed number. Set CUMLX_MAX_USED_GB to choose another one.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess

DEFAULT_LIMIT_FRACTION = 0.73  # A 48 GiB Mac gets 37.6, under its owner's rule of 38 GB. It leaves about a quarter free.
LIMIT_VARIABLE = 'CUMLX_MAX_USED_GB'


def parse_vm_stat(text: str) -> float | None:
    """Used memory in GB from `vm_stat` output, with the page size that it states. Output without one is not read."""
    size = re.search(r'page size of (\d+)', text)
    counts = [re.search(rf'{label}:\s+(\d+)', text) for label in ('Pages active', 'Pages wired down', 'Pages occupied by compressor')]
    if size is None or None in counts:
        return None
    return sum(int(c.group(1)) for c in counts if c) * int(size.group(1)) / 1e9


def parse_meminfo(text: str) -> float | None:
    """Used memory in GB from /proc/meminfo. A kernel without MemAvailable is not read."""
    total = re.search(r'^MemTotal:\s+(\d+) kB', text, re.M)
    available = re.search(r'^MemAvailable:\s+(\d+) kB', text, re.M)
    if total is None or available is None:
        return None
    return (int(total.group(1)) - int(available.group(1))) * 1000 / 1e9


def used_gb() -> float | None:
    """Memory in use on this machine in GB, or None if this platform or its tools cannot say."""
    system = platform.system()
    try:
        if system == 'Darwin':
            return parse_vm_stat(subprocess.run(['vm_stat'], capture_output=True, text=True, check=False, timeout=20).stdout)
        if system == 'Linux':
            with open('/proc/meminfo', encoding='ascii') as handle:
                return parse_meminfo(handle.read())
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def total_gb() -> float | None:
    """This machine's RAM in GB, or None if the platform cannot say."""
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    except (AttributeError, OSError, ValueError):
        return None


def limit_gb(default_fraction: float = DEFAULT_LIMIT_FRACTION) -> float | None:
    """The limit for a long run: CUMLX_MAX_USED_GB, or a fraction of RAM. None if RAM is unknown and nothing is set."""
    value = os.environ.get(LIMIT_VARIABLE)
    if value:
        try:
            limit = float(value)
        except ValueError:
            raise ValueError(f'{LIMIT_VARIABLE} must be a number of GB, got {value!r}') from None
        if limit <= 0:
            raise ValueError(f'{LIMIT_VARIABLE} must be above 0, got {value!r}')
        return limit
    total = total_gb()
    return None if total is None else total * default_fraction


def over_limit() -> str | None:
    """A reason to stop if memory in use is above the limit, else None. An unknown reading is not a reason."""
    used, limit = used_gb(), limit_gb()
    if used is None or limit is None or used <= limit:
        return None
    return f'system memory in use is {used:.1f} GB, above the limit of {limit:.1f} GB. Set {LIMIT_VARIABLE} to change the limit'
