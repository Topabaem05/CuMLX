"""Combine result files from run_all.sh into one evidence file, without local paths."""
import argparse
import json
from pathlib import Path
import platform
import re
import subprocess


def clean(text):
    return re.sub(r'/(?:private/tmp|private/var|var/folders|Users|tmp)/[^\s"\':,]*/(?=[^/\s"\':,]+)', '', text)


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True)
parser.add_argument('results', nargs='+')
args = parser.parse_args()
chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
evidence = {
    'schema': 'cumlx.spikes.v1',
    'date': '2026-11-02',
    'environment': {'chip': chip, 'macos': platform.mac_ver()[0], 'python': '3.12'},
    'versions': {'genesis-world': '1.4.5', 'torch': '2.15.0', 'mitsuba': '3.9.1', 'usd-core': '26.11', 'quadrants': '1.3.3',
                 'mujoco': '3.17.0', 'metal-woof': '1.1.5.2', 'kernelport': '0.1.0'},
    'runs': [[json.loads(clean(line)) for line in Path(path).read_text().splitlines() if line.startswith('{')] for path in args.results],
}
Path(args.output).write_text(json.dumps(evidence, indent=1) + '\n')
