#!/bin/sh
# The Mac side of validation/m6-f3-limits.json, inside spikes/m6_same_mac.sh, which gives it an empty home folder, a public checkout and the limits.
# It installs metal-woof in its own environment, as spikes/m3_kernels.sh does. A failed step does not stop the next one. spikes/m6_f3_judge.py gives the verdict.
#
#   sh spikes/m6_f3_work.sh OUT
set -u
OUT=${1:?the result folder}
WORK=$HOME/f3
mkdir -p "$OUT" "$WORK"
say() { echo "[$(date +%H:%M:%S)] f3: $1"; }

say "installing metal-woof 1.1.5.2 in its own environment"
uv venv -q --python 3.12 "$WORK/venv" >"$OUT/install.txt" 2>&1
# woof builds its Metal launcher with cmake, which this run's PATH lacks, so the environment has its own.
uv pip install -q --python "$WORK/venv/bin/python" -e . metal-woof==1.1.5.2 'libclang>=18.1' 'mlx>=0.29' 'nanobind>=3.0,<3.2' cmake >>"$OUT/install.txt" 2>&1
echo "$?" >"$OUT/install.exit"
[ "$(cat "$OUT/install.exit")" = 0 ] || exit 0
uv pip freeze --python "$WORK/venv/bin/python" >"$OUT/freeze.txt" 2>&1
PY=$WORK/venv/bin/python
PATH=$WORK/venv/bin:$PATH
export PATH
say "fetch sam2's file at its pinned commit"
"$PY" spikes/m6_f3_app.py fetch --out "$WORK/src" >"$OUT/fetch.json" 2>"$OUT/fetch.txt"
echo "$?" >"$OUT/fetch.exit"
say "translate with cumlx kernel port and woof"
mkdir -p "$WORK/metal"
"$WORK/venv/bin/cumlx" kernel port "$WORK/src/connected_components.cu" -o "$WORK/metal/connected_components.metal" --translator woof --format json >"$OUT/translate.json" 2>"$OUT/translate.txt"
echo "$?" >"$OUT/translate.exit"
cp "$WORK/metal/connected_components.metal" "$WORK/metal/connected_components.port.json" "$OUT/" 2>/dev/null
say "the app on the Mac"
"$PY" spikes/m6_f3_app.py mac --metal "$WORK/metal/connected_components.metal" --output "$OUT/mac.npz" >"$OUT/mac.txt" 2>&1
echo "$?" >"$OUT/mac.exit"
for fault in skip_compression half_grid swapped_wh; do
    say "control: $fault"
    "$PY" spikes/m6_f3_app.py mac --metal "$WORK/metal/connected_components.metal" --output "$OUT/fault_$fault.npz" --fault "$fault" >"$OUT/fault_$fault.txt" 2>&1
    echo "$?" >"$OUT/fault_$fault.exit"
done
say "the serial reference"
"$PY" spikes/m6_f3_app.py reference --output "$OUT/reference.npz" >"$OUT/reference.txt" 2>&1
echo "$?" >"$OUT/reference.exit"
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/f3_mac_done.txt"
