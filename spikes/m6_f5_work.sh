#!/bin/sh
# The Mac side of validation/m6-f5-cuda-limits.json, inside spikes/m6_same_mac.sh, with the checkout's .venv (Mitsuba 3.9.1 from the render extra).
#   sh spikes/m6_f5_work.sh OUT
set -u
OUT=${1:?the result folder}
mkdir -p "$OUT"
PY=$PWD/.venv/bin/python
"$PY" -c 'import mitsuba; print(mitsuba.__version__)' >"$OUT/mitsuba.txt" 2>&1
echo "$?" >"$OUT/install.exit"
"$PY" spikes/m6_f5_sensors.py --variant metal_ad_rgb --seeds 11,12,13 --out "$OUT/sensors" >"$OUT/sensors.txt" 2>&1
echo "$?" >"$OUT/sensors.exit"
for fault in box_5cm light_2pct; do
    "$PY" spikes/m6_f5_sensors.py --variant metal_ad_rgb --seeds 11,12,13 --out "$OUT/fault_$fault" --fault "$fault" >"$OUT/fault_$fault.txt" 2>&1
    echo "$?" >"$OUT/fault_$fault.exit"
done
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/f5_mac_done.txt"
