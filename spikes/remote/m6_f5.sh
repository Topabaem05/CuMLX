#!/bin/bash
# F5 on CUDA, under spikes/m6_cuda_v100.sh: the cumlx.render sensor on Mitsuba 3.8.0 cuda_ad_rgb, for validation/m6-f5-cuda-limits.json.
#   bash m6_f5.sh REPO OUT WORK
REPO=$1 OUT=$2 WORK=$3
mkdir -p "$OUT" "$WORK"
export PYTHONUNBUFFERED=1
stage() { echo "[$(date +%H:%M:%S)] f5: $1"; echo "f5: $1" > "$OUT/stage.txt"; }
stage "install Mitsuba 3.8.0 in its own environment"
rm -rf "$WORK/venv"
{ uv venv -q --python 3.12 "$WORK/venv" && uv pip install -q --python "$WORK/venv/bin/python" mitsuba==3.8.0 numpy; } > "$OUT/install.txt" 2>&1
echo $? > "$OUT/install.exit"
stage "render"
"$WORK/venv/bin/python" "$REPO/spikes/m6_f5_sensors.py" --variant cuda_ad_rgb --seeds 1,2,3 --out "$OUT/sensors" > "$OUT/sensors.txt" 2>&1
echo $? > "$OUT/sensors.exit"
cp "$OUT/../cumlx_commit.txt" "$OUT/cumlx_commit.txt" 2>/dev/null
stage done
