#!/bin/bash
# The table of __fdividef(127, absmax) for every fp16 absmax, on the V100, under spikes/m6_cuda_v100.sh. It installs nothing.
#   bash m6_fdiv.sh REPO OUT WORK
REPO=$1 OUT=$2 WORK=$3
mkdir -p "$OUT" "$WORK"
echo "fdiv: build and run" > "$OUT/stage.txt"
{ nvcc --version; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv; } > "$OUT/machine.txt" 2>&1
nvcc -O3 -arch=sm_70 "$REPO/spikes/remote/fdividef_127.cu" -o "$WORK/fdividef_127" > "$OUT/build.txt" 2>&1
echo $? > "$OUT/build.exit"
"$WORK/fdividef_127" "$OUT/fdividef_127_sm70.bin" > "$OUT/run.txt" 2>&1
echo $? > "$OUT/run.exit"
sha256sum "$REPO/spikes/remote/fdividef_127.cu" "$OUT/fdividef_127_sm70.bin" > "$OUT/sha256.txt" 2>&1
cp "$OUT/../cumlx_commit.txt" "$OUT/cumlx_commit.txt" 2>/dev/null
echo "fdiv: done" > "$OUT/stage.txt"
