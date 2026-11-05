#!/bin/sh
# The Mac side of validation/m6-f2-cuda-limits.json, inside spikes/m6_same_mac.sh, which gives it an empty home folder, a public checkout with .venv, and the limits.
# A failed part does not stop the next one. spikes/m6_f2_cuda_judge.py compares the result with the V100's, from spikes/m6_f2_cuda_v100.sh.
#
#   sh spikes/m6_same_mac.sh start COMMIT f2cuda
set -u
OUT=${1:?the result folder}
METALSPLAT=84e8c3a433f71c107e9387d3f72a765436c11f6e
ADAPTER=cumlx.compat.gsplat_metalsplat
WORK=$HOME/f2cuda
mkdir -p "$WORK"
say() { echo "[$(date +%H:%M:%S)] $1"; }

say "installing metalsplat"
uv venv -q --python 3.12 "$WORK/venv" >"$OUT/install.txt" 2>&1
uv pip install -q --python "$WORK/venv/bin/python" -e . "metalsplat @ git+https://github.com/tchauffi/metalsplat@$METALSPLAT" torch==2.15.0 >>"$OUT/install.txt" 2>&1
echo "$?" >"$OUT/install.exit"
[ "$(cat "$OUT/install.exit")" = 0 ] || exit 0
uv pip freeze --python "$WORK/venv/bin/python" >"$OUT/mac_freeze.txt" 2>&1
CUMLX=$WORK/venv/bin/cumlx

say "renders and gradients through the adapter"
"$CUMLX" run spikes/m6_f2_cuda_record.py --substitute "gsplat=$ADAPTER" --report "$OUT/record.run-report.json" --format json -- --output "$OUT/record.npz" >"$OUT/record.txt" 2>&1
echo "$?" >"$OUT/record.exit"
for fault in quaternion_xyzw camera_to_world_view_matrix fx_fy_swapped half_pixel_shift no_eps2d antialiased_for_classic alpha_clamp_0.99 means_gradient_1.1; do
    say "control: $fault"
    CUMLX_F2_FAULT=$fault "$CUMLX" run spikes/m6_f2_cuda_record.py --substitute gsplat=m6_f2_fault_gsplat --format json -- --output "$OUT/fault_$fault.npz" --no-hand >"$OUT/fault_$fault.txt" 2>&1
    echo "$?" >"$OUT/fault_$fault.exit"
done
for seed in 0 1 2 3 4; do
    for mode in right frozen wrong_cameras; do
        say "training $mode seed $seed"
        "$CUMLX" run spikes/m6_f2_gsplat_train.py --substitute "gsplat=$ADAPTER" --format json -- --mode $mode --seed $seed --output "$OUT/train_${mode}_$seed.json" >"$OUT/train_${mode}_$seed.txt" 2>&1
        echo "$?" >"$OUT/train_${mode}_$seed.exit"
    done
    say "control: colors_gradient_zero seed $seed"
    CUMLX_F2_FAULT=colors_gradient_zero "$CUMLX" run spikes/m6_f2_gsplat_train.py --substitute gsplat=m6_f2_fault_gsplat --format json -- --mode right --seed $seed --output "$OUT/train_colors_gradient_zero_$seed.json" >"$OUT/train_colors_gradient_zero_$seed.txt" 2>&1
    echo "$?" >"$OUT/train_colors_gradient_zero_$seed.exit"
done
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/f2cuda_mac_done.txt"
echo mac_done >"$OUT/f2cuda_verdict.txt"
