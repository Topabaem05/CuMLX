#!/bin/sh
# The gsplat adapter work inside spikes/m6_same_mac.sh, which gives it an empty home folder, a public checkout with .venv, and the limits. A failed part does not stop the next one.
#
#   sh spikes/m6_same_mac.sh start COMMIT f2adapter
set -u
OUT=${1:?the result folder}
METALSPLAT=84e8c3a433f71c107e9387d3f72a765436c11f6e
ADAPTER=cumlx.compat.gsplat_metalsplat
WORK=$HOME/f2adapter
mkdir -p "$WORK"
say() { echo "[$(date +%H:%M:%S)] $1"; }

say "installing metalsplat"
uv venv -q --python 3.12 "$WORK/venv" >"$OUT/install.txt" 2>&1
uv pip install -q --python "$WORK/venv/bin/python" -e . "metalsplat @ git+https://github.com/tchauffi/metalsplat@$METALSPLAT" torch==2.15.0 >>"$OUT/install.txt" 2>&1
echo "$?" >"$OUT/install.exit"
if [ "$(cat "$OUT/install.exit")" = 0 ]; then
    uv pip freeze --python "$WORK/venv/bin/python" >"$OUT/adapter_freeze.txt" 2>&1
    say "the adapter against the hand render, wrong conventions, gradients and refusals"
    "$WORK/venv/bin/python" spikes/m6_f2_conformance.py --output "$OUT/conformance.json" >"$OUT/conformance.txt" 2>&1
    echo "$?" >"$OUT/conformance.exit"
    say "the original program, not changed"
    "$WORK/venv/bin/cumlx" run spikes/m6_f2_gsplat_program.py --substitute "gsplat=$ADAPTER" --report "$OUT/program.run-report.json" --format json -- --output "$OUT/program.json" >"$OUT/program.txt" 2>&1
    echo "$?" >"$OUT/program.exit"
    for mode in right frozen wrong_cameras; do
        say "training through gsplat.rasterization: $mode"
        "$WORK/venv/bin/cumlx" run spikes/m6_f2_gsplat_train.py --substitute "gsplat=$ADAPTER" --report "$OUT/train_$mode.run-report.json" --format json -- --mode "$mode" --output "$OUT/train_$mode.json" >"$OUT/train_$mode.txt" 2>&1
        echo "$?" >"$OUT/train_$mode.exit"
    done
fi

.venv/bin/python spikes/m6_f2_adapter_judge.py "$OUT"
code=$?
case "$code" in
0) echo pass ;;
1) echo fail ;;
*) echo "not_run_exit_$code" ;;
esac >"$OUT/f2adapter_verdict.txt"
