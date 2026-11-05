#!/bin/sh
# The F1 and F2 work inside spikes/m6_same_mac.sh, which gives it an empty home folder, a public checkout with .venv, and the limits. A failed part does not stop the next one.
#
#   sh spikes/m6_same_mac.sh start COMMIT f1f2
set -u
OUT=${1:?the result folder}
MODEL=Qwen/Qwen2.5-0.5B
REVISION=060db6499f32faf8b98477b0a26969ef7d8b9987
METALSPLAT=84e8c3a433f71c107e9387d3f72a765436c11f6e
WORK=$HOME/f1f2
mkdir -p "$WORK"
say() { echo "[$(date +%H:%M:%S)] $1"; }

say "F1 port at bf16"
.venv/bin/cumlx port "$MODEL" --revision "$REVISION" --dtype bfloat16 -o "$WORK/port" --format json >"$OUT/f1_port.json" 2>"$OUT/f1_port.err"
echo "$?" >"$OUT/f1_port.exit"
say "F1 port with 8-bit weights"
.venv/bin/cumlx port "$MODEL" --revision "$REVISION" --q-bits 8 -o "$WORK/port8" --format json >"$OUT/f1_port8.json" 2>"$OUT/f1_port8.err"
echo "$?" >"$OUT/f1_port8.exit"
say "F1 a wrong port must fail"
.venv/bin/python spikes/m6_f1_fault.py "$WORK/port/mlx" "$WORK/port-bad" >"$OUT/f1_fault.txt" 2>&1 \
    && .venv/bin/cumlx diff "$WORK/port/golden" "$WORK/port-bad" -o "$WORK/bad-report" --format json >"$OUT/f1_bad.json" 2>"$OUT/f1_bad.err"
echo "$?" >"$OUT/f1_bad.exit"

say "F1 the original program"
uv venv -q --python 3.14 "$WORK/f1-venv" >"$OUT/f1_program_install.txt" 2>&1
uv pip install -q --python "$WORK/f1-venv/bin/python" -e . mps-flash-attn==0.6.3 mps-bitsandbytes==0.7.0 transformers accelerate >>"$OUT/f1_program_install.txt" 2>&1
echo "$?" >"$OUT/f1_program_install.exit"
if [ "$(cat "$OUT/f1_program_install.exit")" = 0 ]; then
    "$WORK/f1-venv/bin/cumlx" run spikes/m6_f1_program.py --substitute flash_attn=mps_flash_attn --substitute bitsandbytes=mps_bitsandbytes --report "$OUT/f1_program.run-report.json" --format json -- --model "$MODEL" --revision "$REVISION" --output "$OUT/f1_program.json" >"$OUT/f1_program.txt" 2>&1
    echo "$?" >"$OUT/f1_program.exit"
fi

say "F2 installing metalsplat"
uv venv -q --python 3.12 "$WORK/f2-venv" >"$OUT/f2_install.txt" 2>&1
uv pip install -q --python "$WORK/f2-venv/bin/python" -e . "metalsplat @ git+https://github.com/tchauffi/metalsplat@$METALSPLAT" torch==2.15.0 >>"$OUT/f2_install.txt" 2>&1
echo "$?" >"$OUT/f2_install.exit"
if [ "$(cat "$OUT/f2_install.exit")" = 0 ]; then
    for mode in right frozen wrong_cameras; do
        say "F2 training: $mode"
        "$WORK/f2-venv/bin/python" spikes/m6_f2_train.py --mode "$mode" --output "$OUT/f2_$mode.json" >"$OUT/f2_$mode.txt" 2>&1
        echo "$?" >"$OUT/f2_$mode.exit"
    done
    say "F2 the original program"
    "$WORK/f2-venv/bin/cumlx" run spikes/m6_f2_gsplat_program.py --substitute gsplat=metalsplat --report "$OUT/f2_program.run-report.json" --format json -- --output "$OUT/f2_program.json" >"$OUT/f2_program.txt" 2>&1
    echo "$?" >"$OUT/f2_program.exit"
fi

.venv/bin/python spikes/m6_f1_f2_judge.py "$OUT"
code=$?
case "$code" in
0) echo pass ;;
1) echo fail ;;
*) echo "not_run_exit_$code" ;;
esac >"$OUT/f1f2_verdict.txt"
