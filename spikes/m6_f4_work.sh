#!/bin/sh
# F4 inside spikes/m6_same_mac.sh, with the checkout's .venv: training, the MLX policy and the physics comparison in one run, under validation/m6-f4-limits.json.
#   sh spikes/m6_f4_work.sh OUT
set -u
OUT=${1:?the result folder}
mkdir -p "$OUT"
PY=$PWD/.venv/bin/python
say() { echo "[$(date +%H:%M:%S)] f4: $1"; }
say "train"
"$PY" spikes/m4_sim.py train --task go2_flat --engine genesis --envs 4096 --iterations 300 --seed 1 --log-dir "$HOME/f4" >"$OUT/train.txt" 2>&1
echo "$?" >"$OUT/train.exit"
CK=$HOME/f4/model_299.pt
if [ -f "$CK" ]; then
    for run in fwd:0.8 back:-0.8 stop:0; do
        name=${run%%:*}
        say "evaluate $name"
        "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --forward "${run#*:}" --output "$OUT/eval_$name.json" >"$OUT/eval_$name.txt" 2>&1
    done
fi
say "evaluate the no-action control"
"$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --forward 0.8 --output "$OUT/eval_zero.json" >"$OUT/eval_zero.txt" 2>&1
mkdir -p "$OUT/compare"
for seed in 101 102 103; do
    say "trajectories, seed $seed"
    "$PY" spikes/m4_sim.py record --task go2_flat --engine mujoco --physics cpu --seed $seed --out "$OUT/compare/mujoco_$seed.npz" >"$OUT/compare/mujoco_$seed.txt" 2>&1
    "$PY" spikes/m4_sim.py record --task go2_flat --engine mujoco --physics cpu --seed $seed --substep-scale 4 --out "$OUT/compare/mujoco_fine_$seed.npz" >"$OUT/compare/mujoco_fine_$seed.txt" 2>&1
    "$PY" spikes/m4_sim.py record --task go2_flat --engine genesis --physics metal --seed $seed --out "$OUT/compare/genesis_$seed.npz" >"$OUT/compare/genesis_$seed.txt" 2>&1
    "$PY" spikes/m4_sim.py record --task go2_flat --engine genesis --physics metal --seed $seed --kp-scale 0.5 --out "$OUT/compare/fault_kp_$seed.npz" >"$OUT/compare/fault_kp_$seed.txt" 2>&1
done
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/f4_done.txt"
