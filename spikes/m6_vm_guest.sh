#!/bin/sh
# Runs inside a fresh macOS VM, started by spikes/m6_vm_reproduce.sh. It treats the VM as a new Mac, and gets CuMLX from its public repository.
#
#   sh m6_vm_guest.sh COMMIT
#
# Each part writes to ~/repro-result. spikes/m6_vm_judge.py gives the verdict, by the rules in validation/m6-vm-limits.json. A failed part does not stop the next one.
set -u
COMMIT=${1:?the commit of CuMLX to test}
# This script changes the machine it runs on, and it uses sudo with the guest's password. So it runs only inside a VM.
if [ "$(sysctl -n kern.hv_vmm_present 2>/dev/null)" != 1 ]; then
    echo "error: this is not a virtual machine, so nothing was done." >&2
    exit 3
fi
OUT=$HOME/repro-result
mkdir -p "$OUT"
stage() { echo "[$(date +%H:%M:%S)] $1"; echo "$1" >"$OUT/stage.txt"; }
finish() { echo "$1" >"$OUT/guest_verdict.txt"; stage "finished: $1"; exit 0; }
PASSWORD=${CUMLX_VM_PASSWORD:-admin}

stage "1/9 what this machine is"
{
    sw_vers
    uname -m
    sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu
    system_profiler SPDisplaysDataType
    df -k /
} >"$OUT/machine.txt" 2>&1

stage "2/9 the disk"
# An image from a registry can hold less than its disk. A fresh install already fills it, and then this is left alone.
if [ "$(df -k / | awk 'NR==2 {print int($4 / 1000000)}')" -lt 40 ]; then
    echo "$PASSWORD" | sudo -S sh -c 'yes | diskutil repairDisk disk0; diskutil apfs resizeContainer disk0s2 0' >"$OUT/disk.txt" 2>&1
fi
df -k / | tail -n 1 >>"$OUT/disk.txt"

stage "3/9 command line tools, for git"
if ! xcode-select -p >/dev/null 2>&1; then
    trigger=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
    touch "$trigger"
    label=$(softwareupdate -l 2>/dev/null | sed -n 's/^\* Label: \(Command Line Tools.*\)$/\1/p' | tail -n 1)
    [ -n "$label" ] || finish setup_error_no_command_line_tools_label
    echo "$PASSWORD" | sudo -S softwareupdate -i "$label" >"$OUT/clt.txt" 2>&1
    rm -f "$trigger"
fi
xcode-select -p >/dev/null 2>&1 || finish setup_error_command_line_tools
git --version >>"$OUT/machine.txt"

stage "4/9 uv, at a fixed version"
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh >"$OUT/uv.txt" 2>&1
PATH=$HOME/.local/bin:$PATH
export PATH
command -v uv >/dev/null 2>&1 || finish setup_error_uv

stage "5/9 CuMLX from its public repository, at $COMMIT"
mkdir -p "$HOME/code/CuMLX"
cd "$HOME/code/CuMLX" || finish setup_error_folder
git init -q . && git remote add origin https://github.com/Topabaem05/CuMLX || finish setup_error_git
git fetch -q --depth 1 origin "$COMMIT" && git checkout -q FETCH_HEAD || finish setup_error_public_commit
uv venv -q --python 3.12 .venv || finish setup_error_venv
uv pip install -q --python .venv/bin/python -e '.[dev,capture,diff,render,sim]' rsl-rl-lib==5.5.2 >"$OUT/install.txt" 2>&1 || finish setup_error_install
uv pip freeze --python .venv/bin/python >"$OUT/freeze.txt" 2>&1
PY=$PWD/.venv/bin/python
# The fractions in sysmem are for a Mac with room to spare. A guest holds 4 or 6 GiB, so these sit above what it can use, and the host's watcher keeps the host's limits.
CUMLX_MAX_USED_GB=${CUMLX_MAX_USED_GB:-6.3} CUMLX_START_USED_GB=${CUMLX_START_USED_GB:-5.6}
export CUMLX_MAX_USED_GB CUMLX_START_USED_GB

stage "6/9 the base: doctor, self test, tests without a GPU"
.venv/bin/cumlx doctor --require capture,diff --format json >"$OUT/doctor.json" 2>"$OUT/doctor.err"
echo "$?" >"$OUT/doctor.exit"
.venv/bin/cumlx selftest --format json >"$OUT/selftest.json" 2>"$OUT/selftest.err"
echo "$?" >"$OUT/selftest.exit"
"$PY" -c 'import json, mlx.core as mx; print(json.dumps({"default_device": str(mx.default_device()), "metal": mx.metal.is_available(), "info": {k: str(v) for k, v in mx.metal.device_info().items()}, "sum": float(mx.array([1.0, 2.0]).sum())}))' >"$OUT/mlx_device.json" 2>"$OUT/mlx_device.err"
"$PY" -m pytest -q -m 'not gpu' >"$OUT/tests.txt" 2>&1
echo "$?" >"$OUT/tests.exit"

stage "7/9 F3: the hand-written Metal kernels against the V100 goldens"
"$PY" -m pytest -q -rs tests/test_kernel_gpu.py >"$OUT/f3.txt" 2>&1
echo "$?" >"$OUT/f3.exit"

stage "8/9 F4: Go2 on Genesis through cumlx.sim, then the MLX policy"
"$PY" spikes/m4_sim.py train --task go2_flat --engine genesis --envs 4096 --iterations 300 --seed 1 --log-dir "$HOME/f4" >"$OUT/f4_train.txt" 2>&1
echo "$?" >"$OUT/f4_train.exit"
CK=$HOME/f4/model_299.pt
if [ -f "$CK" ]; then
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --controller mlx --forward 0.8 --output "$OUT/f4_fwd.json" >"$OUT/f4_fwd.txt" 2>&1
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --controller mlx --forward -0.8 --output "$OUT/f4_back.json" >"$OUT/f4_back.txt" 2>&1
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --checkpoint "$CK" --controller mlx --forward 0 --output "$OUT/f4_stop.json" >"$OUT/f4_stop.txt" 2>&1
    "$PY" spikes/m4_sim.py eval --task go2_flat --engine genesis --envs 128 --steps 1000 --seed 7 --controller zero --forward 0.8 --output "$OUT/f4_zero.json" >"$OUT/f4_zero.txt" 2>&1
fi

stage "9/9 F5: render on Metal"
"$PY" spikes/m5_render.py render --variant metal_ad_rgb --seeds 11,12,13 --out "$OUT/f5" >"$OUT/f5.txt" 2>&1
echo "$?" >"$OUT/f5.exit"

"$PY" spikes/m6_vm_judge.py "$OUT" >"$OUT/judge.txt" 2>&1
code=$?
case "$code" in
0) finish pass ;;
1) finish fail ;;
*) finish "not_run_exit_$code" ;;
esac
