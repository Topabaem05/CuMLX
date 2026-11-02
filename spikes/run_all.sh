#!/bin/bash
# Install the M0 candidates into new venvs, fetch the test inputs, and run each probe with a time limit.
# A failed probe is a result, not an error. Results go to WORK_DIR/results.jsonl.
set -u
WORK="${1:?usage: run_all.sh WORK_DIR}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Classic/Cartpole
KERNELS="$HERE/../src/cumlx/kernels"
mkdir -p "$WORK/asset/Props" "$WORK/foreign" "$WORK/logs"

for venv in venv woof kernelport; do uv venv -q --python 3.12 "$WORK/$venv"; done
uv pip install -q --python "$WORK/venv/bin/python" genesis-world==1.4.5 torch==2.15.0 mitsuba==3.9.1 usd-core==26.11 quadrants==1.3.3 mujoco==3.17.0 numpy==2.5.4
uv pip install -q --python "$WORK/woof/bin/python" metal-woof==1.1.5.2 'libclang>=18.1' 'mlx>=0.29' 'nanobind>=3.0,<3.2'
uv pip install -q --python "$WORK/kernelport/bin/python" 'kernelport[metal]==0.1.0'

curl -sf -m 60 -o "$WORK/asset/cartpole.usd" "$ISAAC/cartpole.usd"
curl -sf -m 60 -o "$WORK/asset/Props/instanceable_meshes.usd" "$ISAAC/Props/instanceable_meshes.usd"
curl -sf -m 60 -o "$WORK/foreign/connected_components.cu" https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/csrc/connected_components.cu
git clone -q --depth 1 https://github.com/NVlabs/nvdiffrast "$WORK/foreign/nvdiffrast"
git clone -q --depth 1 https://github.com/PS12007/kernelport "$WORK/foreign/kernelport"

: > "$WORK/results.jsonl"
run() {  # run NAME LIMIT PYTHON SCRIPT [ARGS...]
  local name="$1" limit="$2" python="$3" script="$4"; shift 4
  echo "== $name"
  ( cd "$WORK/logs" && perl -e 'alarm shift; exec @ARGV' "$limit" "$WORK/$python/bin/python" -I "$HERE/probes/probe_$script.py" "$@" ) > "$WORK/logs/$name.log" 2>&1
  echo "exit=$?"
  grep '^{' "$WORK/logs/$name.log" >> "$WORK/results.jsonl"
}
NV="$WORK/foreign/nvdiffrast/csrc/common"
run usd 120 venv usd
run usd_genesis 400 venv usd_genesis "$WORK/asset/cartpole.usd"
run mitsuba 400 venv mitsuba
run quadrants 300 venv quadrants
run kernelport 600 kernelport kernelport "$WORK/kernelport/bin/kernelport" "$WORK/foreign/kernelport/zoo"
run woof 600 woof woof "$KERNELS" "$WORK/foreign/connected_components.cu" "$NV/interpolate.cu" "$NV/rasterize.cu" "$NV/antialias.cu" "$NV/texture_kernel.cu" "$WORK"/foreign/kernelport/zoo/*/kernel.cu
run genesis 1200 venv genesis
cut -c1-300 "$WORK/results.jsonl"
