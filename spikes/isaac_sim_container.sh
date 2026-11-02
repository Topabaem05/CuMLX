#!/bin/bash
# Run Isaac Sim 6.2 in a Linux arm64 container on an Apple Silicon Mac: CPU PhysX only, with no GPU and no RTX.
# NVIDIA does not support this setup. It needs about 19 GB in the Docker volume.
# Accepting the NVIDIA EULA is your decision, so this script never sets it for you:
#   OMNI_KIT_ACCEPT_EULA=YES bash spikes/isaac_sim_container.sh
set -eu
if [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
  echo "Isaac Sim needs the NVIDIA EULA. Read it, then set OMNI_KIT_ACCEPT_EULA=YES yourself." >&2
  exit 3
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
VOLUME="${ISAAC_VOLUME:-cumlx-isaac}"
LIMIT="${ISAAC_LIMIT:-520}"
docker volume create "$VOLUME" >/dev/null
docker run --rm -i --platform linux/arm64 -v "$VOLUME:/work" ubuntu:24.04 bash -c 'cat > /work/probe_isaac_sim.py' < "$HERE/probes/probe_isaac_sim.py"
docker run --rm --platform linux/arm64 -v "$VOLUME:/work" -e OMNI_KIT_ACCEPT_EULA=YES -e LIMIT="$LIMIT" ubuntu:24.04 bash -c '
  set -u
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev build-essential libgl1 libglib2.0-0 libxt6 libxi6 libxrender1 libsm6 libice6 libgomp1 libvulkan1 mesa-vulkan-drivers >/dev/null 2>&1
  cd /work
  [ -d venv ] || python3.12 -m venv venv
  . venv/bin/activate
  pip list 2>/dev/null | grep -q "^isaacsim " || pip install -q "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
  LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ISAAC_WORK=/work/assets timeout "$LIMIT" python /work/probe_isaac_sim.py > /work/probe.log 2>&1
  echo "exit=$?"
  grep "^RESULT" /work/probe.log | cut -c8-
  echo "error_lines=$(grep -c "\[Error\]" /work/probe.log)"
  grep -m1 "No device could be created" /work/probe.log | sed -E "s/^[^ ]+ \[[0-9,]+ms\] //"
'
echo "Delete the install when you are done: docker volume rm $VOLUME"
