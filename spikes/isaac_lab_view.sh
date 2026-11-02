#!/bin/sh
# Show a trained Go2 policy in Newton's viewer, without Isaac Sim. Close the window or press Ctrl+C to stop.
#
#   sh spikes/isaac_lab_view.sh                  # the newest checkpoint
#   sh spikes/isaac_lab_view.sh path/model.pt    # a chosen checkpoint
#
# It needs the environment from: python spikes/isaac_lab_mac.py --work ~/cumlx-isaaclab/go2 ...
set -u
W=${CUMLX_ISAACLAB_ROOT:-$HOME/cumlx-isaaclab}/go2
if [ ! -x "$W/venv/bin/isaaclab" ]; then
    echo "error: no environment in $W" >&2
    exit 3
fi
CHECKPOINT=${1:-$(ls -t "$W"/run/logs/rsl_rl/unitree_go2_flat/*/model_*.pt 2>/dev/null | head -n 1)}
if [ ! -f "$CHECKPOINT" ]; then
    echo "error: no checkpoint: $CHECKPOINT" >&2
    exit 4
fi
echo "Checkpoint: $CHECKPOINT"
echo "A window opens in about 20 seconds."
cd "$W/run" || exit 3
ISAACSIM_ASSET_ROOT="$W/assets" exec "$W/venv/bin/isaaclab" play --rl_library rsl_rl \
    --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint "$CHECKPOINT" --device cpu --num_envs 16 \
    --viz newton_gl presets=newton_mjwarp
