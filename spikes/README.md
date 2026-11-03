# M0 spikes

These scripts check the assumptions in [the platform roadmap](../docs/ROADMAP-PLATFORM.md). The measured results are in [validation/spikes.json](../validation/spikes.json).

They install third-party packages into new virtual environments and run them, so run them when you accept that. They also download one Isaac Lab asset, two kernel repositories and one kernel file.

```bash
bash spikes/run_all.sh <work dir>
python spikes/collect.py -o validation/spikes.json <work dir>/results.jsonl
```

| Probe | Question |
|---|---|
| `probe_usd.py` | Does OpenUSD read and write a physics scene? |
| `probe_usd_genesis.py` | Does Genesis load and simulate a real Isaac Lab asset? |
| `probe_mitsuba.py` | Does Mitsuba render on Metal, and does it agree with the CPU? |
| `probe_quadrants.py` | Does a Quadrants kernel agree between Metal and CPU? |
| `probe_kernelport.py` | Does kernelport pass correct Metal ports and fail a wrong one? |
| `probe_woof.py` | Does woof's translator run kernels that it was not written for? |
| `probe_genesis.py` | What is the Go2 throughput on Metal and CPU? |

The Isaac Sim and Isaac Lab scripts moved to the private repository [OrchardSim](https://github.com/Topabaem05/OrchardSim) on 2026-11-03. `validation/isaac-sim.json`, `validation/isaac-lab-mac.json` and `validation/isaac-lab-go2.json` stay here, because the recipes and the checkpoint test use them. The script paths in those records are the paths at the time of the run.

M5 scripts run outside `run_all.sh` too. The limits are in [validation/m5-render-limits.json](../validation/m5-render-limits.json).

| Script | Question |
|---|---|
| `m5_render.py render`, `compare`, `rate` | Do Metal and another device render the same image, by the grade B rule? What is the 256x256 frame rate? |
| `m5_render_v100.sh` | Renders the same scenes with `cuda_ad_rgb` on the V100, on Mitsuba 3.8.0 because 3.9 refuses that GPU, and `scalar_rgb` as a control of the version difference. `start`, `status`, `fetch`. It refuses a busy GPU. Then run `m5_render.py compare ~/cumlx-m5/cuda ~/cumlx-m5/metal` and `compare ~/cumlx-m5/scalar-3.8.0 ~/cumlx-m5/scalar`. |
| `m5_record.py` | Writes `validation/m5-render.json` from the raw results. |
| `m5_lidar.py` | Does a raycast on Metal match exact distances, and do wrong cases fail? |
| `m4_sim.py`, `m4_sim_record.py` | Do `cumlx.sim` tasks train and agree between Genesis and MuJoCo? Writes `validation/m4-sim.json`. |
| `run_guarded.py` | Runs one command under start and stop limits on memory in use. |

The `probe_` prefix stops a script from shadowing the package that it imports. Delete the work directory afterwards: it took 4.8 GB. Genesis also creates `~/.cache/genesis`. In `validation/spikes.json` the last run is the complete reproduction from an empty directory, and the earlier runs repeat the throughput measurement.

## M3 kernels

`m3_kernels.sh` works toward the M3 gates in [the platform roadmap](../docs/ROADMAP-PLATFORM.md). It installs metal-woof into `~/cumlx-m3/woof-venv` and runs it with your permissions. `sh spikes/m3_kernels.sh capture` compiles and runs third-party kernels on the V100, and it stops if the GPU is busy. `status` shows the state, and `tools` prints the programs that it would run. It takes the kernels at the commits in [sources.json](../validation/kernel-cases-m3/sources.json), and checks each hash. Delete `~/cumlx-m3` afterwards.

## Settings on another machine

| Setting | Meaning | Default |
|---|---|---|
| `CUMLX_GPU_HOST`, or the file `~/.config/cumlx/gpu-host` | The ssh host of an NVIDIA machine, for the launchers that use one | None: they end with exit 3 |
| `CUMLX_MAX_USED_GB` | Long runs stop above this much memory in use | 73% of RAM |
| `CUMLX_M2_ROOT`, `CUMLX_M2_GOLDEN`, `CUMLX_M2_SUBSTITUTES_ROOT`, `CUMLX_M3_ROOT`, `CUMLX_M5_GOLDEN` | Where results go | Folders named `cumlx-*` under your home |

`uv` is required for the runs that make environments. Without it they end with exit 3 and the install link. See `tests/test_launchers_portable.py`, `tests/test_launcher_host.py` and `scripts/compat_check.py`.
