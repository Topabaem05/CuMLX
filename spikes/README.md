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
| `m5_render_v100.sh` | Renders the same scenes with `cuda_ad_rgb` on the V100. `start`, `status`, `fetch`. It refuses a busy GPU. Then run `m5_render.py compare ~/cumlx-m5/cuda ~/cumlx-m5/metal`. |
| `m5_record.py` | Writes `validation/m5-render.json` from the raw results. |

The `probe_` prefix stops a script from shadowing the package that it imports. Delete the work directory afterwards: it took 4.8 GB. Genesis also creates `~/.cache/genesis`. In `validation/spikes.json` the last run is the complete reproduction from an empty directory, and the earlier runs repeat the throughput measurement.
