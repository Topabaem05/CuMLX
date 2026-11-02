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
| `probe_isaac_sim.py` | Does Isaac Sim run? Use `isaac_sim_container.sh` on a Mac. It is separate from `run_all.sh`, because it needs about 19 GB and your acceptance of the NVIDIA EULA. |

The `probe_` prefix stops a script from shadowing the package that it imports. Delete the work directory afterwards: it took 4.8 GB. Genesis also creates `~/.cache/genesis`. In `validation/spikes.json` the last run is the complete reproduction from an empty directory, and the earlier runs repeat the throughput measurement.
