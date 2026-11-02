# M0 spikes

These scripts check the assumptions in [the platform roadmap](../docs/ROADMAP-PLATFORM.md). **They have not been run yet**, so their API calls may need fixes. Each probe installs nothing itself and prints one JSON line per check. The `probe_` prefix stops a script from shadowing the package that it imports.

They install third-party packages into a new virtual environment and run them, so run them when you accept that.

```bash
bash spikes/run_all.sh /tmp/cumlx-spikes
```

Results go to `<work dir>/results.jsonl`. Delete the work directory afterwards. The disk space it needs has not been measured.
