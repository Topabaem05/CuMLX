# Changelog

## Unreleased

### Added

- `cumlx.compat.flash_attn_sdpa` and `cumlx.compat.bitsandbytes_int8` let an unmodified Transformers program with `flash_attention_2` and `BitsAndBytesConfig(load_in_8bit=True)` run on a Mac: `cumlx run infer.py --substitute flash_attn=cumlx.compat.flash_attn_sdpa --substitute bitsandbytes=cumlx.compat.bitsandbytes_int8`. The first is exact attention in float32 with FlashAttention-2's bottom-right causal mask. The second follows the int8 arithmetic of bitsandbytes 0.50.2 on CUDA, with an exact int8 product. Both refuse what they do not implement. A local run generated text with Qwen2.5-0.5B. No comparison with CUDA has run.
- `cumlx run` reports a distribution for these two adapters when `--substitute` names them (`flash-attn 2.8.3+cumlx`, `bitsandbytes 0.50.2+cumlx`), so that a library that checks a version finds one, and it records each answer in the run report. Other substitutes get none.
- The rules for F1 to F5 at their target grades are committed before any run: `validation/m6-f1-limits.json`, `m6-f2-cuda-limits.json`, `m6-f3-limits.json`, `m6-f4-limits.json` and `m6-f5-cuda-limits.json`. `sh spikes/m6_finish.sh start` runs the CUDA parts on the V100 when its GPU is idle and the Mac parts from nothing on this Mac, and writes the five verdicts. Nothing has run yet.

### Changed

- `cumlx.compat.gsplat_metalsplat` clamps each alpha at 0.999, as gsplat 1.5.3 does, where metalsplat has 0.99. It compiles metalsplat's rasterize kernels again with that value, and it stops if metalsplat's source lacks the two lines. No scene of `validation/m6-f2-adapter-run.json` had an alpha above 0.9.
- `cumlx.compat.gsplat_metalsplat` runs an unmodified program that calls `gsplat.rasterization` on metalsplat: `cumlx run train.py --substitute gsplat=cumlx.compat.gsplat_metalsplat`. It covers pinhole cameras and RGB, and any other option raises an error. Its run passed: `validation/m6-f2-adapter-run.json`. It equals a hand render of the documented formula, and no comparison with gsplat on CUDA exists, so F2 is at grade C against a target of B.
- F1 and F2 ran on the one real Mac, and the verdict is fail: `validation/m6-f1-f2-run.json`. `cumlx port` and training with the metalsplat API passed. The original programs stopped, because mps-bitsandbytes and metalsplat do not give the API that they need. No CUDA run took part, and M6-1 stays open.
- `cumlx port --revision` failed on every use, because the mlx-lm command line has no such flag. The converter now reads a local snapshot of that revision, and the weight comparison uses that revision too, where it used the default branch.
- `spikes/m6_f1_f2_work.sh` tries F1 and F2 on the one real Mac, by the original programs and by the Mac routes, under `validation/m6-f1-f2-limits.json`. No CUDA run takes part, so it cannot reach their target grades.
- The run on the one real Mac passed: `validation/m6-same-mac-run.json`. The base, F3, F4 and F5 passed under the VM's rules, from an empty home folder. The F4 values equal the M4 record exactly. M6-1 stays open, because this is the same Mac and F1 and F2 remain.
- `spikes/m6_same_mac.sh` repeats the VM's work on the one real Mac, from the public repository, with an empty home folder and no sudo. The rules are in `validation/m6-same-mac-limits.json`, and they are the VM's. A pass does not tick M6-1, because that condition names a new Mac.
- M6-1 ran in a real macOS 27.2 VM from `terminal-mac`, and the verdict is fail: `validation/m6-vm-run.json`. The public repository installs and its base passes, and the VM's paravirtual GPU cannot run F3, F4 or F5. So a VM cannot close M6-1. `CUMLX_VM_MEMORY_GIB`, `CUMLX_VM_USER`, `CUMLX_VM_PASSWORD` and `CUMLX_VM_PROXY` choose how it runs.
- The separate repository OrchardSim is deleted, as of 2026-11-04: its Isaac Sim work could not easily be published under that licence. The records that the reports cite are in `validation/archive-orchardsim/`, and they cannot be measured again from here.
- The repository is public as of 2026-11-04. `validation/m4-throughput.json` is a copy from the private repository OrchardSim, so that the M4 range can be read here.
- The history of `main` after `4709604` was rewritten on 2026-11-05 to remove a server name from two commits. Only the one line in `validation/archive-orchardsim/isaac-sim-o5.json` changed, and the 13 commits have new hashes. `validation/m6-vm-run.json` gives the run's commit under both hashes. Other private details in older commits stay.
- The history of `main` after `645450b` was rewritten on 2026-11-04, before the repository went public, to remove a local machine identity, a private ssh alias and a user name. Commit hashes after that point changed, and the references in the documents and records now use the new hashes. `tests/test_no_private_strings.py` keeps these strings out, by hash.

### Fixed

- `spikes/m6_vm_guest.sh` passed a flag that `spikes/m4_sim.py eval` does not have, so the F4 evaluations could never run. No earlier run had reached them. A test now checks each such line against the flags in the source.
- `spikes/m6_vm_reproduce.sh` no longer reads a verdict that an earlier run left in the guest, and ends its proxy tunnel with the run. `spikes/m6_terminal_mac.sh` holds its limits during the install, and survives a failed first boot.
- `spikes/run_all.sh` and `spikes/m3_kernels.sh` failed partway when `uv` was missing. They now end with exit 3 and the install link before they start.
- `spikes/t4_action_example.sh` used `sed -i ''`, which only BSD sed reads, and it failed late without `.venv`. The edit is now `spikes/t4_edit_workflow.py`, which refuses a workflow without the expected lines, and a missing `.venv` is exit 3 with the fix.
- `spikes/m2_golden_v100.sh`, `m5_render_v100.sh` and `m3_kernels.sh capture` defaulted to one person's ssh host. The host now comes from `CUMLX_GPU_HOST` or `~/.config/cumlx/gpu-host`, and with neither the launcher ends with exit 3 before it contacts anything.
- The long-run memory limit was a fixed number for one 48 GB Mac. `cumlx.sysmem` reads memory on macOS and Linux, with the page size that `vm_stat` states, and the limit is now 73% of this machine's RAM or `CUMLX_MAX_USED_GB`. On a 48 GiB Mac that is 37.6 GB. `spikes/m3_kernels.py` and `spikes/m5_render.py` use it.
- The spikes stop at 38 GB of system memory in use, down from 43 GB, on the user's instruction.
- `spikes/m5_render_v100.sh` installs Mitsuba 3.8.0, because Mitsuba 3.9 refuses a GPU under compute capability 7.5 and the V100 is 7.0. It also renders `scalar_rgb` as a control of the version difference.
- `spikes/m5_record.py` records the version control and the faults against the CUDA images.
- `spikes/m5_render.py render` now prints each stage and ends with a traceback if one stage takes longer than `--stall-seconds` (default 900). A run on the V100 had sat at full CPU for 30 minutes with an empty log.
- The kernel layer read `operator*=` as a global variable named `operator`, because of its `=`, and took the next declaration into it. It now reads a compound assignment operator as a function.
- The recursion check knew one definition per name. A wrapper such as `make_float4(float3, float)` that calls the built-in of the same name read as recursion, and a recursive overload was missed when another definition of its name came later. Each definition is now its own node, and a call goes to each overload that accepts its number of arguments. Overloads with the same number of arguments are still refused, because this layer does not read types.
- `is_complete_metal` and the signature reader did not see the `[[kernel]] void` form that woof writes, so a complete library read as a function body.

### Changed

- The separate repository OrchardSim is deleted, as of 2026-11-04: its Isaac Sim work could not easily be published under that licence. The records that the reports cite are in `validation/archive-orchardsim/`, and they cannot be measured again from here.
- The repository is public as of 2026-11-04. `validation/m4-throughput.json` is a copy from the private repository OrchardSim, so that the M4 range can be read here.
- The Isaac Sim and Isaac Lab scripts moved to the private repository [OrchardSim](https://github.com/Topabaem05/OrchardSim). The `spikes/isaac_*` and `spikes/rsl_rl_golden.py` entries below describe files that are now there. The records stay in `validation/`, and the recipes name the new location.

### Added

- `cumlx.render.usd` reads polygon meshes, one perspective camera, RectLight, DistantLight and a diffuse color from USD into a Mitsuba scene, and refuses anything else with `E_RENDER_USD_UNSUPPORTED`. Two small scenes match the same scene written in Mitsuba at grade B on Metal, and 6 injected faults fail. `validation/m5-usd.json`. The `usd` extra installs `usd-core`.
- CI: a base-install job on Linux (Python 3.10, 3.14) and macOS (3.12), Python 3.14 on the Linux CPU job, and required `shellcheck`. No runner tests a GPU: macOS runners may have no usable Metal device, and none has an NVIDIA GPU.
- `cumlx doctor --probe` reports memory in use, RAM and the limit for long runs, on any size of machine. `tests/test_m_series_generic.py` fails if code branches on one chip's name.
- `scripts/compat_check.py`: checks the CLI contract on a base install, with no MLX, torch or NVIDIA tools. It refuses an environment that has them. Run on a fresh venv with the built wheel: 14 of 14, on one Mac with Python 3.15 ([record](validation/compat-base-install.json)).
- `tests/test_launchers_portable.py`: every launcher parses in the shell that it names, and uses no flag or syntax of only one system. `shellcheck` runs where it is installed, and CI requires it.
- `cumlx.sim`: cartpole and Go2 flat task definitions on Genesis and MuJoCo, an rsl_rl environment, a camera on a task, and a policy in MLX checked with `check_op`. It is not Isaac Lab. Results and failed runs are in `validation/m4-sim.json`.
- `cumlx.render.lidar`: a raycast sensor on Mitsuba Metal, checked against exact distances. `validation/m5-lidar.json`.
- `cumlx doctor --stack --env PYTHON`: a stack spread over several environments. A missing part or an environment that cannot answer is still exit 3.
- `spikes/run_guarded.py`: run a command under start and stop limits on memory in use.
- `cumlx doctor --stack`: report each platform layer in this Python environment, and exit 3 with `E_DEP_MISSING` and an install hint unless all are ready. It installs nothing, uses no network and does not look for nvcc.
- `docs/PLATFORM-REPORT.md` and `docs/UNVERIFIED.md`: the result of each platform gate and flagship task with its grade, the failures, and what was not verified. A test keeps the report in step with the roadmap's gates. The repository is private, so neither is published yet.
- `spikes/m3_kernels.sh`: record the kernelport zoo and sam2's kernels on the V100, translate them with woof in their own environment, and compare. Prepared and tested on stubs only; `validation/kernel-m3.json` says that no third-party kernel has run yet.
- `cumlx kernel verify --engine woof`: run a woof translation through woof's own launcher, which binds its scalars and threadgroup memory. Tested on a stub only.
- `cumlx kernel port --allow-warp-level`: pass warp-level built-ins to the translator instead of refusing them, with `W_KERNEL_WARP_LEVEL_PASSED`. The default is still to refuse.
- `call` and `initial` in `cumlx.opgolden.v1` and `cumlx.kernel-cases.v1`: scalar parameters in any position, and buffers that a kernel reads and writes. Older goldens read the same.
- `cumlx.render` (experimental, extra `render`): RGB, depth and segmentation from Mitsuba on Metal, frame checks against a physics state, and a grade B comparison of path-traced images between devices. See `docs/RENDER-SENSORS.md`.
- `examples/physics-camera/drop_box.py`: Genesis physics and a Mitsuba camera in one loop on Metal, checked frame by frame, with two wrong scenarios that must fail.
- `spikes/m5_render.py`, `spikes/m5_render_v100.sh`: the Metal and CUDA render comparison, and the frame rate measurement. The V100 side has not run.

- `cumlx run` (experimental): run an unmodified PyTorch script that assumes CUDA on MPS, and record each redirected call in `run-report.json`. A CUDA call with no equivalent, a second GPU, an operation that MPS lacks and float64 all end with an error. `PYTORCH_ENABLE_MPS_FALLBACK` is removed unless you pass `--allow-cpu-fallback`. Importing `cumlx` changes nothing in torch. MCP shows this command only with `--allow-code-execution`.
- `spikes/m2_scripts.sh`: run 5 open-source scripts under `cumlx run`, once per seed for those that train. `--target cuda` records the golden.
- `spikes/m2_substitutes.sh`: evaluate `mps-*`, metalsplat, msplat and mlx3D, each in its own venv at a pinned version, with checks against known numbers.
- `cumlx rundiff` (experimental): compare the loss traces of `cumlx run` from CUDA and from the target, and give grade A (first two steps) or B (training curve over 3 or more seeds). `cumlx run` gains `--trace-losses`, `--seed` and `--target cuda`.
- `validation/isaac-lab-go2.json`: Unitree Go2 flat velocity on a Mac CPU with Newton. With 4096 envs and 300 iterations, the policy in MLX measures +0.74 m/s at a +0.8 command and -0.87 m/s at -0.8, in simulation, with one seed.
- `spikes/isaac_lab_mac.sh`: train Isaac Lab cartpole on a Mac CPU with Newton, without Isaac Sim. NVIDIA does not support this, and Isaac Lab's `uv sync` refuses macOS, so it builds the environment by hand.
- `spikes/rsl_rl_golden.py` and `verify_policy.py --golden`: compare an MLX policy with rsl_rl's own outputs. Before this, the reference was a torch version written in the same file.
- `spikes/isaac_lab_mlx_loop.py`: let an MLX policy control an Isaac Lab environment, with controls.
- `cumlx kernel capture` (experimental): compile the original CUDA kernels with nvcc, run them on a real NVIDIA GPU and record an op-level golden, `cumlx.opgolden.v1`. It executes the code, so MCP shows it only with `--allow-code-execution`.
- `cumlx kernel verify` (experimental): run a Metal kernel through `mx.fast.metal_kernel` or `torch.mps.compile_shader` on the golden inputs and compare the outputs. An unknown kernel, a gap, a failed compile and float64 are exit 5. Each report states fp64, FTZ, FMA contraction and execution order, measured on the device.
- `cumlx kernel port` (experimental): prepare a `.cu` file (standard headers, template instances, host code, PTX built-ins) and call a translator through `TranslatorAdapter`. A missing translator is exit 3. A feature with no Metal equivalent, such as warp-level behaviour, is refused with its reason. The woof adapter was tested only with a stub, not with woof.
- `validation/kernel-translation.json`: the first CUDA runs of the 8 FP4 kernels, on a Tesla V100. The hand-written Metal kernels pass 29 of 29 cases on an M5 Pro, worst `rel_l2` 7.5e-8. All 9 wrong translations of 8 kinds are caught, and the 9 right ones pass.

### Changed

- The separate repository OrchardSim is deleted, as of 2026-11-04: its Isaac Sim work could not easily be published under that licence. The records that the reports cite are in `validation/archive-orchardsim/`, and they cannot be measured again from here.
- The repository is public as of 2026-11-04. `validation/m4-throughput.json` is a copy from the private repository OrchardSim, so that the M4 range can be read here.
- Recipes: metalsplat, mps-flash-attn on torch 2.13, PyTorch sdpa on MPS and `torchvision.ops.deform_conv2d` now have `run` evidence, against the CPU or hand values, not CUDA. mps-bitsandbytes and the other `mps-*` packages failed on torch 2.15. `cumlx run` maps no package automatically, because none of them passed as a drop-in under the original name. `validation/m2-substitutes.json`.

### Fixed

- `spikes/m6_vm_reproduce.sh` no longer reads a verdict that an earlier run left in the guest, and ends its proxy tunnel with the run. `spikes/m6_terminal_mac.sh` holds its limits during the install, and survives a failed first boot.
- `cumlx run script.py --target x -- args` failed with `E_USAGE` on Python 3.12.3 and 3.10, because argparse there handles this mix of positionals differently. `--help` and `--format` after `--` also went to CUMLX and not to the script. `--module` could not take arguments at all.
- `examples/robot-policy` refused real rsl_rl 5 checkpoints: it expected `model_state_dict` and `actor.N`, but rsl_rl 5 writes `actor_state_dict` and `mlp.N`. Only random weights had been tried. It now also refuses an observation normalizer, which it would otherwise have left out with no error.
- The README said that `pip install isaacsim` installs an empty package on a Mac. It fails.

## 0.5.0 (2026-11-02)

### Added

- `cumlx recipe`: name what to use on Apple Silicon for a CUDA dependency. The catalog has 33 recipes. Each option states whether it was run on Apple Silicon (`evidence: run`, with its file under `validation/` and a measured grade) or only read about (`evidence: docs`). `--verified-only` keeps what was run. A test refuses a catalog that claims a run without evidence.
- `cumlx scan` reads declared dependencies from `requirements*.txt`, `pyproject.toml` and `environment*.yml`, and links each finding to its recipes. New rules: `gsplat`, `pytorch3d`, `mmcv-ops`, `nvdiffrast`, `triton`, `isaac-lab`.
- `cumlx port` names a converter for a vision model or a diffusers pipeline when mlx-lm does not implement it.
- `spikes/isaac_sim_container.sh`: run Isaac Sim 6.2 in a Linux arm64 container on a Mac. Kit and CPU PhysX run; RTX and cameras do not. NVIDIA does not support this. The script never accepts the EULA for you.
- `docs/ROADMAP-PLATFORM.md`: the next goal, running RTX and CUDA workloads on Apple Silicon, with the measured M0 results in `validation/spikes.json`.

## 0.4.1 (2026-11-02)

### Added

- `cumlx scan`: rank the work to run a CUDA project or model on Apple Silicon, from a text scan. It never runs the scanned code.
- `cumlx port`: convert with mlx-lm and prove the result in one command. It refuses a model that mlx-lm does not implement.
- `docs/PORTING-CUDA-MODELS.md`: what can be converted, and how to use Isaac Sim from a Mac.
- `examples/robot-policy`: move a robot policy from Isaac Lab to MLX and prove that it computes the same.
- Mixtral (`block_sparse_moe`) and channels-last convolution support in `weights`, `diff` and `--weights-from-mlx`.

### Fixed

- `spikes/m6_vm_reproduce.sh` no longer reads a verdict that an earlier run left in the guest, and ends its proxy tunnel with the run. `spikes/m6_terminal_mac.sh` holds its limits during the install, and survives a failed first boot.
Each came from running recent models, and each has a test.

- Replay now uses the mask that the port itself passed. A causal mask broke the LFM2 convolution.
- A decode-phase module with a float input is a hidden state, not tokens.
- A constant factor that the parent module applies, as in Gemma embeddings, is no longer a false alarm. A real scale error still fails, because the parent then differs.
- `capture` refuses a checkpoint with missing weights. transformers only logs them and fills them with random values, so a golden from it was meaningless.
- `--weights-from-mlx` refuses to write a golden that leaves out part of the port's weights.

### Measured

- CUDA golden on a Tesla V100: repeat runs are identical, CUDA against CPU is at most 1.9e-5, and an MLX port on a Mac passes against it.
- 12 recent Hugging Face models: 7 ok, 4 not comparable (partial checkpoints), 1 not loadable by mlx-lm.

## 0.4.0 (2026-11-02)

The first release of the porting parity verifier. CUMLX does not convert models. It proves that an MLX port computes the same as the original PyTorch model and locates the first module that differs.

### Added

- `cumlx capture`, `diff`, `report`: golden traces, isolated replay of every module, chain comparison in the native dtype, and JSON, Markdown and HTML reports.
- Replay of attention and decoder layers with a causal mask, of the decode phase on the port's own KV cache, and of MoE experts on the golden routing.
- `cumlx weights`: static comparison that names transposed, sign-flipped, rescaled and reordered weights, per expert for MoE.
- `cumlx conform`: FP4 decode conformance of MLX on CPU and GPU. `--checkpoint` compares a ModelOpt NVFP4 checkpoint with an MLX repack, byte by byte.
- `cumlx tracediff`: how far a golden moves between devices.
- `cumlx.check_op`: parity of one op, for hand-written kernels.
- `cumlx selftest`: 11 injected faults must be located in the current environment.
- `action.yml`, `cumlx-mcp`, `skills/cumlx-verifier`, and an agent-friendly CLI contract (`cumlx.result.v1`).
- `--device` and `--tf32`. MLX runs float32 matmul in TF32 by default on M5 and CUDA, so comparisons default to full float32 and record the measured error.
- Thresholds from measured clean ports. Each run sets them from its median leaf error, so a defect cannot loosen its own threshold.
- `docs/TRACE-FORMAT.md`: the trace specification.

### Changed

- The separate repository OrchardSim is deleted, as of 2026-11-04: its Isaac Sim work could not easily be published under that licence. The records that the reports cite are in `validation/archive-orchardsim/`, and they cannot be measured again from here.
- The repository is public as of 2026-11-04. `validation/m4-throughput.json` is a copy from the private repository OrchardSim, so that the M4 range can be read here.
- The kernel lab accepts MLX 0.32.2 and later 0.32.x. Its 35 GPU tests passed on Apple M5 Pro with 0.32.2 and 0.32.4.
- The FP4 kernel lab moved to `docs/FP4-KERNEL-LAB.md`. The README is organized by use case.

### Known limits

- Not run: Apple GPU generations other than M5, padded inputs, models that are not LLMs.
- The thresholds come from two model families on one machine.
- Not published to PyPI.
