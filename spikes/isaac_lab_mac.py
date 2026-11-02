#!/usr/bin/env python3
"""Try Isaac Lab Cartpole training with Newton on macOS, without Isaac Sim.

Isaac Lab does not support macOS. Its lock file names Linux and Windows only, so `uv sync` refuses
this platform. This script builds an unsupported environment by hand: the dependencies from Isaac
Lab's pyproject.toml, with the packages that have no macOS wheel removed, and usd-core in place of
usd-exchange.

Each stage records what happened in result.json. A failed stage stays failed. It is never a pass.

    python spikes/isaac_lab_mac.py --checkout IsaacLab --work /path/to/empty/dir

Stdlib only, Python 3.11 or later for this driver. The training runs on Python 3.12 in its own venv.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ASSET_ROOT_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2"
ROOT_ASSETS = (
    "Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd",
    "Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1/default_ground_plane.usda",
)
# Measured on PyPI, 2026-11-02: no macOS arm64 wheel for these pinned versions.
NO_MAC_WHEEL = {"usd-exchange", "omniverseclient", "usd-optimize"}
USD_PROVIDER = "usd-core==26.8"  # usd-exchange 3.0.1 vendors USD 26.08.
WORKSPACE_MEMBERS = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_experimental",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_physx",
    "isaaclab_ppisp",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_visualizers",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
REWARD = re.compile(r"Mean reward:\s+(-?\d+(?:\.\d+)?)")
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
ITERATION_TIME = re.compile(r"Iteration time:\s+(\d+(?:\.\d+)?)s")
KEY_PACKAGES = ("torch", "warp-lang", "newton", "mujoco", "mujoco-warp", "usd-core", "rsl-rl-lib")


def _name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9._-]+)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""


def plan_requirements(pyproject: dict) -> tuple[list[str], list[str]]:
    """Return (requirements, overrides) for macOS from Isaac Lab's pyproject.toml."""
    keep = []
    for requirement in pyproject["project"]["dependencies"]:
        name = _name(requirement)
        if name.startswith("isaaclab") or name in NO_MAC_WHEEL:
            continue
        keep.append(requirement)
    keep.append(USD_PROVIDER)
    overrides = [
        o for o in pyproject["tool"]["uv"]["override-dependencies"] if _name(o) not in NO_MAC_WHEEL
    ]
    return keep, overrides


def parse_training(text: str) -> dict:
    """Read rsl_rl console output. Missing numbers stay missing; they never become a pass."""
    text = ANSI.sub("", text)
    rewards = [float(m) for m in REWARD.findall(text)]
    iterations = ITERATION.findall(text)
    times = [float(m) for m in ITERATION_TIME.findall(text)]
    return {
        "iterations_seen": len(iterations),
        "last_iteration": int(iterations[-1][0]) if iterations else None,
        "reward_first": rewards[0] if rewards else None,
        "reward_last": rewards[-1] if rewards else None,
        "reward_best": max(rewards) if rewards else None,
        "iteration_seconds_median": sorted(times)[len(times) // 2] if times else None,
    }


def layer_dependencies(python: Path, layer: Path) -> list[str]:
    """Ask USD for the external paths of one layer, using the venv's usd-core."""
    code = (
        "import json,sys\n"
        "from pxr import UsdUtils\n"
        "a,b,c=UsdUtils.ExtractExternalReferences(sys.argv[1])\n"
        "print(json.dumps(list(a)+list(b)+list(c)))\n"
    )
    done = subprocess.run(
        [str(python), "-c", code, str(layer)], capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def mirror_assets(python: Path, root: Path, log: list[str]) -> dict:
    """Copy the Cartpole assets locally. The remote path needs omni.client, which has no macOS wheel."""
    pending = list(ROOT_ASSETS)
    fetched, unresolved, placeholders = [], [], []
    while pending:
        relative = pending.pop()
        if relative in fetched or relative in unresolved or relative in placeholders:
            continue
        target = root / relative
        url = f"{ASSET_ROOT_URL}/{urllib.parse.quote(relative)}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        except urllib.error.URLError as exc:
            log.append(f"{relative}: {exc}")
            if relative.endswith(".mdl") and getattr(exc, "code", None) == 404:
                # OmniPBR.mdl is built into Kit, so the asset server does not have it. Isaac Lab asks
                # omni.client about a missing local file, and omni.client has no macOS wheel. MDL is
                # visual only and this run has no renderer, so an empty file answers that question.
                target.write_bytes(b"")
                placeholders.append(relative)
            else:
                unresolved.append(relative)
            continue
        fetched.append(relative)
        if target.suffix.lower() in {".usd", ".usda", ".usdc"}:
            for dependency in layer_dependencies(python, target):
                if "://" in dependency or dependency.startswith("/"):
                    unresolved.append(dependency)
                    continue
                pending.append(
                    os.path.normpath(os.path.join(os.path.dirname(relative), dependency))
                )
    missing = [asset for asset in ROOT_ASSETS if asset not in fetched]
    if missing:
        raise OSError(f"could not fetch {missing}")
    return {"fetched": fetched, "placeholders": sorted(placeholders), "unresolved": sorted(set(unresolved))}


class Stages:
    def __init__(self, work: Path, head: str) -> None:
        self.work = work
        self.path = work / "result.json"
        self.data: dict = {
            "schema": "cumlx.spike.isaac_lab_mac.v1",
            "isaac_lab_commit": head,
            "supported_by_nvidia": False,
            "stages": {},
        }

    def record(self, name: str, **fields) -> None:
        self.data["stages"][name] = fields
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False) + "\n")
        status = fields.get("status")
        print(f"[{name}] {status} {fields.get('note', '')}".rstrip(), flush=True)

    def run(self, name: str, argv: list[str], *, cwd: Path, env=None, timeout: int) -> tuple[int, str, float]:
        log = self.work / f"{name}.log"
        start = time.monotonic()
        print(f"[{name}] running, log {log}", flush=True)
        try:
            with log.open("w") as handle:
                code = subprocess.run(
                    argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout
                ).returncode
        except subprocess.TimeoutExpired:
            code = 124
        except FileNotFoundError as exc:
            log.write_text(str(exc))
            code = 127
        seconds = round(time.monotonic() - start, 1)
        text = log.read_text(errors="replace")
        return code, text, seconds


def _tail(text: str, lines: int = 25) -> list[str]:
    return [ANSI.sub("", line)[:300] for line in text.splitlines()[-lines:]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True, help="Isaac Lab source checkout")
    parser.add_argument("--work", type=Path, required=True, help="directory for the venv, assets, logs")
    parser.add_argument("--learn-envs", type=int, default=256)
    parser.add_argument("--learn-iterations", type=int, default=100)
    parser.add_argument("--smoke-timeout", type=int, default=1800)
    parser.add_argument("--learn-timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    import tomllib  # Python 3.11. The rest of this file also loads on 3.10, for its tests.

    checkout = args.checkout.resolve()
    work = args.work.resolve()
    if sys.platform != "darwin":
        print("This script exists for macOS.", file=sys.stderr)
        return 3
    if shutil.which("uv") is None:
        print("uv is required.", file=sys.stderr)
        return 3
    if not (checkout / "pyproject.toml").is_file():
        print(f"Not an Isaac Lab checkout: {checkout}", file=sys.stderr)
        return 4
    free_gb = shutil.disk_usage(work.parent if not work.exists() else work).free / 1e9
    if free_gb < 15:
        print(f"Need 15 GB free, found {free_gb:.1f} GB.", file=sys.stderr)
        return 3
    work.mkdir(parents=True, exist_ok=True)

    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    stages = Stages(work, head)
    print(f"Isaac Lab {head[:7]}, work folder {work}", flush=True)
    venv = work / "venv"
    python = venv / "bin" / "python"
    pyproject = tomllib.loads((checkout / "pyproject.toml").read_text())
    requirements, overrides = plan_requirements(pyproject)
    (work / "requirements.txt").write_text("\n".join(requirements) + "\n")
    (work / "overrides.txt").write_text("\n".join(overrides) + "\n")

    # --no-config keeps Isaac Lab's Linux-only index and environment settings out of this install.
    code, text, seconds = stages.run(
        "venv", ["uv", "venv", "--no-config", "--clear", "--python", "3.12", str(venv)], cwd=work, timeout=600
    )
    if code:
        stages.record("venv", status="failed", exit_code=code, tail=_tail(text))
        return 1
    stages.record("venv", status="ok", seconds=seconds)
    code, text, seconds = stages.run(
        "install_deps",
        ["uv", "pip", "install", "--no-config", "--python", str(python),
         "-r", "requirements.txt", "--override", "overrides.txt"],
        cwd=work,
        timeout=1800,
    )
    if code:
        stages.record("install_deps", status="failed", exit_code=code, seconds=seconds, tail=_tail(text))
        return 1
    stages.record("install_deps", status="ok", seconds=seconds, removed=sorted(NO_MAC_WHEEL), added=USD_PROVIDER)

    editable: list[str] = []
    for member in WORKSPACE_MEMBERS:
        editable += ["-e", str(checkout / "source" / member)]
    code, text, seconds = stages.run(
        "install_isaaclab",
        ["uv", "pip", "install", "--no-config", "--no-deps", "--python", str(python), *editable],
        cwd=work,
        timeout=900,
    )
    if code:
        stages.record("install_isaaclab", status="failed", exit_code=code, seconds=seconds, tail=_tail(text))
        return 1
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--no-config", "--python", str(python)],
        capture_output=True, text=True, check=False, cwd=work,
    ).stdout
    versions = {}
    for line in freeze.splitlines():
        if _name(line) in KEY_PACKAGES:
            versions[_name(line)] = line
    stages.record("install_isaaclab", status="ok", seconds=seconds, versions=versions)

    assets = work / "assets"
    notes: list[str] = []
    try:
        mirrored = mirror_assets(python, assets, notes)
    except (OSError, subprocess.CalledProcessError) as exc:
        stages.record("assets", status="failed", error=str(exc)[:400], notes=notes)
        return 1
    stages.record("assets", status="ok", note=f"{len(mirrored['fetched'])} files", notes=notes, **mirrored)

    env = dict(os.environ, ISAACSIM_ASSET_ROOT=str(assets), PYTHONUNBUFFERED="1")
    env.pop("VIRTUAL_ENV", None)
    train = [
        str(venv / "bin" / "isaaclab"), "train", "--rl_library", "rsl_rl",
        "--task", "Isaac-Cartpole-Direct", "--device", "cpu", "presets=newton_mjwarp",
    ]
    # The same probe as upstream's source/isaaclab/test/install_ci/misc/cartpole_training_smoke.py, plus --device cpu.
    run_dir = work / "run"
    run_dir.mkdir(exist_ok=True)
    earlier = set(run_dir.glob("logs/rsl_rl/**/model_*.pt"))  # A rerun must not count an old checkpoint.
    code, text, seconds = stages.run(
        "smoke", [*train, "--num_envs", "16", "--max_iterations", "5"],
        cwd=run_dir, env=env, timeout=args.smoke_timeout,
    )
    checkpoints = [c for c in sorted(run_dir.glob("logs/rsl_rl/**/model_*.pt")) if c not in earlier]
    fields = parse_training(text)
    if code or not checkpoints:
        stages.record(
            "smoke", status="failed", exit_code=code, seconds=seconds, checkpoints=len(checkpoints),
            tail=_tail(text), **fields,
        )
        return 1
    stages.record("smoke", status="ok", seconds=seconds, checkpoints=len(checkpoints), **fields)

    before = earlier | set(checkpoints)
    code, text, seconds = stages.run(
        "learn",
        [*train, "--num_envs", str(args.learn_envs), "--max_iterations", str(args.learn_iterations)],
        cwd=run_dir, env=env, timeout=args.learn_timeout,
    )
    fields = parse_training(text)
    created = [c for c in sorted(run_dir.glob("logs/rsl_rl/**/model_*.pt"), key=os.path.getmtime) if c not in before]
    if code or not created:
        stages.record("learn", status="failed", exit_code=code, seconds=seconds, tail=_tail(text), **fields)
        return 1
    final = work / "checkpoint.pt"
    shutil.copy2(created[-1], final)
    first, last = fields["reward_first"], fields["reward_last"]
    # Running to the end does not show learning, so the reward is judged separately.
    improved = first is not None and last is not None and last > first
    stages.record(
        "learn",
        status="ok" if improved else "ran_without_improvement",
        seconds=seconds, envs=args.learn_envs, requested_iterations=args.learn_iterations,
        checkpoint=str(final), source=created[-1].name, reward_improved=improved, **fields,
    )
    print(f"Result: {stages.path}", flush=True)
    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
