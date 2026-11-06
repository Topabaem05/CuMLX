"""spikes/m6_finish.sh with stub launchers and a stub Python, given by its environment variables. Nothing real starts."""

import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "spikes" / "m6_finish.sh"


def stub(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture
def box(tmp_path):
    log = tmp_path / "calls.log"
    repro, cuda = tmp_path / "repro", tmp_path / "cuda"
    same_mac = stub(tmp_path / "same_mac.sh", f'echo "same_mac $* golden=$CUMLX_F1_GOLDEN wait=$CUMLX_F1_WAIT_SECONDS" >> "{log}"\n'
                                              f'mkdir -p "{repro}/result" && echo mac_done > "{repro}/result/verdict.txt"\n')
    v100 = stub(tmp_path / "v100.sh", f'echo "v100 $*" >> "{log}"\n'
                                      f'if [ "$1" = start ]; then n=$(grep -c "v100 start" "{log}"); [ "$n" -ge 2 ] && exit 0; exit 3; fi\n'
                                      f'if [ "$1" = fetch ]; then mkdir -p "{cuda}/f1"; fi\nexit 0\n')
    python = stub(tmp_path / "python", f'echo "judge $1" >> "{log}"\nexit "${{JUDGE_EXIT:-0}}"\n')
    env = {**os.environ, "CUMLX_FINISH_ROOT": str(tmp_path / "finish"), "CUMLX_M6_CUDA": str(cuda), "CUMLX_REPRO_ROOT": str(repro), "CUMLX_FINISH_POLL": "1",
           "CUMLX_FINISH_SAME_MAC": str(same_mac), "CUMLX_FINISH_V100": str(v100), "CUMLX_FINISH_PYTHON": str(python)}
    (tmp_path / "finish").mkdir()
    return tmp_path, env, log


def run(env, **extra):
    return subprocess.run(["sh", str(SCRIPT), "--work", "abc123"], env={**env, **extra}, capture_output=True, text=True, timeout=120, check=False)


def test_it_retries_a_busy_gpu_runs_both_sides_and_judges_five(box):
    tmp, env, log = box
    done = run(env)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == f"same_mac start abc123 m6 golden={tmp / 'cuda'}/f1/golden wait=108000"
    assert [c for c in calls if c.startswith("v100")] == ["v100 start f1 f2 f3 f5", "v100 start f1 f2 f3 f5", "v100 status", "v100 fetch"]
    assert [c.split("/")[-1] for c in calls if c.startswith("judge")] == ["m6_f1_judge.py", "m6_f2_cuda_judge.py", "m6_f3_judge.py", "m6_f4_judge.py", "m6_f5_judge.py"]
    summary = (tmp / "finish" / "summary.txt").read_text()
    assert "all: pass" in summary and summary.count(": pass") == 6


def test_a_run_that_ended_before_the_launcher_looked_is_still_fetched(box):
    tmp, env, log = box
    v100 = stub(tmp / "v100.sh", f'echo "v100 $*" >> "{log}"\n[ "$1" = start ] && exit 1\n[ "$1" = fetch ] && mkdir -p "{tmp / 'cuda'}/f1"\nexit 0\n')
    assert run({**env, "CUMLX_FINISH_V100": str(v100)}).returncode == 0
    assert "v100 fetch" in log.read_text()


def test_a_run_that_died_is_not_waited_for_forever(box):
    tmp, env, log = box
    v100 = stub(tmp / "v100.sh", f'echo "v100 $*" >> "{log}"\n[ "$1" = start ] && exit 0\n[ "$1" = status ] && {{ echo "not running"; exit 5; }}\nexit 0\n')
    done = run({**env, "CUMLX_FINISH_V100": str(v100)})
    assert done.returncode == 0
    assert "v100 fetch" not in log.read_text() and "ended without finishing" in done.stdout


def test_one_failed_judge_is_not_a_pass(box):
    tmp, env, _ = box
    assert run(env, JUDGE_EXIT="1").returncode == 0
    summary = (tmp / "finish" / "summary.txt").read_text()
    assert "all: not a pass" in summary and "f1: fail" in summary
    status = subprocess.run(["sh", str(SCRIPT), "status"], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert status.returncode == 1


def test_a_missing_judge_output_is_not_run(box):
    tmp, env, _ = box
    assert run(env, JUDGE_EXIT="5").returncode == 0
    assert "f3: not_run" in (tmp / "finish" / "summary.txt").read_text()


def test_start_refuses_earlier_cuda_results(box):
    tmp, env, _ = box
    (tmp / "cuda").mkdir()
    started = subprocess.run(["sh", str(SCRIPT), "start"], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert started.returncode in (2, 4) and "Started" not in started.stdout
