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


def v100_stub(tmp: Path, log: Path, start: str = 'n=$(grep -c "v100 start" "{log}"); [ "$n" -ge 2 ] && exit 0; exit 3', status: str = "exit 0", commit: str = "abc123") -> Path:
    cuda = tmp / "cuda"
    return stub(tmp / "v100.sh", f'echo "v100 $* commit=$CUMLX_V100_COMMIT" >> "{log}"\n'
                                 f'if [ "$1" = start ]; then {start.format(log=log)}; fi\n'
                                 f'if [ "$1" = status ]; then {status}; fi\n'
                                 f'if [ "$1" = fetch ]; then mkdir -p "{cuda}/f1" && echo {commit} > "{cuda}/cumlx_commit.txt"; fi\nexit 0\n')


@pytest.fixture
def box(tmp_path):
    log = tmp_path / "calls.log"
    repro = tmp_path / "repro"
    same_mac = stub(tmp_path / "same_mac.sh", f'echo "same_mac $* golden=$CUMLX_F1_GOLDEN ready=$CUMLX_F1_READY" >> "{log}"\n'
                                              f'[ "$1" = status ] && {{ [ -f "{repro}/result/verdict.txt" ] && exit 0; echo running; exit 5; }}\n'
                                              f'mkdir -p "{repro}/result" && echo abc123 > "{repro}/result/cumlx_commit.txt" && echo mac_done > "{repro}/result/verdict.txt"\n')
    python = stub(tmp_path / "python", f'echo "judge $1" >> "{log}"\nexit "${{JUDGE_EXIT:-0}}"\n')
    env = {**os.environ, "CUMLX_FINISH_ROOT": str(tmp_path / "finish"), "CUMLX_M6_CUDA": str(tmp_path / "cuda"), "CUMLX_REPRO_ROOT": str(repro),
           "CUMLX_FINISH_POLL": "1", "CUMLX_FINISH_SHORT_POLL": "1", "CUMLX_FINISH_REPO": str(REPO),
           "CUMLX_FINISH_SAME_MAC": str(same_mac), "CUMLX_FINISH_V100": str(v100_stub(tmp_path, log)), "CUMLX_FINISH_PYTHON": str(python)}
    (tmp_path / "finish").mkdir()
    return tmp_path, env, log


def run(env, **extra):
    return subprocess.run(["sh", str(SCRIPT), "--work", "abc123"], env={**env, **extra}, capture_output=True, text=True, timeout=120, check=False)


def summary(tmp: Path) -> str:
    return (tmp / "finish" / "summary.txt").read_text()


def test_it_retries_a_busy_gpu_runs_both_sides_at_one_commit_and_judges_five(box):
    tmp, env, log = box
    done = run(env)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == f"same_mac start abc123 m6 golden={tmp / 'cuda'}/f1/golden ready={tmp / 'finish'}/cuda_ready.txt"
    assert [c for c in calls if c.startswith("v100")] == ["v100 start f1 f2 f3 f5 commit=abc123"] * 2 + ["v100 status commit=", "v100 fetch commit="]
    assert [c.split("/")[-1] for c in calls if c.startswith("judge")] == ["m6_f1_judge.py", "m6_f2_cuda_judge.py", "m6_f3_judge.py", "m6_f4_judge.py", "m6_f5_judge.py"]
    assert (tmp / "finish" / "cuda_ready.txt").read_text().strip() == "fetched"
    assert "all: pass" in summary(tmp) and summary(tmp).count(": pass") == 6


def test_one_failed_judge_is_not_a_pass(box):
    tmp, env, _ = box
    assert run(env, JUDGE_EXIT="1").returncode == 0
    assert "all: not a pass" in summary(tmp) and "f1: fail" in summary(tmp)
    status = subprocess.run(["sh", str(SCRIPT), "status"], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert status.returncode == 1


def test_a_missing_judge_output_is_not_run(box):
    tmp, env, _ = box
    assert run(env, JUDGE_EXIT="5").returncode == 0
    assert "f3: not_run" in summary(tmp)


def test_a_run_that_ended_before_the_launcher_looked_is_still_fetched(box):
    tmp, env, log = box
    v100 = v100_stub(tmp, log, start="exit 1")
    assert run({**env, "CUMLX_FINISH_V100": str(v100)}).returncode == 0
    assert "v100 fetch" in log.read_text() and "all: pass" in summary(tmp)


def test_a_run_that_died_is_not_waited_for_forever_and_the_mac_hears_of_it(box):
    tmp, env, log = box
    v100 = v100_stub(tmp, log, start="exit 0", status='echo "not running"; exit 5')
    done = run({**env, "CUMLX_FINISH_V100": str(v100)})
    assert done.returncode == 0 and "ended without finishing" in done.stdout
    assert "v100 fetch" not in log.read_text()
    assert (tmp / "finish" / "cuda_ready.txt").read_text().strip() == "none"
    assert "cuda: no results" in summary(tmp) and "all: not a pass" in summary(tmp)


def test_results_of_another_commit_are_not_a_pass(box):
    tmp, env, log = box
    v100 = v100_stub(tmp, log, commit="zzz999")
    assert run({**env, "CUMLX_FINISH_V100": str(v100)}).returncode == 0
    assert "cuda: ran zzz999, not abc123" in summary(tmp) and "all: not a pass" in summary(tmp)


def test_a_busy_gpu_at_the_start_of_the_run_is_not_a_pass(box):
    tmp, env, log = box
    v100 = stub(tmp / "v100b.sh", f'[ "$1" = fetch ] && mkdir -p "{tmp / "cuda"}" && echo abc123 > "{tmp / "cuda"}/cumlx_commit.txt" && echo 4242 > "{tmp / "cuda"}/busy_at_start.txt"\nexit 0\n')
    assert run({**env, "CUMLX_FINISH_V100": str(v100)}).returncode == 0
    assert "another process took the GPU" in summary(tmp) and "all: not a pass" in summary(tmp)


def test_a_mac_run_that_cannot_start_is_not_waited_for(box):
    tmp, env, log = box
    same_mac = stub(tmp / "same_mac2.sh", 'exit 4\n')
    done = run({**env, "CUMLX_FINISH_SAME_MAC": str(same_mac)})
    assert done.returncode == 0
    text = summary(tmp)
    assert "mac run: not_started" in text and "f1: not_run" in text and "all: not a pass" in text
    assert "judge" not in log.read_text()


def test_start_refuses_earlier_cuda_results(box):
    tmp, env, _ = box
    (tmp / "cuda").mkdir()
    started = subprocess.run(["sh", str(SCRIPT), "start"], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert started.returncode in (2, 4) and "Started" not in started.stdout


def test_the_v100_launcher_pins_the_commit_fetches_in_one_step_and_checks_the_gpu_again():
    text = (REPO / "spikes" / "m6_cuda_v100.sh").read_text()
    assert 'git -C "$GIT_REPO" archive "$COMMIT"' in text and "CUMLX_V100_COMMIT" in text
    assert 'tar -x -C "$LOCAL.partial"' in text and 'mv "$LOCAL.partial" "$LOCAL"' in text
    assert "busy_at_start.txt" in text
    assert "(cd $REMOTE && exec nohup bash run.sh) > $REMOTE/run.log 2>&1 < /dev/null &" in text


def test_the_f1_part_waits_for_the_ready_marker_not_for_a_file_of_the_golden():
    text = (REPO / "spikes" / "m6_f1_work.sh").read_text()
    assert 'while [ -n "$READY" ] && [ ! -f "$READY" ]' in text and "CUMLX_F1_READY" in text
