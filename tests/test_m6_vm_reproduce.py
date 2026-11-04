"""spikes/m6_vm_reproduce.sh on stubs. Nothing third-party runs: tart, ssh, df and vm_stat are small scripts here, and each test proves that they were the ones called."""

import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "spikes" / "m6_vm_reproduce.sh"
GUEST = Path(__file__).resolve().parents[1] / "spikes" / "m6_vm_guest.sh"


def vm_stat_text(used_gb: float) -> str:
    pages = int(used_gb * 1e9 / 16384)
    return f"Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages active:   {pages}.\nPages wired down:  0.\nPages occupied by compressor:  0.\n"


@pytest.fixture
def box(tmp_path):
    """A folder of stubs, first on PATH. Each stub appends its arguments to calls.log."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    root = tmp_path / "vm"
    root.mkdir()
    log = tmp_path / "calls.log"

    def stub(name, body):
        path = stubs / name
        path.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n{body}\n')
        path.chmod(0o755)

    guest_home = tmp_path / "guest_home"
    (guest_home / "repro-result").mkdir(parents=True)
    stub("uname", 'echo "Darwin arm64"')
    stub("df", f'echo header; echo "disk 1 1 $(cat "{tmp_path}/free_kb")"')
    stub("vm_stat", f'cat "{tmp_path}/vm_stat.txt"')
    stub("tart", 'case "$1" in ip) echo 192.0.2.1 ;; --version) echo stub ;; run) sleep 30 ;; esac; exit 0')
    # A stub ssh runs the remote text in a shell, with HOME set to the guest's folder.
    stub("ssh", f'shift_to=""; for last; do :; done; HOME="{guest_home}" /bin/sh -c "$last"')
    (tmp_path / "free_kb").write_text(str(100_000_000))
    (tmp_path / "vm_stat.txt").write_text(vm_stat_text(20.0))
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path / "host_home"), "CUMLX_VM_ROOT": str(root), "CUMLX_VM_TART": str(stubs / "tart"), "CUMLX_VM_POLL": "0"}

    class Box:
        pass

    b = Box()
    b.env, b.root, b.log, b.guest_home, b.tmp, b.stub = env, root, log, guest_home, tmp_path, stub
    return b


def run(box, *args, timeout=60, **env):
    return subprocess.run(["sh", str(SCRIPT), *args], env={**box.env, **env}, capture_output=True, text=True, timeout=timeout, check=False)


def test_the_stubs_are_the_ones_called(box):
    assert subprocess.run(["sh", "-c", "command -v tart"], env=box.env, capture_output=True, text=True, check=False).stdout.startswith(str(box.tmp))


def test_no_mode_is_a_usage_error(box):
    assert run(box).returncode == 2
    assert run(box, "nonsense").returncode == 2


def test_status_with_no_run_is_not_a_pass(box):
    assert run(box, "status").returncode == 5


@pytest.mark.parametrize("verdict, code", [("pass", 0), ("fail", 1), ("stopped_memory", 1), ("stopped_disk", 1), ("setup_error_tart", 1)])
def test_status_passes_only_on_a_reproduced_pass(box, verdict, code):
    (box.root / "result").mkdir()
    (box.root / "result" / "verdict.txt").write_text(verdict + "\n")
    assert run(box, "status").returncode == code


def test_another_machine_is_refused(box):
    box.stub("uname", 'echo "Linux x86_64"')
    assert run(box, "start").returncode == 3
    assert "tart" not in box.log.read_text().replace("uname", "")


def test_low_disk_is_refused_before_any_download(box):
    (box.tmp / "free_kb").write_text(str(64_000_000))
    done = run(box, "start")
    assert done.returncode == 3 and "65 GB" in done.stderr
    assert "tart" not in box.log.read_text()


@pytest.mark.parametrize("used, code", [(29.0, 0), (31.0, 3)])
def test_memory_at_the_start_counts_the_vm(box, used, code):
    """6 GiB is 6.4e9 bytes, against 37 GB at the start."""
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(used))
    box.stub("sh", 'exit 0')  # Keeps the background work from starting in the case that passes.
    done = subprocess.run(["/bin/sh", str(SCRIPT), "start"], env=box.env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == code, done.stderr


def test_a_second_run_never_overwrites_a_verdict(box):
    (box.root / "result").mkdir()
    (box.root / "result" / "verdict.txt").write_text("fail\n")
    assert run(box, "start").returncode == 4


def test_a_wrong_tart_download_stops_the_run(box):
    box.stub("curl", 'while [ $# -gt 1 ]; do [ "$1" = "-o" ] && out=$2; shift; done; echo wrong > "$out"')
    env = dict(box.env)
    del env["CUMLX_VM_TART"]
    done = subprocess.run(["sh", str(SCRIPT), "--work"], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 1 and "SHA-256" in done.stdout
    assert (box.root / "result" / "verdict.txt").read_text().strip() == "setup_error_tart"
    assert "tart " not in box.log.read_text().replace("tart.tar.gz", "")


def repo_with_git(box):
    """The script archives its own repository, so a stub git keeps this test away from the real one."""
    box.stub("git", 'echo abc')


@pytest.mark.parametrize("answer", ["pass", "fail"])
def test_the_run_reports_what_the_guest_reports(box, answer):
    repo_with_git(box)
    (box.tmp / "host_home").mkdir()
    (box.guest_home / "repro-result" / "guest_verdict.txt").write_text(answer + "\n")
    (box.guest_home / "repro-result" / "machine.txt").write_text("a machine\n")
    done = run(box, "--work")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (box.root / "result" / "verdict.txt").read_text().strip() == answer
    assert (box.root / "result" / "guest" / "repro-result" / "machine.txt").exists()
    calls = box.log.read_text()
    assert "tart pull" in calls and "tart clone" in calls and "tart run --no-graphics cumlx-repro" in calls and "tart stop cumlx-repro" in calls
    assert "--memory 6144" in calls
    assert run(box, "status").returncode == (0 if answer == "pass" else 1)


def test_memory_over_the_limit_stops_the_vm_and_is_not_a_pass(box):
    repo_with_git(box)
    (box.tmp / "host_home").mkdir()
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(38.5))
    # The guest would report a pass, and the limit comes first.
    (box.guest_home / "repro-result" / "guest_verdict.txt").write_text("pass\n")
    done = run(box, "--work")
    assert done.returncode == 1
    assert (box.root / "result" / "verdict.txt").read_text().strip() == "stopped_memory"
    assert "tart stop cumlx-repro" in box.log.read_text()
    assert run(box, "status").returncode == 1


def test_low_disk_during_the_run_stops_the_vm_and_is_not_a_pass(box):
    repo_with_git(box)
    (box.tmp / "host_home").mkdir()
    (box.tmp / "free_kb").write_text(str(15_000_000))
    (box.guest_home / "repro-result" / "guest_verdict.txt").write_text("pass\n")
    done = run(box, "--work")
    assert done.returncode == 1
    assert (box.root / "result" / "verdict.txt").read_text().strip() == "stopped_disk"


def test_a_vm_that_does_not_boot_is_an_error(box):
    repo_with_git(box)
    box.stub("tart", 'case "$1" in ip) exit 1 ;; run) sleep 30 ;; esac; exit 0')
    done = run(box, "--work", timeout=120, CUMLX_VM_LIMIT_SECONDS="5")
    assert done.returncode == 1
    assert (box.root / "result" / "verdict.txt").read_text().strip() == "setup_error_boot"


def test_clean_refuses_while_a_run_is_going_and_keeps_results(box):
    (box.root / "result").mkdir()
    (box.root / "result" / "verdict.txt").write_text("fail\n")
    (box.root / "tart-home").mkdir()
    (box.root / "run.pid").write_text(str(os.getpid()))
    assert run(box, "clean").returncode == 4
    (box.root / "run.pid").unlink()
    assert run(box, "clean").returncode == 0
    assert not (box.root / "tart-home").exists() and (box.root / "result" / "verdict.txt").exists()


def test_the_guest_script_parses_and_gets_cumlx_from_the_public_repository():
    assert subprocess.run(["sh", "-n", str(GUEST)], check=False).returncode == 0
    text = GUEST.read_text()
    assert "https://github.com/Topabaem05/CuMLX" in text and "m6_vm_judge.py" in text
    assert "finish pass" in text and "finish setup_error_uv" in text
