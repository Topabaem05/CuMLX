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
    # It never starts the guest script: that line is only logged, so that a test cannot run the real work on this machine.
    stub("ssh", f'for last; do :; done; case "$last" in *m6_vm_guest.sh*nohup*|*nohup*m6_vm_guest.sh*) echo started; exit 0 ;; esac; HOME="{guest_home}" /bin/sh -c "$last"')
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


def test_attach_needs_an_address_and_a_login(box):
    assert run(box, "attach").returncode == 2
    assert run(box, "attach", "not-an-ip; rm").returncode == 2
    box.stub("ssh", "exit 255")
    done = run(box, "attach", "192.0.2.9")
    assert done.returncode == 3 and "Remote Login" in done.stderr


def test_attach_refuses_low_disk_high_memory_and_an_old_verdict(box):
    (box.tmp / "free_kb").write_text(str(29_000_000))
    assert run(box, "attach", "192.0.2.9").returncode == 3
    (box.tmp / "free_kb").write_text(str(100_000_000))
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(38.5))
    assert run(box, "attach", "192.0.2.9").returncode == 3
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(20.0))
    (box.root / "result").mkdir()
    (box.root / "result" / "verdict.txt").write_text("fail\n")
    assert run(box, "attach", "192.0.2.9").returncode == 4


@pytest.mark.parametrize("answer", ["pass", "fail"])
def test_attach_runs_no_tart_and_reports_what_the_guest_reports(box, answer):
    repo_with_git(box)
    (box.tmp / "host_home").mkdir()
    (box.guest_home / "repro-result" / "guest_verdict.txt").write_text(answer + "\n")
    box.stub("tm", "exit 0")
    done = run(box, "--work-attach", "192.0.2.9", CUMLX_VM_TERMINAL_MAC=str(box.tmp / "stubs" / "tm"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (box.root / "result" / "verdict.txt").read_text().strip() == answer
    calls = box.log.read_text()
    assert "tart " not in calls and "tm stop default" in calls


def test_attach_stops_the_vm_at_the_memory_limit_and_is_not_a_pass(box):
    repo_with_git(box)
    (box.tmp / "host_home").mkdir()
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(38.5))
    (box.guest_home / "repro-result" / "guest_verdict.txt").write_text("pass\n")
    box.stub("tm", "exit 0")
    done = run(box, "--work-attach", "192.0.2.9", CUMLX_VM_TERMINAL_MAC=str(box.tmp / "stubs" / "tm"))
    assert done.returncode == 1
    assert (box.root / "result" / "verdict.txt").read_text().strip() == "stopped_memory"
    assert "tm stop default" in box.log.read_text()


def test_a_test_run_never_starts_the_guest_script(box):
    repo_with_git(box)
    (box.tmp / "host_home").mkdir()
    (box.guest_home / "repro-result" / "guest_verdict.txt").write_text("pass\n")
    assert run(box, "--work").returncode == 0
    assert (box.guest_home / "m6_vm_guest.sh").exists()  # It was copied.
    assert not (box.guest_home / "repro-result" / "stage.txt").exists()  # It did not run.
    assert "m6_vm_guest.sh" in box.log.read_text()


def guest_box(box, in_vm):
    """Each tool that the guest script would use is a stub, so the script can run here, and the log shows how far it got."""
    box.stub("sysctl", f'echo {1 if in_vm else 0}')
    for tool in ("sudo", "diskutil", "softwareupdate", "git", "curl", "sw_vers", "system_profiler", "xcode-select", "touch"):
        box.stub(tool, "exit 0")
    (box.tmp / "home").mkdir(exist_ok=True)
    return subprocess.run(["sh", str(GUEST), "abc"], env={**box.env, "HOME": str(box.tmp / "home")}, capture_output=True, text=True, timeout=30, check=False)


def test_the_guest_script_does_nothing_outside_a_vm(box):
    done = guest_box(box, in_vm=False)
    assert done.returncode == 3 and "not a virtual machine" in done.stderr
    assert not (box.tmp / "home" / "repro-result").exists()
    calls = box.log.read_text()
    assert "sudo" not in calls and "curl" not in calls and "git" not in calls and "diskutil" not in calls


def test_the_guest_script_goes_on_inside_a_vm(box):
    done = guest_box(box, in_vm=True)
    assert done.returncode == 0, done.stderr
    assert (box.tmp / "home" / "repro-result" / "machine.txt").exists()
    # The stub curl installs no uv, so it stops there with a verdict that is not a pass.
    assert (box.tmp / "home" / "repro-result" / "guest_verdict.txt").read_text().strip() == "setup_error_uv"
