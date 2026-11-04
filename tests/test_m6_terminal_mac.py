"""spikes/m6_terminal_mac.sh on stubs. Nothing third-party runs: git, df, vm_stat and terminal-mac are small scripts here, and each test reads the calls that they logged."""

from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "spikes" / "m6_terminal_mac.sh"
COMMIT = "7009117f9e910c3dd4b4bd863e8bc331bd3c9d96"


def vm_stat_text(used_gb):
    pages = int(used_gb * 1e9 / 16384)
    return f"Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages active:   {pages}.\nPages wired down:  0.\nPages occupied by compressor:  0.\n"


@pytest.fixture
def box(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    root = tmp_path / "vm"
    root.mkdir()
    log = tmp_path / "calls.log"

    def stub(name, body):
        path = stubs / name
        path.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n{body}\n')
        path.chmod(0o755)

    stub("uname", 'echo "Darwin arm64"')
    stub("df", f'echo header; echo "disk 1 1 $(cat "{tmp_path}/free_kb")"')
    stub("vm_stat", f'cat "{tmp_path}/vm_stat.txt"')
    # A stub clone copies this folder. Its build script installs a stub terminal-mac that logs its calls.
    fake = tmp_path / "fake_src"
    (fake / "scripts").mkdir(parents=True)
    (fake / "tm").write_text(f'#!/bin/sh\necho "tm $*" >> "{log}"\ncase "$1" in status) echo stopped ;; esac\nexit 0\n')
    (fake / "scripts" / "build.sh").write_text("mkdir -p dist\ncp tm dist/terminal-mac\nchmod +x dist/terminal-mac\n")
    stub("git", f'case "$*" in *clone*) for last; do :; done; cp -R "{fake}" "$last" ;; *rev-parse*) cat "{tmp_path}/head" ;; esac')
    (tmp_path / "head").write_text(COMMIT)
    (tmp_path / "free_kb").write_text(str(100_000_000))
    (tmp_path / "vm_stat.txt").write_text(vm_stat_text(20.0))
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path / "home"), "CUMLX_VM_ROOT": str(root), "CUMLX_VM_POLL": "0"}

    class Box:
        pass

    b = Box()
    b.env, b.root, b.log, b.tmp, b.stub = env, root, log, tmp_path, stub
    return b


def run(box, *args, **env):
    return subprocess.run(["sh", str(SCRIPT), *args], env={**box.env, **env}, capture_output=True, text=True, timeout=60, check=False)


def test_no_mode_is_a_usage_error(box):
    assert run(box).returncode == 2
    assert run(box, "nonsense").returncode == 2


def test_status_with_no_run_is_not_a_pass(box):
    assert run(box, "status").returncode == 5


def test_another_machine_is_refused(box):
    box.stub("uname", 'echo "Linux x86_64"')
    assert run(box, "start").returncode == 3


def test_low_disk_is_refused_before_any_download(box):
    (box.tmp / "free_kb").write_text(str(64_000_000))
    done = run(box, "start")
    assert done.returncode == 3 and "65 GB" in done.stderr
    assert "git" not in box.log.read_text()


def test_memory_at_the_start_counts_the_vm(box):
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(33.0))  # 4 GiB is 4.3e9 bytes, against 37 GB.
    done = run(box, "start")
    assert done.returncode == 3 and "37 GB" in done.stderr
    assert "git" not in box.log.read_text()


def test_the_right_commit_is_built_and_the_window_opens(box):
    done = run(box, "--work")
    assert done.returncode == 0, done.stdout + done.stderr
    calls = box.log.read_text()
    assert f"checkout -q {COMMIT}" in calls
    assert "tm up default --detach --cpus 8 --memory 4 --disk 80" in calls and "tm show default" in calls
    assert (box.root / "setup_state.txt").read_text().strip() == "window_open"
    assert run(box, "status").returncode == 0


def test_another_commit_builds_nothing(box):
    (box.tmp / "head").write_text("0" * 40)
    assert run(box, "--work").returncode == 1
    assert (box.root / "setup_state.txt").read_text().strip() == "error_wrong_commit"
    assert "tm " not in box.log.read_text()
    assert run(box, "status").returncode == 1


def test_a_failed_up_is_not_a_success(box):
    run(box, "--work")
    tm = box.root / "terminal-mac-src" / "dist" / "terminal-mac"
    tm.write_text(f'#!/bin/sh\necho "tm $*" >> "{box.log}"\ncase "$1" in up) exit 1 ;; esac\nexit 0\n')
    assert run(box, "--work").returncode == 1
    assert (box.root / "setup_state.txt").read_text().strip() == "error_up"


def test_the_memory_limit_stops_the_vm(box):
    tm_running = 'case "$1" in status) echo \'"state": "running"\' ;; esac; exit 0'
    run(box, "--work")
    tm = box.root / "terminal-mac-src" / "dist" / "terminal-mac"
    tm.write_text(f'#!/bin/sh\necho "tm $*" >> "{box.log}"\n{tm_running}\n')
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(38.5))
    assert run(box, "--work").returncode == 1
    assert (box.root / "setup_state.txt").read_text().strip().startswith("stopped_memory")
    assert "tm stop default" in box.log.read_text()


@pytest.mark.parametrize("over", ["memory", "disk"])
def test_the_limits_hold_during_the_install(box, over):
    run(box, "--work")
    tm = box.root / "terminal-mac-src" / "dist" / "terminal-mac"
    tm.write_text(f'#!/bin/sh\necho "tm $*" >> "{box.log}"\ncase "$1" in up) exec sleep 50 ;; esac\nexit 0\n')
    if over == "memory":
        (box.tmp / "vm_stat.txt").write_text(vm_stat_text(38.5))
    else:
        (box.tmp / "free_kb").write_text(str(15_000_000))
    assert run(box, "--work").returncode == 1  # The timeout of 60 seconds would raise if it waited for the install.
    assert (box.root / "setup_state.txt").read_text().strip().startswith(f"stopped_{over}")
    assert "tm stop default --force" in box.log.read_text()
