"""spikes/m6_same_mac.sh on stubs. Nothing third-party runs: uv, git, curl and the run's Python are small scripts here, and a test proves that they are the ones called."""

import json
from pathlib import Path
import re
import subprocess
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "spikes" / "m6_same_mac.sh"
GUEST = REPO / "spikes" / "m6_vm_guest.sh"


def vm_stat_text(used_gb: float) -> str:
    pages = int(used_gb * 1e9 / 16384)
    return f"Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages active:   {pages}.\nPages wired down:  0.\nPages occupied by compressor:  0.\n"


@pytest.fixture
def box(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    log = tmp_path / "calls.log"

    def stub(name, body):
        path = stubs / name
        path.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n{body}\n')
        path.chmod(0o755)

    # The run's Python: nothing real runs. Files name what it does for the training and for the judge.
    fake_py = tmp_path / "fake_py"
    fake_py.write_text(f'''#!/bin/sh
echo "py $*" >> "{log}"
case "$*" in
*"m4_sim.py train"*) [ -f "{tmp_path}/hang" ] && {{ echo running > "{tmp_path}/hanging"; exec sleep 300; }} ;;
*m6_vm_judge.py*) exit "$(cat "{tmp_path}/judge_exit")" ;;
esac
exit 0
''')
    fake_py.chmod(0o755)
    stub("uname", 'case "$*" in -sm) echo "Darwin arm64" ;; *) echo arm64 ;; esac')
    stub("df", f'echo header; echo "disk 1 1 $(cat "{tmp_path}/free_kb")"')
    stub("vm_stat", f'cat "{tmp_path}/vm_stat.txt"')
    stub("caffeinate", "exit 0")
    stub("sudo", "exit 1")
    stub("sw_vers", "echo 27.0")
    stub("sysctl", "echo 1")
    stub("system_profiler", "echo gpu")
    stub("curl", f'case "$*" in *install.sh*) echo "exit 0" ;; esac; [ -f "{tmp_path}/no_internet" ] && exit 7; exit 0')
    stub("git", f'case "$1" in fetch) [ -f "{tmp_path}/not_public" ] && exit 1 ;; esac; exit 0')
    stub("uv", f'case "$1" in venv) mkdir -p .venv/bin; cp "{fake_py}" .venv/bin/python; cp "{fake_py}" .venv/bin/cumlx ;; esac; exit 0')
    (tmp_path / "free_kb").write_text(str(100_000_000))
    (tmp_path / "vm_stat.txt").write_text(vm_stat_text(20.0))
    (tmp_path / "judge_exit").write_text("0")
    (tmp_path / "person").mkdir()
    root = tmp_path / "repro"
    path = f"{stubs}:/usr/bin:/bin"
    env = {"PATH": path, "HOME": str(tmp_path / "person"), "CUMLX_REPRO_ROOT": str(root), "CUMLX_REPRO_PATH": path, "CUMLX_REPRO_POLL": "0.1", "CUMLX_REPRO_ROOM_WAIT_SECONDS": "0"}

    class Box:
        pass

    b = Box()
    b.env, b.root, b.result, b.log, b.tmp, b.stub = env, root, root / "result", log, tmp_path, stub
    yield b
    if (root / "run.pid").exists():
        run(b, "stop")


def run(box, *args, **env):
    return subprocess.run(["sh", str(SCRIPT), *args], env={**box.env, **env}, capture_output=True, text=True, timeout=60, check=False)


def wait_for(condition, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.05)
    return False


def verdict(box):
    done = wait_for(lambda: (box.result / "verdict.txt").exists() and "finished" in (box.result / "stage.txt").read_text())
    assert done, (box.root / "run.log").read_text()
    return (box.result / "verdict.txt").read_text().strip()


def ended(box):
    return wait_for(lambda: run(box, "status").stdout.startswith("not running"))


def test_the_stubs_are_the_ones_called(box):
    found = subprocess.run(["sh", "-c", "command -v uv; command -v git; command -v curl"], env=box.env, capture_output=True, text=True, check=False).stdout
    assert found.splitlines() == [str(box.tmp / "stubs" / name) for name in ("uv", "git", "curl")]


def test_no_mode_is_a_usage_error(box):
    assert run(box).returncode == 2
    assert run(box, "nonsense").returncode == 2


def test_status_with_no_run_is_not_a_pass(box):
    assert run(box, "status").returncode == 5


@pytest.mark.parametrize("judge_exit, expected, code", [(0, "pass", 0), (1, "fail", 1), (5, "not_run_exit_5", 1)])
def test_the_verdict_comes_from_the_judge(box, judge_exit, expected, code):
    (box.tmp / "judge_exit").write_text(str(judge_exit))
    assert run(box, "start", "abc123").returncode == 0
    assert verdict(box) == expected
    assert run(box, "status").returncode == code


def test_the_run_uses_no_sudo_and_stays_in_its_own_home(box):
    assert run(box, "start", "abc123").returncode == 0
    verdict(box)
    calls = box.log.read_text()
    assert "sudo " not in calls
    assert "sudo" not in "\n".join(line for line in SCRIPT.read_text().splitlines() if not line.startswith("#"))
    assert list((box.tmp / "person").iterdir()) == []
    assert "fetch -q --depth 1 origin abc123" in calls
    assert "--limits validation/m6-same-mac-limits.json" in calls


@pytest.mark.parametrize("used, free, word", [(34.0, 100_000_000, "memory"), (20.0, 20_000_000, "free")])
def test_a_busy_or_full_mac_does_not_start(box, used, free, word):
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(used))
    (box.tmp / "free_kb").write_text(str(free))
    result = run(box, "start", "abc123")
    assert result.returncode == 3 and word in result.stderr
    assert not (box.root / "run.pid").exists()


def test_memory_at_the_limit_stops_the_run_and_is_not_a_pass(box):
    (box.tmp / "hang").write_text("")
    assert run(box, "start", "abc123").returncode == 0
    assert wait_for(lambda: (box.tmp / "hanging").exists()), (box.root / "run.log").read_text()
    (box.tmp / "vm_stat.txt").write_text(vm_stat_text(38.5))
    assert wait_for(lambda: (box.result / "verdict.txt").exists())
    assert (box.result / "verdict.txt").read_text().startswith("stopped_memory_")
    pid = (box.root / "run.pid").read_text().strip()
    assert wait_for(lambda: subprocess.run(["sh", "-c", f"kill -0 -{pid}"], capture_output=True, check=False).returncode != 0)
    assert run(box, "status").returncode == 1


def test_a_heavy_part_that_finds_no_room_does_not_start(box):
    (box.tmp / "high.txt").write_text(vm_stat_text(34.0))
    (box.tmp / "low.txt").write_text(vm_stat_text(20.0))
    # The start reads low. The first gate before a heavy part reads high, because of the marker that the stub for pytest's stage leaves.
    box.stub("system_profiler", f'cp "{box.tmp}/high.txt" "{box.tmp}/vm_stat.txt"')
    assert run(box, "start", "abc123").returncode == 0
    assert verdict(box) == "stopped_memory_before_f3"
    assert "test_kernel_gpu" not in box.log.read_text()


def test_the_default_folder_works_too(box):
    env = {key: value for key, value in box.env.items() if key != "CUMLX_REPRO_ROOT"}
    started = subprocess.run(["sh", str(SCRIPT), "start", "abc123"], env=env, capture_output=True, text=True, timeout=60, check=False)
    assert started.returncode == 0, started.stderr
    result = box.tmp / "person" / "cumlx-repro" / "result"
    try:
        assert wait_for(lambda: (result / "verdict.txt").exists()), (result.parent / "run.log").read_text()
        assert (result / "verdict.txt").read_text().strip() == "pass"
        assert not (result.parent / "home" / "cumlx-repro").exists()
    finally:
        subprocess.run(["sh", str(SCRIPT), "stop"], env=env, capture_output=True, timeout=60, check=False)


def test_the_f1_f2_part_replaces_the_rest_and_a_missing_verdict_is_not_a_pass(box):
    # The stub git leaves no spikes/m6_f1_f2_work.sh, so the part ends with no verdict file.
    assert run(box, "start", "abc123", "f1f2").returncode == 0
    assert verdict(box) == "not_run_no_verdict"
    assert "test_kernel_gpu" not in box.log.read_text()


def test_a_commit_that_is_not_public_is_a_setup_error(box):
    (box.tmp / "not_public").write_text("")
    assert run(box, "start", "abc123").returncode == 0
    assert verdict(box) == "setup_error_public_commit"


def test_no_internet_is_named(box):
    (box.tmp / "no_internet").write_text("")
    assert run(box, "start", "abc123").returncode == 0
    assert verdict(box) == "setup_error_no_internet"


def test_a_new_run_does_not_read_an_earlier_verdict(box):
    assert run(box, "start", "abc123").returncode == 0
    assert verdict(box) == "pass"
    assert ended(box)
    (box.tmp / "not_public").write_text("")
    assert run(box, "start", "abc123").returncode == 0
    assert verdict(box) == "setup_error_public_commit"
    assert [p for p in box.root.iterdir() if p.name.startswith("result-")], "the earlier results stay"


def test_a_folder_of_other_files_is_left_alone(box):
    box.root.mkdir()
    (box.root / "theirs.txt").write_text("keep")
    assert run(box, "start", "abc123").returncode == 4
    assert run(box, "clean").returncode == 0
    assert (box.root / "theirs.txt").read_text() == "keep"


def test_clean_deletes_the_home_and_keeps_the_results(box):
    assert run(box, "start", "abc123").returncode == 0
    verdict(box)
    assert ended(box)
    assert run(box, "clean").returncode == 0
    assert not (box.root / "home").exists()
    assert (box.result / "verdict.txt").exists()


def work_lines(path: Path) -> list[str]:
    """The lines that do the work, in order."""
    keep = (".venv/bin/cumlx", '"$PY"', "uv venv", "uv pip", "git fetch", "curl -LsSf")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip().startswith(keep) and "m6_vm_judge" not in line]
    return [line.replace("| UV_NO_MODIFY_PATH=1 sh", "| sh") for line in lines]


def test_the_work_is_the_same_as_in_the_vm():
    assert work_lines(SCRIPT) == work_lines(GUEST)
    assert len(work_lines(SCRIPT)) >= 12


def m4_flags(command: str) -> set[str]:
    """The flags that spikes/m4_sim.py accepts for one command, from its source, so that this needs none of its packages."""
    source = (REPO / "spikes" / "m4_sim.py").read_text()
    common = source[source.index("def add(name"):source.index("sub = add('smoke'")]
    start = source.index(f"sub = add('{command}'")
    own = source[start:source.index("sub = add(", start + 1)] if "sub = add(" in source[start + 1:] else source[start:]
    return set(re.findall(r"add_argument\('(--[\w-]+)'", common + own))


@pytest.mark.parametrize("script", [SCRIPT, GUEST], ids=["same_mac", "vm"])
def test_each_m4_sim_line_uses_only_flags_that_exist(script):
    """The first run on a real GPU ended here: four lines passed a flag that spikes/m4_sim.py never had, and no earlier run had reached them."""
    lines = [line for line in script.read_text().splitlines() if "spikes/m4_sim.py" in line]
    assert len(lines) == 5
    for line in lines:
        command = line.split("spikes/m4_sim.py", 1)[1].split()[0]
        assert set(re.findall(r"(?<=\s)--[\w-]+", line)) <= m4_flags(command), line


def test_the_flag_check_can_fail():
    assert "--checkpoint" in m4_flags("eval")
    assert "--controller" not in m4_flags("eval")


def test_the_rules_are_the_same_as_in_the_vm():
    vm = json.loads((REPO / "validation" / "m6-vm-limits.json").read_text())
    here = json.loads((REPO / "validation" / "m6-same-mac-limits.json").read_text())
    for part in ("covers", "base", "f3", "f4", "f5"):
        assert here[part] == vm[part], part
    assert "not a run on a new Mac" in here["scope"]
    assert any("does not tick M6-1" in line for line in here["does_not_cover"])


def test_the_record_keeps_m6_1_open_and_names_no_private_path():
    text = (REPO / "validation" / "m6-same-mac-run.json").read_text()
    record = json.loads(text)
    assert record["verdict"] == "pass" and record["ticks_m6_1"] is False
    assert set(record["parts"]) == {"base", "f3", "f4", "f5"}
    assert all(part["state"] == "pass" for part in record["parts"].values())
    assert re.fullmatch(r"[0-9a-f]{40}", record["cumlx_commit"])
    assert "/Users/" not in text and "local" not in text.split('"what_this_shows"')[0]
