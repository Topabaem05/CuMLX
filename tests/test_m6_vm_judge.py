"""spikes/m6_vm_judge.py on made-up output folders: a right run, each wrong one, and each part that did not run."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spikes"))
import m6_vm_judge as j  # noqa: E402


def good(out: Path) -> Path:
    for name in ("doctor", "selftest", "tests", "f3", "f4_train", "f5"):
        (out / f"{name}.exit").write_text("0\n")
    (out / "tests.txt").write_text("....\n774 passed, 46 skipped in 70.0s\n")
    (out / "f3.txt").write_text("....\n20 passed in 9.0s\n")
    for name, speed in (("fwd", 0.81), ("back", -0.86), ("stop", 0.0), ("zero", 0.0)):
        (out / f"f4_{name}.json").write_text(json.dumps({"mean_forward_speed": speed, "falls_per_world": 0.0}))
    (out / "f5").mkdir()
    (out / "f5" / "summary.json").write_text(json.dumps({"status": "rendered"}))
    return out


def test_a_right_run_passes(tmp_path):
    assert j.judge(good(tmp_path))["verdict"] == "pass"


@pytest.mark.parametrize("change, broken", [
    (lambda o: (o / "selftest.exit").write_text("1"), "base"),
    (lambda o: (o / "tests.txt").write_text("3 passed in 1s"), "base"),
    (lambda o: (o / "f3.txt").write_text("19 passed, 1 skipped in 1s"), "f3"),
    (lambda o: (o / "f3.exit").write_text("1"), "f3"),
    (lambda o: (o / "f4_fwd.json").write_text(json.dumps({"mean_forward_speed": 0.5, "falls_per_world": 0.0})), "f4"),
    (lambda o: (o / "f4_back.json").write_text(json.dumps({"mean_forward_speed": -0.8, "falls_per_world": 0.5})), "f4"),
    (lambda o: (o / "f4_zero.json").write_text(json.dumps({"mean_forward_speed": 0.5, "falls_per_world": 0.0})), "f4"),
    (lambda o: (o / "f5" / "summary.json").write_text(json.dumps({"status": "failed"})), "f5"),
])
def test_each_wrong_run_fails_in_its_own_part(tmp_path, change, broken):
    good(tmp_path)
    change(tmp_path)
    result = j.judge(tmp_path)
    assert result["verdict"] == "fail" and result["parts"][broken]["state"] == "fail"


@pytest.mark.parametrize("gone, part", [("doctor.exit", "base"), ("f3.exit", "f3"), ("f4_stop.json", "f4"), ("f4_train.exit", "f4"), ("f5/summary.json", "f5")])
def test_a_part_with_no_output_is_not_a_pass(tmp_path, gone, part):
    good(tmp_path)
    (tmp_path / gone).unlink()
    result = j.judge(tmp_path)
    assert result["verdict"] == "not_run" and result["parts"][part]["state"] == "not_run"


def test_an_empty_folder_is_not_a_pass(tmp_path):
    assert j.judge(tmp_path)["verdict"] == "not_run"


def test_the_verdict_says_what_it_does_not_cover(tmp_path):
    result = j.judge(good(tmp_path))
    assert "paravirtual" in result["scope"] and any("F1" in item for item in result["does_not_cover"])


def test_the_same_mac_rules_name_their_own_scope(tmp_path):
    limits = json.loads((Path(__file__).resolve().parent.parent / "validation" / "m6-same-mac-limits.json").read_text())
    result = j.judge(good(tmp_path), limits)
    assert result["verdict"] == "pass"
    assert "not a run on a new Mac" in result["scope"]
