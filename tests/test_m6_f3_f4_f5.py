"""The judges of F3, F4 and F5 on made-up output folders, and the pieces of their scripts that need nothing third-party."""

import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spikes"))
import m6_f3_app as app  # noqa: E402
import m6_f3_judge as j3  # noqa: E402
import m6_f4_judge as j4  # noqa: E402
import m6_f5_judge as j5  # noqa: E402


# ----------------------------------------------------------------------------------------------- F3

@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    out = tmp_path_factory.mktemp("ref") / "reference.npz"
    assert app.run_reference(out) == 0
    return j3.read_npz(out)


def test_the_mask_set_has_every_size_and_kind_with_even_sides():
    masks = app.mask_set()
    assert sorted(masks) == sorted(j3.SIZES)
    for size, batch in masks.items():
        h, w = map(int, size.split("x"))
        assert batch.shape == (len(j3.KINDS), 1, h, w) and batch.dtype == np.uint8 and h % 2 == 0 and w % 2 == 0
        assert batch[4].sum() == 0 and batch[5].min() == 1


def test_the_reference_counts_each_component(reference):
    for size, kind, labels, counts in j3.masks_of(reference):
        for value in set(labels.reshape(-1).tolist()) - {0}:
            area = int((labels == value).sum())
            assert set(counts[labels == value].tolist()) == {area}
        assert (counts[labels == 0] == 0).all()


def test_the_launch_geometry_is_that_of_the_host_function():
    assert app.grid_for("merge", 64, 96, None) == (math.ceil(48 / 16) * 16, math.ceil(32 / 16) * 16, 1)
    assert app.grid_for("init_counting", 64, 96, None) == ((96 + 16) // 16 * 16, (64 + 16) // 16 * 16, 1)
    assert app.grid_for("merge", 64, 96, "half_grid")[0] < app.grid_for("merge", 64, 96, None)[0]


def write_f3(cuda: Path, mac: Path, reference: dict) -> None:
    cuda.mkdir()
    mac.mkdir()
    np.savez(cuda / "cuda.npz", **reference)
    np.savez(mac / "mac.npz", **reference)
    np.savez(mac / "reference.npz", **reference)
    for name in ("fetch", "cuda"):
        (cuda / f"{name}.exit").write_text("0\n")
    for name in ("install", "fetch", "mac", "reference", "translate"):
        (mac / f"{name}.exit").write_text("0\n")
    (mac / "connected_components.metal").write_text("kernel")
    (mac / "connected_components.port.json").write_text("{}")
    for fault in j3.FAULTS:
        broken = {k: v.copy() for k, v in reference.items()}
        broken["labels/64x64"][0, 0, 3, 3] += 1
        np.savez(mac / f"fault_{fault}.npz", **broken)
        (mac / f"fault_{fault}.exit").write_text("0\n")


def test_f3_a_right_pair_passes_and_each_break_fails(tmp_path, reference):
    cuda, mac = tmp_path / "cuda", tmp_path / "mac"
    write_f3(cuda, mac, reference)
    result = j3.judge(cuda, mac)
    assert result["verdict"] == "pass", result["parts"]
    relabeled = {k: v.copy() for k, v in reference.items()}
    relabeled["labels/32x32"] = np.where(relabeled["labels/32x32"] > 0, relabeled["labels/32x32"] + 7, 0)
    np.savez(mac / "mac.npz", **relabeled)
    result = j3.judge(cuda, mac)
    assert result["parts"]["cuda_vs_mac"]["state"] == "fail" and result["parts"]["reference"]["state"] == "pass"  # the same partition, other labels
    np.savez(mac / "fault_half_grid.npz", **reference)
    np.savez(mac / "mac.npz", **reference)
    result = j3.judge(cuda, mac)
    assert [k for k, v in result["parts"].items() if v["state"] != "pass"] == ["controls"]


def test_f3_another_partition_fails_the_reference(tmp_path, reference):
    cuda, mac = tmp_path / "cuda", tmp_path / "mac"
    write_f3(cuda, mac, reference)
    merged = {k: v.copy() for k, v in reference.items()}
    lab = merged["labels/64x64"][1, 0]
    values = sorted(set(lab.reshape(-1).tolist()) - {0})
    lab[lab == values[1]] = values[0]
    np.savez(cuda / "cuda.npz", **merged)
    np.savez(mac / "mac.npz", **merged)
    result = j3.judge(cuda, mac)
    assert result["parts"]["reference"]["state"] == "fail"


def test_f3_scripts_run_the_controls_that_the_rules_name():
    work = (REPO / "spikes" / "m6_f3_work.sh").read_text()
    assert sorted(re.search(r"for fault in (.*?); do", work).group(1).split()) == sorted(j3.FAULTS) == sorted(app.FAULTS)
    assert "--translator woof" in work and "metal-woof==1.1.5.2" in work


# ----------------------------------------------------------------------------------------------- F4

def write_f4(run: Path) -> None:
    (run / "compare").mkdir(parents=True)
    (run / "train.exit").write_text("0\n")
    good = {"right": 3e-7, "wrong_activation": 0.2}
    for name, speed in (("fwd", 0.81), ("back", -0.86), ("stop", 0.0)):
        (run / f"eval_{name}.json").write_text(json.dumps({"mean_forward_speed": speed, "falls_per_world": 0.0, "absolute_error": good}))
    (run / "eval_zero.json").write_text(json.dumps({"mean_forward_speed": 0.01, "falls_per_world": 0.5}))
    rng = np.random.default_rng(0)
    for seed in j4.GO2["seeds"]:
        base = rng.normal(size=(50, 64, 15))
        np.savez(run / "compare" / f"mujoco_{seed}.npz", values=base)
        np.savez(run / "compare" / f"mujoco_fine_{seed}.npz", values=base + 0.01)
        np.savez(run / "compare" / f"genesis_{seed}.npz", values=base + 0.02)
        np.savez(run / "compare" / f"fault_kp_{seed}.npz", values=base + 1.0)


def test_f4_a_right_run_passes(tmp_path):
    write_f4(tmp_path)
    result = j4.judge(tmp_path)
    assert result["verdict"] == "pass", result["parts"]
    assert result["grade"] == {"inference": "A", "physics": "B"}


@pytest.mark.parametrize("breaking, broken", [
    (lambda r: (r / "eval_fwd.json").write_text(json.dumps({"mean_forward_speed": 0.5, "falls_per_world": 0.0, "absolute_error": {"right": 1e-7, "wrong_activation": 1}})), "commands"),
    (lambda r: (r / "eval_back.json").write_text(json.dumps({"mean_forward_speed": -0.8, "falls_per_world": 0.2, "absolute_error": {"right": 1e-7, "wrong_activation": 1}})), "commands"),
    (lambda r: (r / "eval_stop.json").write_text(json.dumps({"mean_forward_speed": 0.0, "falls_per_world": 0.0, "absolute_error": {"right": 2e-5, "wrong_activation": 1}})), "inference"),
    (lambda r: (r / "eval_stop.json").write_text(json.dumps({"mean_forward_speed": 0.0, "falls_per_world": 0.0, "absolute_error": {"right": 1e-7, "wrong_activation": 1e-6}})), "inference"),
    (lambda r: np.savez(r / "compare" / "genesis_102.npz", values=np.load(r / "compare" / "mujoco_102.npz")["values"] + 0.3), "physics"),
    (lambda r: np.savez(r / "compare" / "fault_kp_103.npz", values=np.load(r / "compare" / "mujoco_103.npz")["values"]), "physics"),
    (lambda r: (r / "train.exit").write_text("1\n"), "train"),
])
def test_f4_each_break_fails_its_part(tmp_path, breaking, broken):
    write_f4(tmp_path)
    breaking(tmp_path)
    result = j4.judge(tmp_path)
    assert [k for k, v in result["parts"].items() if v["state"] != "pass"] == [broken]


# ----------------------------------------------------------------------------------------------- F5

def write_f5(side: Path, seeds: list[int], noise_seed: int, depth_offset: float = 0.0, folder: str = "sensors") -> None:
    (side / folder).mkdir(parents=True)
    (side / "install.exit").write_text("0\n")
    (side / "sensors.exit").write_text("0\n")
    rng = np.random.default_rng(noise_seed)
    arrays = {}
    base = np.full((16, 16, 3), 0.4)
    for view in j5.VIEWS:
        for seed in seeds:
            arrays[f"rgb/{view}/{seed}"] = (base + rng.normal(0, 0.003, base.shape)).astype(np.float32)
        depth = np.linspace(2, 4, 256).reshape(16, 16)
        label = np.where(depth > 3, 0, 1)
        arrays[f"depth/{view}"] = (depth + depth_offset).astype(np.float32)
        arrays[f"segmentation/{view}"] = label
        arrays[f"exact_depth/{view}"], arrays[f"exact_label/{view}"] = depth, label
    np.savez(side / folder / "sensors.npz", **arrays)
    (side / folder / "summary.json").write_text(json.dumps({"mitsuba": "x"}))


def f5_sides(tmp_path):
    cuda, mac = tmp_path / "cuda", tmp_path / "mac"
    write_f5(cuda, [1, 2, 3], 0)
    write_f5(mac, [11, 12, 13], 1)
    for fault, offset in (("box_5cm", 0.05), ("light_2pct", 0.0)):
        write_f5(mac, [11, 12, 13], 2, depth_offset=offset, folder=f"fault_{fault}")
        (mac / f"fault_{fault}.exit").write_text("0\n")
    light = j5.read_npz(mac / "fault_light_2pct" / "sensors.npz")
    np.savez(mac / "fault_light_2pct" / "sensors.npz", **{k: (v * 1.05 if k.startswith("rgb/") else v) for k, v in light.items()})
    return cuda, mac


def test_f5_a_right_pair_passes(tmp_path):
    result = j5.judge(*f5_sides(tmp_path))
    assert result["verdict"] == "pass", {k: v["reasons"] for k, v in result["parts"].items()}


def test_f5_depth_off_by_2_mm_fails_and_a_control_that_passes_fails(tmp_path):
    cuda, mac = f5_sides(tmp_path)
    data = j5.read_npz(mac / "sensors" / "sensors.npz")
    data["depth/top/centre"] = data["depth/top/centre"] + 0.002
    np.savez(mac / "sensors" / "sensors.npz", **data)
    result = j5.judge(cuda, mac)
    assert result["parts"]["geometry_mac"]["state"] == "fail" and result["parts"]["geometry_between"]["state"] == "fail"
    cuda, mac = f5_sides(tmp_path / "again")
    good = j5.read_npz(mac / "sensors" / "sensors.npz")
    np.savez(mac / "fault_box_5cm" / "sensors.npz", **good)
    assert [k for k, v in j5.judge(cuda, mac)["parts"].items() if v["state"] != "pass"] == ["controls"]


def test_f5_the_exact_depth_of_a_camera_looking_down_is_its_height():
    import m6_f5_sensors as sensors
    from cumlx.render.sensors import Camera
    camera = Camera(origin=(5.0, 5.0, 3.0), target=(5.0, 5.0, 0.0), up=(0.0, 1.0, 0.0), width=8, height=8)
    depth, label = sensors.exact(camera, (50.0, 50.0, 0.5), ["floor", "box"])
    rays = camera.ray_grid()
    assert np.allclose(depth, 3.0 / -rays[..., 2]) and (label == 0).all()
