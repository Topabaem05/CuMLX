"""spikes/m6_f2_cuda_judge.py on made-up output folders, and the scripts around it. Nothing third-party runs."""

import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spikes"))
import m6_f2_cuda_judge as j  # noqa: E402

WORK = (REPO / "spikes" / "m6_f2_cuda_work.sh").read_text()
V100 = (REPO / "spikes" / "remote" / "m6_f2.sh").read_text()
LAUNCHER = (REPO / "spikes" / "m6_cuda_v100.sh").read_text()
FAULT_MODULE = (REPO / "spikes" / "m6_f2_fault_gsplat.py").read_text()


def scene_arrays(rng: np.random.Generator, offset: float = 0.0) -> dict[str, np.ndarray]:
    arrays = {}
    for scene in j.ALL_SCENES:
        image = np.zeros((16, 16, 3))
        image[4 + int(offset), 6] = 1.0
        image[8:12, 8:12] = rng.uniform(0.2, 0.8, (4, 4, 3))
        alpha = image.mean(-1)
        arrays[f"image/{scene}"], arrays[f"alpha/{scene}"] = image, alpha
        arrays[f"hand_image/{scene}"], arrays[f"hand_alpha/{scene}"] = image.copy(), alpha.copy()
    for scene in j.GRADIENT_SCENES:
        for name in j.PARAMETERS:
            arrays[f"grad/{scene}/{name}"] = np.linspace(1.0, 2.0, 12)
    return arrays


def write_side(out: Path, device: str, psnr: list[float], arrays: dict[str, np.ndarray], faults: bool = False) -> Path:
    out.mkdir(parents=True)
    for name in ("install", "build", "record", "program"):
        (out / f"{name}.exit").write_text("0\n")
    np.savez(out / "record.npz", **arrays)
    (out / "record.json").write_text(json.dumps({"scenes": {s: {"background": None} for s in j.ALL_SCENES}}))
    (out / "program.json").write_text(json.dumps({"first_loss": 0.5, "last_loss": 0.1, "device": device}))
    for seed, value in zip(j.SEEDS, psnr, strict=True):
        for mode, end in (("right", value), ("frozen", 15.0), ("wrong_cameras", 18.0)):
            (out / f"train_{mode}_{seed}.exit").write_text("0\n")
            (out / f"train_{mode}_{seed}.json").write_text(json.dumps({"device": device, "heldout_psnr_start": 15.0, "heldout_psnr_end": end}))
        if faults:
            (out / f"train_colors_gradient_zero_{seed}.exit").write_text("0\n")
            (out / f"train_colors_gradient_zero_{seed}.json").write_text(json.dumps({"device": device, "heldout_psnr_start": 15.0, "heldout_psnr_end": 20.0}))
    if faults:
        for fault in j.RENDER_FAULTS + j.GRADIENT_FAULTS:
            broken = {key: value.copy() for key, value in arrays.items()}
            if fault in j.RENDER_FAULTS:
                broken["image/one_gaussian"] = np.roll(broken["image/one_gaussian"], 1, axis=0)
            else:
                broken["grad/covering/means"] = broken["grad/covering/means"] * 1.1
            (out / f"fault_{fault}.exit").write_text("0\n")
            np.savez(out / f"fault_{fault}.npz", **broken)
    return out


@pytest.fixture
def sides(tmp_path):
    cuda = write_side(tmp_path / "cuda", "cuda", [40.0, 41.0, 42.0, 41.5, 40.5], scene_arrays(np.random.default_rng(0)))
    mac = write_side(tmp_path / "mac", "mps", [40.2, 41.1, 41.8, 41.4, 40.9], scene_arrays(np.random.default_rng(0)), faults=True)
    return cuda, mac


def test_a_right_pair_passes(sides):
    result = j.judge(*sides)
    assert result["verdict"] == "pass", {k: v["reasons"] for k, v in result["parts"].items()}
    assert result["grade"] == "B"


def change_npz(out: Path, name: str, change) -> None:
    with np.load(out / f"{name}.npz") as data:
        arrays = {key: data[key] for key in data.files}
    change(arrays)
    np.savez(out / f"{name}.npz", **arrays)


def change_json(out: Path, name: str, change) -> None:
    path = out / f"{name}.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize("breaking, broken", [
    (lambda c, m: change_npz(m, "record", lambda a: a.update({"image/learned_like_train_view": a["image/learned_like_train_view"] * 1.01})), "renders_between"),
    # The Mac's own hand render moves with these, so only the comparison between machines sees them.
    (lambda c, m: change_npz(m, "record", lambda a: a.update({k: a[k] * 1.01 for k in ("alpha/opaque_wall", "hand_alpha/opaque_wall")})), "renders_between"),
    (lambda c, m: change_npz(m, "record", lambda a: a.update({k: np.roll(a[k], 1, axis=1) for k in ("image/small_classic", "hand_image/small_classic")})), "renders_between"),
    (lambda c, m: change_npz(c, "record", lambda a: a.update({"hand_image/opaque_wall": a["hand_image/opaque_wall"] * 1.01})), "hand_cuda"),
    (lambda c, m: change_npz(m, "record", lambda a: a.pop("hand_alpha/near_and_far")), "hand_mac"),
    (lambda c, m: change_npz(m, "record", lambda a: a.update({"grad/learned_like/quats": a["grad/learned_like/quats"] * 1.06})), "gradients"),
    (lambda c, m: change_npz(m, "record", lambda a: a.pop("grad/covering/colors")), "gradients"),
    (lambda c, m: (m / "fault_no_eps2d.exit").write_text("1\n"), "controls"),
    (lambda c, m: change_npz(m, "fault_alpha_clamp_0.99", lambda a: a.update({"image/one_gaussian": np.roll(a["image/one_gaussian"], -1, axis=0)})), "controls"),
    (lambda c, m: (c / "build.exit").write_text("1\n"), "setup"),
    (lambda c, m: change_json(c, "program", lambda d: d.update(last_loss=0.9)), "original_program"),
    (lambda c, m: (c / "program.exit").write_text("1\n"), "original_program"),
    (lambda c, m: change_json(m, "train_right_3", lambda d: d.update(heldout_psnr_end=24.0)), "training_mac"),
    (lambda c, m: change_json(c, "train_frozen_0", lambda d: d.update(heldout_psnr_end=26.0)), "training_cuda"),
    (lambda c, m: change_json(m, "train_wrong_cameras_1", lambda d: d.update(heldout_psnr_end=39.0)), "training_mac"),
    (lambda c, m: change_json(m, "train_right_2", lambda d: d.update(device="cpu")), "training_mac"),
    (lambda c, m: [change_json(m, f"train_right_{s}", lambda d: d.update(heldout_psnr_end=d["heldout_psnr_end"] - 1.5)) for s in j.SEEDS], "equivalence"),
    (lambda c, m: [change_json(m, f"train_colors_gradient_zero_{s}", lambda d: d.update(heldout_psnr_end=41.0)) for s in j.SEEDS], "equivalence_controls"),
])
def test_each_break_fails_its_part(sides, breaking, broken):
    breaking(*sides)
    result = j.judge(*sides)
    assert result["verdict"] == "fail"
    assert result["parts"][broken]["state"] == "fail", result["parts"][broken]
    assert [name for name, item in result["parts"].items() if item["state"] != "pass"] == [broken]


def test_a_missing_record_is_not_a_pass(sides):
    (sides[1] / "record.npz").unlink()
    result = j.judge(*sides)
    assert result["verdict"] != "pass" and result["parts"]["renders_between"]["state"] == "not_run"


def test_the_tolerance_is_three_standard_errors_or_one_percent():
    # Equal values give a standard error of 0, so the 1% of the mean is the tolerance.
    flat = j.equivalent([40.0] * 5, [40.0] * 5)
    assert flat["tolerance"] == pytest.approx(0.4)
    assert j.equivalent([40.39] * 5, [40.0] * 5)["within"] and not j.equivalent([40.41] * 5, [40.0] * 5)["within"]
    spread = j.equivalent([38.0, 42.0, 38.0, 42.0, 40.0], [40.0] * 5)
    assert spread["tolerance"] == pytest.approx(3 * np.std([38.0, 42.0, 38.0, 42.0, 40.0], ddof=1) / np.sqrt(5))


def test_the_peak_is_judged_on_the_six_scenes_only():
    rule = j.LIMITS["renders"]["between_machines"]
    moved = {"image_rel_l2": 0.0, "alpha_rel_l2": 0.0, "peak_error_px": 3.0}
    assert j.within(moved, rule, "one_gaussian") and not j.within(moved, rule, "opaque_wall") and not j.within(moved, rule, "learned_like_train_view")


def test_the_scripts_run_every_control_and_seed_that_the_rules_name():
    faults = re.search(r"FAULTS = \((.*?)\)\n", FAULT_MODULE, re.S).group(1)
    for fault in j.RENDER_FAULTS + j.GRADIENT_FAULTS + ("colors_gradient_zero",):
        assert f"'{fault}'" in faults
    loop = re.search(r"for fault in (.*?); do", WORK).group(1).split()
    assert sorted(loop) == sorted(j.RENDER_FAULTS + j.GRADIENT_FAULTS)
    for text in (WORK, V100):
        assert "for seed in 0 1 2 3 4; do" in text and "right frozen wrong_cameras" in text
    assert "--substitute \"gsplat=$ADAPTER\"" in WORK and "ADAPTER=cumlx.compat.gsplat_metalsplat" in WORK
    assert "gsplat==1.5.3" in V100 and j.LIMITS["cuda"]["gsplat"].startswith("1.5.3")


def test_the_v100_launcher_refuses_a_busy_gpu_and_keeps_to_its_folder():
    assert "--query-compute-apps=pid" in LAUNCHER and "Nothing was started" in LAUNCHER
    assert "TORCH_EXTENSIONS_DIR=$WORK/torch_extensions" in V100 and '--target "$WORK/pkgs"' in V100
    assert V100.count("pip install") == 1
