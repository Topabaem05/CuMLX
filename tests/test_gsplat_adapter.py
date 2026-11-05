"""cumlx.compat.gsplat_metalsplat with stubs for metalsplat's kernels, and on the real kernels where they exist."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cumlx.compat import gsplat_metalsplat as adapter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LIMITS = json.loads((REPO / "validation" / "m6-f2-adapter-limits.json").read_text())
N = 2
VIEW = torch.tensor([[[0.0, -1.0, 0.0, 0.5], [1.0, 0.0, 0.0, -0.25], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]])
KS = torch.tensor([[[100.0, 0.0, 30.0], [0.0, 90.0, 20.0], [0.0, 0.0, 1.0]]])


def inputs(**changes):
    values = dict(means=torch.zeros(N, 3), quats=torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]), scales=torch.ones(N, 3), opacities=torch.full((N,), 0.5), colors=torch.full((N, 3), 0.25), viewmats=VIEW, Ks=KS, width=8, height=6)
    return {**values, **changes}


@pytest.fixture
def kernels(monkeypatch):
    calls = SimpleNamespace(project=[], raster=[])

    def project_gaussians(means, scales, quats, R, t, fx, fy, cx, cy, width, height, near, eps2d):
        calls.project.append(dict(means=means, scales=scales, quats=quats, R=R, t=t, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, near=near, eps2d=eps2d))
        n = means.shape[0]
        return means[:, :2], torch.tensor([1.0, 9.0]), torch.zeros(n, 3), torch.ones(n), torch.ones(n), torch.full((n,), 0.5)

    def rasterize_gaussians(means2d, depths, conics, opacities, colors, radii, valid, width, height, tile_size=16, background=None, return_aux=False):
        calls.raster.append(dict(opacities=opacities, colors=colors, valid=valid, tile_size=tile_size, background=background, return_aux=return_aux))
        image = colors[0] * torch.ones(height, width, 3)
        return (image, None, torch.full((height, width), 0.75)) if return_aux else image

    for name, fields in {"metalsplat": {}, "metalsplat.ops": {}, "metalsplat.ops.project": {"project_gaussians": project_gaussians}, "metalsplat.ops.rasterize": {"rasterize_gaussians": rasterize_gaussians, "_cumlx_alpha_max": adapter.GSPLAT_ALPHA_MAX}, "metalsplat.ops.tiling": {"DEFAULT_TILE_SIZE": 16}}.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_the_conventions_reach_the_kernels(kernels):
    with torch.no_grad():
        adapter.rasterization(**inputs(), near_plane=0.25, eps2d=0.125)
    call = kernels.project[0]
    assert (call["fx"], call["fy"], call["cx"], call["cy"]) == (100.0, 90.0, 30.0, 20.0)
    assert (call["width"], call["height"], call["near"], call["eps2d"]) == (8, 6, 0.25, 0.125)
    assert torch.equal(call["R"], VIEW[0, :3, :3]) and torch.equal(call["t"], VIEW[0, :3, 3])
    assert torch.equal(call["quats"], torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]))


def test_the_shapes_are_those_of_gsplat(kernels):
    with torch.no_grad():
        image, alpha, meta = adapter.rasterization(**inputs())
    assert image.shape == (1, 6, 8, 3) and alpha.shape == (1, 6, 8, 1)
    assert meta["width"] == 8 and meta["n_cameras"] == 1


def test_classic_keeps_the_opacity_and_antialiased_scales_it(kernels):
    with torch.no_grad():
        adapter.rasterization(**inputs())
        adapter.rasterization(**inputs(), rasterize_mode="antialiased")
    assert torch.equal(kernels.raster[0]["opacities"], torch.full((N,), 0.5))
    assert torch.equal(kernels.raster[1]["opacities"], torch.full((N,), 0.25))


def test_far_plane_removes_a_gaussian_behind_it(kernels):
    with torch.no_grad():
        adapter.rasterization(**inputs(), far_plane=5.0)
    assert kernels.raster[0]["valid"].tolist() == [1.0, 0.0]


def test_alpha_without_gradients_comes_from_one_pass(kernels):
    with torch.no_grad():
        _, alpha, _ = adapter.rasterization(**inputs())
    assert len(kernels.raster) == 1 and float(alpha.max()) == 0.25


def test_alpha_with_gradients_is_white_gaussians_on_black(kernels):
    _, alpha, _ = adapter.rasterization(**inputs())
    assert len(kernels.raster) == 2
    second = kernels.raster[1]
    assert torch.equal(second["colors"], torch.ones(N, 3)) and second["background"] is None and second["return_aux"] is False
    assert float(alpha.max()) == 1.0


def test_each_camera_gets_its_own_colors_and_background(kernels):
    views, Ks = VIEW.repeat(2, 1, 1), KS.repeat(2, 1, 1)
    colors = torch.stack([torch.full((N, 3), 0.1), torch.full((N, 3), 0.9)])
    backgrounds = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with torch.no_grad():
        image, _, _ = adapter.rasterization(**inputs(viewmats=views, Ks=Ks, colors=colors), backgrounds=backgrounds)
    assert image.shape[0] == 2
    assert kernels.raster[0]["colors"] is not kernels.raster[1]["colors"]
    assert torch.equal(kernels.raster[1]["colors"], colors[1]) and torch.equal(kernels.raster[1]["background"], backgrounds[1])


REFUSALS = {
    "sh_degree": {"sh_degree": 3}, "render_mode": {"render_mode": "RGB+ED"}, "camera_model": {"camera_model": "fisheye"}, "covars": {"covars": torch.eye(3)[None]},
    "sparse_grad": {"sparse_grad": True}, "absgrad": {"absgrad": True}, "distributed": {"distributed": True}, "with_ut": {"with_ut": True}, "segmented": {"segmented": True},
    "batched_means": {"means": torch.zeros(1, N, 3)}, "viewmats_require_grad": {"viewmats": VIEW.clone().requires_grad_()}, "colors_not_rgb": {"colors": torch.zeros(N, 6)}, "unknown_option": {"no_such_option": True},
}


@pytest.mark.parametrize("name", LIMITS["refusals"]["must_raise"])
def test_an_option_it_lacks_raises(name):
    with pytest.raises(NotImplementedError):
        adapter.rasterization(**{**inputs(), **REFUSALS[name]})


@pytest.mark.parametrize("change", [{"radius_clip": 0.5}, {"with_eval3d": True}, {"return_normals": True}, {"global_z_order": False}, {"rays": torch.zeros(1)}, {"rasterize_mode": "other"}, {"colors": None}])
def test_other_options_raise_too(change):
    with pytest.raises(NotImplementedError):
        adapter.rasterization(**{**inputs(), **change})


def test_a_default_value_of_an_option_is_accepted(kernels):
    with torch.no_grad():
        adapter.rasterization(**inputs(), sh_degree=None, render_mode="RGB", camera_model="pinhole", radius_clip=0.0, absgrad=False, packed=False, channel_chunk=8, viewmats_rs=None)
    assert kernels.project


def test_the_gsplat_names_it_does_not_give_are_absent():
    assert not hasattr(adapter, "rasterization_inria_wrapper") and not hasattr(adapter, "strategy")


@pytest.mark.gpu
def test_on_the_real_kernels_it_equals_the_hand_render(tmp_path):
    pytest.importorskip("metalsplat")
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    sys.path.insert(0, str(REPO / "spikes"))
    import m6_f2_conformance as conformance
    rule = LIMITS["conformance"]
    for name, scene in conformance.conformance(rule).items():
        assert scene["within"], name
    assert all(item["caught"] for item in conformance.controls(rule, LIMITS["controls"]["must_fail"]).values())


METALSPLAT_LINES = "inline float a() {\n    return min(0.99, opacity * exp(power));\n}\n// clamped at 0.99\n    if (raw >= 0.99) d_alpha_raw = 0.0;\n"


def test_the_clamp_becomes_that_of_gsplat():
    text = adapter.gsplat_rasterize_source(METALSPLAT_LINES)
    assert "min(0.999, opacity * exp(power))" in text and "if (raw >= 0.999) d_alpha_raw = 0.0;" in text
    assert "min(0.99," not in text and "raw >= 0.99)" not in text
    assert "// clamped at 0.99" in text


@pytest.mark.parametrize("text", [METALSPLAT_LINES.replace("min(0.99,", "min(0.98,"), METALSPLAT_LINES + METALSPLAT_LINES, METALSPLAT_LINES.replace("raw >= 0.99", "raw > 0.99")])
def test_a_metalsplat_without_those_lines_stops(text):
    with pytest.raises(RuntimeError, match="not the pinned one"):
        adapter.gsplat_rasterize_source(text)


def test_only_the_rasterize_library_is_replaced_and_only_once(monkeypatch):
    compiled = []
    loader = ModuleType("metalsplat.kernels._loader")
    loader.source = lambda name: METALSPLAT_LINES if name == "rasterize.metal" else pytest.fail(name)
    kernels_package = ModuleType("metalsplat.kernels")
    kernels_package._loader = loader
    for name, module in {"metalsplat": ModuleType("metalsplat"), "metalsplat.kernels": kernels_package, "metalsplat.kernels._loader": loader}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.mps, "compile_shader", lambda text: compiled.append(text) or "gsplat library", raising=False)
    module = SimpleNamespace(load_kernel=lambda name: f"metalsplat {name}")
    adapter._use_gsplat_alpha_clamp(module)
    adapter._use_gsplat_alpha_clamp(module)
    assert len(compiled) == 1 and "min(0.999," in compiled[0]
    assert module.load_kernel("rasterize") == "gsplat library"
    assert module.load_kernel("project") == "metalsplat project"
