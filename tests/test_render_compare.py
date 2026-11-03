"""cumlx.render.compare on synthetic images. No Mitsuba, no GPU."""
import json
from pathlib import Path

import numpy as np
import pytest

from cumlx.errors import CumlxError
from cumlx.render.compare import MIN_SEEDS, compare_images, psnr


def _base():
    y, x = np.mgrid[0:64, 0:64] / 63
    return np.stack([0.2 + 0.5 * x, 0.2 + 0.5 * y, 0.3 + 0.2 * x * y], axis=-1)


def _side(base, seeds, noise=2e-3):
    return {seed: base + np.random.default_rng(seed).normal(0, noise, base.shape) for seed in seeds}


GOLDEN_SEEDS = (1, 2, 3)
CANDIDATE_SEEDS = (11, 12, 13)


def test_same_image_with_other_noise_gets_grade_b():
    base = _base()
    result = compare_images(_side(base, GOLDEN_SEEDS), _side(base, CANDIDATE_SEEDS))
    assert result['verdict'] == 'ok' and result['grade'] == 'B'
    assert 0.8 < result['mse_ratio'] < 1.2


@pytest.mark.parametrize('name, change', [
    ('2% too bright', lambda b: b * 1.02),
    ('one pixel to the right', lambda b: np.roll(b, 1, axis=1)),
    ('red channel missing', lambda b: b * np.array([0.0, 1.0, 1.0])),
])
def test_wrong_images_diverge(name, change):
    base = _base()
    result = compare_images(_side(base, GOLDEN_SEEDS), _side(change(base), CANDIDATE_SEEDS))
    assert result['verdict'] == 'diverged', name
    assert result['grade'] is None and result['failures']


def test_a_small_change_in_one_corner_is_caught_by_the_noise_ratio():
    base = _base()
    wrong = base.copy()
    wrong[:16, :16] += 0.02  # The whole image stays above the 40 dB floor, and the mean stays inside 1%.
    result = compare_images(_side(base, GOLDEN_SEEDS), _side(wrong, CANDIDATE_SEEDS))
    assert result['verdict'] == 'diverged'
    assert result['worst_psnr_across_db'] > 40
    assert any('MSE across' in failure for failure in result['failures'])


def test_noise_too_high_to_judge_is_not_a_pass():
    base = _base()
    result = compare_images(_side(base, GOLDEN_SEEDS, noise=0.02), _side(base, CANDIDATE_SEEDS, noise=0.02))
    assert result['verdict'] == 'inconclusive'
    assert result['grade'] is None and result['failures']


def test_a_non_finite_image_fails():
    base = _base()
    candidate = _side(base, CANDIDATE_SEEDS)
    candidate[11][3, 3, 0] = np.nan
    result = compare_images(_side(base, GOLDEN_SEEDS), candidate)
    assert result['verdict'] == 'diverged' and 'an image is not finite' in result['failures']


def test_too_few_seeds_other_shapes_and_shared_seeds_are_refused():
    base = _base()
    golden = _side(base, GOLDEN_SEEDS)
    with pytest.raises(CumlxError):
        compare_images(golden, _side(base, CANDIDATE_SEEDS[:MIN_SEEDS - 1]))
    with pytest.raises(CumlxError):
        compare_images(golden, _side(base[:32], CANDIDATE_SEEDS))
    with pytest.raises(CumlxError):
        compare_images(golden, dict(golden))
    with pytest.raises(CumlxError):
        compare_images(golden, _side(base, (1, 12, 13)))


def test_the_record_does_not_claim_the_cuda_gate_without_a_comparison():
    record = json.loads((Path(__file__).resolve().parent.parent / 'validation' / 'm5-render.json').read_text())
    if record['cuda_comparison']['status'] != 'compared':
        assert record['gates']['metal_vs_v100_cuda_grade_b'] == 'open'
    local = record['local_checks_of_the_comparison']['spp_4096']
    assert all(scene['grade'] == 'B' for scene in local['scalar_metal']['scenes'].values())
    for faulty in ('metal_metal_light2', 'metal_metal_direct', 'scalar_metal_light2'):
        assert all(scene['verdict'] == 'diverged' for scene in local[faulty]['scenes'].values()), faulty
    assert record['physics_and_camera']['ok'] is True


def test_psnr_of_equal_images_and_of_known_error():
    base = _base()
    assert psnr(base, base) == float('inf')
    assert psnr(np.zeros((2, 2, 3)), np.full((2, 2, 3), 0.1)) == pytest.approx(20.0)
