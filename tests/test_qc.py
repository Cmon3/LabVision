import numpy as np

from labvision.qc import (
    has_glare,
    is_blurry,
    is_crowded,
    is_empty,
    laplacian_variance,
    mask_coverage,
    saturated_fraction,
)


def _checker(n: int = 16) -> np.ndarray:
    return (np.indices((n, n)).sum(axis=0) % 2 * 255).astype(np.uint8)


def test_laplacian_variance_uniform_is_zero() -> None:
    assert laplacian_variance(np.full((16, 16), 128, dtype=np.uint8)) == 0.0


def test_blur_flag() -> None:
    assert is_blurry(np.full((16, 16), 128, dtype=np.uint8), threshold=100.0) is True
    assert is_blurry(_checker(), threshold=100.0) is False


def test_glare_flags() -> None:
    assert saturated_fraction(np.zeros((8, 8), dtype=np.uint8)) == 0.0
    assert saturated_fraction(np.full((8, 8), 255, dtype=np.uint8)) == 1.0
    assert has_glare(np.full((8, 8), 255, dtype=np.uint8)) is True
    assert has_glare(np.zeros((8, 8), dtype=np.uint8)) is False


def test_glare_partial_fraction() -> None:
    img = np.zeros((10, 10), dtype=np.uint8)
    img[0, :] = 255  # 10% saturated
    assert saturated_fraction(img) == 0.1
    assert has_glare(img, max_fraction=0.02) is True
    assert has_glare(img, max_fraction=0.5) is False


def test_empty_flag() -> None:
    assert is_empty(0) is True
    assert is_empty(3) is False


def test_crowded_flags() -> None:
    assert is_crowded(5, 0.1) is False
    assert is_crowded(500, 0.1) is True  # too many masks
    assert is_crowded(5, 0.9) is True  # too much coverage


def test_mask_coverage_union() -> None:
    m1 = np.zeros((4, 4), dtype=np.uint8)
    m2 = np.zeros((4, 4), dtype=np.uint8)
    m1[:2, :] = 1
    m2[2:, :] = 1
    assert mask_coverage(np.stack([m1, m2]), 16) == 1.0
    assert mask_coverage(np.zeros((1, 4, 4), dtype=np.uint8), 16) == 0.0


def test_rgb_inputs() -> None:
    gray = _checker()
    rgb = np.stack([gray, gray, gray], axis=2)
    assert is_blurry(rgb, threshold=100.0) is False
    assert saturated_fraction(np.full((4, 4, 3), 255, dtype=np.uint8)) == 1.0
