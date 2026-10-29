"""Image quality flags (T4). Numpy only — no OpenCV dependency."""

import numpy as np

_LAPLACIAN_KERNEL = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a float64 2D view of the image."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if arr.shape[2] >= 3:
            return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    raise ValueError(f"expected 2D or HxWxC image, got shape {arr.shape}")


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian response (sharpness score)."""
    gray = _to_grayscale(image)
    padded = np.pad(gray, 1, mode="reflect")
    lap = (
        _LAPLACIAN_KERNEL[0, 1] * padded[:-2, 1:-1]
        + _LAPLACIAN_KERNEL[1, 0] * padded[1:-1, :-2]
        + _LAPLACIAN_KERNEL[1, 1] * padded[1:-1, 1:-1]
        + _LAPLACIAN_KERNEL[1, 2] * padded[1:-1, 2:]
        + _LAPLACIAN_KERNEL[2, 1] * padded[2:, 1:-1]
    )
    return float(np.var(lap))


DEFAULT_BLUR_THRESHOLD = 6.0  # below 5th pct of AGAR/BBBC (dark p5 6.9, BBBC p5 10.0, see bench/qc_calibration.md)

# Per-resolution thresholds (below 5th percentile): lower-res 52.5→50, bright 15.7→12, dark 6.9→5, vague 47→40, BBBC 10→8
BLUR_THRESHOLDS = {
    "agar_lower": 50.0,
    "agar_bright": 12.0,
    "agar_dark": 5.0,
    "agar_vague": 40.0,
    "bbbc": 8.0,
}


def is_blurry(image: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> bool:
    """True when sharpness is below threshold."""
    return laplacian_variance(image) < threshold


def saturated_fraction(image: np.ndarray, sat_value: float = 250.0) -> float:
    """Fraction of pixels at/above saturation (per-pixel max over channels)."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        peak = arr
    elif arr.ndim == 3:
        peak = arr.max(axis=2)
    else:
        raise ValueError(f"expected 2D or HxWxC image, got shape {arr.shape}")
    if peak.size == 0:
        return 0.0
    return float(np.mean(peak >= sat_value))


def has_glare(image: np.ndarray, sat_value: float = 250.0, max_fraction: float = 0.02) -> bool:
    """True when the saturated-pixel fraction exceeds max_fraction."""
    return saturated_fraction(image, sat_value) > max_fraction


def is_empty(num_masks: int) -> bool:
    """True when there are no masks / no foreground objects."""
    return int(num_masks) <= 0


def mask_coverage(masks: np.ndarray, total_pixels: int) -> float:
    """Union foreground fraction of stacked binary masks (N,H,W)."""
    stacked = np.asarray(masks)
    if stacked.size == 0 or int(total_pixels) <= 0:
        return 0.0
    union = np.any(stacked > 0, axis=0) if stacked.ndim == 3 else stacked > 0
    return float(np.mean(union))


def is_crowded(
    num_masks: int,
    coverage: float = 0.0,
    max_masks: int = 300,
    max_coverage: float = 0.7,
) -> bool:
    """True when mask count or foreground coverage exceeds its threshold."""
    return int(num_masks) > int(max_masks) or float(coverage) > float(max_coverage)
