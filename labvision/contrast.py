"""Lab contrast check for /api/correct add — inside mask vs 10-20px ring, excluding other masks."""
from __future__ import annotations

import numpy as np

def _rgb_to_lab(arr_rgb: np.ndarray) -> np.ndarray:
    # arr_rgb HxWx3 uint8 -> Lab HxWx3 float
    try:
        from skimage.color import rgb2lab
        f = arr_rgb.astype(np.float32) / 255.0
        return rgb2lab(f)
    except Exception:
        # fallback: simple RGB euclidean (approx)
        return arr_rgb.astype(np.float32)

def _ring_masks(mask: np.ndarray, r_in: int = 10, r_out: int = 20, other_union: np.ndarray | None = None) -> np.ndarray:
    """Return bool ring 10-20px outside mask, excluding other masks and mask itself."""
    h, w = mask.shape
    try:
        import cv2
        m_u8 = (mask.astype(np.uint8) * 255)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # dilate by r_out and r_in via iterations (approx Euclidean)
        dil_out = cv2.dilate(m_u8, k3, iterations=r_out)
        dil_in = cv2.dilate(m_u8, k3, iterations=r_in)
        ring = (dil_out > 0) & ~(dil_in > 0)
        # exclude original mask (already) and other masks
        ring = ring & ~mask
        if other_union is not None:
            ring = ring & ~other_union.astype(bool)
        return ring
    except Exception:
        # fallback: no ring
        return np.zeros_like(mask, dtype=bool)

def contrast_delta_e(arr_rgb: np.ndarray, mask: np.ndarray, other_union: np.ndarray | None = None, r_in: int = 10, r_out: int = 20) -> float | None:
    """Delta E between mean Lab inside mask and mean Lab in ring. None if ring empty."""
    if mask is None or mask.size == 0 or not np.any(mask):
        return None
    ring = _ring_masks(mask, r_in=r_in, r_out=r_out, other_union=other_union)
    if not np.any(ring):
        return None
    lab = _rgb_to_lab(arr_rgb)
    # mean inside
    inside = lab[mask]
    outside = lab[ring]
    # filter NaN/inf
    m_in = np.nanmean(inside.reshape(-1, 3), axis=0)
    m_out = np.nanmean(outside.reshape(-1, 3), axis=0)
    if np.any(np.isnan(m_in)) or np.any(np.isnan(m_out)):
        return None
    diff = m_in - m_out
    return float(np.sqrt(np.sum(diff * diff)))

def is_contrast_ok(arr_rgb: np.ndarray, mask: np.ndarray, other_union: np.ndarray | None = None, threshold: float = 5.0) -> tuple[bool, float | None]:
    """True if contrast >= threshold or ring empty (cannot decide -> pass)."""
    d = contrast_delta_e(arr_rgb, mask, other_union)
    if d is None:
        return True, d  # no ring to compare -> allow
    return (d >= threshold), d
