"""Masks -> counts, areas, diameters.

Deterministic: every number in a report comes from here, never from the LLM.
Masks are boolean arrays of shape (H, W). Measurements are plain dicts so
callers (agent, runlog, app) do not need pandas.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_PLATE_DIAMETER_MM = 90.0


def mask_area_px(mask: np.ndarray) -> int:
    """Count foreground pixels in one (H, W) mask."""
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def equivalent_diameter_px(area_px: float) -> float:
    """Diameter of a circle with the same area, in pixels."""
    if area_px <= 0:
        return 0.0
    return math.sqrt(4.0 * area_px / math.pi)


def mm_per_px(plate_diameter_px: float, plate_diameter_mm: float = DEFAULT_PLATE_DIAMETER_MM) -> float:
    """Pixel pitch from a known plate diameter. Both diameters must be > 0."""
    if plate_diameter_px <= 0:
        raise ValueError("plate_diameter_px must be > 0")
    if plate_diameter_mm <= 0:
        raise ValueError("plate_diameter_mm must be > 0")
    return plate_diameter_mm / plate_diameter_px


def measure_masks(
    masks: np.ndarray,
    plate_diameter_px: float | None = None,
    plate_diameter_mm: float = DEFAULT_PLATE_DIAMETER_MM,
) -> list[dict]:
    """Measure each mask in an (N, H, W) stack.

    Returns one dict per mask with area_px and diameter_px, plus area_mm2
    and diameter_mm when plate_diameter_px is given.
    """
    arr = np.asarray(masks, dtype=bool)
    if arr.size == 0:
        return []
    if arr.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {arr.shape}")
    scale: float | None = None
    if plate_diameter_px is not None:
        scale = mm_per_px(plate_diameter_px, plate_diameter_mm)
    rows: list[dict] = []
    for i in range(arr.shape[0]):
        area = mask_area_px(arr[i])
        diam_px = equivalent_diameter_px(area)
        row: dict = {"index": i, "area_px": area, "diameter_px": diam_px}
        if scale is not None:
            row["diameter_mm"] = diam_px * scale
            row["area_mm2"] = area * scale * scale
        rows.append(row)
    return rows


def filter_by_min_diameter(rows: list[dict], min_diameter_mm: float) -> list[dict]:
    """Keep only rows with diameter_mm >= min_diameter_mm."""
    if min_diameter_mm < 0:
        raise ValueError("min_diameter_mm must be >= 0")
    kept = []
    for row in rows:
        if "diameter_mm" not in row:
            raise ValueError("rows need diameter_mm: pass plate_diameter_px to measure_masks")
        if row["diameter_mm"] >= min_diameter_mm:
            kept.append(row)
    return kept


def count_colonies(rows: list[dict]) -> int:
    """Number of colonies in an (already filtered) measurement list."""
    return len(rows)
