import math

import numpy as np

from labvision.measure import (
    count_colonies,
    equivalent_diameter_px,
    filter_by_min_diameter,
    mask_area_px,
    measure_masks,
)


def disk_mask(size: int, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) / 2.0
    return (xx - c) ** 2 + (yy - c) ** 2 <= radius**2


def test_area_and_diameter_of_single_pixel():
    m = np.zeros((3, 3), dtype=bool)
    m[1, 1] = True
    assert mask_area_px(m) == 1
    assert equivalent_diameter_px(1) == math.sqrt(4.0 / math.pi)
    assert equivalent_diameter_px(0) == 0.0


def test_measure_disk_px():
    m = disk_mask(21, 5.0)
    (row,) = measure_masks(np.stack([m]))
    assert row["area_px"] == int(m.sum())
    assert row["diameter_px"] == math.sqrt(4.0 * int(m.sum()) / math.pi)
    assert "diameter_mm" not in row


def test_measure_mm_scale_and_filter_2mm():
    # plate is 100 px wide = 90 mm -> 0.9 mm/px.
    # radius 5 px disk -> diam ~10 px -> ~9 mm (kept); 1-px dot -> ~1 mm (dropped).
    big = disk_mask(21, 5.0)
    small = np.zeros((21, 21), dtype=bool)
    small[10, 10] = True
    rows = measure_masks(np.stack([big, small]), plate_diameter_px=100.0)
    assert rows[0]["diameter_mm"] > 2.0
    assert rows[1]["diameter_mm"] < 2.0
    kept = filter_by_min_diameter(rows, 2.0)
    assert [r["index"] for r in kept] == [0]
    assert count_colonies(kept) == 1


def test_empty_stack_returns_empty():
    assert measure_masks(np.zeros((0, 4, 4), dtype=bool)) == []
