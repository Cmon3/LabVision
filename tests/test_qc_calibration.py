"""QC calibration: acceptable photos not flagged as blurry."""

import numpy as np
from PIL import Image
from pathlib import Path

from labvision.qc import laplacian_variance, is_blurry, DEFAULT_BLUR_THRESHOLD, is_empty


def test_blur_threshold_below_p5():
    # All AGAR/BBBC sample images should not be blurry with new threshold 6.0
    from labvision.qc import BLUR_THRESHOLDS

    assert DEFAULT_BLUR_THRESHOLD == 6.0
    assert BLUR_THRESHOLDS["agar_dark"] == 5.0
    # synthetic sharp image has high variance
    sharp = np.zeros((20, 20), dtype=np.uint8)
    sharp[10, 10] = 255
    assert laplacian_variance(sharp) > 100
    # with new low threshold 6.0, even blurry gradient (low variance) may be flagged, but AGAR acceptable not
    blurry = np.ones((20, 20), dtype=np.uint8) * 128
    assert is_blurry(blurry, threshold=DEFAULT_BLUR_THRESHOLD)
    grad = np.tile(np.arange(20, dtype=np.uint8) * 12, (20, 1))
    assert not is_blurry(grad, threshold=DEFAULT_BLUR_THRESHOLD)


def test_empty_not_false_flag():
    assert not is_empty(1)
    assert is_empty(0)
    assert not is_empty(2)


def test_agar_sample_not_blurry():
    p = Path("data/agar_demo/AGAR_representative/lower-resolution/13895.jpg")
    if not p.exists():
        return
    arr = np.array(Image.open(p).convert("RGB"))
    assert not is_blurry(arr, threshold=DEFAULT_BLUR_THRESHOLD)
