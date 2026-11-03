"""T1b: decode one_bit masks — real cached cheetah payload."""
import json
from pathlib import Path

import numpy as np

from labvision.mask_codec import decode_mask
from labvision.segment import Box, Segmentation, parse_output_text


def test_parse_boxes_tolerant():
    s = '<0f>0< |box;x1=10;y1=10;x2=50;y2=50;w=100;h=100|>< |mask;x=0;y=0;data=10,10,~abc|>'
    boxes, raw, w, h = parse_output_text(s)
    assert len(boxes) == 1 and w == 100


def test_decode_real_cached_one_bit():
    cache = Path("runs/cache/6882e714aa43d60a_cheetah_sam-3.1.json")
    if not cache.exists():
        # allow test to pass when cache missing (CI without data)
        return
    d = json.loads(cache.read_text())
    raw = d["raw_masks"][0]
    h_str, w_str, payload = raw.split(",", 2)
    h, w = int(h_str), int(w_str)
    arr = decode_mask(payload, w, h)
    assert arr.shape == (h, w)
    assert arr.dtype == np.uint8 or arr.dtype == bool
    assert int(arr.sum()) > 0
    # JS reference sum for this cheetah mask is 361269 at HxW before resize
    assert int(arr.sum()) == 361269


def test_segmentation_decoded_resize():
    cache = Path("runs/cache/6882e714aa43d60a_cheetah_sam-3.1.json")
    if not cache.exists():
        return
    d = json.loads(cache.read_text())
    seg = Segmentation(
        boxes=[Box(**b) for b in d["boxes"]],
        raw_masks=d["raw_masks"],
        width=d["width"],
        height=d["height"],
    )
    arr = seg.decoded_masks()
    assert arr.shape == (1, d["height"], d["width"])
    assert arr.shape[1] == 787 and arr.shape[2] == 762
    assert int(arr.sum()) > 0


def test_synthetic_circular_mask_in_box():
    # synthetic circular mask the size of its box, placed at box origin
    box = Box(x1=10, y1=20, x2=30, y2=40)  # 20x20
    seg = Segmentation(boxes=[box], raw_masks=["20,20,!dummy"], width=100, height=100)
    import labvision.segment as segmod

    orig = segmod.decode_mask

    def fake(payload, w, h):
        # 20x20 circular mask (radius 7) -> fill ratio ~0.38, round colony-like
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) / 2, (h - 1) / 2
        return ((xx - cx) ** 2 + (yy - cy) ** 2 <= 7**2).astype(np.uint8)

    segmod.decode_mask = fake
    try:
        arr = seg.decoded_masks()
        assert arr.shape == (1, 100, 100)
        # mask sum equals small mask sum
        small_sum = int(fake("", 20, 20).sum())
        assert int(arr[0].sum()) == small_sum
        # placed at box origin, not at 0,0
        assert not arr[0, 20, 10]  # top-left corner of box is outside circle
        # center of box at (20,30) -> should be True
        assert arr[0, 30, 20]
        # outside box is zero
        assert int(arr[0, :20, :].sum()) == 0
        assert int(arr[0, 40:, :].sum()) == 0
        assert int(arr[0, :, :10].sum()) == 0
        assert int(arr[0, :, 30:].sum()) == 0
        # fill ratio inside box ~ pi r^2 / box area
        box_area = 20 * 20
        ratio = small_sum / box_area
        assert 0.3 < ratio < 0.9
        # also test resize tolerance: mask 22x22 into 20x20 box (diff 2px) should resize
        def fake2(payload, w, h):
            return np.ones((22, 22), dtype=np.uint8)

        segmod.decode_mask = fake2
        arr2 = seg.decoded_masks()
        assert arr2.shape == (1, 100, 100)
        # resized to 20x20 => 400
        assert int(arr2[0].sum()) == 400
    finally:
        segmod.decode_mask = orig
