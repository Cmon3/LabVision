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


def test_mask_clipping_to_box():
    # synthetic: mask covers whole image but box is small -> clipped
    box = Box(x1=10, y1=10, x2=20, y2=20)  # 10x10=100
    seg = Segmentation(boxes=[box], raw_masks=["10,10,!dummy"], width=100, height=100, cached=False)
    # patch decode_mask to return full image mask
    import labvision.segment as segmod

    orig = segmod.decode_mask

    def fake(payload, w, h):
        return np.ones((100, 100), dtype=np.uint8)

    segmod.decode_mask = fake
    try:
        arr = seg.decoded_masks(clip_to_box_ratio=2.0)
        assert arr.shape == (1, 100, 100)
        # full would be 10000, box 100, ratio 2 => clipped to 100
        assert int(arr[0].sum()) == 100
        # without clipping, would be 10000
        arr_no = seg.decoded_masks(clip_to_box_ratio=0)
        assert int(arr_no[0].sum()) == 10000
    finally:
        segmod.decode_mask = orig
