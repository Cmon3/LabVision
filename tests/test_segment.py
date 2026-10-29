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
