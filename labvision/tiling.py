"""Tiling + NMS for AGAR."""
from __future__ import annotations

import numpy as np
from PIL import Image

from labvision.segment import Box, Segmentation, segment


def _iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / float(area_a + area_b - inter + 1e-9)


def nms(boxes: list[Box], iou_thresh: float = 0.5) -> list[int]:
    """Return indices to keep. Simple greedy by area (larger first)."""
    if not boxes:
        return []
    areas = [(b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    keep = []
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if suppressed[j] or i == j:
                continue
            if _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed[j] = True
    return keep


def tile_boxes(image_w: int, image_h: int, n: int, overlap: float = 0.15):
    """Generate n x n tiles with overlap, return list of (x1,y1,x2,y2)."""
    if n == 1:
        return [(0, 0, image_w, image_h)]
    tiles = []
    stride_x = image_w / n
    stride_y = image_h / n
    ov_x = stride_x * overlap
    ov_y = stride_y * overlap
    for iy in range(n):
        for ix in range(n):
            x1 = int(max(0, ix * stride_x - (ov_x if ix > 0 else 0)))
            y1 = int(max(0, iy * stride_y - (ov_y if iy > 0 else 0)))
            x2 = int(min(image_w, (ix + 1) * stride_x + (ov_x if ix < n - 1 else 0)))
            y2 = int(min(image_h, (iy + 1) * stride_y + (ov_y if iy < n - 1 else 0)))
            tiles.append((x1, y1, x2, y2))
    return tiles


def segment_tiled(image_bytes: bytes, prompt: str, tiling: int = 1, overlap: float = 0.15) -> Segmentation:
    """Segment whole or tiled image and merge boxes via NMS."""
    if tiling == 1:
        return segment(image_bytes, prompt)
    # need image size
    from io import BytesIO

    im = Image.open(BytesIO(image_bytes))
    w, h = im.size
    tiles = tile_boxes(w, h, tiling, overlap)
    all_boxes: list[Box] = []
    all_masks: list[str] = []
    for x1, y1, x2, y2 in tiles:
        crop = im.crop((x1, y1, x2, y2))
        import io

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        seg = segment(buf.getvalue(), prompt)
        # shift boxes to global coords
        for b in seg.boxes:
            all_boxes.append(Box(b.x1 + x1, b.y1 + y1, b.x2 + x1, b.y2 + y1))
        all_masks.extend(seg.raw_masks)
        # Note: masks are per-tile local; for tiled mode we use boxes for counting,
        # masks are not globally accurate without re-encoding. Bench uses box count for AGAR.
    # NMS across all tiles
    keep = nms(all_boxes, 0.5)
    kept_boxes = [all_boxes[i] for i in keep]
    kept_masks = [all_masks[i] if i < len(all_masks) else "" for i in keep]
    # width/height is original image size
    return Segmentation(boxes=kept_boxes, raw_masks=kept_masks, width=w, height=h, cached=False)
