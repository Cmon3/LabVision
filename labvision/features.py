"""Measured colony features: color (Lab bg-subtracted), circularity, log diameter, texture."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    # image RGB uint8 HxWx3 -> Lab via OpenCV (expects BGR but we use RGB>Lab via conversion)
    # cv2.COLOR_RGB2Lab gives L 0-255
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)


def _masks_from_boxes(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    """GT boxes -> per-colony masks via Otsu inside each box. Returns (N,H,W) bool."""
    h, w = image.shape[:2]
    n = len(boxes)
    out = np.zeros((n, h, w), dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    for i, b in enumerate(boxes):
        x, y, bw, bh = int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            from skimage.filters import threshold_otsu

            thresh = threshold_otsu(crop)
            mask_crop = crop > thresh
            # Otsu may pick background as foreground if colony is dark: ensure mask is smaller than half
            # heuristic: colony should not be >80% of box (agar). If mask is too large, invert.
            if mask_crop.mean() > 0.7:
                mask_crop = ~mask_crop
            # if still too large or too small, fallback to central ellipse
            if mask_crop.sum() < 10 or mask_crop.mean() > 0.9:
                # fallback: use box center as circle
                mask_crop = np.zeros_like(mask_crop, dtype=bool)
                cy, cx = mask_crop.shape[0] // 2, mask_crop.shape[1] // 2
                r = min(cx, cy) // 2
                yy, xx = np.ogrid[: mask_crop.shape[0], : mask_crop.shape[1]]
                mask_crop[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True
            out[i, y1:y2, x1:x2] = mask_crop
        except Exception:
            # fallback: whole box
            out[i, y1:y2, x1:x2] = True
    return out


def extract_features(image: np.ndarray, boxes: list[dict] | None = None, masks: np.ndarray | None = None) -> np.ndarray:
    """Return (N,6) features: L_adj, a_adj, b_adj, circularity, log_diam, texture.

    image: HxWx3 RGB uint8
    boxes: list of {x,y,width,height} (needed for bg estimation if masks not covering)
    masks: (N,H,W) bool - if None, derived from boxes via Otsu.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {image.shape}")
    h, w = image.shape[:2]

    if masks is None:
        if not boxes:
            raise ValueError("need boxes or masks")
        masks = _masks_from_boxes(image, boxes)
    else:
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3 or masks.shape[1] != h or masks.shape[2] != w:
            raise ValueError(f"masks shape {masks.shape} mismatch image {h}x{w}")

    n = masks.shape[0]
    if n == 0:
        return np.zeros((0, 6), dtype=float)

    lab = _rgb_to_lab(image)  # HxWx3 float
    # background: pixels outside all colony masks/boxes
    union = np.any(masks, axis=0) if n > 0 else np.zeros((h, w), dtype=bool)
    # if union is too large (>90% image), fallback to boxes outside estimate: use image border as bg
    bg_mask = ~union
    if bg_mask.sum() < 100:
        # fallback: border 10px as bg
        bg_mask = np.zeros((h, w), dtype=bool)
        bg_mask[:10, :] = True
        bg_mask[-10:, :] = True
        bg_mask[:, :10] = True
        bg_mask[:, -10:] = True
    bg_vals = lab[bg_mask]  # (M,3)
    bg_median = np.median(bg_vals, axis=0) if bg_vals.size else np.array([128, 128, 128], dtype=float)

    feats = np.zeros((n, 6), dtype=float)
    for i in range(n):
        m = masks[i]
        if m.sum() == 0:
            continue
        # color: mean Lab minus bg median
        colony_lab = lab[m]  # (K,3)
        mean_lab = colony_lab.mean(axis=0)
        adj = mean_lab - bg_median
        feats[i, 0:3] = adj
        # area/perimeter for circularity and diameter
        area = float(m.sum())
        # perimeter via contour
        m_uint8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perim = 0.0
        for cnt in contours:
            perim += cv2.arcLength(cnt, True)
        if perim > 0:
            circ = 4 * math.pi * area / (perim * perim)
            circ = min(circ, 1.0)
        else:
            circ = 0.0
        feats[i, 3] = circ
        # log equivalent diameter
        equiv_d = math.sqrt(4 * area / math.pi) if area > 0 else 1.0
        feats[i, 4] = math.log(max(equiv_d, 1.0))
        # texture: std of L* inside colony
        feats[i, 5] = float(colony_lab[:, 0].std()) if colony_lab.shape[0] > 1 else 0.0

    return feats


def load_image_rgb(path: str | bytes) -> np.ndarray:
    if isinstance(path, (bytes, bytearray)):
        import io

        im = Image.open(io.BytesIO(path)).convert("RGB")
    else:
        im = Image.open(path).convert("RGB")
    return np.array(im)
