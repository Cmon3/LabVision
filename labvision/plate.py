"""Plate detection via HoughCircles (OpenCV)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def detect_plate_diameter_px(image: np.ndarray | bytes) -> float | None:
    """Find largest circular plate via HoughCircles. Returns diameter in px or None."""
    try:
        import cv2

        if isinstance(image, bytes):
            import io

            im = Image.open(io.BytesIO(image)).convert("RGB")
            arr = np.array(im)
        else:
            arr = np.asarray(image)
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.astype(np.uint8)
        gray = cv2.medianBlur(gray, 5)
        h, w = gray.shape
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=h // 4, param1=100, param2=150, minRadius=min(h, w) // 4, maxRadius=max(h, w))
        if circles is None:
            return None
        circles = np.uint16(np.around(circles))
        _x, _y, r = max(circles[0, :], key=lambda c: c[2])
        return float(r * 2)
    except Exception:
        return None


def crop_to_plate(image_bytes: bytes, pad: float = 0.02) -> bytes | None:
    """Try to find largest circle and crop. Returns new PNG bytes or None if not found."""
    try:
        import cv2
        import io

        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(im)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        h, w = gray.shape
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=h // 4, param1=100, param2=150, minRadius=min(h, w) // 4, maxRadius=max(h, w))
        if circles is None:
            return None
        circles = np.uint16(np.around(circles))
        # largest circle
        x, y, r = max(circles[0, :], key=lambda c: c[2])
        r = int(r * (1 + pad))
        x1, y1, x2, y2 = max(0, x - r), max(0, y - r), min(w, x + r), min(h, y + r)
        crop = im.crop((x1, y1, x2, y2))
        out = io.BytesIO()
        crop.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None
