"""SAM 3.1 client: text prompt in, boxes + masks out (decoded one_bit via mask_codec)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests

from labvision.mask_codec import decode_mask

API_URL = "https://api.meta.ai/v1/responses"
MODEL = "sam-3.1"
CACHE_DIR = Path("runs/cache")

BOX_RE = re.compile(
    r"<\s*\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)[^|]*\|>"
)
MASK_RE = re.compile(r"<\s*\|mask;x=(\d+);y=(\d+);data=([^|]*)\|>")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Segmentation:
    boxes: list[Box]
    raw_masks: list[str]  # e.g. "683,692,!!!!..."
    width: int
    height: int
    cached: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def decoded_masks(self, *args, **kwargs) -> np.ndarray:
        """Decode all masks to (N, H, W) bool array.

        SAM returns each mask as a raster the size of its box at (x1,y1).
        We place the HxW raster at the box origin in a full-size zero image.
        Resize only if H,W differ from box size by a few px.
        """
        if not self.raw_masks:
            return np.zeros((0, self.height, self.width), dtype=bool)
        decoded: list[np.ndarray] = []
        for i, rm in enumerate(self.raw_masks):
            try:
                h_str, w_str, payload = rm.split(",", 2)
                h, w = int(h_str), int(w_str)
                small = decode_mask(payload, w, h)
            except Exception:
                continue
            # full-frame mask (legacy) -> use as is
            if small.shape[0] == self.height and small.shape[1] == self.width:
                decoded.append(small.astype(bool))
                continue
            # box-anchored mask
            if i < len(self.boxes):
                b = self.boxes[i]
                bw = b.x2 - b.x1
                bh = b.y2 - b.y1
                # resize if off by a few px
                if small.shape[0] != bh or small.shape[1] != bw:
                    if abs(small.shape[0] - bh) <= 5 and abs(small.shape[1] - bw) <= 5:
                        try:
                            from PIL import Image

                            im = Image.fromarray((small.astype(np.uint8) * 255))
                            small = np.array(im.resize((bw, bh), Image.NEAREST)) > 127
                        except Exception:
                            pass
                arr = np.zeros((self.height, self.width), dtype=bool)
                # clamp box to image bounds
                x1 = max(0, b.x1)
                y1 = max(0, b.y1)
                x2 = min(self.width, b.x2)
                y2 = min(self.height, b.y2)
                # source region in small that maps to clamped box
                # if box was clamped, we need to offset in small
                sx1 = x1 - b.x1
                sy1 = y1 - b.y1
                sx2 = sx1 + (x2 - x1)
                sy2 = sy1 + (y2 - y1)
                # also handle small larger/smaller after resize mismatch
                sx1 = max(0, sx1)
                sy1 = max(0, sy1)
                sx2 = min(small.shape[1], sx2)
                sy2 = min(small.shape[0], sy2)
                # adjust dest if source was clipped
                dx1 = x1 + max(0, b.x1 - x1)
                dy1 = y1 + max(0, b.y1 - y1)
                # but with clamping above, sx already accounts; compute actual paste size
                h_paste = sy2 - sy1
                w_paste = sx2 - sx1
                if h_paste > 0 and w_paste > 0:
                    # ensure dest fits
                    dx1 = x1
                    dy1 = y1
                    # if small had different size,sx/sy already limited
                    arr[dy1 : dy1 + h_paste, dx1 : dx1 + w_paste] = small[sy1:sy2, sx1:sx2]
                decoded.append(arr)
            else:
                # no box: treat as full-frame if possible, else place at 0,0
                arr = np.zeros((self.height, self.width), dtype=bool)
                h0 = min(small.shape[0], self.height)
                w0 = min(small.shape[1], self.width)
                arr[:h0, :w0] = small[:h0, :w0].astype(bool)
                decoded.append(arr)
        if not decoded:
            return np.zeros((0, self.height, self.width), dtype=bool)
        return np.stack(decoded)

    def masks_array(self) -> np.ndarray:
        """Alias for decoded_masks for bench compatibility."""
        return self.decoded_masks()


def _api_key() -> str:
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise RuntimeError("Set META_API_KEY in env (see .env.example).")
    return key


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _slug(prompt: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", prompt.strip().lower()).strip("-")
    return s[:40] or "prompt"


def _cache_path(img_hash: str, prompt: str) -> Path:
    return CACHE_DIR / f"{img_hash[:16]}_{_slug(prompt)}_{MODEL}.json"


def _guess_mime(image_bytes: bytes) -> str:
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "image/png"


def _to_png_bytes(image_bytes: bytes) -> tuple[bytes, str]:
    # Convert TIFF/other to PNG for SAM API (which expects png/jpeg)
    if image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        try:
            from PIL import Image
            import io

            im = Image.open(io.BytesIO(image_bytes))
            # normalize to 8-bit
            if im.mode not in ("RGB", "RGBA", "L"):
                im = im.convert("RGB")
            elif im.mode == "RGBA":
                im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="PNG")
            return out.getvalue(), "image/png"
        except Exception:
            pass
    return image_bytes, _guess_mime(image_bytes)


def _extract_output_text(payload: dict | list) -> str:
    """Pull output_text out of a Responses-API payload (shape varies)."""
    texts: list[str] = []

    def walk(o: object) -> None:
        if isinstance(o, dict):
            t = o.get("type")
            if isinstance(o.get("output_text"), str):
                texts.append(o["output_text"])
            elif t in ("output_text",) and isinstance(o.get("text"), str):
                texts.append(o["text"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(payload)
    return "".join(texts)


def parse_output_text(output_text: str) -> tuple[list[Box], list[str], int, int]:
    """Parse boxes; keep mask payloads raw. Empty string -> no detections."""
    boxes = [Box(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, _, _ in BOX_RE.findall(output_text)]
    raw_masks = [m[2] for m in MASK_RE.findall(output_text)]
    wh = BOX_RE.search(output_text)
    w, h = (int(wh.group(5)), int(wh.group(6))) if wh else (0, 0)
    return boxes, raw_masks, w, h


def _call_sam(image_bytes: bytes, prompt: str, mime: str) -> str:
    image_bytes, mime = _to_png_bytes(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    body = {
        "model": MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ],
        "stream": False,
        "metadata": {"mask_encoding": "one_bit"},
    }
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {_api_key()}"},
        json=body,
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"SAM 3.1 HTTP {r.status_code}: {r.text[:500]}")
    return _extract_output_text(r.json())


def segment(
    image_bytes: bytes,
    prompt: str,
    mime: str = "",
    use_cache: bool = True,
) -> Segmentation:
    """Segment one image for one noun-phrase prompt (one concept per request)."""
    mime = mime or _guess_mime(image_bytes)
    img_hash = _image_hash(image_bytes)
    path = _cache_path(img_hash, prompt)
    if use_cache and path.exists():
        d = json.loads(path.read_text())
        return Segmentation(
            boxes=[Box(**b) for b in d["boxes"]],
            raw_masks=d["raw_masks"],
            width=d["width"],
            height=d["height"],
            cached=True,
        )
    output_text = _call_sam(image_bytes, prompt, mime)
    boxes, raw_masks, w, h = parse_output_text(output_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompt": prompt,
                "model": MODEL,
                "image_sha256": img_hash,
                "output_text": output_text,
                "boxes": [asdict(b) for b in boxes],
                "raw_masks": raw_masks,
                "width": w,
                "height": h,
            }
        )
    )
    return Segmentation(boxes=boxes, raw_masks=raw_masks, width=w, height=h)


def _build_point_prompt(x: int, y: int, w: int, h: int, obj_id: int = 1) -> str:
    return f"{obj_id}< |point;x={x};y={y};w={w};h={h}|>"

def _build_box_prompt(x1: int, y1: int, x2: int, y2: int, w: int, h: int, obj_id: int = 1) -> str:
    return f"{obj_id}< |box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>"

def segment_point(
    image_bytes: bytes,
    x: int,
    y: int,
    x2: int | None = None,
    y2: int | None = None,
    use_cache: bool = True,
) -> Segmentation:
    """Segment one image at a point or box. x,y are image pixels; x2,y2 for drag box (inclusive)."""
    try:
        from PIL import Image
        import io as _io

        im = Image.open(_io.BytesIO(image_bytes))
        w, h = im.size
    except Exception:
        w, h = 1024, 1024
    # if drag rectangle provided and large enough, use box prompt
    if x2 is not None and y2 is not None:
        bw = abs(x2 - x)
        bh = abs(y2 - y)
        if max(bw, bh) >= 6:
            x1 = max(0, min(x, x2))
            y1 = max(0, min(y, y2))
            x2c = min(w - 1, max(x, x2))
            y2c = min(h - 1, max(y, y2))
            prompt = _build_box_prompt(x1, y1, x2c, y2c, w, h)
        else:
            prompt = _build_point_prompt(int(x), int(y), w, h)
    else:
        prompt = _build_point_prompt(int(x), int(y), w, h)
    return segment(image_bytes, prompt, use_cache=use_cache)

def segment_file(image_path: str | Path, prompt: str) -> Segmentation:
    """Convenience wrapper for T2 probe: read file -> segment."""
    return segment(Path(image_path).read_bytes(), prompt)
