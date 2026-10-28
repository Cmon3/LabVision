"""SAM 3.1 client (T1): text prompt in, boxes + raw mask strings out.

Response grammar for masks is not yet in docs/sam_api.md, so masks are
kept as raw strings; boxes are parsed. Every API call is cached on disk
by image hash + prompt so reruns are free.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

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
    raw_masks: list[str]  # undecoded payloads; full grammar pending
    width: int
    height: int
    cached: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


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
    return "image/png"


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


def segment_file(image_path: str | Path, prompt: str) -> Segmentation:
    """Convenience wrapper for T2 probe: read file -> segment."""
    return segment(Path(image_path).read_bytes(), prompt)
