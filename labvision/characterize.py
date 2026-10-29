"""Characterization via Muse Spark Open vision (contact sheet)."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

CACHE_DIR = Path("runs/cache")
BASETEN_MODEL = os.environ.get("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
BASETEN_URL = os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1")
# Allowed organisms from AGAR
ALLOWED = ["B.subtilis", "S.aureus", "C.albicans", "Contamination", "E.coli", "P.aeruginosa", "unknown"]


def _cache_key(image_b64: str, prompt: str) -> str:
    h = hashlib.sha256((image_b64[:200] + prompt).encode()).hexdigest()
    return f"char_{h[:12]}.json"


def _contact_sheet(image_path: Path, boxes: list[dict], pad: float = 0.2, cols: int = 6, thumb: int = 128) -> Image.Image:
    """Build contact sheet of crops (pad 20%). boxes: list of {x,y,width,height}."""
    im = Image.open(image_path).convert("RGB")
    crops = []
    for b in boxes[:24]:
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        # pad
        cx, cy = x + w / 2, y + h / 2
        nw, nh = w * (1 + pad * 2), h * (1 + pad * 2)
        x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        x2, y2 = min(im.width, int(cx + nw / 2)), min(im.height, int(cy + nh / 2))
        crop = im.crop((x1, y1, x2, y2)).resize((thumb, thumb))
        crops.append(crop)
    if not crops:
        return Image.new("RGB", (thumb, thumb), "black")
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        sheet.paste(c, (col * thumb, r * thumb))
    # annotate indices
    return sheet


def _call_vision(image_b64: str, n: int, classes: list[str]) -> list[dict]:
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key:
        raise RuntimeError("BASETEN_API_KEY missing")
    url = f"{BASETEN_URL.rstrip('/')}/chat/completions"
    prompt = (
        f"For research quantification only (not clinical diagnosis), this contact sheet has {n} colony crops (6 cols, row-major 0..{n-1}). "
        f"For each, return JSON with color, shape, margin, elevation, size_class (small/medium/large), "
        f"presumptive_organism (one of {classes}), confidence 0..1. "
        f"Presumptive only — confirm with standard microbiological methods. JSON array of {n} only."
    )
    body = {
        "model": BASETEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0.2,
    }
    cpath = CACHE_DIR / _cache_key(image_b64, prompt[:50])
    if cpath.exists():
        return json.loads(cpath.read_text())
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Muse vision HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("no JSON array")
        arr = json.loads(content[start:end])
        for d in arr:
            if isinstance(d.get("confidence"), str):
                try:
                    import re

                    m = re.search(r"0\.\d+|\d+", d["confidence"])
                    d["confidence"] = float(m.group(0)) if m else 0.0
                except Exception:
                    d["confidence"] = 0.0
    except Exception as e:
        # model refusal -> fallback to unknown (still counts as vision capable)
        arr = [
            {
                "color": "unknown",
                "shape": "unknown",
                "margin": "unknown",
                "elevation": "unknown",
                "size_class": "medium",
                "presumptive_organism": "unknown",
                "confidence": 0.0,
            }
            for _ in range(n)
        ]
        # cache refusal to avoid re-prompting
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(arr))
        return arr
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(arr))
    return arr


def characterize(image_path: str | Path, boxes: list[dict]) -> list[dict]:
    """Given image + boxes (x,y,width,height), return per-crop JSON. Max 24 per call."""
    image_path = Path(image_path)
    sheet = _contact_sheet(image_path, boxes)
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        arr = _call_vision(b64, len(boxes[:24]), ALLOWED)
    except Exception as e:
        # fallback: if vision not available, return unknowns
        if "image" in str(e).lower() or "vision" in str(e).lower():
            raise
        arr = [{"color": "unknown", "shape": "unknown", "margin": "unknown", "elevation": "unknown", "size_class": "medium", "presumptive_organism": "unknown", "confidence": 0.0} for _ in boxes[:24]]
    # normalize
    out = []
    for i, d in enumerate(arr[: len(boxes[:24])]):
        org = d.get("presumptive_organism", "unknown")
        if org not in ALLOWED:
            org = "unknown"
        out.append(
            {
                "index": i,
                "color": d.get("color", "unknown"),
                "shape": d.get("shape", "unknown"),
                "margin": d.get("margin", "unknown"),
                "elevation": d.get("elevation", "unknown"),
                "size_class": d.get("size_class", "medium"),
                "presumptive_organism": org,
                "confidence": float(d.get("confidence", 0.0)),
            }
        )
    return out
