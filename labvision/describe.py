"""Muse wording per morphotype group (constrained vocab, validated)."""
from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

CACHE_DIR = Path("runs/cache")
BASETEN_MODEL = os.environ.get("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
BASETEN_URL = os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1")

VALID_VOCAB = {
    "color": {"white", "cream", "yellow", "orange", "green", "grey", "beige", "brown"},
    "shape": {"circular", "irregular", "punctiform"},
    "margin": {"entire", "undulate", "lobate"},
    "elevation": {"flat", "raised", "convex"},
    "surface": {"smooth", "rough", "mucoid"},
}


def _cache_key(image_bytes: bytes, prompt: str) -> str:
    h = hashlib.sha256(image_bytes + prompt.encode()).hexdigest()
    return f"desc_{h[:16]}.json"


def _contact_sheet_group(image_path: Path, boxes: list[dict], pad: float = 0.2, thumb: int = 128) -> Image.Image:
    """Contact sheet of up to 6 crops for one group."""
    im = Image.open(image_path).convert("RGB")
    crops = []
    for b in boxes[:6]:
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
        cx, cy = x + w / 2, y + h / 2
        nw, nh = w * (1 + pad * 2), h * (1 + pad * 2)
        x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        x2, y2 = min(im.width, int(cx + nw / 2)), min(im.height, int(cy + nh / 2))
        crop = im.crop((x1, y1, x2, y2)).resize((thumb, thumb))
        crops.append(crop)
    if not crops:
        return Image.new("RGB", (thumb, thumb), "black")
    # single row up to 6
    sheet = Image.new("RGB", (len(crops) * thumb, thumb), "white")
    for i, c in enumerate(crops):
        sheet.paste(c, (i * thumb, 0))
    return sheet


def _validate(desc: dict) -> dict:
    out = {}
    for k, allowed in VALID_VOCAB.items():
        v = str(desc.get(k, "")).strip().lower()
        # allow synonyms mapping? strict: must be in allowed
        if v not in allowed:
            # try to extract first allowed word found inside
            found = None
            for a in allowed:
                if a in v:
                    found = a
                    break
            v = found if found else next(iter(allowed))  # fallback to arbitrary allowed (first)
            # for deterministic fallback, pick most common fallback: white for color, circular for shape, etc.
            fallback = {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}
            if v not in allowed:
                v = fallback[k]
        out[k] = v
    return out


def describe_group(image_path: str | Path, boxes: list[dict], feature_summary: dict | None = None) -> dict:
    """Describe one morphotype group via Muse. Returns validated dict."""
    image_path = Path(image_path)
    if not boxes:
        return {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}

    sheet = _contact_sheet_group(image_path, boxes)
    buf = BytesIO()
    sheet.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    import base64 as _b64

    image_b64 = _b64.b64encode(image_bytes).decode()
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key:
        raise RuntimeError("BASETEN_API_KEY missing")
    url = f"{BASETEN_URL.rstrip('/')}/chat/completions"

    feat_str = ""
    if feature_summary:
        feat_str = f" Python features: color Lab adj {feature_summary.get('lab', [])}, circularity {feature_summary.get('circularity', 0):.2f}, log diam {feature_summary.get('log_diam', 0):.2f}, texture {feature_summary.get('texture', 0):.2f}."

    prompt = (
        f"This contact sheet shows {len(boxes[:6])} representative colony crops from one morphotype group.{feat_str} "
        f"Describe the group using ONLY these words: "
        f"color one of white/cream/yellow/orange/green/grey/beige/brown; "
        f"shape one of circular/irregular/punctiform; "
        f"margin one of entire/undulate/lobate; "
        f"elevation one of flat/raised/convex; "
        f"surface one of smooth/rough/mucoid. "
        f"Return JSON with keys color, shape, margin, elevation, surface only. No other text."
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

    cpath = CACHE_DIR / _cache_key(image_bytes, prompt)
    if cpath.exists():
        return json.loads(cpath.read_text())

    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Muse HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    raw_save = (reasoning + "\n" + content) if reasoning else content
    raw_dir = Path("runs/characterize_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{_cache_key(image_bytes, prompt)[:-5]}.txt"
    raw_path.write_text(raw_save[:5000])

    candidate = content
    if not candidate.strip().startswith("{"):
        candidate = reasoning + "\n" + content if reasoning else content
    try:
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("no JSON")
        desc = json.loads(candidate[start:end])
    except Exception:
        desc = {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}
        return _validate(desc)

    validated = _validate(desc)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(validated))
    return validated
