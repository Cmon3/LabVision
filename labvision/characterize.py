"""Characterization via Muse Spark Open vision: morphology -> Python morphotypes."""
from __future__ import annotations

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


def _cache_key(image_bytes: bytes, prompt: str) -> str:
    h = hashlib.sha256(image_bytes + prompt.encode()).hexdigest()
    return f"char_{h[:16]}.json"


def _contact_sheet(image_path: Path, boxes: list[dict], pad: float = 0.2, cols: int = 6, thumb: int = 128) -> Image.Image:
    """Build contact sheet of crops (pad 20%). boxes: list of {x,y,width,height}."""
    im = Image.open(image_path).convert("RGB")
    crops = []
    for b in boxes[:24]:
        x, y, w, h = b["x"], b["y"], b["width"], b["height"]
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
    return sheet


def _call_vision(image_bytes: bytes, n: int) -> list[dict]:
    import base64 as _b64

    image_b64 = _b64.b64encode(image_bytes).decode()
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key:
        raise RuntimeError("BASETEN_API_KEY missing")
    url = f"{BASETEN_URL.rstrip('/')}/chat/completions"
    prompt = (
        f"This contact sheet has {n} colony crops (6 columns, row-major indices 0..{n-1}). "
        f"For each crop, describe visible morphology only. Return a JSON array of {n} objects, each with: "
        f"color (e.g. white, cream, beige, yellow), "
        f"shape (circular, irregular, punctiform, filamentous), "
        f"margin (entire, undulate, lobate, filiform), "
        f"elevation (flat, raised, convex, umbonate), "
        f"opacity (opaque, translucent, transparent), "
        f"surface (smooth, rough, mucoid). "
        f"JSON array of {n} objects only, no other text."
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
        raise RuntimeError(f"Muse vision HTTP {r.status_code}: {r.text[:500]}")
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
    if not candidate.strip().startswith("["):
        candidate = reasoning + "\n" + content if reasoning else content
    try:
        start = candidate.find("[")
        end = candidate.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("no JSON array")
        arr = json.loads(candidate[start:end])
        for d in arr:
            # coerce any non-string values to string
            for k in ("color", "shape", "margin", "elevation", "opacity", "surface"):
                if k in d and not isinstance(d[k], str):
                    d[k] = str(d[k])
    except Exception:
        arr = [
            {"color": "unknown", "shape": "unknown", "margin": "unknown", "elevation": "unknown", "opacity": "unknown", "surface": "unknown"}
            for _ in range(n)
        ]
        return arr
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(arr))
    return arr


def _size_terciles(diameters: list[float]) -> list[str]:
    """Assign small/medium/large by terciles. All equal -> all medium."""
    if not diameters:
        return []
    sorted_d = sorted(diameters)
    n = len(sorted_d)
    # tercile boundaries at 1/3 and 2/3
    lo = sorted_d[n // 3]
    hi = sorted_d[2 * n // 3]
    if lo == hi:
        return ["medium"] * n
    out = []
    for d in diameters:
        if d <= lo:
            out.append("small")
        elif d <= hi:
            out.append("medium")
        else:
            out.append("large")
    return out


def _group_morphotypes(rows: list[dict], diameters_mm: list[float] | None) -> list[dict]:
    """Group colonies into morphotypes in Python. Returns rows annotated with morphotype."""
    if not rows:
        return rows
    # size class per colony
    if diameters_mm and len(diameters_mm) == len(rows):
        size_classes = _size_terciles(diameters_mm)
    else:
        size_classes = ["medium"] * len(rows)

    # build grouping key: identical morphology tuple + size
    # normalize to lowercase stripped
    def norm(v: str) -> str:
        return str(v).strip().lower() if v else "unknown"

    keys = []
    for i, r in enumerate(rows):
        k = (norm(r.get("color")), norm(r.get("shape")), norm(r.get("margin")), norm(r.get("elevation")), norm(r.get("opacity")), norm(r.get("surface")), size_classes[i])
        keys.append(k)

    # cluster identical keys -> morphotype
    unique_keys = list(dict.fromkeys(keys))  # preserve order
    # sort by frequency descending for label assignment A,B...
    from collections import Counter

    cnt = Counter(keys)
    # sort unique keys by count desc, then first occurrence
    unique_keys.sort(key=lambda k: (-cnt[k], keys.index(k)))
    label_map = {k: chr(ord("A") + i) for i, k in enumerate(unique_keys)}

    # per-morphotype mean diameter
    from collections import defaultdict

    mt_diams: dict[str, list[float]] = defaultdict(list)
    for i, k in enumerate(keys):
        lbl = label_map[k]
        if diameters_mm and i < len(diameters_mm):
            mt_diams[lbl].append(diameters_mm[i])

    # description per label
    mt_desc: dict[str, str] = {}
    for k, lbl in label_map.items():
        color, shape, margin, elevation, opacity, surface, sz = k
        mt_desc[lbl] = f"{color}, {shape}, {margin} margin, {elevation}, {opacity}, {surface}, {sz}"

    purity_flag = ""
    n_types = len(unique_keys)
    if n_types == 1:
        purity_flag = "Single morphotype (consistent with pure culture)"
    else:
        purity_flag = f"{n_types} morphotypes detected (possible mixed culture or contamination): confirm with standard microbiological methods."

    # annotate rows
    for i, r in enumerate(rows):
        lbl = label_map[keys[i]]
        r["morphotype"] = lbl
        r["morphotype_desc"] = mt_desc[lbl]
        r["size_class"] = size_classes[i]
        r["purity_flag"] = purity_flag
        # mean diameter for this morphotype
        dlist = mt_diams[lbl]
        r["mean_diameter_mm"] = sum(dlist) / len(dlist) if dlist else 0.0
        r["count_in_morphotype"] = cnt[keys[i]]

    return rows


def characterize(image_path: str | Path, boxes: list[dict], diameters_mm: list[float] | None = None) -> list[dict]:
    """Given image + boxes (x,y,width,height), return per-colony morphology + morphotype. Batches all colonies."""
    image_path = Path(image_path)
    if not boxes:
        return []
    # batch into chunks of 24
    all_results: list[dict] = []
    all_diams: list[float] = []
    for start in range(0, len(boxes), 24):
        chunk = boxes[start : start + 24]
        chunk_diams = diameters_mm[start : start + 24] if diameters_mm else None
        sheet = _contact_sheet(image_path, chunk)
        buf = BytesIO()
        sheet.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        try:
            arr = _call_vision(image_bytes, len(chunk))
        except Exception as e:
            if "image" in str(e).lower() or "vision" in str(e).lower():
                raise
            arr = [{"color": "unknown", "shape": "unknown", "margin": "unknown", "elevation": "unknown", "opacity": "unknown", "surface": "unknown"} for _ in chunk]
        for i, d in enumerate(arr[: len(chunk)]):
            global_idx = start + i
            all_results.append(
                {
                    "index": global_idx,
                    "color": d.get("color", "unknown"),
                    "shape": d.get("shape", "unknown"),
                    "margin": d.get("margin", "unknown"),
                    "elevation": d.get("elevation", "unknown"),
                    "opacity": d.get("opacity", "unknown"),
                    "surface": d.get("surface", "unknown"),
                }
            )
            if chunk_diams:
                all_diams.append(chunk_diams[i])
    # if diameters were provided, pass full list for grouping
    full_diams = diameters_mm if diameters_mm and len(diameters_mm) == len(boxes) else (all_diams if all_diams else None)
    if full_diams and len(full_diams) != len(all_results):
        # pad/truncate to match
        full_diams = (full_diams + [0.0] * len(all_results))[: len(all_results)]
    return _group_morphotypes(all_results, full_diams)
