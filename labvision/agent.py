"""Muse Spark agent: tool loop via Baseten (OpenAI-compatible)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests
import numpy as np

from labvision import measure as measure_mod
from labvision import qc as qc_mod
from labvision.segment import segment

CACHE_DIR = Path("runs/cache")
BASETEN_MODEL = os.environ.get("BASETEN_MODEL", "meta-models/muse-spark-open-1026")
BASETEN_URL = os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1")

# Tool schemas for OpenAI-compatible tool calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_quality",
            "description": "Check image quality: blur, glare, empty, crowded. Call first.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment",
            "description": "Segment image for a noun phrase (one concept, e.g. 'bacterial colony', 'cell'). Returns count and image size.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "short noun phrase"}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure",
            "description": "Measure current masks: count, areas, diameters. Optionally filter by min_diameter_mm (needs plate_diameter_px).",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_diameter_mm": {"type": "number"},
                    "plate_diameter_px": {"type": "number"},
                    "plate_diameter_mm": {"type": "number"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_rephoto",
            "description": "Ask user to retake photo (blur/glare/empty/crowded).",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]

SYSTEM = """You are LabVision Copilot, a lab assistant. You have tools to inspect images.
Rules:
- Always call check_quality first.
- If quality fails (blurry, glare, empty, crowded) call ask_rephoto.
- Otherwise call segment with a short noun phrase for the objects the user asked about, then measure.
- Do NOT invent numbers. The final report numbers will be filled by code from tool outputs.
- Provide a short 2-sentence interpretation for the report (no numbers needed, they are inserted).
"""

REPORT_TEMPLATE = """LabVision Report
Request: {request}
Image: {image_name}

Quality: {qc_summary}
Count: {count} objects
{filter_line}
Mean diameter: {mean_d:.1f} px ({mean_mm:.2f} mm)
Diameters (px): min {min_d:.1f} max {max_d:.1f}
Areas (px): mean {mean_a:.0f}

Note: {assistant_note}

{rephoto_line}
"""


def _cache_key(messages, tools, model) -> str:
    h = hashlib.sha256(json.dumps({"m": messages, "t": tools, "model": model}, sort_keys=True).encode()).hexdigest()
    return f"muse_{h[:16]}.json"


def _call_muse(messages: list[dict], tools=TOOLS, use_cache: bool = True) -> dict:
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key:
        raise RuntimeError("Set BASETEN_API_KEY")
    url = f"{BASETEN_URL.rstrip('/')}/chat/completions"
    body = {"model": BASETEN_MODEL, "messages": messages, "tools": tools, "tool_choice": "auto"}
    # cache
    cpath = CACHE_DIR / _cache_key(messages, tools, BASETEN_MODEL)
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text())
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Muse HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(data))
    return data


def _load_image(path: str | Path) -> tuple[np.ndarray, bytes]:
    p = Path(path)
    data = p.read_bytes()
    # decode via PIL for qc
    from PIL import Image

    im = Image.open(p).convert("RGB")
    arr = np.array(im)
    return arr, data


def run_agent(image_path: str | Path, request: str, use_cache: bool = True) -> dict[str, Any]:
    """Run agent loop. Returns dict with report, count, rows, qc, etc. Numbers from measure only."""
    image_path = Path(image_path)
    img_arr, img_bytes = _load_image(image_path)
    # state
    seg_result = None
    measure_rows: list[dict] = []
    qc_result: dict | None = None
    rephoto_reason: str | None = None
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Image: {image_path.name}\nRequest: {request}\nUse tools to answer."},
    ]
    # ponytail: max 4 tool rounds
    for _ in range(5):
        resp = _call_muse(messages, use_cache=use_cache)
        choice = resp["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            # final assistant note
            assistant_note = choice.get("content") or "Analysis complete."
            break
        # execute tools
        messages.append(choice)
        for tc in tool_calls:
            fname = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            try:
                if fname == "check_quality":
                    bv = qc_mod.laplacian_variance(img_arr)
                    result = {
                        "blur_score": bv,
                        "is_blurry": qc_mod.is_blurry(img_arr),
                        "glare": qc_mod.has_glare(img_arr),
                        "glare_frac": qc_mod.saturated_fraction(img_arr),
                    }
                    qc_result = result
                elif fname == "segment":
                    prompt = args.get("prompt", "bacterial colony")
                    seg_result = segment(img_bytes, prompt)
                    masks = seg_result.decoded_masks()
                    result = {"count": len(seg_result.boxes), "width": seg_result.width, "height": seg_result.height, "masks": int(masks.shape[0])}
                elif fname == "measure":
                    if seg_result is None:
                        result = {"error": "call segment first"}
                    else:
                        masks = seg_result.decoded_masks()
                        plate_px = args.get("plate_diameter_px")
                        plate_mm = args.get("plate_diameter_mm", 90.0)
                        rows = measure_mod.measure_masks(masks, plate_diameter_px=plate_px, plate_diameter_mm=plate_mm)
                        if "min_diameter_mm" in args:
                            rows = measure_mod.filter_by_min_diameter(rows, args["min_diameter_mm"])
                        measure_rows = rows
                        result = {"count": len(rows), "rows": rows[:5]}  # truncate for LLM
                elif fname == "ask_rephoto":
                    rephoto_reason = args.get("reason", "")
                    result = {"asked": True, "reason": rephoto_reason}
                else:
                    result = {"error": f"unknown tool {fname}"}
            except Exception as e:
                result = {"error": str(e)[:200]}
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
    else:
        assistant_note = "Analysis complete."

    # Fill report numbers from tool results only
    qc_summary = "OK"
    if qc_result:
        flags = []
        if qc_result.get("is_blurry"):
            flags.append(f"blurry (score {qc_result['blur_score']:.1f})")
        if qc_result.get("glare"):
            flags.append(f"glare {qc_result['glare_frac']:.1%}")
        if flags:
            qc_summary = ", ".join(flags)
        else:
            qc_summary = f"sharp (score {qc_result['blur_score']:.1f}), no glare"
    # check empty/crowded via final count (fix false empty when count>0)
    if seg_result is not None:
        try:
            masks = seg_result.decoded_masks()
            cov = qc_mod.mask_coverage(masks, seg_result.width * seg_result.height) if masks.size else 0
            final_n = len(measure_rows) if measure_rows else len(seg_result.boxes)
            if qc_mod.is_empty(final_n):
                qc_summary += "; empty (no objects)"
            elif qc_mod.is_crowded(final_n, cov):
                qc_summary += f"; crowded (count {final_n}, coverage {cov:.0%})"
        except Exception:
            pass

    count = len(measure_rows) if measure_rows else (len(seg_result.boxes) if seg_result else 0)
    if measure_rows:
        ds = [r["diameter_px"] for r in measure_rows]
        areas = [r["area_px"] for r in measure_rows]
        mean_d = sum(ds) / len(ds) if ds else 0
        mean_a = sum(areas) / len(areas) if areas else 0
        mean_mm = sum(r.get("diameter_mm", 0) for r in measure_rows) / len(measure_rows) if measure_rows and "diameter_mm" in measure_rows[0] else 0
        min_d, max_d = min(ds) if ds else 0, max(ds) if ds else 0
        filter_line = ""
    elif seg_result and seg_result.boxes:
        # fallback from boxes if measure not called
        mean_d = min_d = max_d = mean_a = mean_mm = 0
        filter_line = "(measure not run)"
    else:
        mean_d = min_d = max_d = mean_a = mean_mm = 0
        filter_line = ""

    # detect filter in request (e.g. "larger than 2 mm")
    import re

    m = re.search(r"larger than\s+([\d.]+)\s*mm", request, re.I)
    filter_line = f"Filter: diameter >= {m.group(1)} mm" if m else ""

    rephoto_line = f"Action: Please retake photo — {rephoto_reason}" if rephoto_reason else ""

    report = REPORT_TEMPLATE.format(
        request=request,
        image_name=image_path.name,
        qc_summary=qc_summary,
        count=count,
        filter_line=filter_line,
        mean_d=mean_d,
        mean_mm=mean_mm,
        min_d=min_d,
        max_d=max_d,
        mean_a=mean_a,
        assistant_note=assistant_note.strip() if "assistant_note" in locals() else "",
        rephoto_line=rephoto_line,
    )

    return {
        "report": report,
        "count": count,
        "rows": measure_rows,
        "qc": qc_result,
        "rephoto_reason": rephoto_reason,
        "assistant_note": locals().get("assistant_note", ""),
        "seg": {"boxes": len(seg_result.boxes) if seg_result else 0, "width": seg_result.width if seg_result else 0, "height": seg_result.height if seg_result else 0},
    }
