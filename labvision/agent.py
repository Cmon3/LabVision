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
            "description": "Segment image for a noun phrase (one concept, e.g. 'round spot', 'cell'). Returns count and image size.",
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
            "description": "Measure current masks: count, areas, diameters (mm if plate detected, otherwise px). Optionally filter by min_diameter_mm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_diameter_mm": {"type": "number", "description": "filter: keep only colonies >= this diameter in mm"},
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
    {
        "type": "function",
        "function": {
            "name": "characterize",
            "description": "Characterize colony morphology (call after segment+measure). Returns morphotypes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM = """You are LabVision Copilot, a lab assistant. You have tools to inspect images.
Rules:
- Always call check_quality first.
- If quality fails (blurry, glare, empty, crowded) call ask_rephoto.
- Otherwise call segment with a short noun phrase for the objects the user asked about, then measure. Optionally call characterize for morphology.
- Do NOT invent numbers. The final report numbers will be filled by code from tool outputs.
- Provide a short 2-sentence interpretation for the report (no numbers needed, they are inserted).
"""

REPORT_TEMPLATE = """LabVision Report
Request: {request}
Image: {image_name}

Quality: {qc_summary}
Count: {count_display}
{filter_line}
Mean diameter: {mean_d:.1f} px{mm_part}
Diameters (px): min {min_d:.1f} max {max_d:.1f}
Areas (px): mean {mean_a:.0f}
{plate_note}

{char_section}
Note: {assistant_note}

{rephoto_line}
Confirm with standard microbiological methods.
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
    char_results: list[dict] = []
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
                    from labvision.config import AGAR_PROMPT as _DEFAULT_PROMPT

                    prompt = args.get("prompt", _DEFAULT_PROMPT)
                    seg_result = segment(img_bytes, prompt)
                    # retry once if 0 but QC says not empty (sharp, no glare)
                    if len(seg_result.boxes) == 0 and qc_result is not None and not qc_result.get("is_blurry") and not qc_result.get("glare"):
                        try:
                            from labvision.config import AGAR_ALT_PROMPT

                            alt = AGAR_ALT_PROMPT
                            if alt and alt != prompt:
                                seg_result = segment(img_bytes, alt)
                                prompt = alt
                        except Exception:
                            pass
                    masks = seg_result.decoded_masks()
                    result = {"count": len(seg_result.boxes), "width": seg_result.width, "height": seg_result.height, "masks": int(masks.shape[0]), "prompt_used": prompt}
                elif fname == "measure":
                    if seg_result is None:
                        result = {"error": "call segment first"}
                    else:
                        masks = seg_result.decoded_masks()
                        # plate diameter from Python, never from model
                        try:
                            from labvision.plate import detect_plate_diameter_px as _plate_px

                            _plate_px_val = _plate_px(img_arr)
                        except Exception:
                            _plate_px_val = None
                        if _plate_px_val:
                            rows = measure_mod.measure_masks(masks, plate_diameter_px=_plate_px_val, plate_diameter_mm=90.0)
                            _plate_info = {"plate_detected": True, "plate_diameter_px": _plate_px_val}
                        else:
                            rows = measure_mod.measure_masks(masks)
                            _plate_info = {"plate_detected": False}
                        if "min_diameter_mm" in args:
                            if rows and "diameter_mm" in rows[0]:
                                rows = measure_mod.filter_by_min_diameter(rows, args["min_diameter_mm"])
                            else:
                                result = {"error": "no plate detected — sizes in px, cannot filter by mm", "plate_detected": False}
                                measure_rows = rows
                                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
                                continue
                        measure_rows = rows
                        result = {"count": len(rows), "rows": rows[:5], **_plate_info}
                elif fname == "ask_rephoto":
                    rephoto_reason = args.get("reason", "")
                    result = {"asked": True, "reason": rephoto_reason}
                elif fname == "characterize":
                    if seg_result is None:
                        result = {"error": "call segment first"}
                    else:
                        try:
                            from labvision.features import extract_features as _feat
                            from labvision.morphotypes import cluster_plate as _clust
                            from labvision.config import MORPHOTYPE_THRESHOLD as _T, MORPHOTYPE_COLOR_WEIGHT as _CW

                            # build masks/boxes for features
                            masks = seg_result.decoded_masks()
                            boxes_for_char = [{"x": b.x1, "y": b.y1, "width": b.x2 - b.x1, "height": b.y2 - b.y1} for b in seg_result.boxes]
                            # extract features: prefer masks if available and valid
                            if masks.size and masks.shape[0] == len(seg_result.boxes):
                                feats = _feat(img_arr, masks=masks)
                            else:
                                feats = _feat(img_arr, boxes=boxes_for_char)
                            labels = _clust(feats, threshold=_T, color_weight=_CW)
                            # per-group Muse wording
                            uniq = sorted(set(labels))
                            # representative crops per group + feature summary
                            char_results = []
                            group_descs = {}
                            for idx, lab in enumerate(uniq):
                                mask = labels == lab
                                # feature summary for this group
                                g_feats = feats[mask]
                                summary = {
                                    "lab": g_feats[:, 0:3].mean(axis=0).tolist() if g_feats.size else [],
                                    "circularity": float(g_feats[:, 3].mean()) if g_feats.size else 0,
                                    "log_diam": float(g_feats[:, 4].mean()) if g_feats.size else 0,
                                    "texture": float(g_feats[:, 5].mean()) if g_feats.size else 0,
                                }
                                # up to 6 boxes from this group
                                group_boxes = [boxes_for_char[i] for i in range(len(boxes_for_char)) if labels[i] == lab][:6]
                                try:
                                    from labvision.describe import describe_group as _desc

                                    desc = _desc(image_path, group_boxes, feature_summary=summary)
                                except Exception:
                                    desc = {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}
                                group_descs[lab] = desc
                            # annotate per-colony results for report
                            for i, lab in enumerate(labels):
                                lab_char = chr(ord("A") + sorted(set(labels)).index(lab))
                                desc = group_descs[lab]
                                # mean diam for group
                                g_mask = labels == lab
                                mean_diam = float(feats[g_mask, 4].mean()) if g_mask.any() else 0
                                char_results.append({
                                    "index": i,
                                    "morphotype": lab_char,
                                    "morphotype_desc": f"{desc['color']}, {desc['shape']}, {desc['margin']} margin, {desc['elevation']}, {desc['surface']}",
                                    "color": desc["color"],
                                    "shape": desc["shape"],
                                    "margin": desc["margin"],
                                    "elevation": desc["elevation"],
                                    "surface": desc["surface"],
                                    "mean_diameter_px": float(np.exp(mean_diam)) if mean_diam else 0,
                                    "count_in_morphotype": int((labels == lab).sum()),
                                })
                            result = {"count": len(char_results), "details": char_results[:3]}
                            if char_results and "morphotype" in char_results[0]:
                                from collections import Counter as _Cnt

                                cnt = _Cnt(r["morphotype"] for r in char_results)
                                result["morphotype_counts"] = dict(cnt)
                        except Exception as e:
                            result = {"error": str(e)[:200]}
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
    # TNTC: >300
    count_display = "Too numerous to count (>300 CFU): dilute and re-plate" if count > 300 else f"{count} objects"
    has_mm = bool(measure_rows and "diameter_mm" in measure_rows[0])
    if measure_rows:
        ds = [r["diameter_px"] for r in measure_rows]
        areas = [r["area_px"] for r in measure_rows]
        mean_d = sum(ds) / len(ds) if ds else 0
        mean_a = sum(areas) / len(areas) if areas else 0
        mean_mm = sum(r.get("diameter_mm", 0) for r in measure_rows) / len(measure_rows) if has_mm else 0
        min_d, max_d = min(ds) if ds else 0, max(ds) if ds else 0
        filter_line = ""
    elif seg_result and seg_result.boxes:
        mean_d = min_d = max_d = mean_a = mean_mm = 0
        filter_line = "(measure not run)"
    else:
        mean_d = min_d = max_d = mean_a = mean_mm = 0
        filter_line = ""

    import re

    m = re.search(r"larger than\s+([\d.]+)\s*mm", request, re.I)
    filter_line = f"Filter: diameter >= {m.group(1)} mm" if m else ""
    mm_part = f" ({mean_mm:.2f} mm)" if has_mm else ""
    plate_note = "Plate detected — sizes in mm (90 mm plate)" if has_mm else "No plate detected — sizes in px"
    # if request asked for mm filter but no plate, note it
    if m and not has_mm:
        plate_note += " (cannot filter by mm without plate)"

    rephoto_line = f"Action: Please retake photo — {rephoto_reason}" if rephoto_reason else ""
    char_section = ""
    if char_results:
        # morphotype-aware: group already has morphotype field if characterize grouped
        if "morphotype" in char_results[0]:
            from collections import Counter as _C2

            # char_results may have morphotypes A,B... with purity_flag attached to first entry
            cnt = _C2(r["morphotype"] for r in char_results)
            lines = []
            for mt in sorted(cnt.keys()):
                ex = next(r for r in char_results if r["morphotype"] == mt)
                desc = ex.get("morphotype_desc", ex.get("color", "") + " " + ex.get("shape", ""))
                n = cnt[mt]
                mdiam = ex.get("mean_diameter_mm", 0)
                diam_str = f", mean {mdiam:.1f} mm" if mdiam else ""
                lines.append(f"  {mt}: {desc} — {n} colonies{diam_str}")
            char_section = "Colony types (descriptive only):\n" + "\n".join(lines)
        else:
            # fallback for old morphology without grouping
            from collections import Counter

            # group by morphology tuple
            def _key(r):
                return (r.get("color"), r.get("shape"), r.get("margin"), r.get("elevation"))
            from collections import Counter as _C3
            cnt = _C3(_key(r) for r in char_results)
            char_section = "Colony types: " + ", ".join(f"{k} x{v}" for k, v in cnt.items())

    report = REPORT_TEMPLATE.format(
        request=request,
        image_name=image_path.name,
        qc_summary=qc_summary,
        count_display=count_display,
        filter_line=filter_line,
        mean_d=mean_d,
        mm_part=mm_part,
        min_d=min_d,
        max_d=max_d,
        mean_a=mean_a,
        plate_note=plate_note,
        assistant_note=assistant_note.strip() if "assistant_note" in locals() else "",
        rephoto_line=rephoto_line,
        char_section=char_section,
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
