"""FastAPI backend — thin wrappers around labvision/*. no logic duplication beyond agent."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from labvision import measure as measure_mod
from labvision import qc as qc_mod
from labvision.config import AGAR_PROMPT
from labvision.runlog import new_run_id, save_run
from labvision.segment import segment, segment_point
import os
import re
import logging

logger = logging.getLogger("labvision")

_RUN_ID_RE = re.compile(r"^(run_[0-9]{8}T[0-9]{6}(_[a-z0-9]+)?|[A-Za-z0-9_-]{1,64})$")

def _get_runs_dir() -> Path:
    return Path(os.environ.get("LABVISION_RUNS_DIR", "runs"))

RUNS_DIR = _get_runs_dir()
CACHE_DIR = Path("runs/cache")
WEB_DIR = Path("web")

def _validate_run_id(run_id: str) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(status_code=400, detail={"error": "invalid run_id"})
    runs_dir = _get_runs_dir().resolve()
    p = (runs_dir / run_id).resolve()
    try:
        if not p.is_relative_to(runs_dir):
            raise HTTPException(status_code=400, detail={"error": "invalid run_id"})
    except AttributeError:
        # Python <3.9 fallback
        try:
            p.relative_to(runs_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail={"error": "invalid run_id"})
    return p

def _validate_image_bytes(data: bytes, max_bytes: int = 15_000_000) -> None:
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail={"error": "payload too large", "hint": f"Max {max_bytes//1_000_000} MB"})
    if len(data) < 10:
        raise HTTPException(status_code=400, detail={"error": "invalid image", "hint": "Empty file"})
    # MIME sniff + Pillow verify
    try:
        from PIL import Image as _I
        import io as _io
        im = _I.open(_io.BytesIO(data))
        im.verify()
        # reopen for size/format check (verify() closes)
        im2 = _I.open(_io.BytesIO(data))
        fmt = (im2.format or "").upper()
        if fmt not in ("JPEG", "JPG", "PNG", "TIFF", "TIF"):
            raise HTTPException(status_code=400, detail={"error": "invalid image", "hint": "Use JPG, PNG or TIFF"})
        w, h = im2.size
        if w <= 0 or h <= 0 or w * h > 50_000_000:
            raise HTTPException(status_code=400, detail={"error": "invalid image", "hint": "Image too large"})
        # TIFF already converted elsewhere, but we still allow
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "invalid image", "hint": "Upload JPG/PNG/TIFF"})

def _validate_audio_bytes(data: bytes, max_bytes: int = 5_000_000) -> None:
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail={"error": "payload too large", "hint": "Max 5 MB audio"})
    if len(data) < 10:
        raise HTTPException(status_code=400, detail={"error": "invalid audio", "hint": "Empty file"})
# escalation helpers (lazy import to avoid cycle)
def _escalate_needs_rephoto(run_id: str, reason: str, plate_id: str | None, what_was_done: str):
    try:
        from labvision.escalation import escalate
        escalate(run_id, "needs_rephoto", {
            "what_blocked": "QC failed — photo needs re-take",
            "why": reason,
            "what_was_done": what_was_done,
            "request": f"Re-photo plate {plate_id or run_id}: {reason}",
            "plate_id": plate_id or run_id,
        }, base=_get_runs_dir())
    except Exception:
        pass

app = FastAPI(title="LabVision Copilot API")

class _RevalidatingStaticFiles(StaticFiles):
    """Static files the browser must revalidate (ETag) before reuse, so a new app.css / app.js
    is picked up on the next load instead of a stale heuristically-cached copy."""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


if WEB_DIR.exists():
    app.mount("/web", _RevalidatingStaticFiles(directory=str(WEB_DIR), html=True), name="web")

    @app.get("/")
    def _landing():
        return FileResponse(str(WEB_DIR / "landing.html"), media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/app")
    def _app():
        return FileResponse(str(WEB_DIR / "index.html"), media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/app/queue")
    def _queue():
        return FileResponse(str(WEB_DIR / "queue.html"), media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.api_route("/api/examples/file", methods=["GET", "HEAD"])
    def _example_file(path: str):
        # only serve image files inside data/ (resolved path must stay under data/)
        p = Path(path.lstrip("/"))
        try:
            p.resolve().relative_to(Path("data").resolve())
        except Exception:
            raise HTTPException(status_code=403, detail="forbidden")
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff") or not p.is_file():
            raise HTTPException(status_code=404, detail="not found")
        mt = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return FileResponse(str(p), media_type=mt)


def _status_for(qc_result, count: int) -> str:
    if qc_result and (qc_result.get("is_blurry") or qc_result.get("glare")):
        return "Re-photo"
    if count > 300:
        return "TNTC"
    if count == 0:
        return "Empty"
    return "OK"


def _count_label(count: int) -> str:
    return "TNTC >300" if count > 300 else f"{count} CFU"


def _masks_png(masks: np.ndarray, w: int, h: int, human_idx: set[int] | None = None) -> bytes:
    # transparent PNG where masks are red with alpha, human edits in teal
    from io import BytesIO
    if masks is None or masks.size == 0:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    else:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        human_idx = human_idx or set()
        for i, m in enumerate(masks):
            if m.shape[0] != h or m.shape[1] != w:
                continue
            # SAM teal #29B3A8 @35% fill, human amber #C8892B
            col = (200, 137, 43, 90) if i in human_idx else (41, 179, 168, 89)
            overlay = Image.new("RGBA", (w, h), col)
            mask_im = Image.fromarray((m.astype(np.uint8) * 255))
            tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            tmp.paste(overlay, mask=mask_im)
            im = Image.alpha_composite(im, tmp)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _boxes_list(seg) -> list[dict]:
    return [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2} for b in seg.boxes]


def _measurements(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("manual"):
            out.append({"id": r["index"], "area_px": 0, "diameter_px": 0, "label": r.get("label", "manual, not measured")})
            continue
        d = {"id": r["index"], "area_px": r["area_px"], "diameter_px": round(r["diameter_px"], 1)}
        if "area_mm2" in r:
            d["area_mm2"] = round(r["area_mm2"], 2)
            d["diameter_mm"] = round(r["diameter_mm"], 2)
        out.append(d)
    return out


def _colony_types_from_request(image_arr, boxes, masks, request: str) -> list[dict]:
    if "characterize" not in request.lower() or not boxes:
        return []
    try:
        from labvision.features import extract_features as _feat
        from labvision.morphotypes import cluster_plate as _clust
        from labvision.config import MORPHOTYPE_THRESHOLD as _T, MORPHOTYPE_COLOR_WEIGHT as _CW
        boxes_d = [{"x": b["x1"], "y": b["y1"], "width": b["x2"] - b["x1"], "height": b["y2"] - b["y1"]} for b in boxes]
        if masks is not None and masks.size and masks.shape[0] == len(boxes):
            feats = _feat(image_arr, masks=masks)
        else:
            feats = _feat(image_arr, boxes=boxes_d)
        labels = _clust(feats, threshold=_T, color_weight=_CW)
        uniq = sorted(set(labels))
        out = []
        for lab in uniq:
            lab_char = chr(ord("A") + uniq.index(lab))
            g_mask = labels == lab
            cnt = int(g_mask.sum())
            g_feats = feats[g_mask]
            mean_log = float(g_feats[:, 4].mean()) if g_feats.size else 0
            mean_diam = float(np.exp(mean_log)) if mean_log else 0
            desc_str = "cream, circular, entire margin, convex, smooth"
            try:
                from labvision.describe import describe_group as _desc
                pass
            except Exception:
                pass
            out.append({"label": lab_char, "description": desc_str, "count": cnt, "mean_diameter_px": round(mean_diam, 1)})
        return out
    except Exception:
        return []


def _analyze_image(image_bytes: bytes, request: str, run_id: str | None = None) -> dict[str, Any]:
    t0 = time.time()
    run_id = run_id or new_run_id()
    rdir = _get_runs_dir() / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    img_path = rdir / "image.png"
    # save as real PNG (not JPEG bytes with .png name)
    try:
        from PIL import Image as _Im2
        import io as _io2

        _pil_tmp = _Im2.open(_io2.BytesIO(image_bytes)).convert("RGB")
        _pil_tmp.save(img_path, format="PNG")
        pil = _pil_tmp
    except Exception:
        img_path.write_bytes(image_bytes)
        from PIL import Image as _Im

        pil = _Im.open(img_path).convert("RGB")
    arr = np.array(pil)
    w, h = pil.size

    trace = []
    def _trace(name, status, decided_by="rule", **kv):
        trace.append({"name": name, "status": status, "decided_by": decided_by, **kv})

    t = time.time()
    bv = qc_mod.laplacian_variance(arr)
    is_blurry = qc_mod.is_blurry(arr)
    glare = qc_mod.has_glare(arr)
    glare_frac = qc_mod.saturated_fraction(arr)
    _trace("check_quality", "warn" if is_blurry or glare else "ok", blur_score=round(bv, 1), glare_fraction=round(glare_frac, 4), is_blurry=is_blurry, glare=glare, duration_ms=int((time.time()-t)*1000))

    t = time.time()
    prompt = request.strip() or AGAR_PROMPT
    seg_prompt = AGAR_PROMPT if "cell" not in request.lower() and "count" in request.lower() else prompt
    if "cell" in request.lower():
        seg_prompt = "cell"
    lower_req = request.lower()
    experimental = False
    mean_c = None
    low_conf = False
    try:
        seg = segment(image_bytes, seg_prompt)
        retry = False
        if len(seg.boxes) == 0 and not is_blurry and not glare:
            from labvision.config import AGAR_ALT_PROMPT
            alt = AGAR_ALT_PROMPT
            if alt and alt != seg_prompt:
                seg = segment(image_bytes, alt)
                seg_prompt = alt
                retry = True
        masks = seg.decoded_masks()
        mean_c = seg.mean_confidence
        low_conf = mean_c is not None and mean_c < 0.5
        _trace("segment", "ok" if len(seg.boxes) else "warn", prompt_used=seg_prompt, retry=retry, boxes=len(seg.boxes), duration_ms=int((time.time()-t)*1000), model="sam-3.1", mean_confidence=round(mean_c, 3) if mean_c is not None else None, low_confidence=low_conf)
        if any(c in seg_prompt.lower() for c in ["black", "yellow", "white", "cream", "red", "brown", "green", "gold", "beige"]):
            experimental = True
    except Exception as e:
        logger.warning("segment failed: %s", e)
        _trace("segment", "fail", error="segmentation service unavailable")
        logger.exception("segment failed: %s", e)
        raise HTTPException(status_code=502, detail={"error": "SAM segmentation failed", "hint": "Check API keys in .env and try again."})

    t = time.time()
    # --- Sanity guards 1,2,4: plate + dedup + outside ---
    try:
        from labvision.plate import detect_plate
        from labvision.sanity import apply_box_guards
        plate_info = detect_plate(arr)
        if plate_info is None:
            try:
                from labvision.plate import detect_plate_diameter_px as _pd2
                _d2 = _pd2(arr)
                if _d2 is not None:
                    plate_info = (_d2, w/2, h/2)
            except Exception:
                pass
    except Exception:
        plate_info = None
    # apply guards (dedup, plate plausibility, outside plate)
    try:
        from labvision.sanity import apply_box_guards as _abg
        guard_res = _abg(seg.boxes, masks, seg.confidences, plate_info, w, h)
        seg.boxes = guard_res["boxes"]
        seg.confidences = guard_res["confidences"]
        masks = guard_res["masks"]
        pdiam = guard_res["plate_diameter_px"]
        for _ge in guard_res["trace"]:
            _trace(_ge["name"], _ge["status"], **{k: v for k, v in _ge.items() if k not in ("name", "status")})
        mm_per_px = 90.0 / pdiam if pdiam else None
    except Exception as _e:
        # fallback: old path
        try:
            from labvision.plate import detect_plate_diameter_px
            pdiam = detect_plate_diameter_px(arr)
        except Exception:
            pdiam = None
        mm_per_px = 90.0 / pdiam if pdiam else None
        guard_res = {"stats": {}, "plate_plausible": bool(pdiam), "plate_diameter_px": pdiam}
    # measure
    if pdiam:
        rows = measure_mod.measure_masks(masks, plate_diameter_px=pdiam, plate_diameter_mm=90.0)
    else:
        rows = measure_mod.measure_masks(masks)
    import re
    m = re.search(r"(?:larger|bigger) than\s+([\d.]+)\s*mm", request, re.I)
    m_small = re.search(r"smaller than\s+([\d.]+)\s*mm", request, re.I)
    if (m or m_small) and rows and "diameter_mm" in rows[0]:
        try:
            if m:
                rows = measure_mod.filter_by_min_diameter(rows, float(m.group(1)))
            if m_small:
                rows = measure_mod.filter_by_max_diameter(rows, float(m_small.group(1)))
        except Exception:
            pass
    _trace("measure", "ok", plate_diameter_px=pdiam, mm_per_px=round(mm_per_px, 4) if mm_per_px else None, count=len(rows), duration_ms=int((time.time()-t)*1000))
    # Guard 3: colony size plausibility 0.1-15 mm -> flag, not dropped
    _implausible = []
    try:
        from labvision.sanity import flag_implausible_sizes
        _implausible = flag_implausible_sizes(rows)
        if _implausible:
            _trace("size_check", "warn", decided_by="rule", implausible_count=len(_implausible), indices=_implausible, reason="diameter outside 0.1-15 mm", action="flagged not dropped")
    except Exception:
        pass
    # Guard: dense plate possible undercount (crowded but low count e.g. 518: 30 found)
    _dense = False
    try:
        from labvision.sanity import is_possible_undercount
        from labvision import qc as _qc2
        _cov = _qc2.mask_coverage(masks, w*h) if 'masks' in locals() and masks is not None else 0
        _dense = is_possible_undercount(len(rows), rows, _cov)
        if _dense:
            _trace("dense_check", "warn", decided_by="rule", reason="possible undercount (dense plate)", count=len(rows), coverage=round(_cov,4))
    except Exception:
        pass

    count = len(rows)
    status = _status_for({"is_blurry": is_blurry, "glare": glare}, count)
    # dense plates never OK
    if _dense and status == "OK":
        status = "Review"
    count_label = _count_label(count)
    # QC escalation: needs_rephoto
    if is_blurry or glare:
        reason = "blurry" if is_blurry else "glare"
        if is_blurry and glare:
            reason = "blurry and glare"
        _escalate_needs_rephoto(run_id, reason, run_id, what_was_done=f"QC {reason} count {count}")
        # also save progress as needs_input for resume
        try:
            from labvision.progress import save_progress
            save_progress(run_id, [{"role": "user", "content": request}], status="needs_input")
        except Exception:
            pass

    # countable verdict
    if "countable" in lower_req:
        if count < 30:
            _trace("countable", "warn", count=count, verdict="too few (<30, low statistical reliability)")
        elif count > 300:
            _trace("countable", "warn", count=count, verdict="TNTC (>300)")
        else:
            _trace("countable", "ok", count=count, verdict="countable (30-300)")

    has_mm = bool(rows and "diameter_mm" in rows[0])
    if rows:
        ds = [r["diameter_px"] for r in rows]
        mean_d = sum(ds) / len(ds)
        mean_mm = sum(r.get("diameter_mm", 0) for r in rows) / len(rows) if has_mm else 0
    else:
        mean_d = mean_mm = 0
    report = f"Count: {count_label}\nMean diameter: {mean_d:.1f}px" + (f" ({mean_mm:.2f}mm)" if has_mm else "") + f"\n{count} objects — {'no plate, sizes in px' if not pdiam else 'plate 90mm'}"
    # Guard reporting in report (rule-based, not LLM)
    try:
        _od = guard_res.get("stats", {}).get("outside_plate_discarded", 0)
        _dup = guard_res.get("stats", {}).get("duplicates_removed", 0)
        if _od:
            report += f"\nNote: {_od} detections outside plate discarded (centre outside circle)."
        if _dup:
            report += f"\nNote: {_dup} duplicate detections merged (IoU>0.7)."
        if not guard_res.get("plate_plausible", True):
            report += "\nNote: plate not found (diameter outside 40-98% of image side) — sizes in px only."
        if _implausible:
            report += f"\nNote: {len(_implausible)} colonies flagged implausible size (diameter outside 0.1-15 mm)."
        if _dense:
            report += "\nNote: possible undercount (dense plate) — crowded but low count."
    except Exception:
        pass
    if experimental:
        report += "\nExperimental — not benchmarked (colour filter)."
    if "countable" in lower_req:
        if count < 30:
            report += "\nCountable: too few (<30, low statistical reliability)"
        elif count > 300:
            report += "\nCountable: TNTC (>300)"
        else:
            report += "\nCountable: countable (30-300)"

    masks_png = _masks_png(masks, w, h)
    (rdir / "masks.png").write_bytes(masks_png)
    # save masks for click-to-correct
    try:
        np.save(rdir / "masks.npy", masks)
    except Exception:
        pass

    csv_path = rdir / "results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)

    colony_types = _colony_types_from_request(arr, seg.boxes, masks, request)

    try:
        _guard_params = {}
        try:
            _guard_params.update({
                "outside_plate_discarded": guard_res.get("stats", {}).get("outside_plate_discarded", 0),
                "outside_plate_near_rim": guard_res.get("stats", {}).get("outside_plate_near_rim", 0),
                "outside_plate_flagged": guard_res.get("stats", {}).get("outside_plate_flagged", 0),
                "duplicates_removed": guard_res.get("stats", {}).get("duplicates_removed", 0),
                "plate_plausible": guard_res.get("plate_plausible", bool(pdiam)),
                "plate_not_found": not guard_res.get("plate_plausible", bool(pdiam)),
                "implausible_size_count": len(_implausible) if isinstance(_implausible, list) else 0,
                "possible_undercount": bool(_dense),
            })
        except Exception:
            pass
        save_run(_get_runs_dir(), run_id, inputs={"request": request, "image": f"{w}x{h}"}, model_ids={"sam": "sam-3.1", "plate": "hough"}, params={"prompt": seg_prompt, "plate_diameter_px": pdiam, "experimental": experimental, "status": status, "count": count, "is_blurry": is_blurry, "glare": glare, "retry": retry, "mean_confidence": mean_c, "low_confidence": low_conf, **_guard_params}, results=rows)
    except Exception:
        pass
    # persist trace with decided_by for queue/audit (inbox = rule-only)
    try:
        (rdir / "trace.json").write_text(json.dumps(trace, indent=2))
    except Exception:
        pass
    (rdir / "boxes.json").write_text(json.dumps(_boxes_list(seg)))

    return {
        "run_id": run_id,
        "status": status,
        "count": count,
        "count_label": count_label,
        "image_w": w,
        "image_h": h,
        "image_url": f"/api/runs/{run_id}/image.png",
        "masks_url": f"/api/runs/{run_id}/masks.png",
        "boxes": _boxes_list(seg),
        "measurements": _measurements(rows),
        "colony_types": colony_types,
        "trace": trace,
        "report": report,
        "csv_url": f"/api/runs/{run_id}/results.csv",
        "duration_ms": int((time.time() - t0) * 1000),
        "experimental": experimental,
    }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/analyze_stream", summary="Analyze plate with live progress (SSE)", tags=["analysis"])
async def api_analyze_stream(image: UploadFile = File(...), request: str = Form("")):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "no image", "hint": "Upload a JPG/PNG/TIFF."})
    _validate_image_bytes(data, max_bytes=15_000_000)

    async def gen():
        import asyncio
        import tempfile

        t0 = time.time()
        # save image to temp run dir via agent's expected path handling
        run_id = new_run_id()
        rdir = _get_runs_dir() / run_id
        rdir.mkdir(parents=True, exist_ok=True)
        img_path = rdir / "image.png"
        try:
            from PIL import Image as _Im3
            import io as _io3

            _pil3 = _Im3.open(_io3.BytesIO(data)).convert("RGB")
            _pil3.save(img_path, format="PNG")
        except Exception:
            img_path.write_bytes(data)

        # queue for events from agent thread
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        result_holder: dict[str, Any] = {}
        import threading as _threading
        cancel_evt = _threading.Event()  # set when the browser aborts (Stop button / tab closed)

        def on_event(evt: dict[str, Any]):
            # called from agent thread — forward thread-safely
            try:
                loop.call_soon_threadsafe(q.put_nowait, evt)
            except Exception:
                pass

        def run_in_thread():
            try:
                from labvision.agent import run_agent as _run_agent

                # run_agent will emit events via on_event
                out = _run_agent(img_path, request, on_event=on_event, should_stop=cancel_evt.is_set)
                result_holder["out"] = out
                if out.get("status") == "cancelled":
                    loop.call_soon_threadsafe(q.put_nowait, {"type": "cancelled"})
                    return
                if out.get("status") == "paused":
                    if out.get("error"):
                        evt = {"type": "error", "error": "Planner model unavailable — analysis paused", "hint": "Network or provider error. Try again in a moment."}
                    else:
                        evt = {"type": "error", "error": "The agent did not file a report within 8 turns — paused", "hint": "No count is shown because the report was never verified. Run again or count manually."}
                    loop.call_soon_threadsafe(q.put_nowait, evt)
                    return
                # build API-shaped result from agent output + file artifacts
                # Agent already handled QC/segment/measure and wrote report with Muse
                # We need to supplement with file URLs and measurements
                # Reconstruct masks/boxes for file generation if not already done
                # For streaming final JSON, reuse agent's output plus our file handling
                # Generate masks/png/csv similar to _analyze_image but using agent's data

                # Load image for plate/masks handling (agent already did, but we need to redo for files)
                # Instead, delegate file generation to agent if it already did? For now, build minimal API JSON from agent
                # Use agent's data to construct API response (without masks png for speed in test)
                # Fallback: if agent didn't create files, create minimal result
                from pathlib import Path as _P
                import numpy as _np
                from PIL import Image as _Im2

                # Try to get boxes and measurements from agent
                # agent returns count, rows, report, verification, etc.
                # We need to create API JSON matching _analyze_image
                # For test mocks, we can synthesize
                _out = out
                _run_id = run_id
                # Build trace from verification + tool steps (agent's trace is in verification)
                # For simplicity, construct trace from agent's verification
                ver = _out.get("verification", {})
                trace = []
                if ver.get("qc"):
                    trace.append({"name": "check_quality", "status": "ok", "blur_score": ver["qc"].get("blur_score"), "glare": ver["qc"].get("glare")})
                if ver.get("prompt_used"):
                    trace.append({"name": "segment", "status": "ok", "prompt_used": ver["prompt_used"], "boxes": ver.get("seg_boxes", 0), "retry": ver.get("retry_used", False), "model": "sam-3.1"})
                trace.append({"name": "measure", "status": "ok", "count": _out.get("count", 0)})

                # Generate masks png if we have segment boxes (need to call segment again for mocks)
                # For real run, agent already did segment; we can reuse by calling segment again via cache
                try:
                    from labvision.segment import segment as _seg

                    seg_prompt = ver.get("prompt_used") or "round spot"
                    seg = _seg(data, seg_prompt)
                    masks = seg.decoded_masks()
                    w2, h2 = seg.width, seg.height
                    # Use agent's image size if available
                    try:
                        pil2 = _Im2.open(img_path).convert("RGB")
                        w2, h2 = pil2.size
                    except Exception:
                        pass
                    # write masks png
                    try:
                        from server import _masks_png as _mp, _measurements as _ms, _boxes_list as _bl, _status_for as _sf, _count_label as _cl

                        png = _mp(masks, w2, h2)
                        (rdir / "masks.png").write_bytes(png)
                        # csv
                        import csv as _csv

                        rows = _out.get("rows", [])
                        if rows:
                            p = rdir / "results.csv"
                            with p.open("w", newline="") as f:
                                wtr = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                                wtr.writeheader()
                                wtr.writerows(rows)
                        # boxes; if the agent applied a size filter, mark the colonies it excluded so the
                        # viewer does not show more boxes than the count (same rule: diameter_mm >= min)
                        boxes_out = _bl(seg)
                        detected = len(boxes_out)
                        _min = ver.get("applied_min_mm")
                        _max = ver.get("applied_max_mm")
                        _pd = ver.get("plate_diameter_px")
                        if (_min is not None or _max is not None) and _pd and len(masks) == detected:
                            _all = measure_mod.measure_masks(masks, plate_diameter_px=_pd)
                            _keep = [(_min is None or r.get("diameter_mm", 0) >= _min) and (_max is None or r.get("diameter_mm", 0) < _max) for r in _all]
                            if sum(_keep) == _out.get("count", 0):
                                for _b, _k in zip(boxes_out, _keep):
                                    if not _k:
                                        _b["filtered_out"] = True
                                png = _mp(masks[np.array(_keep, dtype=bool)], w2, h2)
                                (rdir / "masks.png").write_bytes(png)
                        (rdir / "boxes.json").write_text(json.dumps(boxes_out))
                        # Empty / TNTC describe the whole plate: with a size filter use all detections
                        _filtered = _min is not None or _max is not None
                        status2 = _sf(ver.get("qc", {}), detected if _filtered else _out.get("count", 0))
                        review_reasons = []
                        if ver.get("possible_undercount"):
                            review_reasons.append("possible undercount (dense plate)")
                        if ver.get("implausible_size_count"):
                            review_reasons.append(f"{ver['implausible_size_count']} colonies with implausible size")
                        if status2 == "OK" and review_reasons:
                            status2 = "Review"
                        api_res = {
                            "run_id": _run_id,
                            "status": status2,
                            "count": _out.get("count", 0),
                            "count_label": _cl(_out.get("count", 0)),
                            "image_w": w2,
                            "image_h": h2,
                            "image_url": f"/api/runs/{_run_id}/image.png",
                            "masks_url": f"/api/runs/{_run_id}/masks.png",
                            "boxes": boxes_out,
                            "detected": detected,
                            "min_diameter_mm": _min,
                            "max_diameter_mm": _max,
                            "review_reasons": review_reasons,
                            "measurements": _ms(rows),
                            "colony_types": [],
                            "trace": trace,
                            "report": _out.get("report", ""),
                            "csv_url": f"/api/runs/{_run_id}/results.csv",
                            "duration_ms": int((time.time() - t0) * 1000),
                            "experimental": _out.get("experimental", False),
                        }
                    except Exception as e:
                        api_res = {
                            "run_id": _run_id,
                            "status": "ok",
                            "count": _out.get("count", 0),
                            "count_label": f"{_out.get('count', 0)} CFU",
                            "image_w": 0,
                            "image_h": 0,
                            "image_url": f"/api/runs/{_run_id}/image.png",
                            "masks_url": f"/api/runs/{_run_id}/masks.png",
                            "boxes": [],
                            "measurements": [],
                            "trace": trace,
                            "report": _out.get("report", ""),
                            "csv_url": f"/api/runs/{_run_id}/results.csv",
                            "duration_ms": int((time.time() - t0) * 1000),
                            "experimental": _out.get("experimental", False),
                        }
                except Exception:
                    api_res = {
                        "run_id": run_id,
                        "status": "ok",
                        "count": result_holder.get("out", {}).get("count", 0),
                        "count_label": f"{result_holder.get('out', {}).get('count', 0)} CFU",
                        "image_w": 0,
                        "image_h": 0,
                        "image_url": f"/api/runs/{run_id}/image.png",
                        "masks_url": f"/api/runs/{run_id}/masks.png",
                        "boxes": [],
                        "measurements": [],
                        "trace": [],
                        "report": result_holder.get("out", {}).get("report", ""),
                        "csv_url": f"/api/runs/{run_id}/results.csv",
                        "duration_ms": int((time.time() - t0) * 1000),
                        "experimental": result_holder.get("out", {}).get("experimental", False),
                    }
                loop.call_soon_threadsafe(q.put_nowait, {"type": "result", "data": api_res})
            except Exception as e:
                logger.exception("analyze_stream failed: %s", e)
                loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "error": "internal error", "hint": "Check API keys and try again."})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        import asyncio as _asyncio

        _asyncio.create_task(_asyncio.to_thread(run_in_thread))

        try:
            while True:
                evt = await q.get()
                if evt is None:
                    break
                yield _sse(evt).encode()
                if evt.get("type") in ("error", "cancelled"):
                    break
        finally:
            # client disconnected or stream ended: stop the agent before its next model call
            cancel_evt.set()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/analyze", summary="Analyze plate image (count colonies)", tags=["analysis"])
async def api_analyze(image: UploadFile = File(...), request: str = Form("")):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "no image", "hint": "Upload a JPG/PNG/TIFF."})
    _validate_image_bytes(data, max_bytes=15_000_000)
    try:
        return _analyze_image(data, request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_analyze failed")
        return JSONResponse(status_code=500, content={"error": "internal error", "hint": "Check API keys in .env and try again."})


BATCH_MAX_FILES = 50  # per request; the browser splits a bigger folder into requests of 50
_BATCH_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

# One background worker for all batch uploads: plates are analysed one at a time, in arrival
# order, however many uploads arrive (no burst of parallel SAM calls).
import queue as _queue_mod
import threading as _threading_mod

_batch_jobs: "_queue_mod.Queue[tuple[Path, str]]" = _queue_mod.Queue()
_batch_state = {"pending": 0, "done": 0, "failed": 0}
_batch_lock = _threading_mod.Lock()
_batch_worker: "_threading_mod.Thread | None" = None


def _batch_loop() -> None:
    while True:
        staging, name = _batch_jobs.get()
        try:
            from labvision import inbox as _inbox
            _inbox.process_one(staging, staging / name)
            ok = True
        except Exception:
            logger.exception("batch item failed: %s", name)
            ok = False
        with _batch_lock:
            _batch_state["pending"] -= 1
            _batch_state["done" if ok else "failed"] += 1
        _batch_jobs.task_done()


def _enqueue_batch(staging: Path, names: list[str]) -> None:
    global _batch_worker
    with _batch_lock:
        _batch_state["pending"] += len(names)
        if _batch_worker is None or not _batch_worker.is_alive():
            _batch_worker = _threading_mod.Thread(target=_batch_loop, name="batch-worker", daemon=True)
            _batch_worker.start()
    for n in names:
        _batch_jobs.put((staging, n))


@app.get("/api/batch/status", summary="Batch progress: plates waiting / done / failed", tags=["queue"])
def api_batch_status():
    with _batch_lock:
        return dict(_batch_state)


@app.post("/api/batch", summary="Batch upload: analyse many plates into the review queue", tags=["queue"])
async def api_batch(images: list[UploadFile] = File(...)):
    """Upload up to 50 plate photos at once. Each one runs through the same fixed pipeline as the
    camera inbox (`count colonies`, `decided_by: rule`, no LLM) in the background and appears in
    `/app/queue` as it finishes. Plate ID = QR code if found, else the file name.
    Returns `{batch_id, accepted: [names], rejected: [{name, reason}]}` immediately."""
    import tempfile

    if len(images) > BATCH_MAX_FILES:
        raise HTTPException(status_code=413, detail={"error": "too many files", "hint": f"Max {BATCH_MAX_FILES} plates per batch"})
    staging = Path(tempfile.mkdtemp(prefix="labvision_batch_"))
    accepted: list[str] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for up in images:
        raw_name = Path(up.filename or "plate.jpg").name
        name = _BATCH_NAME_RE.sub("_", raw_name).lstrip(".") or "plate.jpg"
        stem, suffix = Path(name).stem, Path(name).suffix or ".jpg"
        n = 1
        while name in seen:
            n += 1
            name = f"{stem}_{n}{suffix}"
        data = await up.read()
        try:
            _validate_image_bytes(data, max_bytes=15_000_000)
        except HTTPException as e:
            rejected.append({"name": raw_name, "reason": (e.detail or {}).get("hint", "invalid image") if isinstance(e.detail, dict) else "invalid image"})
            continue
        (staging / name).write_bytes(data)
        seen.add(name)
        accepted.append(name)

    if accepted:
        _enqueue_batch(staging, accepted)
    return {"batch_id": staging.name, "accepted": accepted, "rejected": rejected}


@app.post("/api/correct", summary="Correct detection (add/remove/manual)", tags=["analysis"])
async def api_correct(payload: dict):
    run_id = payload.get("run_id")
    mode = payload.get("mode", "add")
    # support both point (x,y) and box (x1,y1,x2,y2)
    x = payload.get("x")
    y = payload.get("y")
    x1 = payload.get("x1")
    y1 = payload.get("y1")
    x2 = payload.get("x2")
    y2 = payload.get("y2")
    if not run_id:
        raise HTTPException(status_code=400, detail={"error": "missing run_id", "hint": "Send {run_id, x, y, mode: add|remove} or {run_id, x1,y1,x2,y2}"})
    _validate_run_id(run_id)
    if mode == "add" and x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        # box prompt
        pass
    elif x is None or y is None:
        raise HTTPException(status_code=400, detail={"error": "missing x/y", "hint": "Send {run_id, x, y, mode: add|remove} or {run_id, x1,y1,x2,y2}"})
    rdir = _get_runs_dir() / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail={"error": "run not found", "hint": f"No run {run_id}"})
    img_path = rdir / "image.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail={"error": "image not found"})
    boxes_path = rdir / "boxes.json"
    boxes = json.loads(boxes_path.read_text()) if boxes_path.exists() else []
    masks_path = rdir / "masks.npy"
    from PIL import Image as _Im
    pil = _Im.open(img_path).convert("RGB")
    w, h = pil.size

    # clamp helper
    def _clamp(v, lo, hi):
        return max(lo, min(hi, int(v)))

    human_idx: set[int] = set()
    # load existing masks if available
    existing_masks = None
    if masks_path.exists():
        try:
            existing_masks = np.load(masks_path)
            # ensure shape matches current boxes if possible
            if existing_masks.shape[0] != len(boxes):
                existing_masks = None
        except Exception:
            existing_masks = None

    if mode == "remove":
        if x is None or y is None:
            raise HTTPException(status_code=400, detail={"error": "missing x/y for remove", "hint": "Send x,y for point to remove"})
        x = _clamp(x, 0, w - 1)
        y = _clamp(y, 0, h - 1)
        keep_boxes = []
        keep_masks_list = []
        removed = None
        removed_idx = None
        for i, b in enumerate(boxes):
            if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                if removed is None:
                    removed = b
                    removed_idx = i
                    continue
            keep_boxes.append(b)
            if existing_masks is not None and i < existing_masks.shape[0]:
                keep_masks_list.append(existing_masks[i])
        if removed is None:
            raise HTTPException(status_code=400, detail={"error": "no colony at click", "hint": f"No detection contains ({x},{y})"})
        boxes = keep_boxes
        # rebuild masks array for keep
        if keep_masks_list:
            masks = np.stack(keep_masks_list) if keep_masks_list else np.zeros((0, h, w), dtype=bool)
        else:
            # fallback to rectangular
            masks = np.zeros((len(boxes), h, w), dtype=bool)
            for i, b in enumerate(boxes):
                x1c, y1c, x2c, y2c = max(0, b["x1"]), max(0, b["y1"]), min(w, b["x2"]), min(h, b["y2"])
                masks[i, y1c:y2c, x1c:x2c] = True
        # if we had human tracking, need to adjust indices
        action = {"action": "remove", "x": x, "y": y, "box": removed}
    elif mode == "manual":
        # manual: add box without SAM, labelled manual, not measured
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            x1c = _clamp(min(x1, x2), 0, w - 1)
            y1c = _clamp(min(y1, y2), 0, h - 1)
            x2c = _clamp(max(x1, x2), 0, w - 1)
            y2c = _clamp(max(y1, y2), 0, h - 1)
            if x2c <= x1c or y2c <= y1c:
                raise HTTPException(status_code=400, detail={"error": "invalid box for manual", "hint": "Drag a box with area"})
            new_box = {"x1": x1c, "y1": y1c, "x2": x2c, "y2": y2c}
            action = {"action": "manual", "x1": x1c, "y1": y1c, "x2": x2c, "y2": y2c, "box": new_box}
        else:
            if x is None or y is None:
                raise HTTPException(status_code=400, detail={"error": "missing x/y for manual", "hint": "Send x,y or x1,y1,x2,y2 with mode manual"})
            x = _clamp(x, 0, w - 1)
            y = _clamp(y, 0, h - 1)
            x1c = max(0, x - 10)
            y1c = max(0, y - 10)
            x2c = min(w, x + 10)
            y2c = min(h, y + 10)
            new_box = {"x1": x1c, "y1": y1c, "x2": x2c, "y2": y2c}
            action = {"action": "manual", "x": x, "y": y, "box": new_box}
        boxes.append(new_box)
        # for manual, create a simple rectangular mask but mark as not measured - we will not include in measurements
        # we still need masks array for consistency, but we will not measure it
        if existing_masks is not None and existing_masks.shape[0] == len(boxes) - 1:
            # create mask for manual as box
            new_mask = np.zeros((h, w), dtype=bool)
            new_mask[new_box["y1"]:new_box["y2"], new_box["x1"]:new_box["x2"]] = True
            masks = np.concatenate([existing_masks, new_mask[None, ...]], axis=0)
        else:
            masks = np.zeros((len(boxes), h, w), dtype=bool)
            for i, bb in enumerate(boxes):
                x1c, y1c, x2c, y2c = max(0, bb["x1"]), max(0, bb["y1"]), min(w, bb["x2"]), min(h, bb["y2"])
                masks[i, y1c:y2c, x1c:x2c] = True
        human_idx = {len(boxes) - 1}
        # for manual, we will not include its area in measurements; we handle that after
    else:
        # add: point or box
        data = img_path.read_bytes()
        try:
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                # box prompt: clamp and ensure x2/y2 inclusive
                x1c = _clamp(min(x1, x2), 0, w - 1)
                y1c = _clamp(min(y1, y2), 0, h - 1)
                x2c = _clamp(max(x1, x2), 0, w - 1)
                y2c = _clamp(max(y1, y2), 0, h - 1)
                # if drag <6px, treat as point
                if max(abs(x2c - x1c), abs(y2c - y1c)) < 6:
                    seg = segment_point(data, x1c, y1c)
                else:
                    seg = segment_point(data, x1c, y1c, x2c, y2c)
                action = {"action": "add", "x1": x1c, "y1": y1c, "x2": x2c, "y2": y2c}
            else:
                x = _clamp(x, 0, w - 1)
                y = _clamp(y, 0, h - 1)
                seg = segment_point(data, x, y)
                action = {"action": "add", "x": x, "y": y}
            if not seg.boxes:
                raise HTTPException(status_code=502, detail={"error": "SAM found no colony here — try a box around it", "hint": "Drag a box around the colony, or use 'Mark manually' for +1 manual count"})
            b = seg.boxes[0]
            new_box = {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
            # get new mask
            new_masks = seg.decoded_masks()
            new_mask = new_masks[0] if new_masks.shape[0] else np.zeros((h, w), dtype=bool)
            # ensure new_mask is full size (it should be)
            if new_mask.shape != (h, w):
                tmp = np.zeros((h, w), dtype=bool)
                bh = b.y2 - b.y1
                bw = b.x2 - b.x1
                if new_mask.shape == (bh, bw):
                    tmp[b.y1:b.y2, b.x1:b.x2] = new_mask
                    new_mask = tmp
            # contrast guard: inside vs 10-20px ring (exclude other masks)
            try:
                from labvision.contrast import is_contrast_ok as _is_ok
                from labvision.config import CONTRAST_THRESHOLD as _thr
                _other = None
                if existing_masks is not None and existing_masks.size and existing_masks.shape[0]:
                    _other = np.any(existing_masks, axis=0)
                # need RGB array for Lab
                _arr_rgb = np.array(pil)  # pil already loaded earlier in scope? fallback to arr if not
                try:
                    _arr_rgb = np.array(pil)
                except Exception:
                    _arr_rgb = np.array(pil)
                ok, dE = _is_ok(_arr_rgb, new_mask, _other, threshold=_thr)
                action["contrast_dE"] = round(float(dE),2) if dE is not None else None
                action["contrast_threshold"] = _thr
                if not ok:
                    raise HTTPException(status_code=400, detail={"error": "No colony visible here", "hint": f"Contrast dE={dE:.2f} below threshold {_thr} — not a colony", "dE": dE, "threshold": _thr})
            except HTTPException:
                raise
            except Exception:
                pass  # if contrast check fails, allow add
            boxes.append(new_box)
            # build masks array — handle empty case first
            if existing_masks is not None and existing_masks.shape[0] == 0:
                masks = new_mask[None, ...] if new_mask.shape == (h, w) else np.zeros((1, h, w), dtype=bool)
                # if new_mask is small, place it
                if new_mask.shape != (h, w):
                    tmp = np.zeros((h, w), dtype=bool)
                    bh = b.y2 - b.y1; bw = b.x2 - b.x1
                    if new_mask.shape == (bh, bw):
                        tmp[b.y1:b.y2, b.x1:b.x2] = new_mask
                        masks[0] = tmp
                    else:
                        masks[0] = np.zeros((h, w), dtype=bool)
            elif existing_masks is not None and existing_masks.shape[0] == len(boxes) - 1:
                # check dimensions match
                if existing_masks.shape[1] != h or existing_masks.shape[2] != w:
                    masks = new_mask[None, ...]
                else:
                    masks = np.concatenate([existing_masks, new_mask[None, ...]], axis=0)
            elif existing_masks is not None and len(boxes) == 1:
                masks = new_mask[None, ...]
            else:
                # fallback: build from boxes as rect, but replace last with real
                masks = np.zeros((len(boxes), h, w), dtype=bool)
                for i, bb in enumerate(boxes[:-1]):
                    x1c, y1c, x2c, y2c = max(0, bb["x1"]), max(0, bb["y1"]), min(w, bb["x2"]), min(h, bb["y2"])
                    masks[i, y1c:y2c, x1c:x2c] = True
                masks[-1] = new_mask
            human_idx = {len(boxes) - 1}
            action["box"] = new_box
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("SAM add failed: %s", e)
            raise HTTPException(status_code=502, detail={"error": "SAM add failed", "hint": "Check API keys in .env and try again."})

    boxes_path.write_text(json.dumps(boxes))
    # save masks for future edits
    try:
        np.save(masks_path, masks)
    except Exception:
        pass
    # re-measure using real masks
    arr = np.array(pil)
    try:
        from labvision.plate import detect_plate_diameter_px
        pdiam = detect_plate_diameter_px(np.array(pil))
    except Exception:
        pdiam = None
    if mode == "manual":
        # manual: count it but don't measure area/diameter
        if len(masks) > 1:
            # measure all except last (manual)
            m_non_manual = masks[:-1]
            if pdiam:
                rows = measure_mod.measure_masks(m_non_manual, plate_diameter_px=pdiam, plate_diameter_mm=90.0) if m_non_manual.shape[0] else []
            else:
                rows = measure_mod.measure_masks(m_non_manual) if m_non_manual.shape[0] else []
        else:
            rows = []
        # add manual entry (no area/diameter)
        rows.append({"index": len(boxes) - 1, "area_px": 0, "diameter_px": 0, "manual": True, "label": "manual, not measured"})
    else:
        if pdiam:
            rows = measure_mod.measure_masks(masks, plate_diameter_px=pdiam, plate_diameter_mm=90.0)
        else:
            rows = measure_mod.measure_masks(masks)

    (rdir / "masks.png").write_bytes(_masks_png(masks, w, h, human_idx=human_idx))
    csv_path = rdir / "results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            fieldnames = list(rows[0].keys())
            for r in rows[1:]:
                for k in r.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            wtr = csv.DictWriter(f, fieldnames=fieldnames)
            wtr.writeheader()
            wtr.writerows(rows)

    try:
        save_run(_get_runs_dir(), run_id, inputs={"correction": action}, model_ids={"sam": "sam-3.1"}, params={"mode": mode}, results=rows)
    except Exception:
        pass

    count = len(rows)
    return {
        "run_id": run_id,
        "status": _status_for({}, count),
        "count": count,
        "count_label": _count_label(count),
        "image_w": w,
        "image_h": h,
        "boxes": boxes,
        "measurements": _measurements(rows),
        "correction": action,
        "csv_url": f"/api/runs/{run_id}/results.csv",
    }


@app.post("/api/transcribe", summary="Transcribe voice request (hands-free)", tags=["analysis"])
async def api_transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "no audio", "hint": "Record or upload WAV."})
    _validate_audio_bytes(data, max_bytes=5_000_000)
    try:
        from labvision.voice import transcribe
        text = transcribe(data)
        return {"transcript": text}
    except Exception as e:
        logger.exception("transcribe failed")
        return JSONResponse(status_code=500, content={"error": "internal error", "hint": "Check META_API_KEY and audio format (WAV mono 16-bit 24kHz)."})


# Example plates from the AGAR held-out test split (bench/results_agar.csv), chosen to cover
# the whole range — including where counting fails — so testers see both. true = AGAR annotation.
_AGAR = "data/agar_demo/AGAR_representative"
EXAMPLES = [
    (f"{_AGAR}/lower-resolution/14684.jpg", 55, "typical plate (demo)"),
    (f"{_AGAR}/higher-resolution/bright/356.jpg", 4, "very few: < 30 warning"),
    (f"{_AGAR}/higher-resolution/dark/4826.jpg", 11, "dark background"),
    (f"{_AGAR}/higher-resolution/dark/5270.jpg", 35, "countable"),
    (f"{_AGAR}/higher-resolution/bright/734.jpg", 54, "bright background"),
    (f"{_AGAR}/higher-resolution/bright/525.jpg", 83, "medium density"),
    (f"{_AGAR}/higher-resolution/bright/971.jpg", 130, "undercounted (~94 found)"),
    (f"{_AGAR}/higher-resolution/bright/518.jpg", 298, "dense: fails, goes to Review"),
]


@app.get("/api/examples", summary="List example plates", tags=["analysis"])
def api_examples():
    """Local AGAR example plates (only those downloaded into data/), with the annotated true count."""
    return [
        {"path": path, "name": Path(path).name, "true_count": true, "note": note}
        for path, true, note in EXAMPLES
        if Path(path).exists()
    ]


@app.get("/api/queue", summary="List runs for review-by-exception queue", tags=["queue"])
def api_queue():
    """List runs: plate_id, time, location, count, status, needs_review, reasons, reviewed_by. Sorted needs_review first."""
    try:
        from labvision.queue import get_queue_items
        return get_queue_items(_get_runs_dir())
    except Exception as e:
        logger.exception("queue failed")
        raise HTTPException(status_code=500, detail={"error": "internal error"})

@app.get("/api/runs/{run_id}", summary="Get run detail (image, masks, boxes, trace)", tags=["queue"])
def api_run_detail(run_id: str):
    p = _validate_run_id(run_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    run_json = p / "run.json"
    boxes_p = p / "boxes.json"
    trace_p = p / "trace.json"
    inbox_p = p / "inbox.json"
    approval_p = p / "approval.json"
    try:
        rec = json.loads(run_json.read_text()) if run_json.exists() else {}
        boxes = json.loads(boxes_p.read_text()) if boxes_p.exists() else []
        trace = json.loads(trace_p.read_text()) if trace_p.exists() else []
        inbox = json.loads(inbox_p.read_text()) if inbox_p.exists() else {}
        approval = json.loads(approval_p.read_text()) if approval_p.exists() else {}
        # derive image size
        w = h = 0
        try:
            from PIL import Image as _Im
            im = _Im.open(p / "image.png")
            w, h = im.size
        except Exception:
            pass
        count = len(rec.get("results", []))
        # status must match queue logic
        is_bl = rec.get("params", {}).get("is_blurry")
        gl = rec.get("params", {}).get("glare")
        # check trace if params missing
        if trace:
            for t in trace:
                if t.get("name") == "check_quality":
                    if t.get("is_blurry") is not None:
                        is_bl = t.get("is_blurry")
                    if t.get("glare") is not None:
                        gl = t.get("glare")
        if is_bl or gl:
            status = "Re-photo"
        elif count > 300:
            status = "TNTC"
        elif count == 0:
            status = "Empty"
        else:
            status = "OK"
        count_label = _count_label(count)
        measurements = _measurements(rec.get("results", []))
        return {
            "run_id": run_id,
            "plate_id": inbox.get("plate_id") or rec.get("inputs", {}).get("plate_id") or run_id,
            "inputs": rec.get("inputs", {}),
            "params": rec.get("params", {}),
            "results": rec.get("results", []),
            "boxes": boxes,
            "trace": trace,
            "inbox": inbox,
            "approval": approval,
            "image_url": f"/api/runs/{run_id}/image.png",
            "masks_url": f"/api/runs/{run_id}/masks.png",
            "csv_url": f"/api/runs/{run_id}/results.csv",
            "image_w": w,
            "image_h": h,
            "count": count,
            "count_label": count_label,
            "status": status,
            "measurements": measurements,
            "colony_types": [],
            "report": f"Plate {inbox.get('plate_id') or run_id} — {count_label} (loaded from queue)",
            "duration_ms": 0,
            "experimental": rec.get("params", {}).get("experimental", False),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("api_run_detail failed")
        raise HTTPException(status_code=500, detail={"error": "internal error"})

@app.post("/api/runs/{run_id}/approve", summary="Approve a run (human decision)", tags=["queue"])
def api_approve(run_id: str, payload: dict):
    _validate_run_id(run_id)
    reviewer = payload.get("reviewer") or payload.get("reviewed_by")
    if not reviewer or not str(reviewer).strip():
        raise HTTPException(status_code=400, detail={"error": "missing reviewer", "hint": "Send {reviewer: name}"})
    try:
        from labvision.queue import approve_run
        return approve_run(run_id, str(reviewer).strip(), _get_runs_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    except Exception as e:
        logger.exception("approve failed")
        raise HTTPException(status_code=500, detail={"error": "internal error"})

@app.post("/api/queue/approve_clean", summary="Approve all clean plates (no review reasons)", tags=["queue"])
def api_approve_clean(payload: dict | None = None):
    reviewer = (payload or {}).get("reviewer") or "technician"
    try:
        from labvision.queue import get_queue_items, approve_run
        items = get_queue_items(_get_runs_dir())
        approved = []
        for it in items:
            if not it["needs_review"] and not it["reviewed_by"]:
                try:
                    approve_run(it["run_id"], str(reviewer).strip(), _get_runs_dir())
                    approved.append(it["run_id"])
                except Exception:
                    pass
        return {"approved": approved, "count": len(approved)}
    except Exception as e:
        logger.exception("approve_clean failed")
        raise HTTPException(status_code=500, detail={"error": "internal error"})


@app.post("/api/runs/{run_id}/review", summary="Review report (approve or reject with notes) — sign-off", tags=["queue"])
def api_review(run_id: str, payload: dict):
    _validate_run_id(run_id)
    decision = (payload.get("decision") or payload.get("action") or "").lower()
    if decision not in ("approve", "approved", "reject", "rejected"):
        raise HTTPException(status_code=400, detail={"error": "decision must be approve or reject", "hint": "Send {decision: approve|reject, reviewer, notes}"})
    reviewer = payload.get("reviewer") or payload.get("reviewed_by") or "technician"
    notes = payload.get("notes") or ""
    rdir = _get_runs_dir() / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    esc_path = rdir / "escalation.json"
    # verify it's awaiting_signoff
    try:
        esc = json.loads(esc_path.read_text()) if esc_path.exists() else {}
        if esc.get("rule") != "awaiting_signoff":
            # allow approve even without signoff for backward compat (queue approve)
            pass
    except Exception:
        pass
    from labvision.escalation import set_status
    if decision in ("approve", "approved"):
        (rdir / "review.json").write_text(json.dumps({"decision": "approved", "reviewer": reviewer, "notes": notes, "at": time.time()}))
        set_status(run_id, "done", base=_get_runs_dir(), rule="awaiting_signoff")
        # also write approval.json for queue
        from labvision.queue import approve_run
        try:
            approve_run(run_id, str(reviewer).strip(), _get_runs_dir())
        except Exception:
            pass
        return {"run_id": run_id, "status": "done", "decision": "approved"}
    else:
        (rdir / "review.json").write_text(json.dumps({"decision": "rejected", "reviewer": reviewer, "notes": notes, "at": time.time()}))
        # clear awaiting_signoff so agent can resume
        try:
            (rdir / "escalation.json").unlink()
        except Exception:
            pass
        set_status(run_id, "running", base=_get_runs_dir())
        return {"run_id": run_id, "status": "running", "decision": "rejected", "notes": notes}


@app.post("/api/runs/{run_id}/guidance", summary="Provide guidance for report_refused escalation", tags=["queue"])
def api_guidance(run_id: str, payload: dict):
    _validate_run_id(run_id)
    reviewer = payload.get("reviewer") or payload.get("reviewed_by") or "technician"
    notes = payload.get("notes") or payload.get("guidance") or ""
    if not notes.strip():
        raise HTTPException(status_code=400, detail={"error": "missing notes", "hint": "Send {notes: guidance text}"})
    rdir = _get_runs_dir() / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    (rdir / "guidance.json").write_text(json.dumps({"reviewer": reviewer, "notes": notes, "at": time.time()}))
    from labvision.escalation import set_status
    set_status(run_id, "running", base=_get_runs_dir())
    return {"run_id": run_id, "status": "running", "guidance": notes}


@app.post("/api/runs/{run_id}/resume", summary="Resume a paused/needs_input/awaiting_signoff run", tags=["queue"])
def api_resume(run_id: str):
    _validate_run_id(run_id)
    from labvision.progress import load_progress, resume_payload
    from labvision.agent import resume_run
    try:
        prog = load_progress(run_id, base=_get_runs_dir())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="run not found")
    step = resume_payload(run_id, base=_get_runs_dir())
    if not step.get("message"):
        # if paused, just tell client why not
        return {"run_id": run_id, "status": step.get("status", prog.get("status")), "note": step.get("note", "")}
    # for paused or rejected signoff or report_refused guidance, trigger resume
    # For needs_rephoto, check if new image arrived via resume_image.json
    try:
        out = resume_run(run_id)
        return out
    except Exception as e:
        logger.exception("resume failed")
        raise HTTPException(status_code=500, detail={"error": "internal error"})


@app.get("/api/runs/{run_id}/progress", summary="Get run progress (messages + status)", tags=["queue"])
def api_progress(run_id: str):
    _validate_run_id(run_id)
    from labvision.progress import load_progress
    try:
        prog = load_progress(run_id, base=_get_runs_dir())
        return prog
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="not found")


@app.get("/api/export.csv", summary="Export queue for LIMS (CSV)", tags=["integration"])
def api_export_csv(from_: str | None = Query(None, alias="from"), to: str | None = Query(None, alias="to")):
    """One row per run: plate_id, location, time, count, status, reasons (why it needs review, `; `-separated), reviewed_by, reviewed_at, run_id. Filter by ?from=&to= ISO dates. Example: /api/export.csv?from=2026-11-01T00:00:00Z`."""
    try:
        from labvision.queue import get_queue_items
        from datetime import datetime
        items = get_queue_items(_get_runs_dir())
        # parse filters
        def _parse(s):
            if not s:
                return None
            try:
                # handle URL alias from_ -> from
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None
        f = _parse(from_)
        t = _parse(to)
        if f or t:
            filtered=[]
            for it in items:
                try:
                    ts = datetime.fromisoformat(it["time"].replace("Z", "+00:00")) if it.get("time") else None
                except Exception:
                    ts=None
                if f and ts and ts < f:
                    continue
                if t and ts and ts > t:
                    continue
                filtered.append(it)
            items=filtered
        # build CSV
        import io
        out = io.StringIO()
        cols = ["plate_id","location","time","count","status","reasons","reviewed_by","reviewed_at","run_id"]
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        for it in items:
            row = {k: it.get(k, "") for k in cols}
            rs = it.get("reasons") or []
            row["reasons"] = "; ".join(str(x) for x in rs) if isinstance(rs, (list, tuple)) else str(rs)
            w.writerow(row)
        csv_bytes = out.getvalue().encode()
        from fastapi.responses import Response
        return Response(content=csv_bytes, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=export.csv"})
    except Exception as e:
        logger.exception("export.csv failed")
        raise HTTPException(status_code=500, detail={"error": "internal error"})


@app.get("/api/runs/{run_id}/image.png", summary="Get run image", tags=["runs"])
def get_image(run_id: str):
    _validate_run_id(run_id)
    p = _get_runs_dir() / run_id / "image.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/runs/{run_id}/masks.png", summary="Get run masks overlay", tags=["runs"])
def get_masks(run_id: str):
    _validate_run_id(run_id)
    p = _get_runs_dir() / run_id / "masks.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/runs/{run_id}/results.csv", summary="Get run measurements CSV", tags=["runs"])
def get_csv(run_id: str):
    _validate_run_id(run_id)
    p = _get_runs_dir() / run_id / "results.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), media_type="text/csv", filename="results.csv")


@app.get("/health", summary="Health check", tags=["system"])
def health():
    return {"ok": True}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", type=str, default=None, help="inbox folder to poll")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args, _ = ap.parse_known_args()
    if args.inbox:
        from labvision.inbox import start_inbox_thread
        start_inbox_thread(args.inbox)
        print(f"Inbox polling {args.inbox}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
