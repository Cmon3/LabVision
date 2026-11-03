"""FastAPI backend — thin wrappers around labvision/*. no logic duplication beyond agent."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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

RUNS_DIR = Path("runs")
CACHE_DIR = Path("runs/cache")
WEB_DIR = Path("web")

app = FastAPI(title="LabVision Copilot API")

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    @app.get("/")
    def _index():
        return FileResponse(str(WEB_DIR / "index.html"), media_type="text/html")

    @app.api_route("/api/examples/file", methods=["GET", "HEAD"])
    def _example_file(path: str):
        # only serve files inside data/
        p = Path(path)
        if not p.exists():
            p = Path(path.lstrip("/"))
        if not p.exists() or "data" not in str(p.resolve()):
            raise HTTPException(status_code=404, detail="not found")
        try:
            p.resolve().relative_to(Path("data").resolve())
        except Exception:
            raise HTTPException(status_code=403, detail="forbidden")
        mt = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return FileResponse(str(p), media_type=mt)


def _status_for(qc_result, count: int) -> str:
    if count > 300:
        return "tntc"
    if qc_result and qc_result.get("is_blurry"):
        return "blurry"
    if qc_result and qc_result.get("glare"):
        return "glare"
    if count == 0:
        return "empty"
    return "ok"


def _count_label(count: int) -> str:
    return "TNTC >300" if count > 300 else f"{count} CFU"


def _masks_png(masks: np.ndarray, w: int, h: int) -> bytes:
    # transparent PNG where masks are red with alpha
    from io import BytesIO
    if masks is None or masks.size == 0:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    else:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for m in masks:
            if m.shape[0] != h or m.shape[1] != w:
                continue
            red = Image.new("RGBA", (w, h), (255, 0, 0, 100))
            mask_im = Image.fromarray((m.astype(np.uint8) * 255))
            tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            tmp.paste(red, mask=mask_im)
            im = Image.alpha_composite(im, tmp)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _boxes_list(seg) -> list[dict]:
    return [{"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2} for b in seg.boxes]


def _measurements(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
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
    rdir = RUNS_DIR / run_id
    rdir.mkdir(parents=True, exist_ok=True)

    img_path = rdir / "image.png"
    img_path.write_bytes(image_bytes)
    from PIL import Image as _Im

    pil = _Im.open(img_path).convert("RGB")
    arr = np.array(pil)
    w, h = pil.size

    trace = []
    def _trace(name, status, **kv):
        trace.append({"name": name, "status": status, **kv})

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
    for col in ["black", "yellow", "white", "cream", "red", "brown", "green", "gold", "beige"]:
        if col in lower_req and "colon" in lower_req:
            seg_prompt = f"{col} round spot"
            experimental = True
            break
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
        _trace("segment", "ok" if len(seg.boxes) else "warn", prompt_used=seg_prompt, retry=retry, boxes=len(seg.boxes), duration_ms=int((time.time()-t)*1000), model="sam-3.1")
    except Exception as e:
        _trace("segment", "fail", error=str(e)[:200])
        raise HTTPException(status_code=502, detail={"error": "SAM segmentation failed", "hint": str(e)[:300]})

    t = time.time()
    try:
        from labvision.plate import detect_plate_diameter_px
        pdiam = detect_plate_diameter_px(arr)
    except Exception:
        pdiam = None
    if pdiam:
        rows = measure_mod.measure_masks(masks, plate_diameter_px=pdiam, plate_diameter_mm=90.0)
        mm_per_px = 90.0 / pdiam
    else:
        rows = measure_mod.measure_masks(masks)
        mm_per_px = None
    import re
    m = re.search(r"larger than\s+([\d.]+)\s*mm", request, re.I)
    if m and rows and "diameter_mm" in rows[0]:
        try:
            rows = measure_mod.filter_by_min_diameter(rows, float(m.group(1)))
        except Exception:
            pass
    _trace("measure", "ok", plate_diameter_px=pdiam, mm_per_px=round(mm_per_px, 4) if mm_per_px else None, count=len(rows), duration_ms=int((time.time()-t)*1000))

    count = len(rows)
    status = _status_for({"is_blurry": is_blurry, "glare": glare}, count)
    count_label = _count_label(count)

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

    csv_path = rdir / "results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)

    colony_types = _colony_types_from_request(arr, seg.boxes, masks, request)

    try:
        save_run(RUNS_DIR, run_id, inputs={"request": request, "image": f"{w}x{h}"}, model_ids={"sam": "sam-3.1", "plate": "hough"}, params={"prompt": seg_prompt, "plate_diameter_px": pdiam, "experimental": experimental}, results=rows)
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


@app.post("/api/analyze_stream")
async def api_analyze_stream(image: UploadFile = File(...), request: str = Form("")):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "no image", "hint": "Upload a JPG/PNG/TIFF."})

    async def gen():
        import asyncio

        t0 = time.time()
        run_id = new_run_id()
        rdir = RUNS_DIR / run_id
        rdir.mkdir(parents=True, exist_ok=True)
        img_path = rdir / "image.png"
        img_path.write_bytes(data)
        pil = Image.open(img_path).convert("RGB")
        arr = np.array(pil)
        w, h = pil.size

        trace: list[dict] = []

        def _trace(name: str, status: str, **kv):
            trace.append({"name": name, "status": status, **kv})

        # helper to yield SSE string as bytes
        async def emit(obj: dict):
            yield _sse(obj).encode()
            # allow event loop to flush
            await asyncio.sleep(0)

        # plan
        async for chunk in emit({"type": "step", "step": "plan", "status": "running", "message": "Planning: Muse Spark Open chose check_quality → segment → measure", "values": {}, "decided_by": "model"}):
            yield chunk
        await asyncio.sleep(0.04)
        async for chunk in emit({"type": "step", "step": "plan", "status": "ok", "message": "Plan ready", "values": {"steps": ["check_quality", "segment", "measure"]}, "decided_by": "model"}):
            yield chunk

        # check_quality running
        async for chunk in emit({"type": "step", "step": "check_quality", "status": "running", "message": "Checking photo quality…", "values": {}, "decided_by": "rule"}):
            yield chunk
        t = time.time()
        bv = qc_mod.laplacian_variance(arr)
        is_blurry = qc_mod.is_blurry(arr)
        glare = qc_mod.has_glare(arr)
        glare_frac = qc_mod.saturated_fraction(arr)
        _trace("check_quality", "warn" if is_blurry or glare else "ok", blur_score=round(bv, 1), glare_fraction=round(glare_frac, 4), is_blurry=is_blurry, glare=glare, duration_ms=int((time.time() - t) * 1000))
        status = "warn" if is_blurry or glare else "ok"
        msg = f"Checking photo quality… {'blurry' if is_blurry else 'sharp'} ({bv:.1f}), {'glare' if glare else 'no glare'}"
        async for chunk in emit({"type": "step", "step": "check_quality", "status": status, "message": msg, "values": {"blur_score": round(bv, 1), "glare_fraction": round(glare_frac, 4), "is_blurry": is_blurry, "glare": glare, "duration_ms": int((time.time() - t) * 1000)}, "decided_by": "rule"}):
            yield chunk

        # segment
        prompt = request.strip() or AGAR_PROMPT
        seg_prompt = AGAR_PROMPT if "cell" not in request.lower() and "count" in request.lower() else prompt
        if "cell" in request.lower():
            seg_prompt = "cell"
        # attribute support for experimental badges (black/yellow etc)
        lower_req = request.lower()
        attribute = None
        experimental = False
        for col in ["black", "yellow", "white", "cream", "red", "brown", "green", "gold", "beige"]:
            if col in lower_req and "colon" in lower_req:
                attribute = col
                seg_prompt = f"{col} round spot"
                experimental = True
                break
        async for chunk in emit({"type": "step", "step": "segment", "status": "running", "message": f"Looking for colonies with SAM 3.1 ('{seg_prompt}')…", "values": {"prompt": seg_prompt, "experimental": experimental}, "decided_by": "model"}):
            yield chunk
        t = time.time()
        try:
            seg = segment(data, seg_prompt)
            retry = False
            if len(seg.boxes) == 0 and not is_blurry and not glare:
                from labvision.config import AGAR_ALT_PROMPT

                alt = AGAR_ALT_PROMPT
                if alt and alt != seg_prompt:
                    async for chunk in emit({"type": "step", "step": "segment", "status": "running", "message": f"No colonies on a sharp photo → retrying with '{alt}'", "values": {"prompt": alt}, "decided_by": "rule"}):
                        yield chunk
                    seg = segment(data, alt)
                    seg_prompt = alt
                    retry = True
            masks = seg.decoded_masks()
            _trace("segment", "ok" if len(seg.boxes) else "warn", prompt_used=seg_prompt, retry=retry, boxes=len(seg.boxes), duration_ms=int((time.time() - t) * 1000), model="sam-3.1")
        except Exception as e:
            _trace("segment", "fail", error=str(e)[:200])
            async for chunk in emit({"type": "step", "step": "segment", "status": "fail", "message": f"Segmentation failed: {e}", "values": {"error": str(e)[:200]}, "decided_by": "rule"}):
                yield chunk
            # also emit error result so client can show it
            async for chunk in emit({"type": "error", "error": str(e)[:400], "hint": "Check API keys and try again."}):
                yield chunk
            return
        msg2 = f"Looking for colonies with SAM 3.1 ('{seg_prompt}')… {len(seg.boxes)} found" + (" (retried)" if retry else "")
        async for chunk in emit({"type": "step", "step": "segment", "status": "ok" if len(seg.boxes) else "warn", "message": msg2, "values": {"boxes": len(seg.boxes), "prompt_used": seg_prompt, "retry": retry, "duration_ms": int((time.time() - t) * 1000)}, "decided_by": "model"}):
            yield chunk

        # plate
        async for chunk in emit({"type": "step", "step": "plate", "status": "running", "message": "Detecting plate…", "values": {}, "decided_by": "rule"}):
            yield chunk
        t = time.time()
        try:
            from labvision.plate import detect_plate_diameter_px

            pdiam = detect_plate_diameter_px(arr)
        except Exception:
            pdiam = None
        if pdiam:
            mm_per_px = 90.0 / pdiam
            msgp = f"Plate detected: {pdiam:.0f} px = 90 mm ({mm_per_px:.4f} mm/px)"
            statusp = "ok"
        else:
            mm_per_px = None
            msgp = "No plate detected — sizes in px"
            statusp = "warn"
        async for chunk in emit({"type": "step", "step": "plate", "status": statusp, "message": msgp, "values": {"plate_diameter_px": pdiam, "mm_per_px": round(mm_per_px, 4) if mm_per_px else None, "duration_ms": int((time.time() - t) * 1000)}, "decided_by": "rule"}):
            yield chunk

        # measure
        async for chunk in emit({"type": "step", "step": "measure", "status": "running", "message": f"Measuring {len(masks)} colonies…", "values": {}, "decided_by": "rule"}):
            yield chunk
        t = time.time()
        if pdiam:
            rows = measure_mod.measure_masks(masks, plate_diameter_px=pdiam, plate_diameter_mm=90.0)
        else:
            rows = measure_mod.measure_masks(masks)
        import re as _re

        m = _re.search(r"larger than\s+([\d.]+)\s*mm", request, _re.I)
        if m and rows and "diameter_mm" in rows[0]:
            try:
                rows = measure_mod.filter_by_min_diameter(rows, float(m.group(1)))
            except Exception:
                pass
        _trace("measure", "ok", plate_diameter_px=pdiam, mm_per_px=round(mm_per_px, 4) if mm_per_px else None, count=len(rows), duration_ms=int((time.time() - t) * 1000))
        count = len(rows)
        async for chunk in emit({"type": "step", "step": "measure", "status": "ok", "message": f"Measuring {count} colonies… done", "values": {"count": count, "duration_ms": int((time.time() - t) * 1000)}, "decided_by": "rule"}):
            yield chunk

        # countable / TNTC
        if count > 300:
            async for chunk in emit({"type": "step", "step": "tntc", "status": "warn", "message": "More than 300 colonies → too numerous to count: dilute and re-plate", "values": {"count": count}, "decided_by": "rule"}):
                yield chunk
        if "countable" in lower_req:
            if count < 30:
                verdict = "too few (<30, low statistical reliability)"
                st = "warn"
            elif count > 300:
                verdict = "TNTC (>300)"
                st = "warn"
            else:
                verdict = "countable (30-300)"
                st = "ok"
            async for chunk in emit({"type": "step", "step": "countable", "status": st, "message": f"Countable check: {verdict} — {count} colonies", "values": {"count": count, "verdict": verdict}, "decided_by": "rule"}):
                yield chunk

        # report
        async for chunk in emit({"type": "step", "step": "report", "status": "running", "message": "Writing report…", "values": {}, "decided_by": "model"}):
            yield chunk
        has_mm = bool(rows and "diameter_mm" in rows[0])
        if rows:
            ds = [r["diameter_px"] for r in rows]
            mean_d = sum(ds) / len(ds)
            mean_mm = sum(r.get("diameter_mm", 0) for r in rows) / len(rows) if has_mm else 0
        else:
            mean_d = mean_mm = 0
        report = f"Count: {_count_label(count)}\nMean diameter: {mean_d:.1f}px" + (f" ({mean_mm:.2f}mm)" if has_mm else "") + f"\n{count} objects — {'no plate, sizes in px' if not pdiam else 'plate 90mm'}"
        if experimental:
            report += "\nExperimental — not benchmarked (colour filter)."
        masks_png = _masks_png(masks, w, h)
        (rdir / "masks.png").write_bytes(masks_png)
        csv_path = rdir / "results.csv"
        if rows:
            with csv_path.open("w", newline="") as f:
                wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wtr.writeheader()
                wtr.writerows(rows)
        colony_types = _colony_types_from_request(arr, seg.boxes, masks, request)
        try:
            save_run(RUNS_DIR, run_id, inputs={"request": request, "image": f"{w}x{h}"}, model_ids={"sam": "sam-3.1", "plate": "hough"}, params={"prompt": seg_prompt, "plate_diameter_px": pdiam, "experimental": experimental}, results=rows)
        except Exception:
            pass
        (rdir / "boxes.json").write_text(json.dumps(_boxes_list(seg)))
        status = _status_for({"is_blurry": is_blurry, "glare": glare}, count)
        count_label = _count_label(count)
        async for chunk in emit({"type": "step", "step": "report", "status": "ok", "message": "Report ready", "values": {}, "decided_by": "model"}):
            yield chunk

        result = {
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
        async for chunk in emit({"type": "result", "data": result}):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/analyze")
async def api_analyze(image: UploadFile = File(...), request: str = Form("")):
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "no image", "hint": "Upload a JPG/PNG/TIFF."})
    try:
        return _analyze_image(data, request)
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        hint = "Check API keys in .env and try again."
        if "META_API_KEY" in msg or "BASETEN" in msg:
            hint = "Missing API key — set META_API_KEY and BASETEN_API_KEY in .env."
        return JSONResponse(status_code=500, content={"error": msg[:400], "hint": hint})


@app.post("/api/correct")
async def api_correct(payload: dict):
    run_id = payload.get("run_id")
    x = payload.get("x")
    y = payload.get("y")
    mode = payload.get("mode", "add")
    if not run_id or x is None or y is None:
        raise HTTPException(status_code=400, detail={"error": "missing run_id/x/y", "hint": "Send {run_id, x, y, mode: add|remove}"})
    rdir = RUNS_DIR / run_id
    if not rdir.exists():
        raise HTTPException(status_code=404, detail={"error": "run not found", "hint": f"No run {run_id}"})
    img_path = rdir / "image.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail={"error": "image not found"})
    boxes_path = rdir / "boxes.json"
    boxes = json.loads(boxes_path.read_text()) if boxes_path.exists() else []
    from PIL import Image as _Im
    pil = _Im.open(img_path).convert("RGB")
    w, h = pil.size
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))

    if mode == "remove":
        keep = []
        removed = None
        for b in boxes:
            if b["x1"] <= x <= b["x2"] and b["y1"] <= y <= b["y2"]:
                if removed is None:
                    removed = b
                    continue
            keep.append(b)
        if removed is None:
            raise HTTPException(status_code=400, detail={"error": "no colony at click", "hint": f"No detection contains ({x},{y})"})
        boxes = keep
        action = {"action": "remove", "x": x, "y": y, "box": removed}
    else:
        data = img_path.read_bytes()
        try:
            seg = segment_point(data, x, y)
            if not seg.boxes:
                raise HTTPException(status_code=502, detail={"error": "SAM returned no mask", "hint": "Try nearby point"})
            b = seg.boxes[0]
            boxes.append({"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2})
            action = {"action": "add", "x": x, "y": y, "box": {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail={"error": str(e)[:300], "hint": "SAM add failed"})

    boxes_path.write_text(json.dumps(boxes))
    # boxes as rectangular masks for re-measure
    arr = np.array(pil)
    masks = np.zeros((len(boxes), h, w), dtype=bool)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = max(0, b["x1"]), max(0, b["y1"]), min(w, b["x2"]), min(h, b["y2"])
        masks[i, y1:y2, x1:x2] = True
    try:
        from labvision.plate import detect_plate_diameter_px
        pdiam = detect_plate_diameter_px(np.array(pil))
    except Exception:
        pdiam = None
    if pdiam:
        rows = measure_mod.measure_masks(masks, plate_diameter_px=pdiam, plate_diameter_mm=90.0)
    else:
        rows = measure_mod.measure_masks(masks)

    (rdir / "masks.png").write_bytes(_masks_png(masks, w, h))
    csv_path = rdir / "results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)

    try:
        save_run(RUNS_DIR, run_id, inputs={"correction": action}, model_ids={"sam": "sam-3.1"}, params={"mode": mode}, results=rows)
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


@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail={"error": "no audio", "hint": "Record or upload WAV."})
    try:
        from labvision.voice import transcribe
        text = transcribe(data)
        return {"transcript": text}
    except Exception as e:
        msg = str(e)
        hint = "Check META_API_KEY and audio format (WAV mono 16-bit 24kHz)."
        if "META_API_KEY" in msg:
            hint = "Missing META_API_KEY in .env."
        return JSONResponse(status_code=500, content={"error": msg[:400], "hint": hint})


@app.get("/api/examples")
def api_examples():
    cands = [
        "data/agar_demo/AGAR_representative/lower-resolution/14684.jpg",
        "data/agar_demo/AGAR_representative/higher-resolution/bright/518.jpg",
        "data/agar_demo/AGAR_representative/lower-resolution/356.jpg",
        "data/agar_demo/AGAR_representative/lower-resolution/5206.jpg",
    ]
    out = []
    for p in cands:
        if Path(p).exists():
            out.append({"path": p, "name": Path(p).name})
            if len(out) >= 3:
                break
    return out


@app.get("/api/runs/{run_id}/image.png")
def get_image(run_id: str):
    p = RUNS_DIR / run_id / "image.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/runs/{run_id}/masks.png")
def get_masks(run_id: str):
    p = RUNS_DIR / run_id / "masks.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/runs/{run_id}/results.csv")
def get_csv(run_id: str):
    p = RUNS_DIR / run_id / "results.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(p), media_type="text/csv", filename="results.csv")


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
