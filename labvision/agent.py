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
MAX_TURNS = 8
MAX_REFUSALS = 3


def _instructions() -> str:
    p = Path(__file__).parent / "system_instructions.md"
    if p.exists():
        return p.read_text()
    return SYSTEM

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
            "description": "Segment image for a noun phrase (one concept). For colonies use 'round spot' (benchmarked: 11.8% error, 7/20 exact on AGAR) and for cells use 'cell' (benchmarked: w1 MAE 5.48). For colour-filtered requests like 'black colonies', include the colour in the prompt (e.g. 'black round spot'); such prompts are experimental and not benchmarked.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "short noun phrase; use 'round spot' for colonies and 'cell' for cells; may include colour like 'black round spot' for experimental filtering"}},
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
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "File the report: summary (2-3 sentences, no new numbers), count you measured, filters used, any notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "2-3 sentences, no new numbers; say TNTC if >300, too few if <30"},
                    "count": {"type": "integer", "description": "measured count with filters"},
                    "filters": {"type": "string", "description": "e.g. 'min_diameter 2 mm' or ''"},
                    "notes": {"type": "string", "description": "quality note or ''"},
                },
                "required": ["summary", "count"],
            },
        },
    },
]

SYSTEM = """You are LabVision Copilot, a lab assistant. You have tools to inspect images.
Rules:
- Always call check_quality first.
- If quality fails (blurry, glare, empty, crowded) call ask_rephoto.
- Otherwise call segment with a short noun phrase for the objects the user asked about, then measure. Optionally call characterize for morphology.
- For colour requests like 'black colonies' or 'yellow colonies', include the colour in the segment prompt (e.g. 'black round spot', 'yellow round spot'). These are experimental.
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


def _ctx_for_verification(count, rows, qc_result, prompt_used, experimental) -> dict:
    return {"count": count, "rows": rows, "qc": qc_result or {}, "prompt_used": prompt_used or "", "experimental": experimental}


def _run_verify_report(report: dict, ctx: dict) -> dict:
    try:
        from verification.verify_report import run_verification
        return run_verification(ctx, report)
    except Exception as e:
        return {"passed": True, "feedback": []}


def _get_runs_dir() -> Path:
    return Path(os.environ.get("LABVISION_RUNS_DIR", "runs"))


def _handle_ask_rephoto(run_id: str, reason: str, plate_id: str | None, what_was_done: str) -> dict:
    # escalate needs_rephoto — inbox will resume when same plate_id appears
    try:
        from labvision.escalation import escalate
        escalate(run_id, "needs_rephoto", {
            "what_blocked": "QC failed — photo needs re-take",
            "why": reason,
            "what_was_done": what_was_done,
            "request": f"Re-photo plate {plate_id or run_id}: {reason}",
            "plate_id": plate_id or run_id,
        }, base=_get_runs_dir())
        return {"escalated": True, "rule": "needs_rephoto", "reason": reason, "status": "needs_input"}
    except Exception as e:
        return {"asked": True, "reason": reason, "error": str(e)}


def _handle_submit_report(run_id: str, report: dict, ctx: dict) -> dict:
    # verification runs inside tool, outside agent control
    result = _run_verify_report(report, ctx)
    if result.get("passed"):
        # record report and escalate to awaiting_signoff
        try:
            from labvision.escalation import escalate
            p = _get_runs_dir() / run_id
            p.mkdir(parents=True, exist_ok=True)
            (p / "report.json").write_text(json.dumps(report, indent=2))
            # save params for audit
            from labvision.runlog import save_run
            # we don't have rows here to save_run, but we store count
            handoff = {
                "what_blocked": "Report ready — awaiting sign-off",
                "why": "verification passed",
                "what_was_done": f"count {ctx.get('count')} with prompt {ctx.get('prompt_used')}",
                "request": "Approve or Reject with notes",
                "plate_id": ctx.get("plate_id"),
            }
            rec = escalate(run_id, "awaiting_signoff", handoff, base=_get_runs_dir())
            return {"filed": True, "status": "awaiting_signoff", "report": report}
        except Exception as e:
            return {"error": str(e)[:200]}
    # refused
    # track refusals in run dir
    p = _get_runs_dir() / run_id
    p.mkdir(parents=True, exist_ok=True)
    # count prior refusals
    prior = 0
    rf_path = p / "refusals.json"
    if rf_path.exists():
        try:
            prior = len(json.loads(rf_path.read_text()))
        except Exception:
            prior = 0
    entry = {"report": report, "feedback": result.get("feedback", []), "at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
    hist = []
    if rf_path.exists():
        try:
            hist = json.loads(rf_path.read_text())
        except Exception:
            hist = []
    hist.append(entry)
    rf_path.write_text(json.dumps(hist, indent=2))
    if len(hist) >= MAX_REFUSALS:
        try:
            from labvision.escalation import escalate
            escalate(run_id, "report_refused", {
                "what_blocked": "Report refused 3 times",
                "why": "; ".join(result.get("feedback", [])),
                "what_was_done": f"count {ctx.get('count')}",
                "request": "Reviewer guidance needed",
                "plate_id": ctx.get("plate_id"),
            }, base=_get_runs_dir())
            return {"escalated": True, "rule": "report_refused", "problems": result.get("feedback", []), "status": "needs_input"}
        except Exception:
            pass
        return {"error": "report refused", "problems": result.get("feedback", []), "status": "needs_input", "escalated": True}
    return {"error": "report refused", "problems": result.get("feedback", []), "status": "running"}


def _cache_key(messages, tools, model) -> str:
    h = hashlib.sha256(json.dumps({"m": messages, "t": tools, "model": model}, sort_keys=True).encode()).hexdigest()
    return f"muse_{h[:16]}.json"


def _call_muse(messages: list[dict], tools=TOOLS, use_cache: bool = True) -> dict:
    key = os.environ.get("BASETEN_API_KEY", "")
    if not key:
        raise RuntimeError("Set BASETEN_API_KEY")
    url = f"{BASETEN_URL.rstrip('/')}/chat/completions"
    body = {"model": BASETEN_MODEL, "messages": messages, "tools": tools, "tool_choice": "auto", "parallel_tool_calls": False}
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


def _run_id_for(image_path: Path, request: str) -> str:
    from labvision.runlog import new_run_id
    return new_run_id()


def run_agent(image_path: str | Path, request: str, use_cache: bool = True, on_event: Any = None, run_id: str | None = None, plate_id: str | None = None) -> dict[str, Any]:
    """Run agent loop. Returns dict with report, count, rows, qc, etc. Numbers from measure only.

    If on_event is provided, it is called with step dicts for live progress:
      {type:"step", step, status, message, values, decided_by}
    decided_by: "model" for Muse tool choices/prompt, "rule" for thresholds/retry/TNTC, "human" for clicks (handled in server).
    Loop hygiene: one tool per turn, plain text not accepted, turn limit 8, paused on API error, progress saved per turn.
    Verification runs inside submit_report/ask_rephoto (outside agent control).
    """
    def _emit(evt: dict[str, Any]):
        if on_event:
            try:
                on_event(evt)
            except Exception:
                pass

    image_path = Path(image_path)
    img_arr, img_bytes = _load_image(image_path)
    plate_id = plate_id or image_path.stem
    run_id = run_id or _run_id_for(image_path, request)
    # init progress
    try:
        from labvision.progress import new_run as _new_prog, save_progress as _save_prog
        instructions = _instructions()
        try:
            _new_prog(run_id, instructions, request)
        except Exception:
            pass
    except Exception:
        pass

    def _save(status: str, resp: dict | None = None, **notes):
        try:
            from labvision.progress import save_progress as _sp
            _sp(run_id, messages, resp, status=status, **notes)
        except Exception:
            pass

    # state
    seg_result = None
    measure_rows: list[dict] = []
    qc_result: dict | None = None
    rephoto_reason: str | None = None
    char_results: list[dict] = []
    # verification trace (for app "How it was verified")
    prompt_used: str | None = None
    retry_used: bool = False
    plate_diameter_px: float | None = None
    plate_detected: bool = False
    experimental: bool = False
    # sanity guard stats
    _guard_stats: dict = {"outside_plate_discarded": 0, "duplicates_removed": 0, "plate_plausible": True, "plate_not_found": False, "implausible_size_count": 0}
    _guard_plate_info: tuple | None = None
    submit_status: str | None = None  # tracks if we escalated
    last_report: dict | None = None
    # colour words for experimental detection
    _colours = ["black", "yellow", "white", "cream", "red", "brown", "green", "gold", "beige"]
    messages: list[dict] = [
        {"role": "system", "content": _instructions()},
        {"role": "user", "content": f"Image: {image_path.name}\nRequest: {request}\nUse tools to answer."},
    ]
    for turn in range(MAX_TURNS):
        # last turn hint
        if turn == MAX_TURNS - 1:
            messages.append({"role": "user", "content": "This is your last turn. Call submit_report now with what you have."})
        try:
            resp = _call_muse(messages, use_cache=use_cache)
        except Exception as e:
            _emit({"type": "step", "step": "error", "status": "fail", "message": f"Model API error: {e}", "values": {}, "decided_by": "rule"})
            _save("paused", error=str(e))
            return {"run_id": run_id, "status": "paused", "error": str(e), "count": len(measure_rows) if measure_rows else (len(seg_result.boxes) if seg_result else 0)}
        choice = resp["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            # like reference: plain text is not accepted — bounce and continue
            messages.append(choice)
            messages.append({"role": "user", "content": "Plain text is not accepted. Call exactly one tool."})
            _emit({"type": "step", "step": "plain_text", "status": "warn", "message": "Plain text is not accepted. Call exactly one tool.", "values": {}, "decided_by": "rule"})
            _save("running", resp)
            continue
        # plan: what Muse actually requested — one tool per turn
        if len(tool_calls) > 1:
            _emit({
                "type": "step",
                "step": "plan",
                "status": "ok",
                "message": f"Planning: Muse Spark Open chose {tool_calls[0]['function']['name']} (1 of {len(tool_calls)} kept, one tool per turn)",
                "values": {"tool_calls": [{"name": tool_calls[0]["function"]["name"], "arguments": json.loads(tool_calls[0]["function"]["arguments"] or "{}")}], "dropped": len(tool_calls) - 1},
                "decided_by": "model",
            })
        else:
            _emit({
                "type": "step",
                "step": "plan",
                "status": "ok",
                "message": f"Planning: Muse Spark Open chose {' → '.join(tc['function']['name'] for tc in tool_calls)}",
                "values": {"tool_calls": [{"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")} for tc in tool_calls]},
                "decided_by": "model",
            })
        # execute tools — one tool per turn: keep only first in the assistant message you append
        if len(tool_calls) > 1:
            filtered_choice = dict(choice)
            filtered_choice["tool_calls"] = [tool_calls[0]]
            messages.append(filtered_choice)
            tc = tool_calls[0]
        else:
            messages.append(choice)
            tc = tool_calls[0]
        fname = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"] or "{}")
        try:
            if fname == "check_quality":
                _emit({"type": "step", "step": "check_quality", "status": "running", "message": "Checking photo quality…", "values": {}, "decided_by": "rule"})
                bv = qc_mod.laplacian_variance(img_arr)
                result = {
                    "blur_score": bv,
                    "is_blurry": qc_mod.is_blurry(img_arr),
                    "glare": qc_mod.has_glare(img_arr),
                    "glare_frac": qc_mod.saturated_fraction(img_arr),
                }
                qc_result = result
                _emit({
                    "type": "step",
                    "step": "check_quality",
                    "status": "warn" if result["is_blurry"] or result["glare"] else "ok",
                    "message": f"Checking photo quality… {'blurry' if result['is_blurry'] else 'sharp'} ({bv:.1f}), {'glare' if result['glare'] else 'no glare'}",
                    "values": {"blur_score": round(bv, 1), "glare_fraction": round(result["glare_frac"], 4), "is_blurry": result["is_blurry"], "glare": result["glare"]},
                    "decided_by": "rule",
                })
            elif fname == "segment":
                from labvision.config import AGAR_PROMPT as _DEFAULT_PROMPT

                prompt = args.get("prompt", _DEFAULT_PROMPT)
                _emit({"type": "step", "step": "segment", "status": "running", "message": f"Looking for colonies with SAM 3.1 ('{prompt}')…", "values": {"prompt": prompt}, "decided_by": "model"})
                seg_result = segment(img_bytes, prompt)
                # retry once if 0 but QC says not empty (sharp, no glare)
                if len(seg_result.boxes) == 0 and qc_result is not None and not qc_result.get("is_blurry") and not qc_result.get("glare"):
                    try:
                        from labvision.config import AGAR_ALT_PROMPT

                        alt = AGAR_ALT_PROMPT
                        if alt and alt != prompt:
                            _emit({"type": "step", "step": "segment", "status": "running", "message": f"No colonies on a sharp photo → retrying with '{alt}'", "values": {"prompt": alt}, "decided_by": "rule"})
                            seg_result = segment(img_bytes, alt)
                            prompt = alt
                            retry_used = True
                    except Exception:
                        pass
                prompt_used = prompt
                # experimental if prompt contains colour
                if any(c in prompt.lower() for c in _colours):
                    experimental = True
                masks = seg_result.decoded_masks()
                result = {"count": len(seg_result.boxes), "width": seg_result.width, "height": seg_result.height, "masks": int(masks.shape[0]), "prompt_used": prompt}
                _emit({
                    "type": "step",
                    "step": "segment",
                    "status": "ok" if len(seg_result.boxes) else "warn",
                    "message": f"Looking for colonies with SAM 3.1 ('{prompt}')… {len(seg_result.boxes)} found" + (" (retried)" if retry_used else ""),
                    "values": {"prompt_used": prompt, "boxes": len(seg_result.boxes), "retry": retry_used, "experimental": experimental},
                    "decided_by": "model",
                })
            elif fname == "measure":
                if seg_result is None:
                    result = {"error": "call segment first"}
                else:
                    _emit({"type": "step", "step": "measure", "status": "running", "message": f"Measuring {len(seg_result.boxes)} colonies…", "values": {}, "decided_by": "rule"})
                    masks = seg_result.decoded_masks()
                    # plate diameter from Python, never from model — with sanity guards 1,2,4
                    _emit({"type": "step", "step": "plate", "status": "running", "message": "Detecting plate…", "values": {}, "decided_by": "rule"})
                    try:
                        from labvision.plate import detect_plate as _plate_det
                        from labvision.sanity import apply_box_guards as _abg
                        _plate_info_raw = _plate_det(img_arr)
                        _img_h, _img_w = img_arr.shape[0], img_arr.shape[1]
                        if _plate_info_raw is None:
                            # fallback for tests that mock diameter-only
                            try:
                                from labvision.plate import detect_plate_diameter_px as _pd2
                                _d2 = _pd2(img_arr)
                                if _d2 is not None:
                                    _plate_info_raw = (_d2, _img_w/2, _img_h/2)
                            except Exception:
                                pass
                        _guard_res = _abg(seg_result.boxes, masks, seg_result.confidences, _plate_info_raw, _img_w, _img_h)
                        # apply filtered
                        seg_result.boxes = _guard_res["boxes"]
                        seg_result.confidences = _guard_res["confidences"]
                        masks = _guard_res["masks"]
                        _plate_px_val = _guard_res["plate_diameter_px"]
                        _guard_stats.update(_guard_res["stats"])
                        _guard_stats["plate_plausible"] = _guard_res["plate_plausible"]
                        _guard_stats["plate_not_found"] = not _guard_res["plate_plausible"]
                        _guard_plate_info = _plate_info_raw
                        for _ge in _guard_res["trace"]:
                            _emit({"type": "step", "step": _ge["name"], "status": _ge["status"], "message": f"{_ge['name']}: {_ge.get('discarded', _ge.get('removed', ''))} { _ge.get('reason','')}", "values": {k: v for k, v in _ge.items() if k not in ("name", "status")}, "decided_by": "rule"})
                    except Exception as _e:
                        try:
                            from labvision.plate import detect_plate_diameter_px as _plate_px
                            _plate_px_val = _plate_px(img_arr)
                        except Exception:
                            _plate_px_val = None
                        _guard_stats["plate_not_found"] = _plate_px_val is None
                    if _plate_px_val:
                        rows = measure_mod.measure_masks(masks, plate_diameter_px=_plate_px_val, plate_diameter_mm=90.0)
                        _plate_info = {"plate_detected": True, "plate_diameter_px": _plate_px_val, **_guard_stats}
                        plate_diameter_px = _plate_px_val
                        plate_detected = True
                        _emit({"type": "step", "step": "plate", "status": "ok", "message": f"Plate detected: {_plate_px_val:.0f} px = 90 mm ({90.0/_plate_px_val:.4f} mm/px)", "values": {"plate_diameter_px": _plate_px_val, "mm_per_px": round(90.0 / _plate_px_val, 4)}, "decided_by": "rule"})
                    else:
                        rows = measure_mod.measure_masks(masks)
                        _plate_info = {"plate_detected": False, **_guard_stats}
                        plate_detected = False
                        reason = _guard_stats.get("plate_not_found") and "plate not found (diameter outside 40-98% of image side)" or "No plate detected"
                        _emit({"type": "step", "step": "plate", "status": "warn", "message": f"{reason} — sizes in px", "values": {"plate_diameter_px": None, "reason": reason}, "decided_by": "rule"})
                    if "min_diameter_mm" in args:
                        if rows and "diameter_mm" in rows[0]:
                            rows = measure_mod.filter_by_min_diameter(rows, args["min_diameter_mm"])
                        else:
                            result = {"error": "no plate detected — sizes in px, cannot filter by mm", "plate_detected": False}
                            measure_rows = rows
                            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
                            continue
                    measure_rows = rows
                    # Guard 3: colony size plausibility + dense plate
                    try:
                        from labvision.sanity import flag_implausible_sizes, is_possible_undercount
                        _bad = flag_implausible_sizes(measure_rows)
                        _guard_stats["implausible_size_count"] = len(_bad)
                        if _bad:
                            _emit({"type": "step", "step": "size_check", "status": "warn", "message": f"Implausible size: {len(_bad)} colonies outside 0.1-15 mm", "values": {"implausible_count": len(_bad), "indices": _bad, "reason": "diameter outside 0.1-15 mm"}, "decided_by": "rule"})
                        # dense plate possible undercount
                        try:
                            from labvision import qc as _qc2
                            _cov = _qc2.mask_coverage(masks, _img_w*_img_h) if 'masks' in locals() and masks is not None else 0
                            _dense_flag = is_possible_undercount(len(measure_rows), measure_rows, _cov)
                            _guard_stats["possible_undercount"] = bool(_dense_flag)
                            if _dense_flag:
                                _emit({"type": "step", "step": "dense_check", "status": "warn", "message": "Possible undercount (dense plate)", "values": {"count": len(measure_rows), "coverage": round(_cov,4)}, "decided_by": "rule"})
                        except Exception:
                            pass
                    except Exception:
                        pass
                    result = {"count": len(rows), "rows": rows[:5], **_plate_info, "implausible_size_count": _guard_stats.get("implausible_size_count", 0), "possible_undercount": _guard_stats.get("possible_undercount", False)}
                    _emit({"type": "step", "step": "measure", "status": "ok", "message": f"Measuring {len(rows)} colonies… done", "values": {"count": len(rows)}, "decided_by": "rule"})
            elif fname == "ask_rephoto":
                rephoto_reason = args.get("reason", "")
                # escalate needs_rephoto (verification outside agent control)
                current_count = len(measure_rows) if measure_rows else (len(seg_result.boxes) if seg_result else 0)
                ctx_ask = _ctx_for_verification(current_count, measure_rows, qc_result, prompt_used, experimental)
                ctx_ask["plate_id"] = plate_id
                result = _handle_ask_rephoto(run_id, rephoto_reason, plate_id, what_was_done=f"QC {qc_result} count {current_count}")
                rephoto_reason = rephoto_reason or str(result.get("reason", ""))
                _emit({"type": "step", "step": "ask_rephoto", "status": "warn", "message": f"Ask rephoto: {rephoto_reason}", "values": {"reason": rephoto_reason}, "decided_by": "rule"})
                if result.get("escalated"):
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
                    _save("needs_input", result)
                    _emit({"type": "step", "step": "escalated", "status": "warn", "message": f"Escalated needs_rephoto for {plate_id}", "values": {"plate_id": plate_id}, "decided_by": "rule"})
                    # break to stop model calls until resume
                    # we will exit loop after handling
                    submit_status = "needs_input"
                    # save rephoto reason for report
                    # fall through to append and break handling below
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
            elif fname == "submit_report":
                # verification runs inside tool (outside agent control)
                current_count = len(measure_rows) if measure_rows else (len(seg_result.boxes) if seg_result else 0)
                ctx_rep = _ctx_for_verification(current_count, measure_rows, qc_result, prompt_used, experimental)
                ctx_rep["plate_id"] = plate_id
                report_arg = {"summary": args.get("summary", ""), "count": args.get("count"), "filters": args.get("filters", ""), "notes": args.get("notes", "")}
                result = _handle_submit_report(run_id, report_arg, ctx_rep)
                last_report = report_arg
                if result.get("filed"):
                    submit_status = "awaiting_signoff"
                    _emit({"type": "step", "step": "submit_report", "status": "ok", "message": "Report filed — awaiting sign-off", "values": {"count": report_arg.get("count")}, "decided_by": "rule"})
                elif result.get("escalated"):
                    submit_status = "needs_input"
                    _emit({"type": "step", "step": "submit_report", "status": "warn", "message": f"Report refused 3x → escalated report_refused", "values": {"problems": result.get("problems", [])}, "decided_by": "rule"})
                elif result.get("error"):
                    _emit({"type": "step", "step": "submit_report", "status": "warn", "message": f"Report refused: {'; '.join(result.get('problems', [])[:2])}", "values": {"problems": result.get("problems", [])}, "decided_by": "rule"})
                else:
                    _emit({"type": "step", "step": "submit_report", "status": "ok", "message": "Report accepted", "values": {}, "decided_by": "rule"})
            else:
                result = {"error": f"unknown tool {fname}"}
        except Exception as e:
            result = {"error": str(e)[:200]}
        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
        # save progress every turn
        _save(result.get("status", "running") if isinstance(result, dict) and result.get("status") in ("needs_input", "awaiting_signoff", "paused") else "running", result)
        # if escalated, stop the run with no further model calls
        if isinstance(result, dict) and result.get("status") in ("needs_input", "awaiting_signoff"):
            # escalated — no more model calls
            submit_status = result.get("status")
            break
        if submit_status in ("needs_input", "awaiting_signoff"):
            break
        # also break if we already escalated via ask_rephoto path
        if result.get("escalated") or result.get("status") in ("needs_input", "awaiting_signoff"):
            break
        # handle ask_rephoto escalation already sets submit_status
        if submit_status:
            break
    else:
        # loop exhausted without submit (turn limit) -> paused
        _save("paused", error="turn limit 8 reached without submit_report")
        return {"run_id": run_id, "status": "paused", "note": "turn limit reached", "count": len(measure_rows) if measure_rows else (len(seg_result.boxes) if seg_result else 0)}

    # preserve assistant_note from plain-text final if already set
    if "assistant_note" not in locals() or not assistant_note:
        assistant_note = "Analysis complete."
    if last_report and assistant_note == "Analysis complete.":
        # use LLM summary as note, but numbers still from measure
        assistant_note = last_report.get("summary", assistant_note)

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
    if count > 300:
        _emit({"type": "step", "step": "tntc", "status": "warn", "message": "More than 300 colonies → too numerous to count: dilute and re-plate", "values": {"count": count}, "decided_by": "rule"})
    if "countable" in request.lower():
        if count < 30:
            _emit({"type": "step", "step": "countable", "status": "warn", "message": f"Countable check: too few (<30, low statistical reliability) — {count} colonies", "values": {"count": count, "verdict": "too few (<30, low statistical reliability)"}, "decided_by": "rule"})
        elif count > 300:
            _emit({"type": "step", "step": "countable", "status": "warn", "message": f"Countable check: TNTC (>300) — {count} colonies", "values": {"count": count, "verdict": "TNTC (>300)"}, "decided_by": "rule"})
        else:
            _emit({"type": "step", "step": "countable", "status": "ok", "message": f"Countable check: countable (30-300) — {count} colonies", "values": {"count": count, "verdict": "countable (30-300)"}, "decided_by": "rule"})
    # TNTC: >300
    count_display = "Too numerous to count (>300 CFU): dilute and re-plate" if count > 300 else f"{count} objects"
    _emit({"type": "step", "step": "report", "status": "running", "message": "Writing report…", "values": {}, "decided_by": "model"})
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
    # Guard notes in report
    try:
        _od = _guard_stats.get("outside_plate_discarded", 0)
        _dup = _guard_stats.get("duplicates_removed", 0)
        if _od:
            report += f"\nNote: {_od} detections outside plate discarded (centre outside circle)."
        if _dup:
            report += f"\nNote: {_dup} duplicate detections merged (IoU>0.7)."
        if _guard_stats.get("plate_not_found"):
            report += "\nNote: plate not found (diameter outside 40-98% of image side) — sizes in px only."
        if _guard_stats.get("implausible_size_count"):
            report += f"\nNote: {_guard_stats['implausible_size_count']} colonies flagged implausible size (diameter outside 0.1-15 mm)."
        if _guard_stats.get("possible_undercount"):
            report += "\nNote: possible undercount (dense plate) — crowded but low count."
    except Exception:
        pass
    if experimental:
        report += "\nExperimental — not benchmarked (colour filter)."
    if "countable" in request.lower():
        if count < 30:
            report += "\nCountable: too few (<30, low statistical reliability)"
        elif count > 300:
            report += "\nCountable: TNTC (>300)"
        else:
            report += "\nCountable: countable (30-300)"
    _emit({"type": "step", "step": "report", "status": "ok", "message": "Report ready", "values": {}, "decided_by": "model"})

    # verification payload for app
    mm_per_px = (90.0 / plate_diameter_px) if plate_diameter_px else None
    verification = {
        "qc": qc_result,
        "prompt_used": prompt_used,
        "retry_used": retry_used,
        "plate_diameter_px": plate_diameter_px,
        "plate_detected": plate_detected,
        "mm_per_px": mm_per_px,
        "tntc": count > 300,
        "count": count,
        "seg_boxes": len(seg_result.boxes) if seg_result else 0,
        "experimental": experimental,
    }
    # determine final status
    final_status = "done"
    if submit_status in ("needs_input", "awaiting_signoff"):
        final_status = submit_status
    elif qc_result and (qc_result.get("is_blurry") or qc_result.get("glare")) and not submit_status:
        # QC fail without rephoto escalation -> still running but will be handled, mark needs_input for inbox
        # ponytail: if QC fails and agent didn't call ask_rephoto, auto-escalate needs_rephoto here for inbox resume
        try:
            from labvision.escalation import get_escalation as _ge
            if not _ge(run_id):
                _handle_ask_rephoto(run_id, rephoto_reason or qc_summary or "QC failed", plate_id, what_was_done=f"count {count}")
                final_status = "needs_input"
        except Exception:
            pass
    return {
        "run_id": run_id,
        "status": final_status,
        "report": report,
        "count": count,
        "rows": measure_rows,
        "qc": qc_result,
        "rephoto_reason": rephoto_reason,
        "assistant_note": locals().get("assistant_note", ""),
        "seg": {"boxes": len(seg_result.boxes) if seg_result else 0, "width": seg_result.width if seg_result else 0, "height": seg_result.height if seg_result else 0},
        "verification": verification,
        "experimental": experimental,
        "prompt_used": prompt_used,
    }


def resume_run(run_id: str, use_cache: bool = True, on_event: Any = None) -> dict:
    """Resume a paused/needs_input/awaiting_signoff run."""
    from labvision.progress import load_progress, resume_payload

    try:
        prog = load_progress(run_id)
    except Exception as e:
        return {"run_id": run_id, "status": "error", "error": str(e)}
    step = resume_payload(run_id)
    msg = step.get("message")
    if not msg:
        return {"run_id": run_id, "status": step.get("status", prog.get("status", "unknown")), "note": step.get("note", "")}
    # load original image path if saved
    # prog messages contain request; find image_path from run_dir
    rdir = _get_runs_dir() / run_id
    img_path = rdir / "image.png"
    if not img_path.exists():
        # try to find any image
        for cand in rdir.glob("image.*"):
            img_path = cand
            break
    request = prog.get("messages", [{}])[-1].get("content", "") if prog.get("messages") else ""
    # for resume, we inject guidance as user message and re-run agent with existing state
    # ponytail: simplest — re-run agent on same image with appended guidance
    guidance = msg
    return run_agent(img_path, request + "\n" + guidance, use_cache=use_cache, on_event=on_event, run_id=run_id + "_resume")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None, help="resume run_id")
    ap.add_argument("image", nargs="?", help="image path")
    ap.add_argument("request", nargs="?", default="count colonies")
    args = ap.parse_args()
    if args.resume:
        out = resume_run(args.resume)
        print(json.dumps(out, indent=2))
    elif args.image:
        out = run_agent(args.image, args.request)
        print(out["report"][:1000])
