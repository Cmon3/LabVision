"""Queue helpers — needs_review logic."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

import os
def _get_runs_dir() -> Path:
    return Path(os.environ.get("LABVISION_RUNS_DIR", "runs"))
RUNS_DIR = _get_runs_dir()

# example alert limit (clearly labelled as example in UI/docs)
ALERT_LIMIT_EXAMPLE = 100

def _parse_run_time(run_id: str, mtime: float | None = None) -> str:
    # run_id like run_20261103T120633 or run_20261103T120633_1
    try:
        # extract timestamp part
        base = run_id.split("_", 1)[1] if "_" in run_id else run_id
        # handle suffix _1
        base = base.split("_")[0]
        dt = datetime.strptime(base, "%Y%m%dT%H%M%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        if mtime:
            return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        return ""

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def get_queue_items(runs_dir: Path | None = None) -> list[dict]:
    if runs_dir is None:
        runs_dir = _get_runs_dir()
    else:
        runs_dir = Path(runs_dir)
    items = []
    if not runs_dir.exists():
        return items
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        run_id = p.name
        # skip cache and non-run dirs
        if run_id in ("cache", "probe", "characterize_raw"):
            continue
        run_json = p / "run.json"
        if not run_json.exists():
            continue
        rec = _load_json(run_json)
        inbox = _load_json(p / "inbox.json")
        approval = _load_json(p / "approval.json")
        # count from results
        results = rec.get("results", [])
        count = len(results)
        # plate_id
        plate_id = inbox.get("plate_id") or rec.get("inputs", {}).get("plate_id") or rec.get("params", {}).get("plate_id") or run_id
        # location
        location = inbox.get("location") or rec.get("inputs", {}).get("location") or ""
        # alert_limit
        alert_limit = inbox.get("alert_limit")
        if alert_limit is None:
            alert_limit = rec.get("inputs", {}).get("alert_limit")
        # time
        try:
            mtime = run_json.stat().st_mtime
        except Exception:
            mtime = None
        time_iso = _parse_run_time(run_id, mtime)
        # experimental, retry, qc
        experimental = rec.get("params", {}).get("experimental", False)
        # also check trace file if exists
        trace = []
        # try to load trace.json if saved
        trace_path = p / "trace.json"
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text())
            except Exception:
                trace = []
        # fallback: check rec params for retry/qc
        retry = rec.get("params", {}).get("retry", False)
        is_blurry = rec.get("params", {}).get("is_blurry")
        glare = rec.get("params", {}).get("glare")
        # if trace has entries, extract
        if trace:
            for t in trace:
                if t.get("name") == "segment" and t.get("retry"):
                    retry = True
                if t.get("name") == "check_quality":
                    if t.get("is_blurry") is not None:
                        is_blurry = t.get("is_blurry")
                    if t.get("glare") is not None:
                        glare = t.get("glare")
                    if "experimental" in t:
                        experimental = t["experimental"]
        # confidence: from params or trace
        mean_conf = rec.get("params", {}).get("mean_confidence")
        low_conf = rec.get("params", {}).get("low_confidence", False)
        if trace:
            for t in trace:
                if t.get("name") == "segment":
                    if t.get("mean_confidence") is not None:
                        mean_conf = t.get("mean_confidence")
                    if t.get("low_confidence") is not None:
                        low_conf = t.get("low_confidence")
        # fallback compute low_conf from mean_conf if not stored
        if mean_conf is not None and not low_conf:
            try:
                if float(mean_conf) < 0.5:
                    low_conf = True
            except Exception:
                pass

        # check escalation/status.json first (named rules)
        esc = _load_json(p / "escalation.json")
        status_file = _load_json(p / "status.json")
        # status override from escalation
        esc_status = None
        esc_rule = None
        if esc:
            esc_rule = esc.get("rule")
            if esc_rule == "needs_rephoto":
                esc_status = "needs_input"
                status = "Re-photo"
            elif esc_rule == "report_refused":
                esc_status = "needs_input"
                # keep computed status for display but queue status is needs_input
                if is_blurry or glare:
                    status = "Re-photo"
                elif count > 300:
                    status = "TNTC"
                elif count == 0:
                    status = "Empty"
                else:
                    status = esc.get("status", "needs_input")
            elif esc_rule == "awaiting_signoff":
                esc_status = "awaiting_signoff"
                status = "awaiting_signoff"
            if esc_status:
                pass
            else:
                esc_status = esc.get("status")
        elif status_file:
            sf_status = status_file.get("status")
            if sf_status in ("paused", "needs_input", "awaiting_signoff", "done", "running"):
                if sf_status == "paused":
                    status = "paused"
                    esc_status = "paused"
                elif sf_status == "done":
                    status = "done"
                    esc_status = "done"
        if not esc_status:
            # status must agree with reasons: blurry/glare -> Re-photo, TNTC -> TNTC, empty -> Empty, else OK
            if is_blurry or glare:
                status = "Re-photo"
            elif count > 300:
                status = "TNTC"
            elif count == 0:
                status = "Empty"
            else:
                status = "OK"

        reasons = []
        if is_blurry:
            reasons.append("blurry")
        if glare:
            reasons.append("glare")
        if count == 0:
            reasons.append("empty")
        if count > 300:
            reasons.append("TNTC")
        # alert limit
        if alert_limit is not None:
            try:
                if count >= int(alert_limit):
                    reasons.append(f"alert ≥{alert_limit}")
            except Exception:
                pass
        if experimental:
            reasons.append("experimental")
        if retry:
            reasons.append("retry")
        if low_conf:
            reasons.append(f"low conf {mean_conf:.2f}" if mean_conf is not None else "low confidence")
        # sanity guards
        if rec.get("params", {}).get("plate_not_found") or any(t.get("name") == "plate_check" and "plate not found" in str(t.get("reason", "")) for t in trace):
            reasons.append("plate not found")
        # outside plate only if not near rim (far outside)
        _flagged = rec.get("params", {}).get("outside_plate_flagged", 0)
        _discarded = rec.get("params", {}).get("outside_plate_discarded", 0)
        _near = rec.get("params", {}).get("outside_plate_near_rim", 0)
        # fallback: compute flagged as discarded - near_rim if not stored
        if _flagged == 0 and _discarded:
            _flagged = max(0, _discarded - rec.get("params", {}).get("outside_plate_near_rim", 0))
        if _flagged > 0 or any(t.get("name") == "plate_filter" and (t.get("discarded", 0) - t.get("near_rim", 0) > 0) for t in trace):
            reasons.append("outside plate")
        if rec.get("params", {}).get("duplicates_removed", 0) > 0 or any(t.get("name") == "dedup" and t.get("removed", 0) > 0 for t in trace):
            reasons.append("duplicate")
        if rec.get("params", {}).get("implausible_size_count", 0) > 0 or any(t.get("name") == "size_check" for t in trace):
            reasons.append("implausible size")
        if rec.get("params", {}).get("possible_undercount") or any(t.get("name") == "dense_check" for t in trace):
            reasons.append("possible undercount (dense plate)")

        needs_review = len(reasons) > 0
        # dense never OK
        if any("possible undercount" in r for r in reasons) and status == "OK":
            status = "Review"

        # handoff for escalation display
        handoff = {}
        if esc:
            handoff = {k: esc.get(k) for k in ("rule", "what_blocked", "why", "what_was_done", "request", "plate_id")}
            # also include reasons from escalation
            if esc_rule == "needs_rephoto" and "blurry" not in reasons and "glare" not in reasons:
                if "needs_rephoto" not in reasons:
                    reasons.append("needs_rephoto")
            if esc_rule == "report_refused":
                reasons.append("report_refused")
            if esc_rule == "awaiting_signoff":
                reasons.append("awaiting_signoff")
                needs_review = True
            if esc_status in ("needs_input", "awaiting_signoff", "paused"):
                needs_review = True

        items.append({
            "run_id": run_id,
            "plate_id": plate_id,
            "time": time_iso,
            "location": location,
            "count": count,
            "status": status,
            "needs_review": needs_review,
            "reasons": reasons,
            "reviewed_by": approval.get("reviewed_by"),
            "reviewed_at": approval.get("reviewed_at"),
            "alert_limit": alert_limit,
            "experimental": experimental,
            "mean_confidence": mean_conf,
            "low_confidence": low_conf,
            "image_url": f"/api/runs/{run_id}/image.png",
            "masks_url": f"/api/runs/{run_id}/masks.png",
            "escalation": esc if esc else None,
            "handoff": handoff if handoff else None,
            "queue_status": esc_status or status,
        })
    # sort: needs_review first, then time desc
    items.sort(key=lambda x: (not x["needs_review"], x["time"]), reverse=False)
    # actually needs_review first means True first: sort with 0 for needs_review True
    # our key is (not needs_review, time) -> False (0) first for needs_review True
    # but time we want newest first within group: reverse time
    # So we should sort with two-stage: group, then time descending
    # To get time descending, we can sort by time reverse after grouping
    # Simpler: do stable sort
    # First sort by time desc
    items.sort(key=lambda x: x["time"] or "", reverse=True)
    # Then stable sort by needs_review (True first)
    items.sort(key=lambda x: 0 if x["needs_review"] and not x["reviewed_by"] else (1 if not x["reviewed_by"] else 2))
    # Actually we want: Needs review first, then clean, then approved last? Spec says table sorted with Needs review first, but filter allows all/needs review/approved.
    # We'll keep needs_review first, then others by time.
    return items

def approve_run(run_id: str, reviewer: str, runs_dir: Path | None = None) -> dict:
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
        raise ValueError("invalid run_id")
    runs_dir_resolved = Path(runs_dir).resolve() if runs_dir else _get_runs_dir().resolve()
    p_check = (runs_dir_resolved / run_id).resolve()
    try:
        if not p_check.is_relative_to(runs_dir_resolved):
            raise ValueError("invalid run_id")
    except AttributeError:
        try:
            p_check.relative_to(runs_dir_resolved)
        except ValueError:
            raise ValueError("invalid run_id")
    if runs_dir is None:
        runs_dir = _get_runs_dir()
    else:
        runs_dir = Path(runs_dir)
    p = runs_dir / run_id
    if not p.exists():
        raise FileNotFoundError(run_id)
    run_json = p / "run.json"
    rec = _load_json(run_json)
    count = len(rec.get("results", []))
    now = datetime.now(timezone.utc).isoformat()
    approval = {
        "run_id": run_id,
        "reviewed_by": reviewer,
        "reviewed_at": now,
        "final_count": count,
        "decided_by": "human",
    }
    (p / "approval.json").write_text(json.dumps(approval, indent=2))
    # also update run.json for audit trail
    try:
        rec.setdefault("audit", []).append({"action": "approve", "reviewer": reviewer, "at": now, "final_count": count, "decided_by": "human"})
        run_json.write_text(json.dumps(rec, indent=2, sort_keys=True))
    except Exception:
        pass
    return approval
