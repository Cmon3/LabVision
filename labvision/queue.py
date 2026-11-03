"""Queue helpers — needs_review logic."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

RUNS_DIR = Path("runs")

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

def get_queue_items(runs_dir: Path = RUNS_DIR) -> list[dict]:
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
        # try to get count from params if results empty but trace had count? fallback
        # status
        status = rec.get("params", {}).get("status") or rec.get("inputs", {}).get("status") or "ok"
        # derive status from count if not stored
        if count > 300:
            status = "tntc"
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
        # confidence: check boxes.json for confidence? not stored; fallback
        # try to get low confidence flag from params
        low_conf = rec.get("params", {}).get("low_confidence", False)

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
            reasons.append("low confidence")

        needs_review = len(reasons) > 0

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
            "image_url": f"/api/runs/{run_id}/image.png",
            "masks_url": f"/api/runs/{run_id}/masks.png",
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

def approve_run(run_id: str, reviewer: str, runs_dir: Path = RUNS_DIR) -> dict:
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
