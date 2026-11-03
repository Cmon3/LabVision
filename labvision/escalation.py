"""Escalations with named rules (Session 9 Step 2).

Each run can be escalated once. Stored as runs/<run_id>/escalation.json.
Status: running / needs_input / awaiting_signoff / done / paused
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_RULES = {"needs_rephoto", "report_refused", "awaiting_signoff"}


def _runs_dir(base: str | Path | None = None) -> Path:
    import os
    return Path(base or os.environ.get("LABVISION_RUNS_DIR", "runs"))


def _run_dir(run_id: str, base: str | Path | None = None) -> Path:
    return _runs_dir(base) / run_id


def escalate(run_id: str, rule: str, handoff: dict, base: str | Path | None = None) -> dict:
    """Open escalation for run. handoff = {what_blocked, why, what_was_done, request}."""
    if rule not in ALLOWED_RULES:
        raise ValueError(f"unknown rule {rule!r}")
    p = _run_dir(run_id, base)
    p.mkdir(parents=True, exist_ok=True)
    rec = {
        "run_id": run_id,
        "rule": rule,
        "what_blocked": handoff.get("what_blocked", ""),
        "why": handoff.get("why", ""),
        "what_was_done": handoff.get("what_was_done", ""),
        "request": handoff.get("request", ""),
        "at": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_signoff" if rule == "awaiting_signoff" else "needs_input",
    }
    # also carry plate_id for inbox resume
    if "plate_id" in handoff:
        rec["plate_id"] = handoff["plate_id"]
    (p / "escalation.json").write_text(json.dumps(rec, indent=2))
    # also write status.json for queue
    (p / "status.json").write_text(json.dumps({"status": rec["status"], "rule": rule}, indent=2))
    return rec


def get_escalation(run_id: str, base: str | Path | None = None) -> dict | None:
    p = _run_dir(run_id, base) / "escalation.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def clear_escalation(run_id: str, base: str | Path | None = None) -> None:
    for name in ("escalation.json", "status.json"):
        try:
            (_run_dir(run_id, base) / name).unlink()
        except Exception:
            pass


def set_status(run_id: str, status: str, base: str | Path | None = None, rule: str | None = None) -> None:
    """Set run status: running / needs_input / awaiting_signoff / done / paused."""
    p = _run_dir(run_id, base)
    p.mkdir(parents=True, exist_ok=True)
    data = {"status": status, "at": datetime.now(timezone.utc).isoformat()}
    if rule:
        data["rule"] = rule
    (p / "status.json").write_text(json.dumps(data, indent=2))


def get_status(run_id: str, base: str | Path | None = None) -> dict | None:
    p = _run_dir(run_id, base) / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def find_needs_rephoto_for_plate(plate_id: str, base: str | Path | None = None) -> list[str]:
    """Return run_ids with needs_rephoto escalation for this plate_id."""
    out = []
    rd = _runs_dir(base)
    if not rd.exists():
        return out
    for p in rd.iterdir():
        if not p.is_dir():
            continue
        esc = p / "escalation.json"
        if not esc.exists():
            continue
        try:
            rec = json.loads(esc.read_text())
            if rec.get("rule") == "needs_rephoto" and rec.get("plate_id") == plate_id and rec.get("status") == "needs_input":
                out.append(p.name)
        except Exception:
            continue
    return out
