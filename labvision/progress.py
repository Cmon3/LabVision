"""Progress per turn (labvision/progress.py) — adapted from legal_agent progress.

A run on disk: runs/<run_id>/progress.json stores messages + status + turn.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import os


def _runs_dir(base: str | Path | None = None) -> Path:
    return Path(base or os.environ.get("LABVISION_RUNS_DIR", "runs"))


def _folder(run_id: str, base: str | Path | None = None) -> Path:
    return _runs_dir(base) / run_id


def new_run(run_id: str, instructions: str, request: str, base: str | Path | None = None) -> dict:
    d = _folder(run_id, base)
    d.mkdir(parents=True, exist_ok=True)
    (d / "instructions.md").write_text(instructions)
    (d / "objective.md").write_text(request)
    prog = {
        "run_id": run_id,
        "started": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "turn": 0,
        "messages": [{"role": "user", "content": request}],
        "turns": [],
    }
    _write(run_id, prog, base)
    return prog


def load_progress(run_id: str, base: str | Path | None = None) -> dict:
    p = _folder(run_id, base) / "progress.json"
    if not p.exists():
        raise FileNotFoundError(f"no such run: {run_id}")
    return json.loads(p.read_text())


def save_progress(run_id: str, messages: list, response: dict | None = None, status: str = "running", base: str | Path | None = None, **notes) -> None:
    try:
        prog = load_progress(run_id, base)
    except FileNotFoundError:
        # first save (should have been new_run)
        prog = {"run_id": run_id, "started": datetime.now(timezone.utc).isoformat(), "status": status, "turn": 0, "messages": [], "turns": []}
    # persist messages/status/turn
    prog["messages"] = messages
    prog["status"] = status
    prog["turn"] += 1
    entry = {"time": datetime.now(timezone.utc).isoformat(), "status": status, "turn": prog["turn"]}
    if response is not None:
        entry["response"] = response
    if notes:
        entry.update(notes)
    prog.setdefault("turns", []).append(entry)
    # also mirror status to status.json
    try:
        from labvision.escalation import set_status
        if status in ("running", "paused", "needs_input", "awaiting_signoff", "done"):
            set_status(run_id, status, base)
    except Exception:
        pass
    _write(run_id, prog, base)


def resume_payload(run_id: str, base: str | Path | None = None) -> dict:
    """Return message to inject on resume, or None if nothing to resume."""
    prog = load_progress(run_id, base)
    status = prog.get("status", "")
    if status == "paused":
        return {"message": "You have more turns. Continue, and finish with submit_report.", "status": "running"}
    from labvision.escalation import get_escalation
    from labvision.queue import _load_json  # type: ignore
    # check signoff status from escalation.json
    esc = get_escalation(run_id, base)
    if status == "awaiting_signoff":
        # check for reviewer decision in escalation or approval
        p = _folder(run_id, base)
        # approval.json is the reviewer decision for queue; for signoff we use review.json
        review_path = p / "review.json"
        if review_path.exists():
            try:
                rev = json.loads(review_path.read_text())
                if rev.get("decision") == "approved":
                    # mark done
                    save_progress(run_id, prog["messages"], status="done", base=base)
                    return {"message": None, "status": "done", "note": "approved"}
                if rev.get("decision") == "rejected":
                    notes = rev.get("notes", "")
                    return {"message": f"The reviewer rejected the report: {notes}. Address the notes and submit a revised report.", "status": "running"}
            except Exception:
                pass
        return {"message": None, "status": status, "note": "waiting for reviewer to approve or reject"}
    if status == "needs_input":
        # needs_rephoto or report_refused
        esc = get_escalation(run_id, base)
        if not esc:
            return {"message": None, "status": status, "note": "no escalation"}
        if esc.get("rule") == "report_refused":
            # check for guidance
            p = _folder(run_id, base)
            guidance = p / "guidance.json"
            if guidance.exists():
                try:
                    g = json.loads(guidance.read_text())
                    return {"message": f"Reviewer guidance: {g.get('notes','')}\nFix the report and submit again.", "status": "running"}
                except Exception:
                    pass
            # also check new photo for needs_rephoto auto-resume is handled in inbox, not here
            return {"message": None, "status": status, "note": "waiting for reviewer guidance"}
        if esc.get("rule") == "needs_rephoto":
            # if new image arrived, inbox would have linked it; check for resume_image link
            p = _folder(run_id, base)
            link = p / "resume_image.json"
            if link.exists():
                try:
                    info = json.loads(link.read_text())
                    return {"message": f"A new photo for plate {esc.get('plate_id')} arrived: {info.get('new_run_id')}. Use it to continue. Re-evaluate quality and measurements.", "status": "running"}
                except Exception:
                    pass
            return {"message": None, "status": status, "note": "waiting for new photo with same plate_id in inbox"}
    return {"message": None, "status": status, "note": f"run is {status}; nothing to resume"}


def _write(run_id: str, prog: dict, base: str | Path | None = None) -> None:
    p = _folder(run_id, base) / "progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog, indent=2))
    tmp.replace(p)
