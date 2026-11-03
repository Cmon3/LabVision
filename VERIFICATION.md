# Verification layer

On for runs started with `VERIFICATION=on` (default). Every check is one file in `verification/` with one entry point, `run_verification(ctx, report) -> {"passed": bool, "feedback": [...]}`. The tools in `labvision/agent.py` call them at fixed points, so the agent cannot skip them.

| When | File | Check | On failure |
|---|---|---|---|
| `submit_report` | `verification/verify_report.py` | Count equals measure count (with filters); every number in summary appears in measured results; QC passed or escalated for re-photo; count in 30-300 or report says TNTC / too few; prompt is benchmarked (`round spot`/`colonies`/`cell`) or report marked experimental; no organism names, no diagnosis | Refused with feedback. After 3 tries escalated `report_refused` |
| `ask_rephoto` | (inside tool) | QC is blurry/glare → escalate `needs_rephoto` | Stops as `needs_input` until new photo |

Also: 3 refused reports → escalated to a person; a passed report → `awaiting_signoff` (attorney sign-off); `flag` not needed; always at most one report per run. The run’s progress is saved every turn in `runs/<id>/progress.json`.

## What was built (Session 9)

```python
# verification/verify_report.py
def run_verification(ctx, report: dict) -> dict:
    """{"passed": bool, "feedback": [problem, ...]}"""
```

Checks are code, not the model. Until a check is built, a run with verification on stops at it. Tests use a scripted fake model (no network) with recorded SAM outputs.

## On work already done

`python -m labvision.agent --resume RUN_ID` continues a run; `GET /api/runs/{id}/progress` shows its messages.
