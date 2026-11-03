# Escalation

When a run stops for a person, what the person sees, and how the run resumes. All pre-built: `labvision/escalation.py` opens the requests and `labvision/progress.py` resumes; whether an escalation is allowed is decided by the checks in `VERIFICATION.md`.

## When a run escalates (verification on)

| Trigger | Rule | What unblocks it |
|---|---|---|
| QC fails (blur / glare) | `needs_rephoto` | A new photo of the same plate arrives in the inbox (same QR/plate_id) — auto-resumed |
| The report is refused 3 times | `report_refused` | The reviewer replies with guidance (`POST /api/runs/{id}/guidance` or dashboard) |
| The report passes the report rules | `awaiting_signoff` | The reviewer approves, or rejects with notes (`POST /api/runs/{id}/review` → `approve`/`reject`; `python -m labvision.agent --resume RUN_ID`) |

Every escalation names one plate / report. A run that stops for a person makes no model calls until someone acts.

## Handoff

The queue shows the reviewer the blocked item, the rule that fired, what was done (count, prompt, QC), the one specific request (re-photo, guidance, approve/reject), and the measured results so far. Detail view (`/app?run_id=`) shows the same.

## Resume

- **After needs_rephoto**: inbox watches `./inbox`; when a new image with same `plate_id` (QR or stem) lands, it links `resume_image.json` to the old run. `Resume` button or `python -m labvision.agent --resume RUN_ID` continues.
- **After report_refused**: `POST /api/runs/{id}/guidance {reviewer, notes}` → status `running`; agent gets guidance text.
- **After awaiting_signoff**: `POST /api/runs/{id}/review {decision: approve|reject, reviewer, notes}` → `approve` ends as `done`; `reject` sends notes to agent which revises.
- **After turn limit (`paused`)**: `POST /api/runs/{id}/resume` → agent gets “You have more turns.”

Run status: `running` / `needs_input` / `awaiting_signoff` / `done` / `paused`.
