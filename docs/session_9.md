# Session 9 — verification layer, modelled on Meta's reference agent (optional; Simon decides)
Reference read: legal-agent-workshop (Meta workshop): agent loop + VERIFICATION.md +
ESCALATION_RULES.md. Run AFTER session 8 is committed. The user is present.
After each step: tests green -> BUILD_LOG line -> commit -> push.

## Patterns to copy (from the reference)
- One tool per turn (parallel_tool_calls=False); plain text is not accepted: "Call exactly
  one tool." Turn limit; on Model API failure the run is saved as "paused" (resumable).
- System instructions live in a markdown file (labvision/system_instructions.md), sent
  unchanged.
- Progress saved every turn (messages + status) so a run can resume.
- Verification runs INSIDE the tools at fixed points, outside the agent's control. A failed
  check REFUSES the step with feedback; after 3 refusals the run escalates to a person.
- Escalations have named rules, stop the run with no further model calls, and show the
  reviewer a handoff (what is blocked, why, what was done, the one specific request).
- Sign-off: a person approves, or rejects with notes that go back to the agent.
- Tests with a scripted fake model (no network), including SAM answers from recorded outputs.

## Step 1 — submit_report tool + verification/verify_report.py
The agent must finish by calling submit_report {summary, count, filters, notes}. Before it is
accepted, verify_report checks (code, not the model):
- count equals the measure tool's count (with the same filters); every number in the summary
  text appears in the measured results;
- QC passed, or the run escalated for a re-photo;
- count within the countable range, else the report must say TNTC / too few;
- prompt used is a benchmarked one, or the report is marked experimental;
- no organism names and no "diagnosis".
Refused -> feedback to the agent; 3 refusals -> escalation "report_refused".

## Step 2 — escalations with named rules (labvision/escalation.py + ESCALATION_RULES.md)
| Trigger | Rule | What unblocks it |
| QC fails (blur/glare) | needs_rephoto | a new photo of the same plate arrives in the inbox (same QR/plate_id) |
| report refused 3 times | report_refused | reviewer guidance text |
| report passes | awaiting_signoff | Approve, or Reject with notes -> agent revises |
Run status: running / needs_input / awaiting_signoff / done / paused. The queue shows status
and the handoff text. Resume: `python -m labvision.agent --resume RUN_ID` and a Resume
button in the dashboard.

## Step 3 — loop hygiene
One tool per turn, "plain text not accepted", turn limit 8, paused on API errors, progress
saved per turn in runs/<id>/progress.json, system instructions in a .md file.

## Step 4 — tests with a scripted fake model
tests/fake_model.py: scripted tool calls + recorded SAM outputs. Tests: report with a wrong
count is refused; 3 refusals escalate; blurry photo -> needs_rephoto -> new photo resumes;
reject with notes -> revised report.

## Step 5 — docs
VERIFICATION.md + ESCALATION_RULES.md in the repo (tables like the reference). README:
"Verification layer" section. Video: show one refused report or the re-photo escalation.
