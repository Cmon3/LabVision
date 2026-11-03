# Session 7 — guide the user + show the agent working (Tue Nov 3). Decided by Simon.
Run AFTER the fix session (round masks, load_dotenv, [hidden] CSS, footer).
The user is present: ask only for real decisions. After each step: tests green -> BUILD_LOG
line -> commit -> push. Keep app.py (Gradio) untouched as fallback.

## Step 1 — live progress (the agent visibly working)
- Add POST /api/analyze_stream (same inputs as /api/analyze) returning text/event-stream.
  Emit one event per step AS IT HAPPENS, then a final "result" event with the same JSON as
  /api/analyze. Event shape: {type:"step", step, status:"running"|"ok"|"warn"|"fail",
  message (plain English), values{...}, decided_by:"model"|"rule"|"human"}.
- Messages, e.g.: "Planning: Muse Spark Open chose check_quality -> segment -> measure",
  "Checking photo quality... sharp (53.8), no glare", "Looking for colonies with SAM 3.1
  ('round spot')... 56 found", "No colonies on a sharp photo -> retrying with 'colonies'",
  "Plate detected: 1650 px = 90 mm (0.0545 mm/px)", "Measuring 56 colonies...",
  "More than 300 colonies -> too numerous to count: dilute and re-plate", "Writing report".
- decided_by makes the agent honest: tool order and segment prompt/attributes = "model";
  retry, TNTC, QC thresholds = "rule"; click corrections = "human". Show it as a small badge.
- Frontend: read the stream with fetch() + ReadableStream (no EventSource: it's a POST);
  render the steps live in the right panel (spinner -> check / warning icon), then the result.
  Keep /api/analyze for tests and fallback.

## Step 2 — suggested questions
Chips under the request box that fill it in one click:
"count colonies" · "count colonies larger than 2 mm" · "count colonies smaller than 1 mm" ·
"is this plate countable?" · "measure colony sizes" · "count black colonies (experimental)" ·
"count yellow colonies (experimental)".
- Make the agent support colour/attribute requests: the model may pass an attribute to the
  segment tool, which becomes the SAM prompt "<attribute> round spot" (e.g. "black round
  spot"). Mark these results "experimental — not benchmarked" in the UI and report.
- "is this plate countable?" -> QC + count -> answer countable (30-300) / too few (<30,
  low statistical reliability) / TNTC (>300).
- Empty state of the dashboard: 3 numbered hints: 1) pick an example plate or upload,
  2) pick or say a question, 3) Run. Short tooltips on Add colony / Remove / Masks.

## Step 3 — landing page (same deploy)
- Routes: "/" = landing (web/landing.html, same CSS tokens), "/app" = current dashboard.
  Keep "/web/..." static. Header link "Open the app".
- Sections (short, concrete, no hype):
  1. Hero: "AI counts. Technicians decide." + one line: "Photo of an agar plate + a plain
     request -> a colony count you can audit." Buttons: "Try an example plate" (/app?example=14684)
     and "View on GitHub".
  2. The problem: QC, food and water labs count colonies by eye; counts decide batch release.
  3. How it works (3 steps with icons): ask (type or speak) -> the agent plans, SAM 3.1 finds
     colonies, Python measures -> you review, click to fix, export CSV with an audit log.
  4. Results: 11.8% average error on 20 plates it never saw, 7 exact; bar "bacterial colony"
     99% vs "round spot" 11.8%. Link to bench/RESULTS.md.
  5. Built with Meta: SAM 3.1, Muse Spark Open (open weights), Muse Voice Transcribe.
  6. What it won't do: no organism names (model declined, we kept that line); mixed-culture
     flag removed because it did not beat the baseline; >300 colonies = TNTC.
  7. Footer: "Built during the Meta Global AI Developer Hackathon, Oct–Nov 2026." + GitHub.
- /app?example=14684 preloads that example.
- Responsive, light/dark, same fonts and colours as the dashboard.

## Step 4 — README + check
README: screenshot of the landing + one of the live progress; mention "/" and "/app".
Give Simon a 6-point manual check: landing loads, "Try an example" opens the app with the
plate loaded, live steps appear one by one, a chip fills the request, "black colonies"
shows the experimental badge, report still matches the numbers. 5-line summary in BUILD_LOG.
