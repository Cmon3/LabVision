# Session 6 — custom web UI (Tue Nov 3). Decided by Simon: option B.
The user is present: ask only for real decisions. Timebox ~5 h with a checkpoint after Step 2.
After each step: pytest green -> BUILD_LOG line -> commit -> push.

## Safety net
Do NOT delete or break app.py (Gradio). It stays as a fallback for the video until the new
UI passes the manual QA in Step 6. All logic stays in labvision/*; the new UI only calls it.

## Stack (keep it build-free)
- Backend: FastAPI + uvicorn + python-multipart (add to requirements.txt). `server.py`.
- Frontend: plain HTML + CSS + vanilla JS in `web/` (index.html, app.css, app.js), served
  by FastAPI as static files. No Node build step, no CDN dependencies except Google Fonts.
- Run: `python server.py` -> http://127.0.0.1:8000

## Step 1 — API (server.py), thin wrappers around existing functions
- POST /api/analyze (multipart: image, request) -> JSON:
  {run_id, status: "ok"|"tntc"|"blurry"|"glare"|"empty"|"error", count, count_label
  ("56 CFU" or "TNTC >300"), image_w, image_h, image_url, masks_url (transparent PNG layer
  with all masks), boxes[], measurements[] (id, area_mm2/px, diameter_mm/px),
  colony_types[], trace[] (ordered steps: name, ok/warn/fail, key values: blur score, glare
  fraction, prompt used, retry yes/no, plate diameter px, mm/px, model ids, duration ms),
  report (text), csv_url}
- POST /api/correct {run_id, x, y, mode: "add"|"remove"} -> same JSON shape, updated
  (uses segment_point / remove logic already in the repo; logs the human edit).
- POST /api/transcribe (multipart: audio WAV mono 16-bit 24 kHz from the browser)
  -> {transcript}. Uses labvision/voice.py (META_API_KEY).
- GET /api/examples -> list of local example images (14684, 518, 356) if present in data/.
- GET /api/runs/{run_id}/results.csv, GET /api/runs/{run_id}/image.png, .../masks.png
- Errors -> JSON {error, hint} with a human message (missing key, API down, bad image).
- Tests with FastAPI TestClient and mocked labvision calls.

## Step 2 — CHECKPOINT
Run the server, call /api/analyze on example 14684 with curl, paste the JSON (truncated)
in BUILD_LOG. Ask Simon to confirm before building the frontend.

## Step 3 — Frontend (web/)
Design brief (lab instrument, calm, precise; NOT generic AI-purple):
- Colors as CSS variables, light + dark (prefers-color-scheme): background off-white
  #F7F7F4 / dark #111615, ink #18201F, muted #5E6B69, line #DDE2DF,
  accent agar amber #C8892B, teal #1F6F6A for "verified", ok green #2F7D4F,
  warn amber #B7791F, fail red #A94A3C.
- Fonts (Google Fonts): "Bricolage Grotesque" (headings), "Source Sans 3" (body),
  "JetBrains Mono" (numbers, model ids, trace values).
- Layout (desktop: two columns; phone: stacked):
  Header: "LabVision Copilot" + model chips: SAM 3.1 · Muse Spark Open · Muse Voice Transcribe.
  Left: drop zone / file picker + "Use camera" (getUserMedia, capture to canvas), example
    thumbnails (click to load), request input with a big mic button, Run button.
  Right: hero result: big count ("56 CFU") + status pill (OK / TNTC dilute & re-plate /
    blurry: re-photo); image viewer on a <canvas> with the masks layer (toggle) and
    boxes; mode switch "Add colony / Remove" for click-to-correct; tabs:
    "How it was verified" (vertical timeline of trace steps with ok/warn/fail icons and the
    key values in mono), "Measurements" (table + Download CSV), "Colony types"
    (descriptive only), "Report".
  Footer: "Research/QC quantification. Not a medical device." + link to GitHub repo.
- Click-to-correct coordinates: x_img = (clientX - rect.left) * image_w / rect.width,
  same for y; floor and clamp (see docs/sam_points_api.md). Show a small marker where clicked
  and a spinner while SAM runs.
- Mic: record with Web Audio API (AudioContext + AudioWorklet or ScriptProcessor), resample
  to 24 kHz mono, encode 16-bit PCM WAV in JS, POST to /api/transcribe; put the transcript
  in the request box for editing. Do NOT rely on MediaRecorder webm (server has no ffmpeg).
- States: empty (example thumbnails invite a first run), loading (step-by-step progress
  text), error (human message + hint), result.
- Accessibility: keyboard focus visible, buttons with labels, reduced motion respected.

## Step 4 — Demo polish
- First load: example 14684 preselected so the demo starts with one click.
- Smooth transitions only where useful (result appearing, timeline steps ticking in).

## Step 5 — README
Update screenshot (docs/img/demo.png from the new UI), Setup: `python server.py`.
Mention app.py as the minimal Gradio fallback.

## Step 6 — Manual QA with Simon (he does it, you fix)
Give Simon this checklist and wait for his results:
1) example 14684 -> ~56 CFU, timeline shows all steps; 2) example 518 -> TNTC;
3) mic -> transcript in the box; 4) Add colony click lands ON the clicked colony;
5) Remove click drops a detection and the count updates; 6) CSV downloads;
7) camera button works; 8) dark mode readable; 9) phone width (devtools) usable.
Fix what fails. Only then: 5-line summary at the end of BUILD_LOG.md.
