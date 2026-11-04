# Session 8 — from demo to lab workflow (Tue Nov 3). Direction decided by Simon.
Run AFTER the click-to-correct fix session is committed (both touch server.py and web/).
The user is present: ask only for real decisions. Timebox ~3-4 h. After each step: tests
green -> BUILD_LOG line -> commit -> push. No new dependencies (opencv already has a QR
detector; polling instead of a watcher library).

## Product idea (put this in README "Where it fits")
LabVision is a module, not a monolith:
- Labs WITH automated incubators/imagers: the same agent runs as a service behind an API;
  the device's software sends each plate image and receives count + masks + audit record.
- Small labs WITHOUT automation: any camera (phone on a stand, USB camera, webcam) saves
  photos into an inbox folder; LabVision picks them up, analyses them, and the technician
  only reviews the plates that need a human ("review by exception").

## Step 1 — inbox folder ingest (labvision/inbox.py)
- Poll a folder (default ./inbox, configurable) every 2 s for new .jpg/.png/.tif files.
- Plate ID: read a QR code on the plate photo with cv2.QRCodeDetector; else use the file name.
  Optional sidecar JSON (same name) with {location, sample_type, alert_limit}.
- For each new image: run the agent (non-interactive request "count colonies"), save the run
  as usual, move the image to inbox/processed/. Failures go to inbox/failed/ with the error.
- Start it with `python server.py --inbox ./inbox` (background thread) or `python -m
  labvision.inbox ./inbox`.

## Step 2 — review-by-exception queue (/app/queue)
- GET /api/queue -> list of runs: plate_id, time, location, count, status, needs_review,
  reasons[], reviewed_by, reviewed_at.
- needs_review = any of: QC warn/fail (blur, glare), TNTC, count >= alert_limit (from sidecar
  or config; example values only, clearly labelled as examples), any experimental prompt,
  any SAM retry, low SAM confidence (request include_confidence and flag mean c < 0.5).
- UI: table sorted with "Needs review" first, coloured reason chips, filter (all / needs
  review / approved). Click a row -> opens the existing detail view (correct by click/box).
- "Approve" button (reviewer name field, remembered in localStorage) -> POST
  /api/runs/{id}/approve -> audit log records reviewer + timestamp + final count as a
  HUMAN decision. "Approve all clean plates" button for runs without review reasons.
- Live: the queue refreshes every 3 s so new photos appear while the camera works.

## Step 3 — integration surface
- FastAPI already serves OpenAPI docs at /docs: give every endpoint a summary and example.
- GET /api/export.csv?from=&to= -> one row per run (plate_id, location, time, count,
  status, reviewed_by, reviewed_at, run_id) for LIMS import.
- README section "Integrating with an incubator/imager or LIMS": POST /api/analyze,
  GET /api/runs/{id}, GET /api/export.csv, audit log fields.

## Step 4 — demo simulator (for the video)
scripts/simulate_camera.py: copies N example images from data/agar_demo into ./inbox every
X seconds (some with a sidecar alert_limit) so the queue fills live. Include the dense 518
plate (TNTC) and one deliberately blurred copy (cv2.GaussianBlur) so "needs review" appears.

## Step 5 — landing + README
Landing: add the two deployment modes (module for automated systems / camera + inbox for
small labs) with a simple diagram. README: same + screenshot of the queue. Update
docs/video_script.md: one new 20-second block "From one plate to a whole session": photos
arrive -> queue -> only flagged plates need a human -> approve -> CSV for LIMS.
5-line summary in BUILD_LOG.md.
