# User stories — LabVision Copilot

| # | As a… | I want… | Done when… | Test |
|---|-------|---------|------------|------|
| 1 | technician | count colonies on a plate from a photo | a count is returned with round masks over each colony | `tests/test_server.py::test_analyze_ok` |
| 2 | technician with gloves | ask by voice | what I say appears in the request box | manual (needs microphone) |
| 3 | technician | filter by size (“larger than 2 mm”, “smaller than 3 mm”) | the count changes, sizes are in mm, excluded colonies are shown dashed | `tests/test_measure.py::test_measure_mm_scale_and_filter_2mm`, `tests/test_measure.py::test_filter_by_max_diameter_smaller_than`, `tests/test_agent.py::test_smaller_than_filter_applied_and_reported` |
| 4 | technician | know if the photo is usable | a blurry photo gives “Re-photo” and no count | `tests/test_qc.py::test_blur_flag` |
| 5 | technician | not get a false number on saturated plates | >300 colonies, or a dense plate with few detections, goes to review | manual (crowded flag covered by `tests/test_qc.py::test_crowded_flags`, dense undercount has no automated check) |
| 6 | technician | fix a missed colony | drawing a box adds it, count +1 and `decided_by: human` | `tests/test_server.py::test_correct_add_remove` |
| 7 | technician | not invent colonies | a box on empty agar gives an error and count unchanged | `tests/test_server.py::test_correct_no_fabricated_mask` |
| 8 | reviewer | only review the doubtful | the queue puts flagged items first with plain-word reasons | `tests/test_queue.py::test_queue_needs_review_and_sort` |
| 9 | reviewer | approve and leave a record | approve requires a name and is stored with a timestamp | manual (approve action has no negative-path automated check) |
| 10 | lab | send data to the LIMS | the CSV has one row per plate with count, status and reviewer | manual (CSV shape has no automated check) |
| 11 | lab with fixed camera | not upload by hand | a photo dropped in `inbox/` appears in the queue within 10 s with plate_id from QR | `tests/test_inbox.py::test_process_success_and_move` |
| 12 | auditor | see how each number was obtained | the timeline shows model/rule/human steps and a wrong-count report is refused | `tests/test_verification.py::test_verify_wrong_count_refused` |
| 13 | integrator | connect another device | `/docs` documents the API and `/api/analyze` responds | manual (API docs have no automated check) |
| 14 | technician without a camera | analyse a whole folder of plates at once | every JPG/PNG/TIFF in the folder is analysed in the background (one at a time) and appears in the queue; non-images are ignored | `tests/test_batch.py` (upload, limits, worker progress); folder picker itself manual |

## Verification (2026-11-04 via scripts/check_stories.py, real system no mocks)

| story | check run | result | evidence |
|---|---|---|---|
| 1 count | 1 count | PASS | PASS count=56 boxes=56 masks 42425B |
| 2 voice | 2 voice | manual | manual (needs microphone) — no recorded WAV in repo |
| 3 size filter | 3 size filter | PASS | PASS count 56->56 mm=True (mm present) |
| 4 blurry | 4 blurry | PASS | PASS status=Re-photo blur |
| 5 dense | 5 dense | PASS | PASS 518 count=31 status=Review needs_review=True reasons=['possible undercount (dense plate)'] |
| 6 add by box | 6 add by box | PASS | PASS 56->57 via box 1124,307,1252,435 human dE 23.46 |
| 7 add on empty agar | 7 add on empty agar | PASS | PASS empty agar error 400, count 56 unchanged dE check — contrast Lab inside vs 10-20px ring (exclude other masks) thr 5.0, GT 100% pass (n=10 min 7.9) empty 100% reject (n=10 max 3.1) |
| 8-9 queue + approve | 8-9 queue + approve | PASS | PASS queue flagged first, approve 400 without reviewer, 200 with reviewer, timestamp stored |
| 10 CSV | 10 CSV | PASS | PASS CSV header plate_id,location,time,count,status,reviewed_by,reviewed_at,run_id rows=101 |
| 11 inbox | 11 inbox | PASS | PASS inbox 13895 appeared in queue in <10s plate_id=13895_181329 |
| 12 audit | 12 audit | PASS | PASS trace has {'rule'} and verify refused wrong count |
| 13 API docs | 13 API docs | PASS | PASS /docs 200 and /openapi.json 200 |

*Contrast guard: `labvision/contrast.py` Lab deltaE inside vs ring 10-20px (exclude other masks), threshold `CONTRAST_THRESHOLD=5.0` (`labvision/config.py`) calibrated on AGAR GT boxes vs empty-agar boxes (see `bench/contrast_calibration.md`). Empty agar inside real plate (14684) now correctly rejected with “No colony visible here” (400). Synthetic beige 500×500 also dE~0.0 <5 → reject.*
