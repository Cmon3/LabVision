# User stories — LabVision Copilot

| # | As a… | I want… | Done when… | Test |
|---|-------|---------|------------|------|
| 1 | technician | count colonies on a plate from a photo | a count is returned with round masks over each colony | `tests/test_server.py::test_analyze_ok` |
| 2 | technician with gloves | ask by voice | what I say appears in the request box | manual (needs microphone) |
| 3 | technician | filter by size (“larger than 2 mm”) | the count changes and sizes are in mm | `tests/test_measure.py::test_measure_mm_scale_and_filter_2mm` |
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
