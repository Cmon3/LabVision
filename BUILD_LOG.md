# Build log

Timestamped record of what was built during the hackathon (Oct 27 – Nov 4, 2026),
so judges can tell original work from background.

| Date | What was built | Notes |
|------|----------------|-------|
| 2026-10-27 | Repo created, README, license, probe script skeleton | Nothing from earlier projects is reused. |
| 2026-10-27 | T3 labvision/measure.py + tests/test_measure.py (4 passed) | Counts/areas/equiv. diameters, px→mm via 90 mm plate, 2 mm filter. |
| 2026-10-27 | T4 labvision/qc.py + tests/test_qc.py (8 passed) | Laplacian-variance blur, saturated-pixel glare, empty/crowded flags, numpy only. |
| 2026-10-27 | T5 bench/datasets.py + tests/test_datasets.py (9 passed) | BBBC005 loader (F1 subset, count from filename) + AGAR sample loader. |
| 2026-10-27 | T7 labvision/runlog.py + tests/test_runlog.py (5 passed) | One JSON + CSV per run (inputs, model ids, params, results). |
| 2026-10-28 | T1 labvision/segment.py (parser verified, no API call yet) | SAM 3.1 client via requests, boxes parsed + raw mask strings, cache by image hash + prompt. |
| 2026-10-28 | T1 fix: tolerant BOX/MASK regex + confidence | Handles `"< \|box"` space and `;c=` conf, verified on live API. |
| 2026-10-28 | T2 scripts/probe_api.py + live SAM call | 1 image: cheetah 762x787 → 1 box, cache hit on rerun, overlays in runs/probe/. Cost $0.0025/image → ~$0.10 for 40-image bench (well under $60 cap). |
| 2026-10-29 | T1b labvision/mask_codec.py + segment decoded_masks() | Port @meta-sam/parser one_bit range decoder (custom base85). Verified on cached cheetah 683x692→787x762, overlay in runs/probe/cheetah2_mask_overlay.png. Tests 3 passed, total 29. |
| 2026-10-29 | T2 AGAR probe 5 plates (bacterial colony → colonies) | Prompt "bacterial colony" gave 1/0 boxes (undercount); "colonies" gave 17 vs 19 true on 13895 but masks oversized (3M px vs box 15k). Overlays in runs/probe/agar/, measure printed. Blocker: SAM 3.1 not tuned for dense small colonies; will benchmark with "colonies". |
