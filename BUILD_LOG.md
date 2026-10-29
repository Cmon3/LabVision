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
| 2026-10-29 | T6 labvision/agent.py + scripts/run_agent.py | Baseten Muse Spark Open tool loop (qc/segment/measure/ask_rephoto), report numbers from measure only. Mock tests 2 passed, total 31. Live test on 13895: blur false-positive (73<100) + empty misflag; needs QC tuning. |
| 2026-10-29 | T8 bench/run_bench.py 200 BBBC005 F1 + 40 AGAR | BBBC005 prompt "cell" MAE 12.14 MPE 34.1% (n=200, 1200 F1 total, 8-9 per count, seed 0). AGAR prompt "bacterial colony" MAE 46.38 MPE 99.4% (SAM undercounts). TIFF→PNG conversion added. Cost ~$0.61 for 240 images. Results in bench/results.csv + RESULTS.md. |
| 2026-10-29 | T10 app.py Gradio first version | Upload + request → overlay (masks+boxes), table, report, CSV. Uses agent.py. Launch `python app.py` verified (http 7860). Unpolished per spec. |
| 2026-10-29 | T8c labvision/segment clipping + tiling/plate + config | Mask→box clip ratio 2.0 (test), tiling NMS IoU0.5 (1/2/3), plate Hough optional. Prompt sweep dev 19: "round spot" MAE 3.05 best vs "bacterial colony" 34.05. Config saved labvision/config.py, sweep bench/agar_sweep.csv (32 tests). |
| 2026-10-29 | QC calibration blur threshold + empty fix | New DEFAULT 6.0 (<5th pct dark 6.9, BBBC 10.0); per-res map in qc.py. Fixed agent empty false flag (now uses final count). Tests 35 passed. Calibration bench/qc_calibration.md. |

---
Summary (Oct 29 long session, unattended):
- Done: T1b decode masks, T2 AGAR probe+measure, T6 agent, T8 200 BBBC005 + 40 AGAR bench, T10 Gradio app (all pytest 31 green).
- Blocked: AGAR SAM "bacterial colony" undercounts (99% err); QC blur threshold 100 too sensitive (73 false blur); agent empty misflag. Use "colonies" prompt and tune QC next.
- Cost this session: ~240 SAM calls × $0.0025 = $0.60 + Muse Spark ~$0.01, well under $1 cap (total ~$0.9 overall).
- Next: polish app, fix QC threshold, improve AGAR prompt, record video.
