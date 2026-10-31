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
| 2026-10-29 | T8 rerun AGAR+BBBC w1/w2 | AGAR test 20 with "round spot" tiling1: MAE 17.70→3.05 dev / 17.7 test (vs 46.38 old). BBBC w1 MAE 5.48 w2 18.95; nucleus prompt worse (52.6) kept cell. Cost incremental $0.05. RESULTS.md updated with split tables + findings. |
| 2026-10-29 | T6b labvision/characterize.py + bench/run_species.py | Contact sheet 24 crops/call via Muse Spark Open vision. GT boxes test 20 (400 crops): accuracy 0% (model refuses organism ID, returns unknown). Per-class 0% across 6 species. bench/species.csv + RESULTS.md section. Blocker: Muse safety refusal for presumptive ID; fallback to unknown. |
| 2026-10-29 | T6b agent characterize tool + app morphology | Added `characterize` tool to agent.py (presumptive counts + "Confirm with standard microbiological methods."). App shows morphology table (35 tests). |
| 2026-10-29 | Fix characterize cache+raw+reasoning (T6b) | Cache now sha256(FULL bytes+prompt), never cache failed parses, raw saved to runs/characterize_raw/, reasoning handled. Manual 13895 4-crop sheet raw 300ch: "We need to respond. The user asks for research quantification only (not clinical diagnosis), contact sheet has 4 colony crops (6 cols, row-major 0..3). For each, return JSON with color, shape, margin, elevation..." Reran species test 20→390 crops, acc 0% (Muse fallback unknown). |
| 2026-10-29 | Fix agent TNTC + retry (T6) | Count>300 → "Too numerous to count (>300 CFU): dilute and re-plate". SAM 0 but QC not empty → retry once with AGAR_ALT_PROMPT "colonies" from config.py. |
| 2026-10-29 | Rerun bench + Findings (T8) | Reran AGAR test with new config (cached, no new SAM cost), updated bench/RESULTS.md Findings: TNTC threshold, prompt wording "round spot" best, w1 vs w2 5.48 vs 18.95, tiling, retry. |

---
Summary (Oct 29 long session 2, accuracy+characterization):
- Done: T8c mask clip + tiling NMS + prompt sweep (round spot best 3.05 MAE), QC 6.0 threshold + empty fix, T8 rerun w1/w2 split + AGAR test 17.7 MAE, T6b characterize 400 crops (0% organism due Muse refusal), agent+app morphology.
- Blocked: Muse Spark Open vision refuses presumptive organism ID (safety) → fallback unknown; AGAR 518 dense 298→31 undercount even with best prompt; plate Hough optional not evaluated.
- Cost: ~350 SAM images × $0.0025 = $0.88 + 20 Muse vision calls ~$0.05, total session ~$0.93 (overall ~$1.8 under $4 cap).
- Next: tune dense plate handling, polish app, record video.

| 2026-10-31 | Finding: Muse Spark Open refuses presumptive organism ID | Raw responses in runs/characterize_raw/ show a safety refusal (biological-risk / medical-advice policy). The model accepts morphology (color, shape, margin, elevation, size). Decision: drop species ID; group colonies into morphotypes; evaluate grouping purity against AGAR species labels. |
| 2026-10-31 | S3 Step 1 fixes: default prompt AGAR_PROMPT, plate px via plate.py (no model param), batch all colonies | labvision/agent.py: removed plate_diameter_px from TOOLS, auto-detect plate via HoughCircles; labvision/plate.py detect_plate_diameter_px; labvision/characterize.py batch up to 24; tests/test_session3.py 5 passed (total 40). |
| 2026-10-31 | S3 Step 2 T6b morphotypes: morphology only + Python grouping | labvision/characterize.py: prompt color/shape/margin/elevation/opacity/surface (no organism), _group_morphotypes exact tuple+terciles -> A,B,C with counts/mean mm, purity flag; tests 3 more (total 43). |
| 2026-10-31 | S3 Step 3 bench morphotypes: ARI/purity/flag | bench/run_morphotypes.py GT boxes 351 crops/20 plates: ARI 0.01 purity 37.9% flag 20% (4/20); bench/morphotypes.csv + RESULTS.md Morphotypes section (4-line findings); numpy ARI, no sklearn. |
| 2026-10-31 | S3 Step 4 agent+app Colony types table + purity flag | labvision/agent.py report Colony types (A,B,C desc count mean mm) + flag; app.py aggregated morphotype DataFrame; README removed diagnosis wording; 43 tests. |
| 2026-10-31 | S4 Step 4: tune on dev (grid T×color_weight) + report on test | bench/run_morphotypes_v2.py dev 19 / test 20 plates 351 crops; best T=2.5 cw=2.0 dev bal 69% test flag 80% vs baseline 90%, ARI -0.00 purity 97% contam 0/1; n small noted. |

| 2026-10-31 | S4 Step 5: decision — flag removed (rule) | Test flag 80% (16/20) not > baseline 90% (18/20); contamination recall 0/1 with 3 false alarms ≤9 (half pure) but overall fail; agent/app flag removed, Colony types labeled descriptive only. |
| 2026-10-31 | S4 Step 6: Muse wording per group (constrained vocab) | labvision/describe.py per-group contact sheet (≤6 crops) + feature summary → Muse → validated color/shape/margin/elevation/surface; agent+app use Python grouping + Muse wording; tests/test_describe.py 4 passed (57 total). |

---
Summary (Oct 31 Session 4, measured features + honest grouping):
- Done: honest findings (ARI random, flag 20%<90% baseline), features.py (Lab bg-sub, circularity, log diam, texture), morphotypes.py (fixed global scales, avg linkage T=2.5 cw=2.0, small merge), tuning dev 69%→test 80%/90% baseline, flag removed, Muse per-group wording (constrained vocab).
- Blocked: Python features ARI ~0, flag not beating baseline, contamination recall 0/1; n=20 test plates is small; no new SAM needed.
- Numbers: AGAR MAE 17.7 BBBC 5.48/18.95; v1 morphotypes 37.9% purity ARI 0.006 flag 20%; v2 Python 97% purity ARI -0.00 flag 80% bal 67% vs 50% baseline, contam 0/1; 57 pytest passed.
- Cost: 0 new SAM (cache/GT boxes), ~35 Muse calls earlier + ~0 for v2 (features only) + future per-group Muse ~28 calls; total cum ~$1.9 under caps.
- Next: polish app (descriptive-only table), update README, record video Nov 1-4.

---
Summary (Oct 31 Session 3, morphotypes + fixes):
- Done: S3 Step 0-4: default prompt fix, plate px via HoughCircles, batch all colonies, morphotype grouping (morphology+terciles), bench ARI/purity/flag, agent+app tables (43 tests green).
- Blocked: ARI 0.01 / purity 37.9% — morphology overlap + size terciles over-split pure plates (flag 20%); 5270 plate partial refusal (logged); dense plate 518 undercount unresolved.
- Numbers: AGAR MAE 17.7 (test 20), BBBC w1 5.48 w2 18.95; morphotypes 351 crops/20 plates ARI 0.01 purity 37.9% flag 4/20; 43 pytest passed.
- Cost: 0 new SAM this session (cached), ~35 Muse vision calls ~$0.08 (total ~$1.9 cum under $60+$20 caps, <300 SAM images).
- Next: polish app (overlay toggle, CSV), README results table, video script (Nov 1-4 buffer).
