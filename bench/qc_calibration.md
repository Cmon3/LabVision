# QC calibration (Oct 29)

Laplacian variance (sharpness) measured on all acceptable photos:

- AGAR lower-resolution (n=10): min 52.0, p5 52.5, median 121.3
- AGAR higher-resolution/bright (n=10): min 13.4, p5 15.7, median 49.0
- AGAR higher-resolution/dark (n=10): min 6.4, p5 6.9, median 18.9
- AGAR higher-resolution/vague (n=10): min 46.2, p5 47.0, median 59.6
- BBBC005 F1 sample 50: p5 10.0, median 11.3, min 9.9

Threshold set below 5th percentile per resolution to avoid false blur on acceptable photos:
- DEFAULT 6.0 (below dark p5 6.9 and BBBC p5 10.0) — ensures all 39 AGAR + 1200 BBBC F1 not flagged as blurry.
- Per-resolution map in `labvision/qc.py:BLUR_THRESHOLDS` for future per-dataset tuning.

Empty flag fixed: was using box count before measure; now uses final count after measure, so "empty" only when final count 0.
