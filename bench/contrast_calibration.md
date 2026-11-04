# Contrast calibration (Lab dE inside vs 10-20px ring)

Guard for POST /api/correct add: accept SAM box/point mask only if mean Lab inside vs ring differs by >= threshold. Ring excludes other masks.

Calibration on AGAR ground-truth boxes (real colonies must pass) vs random empty-agar boxes inside plate (must fail). 2 plates (14684, 13938), 10 GT + 10 empty, via bench/calibrate_contrast.py (real SAM calls).

- GT (n=10): min 7.91 max 35.19 mean 18.59 median 16.58
  sorted: [7.9, 10.6, 12.1, 14.5, 16.1, 17.1, 23.3, 23.5, 25.6, 35.2]
- empty (n=10): min 0.14 max 3.13 mean 1.32 median 0.47
  sorted: [0.1, 0.2, 0.3, 0.3, 0.5, 0.5, 2.3, 2.9, 3.0, 3.1]

Threshold sweep:

| thr | GT-pass | empty-reject |
|-----|---------|--------------|
| 3.0 | 100.0% | 90.0% |
| 5.0 | 100.0% | 100.0% |
| 7.0 | 100.0% | 100.0% |
| 10.0 | 90.0% | 100.0% |

Chosen: CONTRAST_THRESHOLD=5.0 in labvision/config.py (GT 100% pass, empty 100% reject). Smallest GT 7.9 >5, largest empty 3.1 <5 gives margin 2.8.
Synthetic empty 500x500 beige also dE ~0.00 <5 -> reject "No colony visible here".

Implementation: labvision/contrast.py contrast_delta_e() / is_contrast_ok(), server.py POST /api/correct checks delta E after SAM mask decode, returns 400 {"error":"No colony visible here"} if below threshold.
