# Session 4 — measured colony features + honest grouping (Sat Oct 31)
The user is present: ask only when a real decision is needed. Timebox: ~3 hours.
Rules: read AGENTS.md once. After each step: pytest green -> one BUILD_LOG line -> commit -> push.
Never output organism names or "diagnosis". No new SAM calls needed (use cache / GT boxes).
scipy is available (installed with scikit-image): scipy.cluster.hierarchy is allowed.

## Why
Session 3 morphotypes: ARI 0.006 (random), purity 37.9%, pure/mixed flag 20% while the
trivial baseline "always pure" scores 90% (18/20 test plates are pure). Cause: exact match
on 7 noisy LLM text attributes incl. size -> over-splitting. New approach: Python measures,
Python decides; Muse only writes the description of each group.

## Step 1 — honest findings
Rewrite the "Morphotypes" findings in bench/RESULTS.md: ARI ~0 = random; flag worse than the
"always pure" baseline (90%). Keep the numbers, fix the interpretation.

## Step 2 — labvision/features.py
Per colony (from SAM mask, or for benchmarks from the GT box with Otsu thresholding inside
the box to separate colony from agar):
- color: mean L*, a*, b* (OpenCV Lab) MINUS the plate background median (agar pixels
  outside all boxes) -> robust to lighting
- circularity 4*pi*area/perimeter^2
- log(equivalent diameter)
- texture: std of L* inside the colony
Tests with synthetic images.

## Step 3 — labvision/morphotypes.py
Per plate: standardize features with FIXED global scales (computed once on the dev split,
saved in config.py) — not per plate, otherwise every plate looks diverse.
Agglomerative clustering (average linkage) cut at distance threshold T.
Clusters with < max(3, 10% of colonies) are merged into the nearest cluster.
Default answer is ONE type; split only when the threshold is exceeded.
Purity flag: 1 type -> "Single morphotype (consistent with pure culture)";
>1 -> "N morphotypes (possible mixed culture or contamination): confirm with standard
microbiological methods."

## Step 4 — tune on dev, report on test
Grid over T and feature weights (color weight 1-3x) on the DEV split only. Objective:
balanced accuracy of the pure/mixed flag, plus per-crop ARI. Note: AGAR labels include a
"Contamination" class — report whether plates containing it are flagged as mixed.
Report on TEST once: flag accuracy, balanced accuracy, ARI, purity, contamination recall,
and ALWAYS the "always pure" baseline next to it. Write bench/morphotypes_v2.csv and replace
the Morphotypes section in RESULTS.md. Mention that n is small (20 plates).

## Step 5 — decide (rule, not opinion)
If test flag accuracy > baseline OR contamination plates are flagged with no more false
alarms than half the pure plates: keep the flag in agent + app.
Otherwise: remove the purity flag from agent, app and README; keep a descriptive
"Colony types" table (Python features + Muse wording) labeled "descriptive only".
Log the decision in BUILD_LOG.md.

## Step 6 — Muse wording
For each final group, send Muse ONE contact sheet of up to 6 representative crops + the
Python feature summary; ask for a short description using only these words:
color (white/cream/yellow/orange/green/grey/beige/brown), shape (circular/irregular/
punctiform), margin (entire/undulate/lobate), elevation (flat/raised/convex), surface
(smooth/rough/mucoid). Validate against the vocabulary in Python.

## Step 7 — summary (5 lines in BUILD_LOG.md)
