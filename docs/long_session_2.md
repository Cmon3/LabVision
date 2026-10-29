# Long unattended session 2 — accuracy + characterization

Work through the steps IN ORDER. Unattended: do not ask questions. If blocked, log it in
BUILD_LOG.md and continue with the next step that does not depend on it.

## Rules
- Read AGENTS.md, PLAN.md, docs/sam_api.md once. Don't re-read the whole repo.
- After EACH step: `.venv/bin/python -m pytest -q`; if green, one BUILD_LOG.md line, commit
  (message starts with task ID), push.
- Budget for this session: max 1,500 SAM images (~$4) and keep Baseten calls batched.
- Runtime models: Meta only. Never write "diagnosis"/"diagnose" anywhere.
- AGAR data: data/agar_demo/AGAR_representative/ (lower-resolution/, higher-resolution/*/),
  each image has a JSON with colonies_number (-1 = uncountable: EXCLUDE) and labels
  (class, x, y, width, height).
- Avoid overfitting: split AGAR images into dev (first half, sorted by name) and test
  (second half). Choose settings on dev only; report final numbers on test.

## Step 1 — T8c: fix AGAR counting
a) Masks: if a mask's pixel area is much larger than its box area, clip the mask to its box.
   Add a test.
b) Prompt sweep on dev: "colony", "colonies", "bacterial colony", "colonies on agar plate",
   "round spot". One concept per request.
c) Tiling on dev: whole image vs 2x2 vs 3x3 tiles with 15% overlap; merge boxes across
   tiles with NMS (IoU 0.5).
d) Optional: crop to the plate (largest circle, OpenCV HoughCircles) before SAM.
e) Pick the best config by MAE on dev; save it in labvision/config.py; run it on test.
   Write a sweep table to bench/agar_sweep.csv.

## Step 2 — QC calibration
All AGAR and BBBC F1 images are acceptable photos. Set the blur threshold below the
5th percentile of their Laplacian variance (computed per dataset/resolution) and fix the
"empty" false flag. Tests.

## Step 3 — T8 rerun + BBBC split
Rerun bench with the chosen AGAR config (test split) and BBBC005 (same 200 images).
In bench/RESULTS.md report BBBC005 separately for w1 (cell body) and w2 (nuclei); try the
prompt "nucleus" for w2 and keep it if MAE improves. Explain findings in 5 short lines.

## Step 4 — T6b: characterization
labvision/characterize.py: given an image + boxes, crop each colony (pad 20%), build a
contact sheet of up to 24 crops per call, send to meta-models/muse-spark-open-1026 via
Baseten (vision; see Baseten docs for image input format), return JSON per crop:
{color, shape, margin, elevation, size_class, presumptive_organism, confidence}.
presumptive_organism must be one of the classes found in the AGAR JSON files or "unknown".
Benchmark on AGAR test split using ground-truth boxes: accuracy, per-class accuracy,
confusion matrix -> bench/species.csv + a section in bench/RESULTS.md. Max 400 crops.
If Muse Spark Open can't take images, log it and skip to Step 6.

## Step 5 — agent + app
Add tool `characterize` to agent.py. Report gets a section "Colony types (presumptive)"
with counts per type and the line "Confirm with standard microbiological methods."
Show a morphology table in app.py.

## Step 6 — end
5-line summary at the end of BUILD_LOG.md: done, blocked, best numbers, cost, next step.
