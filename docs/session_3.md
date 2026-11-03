# Session 3 — morphotypes + fixes (Sat Oct 31). The user is present: ask only when a real decision is needed (things this file already decides are not questions).

Rules: read AGENTS.md once. After each step: pytest green -> one BUILD_LOG line -> commit -> push.
Runtime models Meta only. Never output organism/species names or the word "diagnosis".
Budget: max 300 SAM images, batch Muse calls.

## Step 0 — commit pending edits
BUILD_LOG.md and PLAN.md were edited by hand (species-refusal finding, T6b revised). Commit them.

## Step 1 — fixes from code review
a) agent.py: default segment prompt = config.AGAR_PROMPT ("round spot"), not "bacterial colony".
b) Plate diameter in px must come from Python, never from the model: use plate.py
   (HoughCircles) inside the measure tool; if no plate is found, report sizes in px and say so.
   Remove plate_diameter_px from the tool's model-facing parameters.
c) characterize must cover ALL detected colonies (batch contact sheets of up to 24 crops),
   not only the first 12.
d) Tests for a)-c).

## Step 2 — T6b morphotypes
characterize.py: ask Muse Spark Open ONLY for morphology per crop: color, shape
(circular/irregular/punctiform/filamentous), margin (entire/undulate/lobate/filiform),
elevation (flat/raised/convex/umbonate), opacity, surface (smooth/rough/mucoid).
Remove presumptive_organism from the prompt and the schema.
Then group colonies into morphotypes in PYTHON (not by the model): identical or near-identical
morphology tuples -> same type; also use measured diameter (small/medium/large terciles).
Output per plate: list of types with label (A, B, C...), description, count, mean diameter mm.
Purity flag: "Single morphotype (consistent with pure culture)" or
"N morphotypes detected (possible mixed culture or contamination): confirm with standard
microbiological methods."

## Step 3 — benchmark grouping purity
bench/run_morphotypes.py on the AGAR test split using ground-truth boxes (max 400 crops).
Metrics: cluster purity (for each morphotype, share of its colonies from the majority
species) and adjusted Rand index (sklearn not allowed -> implement ARI in numpy) vs species labels.
Also: plates with 1 species vs >1 species -> does the purity flag match? (accuracy).
Write bench/morphotypes.csv and a "Morphotypes" section in bench/RESULTS.md with 4-line findings.
If Muse still refuses morphology, log raw responses and stop this step.

## Step 4 — agent + app
Report section "Colony types" (table) + purity flag. App shows the same table.
Remove any species/presumptive wording from code, UI and README.

## Step 5 — summary
5 lines at the end of BUILD_LOG.md: done, blocked, numbers, cost, next.
