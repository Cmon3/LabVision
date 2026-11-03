# Session 5 — ship it (Tue Nov 3). The user is present; ask only for real decisions.
Deadline: submit Wed Nov 4. Scope decided by Simon: polish + README + voice input (T10b) + click-to-correct (option A).
After each step: pytest green -> BUILD_LOG line -> commit -> push.

## Step 0 — housekeeping
Commit docs/session_3.md, docs/session_4.md, docs/session_5.md. Make sure .gitignore keeps
data/, runs/, .env, .venv, opencode.json out. Run a secret scan (grep for key prefixes) on
the full history.

## Step 1 — README rewrite (most important; judges read it first)
Replace README.md entirely. It currently claims things we did not build (exemplar prompts,
Cellpose comparison, "no results yet"). Sections, short and factual:
1. One-liner + 1 screenshot of the app (save to docs/img/, generated from a cached AGAR run;
   do not commit dataset images other than this one demo screenshot).
2. "Built during the hackathon": everything in this repo was built Oct 27–Nov 4 2026; link BUILD_LOG.
3. What it does: count colonies / cells with SAM 3.1, measure in Python (mm via plate
   diameter detected with OpenCV), quality checks, TNTC (>300 -> dilute and re-plate),
   self-check retry, descriptive colony-type table, audit log + CSV.
4. Architecture: mermaid diagram (app -> agent (Muse Spark Open via Baseten) -> tools:
   check_quality, segment (SAM 3.1 Meta Model API), measure, characterize) + "numbers come
   from Python, never from the model".
5. Results (from bench/RESULTS.md): AGAR held-out test 20 plates: mean error 11.8% per
   plate, 7/20 exact, prompt "round spot" vs "bacterial colony" (dev MAE 3.05 vs 34.05);
   BBBC005 200 images: w1 MAE 5.48, w2 MAE 18.95. Dev/test split explained in 2 lines.
6. Honest limits (verifiable-agent section): Muse Spark Open declines organism
   identification on safety grounds, so we do not name organisms; the mixed-culture flag
   was removed by a rule fixed in advance (80% vs 90% always-pure baseline); dense plates
   (>300) are reported as TNTC; small n.
7. Meta models used: SAM 3.1 (Meta Model API), Muse Spark Open (open weights, via Baseten).
8. Responsible use: research and QC quantification, not a medical device, no diagnosis.
9. Data: BBBC005 (CC0), AGAR sample (CC BY-NC 2.0, cite Majchrowska et al. 2021, arXiv:2108.01234). No data committed.
10. Setup + run: venv, requirements, .env keys (META_API_KEY, BASETEN_*), `python app.py`,
    `python bench/run_bench.py`.

## Step 2 — app polish (video-ready, 1 hour max)
- gr.Blocks layout: left = image input (upload or webcam) + request + Run; right = tabs:
  "Result" (overlay with a show/hide masks checkbox), "Measurements" (table + CSV download),
  "How it was verified" (the agent's tool calls in order with their key outputs: QC scores,
  prompt used, retry yes/no, plate diameter px, mm/px, TNTC yes/no), "Report".
- gr.Examples with 3 local AGAR images from data/ (loaded only if present; nothing committed):
  one normal plate, one dense plate (518.jpg -> TNTC), one small-count plate.
- Clear title, one-line description, footer "Research/QC quantification. Not a medical device."
- Friendly error messages (missing key, API down).
- Check it runs: `python app.py`, open http://127.0.0.1:7860, run the 3 examples (cached).

## Step 2b — T10b voice input (Muse Voice Transcribe, Meta Model API)
Read docs/voice_api.md (Simon pastes the official docs there). If it is missing, STOP this
step and ask Simon for it; do not guess the API format.
- labvision/voice.py: audio file -> text via Muse Voice Transcribe (META_API_KEY), cached by
  sha256 of the audio bytes.
- app.py: microphone input (gr.Audio, sources=["microphone","upload"]) next to the request
  box; the transcript fills the request box, the user can edit it, then Run.
- Log the model id + transcript in runlog. Tests with a mocked HTTP call.
- README: add Muse Voice Transcribe to "Meta models used"; one line on hands-free use
  (gloves, biosafety cabinet).

## Step 2c — click-to-correct (SAM 3.1 box/point prompts)
Read docs/sam_points_api.md (Simon pastes the official "Box and point prompts" docs). If it
is missing, STOP this step and ask Simon; do not guess the request format.
- In the "Result" tab, clicking the overlay image (gr.Image .select -> evt.index = x, y):
  mode "Add colony": call SAM 3.1 with a point (or small box) prompt at that spot, add the
  returned mask; mode "Remove": drop the detection whose mask/box contains the click.
- Re-run measure on the corrected set (Python), refresh table, report and CSV.
- Log every correction in runlog (action, x, y, mask added/removed) so the audit trail shows
  human edits separately from model output.
- Tests: mocked SAM point call; remove-by-click logic on synthetic boxes.
- README + video: "human in the loop: one click fixes a missed colony".

## Step 3 — final check
pytest; README renders (mermaid ok); no secrets; repo public; tag v1.0-hackathon; push.
5-line summary at the end of BUILD_LOG.md.
