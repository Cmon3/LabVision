# Long unattended session — Oct 29 evening

Work through the steps below IN ORDER. You are unattended: do not ask questions.
If a step is blocked, write the blocker in BUILD_LOG.md, skip to the next step that
does not depend on it, and keep going.

## Rules for this session
- Read AGENTS.md, PLAN.md, docs/sam_api.md once at the start. Do not re-read the whole repo.
- After EACH step: run `.venv/bin/python -m pytest -q`; if green, add one line to
  BUILD_LOG.md and `git commit` (message starts with the task ID). Push after each commit.
- SAM budget for the whole session: max 400 images (~$1). Always use the cache.
- Never print or commit keys. Never add a dependency that is not in requirements.txt,
  except `gradio` (already listed).
- Runtime models: Meta only (sam-3.1 via Meta API; meta-models/muse-spark-open-1026 via Baseten).
- Keep outputs short. Do not paste large command outputs back into the chat.

## Step 1 — T1b: decode masks
In labvision/segment.py decode one_bit masks: base85 payload -> bytes -> bits -> H x W
array -> resize (nearest) to the image w x h. Try `base64.b85decode` first, then
`base64.a85decode`; keep whichever gives exactly H*W bits (or ceil to bytes) on a real
cached response in runs/cache/. Add a unit test with that real cached line.
Done when: an overlay PNG shows mask pixels on top of the object.

## Step 2 — T2 on real plates
Run scripts/probe_api.py on 5 images from data/agar_demo/ with prompt
"bacterial colony". Save overlays (boxes + masks) to runs/probe/agar/.
Then call measure.py on the masks and print count + diameters per image.
If data/agar_demo/ is empty, skip to Step 3 and log it.

## Step 3 — T6: agent
labvision/agent.py: OpenAI-compatible chat completions with tool calling against
Baseten (env: BASETEN_API_KEY, BASETEN_BASE_URL, BASETEN_MODEL). Tools: qc, segment,
measure, ask_rephoto. The model must answer in a fixed report template whose numbers
are filled ONLY from tool results (fill numbers in Python, not by the model).
Add scripts/run_agent.py: `python scripts/run_agent.py IMAGE "request"`.
Tests: mock the HTTP call; test that the report numbers equal measure() output.

## Step 4 — T8: benchmark
bench/run_bench.py on BBBC005 in-focus images (F1) from data/bbbc005/:
pick 200 images spread across cell counts 1..100 (fixed random seed).
Prompt "cell". Metrics: MAE, mean % error, error vs true count -> bench/results.csv
and bench/RESULTS.md (table + short text). Do not run more than 200 images.

## Step 4b — T8b: colonies benchmark (AGAR sample)
If data/agar_demo/ has annotation files with colony counts or boxes, run the same metrics
on every annotated AGAR sample image with prompt "bacterial colony" ->
bench/results_agar.csv and a second table in bench/RESULTS.md. If there are no usable
annotations, log it and skip.

## Step 5 — T10: Gradio app (first version)
app.py: upload image + text request -> overlay image, results table, report text,
CSV download. Uses agent.py. Launch with `python app.py`. Do not polish design.

## End of session
Write a 5-line summary at the end of BUILD_LOG.md: what is done, what is blocked, cost
of SAM calls this session, next step.
