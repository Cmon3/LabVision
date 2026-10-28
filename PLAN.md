# PLAN — LabVision Copilot

Hackathon build window: Oct 27 – Nov 4, 2026 (submit by Nov 4, 21:00 ET = Nov 5, 03:00 Barcelona).
Internal target: **submit Nov 4, 17:00 Barcelona**. Core must work by **Oct 31**; Nov 1–4 is buffer, polish and video.

## Definition of done (what judges see)
1. Upload a plate photo + "count colonies larger than 2 mm" → masks overlay, count, size table, short report, CSV.
2. Agent asks for a re-photo when QC fails (blur, glare, empty).
3. Benchmark table: SAM 3.1 vs a specialist baseline on BBBC005 (+ AGAR sample), error vs ground truth.
4. Public repo, BUILD_LOG.md, 3-minute video, X post.

## Budget ($150 Meta credits)
| Use | Cap |
|-----|-----|
| Coding with Muse Code (muse-spark-1.3) | $40 |
| SAM 3.1 calls (probe + benchmark) | $60 |
| Muse Spark at runtime (agent + reports) | $20 |
| Buffer | $30 |
Set a Spend Limit alert in Billing. Measure the cost of the first SAM call before planning the benchmark size.

## Tasks (one task = one agent session)

### Day 1 — Tue Oct 27 (tonight, short)
- **T0** (you, no agent): create API key, fill `.env`, read SAM 3.1 docs, paste the request/response example into `docs/sam_api.md`.

### Day 2 — Wed Oct 28  (Agents 102 at 17:00)
- **T1** `labvision/segment.py`: SAM 3.1 client from `docs/sam_api.md`; text prompt in, masks + boxes out; cache by image hash + prompt.
- **T2** `scripts/probe_api.py` → call T1 on 5 images, save overlay PNGs to `runs/probe/`. Record cost per call in BUILD_LOG.

### Day 3 — Thu Oct 29
- **T3** `labvision/measure.py`: count, area, equivalent diameter, filter by size; px→mm via plate diameter (default 90 mm). Tests with synthetic masks.
- **T4** `labvision/qc.py`: blur (Laplacian variance), glare (saturated pixels), empty/crowded flags. Tests.
- **T5** `bench/datasets.py`: BBBC005 loader (count from filename, in-focus `F1` subset) + AGAR sample loader.

### Day 4 — Fri Oct 30  (Fireworks session 17:00)
- **T6** `labvision/agent.py`: Muse Spark tool loop. Tools: segment, measure, qc, ask_rephoto. Report template filled only with tool outputs.
- **T7** `labvision/runlog.py`: JSON + CSV per run (inputs, model ids, params, results).

### Day 5 — Sat Oct 31  (core freeze)
- **T8** `bench/run_bench.py`: SAM 3.1 on ~20 BBBC005 + ~20 AGAR images, MAE/% error vs ground truth → `bench/results.csv` + markdown table.
- **T9** baseline: Cellpose-SAM (local, CPU/MPS) on the same images, same metric. Skip if it costs more than 2 h.

### Day 6 — Sun Nov 1
- **T10** `app.py` Gradio: image + request → overlay, table, report, CSV download.
- **T10b** voice input: mic button in Gradio → Muse Voice Transcribe → same text request as typing. Hands-free story (gloves, biosafety cabinet). Before starting, read the Voice Transcribe docs (audio format, model ID) into `docs/voice_api.md`. Cache transcripts. If it fails after 2 h, drop it and keep text input.
- **T11** README: results table, architecture diagram, "built during hackathon" section.

### Day 7 — Mon Nov 2
- **T12** polish: error messages, example images, one-command run. Fix the top 3 bugs only.
- Write the 3-minute video script.

### Day 8 — Tue Nov 3  (last day for questions)
- Record the video. Draft X post. Final README check.

### Day 9 — Wed Nov 4
- Submit by 17:00 Barcelona. Post on X, add link to submission.

## Stretch (only if core is done)
- Muse Spark Open via Baseten as the agent brain (open-weights bonus — confirm eligibility first).
- Folder-watch mode for an imager output folder.
- Hand gestures (thumbs up = accept, thumbs down = re-photo, point = pick colony) via an external library. Only after the benchmark is done; behind a flag. Does not count as Meta tooling.

## Open questions (Discord)
- SAM 3.1 open-weights bonus eligibility; fine-tuning; data terms; third-party baselines OK?
