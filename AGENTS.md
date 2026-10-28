# AGENTS.md — rules for coding agents in this repo

Read this file and PLAN.md only. Do not scan the whole repo.

## Project
LabVision Copilot: an agent that takes a lab image (plate or cell micrograph) plus a
plain-language request, segments with SAM 3.1 (Meta Model API), measures with
deterministic Python, and writes a short lab-style report with Muse Spark.
Hard rule: **no number in a report may come from the LLM.** Counts, areas and
diameters come only from `labvision/measure.py`.

## Layout (target)
```
labvision/segment.py   SAM 3.1 client + on-disk cache (runs/cache/)
labvision/measure.py   masks -> counts, areas, diameters (px or mm)
labvision/qc.py        image quality flags (blur, glare, empty, too crowded)
labvision/agent.py     Muse Spark tool loop: segment -> measure -> qc -> report
labvision/runlog.py    one JSON + CSV per run (params, model ids, results)
bench/                 dataset loaders, benchmark runner, baselines
app.py                 Gradio UI
tests/                 small pytest tests with tiny synthetic arrays
```

## Stack
Python 3.11+, requests, numpy, opencv-python-headless, scikit-image, pandas,
gradio, pytest. Ask before adding any other dependency.

## How to work (token budget matters)
- One task per session. The task comes from PLAN.md, by its ID (e.g. "do T3").
- Open only the files the task names. Never read `data/`, `runs/`, `.env`, images,
  or large outputs.
- Reply with a short plan (max 5 lines), then the code change. No long explanations.
- Prefer small diffs to rewrites. Keep functions small and typed.
- Do not run benchmarks or long scripts yourself; tell the user the command.
- Never print, log or commit API keys. Read config from env vars only.
- Every external call (SAM, Muse) must be cached by input hash so reruns are free.
- When a task is done: add one line to BUILD_LOG.md (date, task ID, what changed).

## Unknowns — do not guess
The exact SAM 3.1 request/response format must come from the Meta docs
(dev.meta.ai/docs/sam/overview). If the user has not pasted it, stop and ask.
Same for Muse Spark tool-calling format.

## Out of scope
No medical/clinical advice. No patient data. Research quantification only.
