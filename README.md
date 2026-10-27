# LabVision Copilot

An agent built on Meta's SAM 3.1 and Muse models that turns lab images plus a plain-language request into quantified, verifiable results, choosing the best tool for each task.

> **Status: work in progress.** Started on 2026-10-27 as a solo entry (Indie track) in the Meta Global AI Developer Hackathon. Everything in this repository is created during the hackathon; see [BUILD_LOG.md](BUILD_LOG.md) for a dated record of what was built and when.

## The idea

A user uploads a photo (for example, an agar plate) and asks something like *"count colonies larger than 2 mm"*. The agent:

1. Segments the objects with SAM 3.1 from a text prompt (with exemplar prompts and one-click mask correction for refinement).
2. Computes counts, areas and diameters with **deterministic Python code**. No number comes from the LLM.
3. Uses a Muse model to write a lab-style report and flag quality problems (blur, overlapping objects, low confidence), asking for a new photo when needed.
4. Logs every decision, parameter and model version so each result is reproducible and exports to CSV.

## Benchmark (core technical contribution)

Planned evaluation of SAM 3.1 against open specialist models (Cellpose-SAM, µSAM) on public datasets with ground truth:

- **AGAR** (colonies on agar plates)
- **BBBC005 / BBBC004** (synthetic cells with increasing blur and overlap)

Metric: error versus manual counts, per dataset and per condition. The agent routes each task to the best-suited tool, and the results (including where each model fails) will be reported here. *No results yet.*

## Responsible use

This is a research quantification tool with human review. It does not provide diagnoses or medical advice. Do not use it for clinical decisions.

## Data and licenses

Only public or synthetic data is used. See [data/README.md](data/README.md) for sources and license checks. No data is committed to this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your own key; never commit it
```

Connectivity check (needs your own API key and model name from the Meta Model API docs):

```bash
python scripts/probe_api.py path/to/image.jpg
```

## License

MIT. See [LICENSE](LICENSE). Third-party models and datasets keep their own licenses.
