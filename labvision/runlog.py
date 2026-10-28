"""Per-run logs: one JSON + one CSV per run under runs/.

Stdlib only (json/csv/datetime/pathlib). No third-party dependencies.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_JSON_NAME = "run.json"
RESULTS_CSV_NAME = "results.csv"


def run_dir(base_dir: str | Path, run_id: str) -> Path:
    """Return the directory for one run: <base_dir>/<run_id>."""
    return Path(base_dir) / run_id


def new_run_id(prefix: str = "run", now: datetime | None = None) -> str:
    """Build a sortable, unique-ish run id from the current UTC time."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}_{stamp}"


def save_run(
    base_dir: str | Path,
    run_id: str,
    *,
    inputs: Mapping[str, Any],
    model_ids: Mapping[str, Any],
    params: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Save one run as <run_id>/run.json plus <run_id>/results.csv.

    Returns the (json_path, csv_path) written.
    """
    rows = [dict(r) for r in results]
    directory = run_dir(base_dir, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / RUN_JSON_NAME
    csv_path = directory / RESULTS_CSV_NAME
    record = {
        "run_id": run_id,
        "inputs": dict(inputs),
        "model_ids": dict(model_ids),
        "params": dict(params),
        "results": rows,
    }
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, rows)
    return json_path, csv_path


def load_run(base_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Read a run back: metadata from run.json, rows from results.csv."""
    directory = run_dir(base_dir, run_id)
    record: dict[str, Any] = json.loads(
        (directory / RUN_JSON_NAME).read_text(encoding="utf-8")
    )
    record["results"] = _read_csv(directory / RESULTS_CSV_NAME)
    return record


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write result rows to CSV; empty rows produce an empty file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read result rows back from CSV; a missing/empty file gives []."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]
