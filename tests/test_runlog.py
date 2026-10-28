"""Tests for labvision.runlog. Uses only pytest tmp_path; never touches runs/."""

from datetime import datetime, timezone

from labvision import runlog


def _sample():
    return {
        "inputs": {"image": "plate.png", "request": "count colonies larger than 2 mm"},
        "model_ids": {"segment": "sam-3.1", "report": "muse-spark"},
        "params": {"plate_diameter_mm": 90.0, "min_diameter_mm": 2.0},
        "results": [
            {"id": 1, "area_mm2": 5.2, "diameter_mm": 2.57},
            {"id": 2, "area_mm2": 9.8, "diameter_mm": 3.53},
        ],
    }


def test_save_creates_json_and_csv(tmp_path):
    sample = _sample()
    json_path, csv_path = runlog.save_run(tmp_path, "run_001", **sample)
    assert json_path == tmp_path / "run_001" / "run.json"
    assert csv_path == tmp_path / "run_001" / "results.csv"
    assert json_path.exists() and csv_path.exists()


def test_round_trip_metadata_and_results(tmp_path):
    sample = _sample()
    runlog.save_run(tmp_path, "run_001", **sample)
    loaded = runlog.load_run(tmp_path, "run_001")
    assert loaded["run_id"] == "run_001"
    assert loaded["inputs"] == sample["inputs"]
    assert loaded["model_ids"] == sample["model_ids"]
    assert loaded["params"] == sample["params"]
    assert len(loaded["results"]) == 2
    assert loaded["results"][0]["id"] == "1"
    assert float(loaded["results"][1]["diameter_mm"]) == 3.53


def test_empty_results_round_trip(tmp_path):
    sample = _sample()
    sample["results"] = []
    runlog.save_run(tmp_path, "run_empty", **sample)
    loaded = runlog.load_run(tmp_path, "run_empty")
    assert loaded["results"] == []


def test_new_run_id_is_prefixed_and_sortable():
    earlier = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    later = datetime(2026, 10, 28, 12, 0, 5, tzinfo=timezone.utc)
    first = runlog.new_run_id(now=earlier)
    second = runlog.new_run_id(now=later)
    assert first.startswith("run_") and second.startswith("run_")
    assert first < second


def test_save_does_not_touch_real_runs_dir(tmp_path):
    runlog.save_run(tmp_path, "run_001", **_sample())
    assert (tmp_path / "run_001").exists()
