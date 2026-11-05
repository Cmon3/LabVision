import io
import json
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from fastapi.testclient import TestClient
from server import app

client = TestClient(app)


def _fake_image_bytes(w=100, h=100):
    im = Image.new("RGB", (w, h), (200, 180, 160))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _fake_segment(image_bytes, prompt, *a, **kw):
    from labvision.segment import Box, Segmentation
    boxes = [Box(10, 10, 30, 30), Box(50, 50, 70, 70)]
    raw = []
    return Segmentation(boxes=boxes, raw_masks=raw, width=100, height=100, confidences=[0.9, 0.95])

def _fake_masks(*a, **kw):
    m = np.zeros((2, 100, 100), dtype=bool)
    m[0, 10:30, 10:30] = True
    m[1, 50:70, 50:70] = True
    return m

def test_analyze_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    import labvision.queue as qmod; monkeypatch.setattr(qmod, "RUNS_DIR", tmp_path/"runs")
    import labvision.inbox as imod; monkeypatch.setattr(imod, "RUNS_DIR", tmp_path/"runs")
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count colonies"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["run_id"]
        assert data["status"] in ("OK", "Empty", "TNTC", "Re-photo", "ok", "empty", "tntc", "blurry", "glare")
        assert data["count"] == 2
        assert "boxes" in data
        assert "measurements" in data
        assert "trace" in data
        assert data["trace"][0]["name"] == "check_quality"

def test_analyze_missing_image():
    r = client.post("/api/analyze", data={"request": "count"})
    assert r.status_code == 422

def test_transcribe_mocked():
    fake_wav = b"RIFF....WAVE"
    with patch("labvision.voice.transcribe", return_value="count colonies"):
        r = client.post("/api/transcribe", files={"audio": ("a.wav", fake_wav, "audio/wav")})
        assert r.status_code == 200
        assert r.json()["transcript"] == "count colonies"

def test_examples():
    r = client.get("/api/examples")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_correct_add_remove(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    import labvision.queue as qmod; monkeypatch.setattr(qmod, "RUNS_DIR", tmp_path/"runs")
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
        run_id = r.json()["run_id"]
        with patch("server.segment_point") as mock_point:
            from labvision.segment import Box, Segmentation
            mock_point.return_value = Segmentation(boxes=[Box(80, 80, 90, 90)], raw_masks=[], width=100, height=100, confidences=[0.9])
            mock_point.return_value.decoded_masks = lambda: np.zeros((1, 100, 100), dtype=bool)
            r2 = client.post("/api/correct", json={"run_id": run_id, "x": 85, "y": 85, "mode": "add"})
            assert r2.status_code == 200
            assert r2.json()["count"] == 3
        r3 = client.post("/api/correct", json={"run_id": run_id, "x": 15, "y": 15, "mode": "remove"})
        assert r3.status_code == 200
        assert r3.json()["count"] == 2


def test_analyze_stream_decided_by_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    captured_prompts = []

    def fake_seg2(image_bytes, prompt, *a, **kw):
        captured_prompts.append(prompt)
        from labvision.segment import Box, Segmentation
        return Segmentation(boxes=[Box(10, 10, 20, 20)], raw_masks=[], width=100, height=100, confidences=[0.9])

    def fake_masks2(*a, **kw):
        m = np.zeros((1, 100, 100), dtype=bool)
        m[0, 10:20, 10:20] = True
        return m

    seq = [
        {"choices": [{"message": {"tool_calls": [{"id": "1", "function": {"name": "check_quality", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"id": "2", "function": {"name": "segment", "arguments": "{\"prompt\": \"round spot\"}"}}]}}]},
        {"choices": [{"message": {"tool_calls": [{"id": "3", "function": {"name": "measure", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"content": "All good.", "tool_calls": []}}]},
    ]
    idx = {"i": 0}

    def fake_muse(messages, tools=None, use_cache=True):
        i = idx["i"]
        idx["i"] += 1
        return seq[min(i, len(seq) - 1)]

    with patch("labvision.agent._call_muse", side_effect=fake_muse), \
         patch("server.segment", side_effect=fake_seg2), \
         patch("labvision.segment.Segmentation.decoded_masks", fake_masks2), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze_stream", files={"image": ("test.png", img, "image/png")}, data={"request": "count black colonies"})
        assert r.status_code == 200, r.text
        assert "text/event-stream" in r.headers.get("content-type", "")
        events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
        muse_tool_names = []
        for call in seq:
            for tc in call["choices"][0]["message"].get("tool_calls") or []:
                muse_tool_names.append(tc["function"]["name"])
        model_events = [e for e in events if e.get("decided_by") == "model"]
        allowed = set(muse_tool_names + ["plan", "segment", "report"])
        for ev in model_events:
            assert ev["step"] in allowed or ev["step"].startswith("plan_"), f"unexpected model step {ev['step']} not from Muse"
            if ev["step"] == "segment" and "prompt" in ev.get("values", {}):
                assert ev["values"]["prompt"] == "round spot", "keyword routing detected: prompt should be Muse's round spot, not black round spot"
        plan_msgs = [e["message"] for e in events if str(e.get("step", "")).startswith("plan_")]
        assert plan_msgs, "expected one plan row per Muse turn"
        for m in plan_msgs:
            assert "check_quality" in m or "segment" in m or "measure" in m
        # Muse ends with plain text and never calls submit_report -> paused, no unverified count shown
        assert events[-1]["type"] == "error" and "8 turns" in events[-1]["error"]
        assert not [e for e in events if e.get("type") == "result"]


def test_correct_no_fabricated_mask(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    import labvision.queue as qmod; monkeypatch.setattr(qmod, "RUNS_DIR", tmp_path/"runs")
    def fake_empty(image_bytes, prompt, *a, **kw):
        from labvision.segment import Box, Segmentation
        return Segmentation(boxes=[], raw_masks=[], width=100, height=100, confidences=[])

    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        with patch("labvision.segment.segment", side_effect=fake_empty), patch("server.segment_point", side_effect=lambda *a, **kw: fake_empty(None, None)):
            import labvision.segment as segmod
            def fake_point(*a, **kw):
                from labvision.segment import Box, Segmentation
                return Segmentation(boxes=[], raw_masks=[], width=100, height=100, confidences=[])
            with patch("server.segment_point", side_effect=fake_point):
                r2 = client.post("/api/correct", json={"run_id": run_id, "x": 50, "y": 50, "mode": "add"})
                assert r2.status_code == 502
                assert "SAM found no colony" in r2.json()["detail"]["error"]
                r3 = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
                assert r.json()["count"] == 2

    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
        run_id = r.json()["run_id"]
        r2 = client.post("/api/correct", json={"run_id": run_id, "x": 10, "y": 10, "x1": 10, "y1": 10, "x2": 20, "y2": 20, "mode": "manual"})
        assert r2.status_code == 200
        assert r2.json()["count"] == 3
        assert any("manual" in str(m).lower() for m in r2.json()["measurements"])


def test_add_on_removed_spot_restores_original_detection(tmp_path, monkeypatch):
    """Removing a SAM detection and adding on the same spot is an undo: the original box + mask
    come back without a new SAM call and without the contrast guard (it was already a colony)."""
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        run_id = client.post("/api/analyze", files={"image": ("t.png", _fake_image_bytes(), "image/png")}, data={"request": "count"}).json()["run_id"]
        assert client.post("/api/correct", json={"run_id": run_id, "x": 15, "y": 15, "mode": "remove"}).json()["count"] == 1
        with patch("server.segment_point", side_effect=AssertionError("restore must not call SAM")):
            r = client.post("/api/correct", json={"run_id": run_id, "x1": 12, "y1": 12, "x2": 28, "y2": 28, "mode": "add"})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["count"] == 2 and j["correction"]["action"] == "restore"
        assert {"x1": 10, "y1": 10, "x2": 30, "y2": 30}.items() <= j["boxes"][-1].items()
        assert j["measurements"][-1]["area_px"] == 400  # real 20x20 mask, not a rectangle guess


def test_correction_keeps_size_filter_exclusions(tmp_path, monkeypatch):
    """Boxes excluded by the run's size filter stay out of the count after a correction."""
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        run_id = client.post("/api/analyze", files={"image": ("t.png", _fake_image_bytes(), "image/png")}, data={"request": "count"}).json()["run_id"]
        rdir = tmp_path/"runs"/run_id
        boxes = json.loads((rdir/"boxes.json").read_text())
        boxes[1]["filtered_out"] = True  # as if a size filter excluded the second colony
        (rdir/"boxes.json").write_text(json.dumps(boxes))
        with patch("server.segment_point") as mock_point:
            from labvision.segment import Box, Segmentation
            seg = Segmentation(boxes=[Box(80, 80, 90, 90)], raw_masks=[], width=100, height=100, confidences=[0.9])
            m = np.zeros((1, 100, 100), dtype=bool); m[0, 80:90, 80:90] = True
            seg.decoded_masks = lambda: m
            mock_point.return_value = seg
            with patch("labvision.contrast.is_contrast_ok", return_value=(True, 30.0)):
                r = client.post("/api/correct", json={"run_id": run_id, "x": 85, "y": 85, "mode": "add"})
        assert r.status_code == 200, r.text
        assert r.json()["count"] == 2  # colony 1 + the added one; the excluded colony is not counted
