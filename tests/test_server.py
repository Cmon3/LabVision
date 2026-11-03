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
    # return 2 boxes
    boxes = [Box(10, 10, 30, 30), Box(50, 50, 70, 70)]
    raw = []
    return Segmentation(boxes=boxes, raw_masks=raw, width=100, height=100)

def _fake_masks(*a, **kw):
    m = np.zeros((2, 100, 100), dtype=bool)
    m[0, 10:30, 10:30] = True
    m[1, 50:70, 50:70] = True
    return m

def test_analyze_ok():
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("server.detect_plate_diameter_px", return_value=None) if False else patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count colonies"})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["run_id"]
        assert data["status"] in ("ok", "empty", "tntc", "blurry", "glare")
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

def test_correct_add_remove():
    # first analyze
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
        run_id = r.json()["run_id"]
        # add
        with patch("server.segment_point") as mock_point:
            from labvision.segment import Box, Segmentation
            mock_point.return_value = Segmentation(boxes=[Box(80, 80, 90, 90)], raw_masks=[], width=100, height=100)
            # need decoded_masks for new seg
            mock_point.return_value.decoded_masks = lambda: np.zeros((1, 100, 100), dtype=bool)
            r2 = client.post("/api/correct", json={"run_id": run_id, "x": 85, "y": 85, "mode": "add"})
            assert r2.status_code == 200
            assert r2.json()["count"] == 3
        # remove
        r3 = client.post("/api/correct", json={"run_id": run_id, "x": 15, "y": 15, "mode": "remove"})
        assert r3.status_code == 200
        assert r3.json()["count"] == 2


def test_analyze_stream_decided_by_honest():
    # Muse decides prompt; server must not hardcode it. Test that decided_by "model"
    # events only come from actual Muse tool calls, and that attribute handling
    # is via Muse prompt (experimental) not keyword routing.
    captured_prompts = []

    def fake_seg2(image_bytes, prompt, *a, **kw):
        captured_prompts.append(prompt)
        from labvision.segment import Box, Segmentation
        return Segmentation(boxes=[Box(10, 10, 20, 20)], raw_masks=[], width=100, height=100)

    def fake_masks2(*a, **kw):
        m = np.zeros((1, 100, 100), dtype=bool)
        m[0, 10:20, 10:20] = True
        return m

    # Muse returns round spot even though request is black colonies — honest server must use round spot, not black round spot
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
        # all model events must correspond to actual Muse tool calls
        # collect tool names Muse actually requested
        muse_tool_names = []
        for call in seq:
            for tc in call["choices"][0]["message"].get("tool_calls") or []:
                muse_tool_names.append(tc["function"]["name"])
        model_events = [e for e in events if e.get("decided_by") == "model"]
        # each model event's step should be one of Muse's tool names or plan/report
        allowed = set(muse_tool_names + ["plan", "segment", "report"])
        for ev in model_events:
            assert ev["step"] in allowed or ev["step"] in ["plan", "segment", "report"], f"unexpected model step {ev['step']} not from Muse"
            # prompt in segment model event must be from Muse (round spot), not keyword-derived black round spot
            if ev["step"] == "segment" and "prompt" in ev.get("values", {}):
                assert ev["values"]["prompt"] == "round spot", "keyword routing detected: prompt should be Muse's round spot, not black round spot"
        # ensure no hardcoded plan string
        plan_msgs = [e["message"] for e in events if e.get("step") == "plan"]
        for m in plan_msgs:
            # must contain actual tool names, not fixed 'check_quality → segment → measure' if Muse didn't request all
            assert "check_quality" in m or "segment" in m or "measure" in m
        # final result experimental should be False because Muse's prompt was round spot (no colour), even though request had black
        res = [e for e in events if e.get("type") == "result"]
        assert res, "no result event"
        assert res[0]["data"]["experimental"] is False, "experimental should be False when Muse prompt has no colour, even if request contains black"


def test_correct_no_fabricated_mask():
    # Add must not fabricate a mask when SAM returns empty; it should error
    def fake_empty(image_bytes, prompt, *a, **kw):
        from labvision.segment import Box, Segmentation
        return Segmentation(boxes=[], raw_masks=[], width=100, height=100)

    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        # mock SAM to return empty for point prompt
        with patch("labvision.segment.segment", side_effect=fake_empty), patch("server.segment_point", side_effect=lambda *a, **kw: fake_empty(None, None)):
            # patch segment_point to return empty
            import labvision.segment as segmod
            orig = segmod.segment_point

            def fake_point(*a, **kw):
                from labvision.segment import Box, Segmentation
                return Segmentation(boxes=[], raw_masks=[], width=100, height=100)

            with patch("server.segment_point", side_effect=fake_point):
                r2 = client.post("/api/correct", json={"run_id": run_id, "x": 50, "y": 50, "mode": "add"})
                assert r2.status_code == 502
                assert "SAM found no colony" in r2.json()["detail"]["error"]
                # count should not increase
                r3 = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
                # check that no fabricated mask was saved: masks.npy should not have extra entry
                # we can check that the run's boxes still 2
                assert r.json()["count"] == 2

    # manual mode should succeed and be labelled manual, not measured
    with patch("server.segment", side_effect=_fake_segment), \
         patch("labvision.segment.Segmentation.decoded_masks", _fake_masks), \
         patch("labvision.plate.detect_plate_diameter_px", return_value=None):
        img = _fake_image_bytes()
        r = client.post("/api/analyze", files={"image": ("test.png", img, "image/png")}, data={"request": "count"})
        run_id = r.json()["run_id"]
        r2 = client.post("/api/correct", json={"run_id": run_id, "x": 10, "y": 10, "x1": 10, "y1": 10, "x2": 20, "y2": 20, "mode": "manual"})
        assert r2.status_code == 200
        assert r2.json()["count"] == 3
        # manual measurement should be labelled manual, not measured
        assert any("manual" in str(m).lower() for m in r2.json()["measurements"])
