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
