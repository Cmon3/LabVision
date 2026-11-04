"""POST /api/batch: many plates -> same fixed pipeline as the camera inbox, background thread."""
import io
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image


def _png(color=(200, 180, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(buf, format="PNG")
    return buf.getvalue()


def test_batch_accepts_valid_rejects_invalid_and_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path / "runs"))
    import server as srv

    processed = []

    def fake_process_one(inbox_dir, image_path):
        processed.append(image_path.name)
        return {"run_id": "x"}

    with patch("labvision.inbox.process_one", side_effect=fake_process_one):
        client = TestClient(srv.app)
        files = [
            ("images", ("../../evil name.png", _png(), "image/png")),  # path + spaces sanitised
            ("images", ("plate2.png", _png((10, 10, 10)), "image/png")),
            ("images", ("notes.txt", b"hello, not an image", "text/plain")),
        ]
        r = client.post("/api/batch", files=files)
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["accepted"] == ["evil_name.png", "plate2.png"]
        assert [x["name"] for x in j["rejected"]] == ["notes.txt"]
        for _ in range(50):
            if len(processed) == 2:
                break
            time.sleep(0.05)
        assert processed == ["evil_name.png", "plate2.png"]


def test_batch_limit(tmp_path, monkeypatch):
    import server as srv

    client = TestClient(srv.app)
    files = [("images", (f"p{i}.png", b"x", "image/png")) for i in range(srv.BATCH_MAX_FILES + 1)]
    r = client.post("/api/batch", files=files)
    assert r.status_code == 413
