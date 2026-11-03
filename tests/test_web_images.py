import re
from pathlib import Path
from fastapi.testclient import TestClient
from server import app

def test_every_img_src_returns_200():
    client = TestClient(app)
    html_files = list(Path("web").glob("*.html"))
    assert html_files, "no html files found"
    img_srcs = set()
    for p in html_files:
        text = p.read_text()
        for m in re.finditer(r'<img[^>]+src\s*=\s*["\']([^"\']+)["\']', text, re.IGNORECASE):
            src = m.group(1).strip()
            # ignore external, data URIs, empty, or pure SVG fallbacks that are not served
            if not src or src.startswith("data:") or src.startswith("http"):
                continue
            # only check same-origin static paths (starting with /)
            if src.startswith("/"):
                img_srcs.add(src)
    assert img_srcs, f"no img srcs found in {html_files}"
    for src in sorted(img_srcs):
        r = client.get(src)
        assert r.status_code == 200, f"{src} returned {r.status_code}, expected 200 (from {html_files})"
        # ensure it's an image, not html
        ctype = r.headers.get("content-type", "")
        assert "image" in ctype or "octet-stream" in ctype, f"{src} content-type {ctype} not image"
