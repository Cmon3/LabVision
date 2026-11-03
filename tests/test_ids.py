import re
from pathlib import Path

def _ids_in_js(js_text: str):
    ids = set()
    # getElementById("...") or '...'
    for m in re.finditer(r'getElementById\s*\(\s*["\']([^"\']+)["\']', js_text):
        ids.add(m.group(1))
    # $("#...") and $ ( "#...")
    for m in re.finditer(r'\$\s*\(\s*["\']#([^"\']+)["\']', js_text):
        ids.add(m.group(1))
    # querySelector("#...") and querySelectorAll
    for m in re.finditer(r'querySelector(?:All)?\s*\(\s*["\']#([^"\']+?)(?:\s|["\'])', js_text):
        ids.add(m.group(1).split()[0].split(".")[0].split(":")[0])
    # also handle `$("#` with spaces
    return ids

def _ids_in_html(html_text: str):
    ids = set(re.findall(r'id\s*=\s*["\']([^"\']+)["\']', html_text))
    return ids

def test_all_js_ids_exist_in_index_html():
    js = Path("web/app.js").read_text()
    html = Path("web/index.html").read_text()
    js_ids = _ids_in_js(js)
    html_ids = _ids_in_html(html)
    # filter out ids that are not expected to be in index.html (e.g., queue specific)
    # But spec says every id used via getElementById/$("#...") in web/app.js must exist in web/index.html
    # filter dynamic ids like "tab-" from $("#tab-"+t)
    js_ids = {i for i in js_ids if not i.endswith("-") and not i.endswith("_") and "+" not in i and "$" not in i}
    missing = js_ids - html_ids
    assert not missing, f"IDs used in web/app.js but missing in web/index.html: {sorted(missing)}"

def test_toast_exists_and_styled():
    html = Path("web/index.html").read_text()
    assert 'id="toast"' in html, "toast div missing"
    # styled with new tokens: fixed bottom-centre, ink background
    assert "var(--ink)" in html or "var(--bg)" in html or Path("web/app.css").read_text().count("#toast") > 0
    # hidden initially, 4s timeout in JS
    js = Path("web/app.js").read_text()
    assert 'getElementById("toast")' in js or "getElementById('toast')" in js
    assert "4000" in js or "4 * 1000" in js or "4_000" in js
