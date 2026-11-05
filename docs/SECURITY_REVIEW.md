# Security review — 2026-11-03 (read-only, no code changed)

## Status — 2026-11-05 (re-checked in code; findings below kept as written on 11-03)

| # | Finding | Status |
|---|---|---|
| 1 | Path traversal via `run_id` | Fixed — `_validate_run_id` regex + resolved-path check (`server.py`), `labvision/queue.py` |
| 2 | Unbounded uploads | Fixed — images 15 MB + Pillow format check (JPEG/PNG/TIFF), audio 5 MB; batch max 50 files per request |
| 3–5 | `innerHTML` with server data | Fixed — rows and tables built with `textContent`; remaining `innerHTML` uses static strings only |
| 6 | Weak `/api/examples/file` allowlist | Fixed 11-05 — only the resolved-path check under `data/` + image-suffix allowlist |
| 7 | Raw exception text to clients | Fixed — generic errors; planner/SAM errors logged server-side only |
| 8 | Unpinned / heavy deps | Partly — all pinned with `==`; `gradio`/`pandas` remain for the `app.py` fallback UI |
| 9 | Google Fonts without SRI/CSP | Open — no CSP header yet |
| 10–15 | Low / good practices | Unchanged (127.0.0.1 bind, no CORS, no secrets in git) |

Scope: `server.py`, `labvision/inbox.py`, `labvision/queue.py`, `web/*`, `.gitignore`, `requirements.txt`, git history (masked). Did not open `.env`, `data/`, `runs/`, `inbox/`, `notes/`.

## High

1. `server.py:833,907,939,980,995,1016,1071,1079,1087` + `labvision/queue.py:246` — **Path traversal via `run_id`. No validation anywhere.** Pattern is `p = _get_runs_dir() / run_id` then `exists()` / `FileResponse` / `read_text` / `write_text`. `run_id` comes from URL path (`/api/runs/{run_id}/...`) or JSON body (`/api/correct`, `/approve`, `/review`, `/guidance`). `..`, `.`, absolute segments are never rejected. Fix: `if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id): raise 400`; resolve and assert `p.resolve().parent == _get_runs_dir().resolve()` (or `is_relative_to`).
2. `server.py:352,537,791` — **Unbounded upload read, no size or type limit.** `await image.read()` / `await audio.read()` with no `Content-Length` cap, no MIME sniffing, no dimension cap. Attacker can OOM the worker (large TIFF/WAV) or burn SAM/Muse budget. Fix: `app.add_middleware` / dependency rejecting `content-length > 15MB`; after read `if len(data) > 15_000_000: raise 413`; verify with PIL (`Image.open`, check `format in {"JPEG","PNG","TIFF"}` + `w*h` cap) before any external call.
3. `web/queue.html:136-146` + `labvision/inbox.py:22-33,96` — **Stored XSS via QR/sidecar fields.** `plate_id` (QR content or filename stem) and `location` (sidecar JSON) are written to `inbox.json`/`run.json` and rendered with `tr.innerHTML=\`...${r.plate_id}...${r.location}...${r.reasons.map...}\``. A malicious QR/sidecar (`<img onerror=...>`) executes in the technician's browser. Fix: stop using `innerHTML` for rows; build with `textContent`/`createElement`, or escape `&<>"'` on every interpolated field.

## Medium

4. `web/app.js:357,364-370` — **Reflected/server XSS via `innerHTML`.** Measurements row (`tr.innerHTML=\`<td>${r.id...}\``) and colony-types table (`html+=...${t.label}...${t.description}...`) interpolate server JSON. Today `description` is a hardcoded string, but any compromised/mocked SAM/Muse response becomes script execution. Fix: same as (3) — `textContent` per cell.
5. `web/queue.html:189`, `web/app.js:119,122,172,194,318,353` — **Lower-risk `innerHTML` with error/static strings.** `Queue unavailable: ${e.message}` (`queue.html:189`) reflects fetch errors; the rest are static (`No trace`, `No measurements`). Fix: use `textContent` for `e.message`.
6. `server.py:67-80` — **`/api/examples/file` allowlist is weak.** First gate is `"data" not in str(p.resolve())` (substring match — bypassable by e.g. `data`-named dirs), only the second `relative_to(Path("data"))` is sound; no extension allowlist (any file under `data/` is served, `mt` just defaults to `image/png`). Fix: drop the substring check, keep only `try: p.resolve().relative_to(...)`; reject suffixes outside `{.jpg,.jpeg,.png}`; reject `HEAD` if unneeded.
7. `server.py:544-549,799-803,830,905,918,936,1013` — **Raw exception text returned to clients.** `str(e)` / `msg[:400]` in `/api/analyze`, `/api/transcribe`, `/api/queue`, `/api/runs/{id}`, `/approve`, `/review` can leak absolute paths, library versions, key names. Fix: log full `e` server-side, return generic `{"error": "internal error"}` + stable `code`; keep the existing friendly hints, never echo `str(e)`.
8. `requirements.txt:1-11` — **Unpinned + unused/heavy deps, no hashes.** No `==` pins (`requests`, `numpy`, `fastapi`, ...); `gradio` + `pandas` are only used by legacy `app.py:7,9` (the running API `server.py` never imports them) yet pull a large web/JS surface. Fix: pin all deps (`pip freeze` / `pip-compile`) and either remove `gradio,pandas` or move to `requirements-dev.txt` if `app.py` is legacy.
9. `web/index.html:7-8`, `web/queue.html:7-8`, `web/landing.html:7-8` — **External Google Fonts without SRI/CSP.** `fonts.googleapis.com` stylesheet + preconnect on every page; no `integrity=`, no Content-Security-Policy. Privacy ping + single-point availability/script-swap risk. Fix: self-host fonts or add `integrity` + `Content-Security-Policy: default-src 'self'; font-src 'self' https://fonts.gstatic.com; style-src 'self' https://fonts.googleapis.com`.

## Low

10. `server.py` (no `CORSMiddleware` found — good default) — **No CORS config.** Currently same-origin only, which is safe. Fix: keep as-is; if a separate frontend origin is ever needed, add explicit `allow_origins=[...]`, never `"*"`.
11. `server.py:1100-1112` — **Bind address is safe.** Defaults `--host 127.0.0.1:8000`; only binds wider if operator passes `--host 0.0.0.0`. Fix: keep default; document that `0.0.0.0` requires auth/reverse-proxy.
12. `labvision/inbox.py:17,180-184` — **Inbox allowlist good, but no size/MIME/symlink checks.** `EXTS={.jpg,.jpeg,.png,.tif,.tiff}` + dotfile skip is correct; however no max-bytes, no magic-byte check, symlinks followed by `read_bytes()`. Fix: `p.is_symlink() → skip`; `st_size` cap; verify magic bytes via PIL before `_analyze_image`.
13. `labvision/inbox.py:66-69,124-129` — **Filenames safely confined, QR never becomes a path (good).** `plate_id` from `read_plate_id()` is only written into JSON (`run.json`, `inbox.json`, `escalation.json`), never interpolated into a filesystem path; moves use `image_path.name` from `iterdir()` inside `inbox_dir`. Keep this invariant; add a regression test asserting `plate_id` never touches `Path`.
14. Secrets — **no leak found (good).** `git ls-files` tracks only `data/README.md` under sensitive dirs; `.env` untracked; `git log -p` shows only variable *names* (`META_API_KEY` ~40 hits, `BASETEN_API_KEY` ~18 hits, all in docs/errors/examples) and no `sk-*/ghp_*/AKIA*/PRIVATE KEY` values (masked search, values never printed). `.env.example:1-5` contains empty placeholders only. Keep: never `git add .env`; keys only via env (`segment.py:140`, `voice.py:120`, `agent.py:236` — all `os.environ.get`, correct).
15. `.gitignore:1-16` — **Coverage complete for the asked paths.** `.env`, `.venv/`, `data/*` (+ `!data/README.md` intentional), `runs/`, `runs_test_leftovers/`, `inbox/` all ignored; `outputs/` also ignored (extra, harmless). Note: `data/*` ignores contents but not a bare `data/` placeholder — fine as-is.

## Fix list (short)

- [ ] Validate `run_id` with strict regex + `resolve().is_relative_to(RUNS_DIR)` on every `/api/runs/*` and `/api/correct` path.
- [ ] Cap uploads (15 MB, image/audio MIME + pixel/duration caps) before SAM/Muse calls.
- [ ] Replace `innerHTML` row/table rendering in `queue.html` + `app.js` with `textContent`; escape QR/sidecar fields.
- [ ] Harden `/api/examples/file`: keep only `relative_to` check + image-suffix allowlist.
- [ ] Stop returning `str(e)`; return generic errors, log details server-side.
- [ ] Pin `requirements.txt`; drop or split out `gradio`/`pandas` if `app.py` is legacy.
- [ ] Self-host fonts or add SRI + CSP; keep CORS closed and bind on `127.0.0.1`.
