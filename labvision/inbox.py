"""Inbox ingest — poll folder for new plate photos."""
from __future__ import annotations

import json
import shutil
import time
import threading
from pathlib import Path

import os
def _get_runs_dir() -> Path:
    return Path(os.environ.get("LABVISION_RUNS_DIR", "runs"))
RUNS_DIR = _get_runs_dir()
# helper to get current runs dir (respects env changes in tests)
def _runs_dir() -> Path:
    return Path(os.environ.get("LABVISION_RUNS_DIR", "runs"))
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# 2 s stability for large copies (camera simulator uses temp+rename)
STABILITY_S = 2.0
_file_state: dict[str, tuple[int, float, float]] = {}  # path -> (size, mtime, first_seen_mono)

def read_plate_id(image_path: Path) -> str:
    """QR code if present, else file stem."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            d, *_ = cv2.QRCodeDetector().detectAndDecode(img)
            if d and d.strip():
                return d.strip()
    except Exception:
        pass
    return image_path.stem

def read_sidecar(image_path: Path) -> dict:
    cand = image_path.parent / (image_path.stem + ".json")
    if cand.exists():
        try:
            return json.loads(cand.read_text())
        except Exception:
            return {}
    return {}

def _unique_run_id():
    from labvision.runlog import new_run_id
    rid = new_run_id()
    # ponytail: collisions within same second are rare; add suffix if dir exists
    base = rid
    i = 0
    runs = _runs_dir()
    while (runs / rid).exists():
        i += 1
        rid = f"{base}_{i}"
        if i > 10:
            rid = f"{base}_{int(time.time()*1000)%100000}"
            break
    return rid

def process_one(inbox_dir: Path, image_path: Path) -> dict:
    inbox_dir = Path(inbox_dir)
    processed = inbox_dir / "processed"
    failed = inbox_dir / "failed"
    processed.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)

    sidecar_path = inbox_dir / (image_path.stem + ".json")
    has_sidecar = sidecar_path.exists()
    sidecar = read_sidecar(image_path)
    plate_id = read_plate_id(image_path)
    image_bytes = image_path.read_bytes()
    run_id = _unique_run_id()

    try:
        # reuse server analysis (creates runs/<id>/image.png, masks.png, run.json)
        try:
            from server import _analyze_image
        except ImportError as e:
            raise RuntimeError(f"server._analyze_image not available: {e}")
        result = _analyze_image(image_bytes, "count colonies", run_id=run_id)

        # enrich run.json with inbox metadata if present
        try:
            rdir = _runs_dir() / result["run_id"]
            jpath = rdir / "run.json"
            if jpath.exists():
                rec = json.loads(jpath.read_text())
                rec.setdefault("inputs", {})
                rec["inputs"].update({"plate_id": plate_id, "inbox_file": image_path.name})
                for k in ("location", "sample_type", "alert_limit"):
                    if k in sidecar:
                        rec["inputs"][k] = sidecar[k]
                rec.setdefault("params", {})
                rec["params"]["plate_id"] = plate_id
                jpath.write_text(json.dumps(rec, indent=2, sort_keys=True))
            # also stash inbox.json for queue reader
            (rdir / "inbox.json").write_text(json.dumps({"plate_id": plate_id, "inbox_file": image_path.name, **sidecar}, indent=2))
            # update escalation plate_id if QC auto-escalated with run_id placeholder
            try:
                esc_p = rdir / "escalation.json"
                if esc_p.exists():
                    esc = json.loads(esc_p.read_text())
                    if esc.get("plate_id") == result["run_id"]:
                        esc["plate_id"] = plate_id
                        esc_p.write_text(json.dumps(esc, indent=2))
            except Exception:
                pass
            # auto-resume any prior needs_rephoto for same plate_id
            try:
                from labvision.escalation import find_needs_rephoto_for_plate
                prior = find_needs_rephoto_for_plate(plate_id, base=_runs_dir())
                for old_id in prior:
                    if old_id == result["run_id"]:
                        continue
                    old_dir = _runs_dir() / old_id
                    # link new image as resumption
                    (old_dir / "resume_image.json").write_text(json.dumps({"new_run_id": result["run_id"], "at": __import__("time").time()}))
                    # mark old as pending resume -> status still needs_input but will resume automatically if requested
                    # also create guidance that new photo arrived
                    (old_dir / "rephoto_arrived.json").write_text(json.dumps({"new_run_id": result["run_id"], "plate_id": plate_id}))
            except Exception:
                pass
        except Exception:
            pass

        # success -> move to processed
        shutil.move(str(image_path), str(processed / image_path.name))
        if has_sidecar:
            try: shutil.move(str(sidecar_path), str(processed / sidecar_path.name))
            except Exception: pass
        return result
    except Exception as e:
        # failure -> move to failed + error file
        try:
            shutil.move(str(image_path), str(failed / image_path.name))
            if has_sidecar:
                try: shutil.move(str(sidecar_path), str(failed / sidecar_path.name))
                except Exception: pass
            (failed / (image_path.stem + ".error.txt")).write_text(str(e)[:1000])
        except Exception:
            pass
        raise

def _is_stable(path: Path, stability_s: float | None = None, now_mono: float | None = None) -> bool:
    thr = STABILITY_S if stability_s is None else stability_s
    if thr <= 0:
        return True
    try:
        st = path.stat()
        size, mtime = st.st_size, st.st_mtime
    except Exception:
        return False
    key = str(path.resolve())
    now_mono = now_mono if now_mono is not None else time.monotonic()
    prev = _file_state.get(key)
    if prev is None or prev[0] != size or prev[1] != mtime:
        _file_state[key] = (size, mtime, now_mono)
        return False
    _, _, first_seen = prev
    if now_mono - first_seen >= thr:
        # ready — clear state so next poll after move doesn't linger
        _file_state.pop(key, None)
        return True
    return False

def process_inbox_once(inbox_dir: str | Path, stability_s: float | None = None) -> list[dict]:
    inbox_dir = Path(inbox_dir)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    (inbox_dir / "processed").mkdir(exist_ok=True)
    (inbox_dir / "failed").mkdir(exist_ok=True)
    thr = STABILITY_S if stability_s is None else stability_s
    out = []
    # cleanup state for disappeared files
    try:
        existing = {str((inbox_dir / p.name).resolve()) for p in inbox_dir.iterdir() if p.is_file()}
        for k in list(_file_state.keys()):
            if k not in existing:
                _file_state.pop(k, None)
    except Exception:
        pass
    for p in inbox_dir.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if not _is_stable(p, stability_s=thr):
            continue
        try:
            res = process_one(inbox_dir, p)
            out.append(res)
        except Exception:
            out.append({"error": str(p.name)})
    return out

def poll_forever(inbox_dir: str | Path, interval: float = 2.0, stop_event: threading.Event | None = None):
    inbox_dir = Path(inbox_dir)
    while True:
        if stop_event and stop_event.is_set():
            break
        try:
            process_inbox_once(inbox_dir)
        except Exception:
            pass
        if stop_event:
            if stop_event.wait(interval):
                break
        else:
            time.sleep(interval)

def start_inbox_thread(inbox_dir: str | Path, interval: float = 2.0) -> threading.Thread:
    stop = threading.Event()
    t = threading.Thread(target=poll_forever, args=(Path(inbox_dir), interval, stop), daemon=True)
    t.start()
    t._stop_event = stop  # type: ignore
    return t

def main():
    import argparse
    ap = argparse.ArgumentParser(description="LabVision inbox poller")
    ap.add_argument("inbox", nargs="?", default="./inbox", help="inbox folder")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="process once and exit")
    args = ap.parse_args()
    inbox_dir = Path(args.inbox)
    if args.once:
        res = process_inbox_once(inbox_dir)
        print(f"processed {len(res)} files")
    else:
        print(f"Polling {inbox_dir.resolve()} every {args.interval}s — Ctrl+C to stop")
        try:
            poll_forever(inbox_dir, interval=args.interval)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
