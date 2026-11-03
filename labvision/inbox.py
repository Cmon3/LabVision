"""Inbox ingest — poll folder for new plate photos."""
from __future__ import annotations

import json
import shutil
import time
import threading
from pathlib import Path

RUNS_DIR = Path("runs")
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

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
    while (RUNS_DIR / rid).exists():
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
            rdir = RUNS_DIR / result["run_id"]
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

def process_inbox_once(inbox_dir: str | Path) -> list[dict]:
    inbox_dir = Path(inbox_dir)
    inbox_dir.mkdir(parents=True, exist_ok=True)
    (inbox_dir / "processed").mkdir(exist_ok=True)
    (inbox_dir / "failed").mkdir(exist_ok=True)
    out = []
    for p in inbox_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        # skip files still being written — quick size check if needed (ponytail: skip, copy is atomic)
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
