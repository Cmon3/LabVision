"""Simulator for inbox demo: copy example plates into inbox every X seconds."""
from __future__ import annotations
import argparse
import random
import shutil
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# curated examples: lower-res normal, higher-res dense 518, etc.
CANDIDATES = [
    "data/agar_demo/AGAR_representative/lower-resolution/14684.jpg",
    "data/agar_demo/AGAR_representative/lower-resolution/13895.jpg",
    "data/agar_demo/AGAR_representative/lower-resolution/14581.jpg",
    "data/agar_demo/AGAR_representative/higher-resolution/bright/518.jpg",  # dense ~298, TNTC story
    "data/agar_demo/AGAR_representative/higher-resolution/bright/510.jpg",
    "data/agar_demo/AGAR_representative/lower-resolution/356.jpg",
]

def _blur_copy(src: Path, dst: Path):
    try:
        import cv2
        img = cv2.imread(str(src))
        if img is not None:
            blurred = cv2.GaussianBlur(img, (31, 31), 0)
            # write to temp then rename handled by caller; here just write to dst
            cv2.imwrite(str(dst), blurred)
            return True
    except Exception:
        pass
    # fallback: just copy
    shutil.copy2(src, dst)
    return False

def _copy_via_temp(src: Path, inbox: Path, final_name: str, sidecar: dict | None = None):
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "processed").mkdir(exist_ok=True)
    (inbox / "failed").mkdir(exist_ok=True)
    tmp = inbox / f".tmp.{final_name}"
    final = inbox / final_name
    # copy image via temp
    shutil.copy2(src, tmp)
    tmp.rename(final)
    if sidecar is not None:
        # sidecar via temp
        tmp_j = inbox / f".tmp.{Path(final_name).stem}.json"
        final_j = inbox / f"{Path(final_name).stem}.json"
        import json
        tmp_j.write_text(json.dumps(sidecar))
        tmp_j.rename(final_j)
    return final

def main():
    ap = argparse.ArgumentParser(description="Simulate camera feeding inbox")
    ap.add_argument("--inbox", default="./inbox", help="inbox folder")
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between copies")
    ap.add_argument("--count", type=int, default=10, help="how many to copy (0=loop forever)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle order")
    ap.add_argument("--once", action="store_true", help="copy one batch and exit")
    args = ap.parse_args()

    inbox = Path(args.inbox)
    # filter to existing
    srcs = [Path(p) for p in CANDIDATES if Path(p).exists()]
    if not srcs:
        print("No candidate images found — check data/agar_demo/ exists")
        return 1
    if args.shuffle:
        random.shuffle(srcs)

    # sidecars: example alert_limits (clearly labelled as example)
    # 14684 normal with alert 50 (will trigger if count 55>50), 518 dense with 40, 510 with 30
    alert_map = {
        "14684.jpg": {"location": "Bench A", "sample_type": "water", "alert_limit": 50},
        "518.jpg": {"location": "Bench B", "sample_type": "food", "alert_limit": 40},
        "510.jpg": {"location": "Incubator 2", "sample_type": "surface", "alert_limit": 30},
    }

    # plan sequence: include blurred 14684
    seq = []
    for p in srcs:
        sidecar = alert_map.get(p.name)
        seq.append((p, sidecar, False))
    # add one blurred copy as extra
    for p in srcs:
        if p.name == "14684.jpg":
            seq.append((p, {"location": "Bench A", "sample_type": "water", "alert_limit": 50, "note": "blurred example"}, True))
            break

    # if count limited, truncate/repeat
    if args.count > 0:
        # repeat seq to fill count
        extended = []
        while len(extended) < args.count:
            extended.extend(seq)
        seq = extended[:args.count]
    else:
        # infinite loop
        pass

    print(f"Simulator: {len(seq)} images -> {inbox.resolve()} every {args.interval}s (temp+rename)")

    idx = 0
    try:
        while True:
            if args.count > 0 and idx >= len(seq):
                break
            src, sidecar, do_blur = seq[idx % len(seq)]
            # final name with timestamp to avoid collision
            stamp = time.strftime("%H%M%S")
            if do_blur:
                final_name = f"{src.stem}_blurred_{stamp}.jpg"
                # need to create blurred image via temp
                tmp_img = inbox / f".tmp.{final_name}"
                _blur_copy(src, tmp_img)
                tmp_img.rename(inbox / final_name)
                if sidecar:
                    import json
                    tmp_j = inbox / f".tmp.{Path(final_name).stem}.json"
                    final_j = inbox / f"{Path(final_name).stem}.json"
                    tmp_j.write_text(json.dumps(sidecar))
                    tmp_j.rename(final_j)
                print(f"[{idx+1}/{len(seq)}] blurred {src.name} -> {final_name} (alert {sidecar.get('alert_limit') if sidecar else 'none'})")
            else:
                # normal copy via temp+rename helper, but need unique name to avoid overwrite
                # use original stem + stamp
                final_name = f"{src.stem}_{stamp}.jpg" if idx >= len(srcs) else src.name
                # if final_name already exists (reuse), make unique
                if (inbox / final_name).exists():
                    final_name = f"{src.stem}_{stamp}_{idx}.jpg"
                _copy_via_temp(src, inbox, final_name, sidecar)
                print(f"[{idx+1}/{len(seq)}] {src.name} -> {final_name} {'alert '+str(sidecar['alert_limit']) if sidecar else ''}")
            idx += 1
            if args.once:
                break
            if args.count == 0 or idx < len(seq):
                time.sleep(args.interval)
            else:
                break
    except KeyboardInterrupt:
        print("\nStopped")
    print("Done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
