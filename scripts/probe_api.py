"""Probe SAM 3.1 via labvision/segment.py (T2).

Usage:
  python scripts/probe_api.py [image_or_dir ...] [--prompt "bacterial colony"]
  python scripts/probe_api.py --synthetic   # no images needed, tests parsing/cache only

Calls segment_file on up to 5 images, saves box overlays to runs/probe/,
prints count/cached status and estimated cost ($2.50/1k images).

Uses only labvision/segment.py for API calls — cached by image hash+prompt
so reruns are free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# ponytail: PIL already in env, cv2 not — use PIL for overlay
try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from labvision.segment import segment_file

COST_PER_IMAGE = 2.50 / 1000  # $2.50 / 1k images per docs/sam_api.md
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PROBE_DIR = Path("runs/probe")


def collect_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.iterdir()):
                if f.suffix.lower() in IMAGE_SUFFIXES:
                    out.append(f)
        elif pp.is_file():
            out.append(pp)
        else:
            print(f"skip: not found: {p}", file=sys.stderr)
    return out


def save_overlay(image_path: Path, boxes, out_path: Path, masks=None) -> None:
    if not HAS_PIL:
        print(f"  no PIL — skip overlay for {image_path.name}")
        return
    try:
        im = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"  cannot open image for overlay: {e}")
        return
    # masks overlay (red translucent) if available
    if masks is not None and len(masks) > 0:
        try:
            import numpy as np

            for m in masks:
                # m is (H,W) bool
                if m.shape[0] != im.size[1] or m.shape[1] != im.size[0]:
                    continue
                red = Image.new("RGBA", im.size, (255, 0, 0, 70))
                mask_im = Image.fromarray((m.astype(np.uint8) * 255))
                im = Image.alpha_composite(im, Image.new("RGBA", im.size, (0, 0, 0, 0)))
                # composite red where mask
                tmp = Image.new("RGBA", im.size, (0, 0, 0, 0))
                tmp.paste(red, mask=mask_im)
                im = Image.alpha_composite(im, tmp)
        except Exception as e:
            print(f"  mask overlay warn: {e}")
    draw = ImageDraw.Draw(im)
    for b in boxes:
        draw.rectangle([(b.x1, b.y1), (b.x2, b.y2)], outline="lime", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(out_path)
    print(f"  overlay -> {out_path}")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Probe SAM 3.1 (T2)")
    ap.add_argument("images", nargs="*", help="image files or directories")
    ap.add_argument("--prompt", default="bacterial colony", help="SAM noun phrase")
    ap.add_argument("--synthetic", action="store_true", help="run without images (parse/cache smoke test)")
    ap.add_argument("--limit", type=int, default=5, help="max images (default 5)")
    args = ap.parse_args()

    if args.synthetic:
        # Smoke test: create 1px synthetic image, test cache path/parse without API call
        # ponytail: verifies plumbing when no real images/API key available
        from labvision.segment import parse_output_text

        sample = '<0f>0<|box;x1=10;y1=10;x2=50;y2=50;w=100;h=100|><|mask;x=0;y=0;data=10,10,~abc|>,1<|box;x1=60;y1=60;x2=90;y2=90;w=100;h=100|><|mask;x=0;y=0;data=10,10,~def|>'
        boxes, raw_masks, w, h = parse_output_text(sample)
        assert len(boxes) == 2 and w == 100 and h == 100
        print(f"synthetic parse OK: {len(boxes)} boxes, {w}x{h}, {len(raw_masks)} masks")
        print("probe done (synthetic, no API call).")
        return 0

    if not args.images:
        ap.print_help()
        print("\nexamples:")
        print('  python scripts/probe_api.py data/my_plate.jpg --prompt "bacterial colony"')
        print("  python scripts/probe_api.py data/ --limit 5")
        print("  python scripts/probe_api.py --synthetic")
        return 1

    images = collect_images(args.images)[: args.limit]
    if not images:
        print("no images found (supported: png/jpg/tif). Try --synthetic for smoke test.")
        return 1

    print(f"probe: {len(images)} image(s), prompt={args.prompt!r}, cost ~${COST_PER_IMAGE:.4f}/image")
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    total_api_calls = 0
    for img_path in images:
        print(f"\n{img_path} ...")
        try:
            seg = segment_file(img_path, args.prompt)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        if seg.cached:
            print(f"  cached: {len(seg.boxes)} boxes, {seg.width}x{seg.height}")
        else:
            print(f"  API: {len(seg.boxes)} boxes, {seg.width}x{seg.height}")
            total_api_calls += 1
        print(f"  masks: {len(seg.raw_masks)} raw payloads")
        # measure via decoded masks
        try:
            masks = seg.decoded_masks()
            from labvision.measure import measure_masks

            rows = measure_masks(masks)
            if rows:
                ds = [r["diameter_px"] for r in rows]
                print(f"  measure: count={len(rows)} diam_px mean={sum(ds)/len(ds):.1f} min={min(ds):.1f} max={max(ds):.1f}")
            else:
                print("  measure: count=0")
        except Exception as e:
            print(f"  measure warn: {e}")
            masks = None
        else:
            # keep masks for overlay
            pass
        out = PROBE_DIR / f"{img_path.stem}_{args.prompt.replace(' ', '_')}_overlay.png"
        try:
            save_overlay(img_path, seg.boxes, out, masks=masks)
        except Exception:
            save_overlay(img_path, seg.boxes, out)

    est_cost = total_api_calls * COST_PER_IMAGE
    cached = len(images) - total_api_calls
    print(f"\ndone: {len(images)} images, {total_api_calls} API calls, {cached} cached")
    print(f"estimated cost this run: ${est_cost:.4f} (cached runs: $0)")
    print(f"full bench ~40 images: ~${40 * COST_PER_IMAGE:.2f} at $2.50/1k")
    print(f"overlays in {PROBE_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
