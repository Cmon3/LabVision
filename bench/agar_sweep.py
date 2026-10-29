"""Prompt + tiling sweep on AGAR dev, pick best by MAE, write config + sweep csv."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from labvision.config import AGAR_CLIP_RATIO
from labvision.plate import crop_to_plate
from labvision.segment import segment
from labvision.tiling import segment_tiled

AGAR_ROOT = Path("data/agar_demo/AGAR_representative")
PROMPTS = ["colony", "colonies", "bacterial colony", "colonies on agar plate", "round spot"]
TILINGS = [1, 2, 3]
OUT = Path("bench/agar_sweep.csv")


def load_agar():
    imgs = []
    for p in AGAR_ROOT.rglob("*.jpg"):
        js = p.with_suffix(".json")
        if not js.exists():
            continue
        data = json.loads(js.read_text())
        cnt = data.get("colonies_number")
        if cnt is None or cnt == -1:
            continue
        imgs.append((p, cnt))
    imgs = sorted(imgs, key=lambda x: x[0].name)
    n = len(imgs)
    dev = imgs[: n // 2]
    test = imgs[n // 2 :]
    print(f"AGAR countable {n}: dev {len(dev)}, test {len(test)}")
    return dev, test


def eval_prompt_tiling(dev, prompt, tiling):
    errs = []
    for path, true in dev:
        try:
            b = path.read_bytes()
            # optional plate crop
            # for now no crop
            if tiling == 1:
                seg = segment(b, prompt)
            else:
                seg = segment_tiled(b, prompt, tiling=tiling)
            # use box count (more reliable than mask area for AGAR); clipping handled in decoded_masks if needed
            pred = len(seg.boxes)
            errs.append(abs(pred - true))
        except Exception as e:
            print(f"err {path.name} {prompt} t{tiling}: {e}")
            errs.append(abs(true))
    mae = sum(errs) / len(errs) if errs else 999
    return mae


def main():
    dev, test = load_agar()
    rows = []
    # prompt sweep whole only
    best_prompt = None
    best_mae = None
    for prompt in PROMPTS:
        mae = eval_prompt_tiling(dev, prompt, 1)
        rows.append({"prompt": prompt, "tiling": 1, "mae_dev": f"{mae:.2f}"})
        print(f"prompt={prompt!r} tiling=1 MAE={mae:.2f}")
        if best_mae is None or mae < best_mae:
            best_mae, best_prompt = mae, prompt
    # tiling sweep with best prompt
    best = (best_mae, best_prompt, 1)
    for tiling in [2, 3]:
        mae = eval_prompt_tiling(dev, best_prompt, tiling)
        rows.append({"prompt": best_prompt, "tiling": tiling, "mae_dev": f"{mae:.2f}"})
        print(f"prompt={best_prompt!r} tiling={tiling} MAE={mae:.2f}")
        if mae < best[0]:
            best = (mae, best_prompt, tiling)
    # write sweep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "tiling", "mae_dev"])
        w.writeheader()
        w.writerows(rows)
    print(f"best {best} -> {OUT}")

    # save config
    cfg = Path("labvision/config.py")
    cfg.write_text(
        f'"""Best AGAR config chosen on dev (updated by bench/agar_sweep.py)."""\n'
        f'AGAR_PROMPT = {best[1]!r}\n'
        f'AGAR_TILING = {best[2]}\n'
        f"AGAR_OVERLAP = 0.15\n"
        f"AGAR_NMS_IOU = 0.5\n"
        f"AGAR_CLIP_RATIO = 2.0\n"
        f"AGAR_PLATE_CROP = False\n"
    )
    # eval on test with best
    mae_test = eval_prompt_tiling(test, best[1], best[2])
    print(f"test MAE {mae_test:.2f} with {best[1]!r} tiling {best[2]}")


if __name__ == "__main__":
    main()
