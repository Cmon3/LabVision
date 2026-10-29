"""Benchmark characterization on AGAR test split using GT boxes."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from labvision.characterize import characterize

AGAR_ROOT = Path("data/agar_demo/AGAR_representative")
OUT_CSV = Path("bench/species.csv")


def load_test():
    all_agar = []
    for p in AGAR_ROOT.rglob("*.jpg"):
        js = p.with_suffix(".json")
        if not js.exists():
            continue
        d = json.loads(js.read_text())
        cnt = d.get("colonies_number")
        if cnt is None or cnt == -1:
            continue
        labels = d.get("labels", [])
        if not labels:
            continue
        all_agar.append((p, d))
    all_agar = sorted(all_agar, key=lambda x: x[0].name)
    test = all_agar[len(all_agar) // 2 :]
    print(f"test {len(test)} images")
    return test


def main():
    test = load_test()
    # collect up to 400 crops (spec)
    rows = []
    correct = 0
    total = 0
    per_class = defaultdict(lambda: [0, 0])  # correct,total
    confusion = defaultdict(Counter)
    crop_count = 0
    for img_path, data in test:
        labels = data["labels"]
        true_classes = [l["class"] for l in labels]
        # limit to 24 per image (contact sheet max) and 400 total
        labels = labels[:24]
        true_classes = true_classes[:24]
        if crop_count + len(labels) > 400:
            labels = labels[: 400 - crop_count]
            true_classes = true_classes[: 400 - crop_count]
        if not labels:
            continue
        boxes = [{"x": l["x"], "y": l["y"], "width": l["width"], "height": l["height"]} for l in labels]
        try:
            preds = characterize(img_path, boxes)
        except Exception as e:
            print(f"vision failed {img_path.name}: {e}")
            preds = [
                {"presumptive_organism": "unknown", "confidence": 0.0, "color": "unknown", "shape": "unknown", "margin": "unknown", "elevation": "unknown", "size_class": "medium"}
                for _ in boxes
            ]
            if "vision" in str(e).lower() or "image" in str(e).lower():
                Path("bench/species_error.log").write_text(str(e))
        for true, pred in zip(true_classes, preds):
            p_org = pred["presumptive_organism"]
            rows.append({"image": img_path.name, "true": true, "pred": p_org, "confidence": pred["confidence"]})
            confusion[true][p_org] += 1
            per_class[true][1] += 1
            if p_org == true:
                correct += 1
                per_class[true][0] += 1
            total += 1
        crop_count += len(labels)
        print(f"{img_path.name} {len(labels)} crops, acc {correct}/{total}")
        if crop_count >= 400:
            break

    # write species.csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true", "pred", "confidence"])
        w.writeheader()
        w.writerows(rows)

    # update RESULTS.md section
    acc = correct / total if total else 0
    print(f"overall acc {acc:.2%} {correct}/{total}")
    # per-class
    for cls, (c, t) in per_class.items():
        print(f"{cls}: {c}/{t} {c/t:.1%}")

    # append to RESULTS.md
    md = Path("bench/RESULTS.md").read_text()
    sec = "\n\n## Species characterization (AGAR test, GT boxes)\n\n"
    sec += f"Accuracy {acc:.1%} ({correct}/{total}) over {len(test)} images, {crop_count} crops (max 400)\n\n"
    sec += "| true \\ pred | " + " | ".join(sorted(set(confusion.keys()) | {k for v in confusion.values() for k in v.keys()})) + " |\n"
    sec += "|---|" + "---|" * len(set(confusion.keys()) | {k for v in confusion.values() for k in v.keys()}) + "\n"
    for true in sorted(confusion.keys()):
        row = [true] + [str(confusion[true].get(pred, 0)) for pred in sorted(set(confusion.keys()) | {k for v in confusion.values() for k in v.keys()})]
        sec += "| " + " | ".join(row) + " |\n"
    sec += "\n*Presumptive organism via Muse Spark Open vision on contact sheets (24 crops/call). Confirm with standard microbiological methods.*\n"
    Path("bench/RESULTS.md").write_text(md + sec)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
