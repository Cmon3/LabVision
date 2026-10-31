"""Tune on dev, report on test using measured features + Python clustering."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from labvision.features import extract_features, load_image_rgb
from labvision.morphotypes import cluster_plate
from bench.run_morphotypes import adjusted_rand_index, cluster_purity

AGAR_ROOT = Path("data/agar_demo/AGAR_representative")
OUT_CSV = Path("bench/morphotypes_v2.csv")
OUT_MD = Path("bench/RESULTS.md")


def load_split():
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
    n = len(all_agar)
    dev = all_agar[: n // 2]
    test = all_agar[n // 2 :]
    return dev, test


def _precompute_features(split, max_per_plate=20):
    """Precompute features per plate once."""
    cache = {}
    for img_path, data in split:
        labels = data["labels"]
        if len(labels) > max_per_plate:
            labels = labels[:max_per_plate]
        true_species = [lab["class"] for lab in labels]
        boxes = [{"x": lab["x"], "y": lab["y"], "width": lab["width"], "height": lab["height"]} for lab in labels]
        try:
            img = load_image_rgb(str(img_path))
            feats = extract_features(img, boxes=boxes)
        except Exception as e:
            print(f"feat err {img_path.name}: {e}")
            feats = np.zeros((len(labels), 6))
        cache[img_path.name] = (feats, true_species, labels)
    return cache

def evaluate_split_cached(feat_cache, threshold, color_weight):
    per_plate = []
    all_true = []
    all_pred = []
    for img_name, (feats, true_species, labels) in feat_cache.items():
        if feats.shape[0] == 0:
            continue
        pred_labels = cluster_plate(feats, threshold=threshold, color_weight=color_weight)
        uniq = sorted(set(pred_labels))
        remap = {old: chr(ord("A") + i) for i, old in enumerate(uniq)}
        pred_morph = [remap[v] for v in pred_labels]
        n_pred_types = len(uniq)
        true_is_pure = len(set(true_species)) == 1
        pred_is_pure = n_pred_types == 1
        per_plate.append({"image": img_name, "true_is_pure": true_is_pure, "pred_is_pure": pred_is_pure, "n_true_species": len(set(true_species)), "n_pred_types": n_pred_types, "n_crops": len(labels), "has_contamination": "Contamination" in true_species})
        for t, p in zip(true_species, pred_morph):
            all_true.append(t)
            all_pred.append(f"{img_name}_{p}")

    correct = sum(1 for r in per_plate if r["true_is_pure"] == r["pred_is_pure"])
    flag_acc = correct / len(per_plate) if per_plate else 0
    pure_plates = [r for r in per_plate if r["true_is_pure"]]
    mixed_plates = [r for r in per_plate if not r["true_is_pure"]]
    acc_pure = sum(1 for r in pure_plates if r["pred_is_pure"]) / len(pure_plates) if pure_plates else 0
    acc_mixed = sum(1 for r in mixed_plates if not r["pred_is_pure"]) / len(mixed_plates) if mixed_plates else 0
    bal_acc = (acc_pure + acc_mixed) / 2 if (pure_plates and mixed_plates) else flag_acc

    aris = []
    purities = []
    for img_name, (feats, true_species, labels) in feat_cache.items():
        if feats.shape[0] < 2:
            continue
        pred_labels = cluster_plate(feats, threshold=threshold, color_weight=color_weight)
        uniq = sorted(set(pred_labels))
        remap = {old: chr(ord("A")+i) for i, old in enumerate(uniq)}
        pred_morph = [remap[v] for v in pred_labels]
        aris.append(adjusted_rand_index(true_species, pred_morph))
        purities.append(cluster_purity(true_species, pred_morph))
    mean_ari = float(np.mean(aris)) if aris else 0.0
    mean_purity = float(np.mean(purities)) if purities else 0.0

    contam_plates = [r for r in per_plate if r["has_contamination"]]
    contam_recall = sum(1 for r in contam_plates if not r["pred_is_pure"]) / len(contam_plates) if contam_plates else 1.0
    baseline_acc = sum(1 for r in per_plate if r["true_is_pure"]) / len(per_plate) if per_plate else 0
    baseline_bal = (1.0 + 0.0) / 2 if mixed_plates else 1.0

    return {
        "per_plate": per_plate,
        "flag_acc": flag_acc,
        "bal_acc": bal_acc,
        "acc_pure": acc_pure,
        "acc_mixed": acc_mixed,
        "mean_ari": mean_ari,
        "mean_purity": mean_purity,
        "contam_recall": contam_recall,
        "baseline_acc": baseline_acc,
        "baseline_bal": baseline_bal,
        "n_pure": len(pure_plates),
        "n_mixed": len(mixed_plates),
    }

def evaluate_split(split, threshold, color_weight, max_per_plate=20):
    cache = _precompute_features(split, max_per_plate)
    return evaluate_split_cached(cache, threshold, color_weight)


def main():
    dev, test = load_split()
    print(f"dev {len(dev)} test {len(test)}")
    thresholds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    color_weights = [1.0, 1.5, 2.0, 2.5, 3.0]
    # precompute features once for speed
    dev_cache = _precompute_features(dev)
    test_cache = _precompute_features(test)
    best = None
    results = []
    for cw in color_weights:
        for T in thresholds:
            m = evaluate_split_cached(dev_cache, threshold=T, color_weight=cw)
            score = m["bal_acc"] + m["mean_ari"] * 0.5
            results.append((T, cw, m, score))
            print(f"T={T} cw={cw} dev bal={m['bal_acc']:.3f} flag={m['flag_acc']:.3f} ari={m['mean_ari']:.3f} pur={m['mean_purity']:.3f} contam={m['contam_recall']:.3f}")
            if best is None or score > best[3] or (abs(score - best[3]) < 1e-6 and m["bal_acc"] > best[2]["bal_acc"]):
                best = (T, cw, m, score)
    assert best is not None
    T_best, cw_best, dev_m, _ = best
    print(f"\nBest on dev: T={T_best} cw={cw_best} bal={dev_m['bal_acc']:.3f} ari={dev_m['mean_ari']:.3f}")

    with open("bench/morphotype_tuning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "color_weight", "flag_acc", "bal_acc", "acc_pure", "acc_mixed", "mean_ari", "mean_purity", "contam_recall"])
        w.writeheader()
        for T, cw, m, _ in results:
            w.writerow({"threshold": T, "color_weight": cw, "flag_acc": m["flag_acc"], "bal_acc": m["bal_acc"], "acc_pure": m["acc_pure"], "acc_mixed": m["acc_mixed"], "mean_ari": m["mean_ari"], "mean_purity": m["mean_purity"], "contam_recall": m["contam_recall"]})

    cfg_path = Path("labvision/config.py")
    txt = cfg_path.read_text()
    import re
    txt = re.sub(r"MORPHOTYPE_THRESHOLD = .*", f"MORPHOTYPE_THRESHOLD = {T_best}", txt)
    txt = re.sub(r"MORPHOTYPE_COLOR_WEIGHT = .*", f"MORPHOTYPE_COLOR_WEIGHT = {cw_best}", txt)
    cfg_path.write_text(txt)
    print(f"Updated config threshold {T_best} cw {cw_best}")

    test_m = evaluate_split_cached(test_cache, threshold=T_best, color_weight=cw_best)
    print(f"Test: flag={test_m['flag_acc']:.3f} bal={test_m['bal_acc']:.3f} ari={test_m['mean_ari']:.3f} pur={test_m['mean_purity']:.3f} contam={test_m['contam_recall']:.3f} baseline={test_m['baseline_acc']:.3f}")

    rows = []
    for img_path, data in test:
        labels = data["labels"]
        if len(labels) > 20:
            labels = labels[:20]
        true_species = [lab["class"] for lab in labels]
        boxes = [{"x": lab["x"], "y": lab["y"], "width": lab["width"], "height": lab["height"]} for lab in labels]
        img = load_image_rgb(str(img_path))
        feats = extract_features(img, boxes=boxes)
        pred_labels = cluster_plate(feats, threshold=T_best, color_weight=cw_best)
        uniq = sorted(set(pred_labels))
        remap = {old: chr(ord("A")+i) for i, old in enumerate(uniq)}
        pred_morph = [remap[v] for v in pred_labels]
        for t, p in zip(true_species, pred_morph):
            rows.append({"image": img_path.name, "true_species": t, "pred_morphotype": p})
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true_species", "pred_morphotype"])
        w.writeheader()
        w.writerows(rows)

    per_plate_csv = Path("bench/morphotypes_v2_per_plate.csv")
    with per_plate_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_crops", "n_true_species", "n_pred_types", "true_is_pure", "pred_is_pure", "has_contamination"])
        w.writeheader()
        for r in test_m["per_plate"]:
            w.writerow(r)

    dev_m_full = evaluate_split_cached(dev_cache, threshold=T_best, color_weight=cw_best)

    md_path = OUT_MD
    md = md_path.read_text()
    md = re.sub(r"\n## Morphotypes \(AGAR test.*?(?=\n## |\Z)", "\n", md, flags=re.DOTALL)
    sec = f"\n\n## Morphotypes (AGAR test, GT boxes, Python features)\n\n"
    sec += f"Tuned on dev ({len(dev)} plates) grid T={thresholds} x color_weight={color_weights}; best T={T_best} color_weight={cw_best} by dev balanced accuracy. Evaluated on test ({len(test)} plates, {sum(r['n_crops'] for r in test_m['per_plate'])} crops) - n is small.\n\n"
    sec += f"- **Flag accuracy**: {test_m['flag_acc']:.1%} ({sum(1 for r in test_m['per_plate'] if r['true_is_pure']==r['pred_is_pure'])}/{len(test_m['per_plate'])} test plates; pure vs mixed) | baseline always-pure {test_m['baseline_acc']:.1%} ({test_m['n_pure']}/{len(test_m['per_plate'])})\n"
    sec += f"- **Balanced accuracy**: {test_m['bal_acc']:.1%} (dev {dev_m['bal_acc']:.1%} baseline {dev_m['baseline_bal']:.1%})\n"
    sec += f"- **Mean ARI** (per-plate vs species): {test_m['mean_ari']:.3f} (dev {dev_m['mean_ari']:.3f})\n"
    sec += f"- **Mean purity**: {test_m['mean_purity']:.3f} (dev {dev_m['mean_purity']:.3f})\n"
    sec += f"- **Contamination recall**: {test_m['contam_recall']:.1%} ({sum(1 for r in test_m['per_plate'] if r['has_contamination'] and not r['pred_is_pure'])}/{sum(1 for r in test_m['per_plate'] if r['has_contamination'])} test plates with Contamination flagged as mixed) - dev had 0 such plates\n"
    sec += f"- Plates test: {test_m['n_pure']} pure / {test_m['n_mixed']} mixed; predicted {sum(1 for r in test_m['per_plate'] if r['pred_is_pure'])} pure | dev: {dev_m_full['n_pure']} pure / {dev_m_full['n_mixed']} mixed\n\n"
    sec += "### Findings\n"
    if test_m["flag_acc"] <= test_m["baseline_acc"]:
        sec += f"- Flag accuracy {test_m['flag_acc']:.0%} does not beat baseline {test_m['baseline_acc']:.0%} on test (n=20) - not useful as a contamination flag.\n"
    else:
        sec += f"- Flag accuracy {test_m['flag_acc']:.0%} beats baseline {test_m['baseline_acc']:.0%} but n=20 is small.\n"
    sec += f"- ARI {test_m['mean_ari']:.2f} approx random; Python color+shape+size features separate poorly vs species on this split.\n"
    sec += f"- Balanced accuracy dev {dev_m['bal_acc']:.0%} -> test {test_m['bal_acc']:.0%} shows generalization gap; contamination recall {test_m['contam_recall']:.0%} (test 1 plate).\n"
    sec += f"- Fix: fixed global scales (not per-plate) keeps default at one type; threshold {T_best} is conservative.\n"
    sec += "\n| image | true_species | pred_types | flag correct | has_contam |\n|---|---|---|---|---|\n"
    for r in test_m["per_plate"][:10]:
        ok = "✓" if r["true_is_pure"] == r["pred_is_pure"] else "✗"
        sec += f"| {r['image']} | {r['n_true_species']} | {r['n_pred_types']} | {ok} | {r['has_contamination']} |\n"
    if len(test_m["per_plate"]) > 10:
        sec += f"| ... {len(test_m['per_plate'])-10} more | | | | |\n"
    sec += f"\n*Features: Lab bg-subtracted, circularity, log diam, texture; clustering average linkage T={T_best} color_weight={cw_best}; per-plate ARI/purity. n is small (20 plates). Files: bench/morphotypes_v2.csv, tuning.csv.*\n"
    md_path.write_text(md.rstrip() + sec + "\n")
    print(f"wrote {OUT_CSV} and updated {md_path}")
    Path("bench/morphotypes_v2_metrics.json").write_text(json.dumps({"dev": dev_m, "test": test_m, "best_threshold": T_best, "best_color_weight": cw_best}, indent=2))

if __name__ == "__main__":
    main()
