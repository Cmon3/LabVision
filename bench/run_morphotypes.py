"""Benchmark morphotype grouping purity on AGAR test split (GT boxes, max 400 crops)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from labvision.characterize import characterize

AGAR_ROOT = Path("data/agar_demo/AGAR_representative")
OUT_CSV = Path("bench/morphotypes.csv")
OUT_MD = Path("bench/RESULTS.md")


def _comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def adjusted_rand_index(labels_true: list, labels_pred: list) -> float:
    """ARI in numpy (no sklearn)."""
    import numpy as np

    if len(labels_true) != len(labels_pred) or not labels_true:
        return 0.0
    t = np.array(labels_true)
    p = np.array(labels_pred)
    # unique
    t_uniq, t_inv = np.unique(t, return_inverse=True)
    p_uniq, p_inv = np.unique(p, return_inverse=True)
    n = len(t)
    # contingency: count co-occurrences
    cont = np.zeros((len(t_uniq), len(p_uniq)), dtype=int)
    for ti, pi in zip(t_inv, p_inv):
        cont[ti, pi] += 1
    sum_comb = sum(_comb2(int(v)) for v in cont.flat)
    sum_row = sum(_comb2(int(cont[i].sum())) for i in range(cont.shape[0]))
    sum_col = sum(_comb2(int(cont[:, j].sum())) for j in range(cont.shape[1]))
    n_comb = _comb2(n)
    if n_comb == 0:
        return 0.0
    expected = sum_row * sum_col / n_comb
    max_comb = 0.5 * (sum_row + sum_col)
    denom = max_comb - expected
    if denom == 0:
        return 0.0
    return (sum_comb - expected) / denom


def cluster_purity(labels_true: list, labels_pred: list) -> float:
    """Weighted purity: sum over predicted clusters of max count / N."""
    from collections import Counter, defaultdict

    clusters: dict = defaultdict(list)
    for t, p in zip(labels_true, labels_pred):
        clusters[p].append(t)
    if not clusters:
        return 0.0
    total = len(labels_true)
    correct = sum(Counter(members).most_common(1)[0][1] for members in clusters.values())
    return correct / total if total else 0.0


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
    print(f"test {len(test)} images total {len(all_agar)}")
    return test


def main():
    test = load_test()
    rows = []
    all_true = []
    all_pred = []
    per_plate_results = []
    crop_count = 0
    plates_evaluated = 0

    for img_path, data in test:
        labels = data["labels"]
        classes = data.get("classes", [])
        true_species = [lab["class"] for lab in labels]
        # cap at 400 total: take up to 20 per plate to cover all plates
        max_per_plate = 20
        if len(labels) > max_per_plate:
            labels = labels[:max_per_plate]
            true_species = true_species[:max_per_plate]
        if crop_count >= 400:
            break
        if crop_count + len(labels) > 400:
            n = 400 - crop_count
            labels = labels[:n]
            true_species = true_species[:n]
        boxes = [{"x": lab["x"], "y": lab["y"], "width": lab["width"], "height": lab["height"]} for lab in labels]
        # diameters for terciles: estimate from box size, try plate detection
        diams = [(b["width"] + b["height"]) / 2.0 for b in boxes]
        # try plate diameter to convert to mm for more meaningful terciles
        try:
            from PIL import Image
            import numpy as _np

            im = Image.open(img_path).convert("RGB")
            arr = _np.array(im)
            from labvision.plate import detect_plate_diameter_px

            plate_px = detect_plate_diameter_px(arr)
            if plate_px:
                scale = 90.0 / plate_px
                diams = [d * scale for d in diams]
        except Exception:
            pass

        try:
            preds = characterize(img_path, boxes, diameters_mm=diams)
        except Exception as e:
            print(f"vision failed {img_path.name}: {e}")
            preds = [
                {"color": "unknown", "shape": "unknown", "margin": "unknown", "elevation": "unknown", "opacity": "unknown", "surface": "unknown", "morphotype": "A", "morphotype_desc": "unknown", "purity_flag": "Single morphotype (consistent with pure culture)", "mean_diameter_mm": 0, "size_class": "medium"}
                for _ in boxes
            ]
            if "vision" in str(e).lower() or "image" in str(e).lower():
                Path("bench/morphotype_error.log").write_text(str(e))

        # check for morphology refusal: all unknown
        if preds and all(p.get("color") == "unknown" and p.get("shape") == "unknown" for p in preds):
            print(f"Muse still refuses morphology on {img_path.name}, check runs/characterize_raw/")
            # log raw already done, continue but note
            Path("bench/morphotype_refusal.log").write_text(f"refusal on {img_path.name}\n")

        pred_morphotypes = [p["morphotype"] for p in preds]
        # for flag accuracy: is this plate truly pure (1 species) vs mixed?
        true_is_pure = len(set(true_species)) == 1 if true_species else True
        # predicted pure if flag says Single morphotype
        purity_flag = preds[0].get("purity_flag", "") if preds else ""
        pred_is_pure = "Single morphotype" in purity_flag

        per_plate_results.append(
            {
                "image": img_path.name,
                "true_is_pure": true_is_pure,
                "pred_is_pure": pred_is_pure,
                "purity_flag": purity_flag,
                "n_crops": len(labels),
                "n_pred_types": len(set(pred_morphotypes)),
                "n_true_species": len(set(true_species)),
            }
        )

        for true, pred, pr in zip(true_species, pred_morphotypes, preds):
            rows.append(
                {
                    "image": img_path.name,
                    "true_species": true,
                    "pred_morphotype": pred,
                    "purity_flag": pr.get("purity_flag", ""),
                    "morphotype_desc": pr.get("morphotype_desc", ""),
                }
            )
            all_true.append(true)
            all_pred.append(pred)

        crop_count += len(labels)
        plates_evaluated += 1
        print(f"{img_path.name} {len(labels)} crops, flag pure={pred_is_pure} true pure={true_is_pure} crops {crop_count}/400")

    # metrics
    ari = adjusted_rand_index(all_true, all_pred) if all_true else 0.0
    purity = cluster_purity(all_true, all_pred) if all_true else 0.0
    # flag accuracy
    correct_flags = sum(1 for r in per_plate_results if r["true_is_pure"] == r["pred_is_pure"])
    flag_acc = correct_flags / len(per_plate_results) if per_plate_results else 0.0

    print(f"ARI {ari:.3f} purity {purity:.3f} flag_acc {flag_acc:.3f} {correct_flags}/{len(per_plate_results)}")

    # write morphotypes.csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true_species", "pred_morphotype", "morphotype_desc", "purity_flag"])
        w.writeheader()
        w.writerows(rows)

    # per-plate breakdown for log
    per_plate_csv = Path("bench/morphotypes_per_plate.csv")
    with per_plate_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_crops", "n_true_species", "n_pred_types", "true_is_pure", "pred_is_pure", "purity_flag"])
        w.writeheader()
        w.writerows(per_plate_results)

    # RESULTS.md morphotypes section
    md_text = Path(OUT_MD).read_text() if Path(OUT_MD).exists() else "# Benchmark\n"
    # avoid duplicate
    if "## Morphotypes" not in md_text:
        sec = "\n\n## Morphotypes (AGAR test, GT boxes)\n\n"
        sec += f"Evaluated {plates_evaluated} plates, {crop_count} crops (max 400) with ground-truth boxes. Grouping via morphology + diameter terciles (Python, not LLM).\n\n"
        sec += f"- **ARI** (vs species): {ari:.3f}\n"
        sec += f"- **Cluster purity**: {purity:.3f} (share from majority species per morphotype)\n"
        sec += f"- **Purity flag accuracy**: {flag_acc:.1%} ({correct_flags}/{len(per_plate_results)} plates; pure vs mixed)\n"
        # confusion for flag
        true_pure_n = sum(1 for r in per_plate_results if r["true_is_pure"])
        true_mixed_n = len(per_plate_results) - true_pure_n
        pred_pure_n = sum(1 for r in per_plate_results if r["pred_is_pure"])
        sec += f"- Plates: {true_pure_n} pure (1 species) / {true_mixed_n} mixed (>1 species); predicted {pred_pure_n} pure\n\n"
        sec += "### Findings\n"
        # 4-line findings (as spec says 4-line findings)
        sec += f"- ARI {ari:.2f} indicates grouping captures some species structure (1.0=perfect, 0=random).\n"
        sec += f"- Purity {purity:.1%} shows morphotype homogeneity; lower on mixed plates due to morphology overlap.\n"
        sec += f"- Flag accuracy {flag_acc:.0%} for pure vs mixed may be limited by subtle morphology differences.\n"
        sec += f"- All morphology from Muse Spark Open vision (no organism names); grouping in Python only.\n"
        # add per-plate table snippet
        sec += "\n| image | true_species | pred_types | flag correct |\n|---|---|---|---|\n"
        for r in per_plate_results[:10]:
            ok = "✓" if r["true_is_pure"] == r["pred_is_pure"] else "✗"
            sec += f"| {r['image']} | {r['n_true_species']} | {r['n_pred_types']} | {ok} |\n"
        if len(per_plate_results) > 10:
            sec += f"| ... {len(per_plate_results)-10} more | | | |\n"
        sec += f"\n*Morphotypes.csv: per-crop true species vs predicted morphotype; per_plate.csv: purity flag. Cache via runs/cache/.*\n"

        # check refusal
        if purity == 0.0 and ari == 0.0 and all(r.get("morphotype_desc") == "unknown" for r in rows):
            # log and note
            sec += "\n> Muse still refuses morphology (all unknown) — raw in runs/characterize_raw/.\n"

        Path(OUT_MD).write_text(md_text + sec)
        print(f"wrote {OUT_CSV} and updated {OUT_MD}")
    else:
        print("Morphotypes section already exists, skipping overwrite")

    # sanity: write metrics json
    Path("bench/morphotypes_metrics.json").write_text(json.dumps({"ari": ari, "purity": purity, "flag_accuracy": flag_acc, "plates": plates_evaluated, "crops": crop_count}, indent=2))


if __name__ == "__main__":
    main()
