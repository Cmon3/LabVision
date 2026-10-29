"""T8 benchmark: SAM 3.1 vs ground truth on BBBC005 (F1) and AGAR."""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from bench.datasets import list_bbbc005, list_image_files
from labvision.measure import measure_masks
from labvision.segment import segment

# ponytail: keep one file, no extra abstractions
OUT_CSV = Path("bench/results.csv")
OUT_CSV_AGAR = Path("bench/results_agar.csv")
OUT_MD = Path("bench/RESULTS.md")
BBBC_ROOT = Path("data/bbbc005/BBBC005_v1_images")
AGAR_ROOT = Path("data/agar_demo")


def pick_bbbc005(n=200, seed=0):
    samples = list_bbbc005(BBBC_ROOT, plane="F1")
    print(f"BBBC005 F1: {len(samples)} images")
    from collections import defaultdict

    groups = defaultdict(list)
    for s in samples:
        groups[s.count].append(s)
    print("counts:", sorted(groups.keys()))
    counts = sorted(groups.keys())
    per = n // len(counts)
    rem = n % len(counts)
    rnd = random.Random(seed)
    picked = []
    for i, c in enumerate(counts):
        k = per + (1 if i < rem else 0)
        grp = groups[c]
        rnd.shuffle(grp)
        picked.extend(grp[:k])
    rnd.shuffle(picked)
    print(f"picked {len(picked)} across {len(counts)} counts")
    return picked


def _run_with_prompt(path: Path, prompt: str):
    # wrapper with fallback for tiff
    return segment(path.read_bytes(), prompt)


def run_one(path: Path, prompt: str, true_count: int | None):
    seg = segment(path.read_bytes(), prompt)
    masks = seg.decoded_masks()
    rows = measure_masks(masks)
    pred = len(rows)
    err = abs(pred - true_count) if true_count is not None else None
    pct = (err / true_count * 100) if true_count and true_count > 0 else None
    return {"image": path.name, "true": true_count, "pred": pred, "err": err, "pct_err": pct, "w": seg.width, "h": seg.height, "cached": seg.cached}


def write_csv(rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true", "pred", "err", "pct_err"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["image", "true", "pred", "err", "pct_err"]})


def main(limit: int | None = None):
    # BBBC005
    picked = pick_bbbc005(limit or 200)
    rows = []
    # resume: load existing csv if any
    done = set()
    if OUT_CSV.exists():
        import csv as _csv

        with OUT_CSV.open() as f:
            for row in _csv.DictReader(f):
                done.add(row["image"])
        print(f"resume: {len(done)} already done")
    for i, s in enumerate(picked):
        if s.path.name in done:
            # reload from cache still needed for metrics but skip API
            r = run_one(s.path, "cell", s.count)
            rows.append(r)
            continue
        try:
            r = run_one(s.path, "cell", s.count)
        except Exception as e:
            print(f"ERR {s.path.name}: {e}")
            r = {"image": s.path.name, "true": s.count, "pred": 0, "err": s.count, "pct_err": 100.0, "w": 0, "h": 0}
        rows.append(r)
        print(f"[{i+1}/{len(picked)}] {s.path.name} true={r['true']} pred={r['pred']} err={r['err']} cached={r.get('cached')}")
        # incremental save every 10
        if (i + 1) % 10 == 0:
            write_csv(rows, OUT_CSV)
    errs = [r["err"] for r in rows if r["err"] is not None]
    pcts = [r["pct_err"] for r in rows if r["pct_err"] is not None]
    mae = sum(errs) / len(errs) if errs else 0
    mpe = sum(pcts) / len(pcts) if pcts else 0
    print(f"BBBC005 MAE {mae:.2f} MPE {mpe:.1f}% n={len(rows)} cost ~${len([r for r in rows if not r.get('cached')])*0.0025:.3f}")
    write_csv(rows, OUT_CSV)

    # AGAR
    agar_rows = []
    # collect all AGAR jsons recursively
    for root, _, files in AGAR_ROOT.walk() if hasattr(AGAR_ROOT, "walk") else []:
        pass
    # fallback: glob
    import glob

    agar_images = []
    for ext in ("*.jpg", "*.png"):
        agar_images.extend(Path(AGAR_ROOT).rglob(ext))
    print(f"AGAR images found {len(agar_images)}")
    for img in sorted(agar_images):
        js = img.with_suffix(".json")
        if not js.exists():
            continue
        try:
            data = json.loads(js.read_text())
            true_c = data.get("colonies_number")
            if true_c is None:
                continue
            r = run_one(img, "bacterial colony", true_c)
            # also try "colonies" if first gave 0? keep "bacterial colony" per spec
            agar_rows.append(r)
            print(f"AGAR {img.name} true={true_c} pred={r['pred']}")
        except Exception as e:
            print(f"AGAR {img.name} err {e}")
    if agar_rows:
        write_csv(agar_rows, OUT_CSV_AGAR)
        ae = [r["err"] for r in agar_rows if r["err"] is not None]
        ap = [r["pct_err"] for r in agar_rows if r["pct_err"] is not None]
        print(f"AGAR MAE {sum(ae)/len(ae):.2f} MPE {sum(ap)/len(ap):.1f}%")

    # RESULTS.md
    out = OUT_MD
    out.parent.mkdir(parents=True, exist_ok=True)
    # build table
    def table(title, rows, mae, mpe):
        lines = [f"## {title}", "", f"MAE {mae:.2f} | mean % err {mpe:.1f}%", "", "| image | true | pred | err | %err |", "|---|---:|---:|---:|---:|"]
        for r in rows[:20]:
            pct = f"{r['pct_err']:.1f}" if r["pct_err"] is not None else "-"
            lines.append(f"| {r['image']} | {r['true']} | {r['pred']} | {r['err']} | {pct} |")
        if len(rows) > 20:
            lines.append(f"| ... {len(rows)-20} more | | | | |")
        return "\n".join(lines)

    errs_b = [r["err"] for r in rows if r["err"] is not None]
    pcts_b = [r["pct_err"] for r in rows if r["pct_err"] is not None]
    mae_b = sum(errs_b) / len(errs_b) if errs_b else 0
    mpe_b = sum(pcts_b) / len(pcts_b) if pcts_b else 0
    md = [f"# Benchmark {Path(__file__).name}", "", f"Prompt BBBC005: `cell`, AGAR: `bacterial colony`", ""]
    md.append(table(f"BBBC005 F1 (n={len(rows)})", rows, mae_b, mpe_b))
    if agar_rows:
        ae = [r["err"] for r in agar_rows if r["err"] is not None]
        ap = [r["pct_err"] for r in agar_rows if r["pct_err"] is not None]
        md.append("")
        md.append(table(f"AGAR (n={len(agar_rows)})", agar_rows, sum(ae)/len(ae) if ae else 0, sum(ap)/len(ap) if ap else 0))
    else:
        md.append("\nAGAR: no annotations found")
    md.append("\n*Counts from SAM 3.1 `sam-3.1` via `segment` + `measure`; cached by image hash.*")
    out.write_text("\n".join(md))
    print(f"wrote {out} and {OUT_CSV}")


if __name__ == "__main__":
    main()
