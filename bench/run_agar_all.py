"""AGAR: count all annotated plates (dev + test) with the shipped config, one row per plate.

Same selection, split and pipeline as bench/run_bench.py (SAM 3.1 + Python measure, prompt and
tiling from labvision/config.py). The headline number stays the TEST split: the dev plates were
used to choose the prompt, so dev and "all" are reported for context only.

    python bench/run_agar_all.py      # uses the on-disk SAM cache when present
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from labvision.config import AGAR_PROMPT, AGAR_TILING  # noqa: E402
from run_bench import AGAR_ROOT, run_one  # noqa: E402

OUT = Path("bench/results_agar_all.csv")


def _plates() -> list[tuple[Path, int, str]]:
    found = []
    for p in AGAR_ROOT.rglob("*.jpg"):
        js = p.with_suffix(".json")
        if not js.exists():
            continue
        cnt = json.loads(js.read_text()).get("colonies_number")
        if cnt is None or cnt == -1:
            continue
        found.append((p, cnt))
    found.sort(key=lambda x: x[0].name)  # same order and split rule as run_bench.py
    half = len(found) // 2
    return [(p, c, "dev" if i < half else "test") for i, (p, c) in enumerate(found)]


def summary(rows: list[dict]) -> str:
    n = len(rows)
    errs = [r["err"] for r in rows]
    pcts = [r["pct_err"] for r in rows]
    exact = sum(1 for e in errs if e == 0)
    within1 = sum(1 for e in errs if e <= 1)
    within10 = sum(1 for p in pcts if p <= 10)
    return (f"n={n} | MAE {sum(errs)/n:.2f} | mean % err {sum(pcts)/n:.1f}% | exact {exact}/{n} | "
            f"within 1 colony {within1}/{n} | within 10% {within10}/{n}")


def main() -> None:
    rows = []
    for path, true_c, split in _plates():
        r = run_one(path, AGAR_PROMPT, true_c, tiling=AGAR_TILING)
        r["split"] = split
        r["folder"] = path.parent.name
        rows.append(r)
        print(f"{split:4} {path.parent.name:16} {path.name:10} true={true_c:4} pred={r['pred']:4} {'cached' if r['cached'] else 'API'}")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "folder", "image", "true", "pred", "err", "pct_err"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print()
    for name in ("test", "dev"):
        print(f"{name:4}: {summary([r for r in rows if r['split'] == name])}")
    print(f"all : {summary(rows)}")
    print(f"\nwrote {OUT}  (prompt {AGAR_PROMPT!r}, tiling {AGAR_TILING})")


if __name__ == "__main__":
    main()
