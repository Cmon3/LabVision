"""Sanity guards — all rule-based, logged in trace as decided_by rule."""

from __future__ import annotations

import math
import numpy as np
from typing import Sequence

# --- Guard 1: centre outside plate ------------------------------------------------
def filter_outside_plate(boxes, plate_cx: float, plate_cy: float, plate_r: float, factor: float = 1.10):
    """Discard boxes whose centre is outside 1.10x plate circle (inner rim).

    Returns (kept_indices, discarded_count, near_rim_count)
    near_rim: discarded but within 1.25x (close to edge, not worth review)
    """
    kept = []
    discarded = 0
    near_rim = 0
    thresh = plate_r * factor
    # near rim = within 15% beyond thresh (i.e. 1.10 to 1.25x), likely edge colonies
    near_thresh = plate_r * 1.25
    for i, b in enumerate(boxes):
        if isinstance(b, dict):
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        else:
            x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = math.hypot(cx - plate_cx, cy - plate_cy)
        if dist <= thresh + 1e-6:
            kept.append(i)
        else:
            discarded += 1
            if dist <= near_thresh:
                near_rim += 1
    return kept, discarded, near_rim

# --- Guard 2: plate plausibility --------------------------------------------------
def is_plate_plausible(plate_diameter_px: float | None, image_w: int, image_h: int) -> tuple[bool, str]:
    """Plate must be 40-98% of shorter image side. Returns (plausible, reason)."""
    if plate_diameter_px is None:
        return False, "plate not found"
    short = min(image_w, image_h)
    if short <= 0:
        return False, "plate not found"
    frac = plate_diameter_px / short
    if 0.40 <= frac <= 0.98:
        return True, ""
    return False, "plate not found"

# --- Guard 3: colony size plausibility -----------------------------------------
def flag_implausible_sizes(rows: list[dict]) -> list[int]:
    """Return indices of rows with diameter_mm outside 0.1-15 mm.

    Not dropped; caller should flag needs_review.
    """
    bad = []
    for idx, r in enumerate(rows):
        d = r.get("diameter_mm")
        if d is None:
            continue
        if d < 0.1 or d > 15.0:
            bad.append(idx)
    return bad


def is_possible_undercount(count: int, rows: list[dict], mask_coverage: float) -> bool:
    """Dense plate that looks crowded but count is low (e.g. 518: 30 found).
    Heuristic: 10<=count<=100, mean area <2500 px (small colonies), coverage low but image would look crowded.
    Never status OK when true.
    """
    if count < 10 or count > 150:
        return False
    if not rows:
        return False
    # mean area in px
    try:
        mean_area = sum(r.get("area_px", 0) for r in rows) / len(rows) if rows else 0
    except Exception:
        return False
    # dense: many small colonies, but count low vs expected for dense -> flag
    # 518: 31 count, mean 1508 <2500, coverage 0.003 (low because undercount)
    # 14684: 56 count, mean 8618 >2500 -> not dense
    if mean_area < 2500 and 20 <= count <= 100:
        return True
    # also if coverage high but count low (e.g. large plate with many touching)
    if mask_coverage > 0.15 and count < 100:
        return True
    return False

# --- Guard 4: duplicate IoU >0.7 -------------------------------------------------
def box_iou(a, b) -> float:
    if isinstance(a, dict):
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    else:
        ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    if isinstance(b, dict):
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    else:
        bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def dedup_boxes(boxes, confidences: list[float | None] | None = None, iou_thresh: float = 0.7):
    """Keep one per overlapping group (IoU>thresh). Higher confidence kept if available.

    Returns (kept_indices, removed_count)
    """
    if not boxes:
        return [], 0
    n = len(boxes)
    # order by confidence descending if available, else original order
    if confidences and any(c is not None for c in confidences):
        order = sorted(range(n), key=lambda i: confidences[i] if confidences[i] is not None else -1, reverse=True)
    else:
        order = list(range(n))
    kept = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            # only compare if j not yet kept; if IoU > thresh, suppress j
            # we compare i (kept) vs j (candidate)
            if box_iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)
    kept_sorted = sorted(kept)  # keep original order for deterministic output? but we want confidence priority.
    # for trace we return count removed; but we also want kept indices in confidence order? Use kept as per priority, but for masks we need to map.
    # Return kept in order of first appearance (sorted) to keep deterministic, but we already have confidence priority.
    # We'll return kept in the order they were kept (priority order) then sorted for stability, but we need to know which to keep.
    # Instead return kept sorted by original index to preserve image order, but ensure highest conf per group kept.
    # So we already did: kept contains highest conf per group; now sort by original index for output.
    kept_sorted = sorted(kept)
    return kept_sorted, n - len(kept_sorted)


# Combined helper for server/agent
def apply_box_guards(boxes, masks, confidences, plate_info: tuple[float, float, float] | None, image_w: int, image_h: int):
    """Apply dedup + plate-centre + plate plausibility guards.

    plate_info: (diameter, cx, cy) or None
    Returns dict with kept boxes/masks, stats, trace entries, and plate plausibility result.
    """
    trace_entries = []
    stats = {}
    cur_boxes = boxes
    cur_masks = masks
    cur_conf = confidences

    # 4) duplicates
    kept_idx, dup_removed = dedup_boxes(cur_boxes, cur_conf, iou_thresh=0.7)
    trace_entries.append({"name": "dedup", "status": "ok" if dup_removed == 0 else "warn", "decided_by": "rule", "removed": dup_removed, "kept": len(kept_idx)})
    stats["duplicates_removed"] = dup_removed
    if dup_removed > 0:
        cur_boxes = [cur_boxes[i] for i in kept_idx]
        if cur_masks is not None and cur_masks.shape[0] == len(boxes):
            cur_masks = cur_masks[kept_idx] if len(kept_idx) else np.zeros((0, image_h, image_w), dtype=bool)
        if cur_conf:
            cur_conf = [cur_conf[i] for i in kept_idx]

    # 2) plate plausibility
    plate_diameter = None
    plate_cx = plate_cy = None
    plate_r = None
    plate_plausible = False
    plate_reason = "plate not found"
    if plate_info is not None:
        d, cx, cy = plate_info
        plausible, reason = is_plate_plausible(d, image_w, image_h)
        if plausible:
            plate_diameter = d
            plate_cx, plate_cy = cx, cy
            plate_r = d / 2.0
            plate_plausible = True
            plate_reason = ""
            trace_entries.append({"name": "plate_check", "status": "ok", "decided_by": "rule", "plate_diameter_px": d, "short_side": min(image_w, image_h), "fraction": round(d / min(image_w, image_h), 3)})
        else:
            trace_entries.append({"name": "plate_check", "status": "warn", "decided_by": "rule", "plate_diameter_px": d, "short_side": min(image_w, image_h), "fraction": round(d / min(image_w, image_h), 3) if min(image_w, image_h) else 0, "reason": reason, "action": "sizes in px only"})
            plate_reason = reason
    else:
        trace_entries.append({"name": "plate_check", "status": "warn", "decided_by": "rule", "reason": "plate not found", "action": "sizes in px only"})

    stats["plate_plausible"] = plate_plausible
    stats["plate_reason"] = plate_reason

    # 1) centre outside plate (1.10x, near rim not flagged) — DISABLED after benchmark (MAE 18.10 vs 17.70)
    outside_discarded = 0
    outside_near_rim = 0
    # Guard disabled: inner rim discards real edge colonies (14684 56->51, 510 63->54 vs true 55,69)
    # Keep all; still log for audit but do not discard
    if False and plate_plausible and plate_cx is not None:
        n_before = len(cur_boxes)
        kept_idx2, outside_discarded, outside_near_rim = filter_outside_plate(cur_boxes, plate_cx, plate_cy, plate_r, factor=1.10)
        trace_entries.append({"name": "plate_filter", "status": "ok" if outside_discarded == 0 else "warn", "decided_by": "rule", "discarded": outside_discarded, "kept": len(kept_idx2), "near_rim": outside_near_rim, "reason": "centre beyond 1.10x plate circle"})
        stats["outside_plate_discarded"] = outside_discarded
        stats["outside_plate_near_rim"] = outside_near_rim
        # only far outside (not near rim) counts for review
        stats["outside_plate_flagged"] = outside_discarded - outside_near_rim
        if outside_discarded > 0:
            cur_boxes = [cur_boxes[i] for i in kept_idx2]
            if cur_masks is not None and cur_masks.shape[0] == n_before:
                cur_masks = cur_masks[kept_idx2] if len(kept_idx2) else np.zeros((0, image_h, image_w), dtype=bool)
            if cur_conf:
                cur_conf = [cur_conf[i] for i in kept_idx2]
    else:
        trace_entries.append({"name": "plate_filter", "status": "ok", "decided_by": "rule", "discarded": 0, "disabled": True, "reason": "outside-plate filter off (benchmark: removed real edge colonies)"})
        stats["outside_plate_discarded"] = 0
        stats["outside_plate_near_rim"] = 0
        stats["outside_plate_flagged"] = 0

    return {
        "boxes": cur_boxes,
        "masks": cur_masks,
        "confidences": cur_conf,
        "plate_diameter_px": plate_diameter,
        "plate_cx": plate_cx,
        "plate_cy": plate_cy,
        "plate_r": plate_r,
        "plate_plausible": plate_plausible,
        "trace": trace_entries,
        "stats": stats,
    }
