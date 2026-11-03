"""Sanity guards — 5 rule-based checks, all logged in trace (decided_by rule)."""
import numpy as np
from unittest.mock import patch
from pathlib import Path
import json, io
from PIL import Image


def test_guard1_outside_plate_discarded(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from labvision.sanity import filter_outside_plate, apply_box_guards
    from labvision.segment import Box

    # plate at (50,50) r=40 in 100x100 image
    boxes = [Box(40,40,50,50, 0.9), Box(90,90,95,95, 0.9), Box(45,45,55,55, 0.95)]  # 2 inside, 1 outside (centre 92.5,92.5 dist ~60 >40)
    kept, discarded = filter_outside_plate(boxes, 50, 50, 40)
    assert discarded == 1
    assert kept == [0, 2]

    # integration via server: outside box should be discarded, trace logged, report mentions it
    from server import _analyze_image
    # create dummy image 100x100
    arr = np.full((100, 100, 3), 180, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO(); pil.save(buf, format="PNG"); img_bytes = buf.getvalue()

    def fake_seg(image_bytes, prompt, *a, **kw):
        from labvision.segment import Segmentation, Box
        boxes = [Box(40,40,50,50,0.9), Box(90,90,98,98,0.9)]
        seg = Segmentation(boxes=boxes, raw_masks=[], width=100, height=100, confidences=[0.9,0.9])
        m = np.zeros((2, 100, 100), dtype=bool)
        m[0, 40:50, 40:50] = True; m[1, 90:98, 90:98] = True
        seg.decoded_masks = lambda: m
        return seg

    def fake_detect_plate(arr):
        return (80.0, 50.0, 50.0)  # diameter 80 (80% of 100, plausible), centre 50,50 r=40

    with patch("server.segment", side_effect=fake_seg):
        with patch("labvision.segment.segment", side_effect=fake_seg):
            with patch("labvision.plate.detect_plate", side_effect=fake_detect_plate):
                with patch("labvision.plate.detect_plate_diameter_px", return_value=80.0):
                    out = _analyze_image(img_bytes, "count colonies", run_id="test_guard1")

    assert out["count"] == 1  # one outside discarded
    trace = out["trace"]
    pf = [t for t in trace if t["name"] == "plate_filter"]
    assert pf and pf[0]["decided_by"] == "rule"
    assert pf[0]["discarded"] == 1
    assert "outside plate" in out["report"].lower()

    # queue shows needs_review reason
    from labvision.queue import get_queue_items
    items = get_queue_items(tmp_path/"runs")
    m = {i["run_id"]: i for i in items}
    assert "outside plate" in m["test_guard1"]["reasons"]


def test_guard2_plate_plausibility(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from labvision.sanity import is_plate_plausible

    # 100x100 image, short side 100, 40-98% is 40-98px
    assert is_plate_plausible(80, 100, 100)[0] is True
    assert is_plate_plausible(30, 100, 100)[0] is False  # 30% <40
    assert is_plate_plausible(99, 100, 100)[0] is False  # >98
    assert is_plate_plausible(None, 100, 100)[0] is False

    # integration: plate too small -> sizes in px only + trace + needs_review
    from server import _analyze_image
    arr = np.full((100, 100, 3), 180, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO(); pil.save(buf, format="PNG"); img_bytes = buf.getvalue()

    def fake_seg2(image_bytes, prompt, *a, **kw):
        from labvision.segment import Segmentation, Box
        b = [Box(10,10,20,20,0.9)]
        return Segmentation(boxes=b, raw_masks=[], width=100, height=100, confidences=[0.9])

    def fake_detect_small(arr):
        return (30.0, 50, 50)  # 30% -> not plausible

    with patch("labvision.segment.segment", side_effect=fake_seg2):
        with patch("labvision.plate.detect_plate", side_effect=fake_detect_small):
            out = _analyze_image(img_bytes, "count colonies", run_id="test_guard2_small")
    # should be px only
    assert "px only" in out["report"]
    assert any(t["name"] == "plate_check" and t["decided_by"] == "rule" and "plate not found" in str(t.get("reason", "")).lower() for t in out["trace"])
    from labvision.queue import get_queue_items
    items = get_queue_items(tmp_path/"runs")
    m = {i["run_id"]: i for i in items}
    assert "plate not found" in m["test_guard2_small"]["reasons"]


def test_guard3_implausible_size_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from labvision.sanity import flag_implausible_sizes
    rows = [{"diameter_mm": 0.05}, {"diameter_mm": 5.0}, {"diameter_mm": 20.0}]
    bad = flag_implausible_sizes(rows)
    assert bad == [0, 2]

    # integration: measure with plate -> implausible flagged in trace, report, queue
    from server import _analyze_image
    arr = np.full((200, 200, 3), 180, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO(); pil.save(buf, format="PNG"); img_bytes = buf.getvalue()

    # need masks that give known diameters: area -> diam_px -> diam_mm
    # With plate 180px =90mm => scale 0.5 mm/px. For diam 0.05mm => diam_px 0.1 => area ~0.008 -> not realistic.
    # Instead we will mock measure to return implausible rows directly via patch
    def fake_seg3(image_bytes, prompt, *a, **kw):
        from labvision.segment import Segmentation, Box
        b = [Box(10,10,20,20,0.9), Box(30,30,40,40,0.9)]
        return Segmentation(boxes=b, raw_masks=[], width=200, height=200, confidences=[0.9,0.9])

    def fake_detect_plate_ok(arr):
        return (180.0, 100, 100)  # plausible 90% of 200

    # patch measure to return one good, one bad
    orig_measure = None
    def fake_measure(masks, plate_diameter_px=None, plate_diameter_mm=90.0):
        # return rows with diameters 5mm and 0.05mm
        return [{"index": 0, "area_px": 100, "diameter_px": 10, "diameter_mm": 5.0, "area_mm2": 25},
                {"index": 1, "area_px": 1, "diameter_px": 1, "diameter_mm": 0.05, "area_mm2": 0.25}]

    with patch("labvision.segment.segment", side_effect=fake_seg3):
        with patch("labvision.plate.detect_plate", side_effect=fake_detect_plate_ok):
            with patch("labvision.measure.measure_masks", side_effect=fake_measure):
                out = _analyze_image(img_bytes, "count colonies", run_id="test_guard3")

    assert any(t["name"] == "size_check" and t["decided_by"] == "rule" and t["implausible_count"] == 1 for t in out["trace"])
    assert "implausible size" in out["report"].lower()
    from labvision.queue import get_queue_items
    items = get_queue_items(tmp_path/"runs")
    m = {i["run_id"]: i for i in items}
    assert "implausible size" in m["test_guard3"]["reasons"]


def test_guard4_duplicate_iou(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from labvision.sanity import dedup_boxes, box_iou
    from labvision.segment import Box

    b1 = {"x1": 10, "y1": 10, "x2": 20, "y2": 20}
    b2 = {"x1": 12, "y1": 12, "x2": 22, "y2": 22}  # IoU ~0.47, not >0.7
    b3 = {"x1": 10, "y1": 10, "x2": 20, "y2": 20}  # identical to b1 => IoU 1.0
    assert box_iou(b1, b2) < 0.7
    assert box_iou(b1, b3) == 1.0

    kept, removed = dedup_boxes([b1, b3], confidences=[0.9, 0.8])
    assert removed == 1
    assert kept == [0]

    # integration via server
    from server import _analyze_image
    arr = np.full((100, 100, 3), 180, dtype=np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO(); pil.save(buf, format="PNG"); img_bytes = buf.getvalue()

    def fake_seg_dup(image_bytes, prompt, *a, **kw):
        from labvision.segment import Segmentation, Box
        boxes = [Box(10,10,20,20,0.9), Box(10,10,20,20,0.8)]
        seg = Segmentation(boxes=boxes, raw_masks=[], width=100, height=100, confidences=[0.9,0.8])
        m = np.zeros((2, 100, 100), dtype=bool)
        m[0, 10:20, 10:20] = True; m[1, 10:20, 10:20] = True
        seg.decoded_masks = lambda: m
        return seg

    with patch("server.segment", side_effect=fake_seg_dup):
        with patch("labvision.segment.segment", side_effect=fake_seg_dup):
            with patch("labvision.plate.detect_plate", return_value=None):
                out = _analyze_image(img_bytes, "count colonies", run_id="test_guard4")

    assert out["count"] == 1
    assert any(t["name"] == "dedup" and t["decided_by"] == "rule" and t["removed"] == 1 for t in out["trace"])
    assert "duplicate" in out["report"].lower()


def test_guard5_verify_all_numbers_and_blocklist(tmp_path):
    from verification.verify_report import run_verification
    # ALL numbers must be checked, even small ones
    ctx = {"count": 5, "rows": [{"area_px": 100, "diameter_px": 11.3, "diameter_mm": 2.0}] * 5, "qc": {"is_blurry": False, "glare": False}, "prompt_used": "round spot", "experimental": False}
    # summary with small number 7 not in measured set should fail even though 7 <=10
    report = {"summary": "Count 5 too few (<30, low statistical reliability) 7 extra", "count": 5, "filters": "", "notes": ""}
    res = run_verification(ctx, report)
    assert not res["passed"]
    assert "7" in res["feedback"][0]

    # generic words species/organism should NOT be blocked now
    report2 = {"summary": "Count 5 too few (<30, low statistical reliability) species present", "count": 5, "filters": "", "notes": ""}
    res2 = run_verification(ctx, report2)
    # should pass (species is allowed now, only real organism names blocked)
    assert res2["passed"], res2["feedback"]

    # real organism names must still be blocked
    report3 = {"summary": "Count 5 too few (<30, low statistical reliability) escherichia detected", "count": 5, "filters": "", "notes": ""}
    res3 = run_verification(ctx, report3)
    assert not res3["passed"]
    assert "escherichia" in res3["feedback"][0].lower()

    # count mismatch still blocked
    report4 = {"summary": "Count 5 too few (<30, low statistical reliability)", "count": 999, "filters": "", "notes": ""}
    res4 = run_verification(ctx, report4)
    assert not res4["passed"]
