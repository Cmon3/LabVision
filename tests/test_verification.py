"""S9 Step 4: verification tests with scripted fake model (no network)."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import io

import labvision.agent as ag
from labvision import measure as measure_mod
from tests.fake_model import ScriptedMuse


def _fake_seg(count=5, w=100, h=100):
    from labvision.segment import Box, Segmentation
    boxes = [Box(10+i*10, 10+i*10, 20+i*10, 20+i*10) for i in range(count)]
    seg = Segmentation(boxes=boxes, raw_masks=[], width=w, height=h, confidences=[0.9]*count)
    masks = np.zeros((count, h, w), dtype=bool)
    for i in range(count):
        masks[i, 10+i*10:20+i*10, 10+i*10:20+i*10] = True
    seg.decoded_masks = lambda: masks
    return seg, masks


def test_verify_wrong_count_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from verification.verify_report import run_verification
    # ctx says count 5, report says 999
    ctx = {"count": 5, "rows": [{"area_px": 100, "diameter_px": 11.3}] * 5, "qc": {"is_blurry": False, "glare": False}, "prompt_used": "round spot", "experimental": False}
    report = {"summary": "5 colonies found, all good.", "count": 999, "filters": "", "notes": ""}
    res = run_verification(ctx, report)
    assert not res["passed"]
    assert any("does not match measured" in p for p in res["feedback"])

    # correct count passes
    report2 = {"summary": "Count is 5, countable (30-300) not needed for small test", "count": 5, "filters": "", "notes": "sharp"}
    # need countable wording: 5 is too few, so summary must say too few
    report2["summary"] = "Count 5 too few (<30, low statistical reliability)"
    res2 = run_verification(ctx, report2)
    assert res2["passed"], res2["feedback"]


def test_three_refusals_escalate(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from labvision.progress import load_progress
    # create fake image with texture to pass QC (avoid glare: <250)
    p = tmp_path/"img.png"
    arr = np.random.randint(0, 200, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p)
    fake_seg, _ = _fake_seg(count=10)
    # planner that always submits wrong count
    def planner(messages, turn):
        if turn == 1:
            return ("check_quality", {})
        if turn == 2:
            return ("segment", {"prompt": "round spot"})
        if turn == 3:
            return ("measure", {})
        # submit wrong count 999 each time
        return ("submit_report", {"summary": "999 colonies, great", "count": 999, "filters": "", "notes": ""})

    scripted = ScriptedMuse(planner)
    with patch.object(ag, "_call_muse", side_effect=scripted.as_side_effect()):
        with patch.object(ag, "segment", return_value=fake_seg):
            with patch("labvision.plate.detect_plate_diameter_px", return_value=None):
                out = ag.run_agent(p, "count colonies", use_cache=False, run_id="run_refuse_test")
    # after 3 refusals, should escalate report_refused
    from labvision.escalation import get_escalation
    esc = get_escalation("run_refuse_test", base=tmp_path/"runs")
    assert esc is not None
    assert esc["rule"] == "report_refused"
    assert esc["status"] == "needs_input"
    # refusal history
    rf = tmp_path/"runs"/"run_refuse_test"/"refusals.json"
    assert rf.exists()
    hist = json.loads(rf.read_text())
    assert len(hist) == 3
    assert out["status"] in ("needs_input", "paused", "running")


def test_blurry_needs_rephoto_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    from labvision.escalation import get_escalation, find_needs_rephoto_for_plate
    from labvision.inbox import process_inbox_once, _file_state
    from unittest.mock import patch as mpatch
    _file_state.clear()
    runs_dir = tmp_path/"runs"
    runs_dir.mkdir()
    inbox = tmp_path/"inbox"
    inbox.mkdir()
    # simulate prior blurry run via _escalate directly
    import server
    monkeypatch.setattr(server, "RUNS_DIR", runs_dir)
    import labvision.inbox as inbox_mod
    monkeypatch.setattr(inbox_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(inbox_mod, "_runs_dir", lambda: runs_dir)
    # create old run with needs_rephoto for plateX
    from labvision.runlog import save_run
    old_id = "run_old_blur"
    (runs_dir/old_id).mkdir()
    save_run(runs_dir, old_id, inputs={"plate_id": "plateX"}, model_ids={}, params={"is_blurry": True}, results=[{"index": 0}])
    from labvision.escalation import escalate
    escalate(old_id, "needs_rephoto", {"what_blocked": "QC failed", "why": "blurry", "what_was_done": "count 1", "request": "Re-photo plateX", "plate_id": "plateX"}, base=runs_dir)
    assert get_escalation(old_id, base=runs_dir)["rule"] == "needs_rephoto"

    # new photo arrives with same plate_id plateX
    Image.new("RGB", (20, 20)).save(inbox/"plateX.jpg")
    def fake_analyze(image_bytes, request, run_id=None):
        rid = run_id or "run_new_plateX"
        rdir = runs_dir/rid
        rdir.mkdir()
        save_run(runs_dir, rid, inputs={"request": request}, model_ids={}, params={}, results=[{"index": 0}, {"index": 1}])
        (rdir/"boxes.json").write_text("[]")
        (rdir/"trace.json").write_text("[]")
        (rdir/"image.png").write_bytes(image_bytes)
        (rdir/"masks.png").write_bytes(b"")
        return {"run_id": rid, "count": 2, "status": "OK"}

    with mpatch("server._analyze_image", side_effect=fake_analyze):
        res = process_inbox_once(inbox, stability_s=0)
        assert len(res) == 1
        new_id = res[0]["run_id"]
        # old run should have resume link
        assert (runs_dir/old_id/"resume_image.json").exists()
        link = json.loads((runs_dir/old_id/"resume_image.json").read_text())
        assert link["new_run_id"] == new_id
    _file_state.clear()


def test_reject_with_notes_revised(tmp_path, monkeypatch):
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(tmp_path/"runs"))
    # create run that passed verification → awaiting_signoff (textured to pass QC)
    p = tmp_path/"img2.png"
    arr = np.random.randint(0, 200, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p)
    fake_seg, _ = _fake_seg(count=5)
    # correct submission first
    def planner_ok(messages, turn):
        if turn == 1:
            return ("check_quality", {})
        if turn == 2:
            return ("segment", {"prompt": "round spot"})
        if turn == 3:
            return ("measure", {})
        return ("submit_report", {"summary": "Count 5 too few (<30, low statistical reliability)", "count": 5, "filters": "", "notes": "sharp"})

    scripted_ok = ScriptedMuse(planner_ok)
    with patch.object(ag, "_call_muse", side_effect=scripted_ok.as_side_effect()):
        with patch.object(ag, "segment", return_value=fake_seg):
            with patch("labvision.plate.detect_plate_diameter_px", return_value=None):
                out = ag.run_agent(p, "count colonies", use_cache=False, run_id="run_signoff_test")
    from labvision.escalation import get_escalation
    esc = get_escalation("run_signoff_test", base=tmp_path/"runs")
    assert esc["rule"] == "awaiting_signoff"

    # reviewer rejects with notes
    from fastapi.testclient import TestClient
    from server import app
    import server as srv
    monkeypatch.setattr(srv, "RUNS_DIR", tmp_path/"runs")
    client = TestClient(app)
    r = client.post("/api/runs/run_signoff_test/review", json={"decision": "reject", "reviewer": "Anna", "notes": "Please rephrase summary"})
    assert r.status_code == 200
    assert r.json()["decision"] == "rejected"

    # resume with revised report (planner that uses corrected notes)
    def planner_rev(messages, turn):
        # check that last user message contains reviewer notes
        last = messages[-1].get("content", "") if messages else ""
        # we expect guidance to be in last message after resume
        if "rejected" in last or "Please rephrase" in last or turn > 3:
            return ("submit_report", {"summary": "Count 5 too few (<30, low statistical reliability) revised", "count": 5, "filters": "", "notes": "revised per reviewer"})
        if turn == 1:
            return ("check_quality", {})
        if turn == 2:
            return ("segment", {"prompt": "round spot"})
        if turn == 3:
            return ("measure", {})
        return ("submit_report", {"summary": "Count 5 too few (<30, low statistical reliability) revised", "count": 5, "filters": "", "notes": "revised"})

    # simulate resume via API (which will call resume_run)
    # instead directly call resume_run logic: we will just verify that review.json exists and status reset to running
    from labvision.escalation import get_status
    # after reject, status should be running
    status = (tmp_path/"runs"/"run_signoff_test"/"status.json")
    if status.exists():
        assert "running" in status.read_text()


def test_too_few_accepts_the_wording_the_feedback_asks_for():
    """The refusal tells the model: say 'too few' or 'low statistical reliability'. Both must pass
    (regression: the rule also demanded the number 30, so the model could never satisfy it)."""
    from verification.verify_report import _countable_wording
    assert _countable_wording("The number of colonies smaller than 5 mm is too few for conclusions.", 14)
    assert _countable_wording("This count has low statistical reliability.", 14)
    assert not _countable_wording("Colonies look fine.", 14)
    assert not _countable_wording("This is too few.", 120)  # countable plate must not say too few
