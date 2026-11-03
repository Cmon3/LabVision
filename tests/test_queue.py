import json, shutil
from pathlib import Path
from fastapi.testclient import TestClient
from server import app
from labvision.runlog import save_run

def _make_run(runs_dir: Path, run_id, count=5, is_blurry=False, glare=False, experimental=False, retry=False, location="", alert_limit=None, plate_id=None, mean_confidence=None):
    p=runs_dir/run_id
    p.mkdir(parents=True, exist_ok=True)
    results=[{"index":i,"area_px":100,"diameter_px":10} for i in range(count)]
    status = "Re-photo" if (is_blurry or glare) else ("TNTC" if count>300 else ("Empty" if count==0 else "OK"))
    save_run(runs_dir, run_id, inputs={"request":"count colonies", "plate_id": plate_id or run_id, "location": location, **({"alert_limit":alert_limit} if alert_limit is not None else {})}, model_ids={"sam":"sam-3.1"}, params={"experimental":experimental, "retry":retry, "is_blurry":is_blurry, "glare":glare, "status":status, "mean_confidence":mean_confidence, "low_confidence": (mean_confidence is not None and mean_confidence <0.5)}, results=results)
    inbox={}
    if plate_id: inbox["plate_id"]=plate_id
    if location: inbox["location"]=location
    if alert_limit is not None: inbox["alert_limit"]=alert_limit
    (p/"inbox.json").write_text(json.dumps(inbox))
    trace=[
        {"name":"check_quality","status":"warn" if is_blurry or glare else "ok","decided_by":"rule","is_blurry":is_blurry,"glare":glare},
        {"name":"segment","status":"ok","decided_by":"rule","retry":retry,"boxes":count, "mean_confidence":mean_confidence, "low_confidence": (mean_confidence is not None and mean_confidence <0.5)},
        {"name":"measure","status":"ok","decided_by":"rule","count":count},
    ]
    if experimental:
        trace[1]["experimental"]=True
    (p/"trace.json").write_text(json.dumps(trace))
    (p/"boxes.json").write_text(json.dumps([]))
    from PIL import Image
    Image.new("RGB",(10,10)).save(p/"image.png")
    (p/"masks.png").write_bytes(b"")
    return p

def test_queue_needs_review_and_sort(tmp_path, monkeypatch):
    runs_dir = tmp_path/"runs"; runs_dir.mkdir()
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(runs_dir))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", runs_dir)
    import labvision.queue as qmod; monkeypatch.setattr(qmod, "RUNS_DIR", runs_dir)
    client=TestClient(app)
    _make_run(runs_dir, "run_queue_a_ok", count=10, location="A", plate_id="plateA")
    _make_run(runs_dir, "run_queue_b_blur", count=10, is_blurry=True, plate_id="plateB")
    _make_run(runs_dir, "run_queue_c_tntc", count=350, plate_id="plateC")
    _make_run(runs_dir, "run_queue_d_alert", count=60, alert_limit=50, plate_id="plateD")
    _make_run(runs_dir, "run_queue_e_exp", count=10, experimental=True, plate_id="plateE")
    _make_run(runs_dir, "run_queue_f_retry", count=10, retry=True, plate_id="plateF")
    _make_run(runs_dir, "run_queue_g_lowconf", count=10, plate_id="plateG", mean_confidence=0.3)
    r=client.get("/api/queue")
    assert r.status_code==200
    data=r.json()
    ours=[d for d in data if d["run_id"].startswith("run_queue_")]
    assert len(ours)==7, ours
    m={d["plate_id"]:d for d in ours}
    assert m["plateA"]["needs_review"]==False
    assert m["plateA"]["status"]=="OK"
    assert m["plateB"]["needs_review"]==True and "blurry" in m["plateB"]["reasons"]
    assert m["plateB"]["status"]=="Re-photo"
    assert m["plateC"]["needs_review"]==True and "TNTC" in m["plateC"]["reasons"]
    assert m["plateC"]["status"]=="TNTC"
    assert m["plateD"]["needs_review"]==True and any("alert" in x for x in m["plateD"]["reasons"])
    assert m["plateC"]["status"]=="TNTC"
    assert m["plateE"]["needs_review"]==True and "experimental" in m["plateE"]["reasons"]
    assert m["plateE"]["status"]=="OK"
    assert m["plateF"]["needs_review"]==True and "retry" in m["plateF"]["reasons"]
    assert m["plateG"]["needs_review"]==True and any("low conf" in x for x in m["plateG"]["reasons"])
    assert m["plateG"]["status"]=="OK"
    assert m["plateG"]["mean_confidence"]==0.3
    # sorted needs_review first
    assert ours[0]["needs_review"]==True
    # approve one
    r2=client.post("/api/runs/run_queue_b_blur/approve", json={"reviewer":"Anna"})
    assert r2.status_code==200
    assert r2.json()["reviewed_by"]=="Anna"
    r3=client.get("/api/queue")
    m2={d["plate_id"]:d for d in r3.json() if d["run_id"].startswith("run_queue_")}
    assert m2["plateB"]["reviewed_by"]=="Anna"
    # approve clean
    r4=client.post("/api/queue/approve_clean", json={"reviewer":"Bob"})
    assert r4.status_code==200
    assert "run_queue_a_ok" in r4.json()["approved"]
    r5=client.get("/api/queue")
    m3={d["plate_id"]:d for d in r5.json() if d["run_id"].startswith("run_queue_")}
    assert m3["plateA"]["reviewed_by"]=="Bob"
    # detail endpoint
    r6=client.get("/api/runs/run_queue_a_ok")
    assert r6.status_code==200
    assert r6.json()["run_id"]=="run_queue_a_ok"
    assert "count_label" in r6.json()
    assert r6.json()["status"]=="OK"

def test_inbox_trace_is_rule_only(tmp_path, monkeypatch):
    from labvision.inbox import process_inbox_once, _file_state
    _file_state.clear()
    runs_dir = tmp_path/"runs"; runs_dir.mkdir()
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(runs_dir))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", runs_dir)
    import labvision.inbox as inbox_mod; monkeypatch.setattr(inbox_mod, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(inbox_mod, "_runs_dir", lambda: runs_dir)
    inbox=tmp_path/"inbox"; inbox.mkdir()
    from PIL import Image
    Image.new("RGB",(20,20)).save(inbox/"plateQ.jpg")
    def fake_analyze(image_bytes, request, run_id=None):
        p=runs_dir/run_id
        p.mkdir(parents=True, exist_ok=True)
        from labvision.runlog import save_run
        save_run(runs_dir, run_id, inputs={"request":request}, model_ids={"sam":"sam-3.1"}, params={"status":"OK","is_blurry":False,"glare":False,"retry":False,"experimental":False}, results=[{"index":0}])
        (p/"boxes.json").write_text(json.dumps([]))
        (p/"trace.json").write_text(json.dumps([{"name":"check_quality","status":"ok","decided_by":"rule"},{"name":"segment","status":"ok","decided_by":"rule"}]))
        (p/"image.png").write_bytes(image_bytes)
        (p/"masks.png").write_bytes(b"")
        return {"run_id":run_id,"count":1,"status":"OK"}
    from unittest.mock import patch
    with patch("server._analyze_image", side_effect=fake_analyze):
        res=process_inbox_once(inbox, stability_s=0)
        assert res[0]["run_id"]
        rid=res[0]["run_id"]
        trace=json.loads((runs_dir/rid/"trace.json").read_text())
        assert all(t["decided_by"]=="rule" for t in trace)
        assert not any(t.get("decided_by")=="model" for t in trace)
    _file_state.clear()

def test_queue_status_agrees_with_reasons(tmp_path, monkeypatch):
    runs_dir=tmp_path/"runs"; runs_dir.mkdir()
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(runs_dir))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", runs_dir)
    import labvision.queue as qmod; monkeypatch.setattr(qmod, "RUNS_DIR", runs_dir)
    client=TestClient(app)
    _make_run(runs_dir, "run_q_ok", count=15, plate_id="ok")
    _make_run(runs_dir, "run_q_rephoto", count=15, is_blurry=True, plate_id="rephoto")
    _make_run(runs_dir, "run_q_empty", count=0, plate_id="empty")
    _make_run(runs_dir, "run_q_tntc", count=400, plate_id="tntc")
    r=client.get("/api/queue")
    m={d["plate_id"]:d for d in r.json() if d["plate_id"] in ("ok","rephoto","empty","tntc")}
    assert m["ok"]["status"]=="OK"
    assert m["rephoto"]["status"]=="Re-photo"
    assert m["empty"]["status"]=="Empty"
    assert m["tntc"]["status"]=="TNTC"
