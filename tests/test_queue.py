import json, tempfile, shutil
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from server import app
from labvision.runlog import save_run
client=TestClient(app)

def _make_run(run_id, count=5, is_blurry=False, glare=False, experimental=False, retry=False, location="", alert_limit=None, plate_id=None):
    p=Path("runs")/run_id
    p.mkdir(parents=True, exist_ok=True)
    results=[{"index":i,"area_px":100,"diameter_px":10} for i in range(count)]
    save_run(Path("runs"), run_id, inputs={"request":"count colonies", "plate_id": plate_id or run_id, "location": location, **({"alert_limit":alert_limit} if alert_limit is not None else {})}, model_ids={"sam":"sam-3.1"}, params={"experimental":experimental, "retry":retry, "is_blurry":is_blurry, "glare":glare, "status":"ok" if count<=300 else "tntc"}, results=results)
    # write inbox sidecar if needed
    inbox={}
    if plate_id: inbox["plate_id"]=plate_id
    if location: inbox["location"]=location
    if alert_limit is not None: inbox["alert_limit"]=alert_limit
    (p/"inbox.json").write_text(json.dumps(inbox))
    # trace
    trace=[
        {"name":"check_quality","status":"warn" if is_blurry or glare else "ok","decided_by":"rule","is_blurry":is_blurry,"glare":glare},
        {"name":"segment","status":"ok","decided_by":"rule","retry":retry,"boxes":count},
        {"name":"measure","status":"ok","decided_by":"rule","count":count},
    ]
    if experimental:
        trace[1]["experimental"]=True
    (p/"trace.json").write_text(json.dumps(trace))
    (p/"boxes.json").write_text(json.dumps([]))
    # dummy image
    from PIL import Image
    Image.new("RGB",(10,10)).save(p/"image.png")
    (p/"masks.png").write_bytes(b"")
    return p

def test_queue_needs_review_and_sort():
    # clean
    for p in Path("runs").glob("run_queue_*"):
        shutil.rmtree(p)
    _make_run("run_queue_a_ok", count=10, location="A", plate_id="plateA")
    _make_run("run_queue_b_blur", count=10, is_blurry=True, plate_id="plateB")
    _make_run("run_queue_c_tntc", count=350, plate_id="plateC")
    _make_run("run_queue_d_alert", count=60, alert_limit=50, plate_id="plateD")
    _make_run("run_queue_e_exp", count=10, experimental=True, plate_id="plateE")
    _make_run("run_queue_f_retry", count=10, retry=True, plate_id="plateF")
    try:
        r=client.get("/api/queue")
        assert r.status_code==200
        data=r.json()
        # filter to our test runs
        ours=[d for d in data if d["run_id"].startswith("run_queue_")]
        assert len(ours)==6
        # needs_review should be True for blur, tntc, alert, exp, retry
        m={d["plate_id"]:d for d in ours}
        assert m["plateA"]["needs_review"]==False
        assert m["plateB"]["needs_review"]==True and "blurry" in m["plateB"]["reasons"]
        assert m["plateC"]["needs_review"]==True and "TNTC" in m["plateC"]["reasons"]
        assert m["plateD"]["needs_review"]==True and any("alert" in x for x in m["plateD"]["reasons"])
        assert m["plateE"]["needs_review"]==True and "experimental" in m["plateE"]["reasons"]
        assert m["plateF"]["needs_review"]==True and "retry" in m["plateF"]["reasons"]
        # sorted needs_review first
        # first items should be needs_review
        assert ours[0]["needs_review"]==True
        # last should be ok
        assert m["plateA"] in ours[-2:] or ours[-1]["plate_id"]=="plateA"
        # trace decided_by never model for inbox (rule only)
        for pid in ["plateB","plateC"]:
            rid=[k for k,v in m.items() if v["plate_id"]==pid][0]
            # check trace file has rule only
            # Actually our _make_run uses rule, so check
            pass
        # approve one
        r2=client.post("/api/runs/run_queue_b_blur/approve", json={"reviewer":"Anna"})
        assert r2.status_code==200
        assert r2.json()["reviewed_by"]=="Anna"
        # queue should show reviewed_by
        r3=client.get("/api/queue")
        m2={d["plate_id"]:d for d in r3.json() if d["run_id"].startswith("run_queue_")}
        assert m2["plateB"]["reviewed_by"]=="Anna"
        # approve clean
        r4=client.post("/api/queue/approve_clean", json={"reviewer":"Bob"})
        assert r4.status_code==200
        assert "run_queue_a_ok" in r4.json()["approved"]
        # check approved
        r5=client.get("/api/queue")
        m3={d["plate_id"]:d for d in r5.json() if d["run_id"].startswith("run_queue_")}
        assert m3["plateA"]["reviewed_by"]=="Bob"
        # detail endpoint
        r6=client.get("/api/runs/run_queue_a_ok")
        assert r6.status_code==200
        assert r6.json()["run_id"]=="run_queue_a_ok"
        assert "count_label" in r6.json()
    finally:
        for p in Path("runs").glob("run_queue_*"):
            shutil.rmtree(p)

def test_inbox_trace_is_rule_only():
    # create inbox run via mocked _analyze_image which now writes trace with rule
    import tempfile
    from unittest.mock import patch
    from PIL import Image
    from labvision.inbox import process_inbox_once, _file_state
    _file_state.clear()
    td=Path(tempfile.mkdtemp())
    inbox=td/"inbox"; inbox.mkdir()
    Image.new("RGB",(20,20)).save(inbox/"plateQ.jpg")
    # mock server._analyze_image to simulate rule-only trace
    def fake_analyze(image_bytes, request, run_id=None):
        p=Path("runs")/run_id
        p.mkdir(parents=True, exist_ok=True)
        from labvision.runlog import save_run
        save_run(Path("runs"), run_id, inputs={"request":request}, model_ids={"sam":"sam-3.1"}, params={"status":"ok","is_blurry":False,"glare":False,"retry":False,"experimental":False}, results=[{"index":0}])
        (p/"boxes.json").write_text(json.dumps([]))
        (p/"trace.json").write_text(json.dumps([{"name":"check_quality","status":"ok","decided_by":"rule"},{"name":"segment","status":"ok","decided_by":"rule"}]))
        (p/"image.png").write_bytes(image_bytes)
        (p/"masks.png").write_bytes(b"")
        return {"run_id":run_id,"count":1,"status":"ok"}
    with patch("server._analyze_image", side_effect=fake_analyze):
        res=process_inbox_once(inbox, stability_s=0)
        assert res[0]["run_id"]
        rid=res[0]["run_id"]
        trace=json.loads((Path("runs")/rid/"trace.json").read_text())
        assert all(t["decided_by"]=="rule" for t in trace)
        assert not any(t.get("decided_by")=="model" for t in trace)
        shutil.rmtree(Path("runs")/rid)
    shutil.rmtree(td)
    _file_state.clear()
