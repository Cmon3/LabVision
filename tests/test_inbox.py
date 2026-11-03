import io, json, shutil, os
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np

from labvision.inbox import read_plate_id, read_sidecar, process_inbox_once, _file_state

def _img_bytes(w=50,h=50):
    im=Image.new("RGB",(w,h),(180,180,160))
    buf=io.BytesIO(); im.save(buf, format="PNG"); return buf.getvalue()

def test_plate_id_fallback():
    p=Path("/tmp/fake_plate123.jpg")
    assert read_plate_id(p)=="fake_plate123"

def test_sidecar(tmp_path):
    # tmp_path is pytest's temp, use it for inbox files (isolated)
    td=tmp_path/"sidecar_test"; td.mkdir()
    img=td/"a.jpg"
    Image.new("RGB",(10,10)).save(img)
    (td/"a.json").write_text(json.dumps({"location":"L1","alert_limit":50}))
    sc=read_sidecar(img)
    assert sc["location"]=="L1" and sc["alert_limit"]==50
    img2=td/"b.jpg"
    Image.new("RGB",(10,10)).save(img2)
    assert read_sidecar(img2)=={}

def test_process_success_and_move(tmp_path, monkeypatch):
    _file_state.clear()
    runs_dir = tmp_path/"runs"
    runs_dir.mkdir()
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(runs_dir))
    # patch server.RUNS_DIR as well for direct access
    import server as srv
    monkeypatch.setattr(srv, "RUNS_DIR", runs_dir)
    import labvision.inbox as inbox_mod
    monkeypatch.setattr(inbox_mod, "RUNS_DIR", runs_dir)
    inbox=tmp_path/"inbox"; inbox.mkdir()
    im=Image.new("RGB",(40,40),(200,180,160))
    im.save(inbox/"plateX.jpg")
    (inbox/"plateX.json").write_text(json.dumps({"location":"Shelf A","alert_limit":25}))
    def fake_analyze(image_bytes, request, run_id=None):
        from labvision.runlog import save_run
        rid=run_id or "run_test_inbox1"
        rdir=runs_dir/rid; rdir.mkdir(parents=True, exist_ok=True)
        (rdir/"image.png").write_bytes(image_bytes)
        save_run(runs_dir, rid, inputs={"request":request}, model_ids={"sam":"sam-3.1"}, params={}, results=[{"index":0}])
        (rdir/"boxes.json").write_text(json.dumps([]))
        (rdir/"masks.png").write_bytes(b"x")
        return {"run_id":rid,"count":1,"status":"OK"}
    with patch("server._analyze_image", side_effect=fake_analyze):
        res=process_inbox_once(inbox, stability_s=0)
        assert len(res)==1 and res[0]["run_id"]
        assert (inbox/"processed"/"plateX.jpg").exists()
        assert (inbox/"processed"/"plateX.json").exists()
        assert not (inbox/"plateX.jpg").exists()
        rid=res[0]["run_id"]
        rec=json.loads((runs_dir/rid/"run.json").read_text())
        assert rec["inputs"]["plate_id"]=="plateX"
        assert rec["inputs"]["location"]=="Shelf A"
        assert rec["inputs"]["alert_limit"]==25
    _file_state.clear()

def test_process_failed_moves_to_failed(tmp_path, monkeypatch):
    _file_state.clear()
    runs_dir = tmp_path/"runs"; runs_dir.mkdir()
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(runs_dir))
    inbox=tmp_path/"inbox"; inbox.mkdir()
    Image.new("RGB",(20,20)).save(inbox/"bad.jpg")
    def fake_fail(*a,**kw): raise RuntimeError("SAM down")
    with patch("server._analyze_image", side_effect=fake_fail):
        res=process_inbox_once(inbox, stability_s=0)
        assert (inbox/"failed"/"bad.jpg").exists()
        assert (inbox/"failed"/"bad.error.txt").exists()
        assert "SAM down" in (inbox/"failed"/"bad.error.txt").read_text()
    _file_state.clear()

def test_stability_waits_2s(tmp_path):
    _file_state.clear()
    from labvision.inbox import _is_stable
    p=tmp_path/"img.jpg"; p.write_bytes(b"abc")
    t0=1000.0
    assert _is_stable(p, stability_s=2.0, now_mono=t0) is False
    assert _is_stable(p, stability_s=2.0, now_mono=t0+1.0) is False
    assert _is_stable(p, stability_s=2.0, now_mono=t0+2.1) is True
    p.write_bytes(b"abcd")
    assert _is_stable(p, stability_s=2.0, now_mono=t0+2.2) is False
    _file_state.clear()

def test_ignores_non_image(tmp_path, monkeypatch):
    _file_state.clear()
    runs_dir = tmp_path/"runs"; runs_dir.mkdir()
    monkeypatch.setenv("LABVISION_RUNS_DIR", str(runs_dir))
    import server as srv; monkeypatch.setattr(srv, "RUNS_DIR", runs_dir)
    import labvision.inbox as inbox_mod; monkeypatch.setattr(inbox_mod, "RUNS_DIR", runs_dir)
    inbox=tmp_path/"inbox"; inbox.mkdir()
    (inbox/"note.txt").write_text("hi")
    Image.new("RGB",(10,10)).save(inbox/"ok.png")
    def fake_analyze(image_bytes, request, run_id=None):
        from labvision.runlog import save_run
        rid=run_id or "run_test_ignore"
        rdir=runs_dir/rid; rdir.mkdir(parents=True, exist_ok=True)
        save_run(runs_dir, rid, inputs={}, model_ids={}, params={}, results=[])
        (rdir/"boxes.json").write_text("[]"); (rdir/"masks.png").write_bytes(b"x")
        (rdir/"image.png").write_bytes(image_bytes)
        return {"run_id":rid,"count":0,"status":"Empty"}
    with patch("server._analyze_image", side_effect=fake_analyze):
        res=process_inbox_once(inbox, stability_s=0)
        assert len(res)==1
        assert (inbox/"processed"/"ok.png").exists()
        assert (inbox/"note.txt").exists()
    _file_state.clear()
