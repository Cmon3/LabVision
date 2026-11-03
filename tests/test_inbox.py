import io, json, tempfile, shutil
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

def test_sidecar():
    td=Path(tempfile.mkdtemp())
    try:
        img=td/"a.jpg"
        Image.new("RGB",(10,10)).save(img)
        (td/"a.json").write_text(json.dumps({"location":"L1","alert_limit":50}))
        sc=read_sidecar(img)
        assert sc["location"]=="L1" and sc["alert_limit"]==50
        # no sidecar
        img2=td/"b.jpg"
        Image.new("RGB",(10,10)).save(img2)
        assert read_sidecar(img2)=={}
    finally:
        shutil.rmtree(td)

def test_process_success_and_move():
    _file_state.clear()
    td=Path(tempfile.mkdtemp())
    inbox=td/"inbox"; inbox.mkdir()
    im=Image.new("RGB",(40,40),(200,180,160))
    im.save(inbox/"plateX.jpg")
    (inbox/"plateX.json").write_text(json.dumps({"location":"Shelf A","alert_limit":25}))
    def fake_analyze(image_bytes, request, run_id=None):
        from labvision.runlog import save_run
        rid=run_id or "run_test_inbox1"
        rdir=Path("runs")/rid; rdir.mkdir(parents=True, exist_ok=True)
        (rdir/"image.png").write_bytes(image_bytes)
        save_run(Path("runs"), rid, inputs={"request":request}, model_ids={"sam":"sam-3.1"}, params={}, results=[{"index":0}])
        (rdir/"boxes.json").write_text(json.dumps([]))
        (rdir/"masks.png").write_bytes(b"x")
        return {"run_id":rid,"count":1,"status":"ok"}
    with patch("server._analyze_image", side_effect=fake_analyze):
        res=process_inbox_once(inbox, stability_s=0)
        assert len(res)==1 and res[0]["run_id"]
        assert (inbox/"processed"/"plateX.jpg").exists()
        assert (inbox/"processed"/"plateX.json").exists()
        # inbox dir should not contain original
        assert not (inbox/"plateX.jpg").exists()
        # run.json enriched
        rid=res[0]["run_id"]
        rec=json.loads((Path("runs")/rid/"run.json").read_text())
        assert rec["inputs"]["plate_id"]=="plateX"
        assert rec["inputs"]["location"]=="Shelf A"
        assert rec["inputs"]["alert_limit"]==25
        # cleanup
        shutil.rmtree(Path("runs")/rid)
    shutil.rmtree(td)

def test_process_failed_moves_to_failed():
    _file_state.clear()
    td=Path(tempfile.mkdtemp())
    inbox=td/"inbox"; inbox.mkdir()
    Image.new("RGB",(20,20)).save(inbox/"bad.jpg")
    def fake_fail(*a,**kw): raise RuntimeError("SAM down")
    with patch("server._analyze_image", side_effect=fake_fail):
        res=process_inbox_once(inbox, stability_s=0)
        assert (inbox/"failed"/"bad.jpg").exists()
        assert (inbox/"failed"/"bad.error.txt").exists()
        assert "SAM down" in (inbox/"failed"/"bad.error.txt").read_text()
    shutil.rmtree(td)

def test_stability_waits_2s():
    _file_state.clear()
    import time as _t
    from labvision.inbox import _is_stable
    td=Path(tempfile.mkdtemp())
    p=td/"img.jpg"; p.write_bytes(b"abc")
    t0=1000.0
    assert _is_stable(p, stability_s=2.0, now_mono=t0) is False
    # same size/mtime within 2s -> not ready
    assert _is_stable(p, stability_s=2.0, now_mono=t0+1.0) is False
    # after 2s -> ready
    assert _is_stable(p, stability_s=2.0, now_mono=t0+2.1) is True
    # size change resets
    p.write_bytes(b"abcd")
    assert _is_stable(p, stability_s=2.0, now_mono=t0+2.2) is False
    shutil.rmtree(td)
    _file_state.clear()

def test_ignores_non_image():
    _file_state.clear()
    td=Path(tempfile.mkdtemp())
    inbox=td/"inbox"; inbox.mkdir()
    (inbox/"note.txt").write_text("hi")
    Image.new("RGB",(10,10)).save(inbox/"ok.png")
    def fake_analyze(image_bytes, request, run_id=None):
        from labvision.runlog import save_run
        rid=run_id or "run_test_ignore"
        rdir=Path("runs")/rid; rdir.mkdir(parents=True, exist_ok=True)
        save_run(Path("runs"), rid, inputs={}, model_ids={}, params={}, results=[])
        (rdir/"boxes.json").write_text("[]"); (rdir/"masks.png").write_bytes(b"x")
        (rdir/"image.png").write_bytes(image_bytes)
        return {"run_id":rid,"count":0,"status":"ok"}
    with patch("server._analyze_image", side_effect=fake_analyze):
        res=process_inbox_once(inbox, stability_s=0)
        assert len(res)==1  # only ok.png
        assert (inbox/"processed"/"ok.png").exists()
        assert (inbox/"note.txt").exists()  # ignored
        shutil.rmtree(Path("runs")/res[0]["run_id"])
    shutil.rmtree(td)
