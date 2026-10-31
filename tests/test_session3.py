"""Session 3 Step 1 fixes: default prompt, plate px from Python, batch all colonies."""
import json
import numpy as np
from unittest.mock import patch, MagicMock

from labvision import agent as ag
from labvision.config import AGAR_PROMPT


def _make_fake_seg(n=2):
    fake = MagicMock()
    fake.boxes = [MagicMock(x1=0, y1=0, x2=10, y2=10) for _ in range(n)]
    for b in fake.boxes:
        b.x1, b.y1, b.x2, b.y2 = 0, 0, 10, 10
    fake.width = 100
    fake.height = 100
    fake_masks = np.zeros((n, 100, 100), dtype=bool)
    for i in range(n):
        fake_masks[i, i * 10 : i * 10 + 5, i * 10 : i * 10 + 5] = True
    fake.decoded_masks = lambda: fake_masks
    return fake, fake_masks


def test_default_segment_prompt_is_agar_prompt(tmp_path):
    """Step 1a: default segment prompt must be config.AGAR_PROMPT, not 'bacterial colony'."""
    # Verify TOOLS does not use bacterial colony as default example
    assert "bacterial colony" not in json.dumps(ag.TOOLS).lower()
    # functional: Muse calls segment with no prompt -> agent uses AGAR_PROMPT
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (100, 100), "white").save(p)

    captured = {}

    def fake_muse(messages, tools=None, use_cache=True):
        # first call: check_quality, second: segment with empty args, third: measure, fourth: final
        n = getattr(fake_muse, "n", 0)
        fake_muse.n = n + 1
        if n == 0:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "check_quality", "arguments": "{}"}}]}}]}
        if n == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "segment", "arguments": "{}"}}]}}]}
        if n == 2:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "3", "function": {"name": "measure", "arguments": "{}"}}]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    fake_seg, _ = _make_fake_seg(1)

    def fake_segment(image_bytes, prompt):
        captured["prompt"] = prompt
        return fake_seg

    fake_muse.n = 0
    with patch.object(ag, "_call_muse", side_effect=fake_muse):
        with patch.object(ag, "segment", side_effect=fake_segment):
            out = ag.run_agent(p, "count", use_cache=False)
    assert captured.get("prompt") == AGAR_PROMPT, f"default prompt was {captured.get('prompt')!r}, expected {AGAR_PROMPT!r}"
    assert AGAR_PROMPT == "round spot"


def test_measure_tool_has_no_plate_diameter_px_param():
    """Step 1b: plate_diameter_px must not be in model-facing TOOLS."""
    tools_json = json.dumps(ag.TOOLS)
    # the measure tool params should not expose plate_diameter_px
    import re

    # find measure tool
    measure_tool = [t for t in ag.TOOLS if t["function"]["name"] == "measure"][0]
    props = measure_tool["function"]["parameters"]["properties"]
    assert "plate_diameter_px" not in props, f"measure tool still exposes plate_diameter_px: {props}"
    assert "plate_diameter_mm" not in props


def test_measure_uses_plate_py_not_model_param(tmp_path):
    """Step 1b: plate diameter comes from plate.py, model param ignored."""
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (100, 100), "white").save(p)

    fake_seg, _ = _make_fake_seg(1)

    # case 1: plate detected -> diameters in mm
    def fake_muse_plate(messages, tools=None, use_cache=True):
        n = getattr(fake_muse_plate, "n", 0)
        fake_muse_plate.n = n + 1
        if n == 0:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "check_quality", "arguments": "{}"}}]}}]}
        if n == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "segment", "arguments": json.dumps({"prompt": "round spot"})}}]}}]}
        if n == 2:
            # model tries to inject plate_diameter_px=999 (should be ignored)
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "3", "function": {"name": "measure", "arguments": json.dumps({"plate_diameter_px": 999, "min_diameter_mm": 1})}}]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    fake_muse_plate.n = 0
    with patch.object(ag, "_call_muse", side_effect=fake_muse_plate):
        with patch.object(ag, "segment", return_value=fake_seg):
            with patch("labvision.plate.detect_plate_diameter_px", return_value=200.0):
                out = ag.run_agent(p, "count larger than 1 mm", use_cache=False)
    # should have diameter_mm computed from 200px plate, not 999
    assert out["rows"], "no rows"
    assert "diameter_mm" in out["rows"][0], "plate detected but no diameter_mm"
    # diameter_mm = diameter_px * (90/200)
    # verify not using 999
    from labvision.measure import equivalent_diameter_px, mask_area_px

    dpx = equivalent_diameter_px(mask_area_px(np.zeros((100, 100), dtype=bool)))
    # our fake mask area is 25
    expected_scale = 90.0 / 200.0
    # check scale used was 200 not 999
    row = out["rows"][0]
    assert abs(row["diameter_mm"] - row["diameter_px"] * expected_scale) < 1e-6

    # case 2: no plate -> sizes in px
    fake_muse_plate.n = 0
    with patch.object(ag, "_call_muse", side_effect=fake_muse_plate):
        with patch.object(ag, "segment", return_value=fake_seg):
            with patch("labvision.plate.detect_plate_diameter_px", return_value=None):
                out = ag.run_agent(p, "count", use_cache=False)
    assert "diameter_mm" not in out["rows"][0]
    assert "No plate detected" in out["report"]


def test_characterize_batches_all_colonies(tmp_path):
    """Step 1c: characterize must cover ALL detected colonies (batch 24)."""
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (200, 200), "white").save(p)

    fake_seg, _ = _make_fake_seg(30)

    captured = {}

    def fake_char(image_path, boxes, diameters_mm=None):
        captured["n"] = len(boxes)
        captured["boxes"] = boxes
        # return dummy morph
        return [{"index": i, "color": "white", "shape": "circular", "margin": "entire", "elevation": "convex", "opacity": "opaque", "surface": "smooth", "morphotype": "A", "morphotype_desc": "white circular", "purity_flag": "Single morphotype (consistent with pure culture)", "mean_diameter_mm": 1.0, "size_class": "medium", "count_in_morphotype": 30} for i in range(len(boxes))]

    def fake_muse_all(messages, tools=None, use_cache=True):
        n = getattr(fake_muse_all, "n", 0)
        fake_muse_all.n = n + 1
        if n == 0:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "check_quality", "arguments": "{}"}}]}}]}
        if n == 1:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "segment", "arguments": json.dumps({"prompt": "round spot"})}}]}}]}
        if n == 2:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "3", "function": {"name": "measure", "arguments": "{}"}}]}}]}
        if n == 3:
            return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "4", "function": {"name": "characterize", "arguments": "{}"}}]}}]}
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    fake_muse_all.n = 0
    with patch.object(ag, "_call_muse", side_effect=fake_muse_all):
        with patch.object(ag, "segment", return_value=fake_seg):
            with patch("labvision.characterize.characterize", side_effect=fake_char):
                out = ag.run_agent(p, "count and characterize", use_cache=False)
    assert captured.get("n") == 30, f"characterize got {captured.get('n')} expected 30 (all colonies)"
    # report should contain colony types
    assert "Colony types" in out["report"]


def test_characterize_module_batches_internally(tmp_path):
    """characterize.py must batch contact sheets of 24 for >24 colonies (no truncation)."""
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (400, 400), "white").save(p)
    boxes = [{"x": 0, "y": 0, "width": 20, "height": 20} for _ in range(30)]

    call_counts = {"n": 0}

    def fake_vision(image_bytes, n):
        call_counts["n"] += 1
        call_counts[f"c{n}"] = call_counts.get(f"c{n}", 0) + 1
        return [{"color": "white", "shape": "circular", "margin": "entire", "elevation": "convex", "opacity": "opaque", "surface": "smooth"} for _ in range(n)]

    with patch("labvision.characterize._call_vision", side_effect=fake_vision):
        from labvision.characterize import characterize

        out = characterize(p, boxes)
    assert len(out) == 30, f"expected 30 results, got {len(out)}"
    assert call_counts["n"] == 2, f"expected 2 batches (24+6), got {call_counts['n']}"
    assert {r["index"] for r in out} == set(range(30))


def test_morphotype_grouping_and_purity_flag():
    """Step 2: Python grouping creates morphotypes, purity flag."""
    from labvision.characterize import _group_morphotypes

    rows = [
        {"color": "white", "shape": "circular", "margin": "entire", "elevation": "convex", "opacity": "opaque", "surface": "smooth"},
        {"color": "white", "shape": "circular", "margin": "entire", "elevation": "convex", "opacity": "opaque", "surface": "smooth"},
        {"color": "yellow", "shape": "irregular", "margin": "undulate", "elevation": "flat", "opacity": "translucent", "surface": "rough"},
    ]
    # with diameter terciles: 1,1,10 -> small,small,large
    diams = [1.0, 1.1, 10.0]
    out = _group_morphotypes(rows, diams)
    # two distinct morphology+size combos -> 2 morphotypes (first two identical -> same type)
    labels = {r["morphotype"] for r in out}
    assert len(labels) == 2
    assert out[0]["morphotype"] == out[1]["morphotype"]
    assert out[2]["morphotype"] != out[0]["morphotype"]
    assert "2 morphotypes" in out[0]["purity_flag"]
    assert "confirm with standard microbiological methods" in out[0]["purity_flag"].lower()

    # single morphotype case
    rows2 = [
        {"color": "white", "shape": "circular", "margin": "entire", "elevation": "convex", "opacity": "opaque", "surface": "smooth"},
        {"color": "white", "shape": "circular", "margin": "entire", "elevation": "convex", "opacity": "opaque", "surface": "smooth"},
    ]
    out2 = _group_morphotypes(rows2, [1.0, 1.0])
    assert "Single morphotype" in out2[0]["purity_flag"]


def test_characterize_schema_no_organism(tmp_path):
    """Step 2: characterize prompt must not request presumptive_organism."""
    from labvision.characterize import _call_vision

    # Check that _call_vision prompt contains morphology fields and not organism
    import inspect

    src = inspect.getsource(_call_vision)
    assert "presumptive_organism" not in src
    assert "organism" not in src.lower() or "presumptive" not in src.lower()
    for field in ("color", "shape", "margin", "elevation", "opacity", "surface"):
        assert field in src, f"missing {field} in vision prompt"


def test_no_presumptive_wording_in_agent(tmp_path):
    """Step 4: agent must not output organism/species names or 'diagnosis'."""
    src = open("labvision/agent.py").read().lower()
    # should not contain banned words in output (except in comments about removal)
    assert "presumptive_organism" not in src
    assert "diagnosis" not in src
    # report template should not mention species
    assert "species" not in src
