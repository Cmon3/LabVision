"""Agent: mock Muse HTTP, verify numbers come from measure only."""
import json
import numpy as np
from unittest.mock import patch

from labvision import agent as ag


def _fake_muse_responses(*args, **kwargs):
    # Sequence: first call -> check_quality, second -> segment, third -> measure, fourth -> final
    calls = getattr(_fake_muse_responses, "calls", 0)
    _fake_muse_responses.calls = calls + 1
    if calls == 0:
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "check_quality", "arguments": "{}"}}]}}]}
    if calls == 1:
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "segment", "arguments": json.dumps({"prompt": "cell"})}}]}}]}
    if calls == 2:
        return {"choices": [{"message": {"role": "assistant", "tool_calls": [{"id": "3", "function": {"name": "measure", "arguments": "{}"}}]}}]}
    return {"choices": [{"message": {"role": "assistant", "content": "All done. Colonies look healthy."}}]}


def test_report_numbers_from_measure(tmp_path):
    # create tiny fake image
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (100, 100), "white").save(p)
    _fake_muse_responses.calls = 0

    # mock segment to return 2 fake masks
    fake_seg = type("S", (), {})()
    fake_seg.boxes = [None, None]
    fake_seg.width = 100
    fake_seg.height = 100
    fake_masks = np.zeros((2, 100, 100), dtype=bool)
    fake_masks[0, 10:20, 10:20] = True
    fake_masks[1, 30:50, 30:50] = True
    fake_seg.decoded_masks = lambda: fake_masks

    with patch.object(ag, "_call_muse", side_effect=_fake_muse_responses):
        with patch.object(ag, "segment", return_value=fake_seg):
            out = ag.run_agent(p, "count colonies", use_cache=False)
    # numbers must equal measure output
    assert out["count"] == 2
    assert len(out["rows"]) == 2
    assert "Count: 2 objects" in out["report"]
    # LLM note should be included but not numbers
    assert "healthy" in out["report"].lower() or "All done" in out["report"]


def test_empty_qc_flag():
    from PIL import Image
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mktemp(suffix=".png"))
    Image.new("RGB", (10, 10), "black").save(p)
    _fake_muse_responses.calls = 99  # force final response
    with patch.object(ag, "_call_muse", return_value={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}):
        out = ag.run_agent(p, "count", use_cache=False)
    assert "Count:" in out["report"]
