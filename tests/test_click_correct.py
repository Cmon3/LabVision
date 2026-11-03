import numpy as np
from unittest.mock import patch, MagicMock

from labvision.segment import _build_point_prompt, _build_box_prompt, Box
from labvision.segment import segment_point


def test_build_point_prompt():
    s = _build_point_prompt(100, 200, 1000, 800, obj_id=1)
    assert "1< |point;x=100;y=200;w=1000;h=800|>" == s
    assert "<0f>" not in s

def test_build_box_prompt():
    s = _build_box_prompt(10, 20, 50, 60, 100, 100)
    assert "box;x1=10;y1=20;x2=50;y2=60;w=100;h=100" in s
    assert "<0f>" not in s

def test_remove_by_click_synthetic():
    from app import _find_box_at, _find_mask_at
    boxes = [Box(10, 10, 50, 50), Box(60, 60, 100, 100)]
    assert _find_box_at(boxes, 20, 20) == 0
    assert _find_box_at(boxes, 80, 80) == 1
    assert _find_box_at(boxes, 5, 5) is None

    masks = np.zeros((2, 100, 100), dtype=bool)
    masks[0, 10:50, 10:50] = True
    masks[1, 60:90, 60:90] = True
    assert _find_mask_at(masks, 20, 20) == 0
    assert _find_mask_at(masks, 70, 70) == 1
    assert _find_mask_at(masks, 5, 5) is None

def test_segment_point_mocked(tmp_path):
    from PIL import Image
    p = tmp_path / "img.png"
    Image.new("RGB", (100, 100), "white").save(p)
    data = p.read_bytes()
    fake_seg = MagicMock()
    fake_seg.boxes = [Box(40, 40, 60, 60)]
    fake_seg.decoded_masks.return_value = np.zeros((1, 100, 100), dtype=bool)
    fake_seg.width = 100
    fake_seg.height = 100
    fake_seg.cached = False
    with patch("labvision.segment.segment", return_value=fake_seg) as mock_seg:
        seg = segment_point(data, 50, 50)
        assert mock_seg.called
        call_arg = mock_seg.call_args[0][1]  # prompt
        assert "point;x=50;y=50" in call_arg
        assert seg.boxes[0].x1 == 40
