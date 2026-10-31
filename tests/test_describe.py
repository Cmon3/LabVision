from labvision.describe import VALID_VOCAB, _validate


def test_validate_allowed():
    desc = {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}
    assert _validate(desc) == desc

def test_validate_correction():
    # invalid words should be mapped to allowed fallback
    desc = {"color": "purple", "shape": "star", "margin": "wavy", "elevation": "high", "surface": "shiny"}
    v = _validate(desc)
    for k, allowed in VALID_VOCAB.items():
        assert v[k] in allowed

def test_validate_case_insensitive():
    desc = {"color": "White", "shape": "CIRCULAR", "margin": "Entire", "elevation": "Flat", "surface": "Rough"}
    v = _validate(desc)
    assert v["color"] == "white"
    assert v["surface"] == "rough"

def test_vocab_completeness():
    # ensure spec vocab is present
    assert "white" in VALID_VOCAB["color"]
    assert "cream" in VALID_VOCAB["color"]
    assert "circular" in VALID_VOCAB["shape"]
    assert "punctiform" in VALID_VOCAB["shape"]
    assert "entire" in VALID_VOCAB["margin"]
    assert "mucoid" in VALID_VOCAB["surface"]
