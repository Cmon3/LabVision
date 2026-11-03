import io
import wave
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from labvision.voice import transcribe, _to_wav_bytes, _cache_key


def _make_wav(sr=24000, duration=0.5):
    n = int(sr * duration)
    data = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 10000).astype(np.int16)
    return sr, data


def test_to_wav_mono_24k():
    sr, data = _make_wav(48000, 0.1)
    # stereo
    stereo = np.stack([data, data], axis=1)
    wav = _to_wav_bytes(sr, stereo)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 24000
        assert wf.getsampwidth() == 2

def test_transcribe_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "fake-key")
    sr, data = _make_wav()
    from labvision.voice import CACHE_DIR, _to_wav_bytes, _cache_key
    wav = _to_wav_bytes(sr, data)
    cpath = CACHE_DIR / _cache_key(wav)
    if cpath.exists():
        cpath.unlink()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"transcript": "count colonies larger than 2 mm", "audioDurationMs": 1000}
    with patch("labvision.voice.requests.post", return_value=fake_resp) as mock_post:
        text = transcribe((sr, data))
        assert text == "count colonies larger than 2 mm"
        assert mock_post.called
        text2 = transcribe((sr, data))
        assert text2 == text
        assert mock_post.call_count == 1
    # cleanup
    if cpath.exists():
        cpath.unlink()

def test_transcribe_cache_key(tmp_path):
    b = b"fake audio bytes"
    k = _cache_key(b)
    assert k.startswith("voice_")
    assert k.endswith(".json")

def test_transcribe_missing_key(monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    # use unique audio to avoid cache hit from previous test
    sr, data = _make_wav(duration=0.6)
    # ensure cache miss by clearing any existing cache for this audio
    from labvision.voice import CACHE_DIR, _to_wav_bytes, _cache_key
    wav = _to_wav_bytes(sr, data)
    cpath = CACHE_DIR / _cache_key(wav)
    if cpath.exists():
        cpath.unlink()
    try:
        transcribe((sr, data))
        assert False, "should raise"
    except RuntimeError as e:
        assert "META_API_KEY" in str(e)
