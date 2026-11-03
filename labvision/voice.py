"""Muse Voice Transcribe — wav file -> text, cached."""
from __future__ import annotations

import hashlib
import io
import json
import os
import wave
from pathlib import Path

import numpy as np
import requests

CACHE_DIR = Path("runs/cache")
API_URL = "https://api.meta.ai/v1/asr/transcribe"
MODEL_ID = "muse-voice-transcribe-1.0"


def _to_wav_bytes(sample_rate: int, data: np.ndarray) -> bytes:
    """Convert audio to mono 24kHz int16 WAV bytes."""
    arr = np.asarray(data)
    # to mono
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1)
    # normalize if float in [-1,1]
    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767).astype(np.int16)
    elif arr.dtype != np.int16:
        # assume int
        arr = arr.astype(np.int16)

    # resample to 24kHz if needed
    target_sr = 24000
    if sample_rate != target_sr:
        try:
            from scipy.signal import resample_poly
            import math

            g = math.gcd(sample_rate, target_sr)
            up, down = target_sr // g, sample_rate // g
            arr = resample_poly(arr.astype(np.float32), up, down).astype(np.int16)
        except Exception:
            # fallback: simple decimation/interp via numpy if scipy fails
            # rough: use linear interpolation
            duration = len(arr) / sample_rate
            target_len = int(duration * target_sr)
            x_old = np.linspace(0, duration, len(arr))
            x_new = np.linspace(0, duration, target_len)
            arr = np.interp(x_new, x_old, arr.astype(float)).astype(np.int16)

    # write WAV to memory
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(arr.tobytes())
    return buf.getvalue()


def _cache_key(audio_bytes: bytes) -> str:
    return f"voice_{hashlib.sha256(audio_bytes).hexdigest()[:16]}.json"


def transcribe(audio, sample_rate: int | None = None) -> str:
    """Transcribe audio (filepath, bytes, or (sr, ndarray) tuple) → transcript.

    Cached by sha256 of the wav bytes sent.
    """
    import tempfile

    # normalize input to wav bytes
    if isinstance(audio, tuple) and len(audio) == 2:
        sr, data = audio
        wav_bytes = _to_wav_bytes(int(sr), np.asarray(data))
    elif isinstance(audio, (str, Path)):
        p = Path(audio)
        if p.suffix.lower() == ".wav":
            # try to read and convert to 24k mono if needed
            with wave.open(str(p), "rb") as wf:
                sr = wf.getframerate()
                nchan = wf.getnchannels()
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16)
                if nchan == 2:
                    arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                wav_bytes = _to_wav_bytes(sr, arr)
        else:
            # other audio file: load via soundfile? fallback: read as bytes and let API handle
            wav_bytes = p.read_bytes()
    elif isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate required for ndarray input")
        wav_bytes = _to_wav_bytes(sample_rate, audio)
    elif isinstance(audio, (bytes, bytearray)):
        # assume already wav bytes; try to normalize if needed
        try:
            with wave.open(io.BytesIO(bytes(audio)), "rb") as wf:
                sr = wf.getframerate()
                if sr != 24000 or wf.getnchannels() != 1:
                    raw = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(raw, dtype=np.int16)
                    if wf.getnchannels() == 2:
                        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
                    wav_bytes = _to_wav_bytes(sr, arr)
                else:
                    wav_bytes = bytes(audio)
        except Exception:
            wav_bytes = bytes(audio)
    else:
        raise ValueError(f"unsupported audio input: {type(audio)}")

    cpath = CACHE_DIR / _cache_key(wav_bytes)
    if cpath.exists():
        return json.loads(cpath.read_text()).get("transcript", "")

    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise RuntimeError("Set META_API_KEY for Voice Transcribe")

    # prepare multipart
    req_json = json.dumps({
        "mode": "PUSH_TO_TALK",
        "model": MODEL_ID,
        "audioEncoding": "WAV",
        "languageBias": ["English", "Spanish"],
        "keywords": ["colonies", "CFU", "agar", "Petri", "millimetres", "mm", "plate", "count"],
    })

    files = {
        "request": (None, req_json, "application/json"),
        "audio": ("audio.wav", wav_bytes, "audio/wav"),
    }
    headers = {"Authorization": f"Bearer {key}"}

    # retry once on 429/500
    for attempt in range(2):
        r = requests.post(API_URL, headers=headers, files=files, timeout=60)
        if r.status_code == 200:
            break
        if r.status_code in (429, 500) and attempt == 0:
            continue
        raise RuntimeError(f"Voice Transcribe HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()
    transcript = data.get("transcript", "")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"transcript": transcript, "model": MODEL_ID, "audioDurationMs": data.get("audioDurationMs", 0)}))
    return transcript
