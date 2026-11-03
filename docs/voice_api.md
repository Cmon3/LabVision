# Muse Voice Transcribe (from dev.meta.ai docs, copied 2026-11-03)

Use the FILE endpoint (simplest for a mic button in Gradio):
- `POST https://api.meta.ai/v1/asr/transcribe?sessionId=<optional-id>`
- Auth: `Authorization: Bearer $META_API_KEY`
- Model ID: `muse-voice-transcribe-1.0`
- multipart/form-data with two parts:
  - `request` (application/json): {"mode":"PUSH_TO_TALK","model":"muse-voice-transcribe-1.0",
    "audioEncoding":"WAV","languageBias":["English","Spanish"],
    "keywords":["colonies","CFU","agar","Petri","millimetres","mm","plate","count"]}
  - `audio`: WAV file — RIFF/WAVE, MONO, 16-bit integer PCM, 16 kHz or 24 kHz.
    (Gradio mic audio is usually 44.1/48 kHz stereo -> convert in Python to mono int16
    24 kHz, e.g. scipy.signal.resample_poly; do not assume ffmpeg is installed.)
- Default response JSON:
  {"sessionId":"...","transcript":"full text","audioDurationMs":8240,"turns":[...]}
  `transcript` is always present; `turns` is empty in PUSH_TO_TALK.
- languageBias: hint, not forced. keywords: bias, no exact-spelling guarantee.
- Limits: body > 32 MB -> 413; audio > 10 min -> 400; bad WAV -> 400;
  429 rate limit -> retry with backoff; 500 -> retry once.
- Price: $0.18 per hour of audio (billed per whole second). Failed / 429 requests not billed.
- Realtime WebSocket endpoint exists (wss://api.meta.ai/v1/asr/realtime) — NOT needed here.
