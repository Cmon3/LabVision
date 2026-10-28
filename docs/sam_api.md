# SAM 3.1 on Meta Model API (from dev.meta.ai docs, copied 2026-10-28)

## Endpoint
- `POST https://api.meta.ai/v1/responses` (Responses API, OpenAI-compatible)
- Auth: `Authorization: Bearer $META_API_KEY` (same key as the rest of the API)
- Model ID: `sam-3.1`
- Price: $2.50 / 1,000 images; video $0.20 / 1,000 frames

## Request
One user message: a short `input_text` noun phrase naming ONE concept + one media part.
- Prompt: short concrete noun phrase ("bacterial colony"). One concept per request;
  several object types = several requests.
- Image: `input_image` part with `image_url`. (Base64 data URL support: NOT yet confirmed.)
- Video: `input_video` part (MP4/MOV, H.264/H.265/MPEG-4).
- `stream`: optional for images; non-streaming returns the full result in one response.
- `metadata: {"mask_encoding": "one_bit"}` selects the binary-mask encoding.
- Box and point prompts exist (to segment already-located objects) — see the
  "Box and point prompts" doc page (NOT yet copied).

Example (TypeScript from docs; Python equivalent: `openai` SDK with `base_url="https://api.meta.ai/v1"`):
```ts
const client = new OpenAI({ baseURL: "https://api.meta.ai/v1", apiKey: process.env.MODEL_API_KEY });
const res = await client.responses.create({
  model: "sam-3.1",
  input: [{ type: "message", role: "user", content: [
    { type: "input_text", text: "yellow school bus" },
    { type: "input_image", image_url: "https://example.com/street.png" },
  ]}],
  stream: true,
  metadata: { mask_encoding: "one_bit" },
});
```

## Response
Result comes in the `output_text` channel as special-token text: one line per image
(one line per frame for video). Each object = one box + one mask. Example (truncated):
```
<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561|><|mask;x=0;y=0;data=46,101,~...|>,1<|...
```
- `<0f>` = frame index; the leading integer = object ordinal; then one box, then one mask.
- Box coordinates are in source pixels; `w`,`h` = image size.
- Mask = binary raster; `~` marks a lossless mask payload.
- Zero matches = empty `output_text` (valid, not an error).
- Official parser is TypeScript (`@meta-sam/parser`: parseImageStream, decodeMaskToRaster).
  In Python we parse it ourselves with the grammar below.

## Other models (models page)
- Muse Spark 1.3: $1.25 in / $0.15 cached / $4.25 out per 1M tokens. Native tool calling + MCP.
- Muse Spark 1.3 Contributor: $0.10 / $0.002 / $0.20, but inputs/outputs are used to train
  Meta models; select countries only. Do not send non-public data.
- Muse Voice Transcribe: $0.18 / hour of audio.
- Muse Spark Open: open weights (Meta, Hugging Face), Muse Model License.
- Muse Glimmer: 30B open model, Apache 2.0, single GPU.
- Muse Wonder Full: Muse Code subscription only (Power / Maximum plans).

## Output grammar ("Read segmentation output" page, copied 2026-10-28)
```
line      ::= FRAME record ("," record)*
record    ::= OBJECT_ID box mask
box       ::= "<|box;x1=" INT ";y1=" INT ";x2=" INT ";y2=" INT ";w=" INT ";h=" INT [";c=" CONF] "|>"
mask      ::= "<|mask;x=0;y=0" [";c=" CONF] ";data=" H "," W "," ENC PAYLOAD "|>"
FRAME     ::= "<" INT "f>"          ; frame index, zero-based
OBJECT_ID ::= INT                   ; stable per object, NOT dense (can be 0,2)
CONF      ::= decimal 0..1          ; only with include_confidence = "true"
H, W      ::= INT                   ; mask raster rows, columns (smaller than the image)
ENC       ::= "~" (lossless, default) | "!" (one_bit)
PAYLOAD   ::= base85 bytes (NOT base64; may contain "~" or "!")
```
Notes:
- Box: x1,y1 top-left, x2,y2 bottom-right, in source pixels; w,h = source image size.
- Mask: `x=0;y=0` origin always 0. `data=37,51,~...` = 37 rows x 51 columns, lossless.
  The mask is a low-res raster of the WHOLE frame -> resize to (h, w) before measuring.
- Decoded mask = row-major array of 0/1, length H*W.
- Request `metadata.mask_encoding = "one_bit"` for the smallest payload.
- Confidence: request option `include_confidence: "true"` adds `;c=` to box and mask.
- Parse fields by key, not by position (c may or may not be present).

### Open points (verify with the first real response in T1/T2)
- Exact base85 alphabet (Python `base64.b85decode` vs `a85decode`) - test both.
- one_bit byte layout (bit order) - check that decoded H*W matches and the overlay
  lands on the object.
- The default "~" lossless codec is not documented here -> always request one_bit.
