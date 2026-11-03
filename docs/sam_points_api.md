# SAM 3.1 box and point prompts (from dev.meta.ai "Segmenting with prompts", copied 2026-11-03)

Same endpoint and request as docs/sam_api.md (POST /v1/responses, model "sam-3.1",
input_text + input_image). The input_text holds EITHER a noun phrase OR box/point prompts,
never both (mixing -> 400).

## Prompt grammar (same token grammar as the output)
<0f>1<|box;x1=444;y1=626;x2=675;y2=849;w=1800;h=1200|>,2<|point;x=1430;y=700;w=1800;h=1200|>-<|point;x=1300;y=450;w=1800;h=1200|>
- Images use frame marker <0f> only.
- Objects are comma-separated: an object id you choose, then one or more tokens.
- Box: <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|> (x2, y2 inclusive). Max one box per object.
- Positive point: <|point;x=..;y=..;w=..;h=..|> = a pixel that belongs to the object.
- Negative point: prefix "-" before a point = a pixel that is NOT part of the object.
  Only points can be negative.
- Every token carries the same w and h = the image's own pixel size (after EXIF rotation).
- Max 16 prompted objects per request. Ids unique per frame.
- Response: one record per prompted object, under YOUR ids, e.g.
  <0f>1<|box;...|><|mask;x=0;y=0;data=226,236,!...|>
  Box/point prompts ALWAYS return one_bit masks ("!" marker) -> our mask_codec handles it.

## Coordinates: convert from the on-screen preview to image pixels
x_img = (x_click - view_left) * img_w / shown_w ; same for y. Round down, clamp inside the
image. Example: click (100,50) on a 640x360 preview of 1920x1080 -> x=300, y=150.
(Check whether Gradio's select event already gives original-image pixels; verify by
clicking a known colony and drawing the returned box.)

## Other useful facts
- metadata.include_confidence = "true" (string) adds ;c=0..1 to box and mask.
- Results can vary slightly between identical runs -> keep caching by (image hash, prompt).
- A concept that isn't present returns empty output_text (not an error).
- Official Python helper exists: package `meta-sam-parser` (>= 0.0.6) with
  build_object_prompt. Not required: building the string by hand is simple.

## Errors (400, do not retry unchanged)
"object prompts cannot be combined with a phrase", "a box ... cannot be negative",
"object 1 carries more than one box", "object id 1 is used more than once on frame 0",
"every prompt token must declare the same w/h", "point (...) lies outside the ... frame",
"17 objects exceed this model's limit of 16", aspect-ratio mismatch of w/h vs media.
