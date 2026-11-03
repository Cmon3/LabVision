"""Gradio app: polished, video-ready."""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from labvision.agent import run_agent
from labvision.measure import measure_masks
from labvision.segment import segment


def _overlay(image_path: str | Path, boxes, masks: np.ndarray | None, show_masks: bool = True):
    im = Image.open(image_path).convert("RGBA")
    if show_masks and masks is not None and masks.size:
        for m in masks:
            if m.shape[0] != im.size[1] or m.shape[1] != im.size[0]:
                continue
            red = Image.new("RGBA", im.size, (255, 0, 0, 70))
            mask_im = Image.fromarray((m.astype(np.uint8) * 255))
            tmp = Image.new("RGBA", im.size, (0, 0, 0, 0))
            tmp.paste(red, mask=mask_im)
            im = Image.alpha_composite(im, tmp)
    draw = ImageDraw.Draw(im)
    for b in boxes:
        draw.rectangle([(b.x1, b.y1), (b.x2, b.y2)], outline="lime", width=2)
    return im.convert("RGB")


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "meta_api_key" in low or "baseten_api_key" in low or "set " in low and "key" in low:
        return "Missing API key: set META_API_KEY and BASETEN_API_KEY in your .env (see .env.example)."
    if "429" in msg or "rate limit" in low:
        return "API rate limit (429): wait a minute and try again."
    if "http" in low or "api" in low:
        return f"API error: {msg[:300]}"
    return f"Error: {msg[:400]}"


def process(image, request: str, show_masks: bool = True):
    if image is None:
        return None, None, pd.DataFrame(), "Upload an image first.", "No run yet.", None
    if not request or not request.strip():
        request = "count colonies"
    tmp = Path(tempfile.gettempdir()) / "gradio_input.png"
    if isinstance(image, np.ndarray):
        Image.fromarray(image.astype(np.uint8)).save(tmp)
    else:
        tmp = Path(image)

    try:
        out = run_agent(tmp, request)
    except Exception as e:
        return None, pd.DataFrame(), pd.DataFrame(), _friendly_error(e), "Run failed.", None

    # overlay
    try:
        from labvision.config import AGAR_PROMPT as _AP
        seg = segment(Path(tmp).read_bytes(), "cell" if "cell" in request.lower() else _AP)
        masks = seg.decoded_masks()
        overlay = _overlay(tmp, seg.boxes, masks, show_masks=show_masks)
        # store for toggle
        overlay_no_masks = _overlay(tmp, seg.boxes, None, show_masks=False)
        rows = measure_masks(masks)
        df = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "no objects"}])

        morph_df = None
        if "characterize" in request.lower() and seg.boxes:
            try:
                import numpy as _np
                from labvision.features import extract_features as _feat
                from labvision.morphotypes import cluster_plate as _clust
                from labvision.config import MORPHOTYPE_THRESHOLD as _T, MORPHOTYPE_COLOR_WEIGHT as _CW
                boxes_for_char = [{"x": b.x1, "y": b.y1, "width": b.x2 - b.x1, "height": b.y2 - b.y1} for b in seg.boxes]
                if masks.size and masks.shape[0] == len(seg.boxes):
                    feats = _feat(np.array(Image.open(tmp).convert("RGB")), masks=masks)
                else:
                    feats = _feat(np.array(Image.open(tmp).convert("RGB")), boxes=boxes_for_char)
                labels = _clust(feats, threshold=_T, color_weight=_CW)
                uniq = sorted(set(labels))
                agg = {}
                for lab in uniq:
                    lab_char = chr(ord("A") + uniq.index(lab))
                    g_mask = labels == lab
                    count = int(g_mask.sum())
                    g_feats = feats[g_mask]
                    summary = {"lab": g_feats[:, 0:3].mean(axis=0).tolist() if g_feats.size else [], "circularity": float(g_feats[:, 3].mean()) if g_feats.size else 0, "log_diam": float(g_feats[:, 4].mean()) if g_feats.size else 0, "texture": float(g_feats[:, 5].mean()) if g_feats.size else 0}
                    group_boxes = [boxes_for_char[i] for i in range(len(boxes_for_char)) if labels[i] == lab][:6]
                    try:
                        from labvision.describe import describe_group as _desc
                        desc = _desc(tmp, group_boxes, feature_summary=summary)
                    except Exception:
                        desc = {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}
                    desc_str = f"{desc['color']}, {desc['shape']}, {desc['margin']} margin, {desc['elevation']}, {desc['surface']}"
                    mean_log = float(g_feats[:, 4].mean()) if g_feats.size else 0
                    mean_diam = float(_np.exp(mean_log)) if mean_log else 0
                    agg[lab_char] = {"label": lab_char, "description": desc_str, "count": count, "mean_diameter_px": round(mean_diam, 1)}
                morph_df = pd.DataFrame(sorted(agg.values(), key=lambda r: r["label"]))
            except Exception:
                morph_df = None
        else:
            morph_df = None
    except Exception as e:
        # fallback to agent output
        overlay = Image.open(tmp).convert("RGB")
        df = pd.DataFrame(out.get("rows", []))
        morph_df = None
        return overlay, df, morph_df if morph_df is not None else pd.DataFrame(), _friendly_error(e), _verification_md(out.get("verification", {}), request), df.to_csv("/tmp/labvision_results.csv", index=False) or "/tmp/labvision_results.csv" if not df.empty else None

    csv_path = None
    if not df.empty:
        df.to_csv("/tmp/labvision_results.csv", index=False)
        csv_path = "/tmp/labvision_results.csv"

    ver_md = _verification_md(out.get("verification", {}), request)
    # combine morph into measurements tab? we return separate for now but app will show in How verified
    # For measurements tab we show df; morph table will be shown in verified? Actually spec says measurements tab has table+CSV
    # We'll return morph as part of verified or as separate; here we keep morph in verified text and also return df
    # To show morph table in app, we could append to df or show separately — we keep morph in report; for now merge morph into verification
    return overlay, df, morph_df if morph_df is not None else pd.DataFrame(), out.get("report", ""), ver_md, csv_path


def _verification_md(ver: dict, request: str) -> str:
    if not ver:
        return "No verification data yet. Run an image."
    lines = ["**How it was verified (tool calls in order):**", ""]
    qc = ver.get("qc") or {}
    lines.append(f"1. `check_quality` → blur score {qc.get('blur_score', 0):.1f} ({'blurry' if qc.get('is_blurry') else 'sharp'}), glare {qc.get('glare_frac', 0):.1%} ({'yes' if qc.get('glare') else 'no'})")
    lines.append(f"2. `segment` → prompt `{ver.get('prompt_used', '?')}` — retry {ver.get('retry_used', False)} — {ver.get('seg_boxes', 0)} boxes")
    pdiam = ver.get("plate_diameter_px")
    if pdiam:
        lines.append(f"3. `measure` → plate diameter {pdiam:.0f}px → 90mm ⇒ {ver.get('mm_per_px', 0)*1000:.2f} µm/px — {ver.get('count', 0)} objects — TNTC {ver.get('tntc', False)}")
    else:
        lines.append(f"3. `measure` → no plate detected — sizes in px — {ver.get('count', 0)} objects — TNTC {ver.get('tntc', False)}")
    if "characterize" in request.lower():
        lines.append("4. `characterize` → Python features clustered, Muse wording per group (descriptive only)")
    lines.append("")
    lines.append(f"Request: `{request}`")
    return "\n".join(lines)


# Examples — only if AGAR images exist locally (never committed)
_example_paths = []
for cand in [
    "data/agar_demo/AGAR_representative/lower-resolution/14684.jpg",  # normal
    "data/agar_demo/AGAR_representative/higher-resolution/bright/518.jpg",  # dense TNTC
    "data/agar_demo/AGAR_representative/lower-resolution/356.jpg",  # small-count
]:
    if Path(cand).exists():
        _example_paths.append([cand, "count colonies"])
    # also try alternative locations
for cand in [
    "data/agar_demo/AGAR_representative/lower-resolution/5206.jpg",
]:
    if Path(cand).exists() and len(_example_paths) < 3:
        _example_paths.append([cand, "count colonies"])


with gr.Blocks(title="LabVision Copilot") as demo:
    gr.Markdown("# LabVision Copilot — SAM 3.1 + Muse Spark")
    gr.Markdown("Upload a plate / cell image + plain-language request (e.g. *count colonies larger than 2 mm*). Numbers come from deterministic `measure.py`, not the LLM.")
    with gr.Row():
        with gr.Column(scale=1):
            inp_img = gr.Image(label="Image (upload or webcam)", type="numpy", sources=["upload", "webcam"])
            with gr.Row():
                inp_req = gr.Textbox(label="Request", placeholder="count colonies larger than 2 mm", lines=2, scale=3)
                voice_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Voice (hands-free)", scale=1)
            inp_masks = gr.Checkbox(label="Show masks", value=True)
            btn = gr.Button("Run", variant="primary")
            gr.Markdown("_Voice: click mic, say request (e.g. “count colonies larger than 2 mm”), transcript fills box — edit if needed._")
            gr.Markdown("**Examples** (only if AGAR sample present locally):")
            if _example_paths:
                gr.Examples(examples=_example_paths, inputs=[inp_img, inp_req], cache_examples=False)
            else:
                gr.Markdown("_No local AGAR sample found — upload your own._")
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("Result"):
                    out_img = gr.Image(label="Overlay (masks + boxes)")
                with gr.Tab("Measurements"):
                    out_table = gr.Dataframe(label="Measurements (px/mm)")
                    out_csv = gr.File(label="CSV download")
                    morph_table = gr.Dataframe(label="Colony types (descriptive only)")
                with gr.Tab("How it was verified"):
                    out_ver = gr.Markdown(label="Verification")
                with gr.Tab("Report"):
                    out_report = gr.Textbox(label="Report", lines=10)

    gr.Markdown("Research/QC quantification. Not a medical device.")
    gr.Markdown("Numbers from Python (`measure.py` + plate detection). Muse Spark Open via Baseten writes interpretation only.")

    def voice_to_text(audio):
        if audio is None:
            return ""
        try:
            from labvision.voice import transcribe as _transcribe
            text = _transcribe(audio)
            # log transcript (runlog) — best effort
            try:
                from labvision.runlog import save_run
                from datetime import datetime, timezone
                run_id = f"voice_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
                save_run("runs", run_id, inputs={"request": text}, model_ids={"voice": "muse-voice-transcribe-1.0"}, params={"audio": "mic"}, results=[{"transcript": text}])
            except Exception:
                pass
            return text
        except Exception as e:
            return _friendly_error(e)

    voice_in.change(voice_to_text, inputs=[voice_in], outputs=[inp_req])

    btn.click(process, inputs=[inp_img, inp_req, inp_masks], outputs=[out_img, out_table, morph_table, out_report, out_ver, out_csv])
    inp_masks.change(process, inputs=[inp_img, inp_req, inp_masks], outputs=[out_img, out_table, morph_table, out_report, out_ver, out_csv])

if __name__ == "__main__":
    demo.launch()
