"""Gradio app: image + request -> overlay, table, report, CSV."""
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


def _overlay(image_path: str, boxes, masks: np.ndarray | None):
    im = Image.open(image_path).convert("RGBA")
    if masks is not None and masks.size:
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


def process(image, request: str):
    if image is None:
        return None, None, None, "Upload an image first.", None
    if not request.strip():
        request = "count colonies"
    tmp = Path(tempfile.gettempdir()) / "gradio_input.png"
    if isinstance(image, np.ndarray):
        Image.fromarray(image.astype(np.uint8)).save(tmp)
    else:
        tmp = Path(image)

    try:
        out = run_agent(tmp, request)
    except Exception as e:
        return None, None, None, f"Error: {e}", None

    # overlay
    try:
        from labvision.config import AGAR_PROMPT as _AP

        seg = segment(Path(tmp).read_bytes(), "cell" if "cell" in request.lower() else _AP)
        masks = seg.decoded_masks()
        overlay = _overlay(tmp, seg.boxes, masks)
        rows = measure_masks(masks)
        df = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "no objects"}])
        # morphology table if characterize was used
        morph_df = None
        if "characterize" in request.lower() and seg.boxes:
            try:
                import numpy as _np
                from labvision.features import extract_features as _feat
                from labvision.morphotypes import cluster_plate as _clust
                from labvision.config import MORPHOTYPE_THRESHOLD as _T, MORPHOTYPE_COLOR_WEIGHT as _CW

                boxes_for_char = [{"x": b.x1, "y": b.y1, "width": b.x2 - b.x1, "height": b.y2 - b.y1} for b in seg.boxes]
                # use masks if available for features
                if masks.size and masks.shape[0] == len(seg.boxes):
                    feats = _feat(np.array(Image.open(tmp).convert("RGB")), masks=masks)
                else:
                    feats = _feat(np.array(Image.open(tmp).convert("RGB")), boxes=boxes_for_char)
                labels = _clust(feats, threshold=_T, color_weight=_CW)
                uniq = sorted(set(labels))
                # per-group Muse wording
                agg = {}
                for lab in uniq:
                    lab_char = chr(ord("A") + uniq.index(lab))
                    g_mask = labels == lab
                    count = int(g_mask.sum())
                    # feature summary for Muse
                    g_feats = feats[g_mask]
                    summary = {"lab": g_feats[:, 0:3].mean(axis=0).tolist() if g_feats.size else [], "circularity": float(g_feats[:, 3].mean()) if g_feats.size else 0, "log_diam": float(g_feats[:, 4].mean()) if g_feats.size else 0, "texture": float(g_feats[:, 5].mean()) if g_feats.size else 0}
                    group_boxes = [boxes_for_char[i] for i in range(len(boxes_for_char)) if labels[i] == lab][:6]
                    try:
                        from labvision.describe import describe_group as _desc
                        desc = _desc(tmp, group_boxes, feature_summary=summary)
                    except Exception:
                        desc = {"color": "cream", "shape": "circular", "margin": "entire", "elevation": "convex", "surface": "smooth"}
                    desc_str = f"{desc['color']}, {desc['shape']}, {desc['margin']} margin, {desc['elevation']}, {desc['surface']}"
                    # mean diameter from features
                    mean_log = float(g_feats[:, 4].mean()) if g_feats.size else 0
                    mean_diam = float(_np.exp(mean_log)) if mean_log else 0
                    agg[lab_char] = {"label": lab_char, "description": desc_str, "count": count, "mean_diameter_px": round(mean_diam, 1)}
                morph_df = pd.DataFrame(sorted(agg.values(), key=lambda r: r["label"]))
            except Exception:
                morph_df = None
        else:
            morph_df = None
    except Exception:
        overlay = Image.open(tmp).convert("RGB")
        df = pd.DataFrame(out.get("rows", []))
        morph_df = None

    csv_path = None
    if not df.empty:
        df.to_csv("/tmp/labvision_results.csv", index=False)
        csv_path = "/tmp/labvision_results.csv"

    # return morph as second dataframe if exists, else None
    return overlay, df, morph_df if morph_df is not None else pd.DataFrame(), out.get("report", ""), csv_path


with gr.Blocks(title="LabVision Copilot") as demo:
    gr.Markdown("# LabVision Copilot — SAM 3.1 + Muse Spark")
    gr.Markdown("Upload a plate/cell image + plain-language request (e.g. *count colonies larger than 2 mm*). Numbers come from deterministic `measure.py`, not the LLM.")
    with gr.Row():
        inp_img = gr.Image(label="Image", type="numpy")
        inp_req = gr.Textbox(label="Request", placeholder='count colonies larger than 2 mm', lines=2)
    btn = gr.Button("Run")
    with gr.Row():
        out_img = gr.Image(label="Overlay (masks + boxes)")
        out_table = gr.Dataframe(label="Measurements")
    morph_table = gr.Dataframe(label="Colony types (descriptive only) — Python grouping, Muse wording")
    out_report = gr.Textbox(label="Report", lines=10)
    out_csv = gr.File(label="CSV download")
    btn.click(process, inputs=[inp_img, inp_req], outputs=[out_img, out_table, morph_table, out_report, out_csv])

if __name__ == "__main__":
    demo.launch()
