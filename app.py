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
                from labvision.characterize import characterize

                boxes_for_char = [{"x": b.x1, "y": b.y1, "width": b.x2 - b.x1, "height": b.y2 - b.y1} for b in seg.boxes]
                diams = [r.get("diameter_mm", r.get("diameter_px", 0)) for r in rows] if rows and isinstance(rows, list) else None
                has_mm = bool(rows and rows[0].get("diameter_mm")) if rows else False
                char = characterize(tmp, boxes_for_char, diameters_mm=diams if has_mm else None)
                # aggregate to morphotype table: label, description, count, mean diameter
                from collections import Counter

                cnt = Counter(c["morphotype"] for c in char)
                # one row per morphotype with mean diameter
                agg = {}
                for c in char:
                    mt = c["morphotype"]
                    if mt not in agg:
                        agg[mt] = {"label": mt, "description": c["morphotype_desc"], "count": cnt[mt], "mean_diameter_mm": round(c["mean_diameter_mm"], 2)}
                morph_df = pd.DataFrame(sorted(agg.values(), key=lambda r: r["label"]))
                # append purity flag as extra info row via report (already in run_agent)
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
    morph_table = gr.Dataframe(label="Colony types — confirm with standard microbiological methods")
    out_report = gr.Textbox(label="Report", lines=10)
    out_csv = gr.File(label="CSV download")
    btn.click(process, inputs=[inp_img, inp_req], outputs=[out_img, out_table, morph_table, out_report, out_csv])

if __name__ == "__main__":
    demo.launch()
