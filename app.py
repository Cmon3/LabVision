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
        return None, None, "Upload an image first.", None
    if not request.strip():
        request = "count colonies"
    # save temp for agent (needs path)
    tmp = Path(tempfile.gettempdir()) / "gradio_input.png"
    # image is numpy array from gr.Image
    if isinstance(image, np.ndarray):
        Image.fromarray(image.astype(np.uint8)).save(tmp)
    else:
        # filepath
        tmp = Path(image)

    # Run agent (handles qc/segment/measure)
    try:
        out = run_agent(tmp, request)
    except Exception as e:
        return None, None, f"Error: {e}", None

    # Build overlay via direct segment for visual (reuse agent's seg)
    # Re-segment for overlay to avoid agent internals
    try:
        seg = segment(Path(tmp).read_bytes(), "cell" if "cell" in request.lower() else "bacterial colony")
        masks = seg.decoded_masks()
        overlay = _overlay(tmp, seg.boxes, masks)
        rows = measure_masks(masks)
        df = pd.DataFrame(rows) if rows else pd.DataFrame([{"info": "no objects"}])
    except Exception:
        overlay = Image.open(tmp).convert("RGB")
        df = pd.DataFrame(out.get("rows", []))

    # CSV
    csv_path = None
    if out.get("rows"):
        df.to_csv("/tmp/labvision_results.csv", index=False)
        csv_path = "/tmp/labvision_results.csv"
    elif not df.empty:
        df.to_csv("/tmp/labvision_results.csv", index=False)
        csv_path = "/tmp/labvision_results.csv"

    return overlay, df, out.get("report", ""), csv_path


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
    out_report = gr.Textbox(label="Report", lines=10)
    out_csv = gr.File(label="CSV download")
    btn.click(process, inputs=[inp_img, inp_req], outputs=[out_img, out_table, out_report, out_csv])

if __name__ == "__main__":
    demo.launch()
