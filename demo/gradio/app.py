"""Gradio front end for the DFD detector. Run locally with `python app.py`."""

import gradio as gr
from PIL import Image, ImageDraw

from backend import ACCEPT_TYPES, DetectError, backend_mode, detect

HEADER = """
# DFD detector
Upload an image or short video. You get the detector's verdict, the reasons it returned, and each specialist's score.
This is a research demo and not for enforcement decisions.
"""

EMPTY = "Upload a file and press **Scan**."
SIZE_NOTE = "_Hosted on Vercel, requests over about 4.5 MB fail before they reach the detector. Compress large files first._"


def annotate(path: str, faces: list[dict]) -> Image.Image | None:
    """Draw face boxes in original pixel coordinates."""
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for face in faces:
        x, y, w, h = face["box"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 64, 64), width=max(2, img.width // 200))
    return img


def summary(result: dict) -> str:
    verdict = result["verdict"]
    lines = [
        f"### {result['label']}",
        f"**Verdict:** `{verdict}`  ·  **Confidence:** {result['confidence'] * 100:.0f}%",
    ]
    if result.get("latency_ms"):
        lines.append(f"**Latency:** {result['latency_ms']} ms")
    return "\n\n".join(lines)


def specialist_rows(result: dict) -> list[list]:
    rows = []
    for s in result["specialists"]:
        rows.append([s["title"], s["applies_to"], f"{s['p_fake']:.3f}", s["verdict"]])
    return rows


def keyframe_rows(result: dict) -> list[list]:
    video = result.get("video")
    if not video:
        return []
    return [[f["t"], f["verdict"], f"{f['p_fake']:.3f}"] for f in video["frames"]]


def reasons_md(result: dict) -> str:
    items = "\n".join(f"- {r}" for r in result["reasons"]) or "_The detector returned no reasons._"
    extra = ""
    if result.get("video"):
        v = result["video"]
        extra = f"\n\nVideo method: `{v['method']}` · {v['n_keyframes']} keyframes · calibrated: {'yes' if v['calibrated'] else 'no'}"
    return f"**Reasons**\n\n{items}{extra}\n\n_{result['disclaimer']}_"


def scan(path: str | None):
    if not path:
        return EMPTY, None, [], [], EMPTY
    try:
        result = detect(path)
    except DetectError as exc:
        return f"**Scan failed:** {exc}", None, [], [], EMPTY
    image = None
    if result["media_type"] == "image":
        image = annotate(path, result["faces"])
    return summary(result), image, specialist_rows(result), keyframe_rows(result), reasons_md(result)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="DFD detector") as demo:
        gr.Markdown(HEADER)
        gr.Markdown(f"_{backend_mode()}_")
        gr.Markdown(SIZE_NOTE)
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="Image or video", file_types=ACCEPT_TYPES, type="filepath")
                run = gr.Button("Scan", variant="primary")
            with gr.Column(scale=1):
                verdict_md = gr.Markdown(EMPTY)
                boxed = gr.Image(label="Face boxes (original pixels)", type="pil", interactive=False)
        specialists = gr.Dataframe(
            headers=["Specialist", "Applies to", "p_fake", "Verdict"],
            label="Specialists",
            interactive=False,
        )
        keyframes = gr.Dataframe(headers=["t (s)", "Verdict", "p_fake"], label="Video keyframes", interactive=False)
        reasons = gr.Markdown(EMPTY)
        run.click(scan, inputs=upload, outputs=[verdict_md, boxed, specialists, keyframes, reasons])
    return demo


if __name__ == "__main__":
    build_ui().launch()
