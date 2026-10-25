"""Vercel entry point: mounts the Gradio UI on a FastAPI app served at /."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from app import build_ui  # noqa: E402

app = gr.mount_gradio_app(FastAPI(), build_ui(), path="/")
