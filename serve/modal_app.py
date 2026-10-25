"""Demo serving backend (Modal app `dfd-demo`).

    modal serve serve/modal_app.py                 # dev
    modal deploy serve/modal_app.py                # prod (lead only)
    modal run serve/modal_app.py --action seed     # build /demo feed + media

Endpoints: `api` (Bearer-key /v1/detect, /v1/models, /v1/health) and `web` (cookie-auth /api/*
plus the SPA + /media from the dfd-demo volume). The GPU `Detector` cls fuses a general
AI-image detector with a YuNet-cropped AI-face specialist via dfd.inference.router.MultiDetector.
ASGI app factories live in serve/asgi_apps.py (imported inside the functions; fastapi is image-only).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
DIST = PROJECT / "demo" / "web" / "dist"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")  # MIT licence

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg", "curl")
    .uv_pip_install(  # pins mirror requirements.lock + web extras (all released >7d ago)
        "torch==2.14.1", "torchvision==0.29.1", "transformers==5.20.1", "datasets==5.1.0",
        "huggingface-hub==2.2.0", "timm==1.0.30", "pillow==13.0.0", "numpy==2.5.4",
        "scikit-learn==1.9.1", "pyyaml==6.0.3", "opencv-python-headless==5.0.0.93",
        "safetensors==0.8.0", "tqdm==4.70.1", "pyarrow==26.0.0",
        "fastapi==0.121.0", "python-multipart==0.0.20",
    )
    .run_commands(f"curl -fsSL {YUNET_URL} -o /opt/yunet.onnx")
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PROJECT / "src", f"{REMOTE}/src")
    .add_local_dir(PROJECT / "scripts", f"{REMOTE}/scripts")
    .add_local_dir(PROJECT / "serve", f"{REMOTE}/serve")
)
if DIST.is_dir():
    image = image.add_local_dir(str(DIST), "/app/static")

app = modal.App("dfd-demo", image=image)
runs_vol = modal.Volume.from_name("dfd-runs")
data_vol = modal.Volume.from_name("dfd-data", create_if_missing=True)
demo_vol = modal.Volume.from_name("dfd-demo", create_if_missing=True)
hf_cache = modal.Volume.from_name("dfd-hf-cache", create_if_missing=True)

GENERAL_CANDIDATES = ["/runs/general_all/config/best.pt",
                      "/runs/full7_dinov2_conformal/SD15-to-Midjourney/best.pt"]
FACE_CKPT = "/runs/face_t2i_b3/config/best.pt"
VIDEO_CKPT = "/runs/video_v3/config/best.pt"


def _hf_secret() -> list:
    token = os.environ.get("HF_TOKEN", "")
    env_file = PROJECT / ".env"
    if not token and modal.is_local() and env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("\"'")
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


def _asgi_module():
    sys.path.insert(0, f"{REMOTE}/serve")
    import asgi_apps

    return asgi_apps


@app.cls(gpu="L4", cpu=4.0, memory=16384, scaledown_window=600, min_containers=0,
         volumes={"/runs": runs_vol, "/data": data_vol, "/cache": hf_cache}, secrets=_hf_secret())
@modal.concurrent(max_inputs=4)
class Detector:
    @modal.enter()
    def load(self):
        from dfd.inference.pipeline import DetectionPipeline
        from dfd.inference.router import FaceDetector, MultiDetector, VideoClipScorer

        general_ckpt = next((p for p in GENERAL_CANDIDATES if Path(p).exists()), None)
        if general_ckpt is None:
            raise FileNotFoundError(f"no general checkpoint found under {GENERAL_CANDIDATES}")
        general = DetectionPipeline(general_ckpt, None, "auto", crop_mode="center")
        face = DetectionPipeline(FACE_CKPT, None, "auto", crop_mode="center") \
            if Path(FACE_CKPT).exists() else None
        video = VideoClipScorer(VIDEO_CKPT, "auto", face_onnx="/opt/yunet.onnx") \
            if Path(VIDEO_CKPT).exists() else None
        versions = {"general": Path(general_ckpt).parents[1].name,
                    "face": Path(FACE_CKPT).parents[1].name if face else None,
                    "video": Path(VIDEO_CKPT).parents[1].name if video else None}
        self.md = MultiDetector(general, face, FaceDetector("/opt/yunet.onnx"), versions,
                                video_scorer=video)

    @modal.method()
    def detect(self, data: bytes, filename: str) -> dict:
        import io
        import tempfile

        ext = Path(filename or "").suffix.lower()
        if ext in {".mp4", ".mov", ".webm"}:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(data)
                tmp = f.name
            try:
                return self.md.detect_video(tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
        from PIL import Image
        return self.md.detect_image(Image.open(io.BytesIO(data)).convert("RGB"))


@app.function(secrets=[modal.Secret.from_name("dfd-api")])
@modal.asgi_app()
def api():
    return _asgi_module().build_api(Detector)


@app.function(cpu=2.0, memory=1024, volumes={"/demo": demo_vol},
              secrets=[modal.Secret.from_name("dfd-api"), modal.Secret.from_name("dfd-demo")])
@modal.asgi_app()
def web():
    return _asgi_module().build_web(Detector, api_fn=api)


@app.function(cpu=4.0, memory=32768, timeout=4 * 3600,
              volumes={"/runs": runs_vol, "/data": data_vol, "/demo": demo_vol, "/cache": hf_cache},
              secrets=_hf_secret())
def seed(tag: str = "") -> dict:
    sys.path.insert(0, f"{REMOTE}/scripts")
    import demo_seed

    return demo_seed.run(detect=lambda data, name: Detector().detect.remote(data, name))


@app.local_entrypoint()
def main(action: str = "health", tag: str = ""):
    if action == "seed":
        print(json.dumps(seed.remote(tag), indent=2)[:4000])
        return
    raise SystemExit(f"Unknown action {action!r}; use `modal serve` for endpoints or `--action seed`")
