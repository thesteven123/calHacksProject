"""Detector client for the Gradio demo.

Set DFD_API_URL (and DFD_API_KEY) to call a hosted detector's POST /v1/detect.
Without DFD_API_URL the app runs a deterministic mock so the UI can be checked
before the model is hosted. The mock is labelled as such in the UI.
"""

import hashlib
import os
import random
import uuid
from pathlib import Path

import httpx

BACKEND_URL = os.environ.get("DFD_API_URL", "").rstrip("/")
API_KEY = os.environ.get("DFD_API_KEY", "")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".webm"}
ACCEPT_TYPES = [".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".webm"]
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MOCK_FAKE_AT = 0.9  # mock thresholds only; the real service uses calibrated ones
MOCK_REVIEW_AT = 0.5

LABELS = {
    "fake": "Likely AI-generated or manipulated",
    "review": "Unverified - needs human review",
    "real": "No manipulation detected",
}
FLAGS = {"fake": "ai_generated", "review": "needs_review", "real": "none"}
DISCLAIMER = "Research demo. Not for enforcement decisions; uncertain cases go to human review."


class DetectError(Exception):
    pass


def backend_mode() -> str:
    return f"Backend: {BACKEND_URL}" if BACKEND_URL else "Demo mode: mock detector (no model configured)"


def media_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise DetectError("Unsupported file type. Use JPG, PNG, WebP, MP4, MOV or WebM.")


def validate(path: str) -> str:
    kind = media_kind(path)
    size = Path(path).stat().st_size
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
    if size > limit:
        raise DetectError(f"File is larger than {limit // (1024 * 1024)} MB.")
    return kind


def detect(path: str) -> dict:
    validate(path)
    if BACKEND_URL:
        return _remote(path)
    return _mock(path)


def _remote(path: str) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    with open(path, "rb") as fh:
        files = {"file": (Path(path).name, fh.read())}
    try:
        res = httpx.post(f"{BACKEND_URL}/v1/detect", headers=headers, files=files, timeout=120)
    except httpx.HTTPError as exc:
        raise DetectError(f"Could not reach the detector: {exc}") from exc
    if res.status_code != 200:
        try:
            message = res.json().get("error", res.text)
        except ValueError:
            message = res.text
        raise DetectError(f"Detector returned {res.status_code}: {message}")
    return res.json()


def _verdict(p: float) -> str:
    return "fake" if p >= MOCK_FAKE_AT else "review" if p >= MOCK_REVIEW_AT else "real"


def _mock(path: str) -> dict:
    """Schema-valid stand-in for the real detector, seeded by file content."""
    data = Path(path).read_bytes()
    rng = random.Random(int(hashlib.sha256(data).hexdigest()[:16], 16))
    kind = media_kind(path)
    specialists = []
    general_p = round(rng.random(), 3)
    specialists.append(
        {
            "name": "general",
            "title": "General AI-image detector" if kind == "image" else "General AI-image detector (keyframes)",
            "applies_to": "whole image" if kind == "image" else "keyframes",
            "p_fake": general_p,
            "verdict": _verdict(general_p),
            "calibration_status": "mock",
        }
    )
    faces = []
    if kind == "image" and rng.random() < 0.5:
        face_p = round(rng.random(), 3)
        box = [40, 40, 120, 150]
        specialists.append(
            {
                "name": "face",
                "title": "AI-generated face detector",
                "applies_to": "face crop",
                "p_fake": face_p,
                "verdict": _verdict(face_p),
                "calibration_status": "mock",
                "box": box,
            }
        )
        faces.append({"box": box, "score": round(0.5 + rng.random() / 2, 3)})
    verdicts = [s["verdict"] for s in specialists]
    verdict = "fake" if "fake" in verdicts else "real" if all(v == "real" for v in verdicts) else "review"
    deciding = [s["p_fake"] for s in specialists if s["verdict"] == verdict]
    confidence = 1 - max(s["p_fake"] for s in specialists) if verdict == "real" else max(deciding)
    video = None
    if kind == "video":
        frames = [{"t": round(i * 0.75, 2), "verdict": _verdict(p), "p_fake": p} for i, p in enumerate(round(rng.random(), 3) for _ in range(8))]
        video = {"method": "keyframe_heuristic", "calibrated": False, "n_keyframes": len(frames), "frames": frames}
    reasons = [f"The {s['name']} specialist scores {s['p_fake']:.2f} (mock threshold {MOCK_FAKE_AT})." for s in specialists]
    return {
        "id": str(uuid.uuid4()),
        "media_type": kind,
        "verdict": verdict,
        "flag": FLAGS[verdict],
        "label": LABELS[verdict],
        "confidence": round(confidence, 3),
        "specialists": specialists,
        "faces": faces,
        "video": video,
        "reasons": reasons,
        "latency_ms": 0,
        "model_versions": {"general": "mock", "face": "mock"},
        "disclaimer": DISCLAIMER,
    }
