"""Torch-free video frame sampling/extraction (runs on the CPU-only Modal ingest image).

Format-shortcut control: every clip, real or fake, is sampled by TIME (not frame index) and every frame gets the
same centre square crop -> INTER_AREA resize -> JPEG, so source resolution/fps are not label cues.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def video_timestamps(duration_s: float, fps_sample: float = 4.0, max_frames: int = 16) -> list[float]:
    """Fixed-rate stamps from t=0 (at most max_frames); clips shorter than max_frames/fps_sample get max_frames
    stamps spread uniformly over [0, duration)."""
    if duration_s >= max_frames / fps_sample:
        return [k / fps_sample for k in range(max_frames)]
    return [k * duration_s / max_frames for k in range(max_frames)]


def probe_video(path: str | Path) -> dict:
    """{fps, n_frames, width, height, duration_s} from container metadata; raises if unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or n <= 0:
        raise IOError(f"Bad video metadata for {path}: fps={fps} frames={n}")
    return {"fps": fps, "n_frames": n, "width": w, "height": h, "duration_s": n / fps}


def square_resize(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    c = min(h, w)
    top, left = (h - c) // 2, (w - c) // 2
    return cv2.resize(frame[top:top + c, left:left + c], (size, size), interpolation=cv2.INTER_AREA)


def extract_clip_frames(path: str | Path, timestamps: list[float], size: int = 256, quality: int = 95,
                        fps: float | None = None) -> list[bytes]:
    """JPEG bytes of the frames at `timestamps` (frame index = floor(t * fps), decoded sequentially, which is
    more reliable than container seeking). One output per timestamp; raises on an unreadable clip."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video {path}")
        fps = fps or float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise IOError(f"Unknown fps for {path}")
        wanted = [math.floor(t * fps + 1e-6) for t in timestamps]
        need, frames, idx, last = set(wanted), {}, 0, None
        while idx <= max(wanted):
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
            if idx in need:
                frames[idx] = frame
            idx += 1
    finally:
        cap.release()
    if last is None:
        raise IOError(f"No decodable frames in {path}")
    out = []
    for i in wanted:  # indices past the decodable end (metadata overcount) fall back to the last decoded frame
        ok, buf = cv2.imencode(".jpg", square_resize(frames.get(i, last), size), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise IOError(f"JPEG encode failed for {path} frame {i}")
        out.append(buf.tobytes())
    return out


def extract_clip_job(job: tuple[dict, float, int, int, int]) -> tuple[dict, list[bytes] | None, str | None]:
    """Pool worker for scripts/extract_video_frames.py: (clip, fps_sample, max_frames, size, quality) ->
    (clips.json metadata, JPEG frames, None) or (clip, None, error). Lives here so spawn-based pools can import it."""
    clip, fps_sample, max_frames, size, quality = job
    try:
        info = probe_video(clip["path"])
        frames = extract_clip_frames(clip["path"], video_timestamps(info["duration_s"], fps_sample, max_frames),
                                     size, quality, fps=info["fps"])
        meta = {k: v for k, v in clip.items() if k != "path"}
        meta.update(n_frames=len(frames), duration_s=round(info["duration_s"], 3), src_fps=round(info["fps"], 3),
                    src_size=[info["width"], info["height"]])
        return meta, frames, None
    except Exception as e:  # noqa: BLE001 - counted and logged by the caller; extraction continues
        return clip, None, f"{type(e).__name__}: {e}"
