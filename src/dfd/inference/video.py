"""Video decoding with change-aware keyframe selection (OpenCV; swap for NVDEC/VideoToolbox in prod).

Two passes over the file:
1. Thumbnail pass: keep only a 32x32 grayscale thumb per sampled frame, strided so candidates cover the
   WHOLE duration (frame-count hint; if the container lies, candidates are halved and the stride doubles
   so coverage still extends to the end). Fixes the review finding that only the first ~64 s was scanned.
2. Decode pass: grab sequentially and decode pixels only for the chosen keyframes.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from dfd.models.temporal import select_keyframes

_THUMB = (32, 32)


def sample_keyframes(path: str, sample_fps: float = 4.0, max_candidates: int = 256,
                     n_keyframes: int = 16) -> tuple[list[Image.Image], list[float], dict]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(native_fps / sample_fps))
    if n_frames > 0:
        step = max(step, -(-n_frames // max_candidates))  # stride wide enough to reach the end

    thumbs: list[np.ndarray] = []
    times: list[float] = []
    frame_nums: list[int] = []  # source frame index of each candidate (needed when step doubles)
    i = 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                thumbs.append(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), _THUMB, interpolation=cv2.INTER_AREA))
                times.append(i / native_fps)
                frame_nums.append(i)
                if len(thumbs) > max_candidates:
                    thumbs, times, frame_nums = thumbs[::2], times[::2], frame_nums[::2]
                    step *= 2
        i += 1
    cap.release()
    if not thumbs:
        raise ValueError(f"No frames decoded from {path}")
    duration_s = i / native_fps

    wanted = {frame_nums[j] for j in select_keyframes(np.stack(thumbs), n_keyframes)}
    frames: dict[int, Image.Image] = {}
    cap = cv2.VideoCapture(path)
    i = 0
    while cap.grab() and wanted:
        if i in wanted:
            ok, bgr = cap.retrieve()
            if ok:
                frames[i] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if i > max(wanted):
            break
        i += 1
    cap.release()
    info = {
        "duration_s": round(duration_s, 2),
        "sampled_fps": round(native_fps / step, 3),
        "n_candidates": len(times),
        "coverage_frac": round(times[-1] / max(duration_s, 1e-9), 4),
    }
    # frame_nums is ascending, so this keeps frames/times aligned even if some decodes failed
    pairs = [(frames[n], times[j]) for j, n in enumerate(frame_nums) if n in wanted and n in frames]
    return [f for f, _ in pairs], [t for _, t in pairs], info
