"""Torch-free video frame sampling/extraction (runs on the CPU-only Modal ingest image).

Format-shortcut control: every clip, real or fake, is sampled by TIME (not frame index) and every frame gets the
same centre square crop -> INTER_AREA resize -> JPEG, so source resolution/fps are not label cues.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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


def encode_aligned(frame_bgr: np.ndarray, size: int = 256, quality: int = 95) -> bytes:
    """The shared per-frame alignment: centre square crop -> size x size INTER_AREA -> JPEG (cv2) bytes."""
    ok, buf = cv2.imencode(".jpg", square_resize(frame_bgr, size), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError("JPEG encode failed")
    return buf.tobytes()


def align_frame(img: Image.Image, size: int = 256, quality: int = 95) -> Image.Image:
    """PIL in/out version of the extractor's per-frame alignment (same cv2 path), e.g. for eval-time frames."""
    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    data = np.frombuffer(encode_aligned(bgr, size, quality), np.uint8)
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


def wdf_clips(index: list[dict], split: str, max_frames: int = 16) -> list[dict]:
    """WildDeepfake clips from an index_<split>.json: group by (kind, video, clip), sort by frame number and keep
    up to max_frames entries evenly spaced over the sorted list (unique). `frames` holds the chosen index entries."""
    by_clip: dict[tuple, list[dict]] = {}
    for e in index:
        by_clip.setdefault((e["label"], e["video"], e["clip"]), []).append(e)
    clips = []
    for (label, video, clip), entries in sorted(by_clip.items()):
        entries = sorted(entries, key=lambda e: e["frame"])
        picks = np.unique(np.linspace(0, len(entries) - 1, min(max_frames, len(entries))).round().astype(int))
        kind = "real" if label == 0 else "fake"
        # Video numbers restart per official split, so ids and groups must carry the split (else train/test collide).
        clips.append({"clip_id": f"wdf_{split}_{kind}_{video}_{clip}", "source": "wdf",
                      "method": "real" if label == 0 else "wilddeepfake", "split": split, "label": int(label),
                      "video_group": f"wdf/{split}/{kind}/{video}", "frames": [entries[i] for i in picks]})
    return clips


def extract_wdf_frames(root: str | Path, frames: list[dict], size: int = 256, quality: int = 95) -> list[bytes]:
    """Read PNG frames at their indexed tar offsets under `root` and align them like video frames."""
    out, handles = [], {}
    try:
        for e in frames:
            f = handles.setdefault(e["tar"], open(Path(root) / e["tar"], "rb"))
            f.seek(e["offset"])
            img = cv2.imdecode(np.frombuffer(f.read(e["size"]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Cannot decode {e['tar']}:{e['member']}")
            out.append(encode_aligned(img, size, quality))
    finally:
        for f in handles.values():
            f.close()
    return out


YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")  # MIT licence; same model as serve/ (dfd.inference.router)
_YUNET: dict[tuple[str, int], object] = {}


def yunet(onnx_path: str | Path, score_threshold: float = 0.8):
    """Per-process cached cv2.FaceDetectorYN (cv2 objects cannot be pickled to pool workers)."""
    import os

    key = (str(onnx_path), os.getpid())
    if key not in _YUNET:
        _YUNET[key] = cv2.FaceDetectorYN.create(str(onnx_path), "", (320, 320), score_threshold)
    return _YUNET[key]


def largest_face(detector, frame_bgr: np.ndarray) -> list[int] | None:
    """[x, y, w, h] of the largest detected face in original pixels, or None."""
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0:
        return None
    x, y, bw, bh = max(faces, key=lambda f: float(f[2]) * float(f[3]))[:4]
    return [int(round(float(v))) for v in (x, y, bw, bh)]


def face_square_crop(frame: np.ndarray, box: list[int], scale: float = 1.3) -> np.ndarray:
    """Square crop centred on the box with side max(w, h) * scale, shifted (not padded) to stay inside the frame."""
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    side = min(int(round(max(bw, bh) * scale)), h, w)
    cx, cy = x + bw / 2, y + bh / 2
    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    return frame[top:top + side, left:left + side]


def extract_clip_frames(path: str | Path, timestamps: list[float], size: int = 256, quality: int = 95,
                        fps: float | None = None, face_onnx: str | Path | None = None,
                        stats: dict | None = None) -> list[bytes]:
    """JPEG bytes of the frames at `timestamps` (frame index = floor(t * fps), decoded sequentially, which is
    more reliable than container seeking). One output per timestamp; raises on an unreadable clip.

    face_onnx: crop each frame to the largest YuNet face (x1.3 square) before the shared alignment; frames without
    a face fall back to the centre crop and are counted in stats["face_fallback"]."""
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
    # Indices past the decodable end (metadata overcount) fall back to the last decoded frame.
    picked = [frames.get(i, last) for i in wanted]
    if face_onnx:
        det, fallback = yunet(face_onnx), 0
        for k, frame in enumerate(picked):
            box = largest_face(det, frame)
            if box is None:
                fallback += 1
            else:
                picked[k] = face_square_crop(frame, box)
        if stats is not None:
            stats["face_fallback"] = stats.get("face_fallback", 0) + fallback
    return [encode_aligned(f, size, quality) for f in picked]


def extract_clip_job(job: tuple[dict, float, int, int, int]) -> tuple[dict, list[bytes] | None, str | None]:
    """Pool worker for scripts/extract_video_frames.py: (clip, fps_sample, max_frames, size, quality) ->
    (clips.json metadata, JPEG frames, None) or (clip, None, error). Lives here so spawn-based pools can import it."""
    clip, fps_sample, max_frames, size, quality = job
    try:
        if clip["source"] == "wdf":  # pre-extracted PNG frames: clip["frames"] index entries, clip["root"] tar dir
            frames = extract_wdf_frames(clip["root"], clip["frames"], size, quality)
            meta = {k: v for k, v in clip.items() if k not in ("frames", "root")}
            meta["n_frames"] = len(frames)
            return meta, frames, None
        info = probe_video(clip["path"])
        stats: dict = {}
        frames = extract_clip_frames(clip["path"], video_timestamps(info["duration_s"], fps_sample, max_frames),
                                     size, quality, fps=info["fps"], face_onnx=clip.get("face_onnx"), stats=stats)
        meta = {k: v for k, v in clip.items() if k not in ("path", "face_onnx")}
        meta.update(n_frames=len(frames), duration_s=round(info["duration_s"], 3), src_fps=round(info["fps"], 3),
                    src_size=[info["width"], info["height"]])
        if clip.get("face_onnx"):
            meta["face_fallback_frames"] = stats.get("face_fallback", 0)
        return meta, frames, None
    except Exception as e:  # noqa: BLE001 - counted and logged by the caller; extraction continues
        return clip, None, f"{type(e).__name__}: {e}"
