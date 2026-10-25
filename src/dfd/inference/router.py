"""Multi-specialist detection router for the demo serving backend (API_CONTRACT.md).

General image detector + optional AI-face specialist on a YuNet face crop. Fusion:
"fake" if any specialist verdict is fake; "real" only if every specialist that ran says real;
otherwise "review". Pipeline "abstain" maps to "review"; an uncalibrated specialist (calibration
status != "matched") is always "review".
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dfd.inference.pipeline import DetectionPipeline
from dfd.inference.video import sample_keyframes


class _ListClipDataset:
    """1-item clip dataset: base[idx] returns (list_of_PIL_frames, 1, 'video') for TransformedDataset._clip."""

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return 1

    def meta(self, idx):
        return 1, "video"

    def __getitem__(self, idx):
        return self.frames, 1, "video"


class VideoClipScorer:
    """Face-swap video specialist (temporal model, WildDeepfake-trained, e.g. video_v2b_r2).

    Scores a whole clip through the SAME preprocessing as training/eval: YuNet 1.3x face crops
    -> encode_aligned -> pick_clip_frames + DetectorTransform(train=False) via a 1-item
    TransformedDataset. Returns None when fewer than half the sampled frames have a face
    (the model only knows face crops).
    """

    def __init__(self, checkpoint: str | Path, device: str = "auto", n_frames: int = 8,
                 face_onnx: str | Path | None = "/opt/yunet.onnx"):
        import torch

        from dfd.config import Config
        from dfd.data.datasets import TransformedDataset
        from dfd.data.transforms import DetectorTransform
        from dfd.eval.calibration import Calibration
        from dfd.models.detector import DeepfakeDetector
        from dfd.training.trainer import resolve_device

        self.device = resolve_device(device)
        self.model, _ = DeepfakeDetector.load(checkpoint, map_location="cpu")
        self.model.to(self.device).eval()
        self.n_frames = n_frames
        self.face_onnx = face_onnx
        cal_path = Path(checkpoint).with_name("calibration.json")
        if cal_path.exists():
            self.cal = Calibration.load(cal_path)
            self.calibration_status = "matched" if self.cal.crop_mode == "center" else "mismatched"
        else:
            self.cal, self.calibration_status = Calibration(), "missing"
        cfg_yaml = Path(checkpoint).with_name("config.yaml")
        from dfd.config import PTEConfig
        pte = Config.from_yaml(cfg_yaml).data.pte if cfg_yaml.exists() else PTEConfig(enabled=False)
        self._transform = DetectorTransform(pte, self.model.cfg.image_size, train=False)
        self._dataset_cls = TransformedDataset
        self._torch = torch

    def _tensor(self, frames):
        # Reuse TransformedDataset._clip (pick_clip_frames + per-frame DetectorTransform) on a
        # 1-item stub dataset so preprocessing is identical to evaluation. Built per call: the
        # Modal class serves concurrent inputs on threads, so no per-request state is shared.
        item = self._dataset_cls(_ListClipDataset(frames), self._transform, seed=0,
                                 frames_per_clip=self.n_frames)[0]
        return item["pixels"].unsqueeze(0).to(self.device), item["mask"].unsqueeze(0).to(self.device)

    def score(self, path: str | Path) -> dict | None:
        import io

        from dfd.data.video_frames import extract_clip_frames, probe_video, video_timestamps

        info = probe_video(path)
        stats: dict = {}
        jpgs = extract_clip_frames(path, video_timestamps(info["duration_s"], 4.0, 16), 256, 95,
                                   fps=info["fps"], face_onnx=self.face_onnx, stats=stats)
        if stats.get("face_fallback", 0) > len(jpgs) / 2:
            return None
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpgs]
        pixels, mask = self._tensor(frames)
        with self._torch.no_grad():
            out = self.model(pixels, mask)
        p = float(self.cal.prob(out["logit"].item()))
        verdict = "review" if self.cal.verdict(p) == "abstain" or self.calibration_status != "matched" \
            else self.cal.verdict(p)
        return {"name": "video", "title": "Face-swap video model",
                "applies_to": "whole clip (face track)", "p_fake": round(p, 4),
                "verdict": verdict, "calibration_status": self.calibration_status}

FLAG_LABEL = {
    "fake": ("ai_generated", "Likely AI-generated or manipulated"),
    "review": ("needs_review", "Unverified - needs human review"),
    "real": ("none", "No manipulation detected"),
}
DISCLAIMER = "Research demo. Not for enforcement decisions; uncertain cases go to human review."

MIN_FACE_PX = 64
MIN_FACE_FRAC = 0.01
FACE_CROP_SCALE = 1.8
N_KEYFRAMES = 8
VIDEO_MAX_S = 60.0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024


class FaceDetector:
    """YuNet (cv2.FaceDetectorYN) bounding boxes in original pixels, largest first."""

    def __init__(self, onnx_path: str | Path, score_threshold: float = 0.8):
        self.det = cv2.FaceDetectorYN.create(str(onnx_path), "", (320, 320), score_threshold)

    def detect(self, img: Image.Image) -> list[dict]:
        arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = arr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(arr)
        out = [{"box": [int(round(float(v))) for v in f[:4]], "score": float(f[-1])}
               for f in (faces if faces is not None else [])]
        out.sort(key=lambda d: d["box"][2] * d["box"][3], reverse=True)
        return out


class MultiDetector:
    def __init__(self, general: DetectionPipeline, face: DetectionPipeline | None,
                 face_detector: FaceDetector | None, versions: dict | None = None,
                 video_scorer=None):
        self.general, self.face, self.face_detector = general, face, face_detector
        self.versions = versions or {}
        self.video_scorer = video_scorer

    def _specialist(self, pipe, name, title, applies_to, res, box=None) -> dict:
        verdict = "review" if res.verdict == "abstain" else res.verdict
        if pipe.calibration_status != "matched":
            verdict = "review"
        row = {"name": name, "title": title, "applies_to": applies_to,
               "p_fake": round(res.p_fake, 4) if res.p_fake is not None else None,
               "verdict": verdict, "calibration_status": pipe.calibration_status}
        if box is not None:
            row["box"] = list(box)
        return row

    @staticmethod
    def _reason(subject: str, row: dict, cal) -> str:
        return (f"The {subject} scores {row['p_fake']:.2f} on the {row['title'].lower()} "
                f"(fake >= {cal.tau_fake:.3f}, real < {cal.tau_real:.3f}).")

    @staticmethod
    def _fuse(rows: list[dict]) -> str:
        verdicts = [r["verdict"] for r in rows]
        if "fake" in verdicts:
            return "fake"
        if verdicts and all(v == "real" for v in verdicts):
            return "real"
        return "review"

    def _crop_square(self, img: Image.Image, box) -> Image.Image:
        x, y, w, h = box
        side = int(round(FACE_CROP_SCALE * max(w, h)))
        cx, cy = x + w / 2, y + h / 2
        W, H = img.size
        side = min(side, W, H)
        left = int(round(min(max(cx - side / 2, 0), W - side)))
        top = int(round(min(max(cy - side / 2, 0), H - side)))
        return img.crop((left, top, left + side, top + side))

    def detect_image(self, img: Image.Image) -> dict:
        t0 = time.time()
        img = img.convert("RGB")
        gen_res = self.general.score_pil(img)
        specialists = [self._specialist(self.general, "general", "General AI-image detector",
                                      "whole image", gen_res)]
        reasons = [self._reason("whole image", specialists[0], self.general.cal)]
        faces = self.face_detector.detect(img) if self.face_detector else []
        chosen = None
        if self.face is not None and faces:
            W, H = img.size
            big = faces[0]
            bw, bh = big["box"][2], big["box"][3]
            if min(bw, bh) >= MIN_FACE_PX and bw * bh >= MIN_FACE_FRAC * W * H:
                chosen = big
        if chosen is not None:
            crop = self._crop_square(img, chosen["box"])
            face_res = self.face.score_pil(crop)
            specialists.append(self._specialist(self.face, "face", "AI-generated face detector",
                                                "face crop", face_res, box=chosen["box"]))
            reasons.append(self._reason("face crop", specialists[-1], self.face.cal))
        elif self.face is not None:
            reasons.append("No face found; only the general detector ran.")
        verdict = self._fuse(specialists)
        flag, label = FLAG_LABEL[verdict]
        p_max = max(r["p_fake"] or 0.0 for r in specialists)
        confidence = p_max if verdict == "fake" else (1.0 - p_max if verdict == "real" else p_max)
        return {
            "id": str(uuid.uuid4()),
            "media_type": "image",
            "verdict": verdict,
            "flag": flag,
            "label": label,
            "confidence": round(float(confidence), 4),
            "specialists": specialists,
            "faces": faces,
            "video": None,
            "reasons": reasons,
            "latency_ms": int(round((time.time() - t0) * 1000)),
            "model_versions": {"general": self.versions.get("general"),
                               "face": self.versions.get("face") if self.face else None},
            "disclaimer": DISCLAIMER,
        }

    def detect_video(self, path: str | Path) -> dict:
        t0 = time.time()
        frames, times, info = sample_keyframes(path, n_keyframes=N_KEYFRAMES)
        if info.get("duration_s", 0) > VIDEO_MAX_S:
            raise ValueError(f"video too long: {info['duration_s']:.1f}s > {VIDEO_MAX_S:.0f}s")
        per_frame = [self.detect_image(f) for f in frames]
        frame_rows = []
        for t, r in zip(times, per_frame):
            frame_rows.append({"t": round(float(t), 2), "verdict": r["verdict"],
                               "p_fake": max(s["p_fake"] or 0.0 for s in r["specialists"])})
        # Face-swap video specialist on the whole clip (None = no face track found / no scorer).
        vrow = self.video_scorer.score(path) if self.video_scorer else None
        n_fake = sum(1 for f in frame_rows if f["verdict"] == "fake")
        if (vrow and vrow["verdict"] == "fake") or n_fake >= 2:
            verdict = "fake"
        elif vrow and vrow["verdict"] == "real" and all(f["verdict"] == "real" for f in frame_rows):
            verdict = "real"
        else:
            verdict = "review"
        # per-specialist: max p_fake across keyframes, verdict taken at that frame
        names = [s["name"] for s in per_frame[0]["specialists"]]
        specialists = []
        for name in names:
            best_row, best_frame = None, 0
            best_p = -1.0
            for fi, fr in enumerate(per_frame):
                row = next((s for s in fr["specialists"] if s["name"] == name), None)
                if row and (row["p_fake"] or 0.0) > best_p:
                    best_p, best_row, best_frame = row["p_fake"] or 0.0, row, fi
            if best_row is not None:
                row = dict(best_row)
                if "box" in row:
                    row["box"] = per_frame[best_frame]["specialists"][
                        [s["name"] for s in per_frame[best_frame]["specialists"]].index(name)
                    ].get("box")
                specialists.append(row)
        top = int(np.argmax([f["p_fake"] for f in frame_rows]))
        faces = per_frame[top]["faces"]
        reasons = [f"{f['verdict']} ({f['p_fake']:.2f}) @ {f['t']}s" for f in frame_rows]
        reasons = ["Per-keyframe verdicts: " + "; ".join(reasons),
                   "The video model is trained on face swaps (WildDeepfake); it does not cover "
                   "talking-head animation."]
        if self.video_scorer is None or vrow is None:
            reasons += ["Video verdict is a frame-level heuristic (not clip-calibrated).",
                        "No face track found; the face-swap video model was not applied."]
            method, calibrated = "keyframe_heuristic", False
        else:
            specialists.append(vrow)
            method, calibrated = "clip_model+keyframes", True
        flag, label = FLAG_LABEL[verdict]
        p_max = max([f["p_fake"] for f in frame_rows] + ([vrow["p_fake"]] if vrow else []))
        confidence = p_max if verdict == "fake" else (1.0 - p_max if verdict == "real" else p_max)
        return {
            "id": str(uuid.uuid4()),
            "media_type": "video",
            "verdict": verdict,
            "flag": flag,
            "label": label,
            "confidence": round(float(confidence), 4),
            "specialists": specialists,
            "faces": faces,
            "video": {"method": method, "calibrated": calibrated,
                      "n_keyframes": len(frame_rows), "frames": frame_rows},
            "reasons": reasons,
            "latency_ms": int(round((time.time() - t0) * 1000)),
            "model_versions": {"general": self.versions.get("general"),
                               "face": self.versions.get("face") if self.face else None,
                               "video": self.versions.get("video") if self.video_scorer else None},
            "disclaimer": DISCLAIMER,
        }
