"""Tiered inference funnel (docs/implementation-architecture.md §5–6).

Stage 0: perceptual-hash lookup against known real/fake media + provenance-manifest presence flag.
Stage 2: core detector (image, or keyframes + top-k pooling for video) -> calibrated 3-way verdict.
Abstentions are marked `escalate=True` for the Stage 3 ensemble / human review queue.
Stage 1 (distilled student) plugs in via `stage1` once trained (S6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image

from dfd.config import PTEConfig
from dfd.data.transforms import DetectorTransform
from dfd.eval.calibration import Calibration
from dfd.inference.crops import MultiCropTransform, pool_crops
from dfd.inference.video import sample_keyframes
from dfd.models.detector import DeepfakeDetector
from dfd.models.temporal import topk_pool
from dfd.training.trainer import resolve_device

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def dhash(img: Image.Image) -> int:
    gray = cv2.resize(np.asarray(img.convert("L")), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def has_provenance_manifest(path: str | Path) -> bool:
    """Presence check only (JUMBF/C2PA box markers). Cryptographic verification needs the c2pa SDK."""
    with open(path, "rb") as f:
        head = f.read(4 * 1024 * 1024)
    return b"c2pa" in head or b"jumb" in head


@dataclass
class Result:
    path: str
    media_type: str
    stage: str
    verdict: str
    p_fake: float | None = None
    logit: float | None = None
    escalate: bool = False
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DetectionPipeline:
    def __init__(self, checkpoint: str | Path, calibration: str | Path | None = None, device: str = "auto",
                 known_hashes: dict[int, str] | None = None, hash_radius: int = 6,
                 stage1: Callable[[torch.Tensor], float] | None = None, stage1_threshold: float = 0.02,
                 crop_mode: str = "center", crop_grid: int = 3, crop_k_frac: float = 0.25):
        if crop_mode not in ("center", "multi"):
            raise ValueError(f"crop_mode must be 'center' or 'multi', got {crop_mode!r}")
        self.device = resolve_device(device)
        self.model, _ = DeepfakeDetector.load(checkpoint, map_location="cpu")
        self.model.to(self.device).eval()
        self.crop_mode, self.crop_k_frac = crop_mode, crop_k_frac
        # Multi-crop top-k scores sit higher than centre-crop scores, so thresholds must be fit per mode.
        # Automatic real/fake verdicts are only issued with thresholds fit for THIS crop mode; otherwise every
        # verdict becomes "abstain" (CODE_REVIEW 2026-10-24: mismatched or missing calibration).
        default_name = "calibration_multicrop.json" if crop_mode == "multi" else "calibration.json"
        cal_path = Path(calibration) if calibration else Path(checkpoint).with_name(default_name)
        if not cal_path.exists() and not calibration and crop_mode == "multi":
            cal_path = Path(checkpoint).with_name("calibration.json")  # better-than-nothing scale, still abstains
        if cal_path.exists():
            self.cal = Calibration.load(cal_path)
            self.calibration_status = "matched" if self.cal.crop_mode == crop_mode else "mismatched"
        else:
            self.cal, self.calibration_status = Calibration(), "missing"
        no_pte = PTEConfig(enabled=False)
        self.transform = DetectorTransform(no_pte, self.model.cfg.image_size, train=False)
        self.multicrop = MultiCropTransform(no_pte, self.model.cfg.image_size, crop_grid)
        self.known_hashes = known_hashes or {}
        self.hash_radius = hash_radius
        self.stage1, self.stage1_threshold = stage1, stage1_threshold

    def _tensor(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img, np.random.default_rng(0))[0]

    def _score_image(self, img: Image.Image) -> tuple[float, dict]:
        """Image logit under the configured crop mode, plus crop diagnostics."""
        if self.crop_mode == "center":
            return float(self.model(self._tensor(img).unsqueeze(0).to(self.device))["logit"].item()), {"n_crops": 1}
        crops = self.multicrop(img, np.random.default_rng(0))[0].to(self.device)
        crop_logits = self.model(crops)["logit"].float().cpu()
        return float(pool_crops(crop_logits, self.crop_k_frac)), {
            "n_crops": len(crops), "max_crop_logit": round(float(crop_logits.max()), 3),
        }

    def _stage0(self, path: str, img: Image.Image, media_type: str) -> Result | None:
        h = dhash(img)
        for known, label in self.known_hashes.items():
            if bin(h ^ known).count("1") <= self.hash_radius:
                return Result(path, media_type, "stage0_hash", label, details={"dhash": f"{h:016x}"})
        return None

    def _finalize(self, path: str, media_type: str, logit: float, details: dict) -> Result:
        p = float(self.cal.prob(logit))
        verdict = self.cal.verdict(p)
        details = {**details, "crop_mode": self.crop_mode, "calibration_status": self.calibration_status}
        if self.calibration_status != "matched":
            # Thresholds were not fit on this score distribution: neither FPR nor FNR is controlled.
            verdict = "abstain"
        return Result(path, media_type, "stage2", verdict, p, logit, verdict == "abstain", details)

    @torch.no_grad()
    def predict_image(self, path: str | Path) -> Result:
        path = str(path)
        img = Image.open(path).convert("RGB")
        details = {"provenance_manifest": has_provenance_manifest(path)}
        if (hit := self._stage0(path, img, "image")) is not None:
            return hit
        if self.stage1 is not None and self.stage1(self._tensor(img).unsqueeze(0).to(self.device)) < self.stage1_threshold:
            return Result(path, "image", "stage1", "real", details=details)
        logit, crop_info = self._score_image(img)
        return self._finalize(path, "image", logit, {**details, **crop_info})

    @torch.no_grad()
    def score_pil(self, img: Image.Image, media_type: str = "image") -> Result:
        """Stage-2 score + calibrated verdict for an in-memory image (no stage0/stage1)."""
        logit, crop_info = self._score_image(img)
        return self._finalize("", media_type, logit, crop_info)

    @torch.no_grad()
    def predict_video(self, path: str | Path, n_keyframes: int = 16) -> Result:
        path = str(path)
        if self.crop_mode == "multi" and self.model.temporal is not None:
            # A temporal adapter needs one aligned crop per frame; multi-crop would silently be ignored.
            raise ValueError("crop_mode='multi' is not supported for video with a temporal checkpoint; use 'center'")
        frames, times, vid_info = sample_keyframes(path, n_keyframes=n_keyframes)
        if (hit := self._stage0(path, frames[0], "video")) is not None:
            return hit
        if self.crop_mode == "multi":
            # No temporal adapter: each keyframe is an independent image, so score it multi-crop.
            frame_logits_t = torch.tensor([self._score_image(f)[0] for f in frames])
            logit = float(topk_pool(frame_logits_t.unsqueeze(0), None, self.model.cfg.topk_frac)[0])
        else:
            x = torch.stack([self._tensor(f) for f in frames]).unsqueeze(0).to(self.device)
            out = self.model(x)
            frame_logits_t, logit = out["frame_logits"][0].float().cpu(), float(out["logit"].item())
        frame_logits = frame_logits_t.tolist()
        top = int(np.argmax(frame_logits))
        details = {
            "provenance_manifest": has_provenance_manifest(path),
            "n_keyframes": len(frames),
            "most_suspicious_t": round(times[top], 2),
            "frame_logits": [round(v, 3) for v in frame_logits],
            **vid_info,
        }
        return self._finalize(path, "video", logit, details)

    def predict(self, path: str | Path) -> Result:
        return self.predict_video(path) if Path(path).suffix.lower() in VIDEO_EXTS else self.predict_image(path)
