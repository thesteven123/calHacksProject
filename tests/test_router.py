"""Offline tests for the multi-specialist router (stub pipelines + face detector, no downloads)."""

import numpy as np
import pytest
from PIL import Image

from dfd.eval.calibration import Calibration
from dfd.inference.pipeline import Result
from dfd.inference.router import FaceDetector, MultiDetector


class _Pipe:
    """Stub DetectionPipeline: fixed verdict/p_fake, real Calibration for taus."""

    def __init__(self, verdict="real", p=0.1, status="matched"):
        self.res = Result("", "image", "stage2", verdict, p_fake=p)
        self.calibration_status = status
        self.cal = Calibration(tau_fake=0.95, tau_real=0.05, method="conformal")
        self.calls = []

    def score_pil(self, img, media_type="image"):
        self.calls.append(img.size)
        return self.res


class _Faces:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, img):
        return [{"box": b, "score": 0.9} for b in self.boxes]


def _img(size=(800, 600)):
    return Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))


def _md(gv="real", gp=0.1, fv="fake", fp=0.97, boxes=((100, 100, 200, 200),), face=True, status="matched"):
    return MultiDetector(_Pipe(gv, gp, status), _Pipe(fv, fp, status) if face else None,
                         _Faces(list(boxes)))


def test_fusion_truth_table():
    assert _md().detect_image(_img())["verdict"] == "fake"
    assert _md(fv="real", fp=0.02).detect_image(_img())["verdict"] == "real"
    assert _md(fv="abstain").detect_image(_img())["verdict"] == "review"
    assert _md(gv="fake", fv="abstain").detect_image(_img())["verdict"] == "fake"


def test_crop_geometry_and_clamping():
    face = _Pipe("fake", 0.9)
    md = MultiDetector(_Pipe("real", 0.1), face, _Faces([(500, 400, 180, 180)]))
    md.detect_image(_img((800, 600)))
    w, h = face.calls[0]
    assert w == h and w <= 800 and h <= 600  # square, clamped into the image
    # face crop = 1.8*180=324 centred on the box
    assert w == 324


def test_no_face_and_small_face_paths():
    md = _md(boxes=())
    r = md.detect_image(_img())
    assert len(r["specialists"]) == 1 and "No face found" in r["reasons"][-1]
    md = _md(boxes=((0, 0, 30, 30),))  # below the 64 px minimum
    r = md.detect_image(_img())
    assert len(r["specialists"]) == 1


def test_uncalibrated_specialist_is_review():
    r = _md(fv="fake", fp=0.99, status="mismatched").detect_image(_img())
    face = r["specialists"][1]
    assert face["verdict"] == "review" and face["calibration_status"] == "mismatched"
    assert r["verdict"] == "review"


def test_schema_keys_and_flags():
    r = _md().detect_image(_img())
    for k in ("id", "media_type", "verdict", "flag", "label", "confidence", "specialists",
              "faces", "video", "reasons", "latency_ms", "model_versions", "disclaimer"):
        assert k in r
    assert r["flag"] == "ai_generated" and r["media_type"] == "image" and r["video"] is None
    assert r["specialists"][1]["box"] == [100, 100, 200, 200]
    assert r["confidence"] == pytest.approx(0.97, abs=1e-3)


def test_video_aggregation(monkeypatch, tmp_path):
    import dfd.inference.router as router

    frames = [_img() for _ in range(8)]
    monkeypatch.setattr(router, "sample_keyframes",
                        lambda p, n_keyframes: (frames, list(range(8)), {"duration_s": 8.0}))
    verdicts = iter(["real", "real", "fake", "review", "fake", "real", "real", "real"])
    md = _md()
    monkeypatch.setattr(md, "detect_image",
                        lambda img: {**_md().detect_image(img), "verdict": next(verdicts)})
    r = md.detect_video(tmp_path / "v.mp4")
    assert r["verdict"] == "fake" and r["video"]["method"] == "keyframe_heuristic"
    assert r["video"]["calibrated"] is False and len(r["video"]["frames"]) == 8
    assert any("heuristic" in s for s in r["reasons"])

    monkeypatch.setattr(md, "detect_image",
                        lambda img: {**_md(fv="real").detect_image(img), "verdict": "real"})
    assert md.detect_video(tmp_path / "v.mp4")["verdict"] == "real"
    monkeypatch.setattr(md, "detect_image",
                        lambda img: {**_md(fv="abstain").detect_image(img), "verdict": "review"})
    assert md.detect_video(tmp_path / "v.mp4")["verdict"] == "review"


def test_video_too_long_raises(monkeypatch, tmp_path):
    import dfd.inference.router as router

    monkeypatch.setattr(router, "sample_keyframes",
                        lambda p, n_keyframes: ([_img()], [0.0], {"duration_s": 61.0}))
    with pytest.raises(ValueError, match="too long"):
        _md().detect_video(tmp_path / "v.mp4")
