import json

import numpy as np
import torch
from PIL import Image

from dfd.config import ModelConfig, PTEConfig
from dfd.inference.crops import MultiCropTransform, crop_boxes, pool_crops
from dfd.inference.pipeline import DetectionPipeline
from dfd.models.detector import DeepfakeDetector

TINY = dict(backbone="random-tiny", feature_layers=(2, 4), lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8)


def test_crop_boxes_reach_every_corner_of_large_image():
    boxes = crop_boxes(1000, 600, 224, grid=3)
    xs0, ys0 = {b[0] for b in boxes}, {b[1] for b in boxes}
    assert 0 in xs0 and 1000 - 224 in xs0 and 0 in ys0 and 600 - 224 in ys0
    assert all(b[2] - b[0] == 224 and b[3] - b[1] == 224 for b in boxes)
    assert ((1000 - 224) // 2, (600 - 224) // 2, (1000 - 224) // 2 + 224, (600 - 224) // 2 + 224) in boxes


def test_crop_boxes_single_crop_when_image_equals_crop():
    assert crop_boxes(224, 224, 224) == [(0, 0, 224, 224)]


def test_multicrop_finds_off_centre_signal_that_centre_crop_misses():
    """Stand-in scorer = mean brightness. A bright patch in a corner must dominate the pooled score."""
    arr = np.zeros((672, 672, 3), dtype=np.uint8)
    arr[:200, :200] = 255  # off-centre "manipulation"
    crops, _ = MultiCropTransform(PTEConfig(enabled=False), 224, grid=3)(Image.fromarray(arr), np.random.default_rng(0))
    crop_scores = crops.mean(dim=(1, 2, 3))
    centre_score = crop_scores[-1] if len(crops) == 10 else crop_scores[4]
    assert float(centre_score) == 0.0
    assert float(pool_crops(crop_scores, k_frac=0.25)) > 0.2


def test_pool_crops_is_topk_mean():
    assert float(pool_crops(torch.tensor([0.0, 1.0, 5.0, 3.0]), k_frac=0.5)) == 4.0


def _ckpt(tmp_path):
    torch.manual_seed(0)
    DeepfakeDetector(ModelConfig(**TINY)).save(tmp_path / "m.pt")
    img = tmp_path / "big.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (500, 700, 3), dtype=np.uint8)).save(img)
    return tmp_path / "m.pt", img


FORCE_FAKE = {"temperature": 1.0, "tau_real": -1.0, "tau_fake": -0.5, "target_fpr": 0.01, "target_fnr": 0.02}
FORCE_REAL = {"temperature": 1.0, "tau_real": 2.0, "tau_fake": 3.0, "target_fpr": 0.01, "target_fnr": 0.02}


def test_pipeline_multicrop_abstains_on_centre_calibration_for_fake_and_real(tmp_path):
    """Mismatched thresholds must never yield an automatic verdict in EITHER direction (CODE_REVIEW 2026-10-24)."""
    ckpt, img = _ckpt(tmp_path)
    for forced in (FORCE_FAKE, FORCE_REAL):
        (tmp_path / "calibration.json").write_text(json.dumps(forced))  # only centre-crop calibration exists
        r = DetectionPipeline(ckpt, device="cpu", crop_mode="multi").predict(img)
        assert r.details["crop_mode"] == "multi" and r.details["n_crops"] > 1
        assert r.details["calibration_status"] == "mismatched"
        assert r.verdict == "abstain" and r.escalate


def test_pipeline_without_any_calibration_abstains_and_reports_missing(tmp_path):
    ckpt, img = _ckpt(tmp_path)
    for mode in ("center", "multi"):
        r = DetectionPipeline(ckpt, device="cpu", crop_mode=mode).predict(img)
        assert r.details["calibration_status"] == "missing" and r.verdict == "abstain"


def test_pipeline_uses_multicrop_calibration_when_present(tmp_path):
    ckpt, img = _ckpt(tmp_path)
    for name in ("calibration.json", "calibration_multicrop.json"):  # legacy files: mode inferred from name
        (tmp_path / name).write_text(json.dumps(FORCE_FAKE))
    r = DetectionPipeline(ckpt, device="cpu", crop_mode="multi").predict(img)
    assert r.details["calibration_status"] == "matched" and r.verdict == "fake"


def test_explicit_calibration_file_with_wrong_mode_is_mismatched(tmp_path):
    ckpt, img = _ckpt(tmp_path)
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({**FORCE_FAKE, "crop_mode": "center"}))
    r = DetectionPipeline(ckpt, calibration=path, device="cpu", crop_mode="multi").predict(img)
    assert r.details["calibration_status"] == "mismatched" and r.verdict == "abstain"


def test_temporal_checkpoint_rejects_multicrop_video(tmp_path):
    import cv2
    import pytest

    torch.manual_seed(0)
    DeepfakeDetector(ModelConfig(**TINY, temporal=True, temporal_dim=16)).save(tmp_path / "t.pt")
    vid = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for t in range(20):
        writer.write(np.full((240, 320, 3), t * 10, dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="temporal"):
        DetectionPipeline(tmp_path / "t.pt", device="cpu", crop_mode="multi").predict(vid)
    assert DetectionPipeline(tmp_path / "t.pt", device="cpu", crop_mode="center").predict(vid).media_type == "video"


def test_pipeline_centre_mode_scores_single_crop(tmp_path):
    ckpt, img = _ckpt(tmp_path)
    r = DetectionPipeline(ckpt, device="cpu", crop_mode="center").predict(img)
    assert r.details["n_crops"] == 1 and r.details["crop_mode"] == "center"
