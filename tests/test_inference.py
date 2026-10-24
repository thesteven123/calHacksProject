import cv2
import numpy as np
import torch
from PIL import Image

from dfd.config import ModelConfig
from dfd.inference.pipeline import DetectionPipeline, dhash
from dfd.models.detector import DeepfakeDetector

TINY = dict(backbone="random-tiny", feature_layers=(2, 4), lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8)


def _pipeline(tmp_path, **kw):
    torch.manual_seed(0)
    DeepfakeDetector(ModelConfig(**TINY)).save(tmp_path / "m.pt")
    return DetectionPipeline(tmp_path / "m.pt", device="cpu", **kw)


def test_image_and_video_predictions(tmp_path):
    rng = np.random.default_rng(0)
    img_path = tmp_path / "a.png"
    Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)).save(img_path)
    vid_path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for t in range(300):
        writer.write(np.full((240, 320, 3), t % 256, dtype=np.uint8))
    writer.release()

    pipe = _pipeline(tmp_path)
    r_img = pipe.predict(img_path)
    r_vid = pipe.predict(vid_path)
    assert r_img.media_type == "image" and r_img.verdict in {"real", "fake", "abstain"} and 0 <= r_img.p_fake <= 1
    assert r_vid.media_type == "video" and r_vid.details["n_keyframes"] > 1
    # sampling must span the whole clip, not just the opening seconds (CODE_REVIEW finding 2)
    assert r_vid.details["coverage_frac"] > 0.9


def test_stage0_hash_short_circuits(tmp_path):
    img = Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (64, 1))[..., None].repeat(3, -1))
    path = tmp_path / "known.png"
    img.save(path)
    pipe = _pipeline(tmp_path, known_hashes={dhash(img): "fake"})
    r = pipe.predict(path)
    assert r.stage == "stage0_hash" and r.verdict == "fake"
