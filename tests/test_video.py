"""Offline tests for video V1: time-sampled frame extraction, frame tars, clip datasets, temporal model, THB source."""

import io
import json
import pickle
import tarfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dfd.config import DataConfig, ModelConfig, PTEConfig
from dfd.data.datasets import TransformedDataset, build_base_datasets, pick_clip_frames
from dfd.data.tar_index import build_image_tar_index, write_index
from dfd.data.transforms import DetectorTransform
from dfd.data.video import VideoClipDataset, extract_clip_frames, probe_video, video_timestamps
from dfd.eval.calibration import Calibration
from dfd.models.detector import DeepfakeDetector
from dfd.training.losses import detector_loss
from dfd.training.trainer import predict, resolve_device
from test_external_data import _script

TINY = dict(backbone="random-tiny", feature_layers=(2, 4), lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8,
            image_size=56, temporal=True, temporal_dim=16, beauty_head=False)
METHODS = ("EmoPortrait", "AniPortraitAudio", "LivePortrait", "Hallo2", "AniPortraitVideo", "Hallo")


def _write_mp4(path: Path, w: int, h: int, fps: float, n: int) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        pytest.skip("no mp4v video encoder available in this OpenCV build")
    for i in range(n):
        vw.write(np.full((h, w, 3), (i * 7) % 255, np.uint8))
    vw.release()
    return path


def _jpeg(value: int, size: int = 64) -> bytes:
    bio = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value, np.uint8)).save(bio, format="JPEG", quality=95)
    return bio.getvalue()


def _frame_root(root: Path, clips: list[dict], n_frames: int = 16, distinct: bool = True) -> Path:
    """Frame tars + index.json + clips.json like scripts/extract_video_frames.py writes."""
    root.mkdir(parents=True, exist_ok=True)
    tars: dict[str, tarfile.TarFile] = {}
    for c in clips:
        name = f"{c['source']}_{c['split']}.tar"
        if name not in tars:
            tars[name] = tarfile.open(root / name, "w")
        tf = tars[name]
        for k in range(c.get("n_frames", n_frames)):
            data = _jpeg(10 * k if distinct else 128)
            info = tarfile.TarInfo(f"{c['clip_id']}/{k:02d}.jpg")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    for tf in tars.values():
        tf.close()
    write_index(root / "index.json", build_image_tar_index(root, [root / n for n in sorted(tars)]))
    (root / "clips.json").write_text(json.dumps(clips))
    return root


def _thb_clips() -> list[dict]:
    clips = []
    for m in METHODS:
        for split in ("train", "val", "test"):
            for i in range(2):
                clips.append({"clip_id": f"img{i}--clip{i}--{m}-{split}", "source": "thb", "method": m,
                              "split": split, "label": 1})
    for split in ("train", "val", "test"):
        for i in range(6):
            clips.append({"clip_id": f"real_{split}_{i}", "source": "celebvhq", "method": "real", "split": split,
                          "label": 0})
    return clips


def test_video_timestamps():
    ts = video_timestamps(10.0)
    assert ts == [k * 0.25 for k in range(16)]
    short = video_timestamps(2.0)
    assert len(short) == 16 and short[0] == 0.0 and max(short) < 2.0
    assert np.allclose(np.diff(short), 2.0 / 16)


def test_extract_clip_frames_two_formats(tmp_path):
    for name, (w, h, fps, n) in {"a": (64, 48, 10, 25), "b": (96, 96, 25, 150)}.items():
        p = _write_mp4(tmp_path / f"{name}.mp4", w, h, fps, n)
        info = probe_video(p)
        assert (info["width"], info["height"]) == (w, h) and info["fps"] == pytest.approx(fps)
        frames = extract_clip_frames(p, video_timestamps(info["duration_s"]))
        assert len(frames) == 16
        assert all(cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR).shape == (256, 256, 3) for f in frames)
    with pytest.raises(IOError):
        extract_clip_frames(tmp_path / "missing.mp4", [0.0])


def test_extract_script_round_trip(tmp_path):
    mod = _script("extract_video_frames")
    thb, cv = tmp_path / "thb", tmp_path / "celebvhq"
    (thb / "fake" / "LivePortrait" / "train").mkdir(parents=True)
    (thb / "fake" / "Hallo2" / "test").mkdir(parents=True)
    (thb / "real").mkdir(parents=True)
    (cv / "clips").mkdir(parents=True)
    _write_mp4(thb / "fake" / "LivePortrait" / "train" / "f1--c1--LivePortrait.mp4", 80, 80, 10, 50)
    _write_mp4(thb / "fake" / "Hallo2" / "test" / "f2--c2--Hallo2.mp4", 96, 96, 25, 40)
    _write_mp4(cv / "clips" / "c1.mp4", 96, 72, 24, 120)
    (thb / "real" / "real_dataset_split.json").write_text(json.dumps(
        {"Train": {"CelebV-HQ": ["c1.mp4"], "FaceForensics++": ["x.mp4"]}, "Val": {"CelebV-HQ": ["gone.mp4"]}}))
    r = mod.main(["--thb-root", str(thb), "--celebvhq-root", str(cv), "--out", str(tmp_path / "out"),
                  "--workers", "2"])
    assert r["clips"] == 3 and r["failed"] == 1 and r["tars"] == ["celebvhq_train.tar", "thb_test.tar", "thb_train.tar"]
    clips = {c["clip_id"]: c for c in json.loads((tmp_path / "out" / "clips.json").read_text())}
    assert clips["c1"]["label"] == 0 and clips["c1"]["method"] == "real" and clips["c1"]["split"] == "train"
    assert clips["f1--c1--LivePortrait"]["method"] == "LivePortrait" and clips["f1--c1--LivePortrait"]["n_frames"] == 16
    assert clips["f2--c2--Hallo2"]["src_size"] == [96, 96]
    ds = VideoClipDataset(tmp_path / "out", list(clips.values()))
    frames, label, gen = ds[0]
    assert len(frames) == 16 and frames[0].size == (256, 256)


def test_clip_dataset_and_transformed_dataset(tmp_path):
    clips = [{"clip_id": "a", "source": "thb", "method": "Hallo", "split": "val", "label": 1},
             {"clip_id": "b", "source": "celebvhq", "method": "real", "split": "val", "label": 0, "n_frames": 3}]
    root = _frame_root(tmp_path, clips)
    base = VideoClipDataset(root, clips)
    assert base.meta(0) == (1, "Hallo") and base.meta(1) == (0, "real") and len(base[0][0]) == 16
    eval_ds = TransformedDataset(base, DetectorTransform(PTEConfig(enabled=False), 224, train=False), 1, 8)
    item = eval_ds[0]
    assert item["pixels"].shape == (8, 3, 224, 224) and item["mask"].all() and item["label"] == 1
    assert item["generator"] == "Hallo" and item["beauty"] == 0
    padded = eval_ds[1]
    assert padded["mask"].tolist() == [True] * 3 + [False] * 5 and torch.equal(padded["pixels"][7], padded["pixels"][2])
    clone = pickle.loads(pickle.dumps(eval_ds))
    assert torch.equal(clone[0]["pixels"], item["pixels"])


def test_clip_transform_consistent_across_frames(tmp_path):
    clips = [{"clip_id": "same", "source": "thb", "method": "Hallo", "split": "train", "label": 1}]
    root = _frame_root(tmp_path, clips, distinct=False)
    ds = TransformedDataset(VideoClipDataset(root, clips), DetectorTransform(PTEConfig(), 56, train=True), 0, 6)
    for epoch in range(3):
        ds.set_epoch(epoch)
        px = ds[0]["pixels"]
        assert all(torch.equal(px[0], px[t]) for t in range(1, 6))  # same PTE/crop/flip on identical input frames
        assert isinstance(ds[0]["severity"], str)


def test_clip_frame_offsets(tmp_path):
    clips = [{"clip_id": "c", "source": "thb", "method": "Hallo", "split": "train", "label": 1}]
    root = _frame_root(tmp_path, clips)
    base = VideoClipDataset(root, clips)

    def picks(ds):
        return [round(float(p.mean()) * 255 / 10) for p in ds[0]["pixels"]]

    tr = TransformedDataset(base, DetectorTransform(PTEConfig(enabled=False), 56, train=True), 0, 4)
    seen = set()
    for epoch in range(6):
        tr.set_epoch(epoch)
        seen.add(tuple(picks(tr)))
    assert len(seen) > 1 and all(np.all(np.diff(p) == 4) for p in seen)
    ev = TransformedDataset(base, DetectorTransform(PTEConfig(enabled=False), 56, train=False), 0, 4)
    first = picks(ev)
    ev.set_epoch(3)
    assert picks(ev) == first == [0, 5, 10, 15]
    assert pick_clip_frames(2, 4, True, np.random.default_rng(0)) == ([0, 1, 1, 1], [True, True, False, False])


def test_temporal_model_loss_and_predict(tmp_path):
    model = DeepfakeDetector(ModelConfig(**TINY))
    model.train()
    mask = torch.tensor([[True] * 4, [True, True, False, False]])
    out = model(torch.rand(2, 4, 3, 56, 56), mask)
    assert out["logit"].shape == (2,) and out["mu"].shape == (8, TINY["latent_dim"]) and "beauty_logit" not in out
    loss, logs = detector_loss(out, torch.tensor([1, 0]), beta=1e-3, beauty_labels=torch.zeros(2), beauty_weight=0.0)
    loss.backward()
    assert "beauty" not in logs
    clips = [{"clip_id": f"c{i}", "source": "thb", "method": "Hallo", "split": "val", "label": i % 2} for i in range(3)]
    ds = TransformedDataset(VideoClipDataset(_frame_root(tmp_path, clips, n_frames=5), clips),
                            DetectorTransform(PTEConfig(enabled=False), 56, train=False), 0, 4)
    preds = predict(model, DataLoader(ds, batch_size=2), resolve_device("cpu"))
    assert len(preds["logits"]) == 3 and preds["labels"] == [0, 1, 0]


def test_build_base_datasets_video_thb(tmp_path):
    root = _frame_root(tmp_path, _thb_clips(), n_frames=2)
    cfg = DataConfig(source="video_thb", video_frames_root=str(root), selection_generators=("AniPortraitVideo",),
                     holdout_generators=("EmoPortrait",))
    train, val = build_base_datasets(cfg, seed=0)
    train_gens = {train.meta(i)[1] for i in range(len(train))}
    val_gens = {val.meta(i)[1] for i in range(len(val))}
    assert train_gens == {"AniPortraitAudio", "LivePortrait", "Hallo2", "Hallo", "real"}
    assert val_gens == set(METHODS) | {"real"}
    assert {c["split"] for c in train.clips} == {"train"} and {c["split"] for c in val.clips} == {"val", "test"}
    only = build_base_datasets(DataConfig(source="video_thb", video_frames_root=str(root),
                                          video_methods=("Hallo",)), seed=0)[0]
    assert {only.meta(i)[1] for i in range(len(only))} == {"Hallo", "real"}


def test_plan_files_thb_and_celebvhq():
    mod = _script("modal_data")
    tree = [(f"fake/{m}/{s}/x{i}--y--{m}.mp4", 10) for m in METHODS for s in ("train", "val", "test") for i in range(3)]
    tree += [("fake/additional_dataset/MAGI-1/test/a.mp4", 1), ("audio/train/a.wav", 1),
             ("real/real_dataset_split.json", 1), ("real/other.json", 1), ("README.md", 1)]
    picked = [p for p, _ in mod.plan_files("thb", tree)]
    assert len(picked) == 6 * 3 * 3 + 2 and not any("additional" in p or p.startswith("audio") for p in picked)
    smoke = [p for p, _ in mod.plan_files("thb", tree, smoke=True)]
    assert len(smoke) == 6 * 2 + 2 and all("/test/" in p for p in smoke if p.endswith(".mp4"))
    cv_tree = [("videos.tar", 100), ("celebvhq_info.json", 1), ("README.md", 1)]
    assert [p for p, _ in mod.plan_files("celebvhq", cv_tree)] == ["videos.tar", "celebvhq_info.json"]
    assert mod.plan_files("celebvhq", cv_tree, smoke=True) == []
    assert mod.celebvhq_wanted({"Train": {"CelebV-HQ": ["a.mp4"], "FaceForensics++": ["z.mp4"]},
                                "Test": {"CelebV-HQ": ["b.mp4"]}}) == {"a", "b"}


def test_extract_tar_members(tmp_path):
    mod = _script("modal_data")
    tar_path = tmp_path / "videos.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in {"videos/a.mp4": b"aaa", "videos/b.mp4": b"bb", "videos/c.mp4": b"c",
                              "videos/a.txt": b"x"}.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    out = tmp_path / "clips"
    r = mod.extract_tar_members(tar_path, {"a", "c", "zz"}, out)
    assert r == {"extracted": 2, "skipped": 0, "missing": ["zz"]}
    assert sorted(p.name for p in out.iterdir()) == ["a.mp4", "c.mp4"] and (out / "a.mp4").read_bytes() == b"aaa"
    assert mod.extract_tar_members(tar_path, {"a", "c"}, out)["skipped"] == 2
    (out / "a.mp4").write_bytes(b"zzz")  # stale content, same size as the member
    r = mod.extract_tar_members(tar_path, {"a", "c"}, out)
    assert r["extracted"] == 1 and r["skipped"] == 1 and (out / "a.mp4").read_bytes() == b"aaa"


def test_eval_external_temporal_wilddeepfake():
    mod = _script("eval_external")

    class _Wild:
        clips = ["r/1", "r/1", "f/2", "f/2", "f/2"]

        def __len__(self):
            return len(self.clips)

        def clip_id(self, i):
            return self.clips[i]

        def meta(self, i):
            return (0 if self.clips[i].startswith("r") else 1), "x"

        def __getitem__(self, i):
            return Image.new("RGB", (64, 64), (i * 40, 0, 0)), *self.meta(i)

    model = DeepfakeDetector(ModelConfig(**TINY)).eval()
    pipe = SimpleNamespace(model=model, device="cpu", calibration_status="matched",
                           cal=Calibration(temperature=1.0, tau_real=0.3, tau_fake=0.7),
                           transform=DetectorTransform(PTEConfig(enabled=False), 56, train=False))
    rep = mod._eval_wilddeepfake_temporal(pipe, _Wild(), frames_per_clip=4, batch_size=8, num_workers=0)
    assert rep["frame"] is None and rep["clip"]["n_clips"] == 2 and rep["clip"]["verdicts"]["n"] == 2
