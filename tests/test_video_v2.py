"""Offline tests for video V2: WildDeepfake train clips as a frame source, video-level val hold-out, extra splits."""

import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from dfd.config import DataConfig
from dfd.data.datasets import build_base_datasets, eval_splits
from dfd.data.video import VideoClipDataset
from dfd.data.video_frames import align_frame, extract_wdf_frames, wdf_clips
from dfd.data.wilddeepfake_index import build_wilddeepfake_index, write_wilddeepfake_index
from test_external_data import _script
from test_video import _frame_root, _thb_clips


def _png(seed: int, size=(40, 30)) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return bio.getvalue()


def _wdf_tar(path: Path, members: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tf:
        for i, name in enumerate(members):
            data = _png(i + len(name))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _wdf_root(tmp_path: Path) -> Path:
    """Train: fake video 1 (40 frames, numbers 100,103,...), real video 1 (5 frames). Test: fake video 1 (8 frames)."""
    root = tmp_path / "wilddeepfake"
    d = root / "deepfake_in_the_wild"
    _wdf_tar(d / "fake_train" / "1.tar.gz", [f"./1/fake/3/{100 + 3 * f}.png" for f in range(40)])
    _wdf_tar(d / "real_train" / "1.tar.gz", [f"./1/real/2/{f}.png" for f in range(5)])
    _wdf_tar(d / "fake_test" / "1.tar.gz", [f"./1/fake/9/{f}.png" for f in range(8)])
    _wdf_tar(d / "real_test" / "4.tar.gz", [f"./4/real/1/{f}.png" for f in range(3)])
    for split in ("train", "test"):
        write_wilddeepfake_index(root, build_wilddeepfake_index(root, split=split), split=split)
    return root


def test_wdf_clip_selection_and_ids(tmp_path):
    root = _wdf_root(tmp_path)
    clips = {c["clip_id"]: c for c in wdf_clips(json.loads((root / "index_train.json").read_text()), "train")}
    assert set(clips) == {"wdf_train_fake_1_3", "wdf_train_real_1_2"}
    fake, real = clips["wdf_train_fake_1_3"], clips["wdf_train_real_1_2"]
    assert (fake["label"], fake["method"], fake["video_group"]) == (1, "wilddeepfake", "wdf/train/fake/1")
    assert (real["label"], real["method"], real["video_group"]) == (0, "real", "wdf/train/real/1")
    frames = [e["frame"] for e in fake["frames"]]
    expected = [100 + 3 * int(i) for i in np.linspace(0, 39, 16).round().astype(int)]
    assert frames == expected and len(set(frames)) == 16
    assert [e["frame"] for e in real["frames"]] == [0, 1, 2, 3, 4]  # < 16 frames: all, unique
    out = extract_wdf_frames(root, fake["frames"][:2])
    assert all(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).shape == (256, 256, 3) for b in out)


def test_align_frame_matches_extractor(tmp_path):
    root = _wdf_root(tmp_path)
    entry = json.loads((root / "index_train.json").read_text())[0]
    extracted = cv2.cvtColor(cv2.imdecode(np.frombuffer(extract_wdf_frames(root, [entry])[0], np.uint8),
                                          cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    with open(root / entry["tar"], "rb") as f:
        f.seek(entry["offset"])
        img = Image.open(io.BytesIO(f.read(entry["size"])))
        aligned = align_frame(img)
    assert aligned.size == (256, 256) and np.array_equal(np.asarray(aligned), extracted)


def test_extract_script_wdf_merge(tmp_path):
    root = _wdf_root(tmp_path)
    out = _frame_root(tmp_path / "frames", [c for c in _thb_clips() if c["split"] == "test"], n_frames=2)
    before = json.loads((out / "clips.json").read_text())
    r = _script("extract_video_frames").main(["--sources", "wdf", "--wdf-root", str(root), "--out", str(out),
                                              "--workers", "2"])
    assert r["clips"] == 4 and r["failed"] == 0 and r["tars"] == ["wdf_test.tar", "wdf_train.tar"]
    clips = json.loads((out / "clips.json").read_text())
    assert [c for c in clips if c["source"] != "wdf"] == before
    wdf = {c["clip_id"]: c for c in clips if c["source"] == "wdf"}
    assert wdf["wdf_train_fake_1_3"]["n_frames"] == 16 and "frames" not in wdf["wdf_train_fake_1_3"]
    assert wdf["wdf_test_fake_1_9"]["split"] == "test" and wdf["wdf_test_real_4_1"]["video_group"] == "wdf/test/real/4"
    tars = {e["tar"] for e in json.loads((out / "index.json").read_text())}
    assert tars == {"thb_test.tar", "celebvhq_test.tar", "wdf_train.tar", "wdf_test.tar"}
    ds = VideoClipDataset(out, clips)
    assert len(ds) == len(clips) and ds.group(clips.index(wdf["wdf_train_fake_1_3"])) == "wdf/train/fake/1"
    assert ds[clips.index(wdf["wdf_train_fake_1_3"])][0][0].size == (256, 256)
    assert ds.group(0) == clips[0]["clip_id"]  # no video_group -> clip_id


def _wdf_fixture_clips(n_videos: int = 400) -> list[dict]:
    clips = []
    for v in range(n_videos):
        kind = "fake" if v % 2 else "real"
        for c in range(2):
            clips.append({"clip_id": f"wdf_{kind}_{v}_{c}", "source": "wdf", "split": "train",
                          "method": "wilddeepfake" if kind == "fake" else "real", "label": int(kind == "fake"),
                          "video_group": f"wdf/{kind}/{v}"})
    return clips


def test_build_base_datasets_video_wdf(tmp_path):
    clips = _wdf_fixture_clips() + [dict(c, split="test", clip_id=c["clip_id"] + "_t") for c in _wdf_fixture_clips(4)]
    root = _frame_root(tmp_path, clips, n_frames=1)
    cfg = DataConfig(source="video_wdf", video_frames_root=str(root))
    train, val = build_base_datasets(cfg, seed=0)
    tg = {train.group(i) for i in range(len(train))}
    vg = {val.group(i) for i in range(len(val))}
    assert not tg & vg and len(tg) + len(vg) == 400
    assert 0.10 < len(vg) / 400 < 0.20
    assert {c["split"] for c in train.clips + val.clips} == {"train"}
    train2, val2 = build_base_datasets(cfg, seed=7)
    assert [c["clip_id"] for c in val2.clips] == [c["clip_id"] for c in val.clips]


def test_eval_splits_wdf_extras(tmp_path):
    only_train = _frame_root(tmp_path / "a", _wdf_fixture_clips(20), n_frames=1)
    cfg = DataConfig(source="video_wdf", video_frames_root=str(only_train))
    splits = eval_splits(cfg, build_base_datasets(cfg)[1])
    assert "wdf_test" not in splits and "thb_cross" not in splits and splits["select"] is not None
    wdf_test = [dict(c, split="test", clip_id=c["clip_id"] + "_t") for c in _wdf_fixture_clips(6)]
    full = _frame_root(tmp_path / "b", _wdf_fixture_clips(20) + wdf_test + _thb_clips(), n_frames=1)
    cfg = DataConfig(source="video_wdf", video_frames_root=str(full))
    splits = eval_splits(cfg, build_base_datasets(cfg)[1])
    assert len(splits["wdf_test"]) == len(wdf_test)
    cross = splits["thb_cross"]
    assert {cross.meta(i)[1] for i in range(len(cross))} >= {"real", "Hallo"}
    assert {c["split"] for c in cross.clips} == {"test"} and {c["source"] for c in cross.clips} == {"thb", "celebvhq"}


def test_manifest_name_and_plan_files():
    mod = _script("modal_data")
    assert mod.manifest_name("wilddeepfake_train", "/data/wilddeepfake") == "MANIFEST_wilddeepfake_train.json"
    assert all(mod.manifest_name(k, v["dest"]) == "MANIFEST.json" for k, v in mod.DATASETS.items()
               if k != "wilddeepfake_train")
    tree = [(f"deepfake_in_the_wild/{k}/{i}.tar.gz", 10) for k in ("real_train", "fake_train", "real_test",
                                                                  "fake_test") for i in range(3)] + [("README.md", 1)]
    picked = [p for p, _ in mod.plan_files("wilddeepfake_train", tree)]
    assert len(picked) == 6 and all("_train/" in p for p in picked)
    smoke = [p for p, _ in mod.plan_files("wilddeepfake_train", tree, smoke=True)]
    assert smoke == ["deepfake_in_the_wild/real_train/0.tar.gz", "deepfake_in_the_wild/fake_train/0.tar.gz"]
    assert all("_test/" in p for p, _ in mod.plan_files("wilddeepfake", tree))
    assert "wilddeepfake_train" not in mod.EVAL_DATASETS


def test_wdf_train_test_same_numbers_do_not_collide(tmp_path):
    """Regression (PR #18 review): video/clip numbers restart per split, so train and test both have fake/1/3."""
    root = tmp_path / "wilddeepfake"
    d = root / "deepfake_in_the_wild"
    _wdf_tar(d / "fake_train" / "1.tar.gz", [f"./1/fake/3/{f}.png" for f in range(6)])
    _wdf_tar(d / "fake_test" / "1.tar.gz", [f"./1/fake/3/{f}.png" for f in range(4)])
    _wdf_tar(d / "real_train" / "2.tar.gz", [f"./2/real/5/{f}.png" for f in range(3)])
    _wdf_tar(d / "real_test" / "2.tar.gz", [f"./2/real/5/{f}.png" for f in range(2)])
    for split in ("train", "test"):
        write_wilddeepfake_index(root, build_wilddeepfake_index(root, split=split), split=split)
    train = wdf_clips(json.loads((root / "index_train.json").read_text()), "train")
    test = wdf_clips(json.loads((root / "index_test.json").read_text()), "test")
    assert {c["clip_id"] for c in train}.isdisjoint(c["clip_id"] for c in test)
    assert {c["video_group"] for c in train}.isdisjoint(c["video_group"] for c in test)
    out = tmp_path / "frames"
    r = _script("extract_video_frames").main(["--sources", "wdf", "--wdf-root", str(root), "--out", str(out),
                                              "--workers", "1"])
    assert r["failed"] == 0
    clips = json.loads((out / "clips.json").read_text())
    ds = VideoClipDataset(out, clips)
    tar_of = {c["clip_id"]: {e["tar"] for e in ds.frames[i]} for i, c in enumerate(ds.clips)}
    assert tar_of == {"wdf_train_fake_1_3": {"wdf_train.tar"}, "wdf_test_fake_1_3": {"wdf_test.tar"},
                      "wdf_train_real_2_5": {"wdf_train.tar"}, "wdf_test_real_2_5": {"wdf_test.tar"}}
    expected = {"wdf_train_fake_1_3": 6, "wdf_test_fake_1_3": 4, "wdf_train_real_2_5": 3, "wdf_test_real_2_5": 2}
    assert {c["clip_id"]: len(ds.frames[i]) for i, c in enumerate(ds.clips)} == expected


def test_video_dataset_keys_frames_by_tar(tmp_path):
    """Same clip id in two tars (legacy ids) must not merge frames across tars."""
    clips = [{"clip_id": "dup", "source": "wdf", "split": "train", "method": "real", "label": 0, "n_frames": 3},
             {"clip_id": "dup", "source": "wdf", "split": "test", "method": "real", "label": 0, "n_frames": 2}]
    ds = VideoClipDataset(_frame_root(tmp_path, clips), clips)
    assert [len(f) for f in ds.frames] == [3, 2]
    assert [{e["tar"] for e in f} for f in ds.frames] == [{"wdf_train.tar"}, {"wdf_test.tar"}]
