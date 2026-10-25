"""Offline tests for video V3: Celeb-DF ingestion helpers, YuNet face crops, multi-source WDF+Celeb-DF training data."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dfd.config import DataConfig
from dfd.data.datasets import build_base_datasets, eval_splits
from dfd.data.video_frames import extract_clip_frames, face_square_crop, largest_face, video_timestamps, yunet
from test_external_data import _script
from test_video import _frame_root, _write_mp4
from test_video_v2 import _wdf_fixture_clips

YUNET = Path("/tmp/yunet_model/yunet.onnx")  # optional local copy of the opencv_zoo model (tests skip without it)


def test_face_square_crop_geometry():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert face_square_crop(frame, [90, 40, 20, 20]).shape == (26, 26, 3)  # 20 * 1.3, centred
    edge = face_square_crop(frame, [0, 0, 30, 40])  # near the corner: shifted inside, not padded
    assert edge.shape == (52, 52, 3)
    assert face_square_crop(frame, [10, 10, 180, 90]).shape == (100, 100, 3)  # capped at the short side


class _FakeResp(io.BytesIO):
    status = 206

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_download_resumable_resumes_and_sanitizes(tmp_path):
    mod = _script("modal_data")
    data = bytes(range(256)) * 40
    calls = []

    def opener(req):
        start = int(req.get_header("Range").split("=")[1].rstrip("-"))
        calls.append(start)
        if len(calls) == 1:  # first attempt: deliver only part, as if the connection dropped
            return _FakeResp(data[start:start + 1000])
        return _FakeResp(data[start:])

    out = tmp_path / "f.zip"
    assert mod.download_resumable("https://x/download?id=SECRET", out, len(data), opener=opener) == len(data)
    assert out.read_bytes() == data and calls == [0, 1000]

    def broken(req):
        raise OSError("boom https://x/download?id=SECRET")

    with pytest.raises(RuntimeError) as e:
        mod.download_resumable("https://x/download?id=SECRET", tmp_path / "g.zip", 10, opener=broken, retries=0)
    assert "SECRET" not in str(e.value)


def test_extract_zip_members_top_level_dir(tmp_path):
    mod = _script("modal_data")
    zp = tmp_path / "c.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for name in ("Celeb-DF-v2/Celeb-real/id0_0000.mp4", "Celeb-DF-v2/Celeb-synthesis/id0_id1_0000.mp4",
                     "Celeb-DF-v2/List_of_testing_videos.txt", "Celeb-DF-v2/README.md"):
            zf.writestr(name, b"x" * 5)
    prefixes = ("Celeb-real/", "YouTube-real/", "Celeb-synthesis/", "List_of_testing_videos.txt")
    r = mod.extract_zip_members(zp, tmp_path / "out", prefixes)
    assert r["extracted"] == 3 and r["counts"]["Celeb-real/"] == 1 and r["counts"]["YouTube-real/"] == 0
    assert (tmp_path / "out" / "Celeb-synthesis" / "id0_id1_0000.mp4").exists()
    assert not (tmp_path / "out" / "README.md").exists()
    assert mod.extract_zip_members(zp, tmp_path / "out", prefixes)["skipped"] == 3


def test_main_checkout_from_worktree(tmp_path):
    mod = _script("modal_data")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {tmp_path / 'main' / '.git' / 'worktrees' / 'wt'}\n")
    assert mod._main_checkout(wt) == tmp_path / "main"
    assert mod._main_checkout(tmp_path / "main") == tmp_path / "main"


def _celebdf_root(tmp_path: Path) -> Path:
    root = tmp_path / "celebdf"
    for folder, names in {"Celeb-real": ["id0_0000", "id1_0001"], "YouTube-real": ["00007"],
                          "Celeb-synthesis": ["id0_id1_0000", "id1_id0_0001"]}.items():
        (root / folder).mkdir(parents=True)
        for n in names:
            _write_mp4(root / folder / f"{n}.mp4", 96, 64, 10, 12)
    (root / "List_of_testing_videos.txt").write_text("1 YouTube-real/00007.mp4\n0 Celeb-synthesis/id1_id0_0001.mp4\n")
    return root


def test_celebdf_clips_and_extraction(tmp_path):
    mod = _script("extract_video_frames")
    root = _celebdf_root(tmp_path)
    clips = {c["clip_id"]: c for c in mod.celebdf_clips(root, "m.onnx")}
    assert set(clips) == {"celebdf_train_celebreal_id0_0000", "celebdf_train_celebreal_id1_0001",
                          "celebdf_test_ytreal_00007", "celebdf_train_synth_id0_id1_0000",
                          "celebdf_test_synth_id1_id0_0001"}
    fake = clips["celebdf_train_synth_id0_id1_0000"]
    assert (fake["label"], fake["method"], fake["video_group"]) == (1, "celebdf", "celebdf/id0")
    assert clips["celebdf_train_celebreal_id0_0000"]["video_group"] == "celebdf/id0"
    assert clips["celebdf_test_ytreal_00007"]["video_group"] == "celebdf/yt_00007"
    with pytest.raises(FileNotFoundError):
        mod.list_clips(tmp_path, tmp_path, ("celebdf",), celebdf_root=root, face_onnx=str(tmp_path / "none.onnx"))
    if not YUNET.exists():
        pytest.skip("YuNet ONNX not available locally")
    r = mod.main(["--sources", "celebdf", "--celebdf-root", str(root), "--yunet", str(YUNET),
                  "--out", str(tmp_path / "frames"), "--workers", "2"])
    assert r["clips"] == 5 and r["failed"] == 0 and r["tars"] == ["celebdf_test.tar", "celebdf_train.tar"]
    # synthetic clips have no faces: every frame falls back to the centre crop, counted per class
    assert r["face_fallback"]["real"]["rate"] == 1.0 and r["face_fallback"]["fake"]["clips"] == 2
    meta = json.loads((tmp_path / "frames" / "clips.json").read_text())
    assert all("face_onnx" not in c and c["face_fallback_frames"] == c["n_frames"] for c in meta)


@pytest.mark.skipif(not YUNET.exists(), reason="YuNet ONNX not available locally")
def test_extract_clip_frames_face_mode_counts_fallback(tmp_path):
    p = _write_mp4(tmp_path / "blank.mp4", 160, 90, 10, 20)
    stats = {}
    frames = extract_clip_frames(p, video_timestamps(2.0), face_onnx=YUNET, stats=stats)
    assert len(frames) == 16 and stats["face_fallback"] == 16
    assert cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR).shape == (256, 256, 3)
    assert largest_face(yunet(YUNET), np.zeros((90, 160, 3), np.uint8)) is None


def test_video_train_sources_and_celebdf_test(tmp_path):
    celeb = [{"clip_id": f"celebdf_{s}_x_{i}", "source": "celebdf", "split": s, "label": i % 2,
              "method": "celebdf" if i % 2 else "real", "video_group": f"celebdf/id{i}"} for s in ("train", "test")
             for i in range(40)]
    root = _frame_root(tmp_path, _wdf_fixture_clips(40) + celeb, n_frames=1)
    only_wdf = DataConfig(source="video_wdf", video_frames_root=str(root))
    train, val = build_base_datasets(only_wdf)
    assert {c["source"] for c in train.clips + val.clips} == {"wdf"}
    both = DataConfig(source="video_wdf", video_frames_root=str(root), video_train_sources=("wdf", "celebdf"))
    train, val = build_base_datasets(both)
    assert {c["source"] for c in train.clips} == {"wdf", "celebdf"} and {c["split"] for c in train.clips} == {"train"}
    assert not {c["video_group"] for c in train.clips} & {c["video_group"] for c in val.clips}
    splits = eval_splits(both, val)
    assert len(splits["celebdf_test"]) == 40 and "wdf_test" not in splits


def test_cap_fakes_per_group_ratio_and_determinism():
    from dfd.data.datasets import cap_fakes_per_group

    clips = [{"clip_id": f"r{i}", "label": 0, "video_group": f"celebdf/id{i}"} for i in range(10)]
    for g, n in {"id0": 30, "id1": 12, "id2": 3, "id3": 1}.items():
        clips += [{"clip_id": f"{g}_f{j}", "label": 1, "video_group": f"celebdf/{g}"} for j in range(n)]
    kept = cap_fakes_per_group(clips, 2.0, seed=0)
    fakes = [c for c in kept if c["label"] == 1]
    assert len([c for c in kept if c["label"] == 0]) == 10 and len(fakes) <= 20
    per = {g: sum(c["video_group"] == f"celebdf/{g}" for c in fakes) for g in ("id0", "id1", "id2", "id3")}
    assert per["id0"] == per["id1"] and per["id2"] == 3 and per["id3"] == 1  # uniform k; small groups intact
    assert kept == cap_fakes_per_group(clips, 2.0, seed=0) and kept != cap_fakes_per_group(clips, 2.0, seed=1)
    assert cap_fakes_per_group(clips, 10.0) == clips


def test_video_wdf_caps_celebdf_and_logs_counts(tmp_path, capsys):
    celeb = [{"clip_id": f"celebdf_train_r_{i}", "source": "celebdf", "split": "train", "label": 0, "method": "real",
              "video_group": f"celebdf/id{i}"} for i in range(20)]
    celeb += [{"clip_id": f"celebdf_train_f_{i}_{j}", "source": "celebdf", "split": "train", "label": 1,
               "method": "celebdf", "video_group": f"celebdf/id{i}"} for i in range(20) for j in range(6)]
    root = _frame_root(tmp_path, _wdf_fixture_clips(20) + celeb, n_frames=1)
    cfg = DataConfig(source="video_wdf", video_frames_root=str(root), video_train_sources=("wdf", "celebdf"))
    train, val = build_base_datasets(cfg)
    c = [x for x in train.clips + val.clips if x["source"] == "celebdf"]
    assert sum(x["label"] for x in c) <= 2 * sum(1 - x["label"] for x in c)
    assert len([x for x in train.clips + val.clips if x["source"] == "wdf"]) == 40  # WDF untouched
    assert "per-source class counts" in capsys.readouterr().out
