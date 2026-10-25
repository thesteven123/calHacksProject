"""Offline tests for FFHQ real portraits (tar index + dataset + paired_faces), FairFace grouping, and B3 config."""

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dfd.config import Config, DataConfig
from dfd.data.datasets import build_base_datasets
from dfd.data.external import FFHQDataset, sfhq_val_row
from dfd.data.paired import FACE_ALIGN_SIZE
from dfd.data.tar_index import build_image_tar_index, write_index
from dfd.eval.calibration import Calibration
from test_external_data import _script
from test_paired_data import _create_synthetic_dff

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _img(color: int, fmt: str, size=(96, 64)) -> bytes:
    bio = io.BytesIO()
    kw = {"lossless": True} if fmt == "WEBP" else {}
    Image.fromarray(np.full((size[1], size[0], 3), color, dtype=np.uint8)).save(bio, format=fmt, **kw)
    return bio.getvalue()


def _add(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _ffhq_root(path: Path, n_tars: int = 2, per_tar: int = 20) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    tars = []
    for t in range(n_tars):
        tars.append(path / f"{t * 1000:05d}.tar")
        with tarfile.open(tars[-1], "w") as tf:
            for i in range(per_tar):
                _add(tf, f"{t * 1000 + i:05d}.webp", _img(10 + i, "WEBP"))
    write_index(path / "index.json", build_image_tar_index(path, tars))
    return path


def test_tar_index_offsets_and_skip_non_images(tmp_path):
    payloads = {"a.webp": _img(1, "WEBP"), "sub/b.png": _img(2, "PNG"), "c.jpg": _img(3, "JPEG")}
    tar_path = tmp_path / "x.tar"
    with tarfile.open(tar_path, "w") as tf:
        for n, p in payloads.items():
            _add(tf, n, p)
        _add(tf, "meta.json", b"{}")
        d = tarfile.TarInfo("dir.png")
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
    entries = build_image_tar_index(tmp_path, [tar_path])
    assert [e["member"] for e in entries] == list(payloads) and all(e["tar"] == "x.tar" for e in entries)
    with open(tar_path, "rb") as f:
        for e in entries:
            f.seek(e["offset"])
            assert f.read(e["size"]) == payloads[e["member"]]


def test_ffhq_dataset_split_cap_contract(tmp_path):
    root = _ffhq_root(tmp_path / "ffhq")
    full = FFHQDataset(root)
    members = {e["member"] for e in full.items}
    train, val = FFHQDataset(root, split="train"), FFHQDataset(root, split="val", seed=5)
    tm, vm = {e["member"] for e in train.items}, {e["member"] for e in val.items}
    assert len(members) == 40 and tm and vm and not (tm & vm) and tm | vm == members
    assert vm == {m for m in members if sfhq_val_row(m, 0.2)}
    capped = FFHQDataset(root, split="train", max_items=5, seed=1)
    assert len(capped) == 5 and {e["member"] for e in capped.items} <= tm
    assert [e["member"] for e in capped.items] == [e["member"] for e in FFHQDataset(root, split="train", max_items=5, seed=1).items]
    img, label, gen = train[0]
    assert img.size == (384, 384) and label == 0 and gen == "real" and train.meta(0) == (0, "real")
    assert train.group(0) == train.items[0]["member"]
    clone = pickle.loads(pickle.dumps(train))
    assert np.array_equal(np.asarray(clone[0][0]), np.asarray(img))
    with pytest.raises(ValueError):
        FFHQDataset(root, split="test")


def _cfg(tmp_path, **kw) -> DataConfig:
    return DataConfig(source="paired_faces", hf_dataset=None,
                      dff_root=str(_create_synthetic_dff(tmp_path / "dff", num_identities=12)),
                      dff_generators=("inpainting", "insight"), ffhq_root=str(_ffhq_root(tmp_path / "ffhq")),
                      ffhq_val_fraction=0.3, **kw)


def test_paired_faces_with_ffhq(tmp_path):
    train, val = build_base_datasets(_cfg(tmp_path), seed=0)
    for ds in (train, val):
        ffhq = [i for i in range(len(ds)) if ds.group(i).endswith(".webp")]
        assert ffhq and all(ds.meta(i) == (0, "real") for i in ffhq)
        assert ds[ffhq[0]][0].size == (FACE_ALIGN_SIZE, FACE_ALIGN_SIZE)


def test_ffhq_requires_dff_format_align(tmp_path):
    with pytest.raises(ValueError, match="FFHQ alignment must match DFF"):
        build_base_datasets(_cfg(tmp_path, dff_format_align=False), seed=0)


def test_plan_files_ffhq():
    mod = _script("modal_data")
    tree = [(f"{i * 1000:05d}.tar", 10) for i in range(70)] + [("README.md", 1), ("sub/00000.tar", 1)]
    picked = [p for p, _ in mod.plan_files("ffhq", tree)]
    assert len(picked) == 15 and picked[0] == "00000.tar" and picked[-1] == "69000.tar"
    assert picked == sorted(picked) and len(set(picked)) == 15
    assert [p for p, _ in mod.plan_files("ffhq", tree, smoke=True)] == ["00000.tar"]
    assert len(mod.plan_files("ffhq", tree, ffhq_shards=100)) == 70
    wild = [("deepfake_in_the_wild/real_test/1.tar.gz", 5), ("deepfake_in_the_wild/fake_test/2.tar.gz", 5)]
    assert mod.plan_files("wilddeepfake", wild, ffhq_shards=1) == mod.plan_files("wilddeepfake", wild)
    assert "ffhq" in mod.DATASETS and "fairface" in mod.EVAL_DATASETS


def test_group_rates_and_status():
    mod = _script("eval_external")
    v = np.array(["fake", "real", "abstain", "real", "fake", "real"])
    p = np.array([0.9, 0.1, 0.5, 0.2, 0.8, 0.0])
    g = ["A", "A", "A", "B", "B", "C"]
    r = mod._group_rates(v, p, g)
    assert r["A"] == {"n": 3, "fpr_fake_verdict": pytest.approx(1 / 3), "abstain_rate": pytest.approx(1 / 3),
                      "mean_p_fake": pytest.approx(0.5)}
    assert r["B"]["fpr_fake_verdict"] == 0.5 and r["B"]["abstain_rate"] == 0.0
    assert r["C"] == {"n": 1, "fpr_fake_verdict": 0.0, "abstain_rate": 0.0, "mean_p_fake": 0.0}
    cal = Calibration(temperature=1.0, tau_real=0.3, tau_fake=0.7)
    logits, labels = np.array([5.0, -5.0, 0.0]), np.zeros(3, dtype=int)
    assert list(mod._verdicts(cal, logits, "mismatched")) == ["abstain"] * 3
    off = mod._verdict_rates(cal, logits, labels, "mismatched")
    assert off["abstain_rate"] == 1.0 and off["fpr_fake_verdict"] == 0.0 and not off["verdicts_valid"]
    on = mod._verdict_rates(cal, logits, labels, "matched")
    assert on["fpr_fake_verdict"] == pytest.approx(1 / 3)


def test_face_t2i_ffhq_config_loads():
    c = Config.from_yaml(CONFIGS / "face_t2i_ffhq_nested.yaml")
    assert c.data.ffhq_root == "/data/ffhq" and c.data.ffhq_max_train == 12000 and c.data.ffhq_max_val == 3000
    assert c.data.hf_dataset is None and c.data.sfhq_zip and c.train.output_dir == "runs/face_t2i_ffhq_nested"
    assert c.data.holdout_generators == ("FLUX1-dev", "FLUX1-pro", "FLUX1-schnell")


def test_fairface_dataset_offline(monkeypatch):
    import datasets as hf

    feats = hf.Features({"image": hf.Image(), "race": hf.ClassLabel(names=["East Asian", "Black"]),
                         "gender": hf.ClassLabel(names=["Male", "Female"]),
                         "age": hf.ClassLabel(names=["0-2", "20-29"])})
    rows = {"image": [Image.fromarray(np.full((40, 40), i * 20, dtype=np.uint8)) for i in range(6)],
            "race": [0, 1, 0, 1, 0, 1], "gender": [0, 0, 1, 1, 0, 1], "age": [1] * 6}
    calls = []
    monkeypatch.setattr(hf, "load_dataset", lambda *a, **k: calls.append((a, k)) or hf.Dataset.from_dict(rows, features=feats))
    ds = _script("eval_external").FairFaceDataset(max_items=4)
    assert calls[0] == (("HuggingFaceM4/FairFace", "1.25"), {"split": "validation"})
    assert len(ds) == 4 and set(ds.race) <= {"East Asian", "Black"} and set(ds.age) == {"20-29"}
    img, label, gen = ds[0]
    assert img.mode == "RGB" and label == 0 and gen == "real"
