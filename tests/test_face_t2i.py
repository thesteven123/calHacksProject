"""Offline tests for face text-to-image training: SFHQ-T2I splits + paired_faces integration + configs."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dfd.config import Config, DataConfig
from dfd.data import external
from dfd.data.align import align_format, quality_for
from dfd.data.datasets import build_base_datasets, eval_splits
from dfd.data.external import SFHQT2IDataset, sfhq_val_row
from dfd.data.paired import FACE_ALIGN_SIZE, FACE_JPEG_QUALITY, DeepFakeFaceDataset
from test_paired_data import _create_synthetic_dff

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SFHQ_PREFIXES = ("DALLE3", "FLUX1_dev", "SDXL")


def _sfhq_zip(path: Path, n: int = 30, prefixes=SFHQ_PREFIXES) -> Path:
    """Tiny SFHQ-T2I zip with non-square 300x200 PNG-in-.jpg payloads (alignment must square them)."""
    path.mkdir(parents=True, exist_ok=True)
    zpath = path / "SyntheticFacesHighQuality-T2I.zip"
    bio = io.BytesIO()
    Image.fromarray(np.full((200, 300, 3), 120, dtype=np.uint8)).save(bio, format="JPEG", quality=98)
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in prefixes:
            for i in range(n):
                zf.writestr(f"images/{p}_image_{i:07d}.jpg", bio.getvalue())
        zf.writestr("images/README.txt", b"readme")
    return zpath


def _names(ds: SFHQT2IDataset) -> set[str]:
    return {n for n, _ in ds.items}


def test_sfhq_split_disjoint_complete_and_seed_independent(tmp_path):
    z = _sfhq_zip(tmp_path)
    full = _names(SFHQT2IDataset(z))
    train, val = _names(SFHQT2IDataset(z, split="train")), _names(SFHQT2IDataset(z, split="val"))
    assert train and val and not (train & val) and train | val == full
    assert val == {n for n in full if sfhq_val_row(n, 0.2)}
    assert _names(SFHQT2IDataset(z, split="val", seed=7)) == val
    assert _names(SFHQT2IDataset(z, split="val", val_fraction=0.5)) > val


def test_sfhq_exclude_and_cap_after_split(tmp_path):
    z = _sfhq_zip(tmp_path)
    ds = SFHQT2IDataset(z, split="train", exclude_generators=("FLUX1-dev", "DALLE3"))
    assert set(ds.stats) == {"SDXL"}
    capped = SFHQT2IDataset(z, split="val", max_per_generator=2, seed=3)
    val = _names(SFHQT2IDataset(z, split="val"))
    assert all(v <= 2 for v in capped.stats.values()) and _names(capped) <= val
    assert SFHQT2IDataset(z, generators=("SDXL",), exclude_generators=("SDXL",)).stats == {}
    with pytest.raises(ValueError):
        SFHQT2IDataset(z, split="test")


def test_sfhq_quality_range_and_defaults(tmp_path, monkeypatch):
    z = _sfhq_zip(tmp_path, n=3)
    seen = []
    def spy(key, lo=75, hi=95):
        seen.append((lo, hi))
        return quality_for(key, lo, hi)

    monkeypatch.setattr(external, "quality_for", spy)
    ds = SFHQT2IDataset(z, quality_range=(90, 90), align_size=64)
    img, label, _ = ds[0]
    assert seen[-1] == (90, 90) and img.size == (64, 64) and label == 1
    default = SFHQT2IDataset(z)
    default[0]
    assert seen[-1] == (75, 95) and default.align_size == 256 and len(default) == 9


def test_dff_alignment_matches_align_format(tmp_path):
    root = _create_synthetic_dff(tmp_path, num_identities=4)
    ds = DeepFakeFaceDataset(root=root, split="all")
    raw = DeepFakeFaceDataset(root=root, split="all", format_align=False)
    for i in range(len(ds)):
        expected = align_format(raw[i][0], FACE_ALIGN_SIZE, quality_for(ds.group(i), *FACE_JPEG_QUALITY))
        assert np.array_equal(np.asarray(ds[i][0]), np.asarray(expected))


def _face_cfg(tmp_path, **kw) -> DataConfig:
    return DataConfig(source="paired_faces", hf_dataset=None,
                      dff_root=str(_create_synthetic_dff(tmp_path / "dff", num_identities=30)),
                      dff_val_fraction=0.3, sfhq_zip=str(_sfhq_zip(tmp_path / "sfhq")),
                      sfhq_val_fraction=0.4, holdout_generators=("FLUX1-dev",), selection_generators=("DALLE3",),
                      **kw)


def _fake_gens(ds, idx=None) -> set[str]:
    return {ds.meta(i)[1] for i in (idx if idx is not None else range(len(ds))) if ds.meta(i)[0] == 1}


def test_paired_faces_with_sfhq(tmp_path):
    cfg = _face_cfg(tmp_path)
    train, val = build_base_datasets(cfg, seed=0)
    assert _fake_gens(train) == {"inpainting", "insight", "text2img", "SDXL"}
    assert {"FLUX1-dev", "DALLE3", "SDXL"} <= _fake_gens(val)
    for ds in (train, val):
        sfhq = [i for i in range(len(ds)) if ds.meta(i)[1] in ("SDXL", "DALLE3", "FLUX1-dev")][:3]
        dff = [i for i in range(len(ds)) if ds.meta(i)[1] in ("real", "insight")][:3]
        assert sfhq and dff and all(ds[i][0].size == (FACE_ALIGN_SIZE, FACE_ALIGN_SIZE) for i in sfhq + dff)
    splits = eval_splits(cfg, val, seed=0)
    assert _fake_gens(splits["test"]) == {"FLUX1-dev"}
    assert _fake_gens(splits["select"]) == {"DALLE3"}
    assert "FLUX1-dev" not in _fake_gens(splits["fit"]) and "DALLE3" not in _fake_gens(splits["fit"])
    groups = {k: {val.group(i) for i in splits[k].indices if val.group(i).startswith("0_")}
              for k in ("fit", "select", "test")}
    assert not (groups["fit"] & groups["select"] or groups["fit"] & groups["test"] or groups["select"] & groups["test"])


def test_sfhq_requires_dff_format_align(tmp_path):
    with pytest.raises(ValueError, match="SFHQ alignment must match DFF"):
        build_base_datasets(_face_cfg(tmp_path, dff_format_align=False), seed=0)


def test_face_t2i_configs_load():
    b1 = Config.from_yaml(CONFIGS / "face_t2i_nested.yaml")
    b2 = Config.from_yaml(CONFIGS / "face_t2i_tiny_nested.yaml")
    assert b1.data.hf_dataset is None and b2.data.hf_dataset == "TheKernel01/Tiny-GenImage"
    for c in (b1, b2):
        assert c.data.source == "paired_faces" and c.data.sfhq_zip.endswith("SyntheticFacesHighQuality-T2I.zip")
        assert c.data.selection_generators == ("DALLE3",)
        assert c.data.holdout_generators == ("FLUX1-dev", "FLUX1-pro", "FLUX1-schnell")
        assert c.train.epochs == 2 and c.train.batch_size == 32 and c.train.calibration_method == "conformal"
