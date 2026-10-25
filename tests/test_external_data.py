"""Offline tests for external-dataset loaders (GenImage-arrow, WildDeepfake, SFHQ-T2I) and modal_data.plan_files."""

import io
import pickle
import tarfile
import zipfile
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pytest
from PIL import Image

from dfd.config import DataConfig
from dfd.data.datasets import build_base_datasets
from dfd.data.external import (
    GENIMAGE_DIRS,
    GenImageArrowDataset,
    SFHQT2IDataset,
    WildDeepfakeDataset,
    build_wilddeepfake_index,
    select_shards,
    write_wilddeepfake_index,
)
from dfd.eval.metrics import report_to_markdown


def _script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _img_bytes(color: int, size=(64, 48), fmt="PNG") -> bytes:
    bio = io.BytesIO()
    Image.fromarray(np.full((*size, 3), color, dtype=np.uint8)).save(bio, format=fmt)
    return bio.getvalue()


def _genimage_leaf(root: Path, split: str, dir_name: str, gen_name: str,
                   n_reals: int = 4, n_fakes: int = 4, n_dupe_reals: int = 0, n_shards: int = 4):
    """Reals-first/fakes-last leaf bundle; n_dupe_reals reals reuse ADM's shared_real_* basenames."""
    leaf = root / "data" / split / dir_name
    leaf.mkdir(parents=True, exist_ok=True)
    paths, imgs, labels, gens = [], [], [], []
    for i in range(n_reals):
        name = f"shared_real_{i:03d}.JPEG" if i < n_dupe_reals else f"{gen_name}_real_{i:03d}.JPEG"
        paths.append(f"{gen_name}/{split}/nature/{name}")
        imgs.append(_img_bytes(30 + i))
        labels.append(0)
        gens.append(gen_name)
    for i in range(n_fakes):
        paths.append(f"{gen_name}/{split}/ai/fake_{i:03d}.PNG")
        imgs.append(_img_bytes(200 + i, size=(64, 64)))
        labels.append(1)
        gens.append(gen_name)
    features = hf_datasets.Features({
        "image_path": hf_datasets.Value("string"),
        "image": hf_datasets.Value("binary"),
        "label": hf_datasets.ClassLabel(names=["real", "fake"]),
        "generator": hf_datasets.ClassLabel(names=list(GENIMAGE_DIRS.values())),
    })
    ds = hf_datasets.Dataset.from_dict(
        {"image_path": paths, "image": imgs, "label": labels, "generator": gens},
        features=features)
    ds.save_to_disk(str(leaf), num_shards=n_shards)
    return leaf


def test_select_shards_spreads_and_caps():
    assert select_shards(10, 4) == [0, 3, 6, 9]
    assert select_shards(3, None) == [0, 1, 2]
    assert select_shards(3, 5) == [0, 1, 2]
    assert select_shards(1, 1) == [0]


def _genimage_root(tmp_path: Path) -> Path:
    root = tmp_path / "genimage"
    for split in ("train", "test"):
        _genimage_leaf(root, split, "ADM", "ADM", n_reals=4, n_fakes=6, n_dupe_reals=4)
        _genimage_leaf(root, split, "glide", "glide", n_reals=4, n_fakes=2, n_dupe_reals=2)
    return root


def test_genimage_loader_meta_align_balance_dedupe(tmp_path):
    root = _genimage_root(tmp_path)
    ds = GenImageArrowDataset(root, "test", generators=("ADM", "GLIDE"), align_size=128, seed=0)
    meta = [ds.meta(i) for i in range(len(ds))]
    assert all(g in ("real", "ADM", "GLIDE") for _, g in meta)
    # balance=True: ADM keeps 4 reals + 4 of 6 fakes; glide keeps its 2 unique reals + 2 fakes
    assert ds.stats["ADM"]["fakes"] == 4 and ds.stats["GLIDE"]["reals"] == 2
    # glide shares 2 shared_real_* basenames with ADM -> dropped on the second leaf
    assert sum(s["dupe_reals_dropped"] for s in ds.stats.values()) == 2
    img, label, gen = ds[0]
    assert img.size == (128, 128) and label in (0, 1)
    groups = {ds.group(i) for i in range(len(ds))}
    assert "shared_real_000.JPEG" in groups


def test_genimage_quality_key_deterministic():
    from dfd.data.align import quality_for

    assert quality_for("ADM/test/ai/354_adm_34.PNG") == quality_for("ADM/test/ai/354_adm_34.PNG")
    assert 75 <= quality_for("x") <= 95


def test_genimage_missing_generator_dir(tmp_path):
    root = _genimage_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="Midjourney"):
        GenImageArrowDataset(root, "test", generators=("Midjourney",))


def test_genimage_arrow_build_base_datasets(tmp_path):
    root = _genimage_root(tmp_path)
    cfg = DataConfig(source="genimage_arrow", data_root=str(tmp_path), align_size=64,
                     genimage_generators=("ADM", "GLIDE"), genimage_train_shards_per_generator=2,
                     holdout_generators=("GLIDE",))
    train, val = build_base_datasets(cfg, seed=0)
    train_gens = {train.meta(i)[1] for i in range(len(train))}
    assert "GLIDE" not in train_gens and "ADM" in train_gens
    val_gens = {val.meta(i)[1] for i in range(len(val))}
    assert {"ADM", "GLIDE"} <= val_gens
    cfg2 = DataConfig(source="genimage_arrow", data_root=str(tmp_path),
                      genimage_train_shards_per_generator=0)
    with pytest.raises(ValueError, match=">= 1"):
        build_base_datasets(cfg2, seed=0)


def _wild_tar(path: Path, members: list[tuple[str, int]]):
    with tarfile.open(path, "w") as tf:  # uncompressed tar named .tar.gz, as in the real repo
        for name, color in members:
            data = _img_bytes(color, size=(32, 32))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _wilddeepfake_root(tmp_path: Path) -> Path:
    root = tmp_path / "wilddeepfake"
    (root / "deepfake_in_the_wild" / "fake_test").mkdir(parents=True)
    (root / "deepfake_in_the_wild" / "real_test").mkdir(parents=True)
    _wild_tar(root / "deepfake_in_the_wild" / "fake_test" / "1.tar.gz",
              [(f"./1/fake/3/{f}.png", 100 + f) for f in range(10)])
    _wild_tar(root / "deepfake_in_the_wild" / "real_test" / "7.tar.gz",
              [(f"./7/real/2/{f}.png", 200 + f) for f in range(6)])
    return root


def test_wilddeepfake_index_and_dataset(tmp_path):
    root = _wilddeepfake_root(tmp_path)
    index = build_wilddeepfake_index(root)
    assert len(index) == 16 and {e["label"] for e in index} == {0, 1}
    write_wilddeepfake_index(root, index)
    ds = WildDeepfakeDataset(root, frames_per_clip=4)
    assert len(ds) == 8  # 10 fake frames -> 4, 6 real frames -> 4
    imgs = [ds[i] for i in range(len(ds))]
    assert all(img.size == (32, 32) for img, _, _ in imgs)
    assert {gen for _, _, gen in imgs} == {"wilddeepfake", "real"}
    assert {ds.clip_id(i) for i in range(len(ds))} == {"fake/1/3", "real/7/2"}
    metas = {ds.meta(i) for i in range(len(ds))}
    assert metas == {(1, "wilddeepfake"), (0, "real")}
    pickle.loads(pickle.dumps(ds))  # dataset must survive DataLoader-worker pickling


def test_wilddeepfake_label_dir_mismatch(tmp_path):
    root = tmp_path / "wild"
    (root / "deepfake_in_the_wild" / "fake_test").mkdir(parents=True)
    (root / "deepfake_in_the_wild" / "real_test").mkdir(parents=True)
    _wild_tar(root / "deepfake_in_the_wild" / "fake_test" / "9.tar.gz",
              [("./9/real/1/0.png", 50)])  # real member inside fake_test
    with pytest.raises(ValueError, match="real"):
        build_wilddeepfake_index(root)


def _sfhq_zip(path: Path) -> Path:
    zpath = path / "SyntheticFacesHighQuality-T2I.zip"
    names = ([f"images/FLUX1_dev_image_{i:07d}.jpg" for i in range(4)]
             + [f"images/SDXL_image_{i:07d}.jpg" for i in range(3)]
             + ["images/README.txt"])
    with zipfile.ZipFile(zpath, "w") as zf:
        for n in names:
            payload = b"readme" if n.endswith(".txt") else _img_bytes(150, fmt="JPEG")
            zf.writestr(n, payload)
    return zpath


def test_sfhq_dataset(tmp_path):
    zpath = _sfhq_zip(tmp_path)
    ds = SFHQT2IDataset(zpath, max_per_generator=2, align_size=96, seed=0)
    assert ds.stats == {"FLUX1-dev": 2, "SDXL": 2}
    img, label, gen = ds[0]
    assert img.size == (96, 96) and label == 1 and gen in ("FLUX1-dev", "SDXL")
    pickle.loads(pickle.dumps(ds))


def test_report_to_markdown_renders_eval_extra():
    report = {"overall": {"n": 4, "auc": 0.9},
              "eval_extra": {"sowaiba": {"report": {"overall": {"n": 8, "auc": 0.5},
                                                   "by_generator": {"inswapper": {"n": 4}}},
                                         "verdicts_by_method": {}},
                             "empty_no_report": {"verdicts_by_method": {}}}}
    md = report_to_markdown(report, "t")
    assert "## eval_extra/sowaiba" in md


def test_plan_files(tmp_path):
    mod = _script("modal_data")

    tree = []
    for split in ("train", "test"):
        for d in ("ADM", "glide"):
            for k in range(6):
                tree.append((f"data/{split}/{d}/data-{k:05d}-of-00006.arrow", 100))
            tree.append((f"data/{split}/{d}/dataset_info.json", 1))
            tree.append((f"data/{split}/{d}/bundle_manifest.json", 1))
    for kind in ("real_test", "fake_test", "real_train", "fake_train"):
        tree.append((f"deepfake_in_the_wild/{kind}/1.tar.gz", 10))
        tree.append((f"deepfake_in_the_wild/{kind}/2.tar.gz", 10))
    tree.append(("SyntheticFacesHighQuality-T2I.zip", 22_900_000_000))

    sel = mod.plan_files("genimage", tree, generators=("ADM",), train_shards=3)
    paths = {p for p, _ in sel}
    assert not any("/glide/" in p for p in paths)
    assert sum(1 for p in paths if p.startswith("data/train/") and p.endswith(".arrow")) == 3
    assert sum(1 for p in paths if p.startswith("data/test/") and p.endswith(".arrow")) == 6
    assert any(p.endswith(".json") for p in paths)

    smoke = mod.plan_files("genimage", tree, generators=("ADM", "GLIDE"), smoke=True)
    assert {p.split("/")[2] for p, _ in smoke} == {"ADM"}
    assert sum(1 for p, _ in smoke if p.endswith(".arrow")) == 2

    wd = mod.plan_files("wilddeepfake", tree)
    assert {p.split("/")[1] for p, _ in wd} == {"real_test", "fake_test"}
    assert not any("_train" in p for p, _ in wd)
    assert len(mod.plan_files("wilddeepfake", tree, smoke=True)) == 2

    assert mod.plan_files("sfhq_t2i", tree) == [("SyntheticFacesHighQuality-T2I.zip", 22_900_000_000)]
    assert mod.plan_files("sfhq_t2i", tree, smoke=True) == []


def test_verdict_rates_respect_calibration_status():
    from types import SimpleNamespace  # noqa: F401
    from dfd.eval.calibration import Calibration

    ee = _script("eval_external")
    cal = Calibration(temperature=1.0, tau_fake=0.9, tau_real=0.1)
    logits = np.array([-5.0, 5.0, 0.0])
    labels = np.array([0, 1, 1])
    m = ee._verdict_rates(cal, logits, labels, "matched")
    assert m["verdicts_valid"] is True and m["calibration_status"] == "matched"
    assert m["fpr_fake_verdict"] == 0.0 and m["fnr_real_verdict"] == 0.0 and m["abstain_rate"] == 1 / 3
    for status in ("missing", "mismatched"):
        r = ee._verdict_rates(cal, logits, labels, status)
        assert r["abstain_rate"] == 1.0 and r["verdicts_valid"] is False
        assert r["fpr_fake_verdict"] == 0.0 and r["fnr_real_verdict"] == 0.0
        assert r["frac_verdict_fake"] == 0.0 and r["calibration_status"] == status


def test_eval_wilddeepfake_clip_verdicts_omitted(tmp_path):
    import torch
    from types import SimpleNamespace

    from dfd.eval.calibration import Calibration

    ee = _script("eval_external")
    root = _wilddeepfake_root(tmp_path)
    write_wilddeepfake_index(root, build_wilddeepfake_index(root))
    ds = WildDeepfakeDataset(root, frames_per_clip=4)

    class _Model:
        def __call__(self, pixels):
            return {"logit": torch.zeros(len(pixels))}

    pipe = SimpleNamespace(
        model=_Model(),
        transform=lambda img, rng: (torch.zeros(3, 8, 8), None),
        cal=Calibration(temperature=1.0, tau_fake=0.9, tau_real=0.1),
        calibration_status="matched",
        device="cpu",
    )
    rep = ee._eval_wilddeepfake(pipe, ds, batch_size=4, num_workers=0)
    assert rep["clip"]["verdicts"] is None and "single-image" in rep["clip"]["note"]
    assert rep["clip"]["n_clips"] == 2
    assert rep["frame"]["verdicts"]["verdicts_valid"] is True


def test_eval_names():
    mod = _script("modal_data")
    assert mod._eval_names("all") == ["wilddeepfake", "sfhq_t2i", "fairface"]
    assert mod._eval_names("sfhq_t2i") == ["sfhq_t2i"]
    with pytest.raises(SystemExit):
        mod._eval_names("genimage")


def test_download_files_always_calls_download_fn(tmp_path):
    mod = _script("modal_data")
    dest = tmp_path / "data"
    files = [("x.bin", 4), ("y.bin", 4)]
    (dest).mkdir(parents=True)
    (dest / "x.bin").write_bytes(b"\x00" * 4)  # pre-existing same-size file must still be revalidated
    calls = []

    def fake_dl(repo_id, filename, repo_type, revision, local_dir):
        calls.append((filename, revision))
        p = Path(local_dir) / filename
        if not p.exists():
            p.write_bytes(b"\x01" * dict(files)[filename])

    downloaded, reused, n_bytes = mod._download_files("repo", files, "sha123", dest, fake_dl)
    assert sorted(c[0] for c in calls) == ["x.bin", "y.bin"] and all(c[1] == "sha123" for c in calls)  # threaded
    assert (downloaded, reused, n_bytes) == (1, 1, 4)
