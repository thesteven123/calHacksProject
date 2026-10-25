import csv
import io
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from dfd.config import Config, DataConfig
from dfd.data.paired import (
    CompositeDataset,
    DeepFakeFaceDataset,
    SowaibaDataset,
    verify_no_group_leakage,
)
from dfd.data.datasets import build_base_datasets, eval_splits, split_nested_holdouts


def _create_synthetic_dff(tmp_path: Path, num_identities: int = 20) -> Path:
    """Creates a miniature offline synthetic DeepFakeFace directory with 4 zips.
    
    Reals (wiki) are non-square (400x550) with lower JPEG quality (75).
    Fakes (inpainting, insight, text2img) are 512x512 with high JPEG quality (95).
    Without format alignment, this creates a major resolution and quantization shortcut.
    """
    root = tmp_path / "deepfakeface"
    root.mkdir(parents=True, exist_ok=True)

    cats = ["wiki", "inpainting", "insight", "text2img"]
    zip_handles = {cat: zipfile.ZipFile(root / f"{cat}.zip", "w") for cat in cats}

    for i in range(num_identities):
        rel_path = f"folder_{(i % 4):02d}/person_{i:04d}.jpg"
        for cat in cats:
            if cat == "wiki":
                # Real: non-square (width=400, height=550), initial JPEG quality 75
                img_arr = np.zeros((550, 400, 3), dtype=np.uint8)
                img_arr[:, :, 0] = 200
                q_init = 75
            else:
                # Fake: 512x512 square, initial JPEG quality 95
                img_arr = np.zeros((512, 512, 3), dtype=np.uint8)
                if cat == "inpainting":
                    img_arr[:, :, 1] = 200
                elif cat == "insight":
                    img_arr[:, :, 2] = 200
                else:
                    img_arr[:, :, :] = 150
                q_init = 95
            img = Image.fromarray(img_arr)

            bio = io.BytesIO()
            img.save(bio, format="JPEG", quality=q_init)
            zip_handles[cat].writestr(f"{cat}/{rel_path}", bio.getvalue())

    for z in zip_handles.values():
        z.close()

    return root


def _create_synthetic_sowaiba(tmp_path: Path, num_pairs: int = 20) -> Path:
    """Creates a miniature offline synthetic Sowaiba directory with fake/, real/, and metadata.csv."""
    root = tmp_path / "sowaiba"
    root.mkdir(parents=True, exist_ok=True)
    (root / "fake").mkdir()
    (root / "real").mkdir()

    rows = []
    for i in range(num_pairs):
        real_file = f"real/real_{i:05d}.jpg"
        fake_file = f"fake/fake_{i:05d}.jpg"

        img_real = Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8))
        img_fake = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))

        img_real.save(root / real_file, format="JPEG", quality=95)
        img_fake.save(root / fake_file, format="JPEG", quality=95)

        rows.append({
            "id": str(i),
            "fake_file": fake_file,
            "real_file": real_file,
            "method": "inswapper_128",
            "source_identity": f"src_{i}.jpg",
            "target_face": f"tgt_{i}.jpg",
            "label_fake": "1",
            "label_real": "0",
        })

    with open(root / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return root


def test_dff_offline_loading_and_contract(tmp_path):
    root = _create_synthetic_dff(tmp_path, num_identities=15)
    ds = DeepFakeFaceDataset(root=root, generators=("inpainting", "insight"), split="all")

    # 15 identities * (1 real + 2 fakes) = 45 images
    assert len(ds) == 45

    img, label, gen = ds[0]
    assert isinstance(img, Image.Image)
    assert label in (0, 1)
    assert gen in ("real", "inpainting", "insight")
    # Default format_align downscales to 384x384
    assert img.size == (384, 384)

    lbl, g = ds.meta(0)
    assert lbl == label and g == gen


def test_dff_format_alignment_and_quantization_parity(tmp_path):
    root = _create_synthetic_dff(tmp_path, num_identities=10)

    # 1. Test WITH format alignment (default)
    ds_aligned = DeepFakeFaceDataset(root=root, split="all", format_align=True)

    # Group items by group_id
    grouped_samples: dict[str, list[tuple[Image.Image, int, str]]] = {}
    for i in range(len(ds_aligned)):
        img, lbl, gen = ds_aligned[i]
        grp = ds_aligned.group(i)
        grouped_samples.setdefault(grp, []).append((img, lbl, gen))

    assert len(grouped_samples) == 10
    for grp, samples in grouped_samples.items():
        reals = [s for s in samples if s[1] == 0]
        fakes = [s for s in samples if s[1] == 1]
        assert len(reals) == 1 and len(fakes) == 3

        real_img, _, _ = reals[0]
        # Size contract: exactly 384x384
        assert real_img.size == (384, 384)
        real_quant = getattr(real_img, "quantization", None)
        assert real_quant is not None

        for fake_img, _, fake_gen in fakes:
            assert fake_img.size == (384, 384)
            fake_quant = getattr(fake_img, "quantization", None)
            assert fake_quant is not None
            # Quantization parity: real and fake in the same pair group share the identical quantization table
            assert fake_quant == real_quant, f"Quant mismatch for generator {fake_gen} in group {grp}"

    # 2. Test WITHOUT format alignment (verify that shortcut exists without alignment)
    ds_unaligned = DeepFakeFaceDataset(root=root, split="all", format_align=False)
    for i in range(len(ds_unaligned)):
        img, lbl, gen = ds_unaligned[i]
        if lbl == 0:
            assert img.size == (400, 550)
        else:
            assert img.size == (512, 512)


def test_dff_pair_grouped_split_no_leakage(tmp_path):
    root = _create_synthetic_dff(tmp_path, num_identities=30)
    train_ds = DeepFakeFaceDataset(root=root, split="train", val_frac=0.3, seed=42)
    val_ds = DeepFakeFaceDataset(root=root, split="val", val_frac=0.3, seed=42)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(train_ds) + len(val_ds) == 30 * 4  # 30 * (1 real + 3 fakes)

    # Prove that no pair group (relative path) leaks across splits
    assert verify_no_group_leakage(train_ds, val_ds) is True

    # Double-check manually with set intersection
    train_groups = {train_ds.group(i) for i in range(len(train_ds))}
    val_groups = {val_ds.group(i) for i in range(len(val_ds))}
    assert len(train_groups & val_groups) == 0


def test_sowaiba_offline_loading_and_grouping(tmp_path):
    root = _create_synthetic_sowaiba(tmp_path, num_pairs=20)
    train_ds = SowaibaDataset(root=root, split="train", val_frac=0.25, seed=7)
    val_ds = SowaibaDataset(root=root, split="val", val_frac=0.25, seed=7)

    assert len(train_ds) + len(val_ds) == 40  # 20 pairs * 2 (1 real + 1 fake)
    assert verify_no_group_leakage(train_ds, val_ds) is True

    # Check labels and generators
    labels = [train_ds.meta(i) for i in range(len(train_ds))]
    fakes = [g for y, g in labels if y == 1]
    reals = [g for y, g in labels if y == 0]
    assert all(g == "inswapper_128" for g in fakes)
    assert all(g == "real" for g in reals)


def test_composite_dataset_contract(tmp_path):
    root_dff = _create_synthetic_dff(tmp_path / "dff", num_identities=10)
    root_sow = _create_synthetic_sowaiba(tmp_path / "sow", num_pairs=10)

    ds_dff = DeepFakeFaceDataset(root=root_dff, split="all")
    ds_sow = SowaibaDataset(root=root_sow, split="all")

    comp = CompositeDataset([ds_dff, ds_sow])
    assert len(comp) == len(ds_dff) + len(ds_sow)

    # Test indexing and metadata
    img0, lbl0, gen0 = comp[0]
    assert isinstance(img0, Image.Image)
    assert comp.meta(0) == (lbl0, gen0)

    # Check last element (from Sowaiba)
    last_idx = len(comp) - 1
    img_last, lbl_last, gen_last = comp[last_idx]
    assert comp.meta(last_idx) == (lbl_last, gen_last)


def test_paired_faces_nested_split_and_sowaiba_isolation(tmp_path):
    root_dff = _create_synthetic_dff(tmp_path / "dff", num_identities=30)
    root_sow = _create_synthetic_sowaiba(tmp_path / "sow", num_pairs=15)

    cfg = DataConfig(
        source="paired_faces",
        hf_dataset="",
        hf_data_files=None,
        dff_root=str(root_dff),
        dff_generators=("inpainting", "insight", "text2img"),
        dff_val_fraction=0.3,
        selection_generators=("text2img",),
        holdout_generators=("inpainting",),
        sowaiba_root=str(root_sow),
        sowaiba_eval=True,
    )

    train_ds, val_ds = build_base_datasets(cfg, seed=123)

    # Verify train_ds has only in-distribution fakes
    train_gens = {train_ds.meta(i)[1] for i in range(len(train_ds)) if train_ds.meta(i)[0] == 1}
    assert train_gens == {"insight"}
    assert "text2img" not in train_gens
    assert "inpainting" not in train_gens
    assert "inswapper_128" not in train_gens

    # Verify val_ds does NOT contain Sowaiba (Sowaiba is isolated from validation/fit/select)
    val_gens = {val_ds.meta(i)[1] for i in range(len(val_ds)) if val_ds.meta(i)[0] == 1}
    assert "inswapper_128" not in val_gens

    # Verify pair groups between DFF train and DFF val do not leak
    assert verify_no_group_leakage(train_ds, val_ds) is True

    # Test eval_splits: returns fit, select, test (LOGO on DFF) and separate "sowaiba" split
    splits = eval_splits(cfg, val_ds, seed=123)
    assert "fit" in splits and "select" in splits and "test" in splits
    assert "sowaiba" in splits

    # Sowaiba split contains all 15 pairs (30 images) with method inswapper_128
    assert len(splits["sowaiba"]) == 30
    sow_labels = [splits["sowaiba"].meta(i) for i in range(len(splits["sowaiba"]))]
    assert all(g == "inswapper_128" for y, g in sow_labels if y == 1)

    # Generators in fit, select, and test splits
    select_fakes = {splits["select"].meta(i)[1] for i in range(len(splits["select"])) if splits["select"].meta(i)[0] == 1}
    test_fakes = {splits["test"].meta(i)[1] for i in range(len(splits["test"])) if splits["test"].meta(i)[0] == 1}
    fit_fakes = {splits["fit"].meta(i)[1] for i in range(len(splits["fit"])) if splits["fit"].meta(i)[0] == 1}

    assert select_fakes == {"text2img"}
    assert test_fakes == {"inpainting"}
    assert fit_fakes == {"insight"}


def _create_partial_dff(tmp_path: Path, cats: list[str], num_identities: int = 10) -> Path:
    """Like _create_synthetic_dff but writes only the given archive categories."""
    root = tmp_path / "dff_partial"
    root.mkdir(parents=True, exist_ok=True)
    for cat in cats:
        with zipfile.ZipFile(root / f"{cat}.zip", "w") as z:
            for i in range(num_identities):
                rel = f"folder_{(i % 4):02d}/person_{i:04d}.jpg"
                img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
                bio = io.BytesIO()
                img.save(bio, format="JPEG", quality=90)
                z.writestr(f"{cat}/{rel}", bio.getvalue())
    return root


class _TinyGenStub:
    """Singleton-row stub: reals + fakes cycling through the given generators (no group())."""

    def __init__(self, n: int = 90, gens=("ADM", "Midjourney")):
        self.n, self.gens = n, gens

    def __len__(self):
        return self.n

    def meta(self, i):
        if i % 3 == 0:
            return 0, "real"
        return 1, self.gens[(i // 3) % len(self.gens)]


def _multi_groups(comp, subset):
    from collections import Counter
    gids = [comp.group(i) for i in subset.indices]
    counts = Counter(gids)
    return {g for g in gids if counts[g] > 1}


def test_nested_split_keeps_pair_groups_isolated(tmp_path):
    root = _create_synthetic_dff(tmp_path / "dff", num_identities=60)
    dff = DeepFakeFaceDataset(root=root, split="all")
    comp = CompositeDataset([_TinyGenStub(), dff])
    fit, select, test = split_nested_holdouts(comp, ("text2img",), ("Midjourney",), seed=0)

    f_g, s_g, t_g = (_multi_groups(comp, s) for s in (fit, select, test))
    assert not (f_g & s_g) and not (f_g & t_g) and not (s_g & t_g)

    def fakes(sub):
        return {comp.meta(i)[1] for i in sub.indices if comp.meta(i)[0] == 1}
    assert fakes(select) == {"text2img"}
    assert fakes(test) == {"Midjourney"}
    assert fakes(fit) <= {"ADM", "inpainting", "insight"} and fakes(fit)


def test_val_holdout_keeps_pair_groups_isolated(tmp_path):
    from dfd.data.datasets import split_val_holdout

    root = _create_synthetic_dff(tmp_path / "dff", num_identities=60)
    dff = DeepFakeFaceDataset(root=root, split="all")
    comp = CompositeDataset([_TinyGenStub(), dff])
    sel, test = split_val_holdout(comp, ("text2img",), seed=0, test_real_frac=0.5)

    s_g, t_g = _multi_groups(comp, sel), _multi_groups(comp, test)
    assert not (s_g & t_g)
    assert all(comp.meta(i)[1] == "text2img" for i in test.indices if comp.meta(i)[0] == 1)
    assert all(comp.meta(i)[1] != "text2img" for i in sel.indices)


def test_singleton_rows_follow_original_algorithm():
    """All-singleton datasets must produce exactly the pre-grouping indices."""
    from dfd.data.datasets import split_val_holdout

    val = _TinyGenStub(gens=("base", "selector", "final"))

    def old_split_val_holdout(v, holdout, seed=0, frac=0.5):
        sel, test_fakes, reals = [], [], []
        for i in range(len(v)):
            y, g = v.meta(i)
            if y == 0:
                reals.append(i)
            elif g in holdout:
                test_fakes.append(i)
            else:
                sel.append(i)
        reals = np.random.default_rng(seed).permutation(reals).tolist()
        n = int(round(len(reals) * frac))
        return sorted(sel + reals[n:]), sorted(test_fakes + reals[:n])

    sel, test = split_val_holdout(val, ("final",), seed=3)
    exp_sel, exp_test = old_split_val_holdout(val, ("final",), seed=3)
    assert sel.indices == exp_sel and test.indices == exp_test

    # reference copy of the pre-grouping nested algorithm
    fit, select, test = split_nested_holdouts(val, ("selector",), ("final",), seed=7)
    f, s, t, reals = [], [], [], []
    for i in range(len(val)):
        y, g = val.meta(i)
        (reals if y == 0 else s if g == "selector" else t if g == "final" else f).append(i)
    reals = np.random.default_rng(7).permutation(reals).tolist()
    th = (len(reals) // 3, 2 * len(reals) // 3)
    assert fit.indices == sorted(f + reals[:th[0]])
    assert select.indices == sorted(s + reals[th[0]:th[1]])
    assert test.indices == sorted(t + reals[th[1]:])


def test_partial_dff_root_fails_on_missing_types(tmp_path):
    root = _create_partial_dff(tmp_path, ["wiki", "inpainting"])
    with pytest.raises(FileNotFoundError, match="insight"):
        DeepFakeFaceDataset(root=root, generators=("inpainting", "insight"), split="all")
    root2 = _create_partial_dff(tmp_path / "nowiki", ["inpainting"])
    with pytest.raises(FileNotFoundError, match="wiki"):
        DeepFakeFaceDataset(root=root2, generators=("inpainting",), include_reals=True, split="all")
    with pytest.raises(ValueError, match="root"):
        DeepFakeFaceDataset(root=None, download=False)
