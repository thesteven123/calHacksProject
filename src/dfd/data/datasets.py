"""Dataset sources (synthetic, Hugging Face, folder) and the transformed training wrapper.

Every source yields `(PIL.Image, label: int, generator: str)` with label 1 = fake.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dfd.config import DataConfig
from dfd.data.transforms import DetectorTransform

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SyntheticDeepfakeDataset(Dataset):
    """Procedural stand-in for smoke tests. Fakes carry a periodic upsampling grid (a GAN-like trace)."""

    def __init__(self, size: int, image_size: int = 256, seed: int = 0):
        self.n, self.image_size, self.seed = size, image_size, seed

    def __len__(self) -> int:
        return self.n

    def meta(self, idx: int) -> tuple[int, str]:
        label = idx % 2
        return label, ("synth_gan" if idx % 4 == 1 else "synth_diff") if label else "real"

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        s = self.image_size
        label = idx % 2
        yy, xx = np.mgrid[0:s, 0:s] / s
        base = np.stack([rng.uniform(0, 1) * xx + rng.uniform(0, 1) * yy + rng.uniform(0, 0.5) for _ in range(3)], -1)
        for _ in range(int(rng.integers(3, 8))):
            cx, cy, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.05, 0.3)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2))
            base += blob[..., None] * rng.uniform(-0.5, 0.5, 3)
        if label == 1:
            low = base[::4, ::4]
            base = np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)[:s, :s]
            base += 0.04 * (np.sin(xx * s * np.pi / 2) * np.sin(yy * s * np.pi / 2))[..., None]
            generator = "synth_gan" if idx % 4 == 1 else "synth_diff"
        else:
            base += rng.normal(0, 0.03, base.shape)
            generator = "real"
        arr = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr), label, generator


class HFImageDataset(Dataset):
    """Hugging Face image dataset with optional generator include/exclude (leave-one-generator-out)."""

    def __init__(
        self,
        cfg: DataConfig,
        split: str,
        include_generators: tuple[str, ...] | None = None,
        exclude_generators: tuple[str, ...] = (),
        max_samples: int | None = None,
    ):
        import datasets

        if cfg.hf_data_files:
            ds = datasets.load_dataset("parquet", data_files=cfg.hf_data_files, split=split)
        else:
            ds = datasets.load_dataset(cfg.hf_dataset, split=split)
        self.cfg = cfg
        gcol = cfg.generator_column
        self.gen_names = None
        self._meta: list[tuple[int, str]] | None = None
        self.has_generators = bool(gcol and gcol in ds.column_names)
        if (include_generators is not None or exclude_generators) and not self.has_generators:
            # Fail fast: silently skipping the filter would train on held-out generators (CODE_REVIEW 2026-10-24).
            raise ValueError(f"Generator filtering requested but column {gcol!r} is missing from {split!r} "
                             f"(columns: {ds.column_names})")
        if self.has_generators:
            feat = ds.features[gcol]
            self.gen_names = feat.names if hasattr(feat, "names") else None
            labels = ds[cfg.label_column]
            self._meta = [(int(y), self._gen_name(g)) for g, y in zip(ds[gcol], labels)]
            keep = [
                i for i, (y, g) in enumerate(self._meta)
                if g not in exclude_generators and (include_generators is None or y == 0 or g in include_generators)
            ]
            ds = ds.select(keep)
            self._meta = [self._meta[i] for i in keep]
        if max_samples is not None and len(ds) > max_samples:
            order = np.random.default_rng(0).permutation(len(ds))[:max_samples].tolist()
            ds = ds.select(order)
            if self._meta is not None:
                self._meta = [self._meta[i] for i in order]  # keep metadata aligned with the subsample
        self.ds = ds

    def meta(self, idx: int) -> tuple[int, str]:
        if self._meta is not None:
            return self._meta[idx]
        row = self.ds[idx]
        label = int(row[self.cfg.label_column])
        gcol = self.cfg.generator_column
        gen = self._gen_name(row[gcol]) if gcol and gcol in row else ("real" if label == 0 else "fake")
        return label, gen

    def _gen_name(self, g) -> str:
        return self.gen_names[g] if self.gen_names is not None and isinstance(g, int) else str(g)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        row = self.ds[idx]
        label = int(row[self.cfg.label_column])
        gcol = self.cfg.generator_column
        gen = self._gen_name(row[gcol]) if gcol and gcol in row else ("real" if label == 0 else "fake")
        return row[self.cfg.image_column].convert("RGB"), label, gen


class FolderDataset(Dataset):
    """Layout: root/real/**/*.jpg and root/fake/<generator>/**/*.png."""

    def __init__(self, root: str | Path, exclude_generators: tuple[str, ...] = ()):
        root = Path(root)
        self.items: list[tuple[Path, int, str]] = []
        for p in sorted((root / "real").rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                self.items.append((p, 0, "real"))
        for p in sorted((root / "fake").rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                rel = p.relative_to(root / "fake").parts
                gen = rel[0] if len(rel) > 1 else "fake"
                if gen not in exclude_generators:
                    self.items.append((p, 1, gen))
        if not self.items:
            raise FileNotFoundError(f"No images found under {root}/real or {root}/fake")

    def __len__(self) -> int:
        return len(self.items)

    def meta(self, idx: int) -> tuple[int, str]:
        return self.items[idx][1], self.items[idx][2]

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        path, label, gen = self.items[idx]
        return Image.open(path).convert("RGB"), label, gen


class SubsetDataset(Dataset):
    def __init__(self, base: Dataset, indices: list[int]):
        self.base, self.indices = base, indices

    def __len__(self) -> int:
        return len(self.indices)

    def meta(self, idx: int) -> tuple[int, str]:
        return self.base.meta(self.indices[idx]) if hasattr(self.base, "meta") else self.__getitem__(idx)[1:3]

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        return self.base[self.indices[idx]]


class TransformedDataset(Dataset):
    """Applies DetectorTransform with a per-(epoch, index) RNG so eval is deterministic and train varies."""

    def __init__(self, base: Dataset, transform: DetectorTransform, seed: int = 0):
        self.base, self.transform, self.seed, self.epoch = base, transform, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        img, label, gen = self.base[idx]
        epoch = self.epoch if self.transform.train else 0
        rng = np.random.default_rng([self.seed, epoch, idx])
        pixels, meta = self.transform(img, rng)
        return {
            "pixels": pixels,
            "label": label,
            "beauty": int(meta.beauty),
            "generator": gen,
            "severity": meta.severity,
        }


def _require_each_generator(fake_counts: dict[str, int], role: str, generators: tuple[str, ...]) -> None:
    """Every configured generator must contribute validation fakes, not just the group as a whole.

    Otherwise a multi-generator rotation where one name matches nothing (typo, absent from the shard subset)
    silently runs as a smaller rotation than requested (CODE_REVIEW 2026-10-24 re-review, Medium).
    """
    missing = [g for g in generators if fake_counts.get(g, 0) == 0]
    if missing:
        raise ValueError(f"{role} generators {missing} have no fakes in validation; "
                         f"generators present: {sorted(fake_counts)}")


def _fake_generator_counts(val: Dataset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(len(val)):
        label, gen = val.meta(i)
        if label:
            counts[gen] = counts.get(gen, 0) + 1
    return counts


def split_val_holdout(
    val: Dataset, holdout_generators: tuple[str, ...], seed: int = 0, test_real_frac: float = 0.5
) -> tuple[SubsetDataset, SubsetDataset]:
    """Split validation into disjoint (selection, test) sets.

    `selection`: in-distribution fakes + a share of reals. It drives best-checkpoint choice and calibration.
    `test`: held-out-generator fakes + the remaining reals. No selection decision touches it, so this is the
    honest leave-one-generator-out estimate. Reals are split (not shared) so FPR thresholds calibrated on
    selection reals are not re-scored on the same images. Without holdouts, test is empty.
    """
    sel, test_fakes, reals = [], [], []
    for i in range(len(val)):
        label, gen = val.meta(i)
        if label == 0:
            reals.append(i)
        elif gen in holdout_generators:
            test_fakes.append(i)
        else:
            sel.append(i)
    if holdout_generators:
        # Each configured holdout must match validation fakes (CODE_REVIEW 2026-10-24 + re-review).
        _require_each_generator(_fake_generator_counts(val), "Holdout", tuple(holdout_generators))
    if not test_fakes:
        return SubsetDataset(val, sorted(sel + reals)), SubsetDataset(val, [])
    reals = np.random.default_rng(seed).permutation(reals).tolist()
    n_test_reals = int(round(len(reals) * test_real_frac))
    return SubsetDataset(val, sorted(sel + reals[n_test_reals:])), SubsetDataset(val, sorted(test_fakes + reals[:n_test_reals]))


def split_nested_holdouts(
    val: Dataset, selection_generators: tuple[str, ...], test_generators: tuple[str, ...], seed: int = 0
) -> tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
    """Disjoint fit, OOD checkpoint-selection, and untouched final-test sets."""
    if not selection_generators or not test_generators or set(selection_generators) & set(test_generators):
        raise ValueError("Nested LOGO requires distinct, non-empty selection and test generator sets")
    counts = _fake_generator_counts(val)
    _require_each_generator(counts, "Selection", tuple(selection_generators))
    _require_each_generator(counts, "Test", tuple(test_generators))
    fit, selection, test, reals = [], [], [], []
    for i in range(len(val)):
        label, generator = val.meta(i)
        if not label:
            reals.append(i)
        elif generator in selection_generators:
            selection.append(i)
        elif generator in test_generators:
            test.append(i)
        else:
            fit.append(i)
    if not (fit and selection and test and len(reals) >= 3):
        seen = sorted({val.meta(i)[1] for i in fit + selection + test})
        raise ValueError(
            "Nested LOGO needs fake samples in each split and at least three real images: "
            f"fit={len(fit)} select={len(selection)} test={len(test)} reals={len(reals)}; "
            f"selection={list(selection_generators)} test={list(test_generators)} generators present: {seen}"
        )
    reals = np.random.default_rng(seed).permutation(reals).tolist()
    thirds = (len(reals) // 3, 2 * len(reals) // 3)
    fit.extend(reals[:thirds[0]])
    selection.extend(reals[thirds[0]:thirds[1]])
    test.extend(reals[thirds[1]:])
    return tuple(SubsetDataset(val, sorted(indices)) for indices in (fit, selection, test))


def eval_splits(cfg: DataConfig, base_val: Dataset, seed: int = 0) -> dict[str, Dataset]:
    """Named validation splits, identical to scripts/train.py for the same seed."""
    sel, test = tuple(cfg.selection_generators), tuple(cfg.holdout_generators)
    if sel:
        fit, select, final = split_nested_holdouts(base_val, sel, test, seed)
    else:
        select, final = split_val_holdout(base_val, test, seed=seed)
        fit = select
    return {"all": base_val, "fit": fit, "select": select, "test": final}


def build_base_datasets(cfg: DataConfig, seed: int = 0) -> tuple[Dataset, Dataset]:
    holdout = tuple(cfg.holdout_generators) + tuple(cfg.selection_generators)
    if cfg.source == "synthetic":
        train = SyntheticDeepfakeDataset(cfg.synthetic_train_size, cfg.synthetic_image_size, seed)
        if holdout:
            train = SubsetDataset(train, [i for i in range(len(train)) if train.meta(i)[1] not in holdout])
        return train, SyntheticDeepfakeDataset(cfg.synthetic_val_size, cfg.synthetic_image_size, seed + 1)
    if cfg.source == "hf":
        train = HFImageDataset(cfg, cfg.hf_train_split, exclude_generators=holdout, max_samples=cfg.max_train_samples)
        val = HFImageDataset(cfg, cfg.hf_val_split, max_samples=cfg.max_val_samples)
        if holdout and not val.has_generators:
            raise ValueError(f"Generator holdouts {list(holdout)} need column {cfg.generator_column!r} "
                             f"in validation split {cfg.hf_val_split!r}")
        return train, val
    if cfg.source == "folder":
        if not cfg.folder_root:
            raise ValueError("data.folder_root is required for source=folder")
        root = Path(cfg.folder_root)
        return FolderDataset(root / "train", holdout), FolderDataset(root / "val")
    raise ValueError(f"Unknown data source: {cfg.source}")
