"""Dataset sources (synthetic, Hugging Face, folder) and the transformed training wrapper.

Every source yields `(PIL.Image, label: int, generator: str)` with label 1 = fake.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dfd.config import DataConfig
from dfd.data.paired import (FACE_ALIGN_SIZE, FACE_JPEG_QUALITY, CompositeDataset, DeepFakeFaceDataset, SowaibaDataset,
                             verify_no_group_leakage)
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


def pick_clip_frames(n_stored: int, n_out: int, train: bool, rng: np.random.Generator) -> tuple[list[int], list[bool]]:
    """Frame indices + validity mask. Eval: evenly spaced; train: same stride from a random start offset. Too few
    stored frames -> all of them, then the last one repeated with mask False."""
    if n_stored < n_out:
        pad = n_out - n_stored
        return list(range(n_stored)) + [n_stored - 1] * pad, [True] * n_stored + [False] * pad
    if not train:
        return np.linspace(0, n_stored - 1, n_out).round().astype(int).tolist(), [True] * n_out
    stride = n_stored / n_out
    start = rng.uniform(0, stride)
    return [min(int(start + k * stride), n_stored - 1) for k in range(n_out)], [True] * n_out


class TransformedDataset(Dataset):
    """Applies DetectorTransform with a per-(epoch, index) RNG so eval is deterministic and train varies.

    Clip items (base image is a list of frames) yield frames_per_clip frames [T,3,H,W] + mask [T]; every frame is
    transformed with an identically re-created rng so PTE/crop/flip parameters are consistent within a clip.
    """

    def __init__(self, base: Dataset, transform: DetectorTransform, seed: int = 0, frames_per_clip: int = 8):
        self.base, self.transform, self.seed, self.epoch = base, transform, seed, 0
        self.frames_per_clip = frames_per_clip

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        img, label, gen = self.base[idx]
        epoch = self.epoch if self.transform.train else 0
        if isinstance(img, list):
            return self._clip(img, label, gen, epoch, idx)
        rng = np.random.default_rng([self.seed, epoch, idx])
        pixels, meta = self.transform(img, rng)
        return {
            "pixels": pixels,
            "label": label,
            "beauty": int(meta.beauty),
            "generator": gen,
            "severity": meta.severity,
        }


    def _clip(self, frames: list, label: int, gen: str, epoch: int, idx: int) -> dict:
        key = [self.seed, epoch, idx]
        picks, mask = pick_clip_frames(len(frames), self.frames_per_clip, self.transform.train,
                                       np.random.default_rng(key))
        out = [self.transform(frames[i], np.random.default_rng(key)) for i in picks]
        return {
            "pixels": torch.stack([p for p, _ in out]),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": label,
            "beauty": 0,
            "generator": gen,
            "severity": out[0][1].severity,
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


def _row_groups(val: Dataset) -> tuple[list[int], dict[str, list[int]]]:
    """Partition rows into singleton indices and multi-row pair groups.

    Group id is `val.group(i)` when the dataset provides it (DeepFakeFace identity; CompositeDataset
    prefixes it with the sub-dataset index), else `str(i)` so plain datasets stay singletons and the
    split is byte-identical to the pre-grouping algorithm.
    """
    by_group: dict[str, list[int]] = {}
    for i in range(len(val)):
        gid = val.group(i) if hasattr(val, "group") else str(i)
        by_group.setdefault(gid, []).append(i)
    singles = [idxs[0] for idxs in by_group.values() if len(idxs) == 1]
    multi = {g: idxs for g, idxs in by_group.items() if len(idxs) > 1}
    return singles, multi


def _fake_counts(val: Dataset, indices: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in indices:
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

    Pair-group aware: a multi-row group (e.g. a DeepFakeFace real + its generated counterparts) is assigned
    wholly to selection or test (seeded shuffle of sorted group ids, split by test_real_frac), keeping only
    the fakes whose generator routes to that split so no identity leaks across selection/test. Singleton
    rows follow the original per-row algorithm unchanged.
    """
    singles, multi = _row_groups(val)
    sel, test_fakes, reals = [], [], []
    for i in singles:
        label, gen = val.meta(i)
        if label == 0:
            reals.append(i)
        elif gen in holdout_generators:
            test_fakes.append(i)
        else:
            sel.append(i)
    test_reals: list[int] = []
    if multi and not holdout_generators:
        # No test split possible; every grouped row belongs to selection.
        for idxs in multi.values():
            sel.extend(idxs)
        multi = {}
    if multi:
        order = np.random.default_rng(seed + 1).permutation(sorted(multi)).tolist()
        n_test_groups = int(round(len(order) * test_real_frac))
        test_groups = set(order[:n_test_groups])
        dropped = 0
        for g in order:
            in_test = g in test_groups
            for i in multi[g]:
                label, gen = val.meta(i)
                if label == 0:
                    (test_reals if in_test else sel).append(i)
                elif (gen in holdout_generators) == in_test:
                    (test_fakes if in_test else sel).append(i)
                else:
                    dropped += 1
        if dropped:
            print(f"[split_val_holdout] dropped {dropped} pair-grouped fakes routed to the other split")
    if holdout_generators:
        # Each configured holdout must contribute fakes to the final test split (grouping can drop them).
        _require_each_generator(_fake_counts(val, test_fakes), "Holdout", tuple(holdout_generators))
    if not test_fakes:
        return SubsetDataset(val, sorted(sel + reals + test_reals)), SubsetDataset(val, [])
    reals = np.random.default_rng(seed).permutation(reals).tolist()
    n_test_reals = int(round(len(reals) * test_real_frac))
    return (SubsetDataset(val, sorted(sel + reals[n_test_reals:])),
            SubsetDataset(val, sorted(test_fakes + reals[:n_test_reals] + test_reals)))


def split_nested_holdouts(
    val: Dataset, selection_generators: tuple[str, ...], test_generators: tuple[str, ...], seed: int = 0
) -> tuple[SubsetDataset, SubsetDataset, SubsetDataset]:
    """Disjoint fit, OOD checkpoint-selection, and untouched final-test sets.

    Pair-group aware: a multi-row group is assigned wholly to fit/select/test (seeded shuffle of sorted
    group ids, by thirds); within it all reals are kept plus only the fakes whose generator routes to
    that split (selection generators -> select, test generators -> test, anything else -> fit). The
    other fakes of the group are dropped so identities cannot leak across splits. Singleton rows follow
    the original per-row algorithm unchanged.
    """
    if not selection_generators or not test_generators or set(selection_generators) & set(test_generators):
        raise ValueError("Nested LOGO requires distinct, non-empty selection and test generator sets")
    singles, multi = _row_groups(val)
    fit, selection, test, reals = [], [], [], []
    for i in singles:
        label, generator = val.meta(i)
        if not label:
            reals.append(i)
        elif generator in selection_generators:
            selection.append(i)
        elif generator in test_generators:
            test.append(i)
        else:
            fit.append(i)
    if multi:
        order = np.random.default_rng(seed + 1).permutation(sorted(multi)).tolist()
        thirds = (len(order) // 3, 2 * len(order) // 3)
        split_of = {}
        for part, ids in zip((fit, selection, test),
                             (order[:thirds[0]], order[thirds[0]:thirds[1]], order[thirds[1]:])):
            for g in ids:
                split_of[g] = part
        dropped = 0
        for g in order:
            part = split_of[g]
            for i in multi[g]:
                label, generator = val.meta(i)
                if label == 0:
                    part.append(i)
                    continue
                required = (selection if generator in selection_generators
                            else test if generator in test_generators else fit)
                if required is part:
                    part.append(i)
                else:
                    dropped += 1
        if dropped:
            print(f"[split_nested_holdouts] dropped {dropped} pair-grouped fakes routed to another split")
    _require_each_generator(_fake_counts(val, selection), "Selection", tuple(selection_generators))
    _require_each_generator(_fake_counts(val, test), "Test", tuple(test_generators))
    n_fit_fakes = sum(1 for i in fit if val.meta(i)[0])
    n_sel_fakes = sum(1 for i in selection if val.meta(i)[0])
    n_test_fakes = sum(1 for i in test if val.meta(i)[0])
    n_reals = len(reals) + sum(
        1 for part in (fit, selection, test) for i in part if val.meta(i)[0] == 0)
    if not (n_fit_fakes and n_sel_fakes and n_test_fakes and n_reals >= 3):
        seen = sorted({val.meta(i)[1] for i in fit + selection + test})
        raise ValueError(
            "Nested LOGO needs fake samples in each split and at least three real images: "
            f"fit={n_fit_fakes} select={n_sel_fakes} test={n_test_fakes} reals={n_reals}; "
            f"selection={list(selection_generators)} test={list(test_generators)} generators present: {seen}"
        )
    reals = np.random.default_rng(seed).permutation(reals).tolist()
    thirds = (len(reals) // 3, 2 * len(reals) // 3)
    fit.extend(reals[:thirds[0]])
    selection.extend(reals[thirds[0]:thirds[1]])
    test.extend(reals[thirds[1]:])
    return tuple(SubsetDataset(val, sorted(indices)) for indices in (fit, selection, test))


def _video_clips(cfg: DataConfig) -> list[dict]:
    import json

    return json.loads((Path(cfg.video_frames_root) / "clips.json").read_text())


def _wdf_val_group(video_group: str, fraction: float) -> bool:
    """Deterministic video-level validation hold-out: md5(video_group) fraction < `fraction`."""
    return int(hashlib.md5(video_group.encode("utf-8")).hexdigest()[:8], 16) / 2**32 < fraction


def cap_fakes_per_group(clips: list[dict], max_ratio: float, seed: int = 0) -> list[dict]:
    """Keep all reals; cap fakes to the largest uniform per-video_group count k with total fakes <= max_ratio x
    reals. Which fakes survive is seeded per group (md5(group) + seed), so the result is deterministic."""
    reals = [c for c in clips if c["label"] == 0]
    fakes: dict[str, list[dict]] = {}
    for c in clips:
        if c["label"] == 1:
            fakes.setdefault(c["video_group"], []).append(c)
    budget = int(max_ratio * len(reals))
    if sum(map(len, fakes.values())) <= budget:
        return list(clips)
    sizes = sorted(len(v) for v in fakes.values())
    k = 0
    while k < sizes[-1] and sum(min(n, k + 1) for n in sizes) <= budget:
        k += 1
    kept = []
    for g, items in sorted(fakes.items()):
        items = sorted(items, key=lambda c: c["clip_id"])
        rng = np.random.default_rng([seed, int(hashlib.md5(g.encode()).hexdigest()[:8], 16)])
        kept += [items[i] for i in sorted(rng.permutation(len(items))[:k])]
    return reals + kept


def source_class_counts(clips: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for c in clips:
        d = out.setdefault(c["source"], {"real": 0, "fake": 0})
        d["fake" if c["label"] else "real"] += 1
    return out


def _wdf_extra_splits(cfg: DataConfig) -> dict[str, Dataset]:
    """wdf_test = official WildDeepfake test clips; celebdf_test = official Celeb-DF v2 test list; thb_cross = THB
    test fakes + CelebV-HQ test reals. Each is added only if its clips exist."""
    from dfd.data.video import VideoClipDataset

    clips = _video_clips(cfg)
    parts = {"wdf_test": [c for c in clips if c["source"] == "wdf" and c["split"] == "test"],
             "celebdf_test": [c for c in clips if c["source"] == "celebdf" and c["split"] == "test"],
             "thb_cross": [c for c in clips if c["source"] in ("thb", "celebvhq") and c["split"] == "test"]}
    return {k: VideoClipDataset(cfg.video_frames_root, v) for k, v in parts.items() if v}


def eval_splits(cfg: DataConfig, base_val: Dataset, seed: int = 0) -> dict[str, Dataset]:
    """Named validation splits, identical to scripts/train.py for the same seed."""
    sel, test = tuple(cfg.selection_generators), tuple(cfg.holdout_generators)
    if sel:
        fit, select, final = split_nested_holdouts(base_val, sel, test, seed)
    else:
        select, final = split_val_holdout(base_val, test, seed=seed)
        fit = select
    splits = {"all": base_val, "fit": fit, "select": select, "test": final}
    if cfg.source == "video_wdf":
        splits.update(_wdf_extra_splits(cfg))
    if cfg.sowaiba_eval:
        splits["sowaiba"] = SowaibaDataset(
            root=cfg.sowaiba_root,
            split="all",
            seed=seed,
            max_samples=cfg.max_val_samples,
            download=(cfg.sowaiba_root is None),
        )
    return splits


def build_base_datasets(cfg: DataConfig, seed: int = 0) -> tuple[Dataset, Dataset]:
    if cfg.source == "deepfake_eval_2024" or "Deepfake-Eval-2024" in (cfg.hf_dataset or ""):
        raise ValueError("Deepfake-Eval-2024 is evaluation-only (licence/terms); use scripts/eval_deepfake_eval_2024.py")
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
    if cfg.source == "paired_faces":
        train_parts: list[Dataset] = []
        val_parts: list[Dataset] = []
        if cfg.hf_data_files or cfg.hf_dataset:
            train_hf = HFImageDataset(cfg, cfg.hf_train_split, exclude_generators=holdout, max_samples=cfg.max_train_samples)
            val_hf = HFImageDataset(cfg, cfg.hf_val_split, max_samples=cfg.max_val_samples)
            train_parts.append(train_hf)
            val_parts.append(val_hf)

        dff_train_gens = tuple(g for g in cfg.dff_generators if g not in holdout)
        dff_val_gens = tuple(cfg.dff_generators)

        dff_train = DeepFakeFaceDataset(
            root=cfg.dff_root,
            generators=dff_train_gens,
            include_reals=True,
            split="train",
            val_frac=cfg.dff_val_fraction,
            format_align=cfg.dff_format_align,
            seed=seed,
            max_samples=cfg.max_train_samples,
            download=(cfg.dff_root is None),
        )
        dff_val = DeepFakeFaceDataset(
            root=cfg.dff_root,
            generators=dff_val_gens,
            include_reals=True,
            split="val",
            val_frac=cfg.dff_val_fraction,
            format_align=cfg.dff_format_align,
            seed=seed,
            max_samples=cfg.max_val_samples,
            download=(cfg.dff_root is None),
        )
        train_parts.append(dff_train)
        val_parts.append(dff_val)
        if cfg.sfhq_zip:
            if not cfg.dff_format_align:
                raise ValueError("data.sfhq_zip requires data.dff_format_align: SFHQ alignment must match DFF")
            from dfd.data.external import SFHQT2IDataset

            sfhq_kw = dict(generators=tuple(cfg.sfhq_generators) or None, val_fraction=cfg.sfhq_val_fraction,
                           align_size=FACE_ALIGN_SIZE, quality_range=FACE_JPEG_QUALITY, seed=seed)
            train_parts.append(SFHQT2IDataset(cfg.sfhq_zip, split="train", exclude_generators=holdout,
                                              max_per_generator=cfg.sfhq_max_per_generator_train, **sfhq_kw))
            val_parts.append(SFHQT2IDataset(cfg.sfhq_zip, split="val",
                                            max_per_generator=cfg.sfhq_max_per_generator_val, **sfhq_kw))
        if cfg.ffhq_root:
            if not cfg.dff_format_align:
                raise ValueError("data.ffhq_root requires data.dff_format_align: FFHQ alignment must match DFF")
            from dfd.data.external import FFHQDataset

            ffhq_kw = dict(val_fraction=cfg.ffhq_val_fraction, align_size=FACE_ALIGN_SIZE,
                           quality_range=FACE_JPEG_QUALITY, seed=seed)
            train_parts.append(FFHQDataset(cfg.ffhq_root, split="train", max_items=cfg.ffhq_max_train, **ffhq_kw))
            val_parts.append(FFHQDataset(cfg.ffhq_root, split="val", max_items=cfg.ffhq_max_val, **ffhq_kw))

        train = CompositeDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
        val = CompositeDataset(val_parts) if len(val_parts) > 1 else val_parts[0]
        return train, val
    if cfg.source == "video_thb":
        import json

        from dfd.data.video import VideoClipDataset

        root = Path(cfg.video_frames_root)
        clips = json.loads((root / "clips.json").read_text())
        clips = [c for c in clips if c["label"] == 0 or not cfg.video_methods or c["method"] in cfg.video_methods]
        # THB splits are identity-disjoint, so train = train split; val = val + test splits (nested LOGO routes them).
        train = [c for c in clips if c["split"] == "train" and (c["label"] == 0 or c["method"] not in holdout)]
        val = [c for c in clips if c["split"] in ("val", "test")]
        return VideoClipDataset(root, train), VideoClipDataset(root, val)
    if cfg.source == "video_wdf":
        from dfd.data.video import VideoClipDataset

        # Official WildDeepfake train/test are video-disjoint; validation holds out whole train videos (Celeb-DF:
        # whole target identities) of every source in video_train_sources.
        train = [c for c in _video_clips(cfg) if c["source"] in cfg.video_train_sources and c["split"] == "train"]
        celeb = [c for c in train if c["source"] == "celebdf"]
        if celeb:  # ~6:1 fake:real; uncapped it skews training and calibration toward "Celeb-DF look = fake"
            train = [c for c in train if c["source"] != "celebdf"] + cap_fakes_per_group(
                celeb, cfg.video_celebdf_max_fake_ratio, seed)
        held = [_wdf_val_group(c["video_group"], cfg.video_val_fraction) for c in train]
        fit = [c for c, h in zip(train, held) if not h]
        val = [c for c, h in zip(train, held) if h]
        print(f"[data] per-source class counts: train={source_class_counts(fit)} val={source_class_counts(val)}")
        return VideoClipDataset(cfg.video_frames_root, fit), VideoClipDataset(cfg.video_frames_root, val)
    if cfg.source == "genimage_arrow":
        from dfd.data.external import GENIMAGE_DIRS, GenImageArrowDataset

        if cfg.genimage_train_shards_per_generator < 1:
            raise ValueError("data.genimage_train_shards_per_generator must be >= 1 "
                             "(never default to the full ~655 GB)")
        root = Path(cfg.data_root) / "genimage"
        all_gens = tuple(cfg.genimage_generators) or tuple(GENIMAGE_DIRS)
        train_gens = tuple(g for g in all_gens if g not in holdout)
        train = GenImageArrowDataset(root, "train", train_gens,
                                     cfg.genimage_train_shards_per_generator,
                                     align_size=cfg.align_size, max_samples=cfg.max_train_samples, seed=seed)
        val = GenImageArrowDataset(root, "test", all_gens,
                                   cfg.genimage_val_shards_per_generator,
                                   align_size=cfg.align_size, max_samples=cfg.max_val_samples, seed=seed)
        return train, val
    raise ValueError(f"Unknown data source: {cfg.source}")
