"""Paired real/fake face datasets (DeepFakeFace, Sowaiba) with pair-grouped splits.

Guarantees:
1. (PIL.Image, label: int, generator: str) contract.
2. meta(idx) -> (label, generator).
3. Group-preserved splitting: a real photo and all its corresponding synthetic counterparts
   strictly land in the same split (grouped by relative path or source identity).
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class CompositeDataset(Dataset):
    """Concatenates multiple datasets while preserving `meta(idx)` and `group(idx)`."""

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = [d for d in datasets if len(d) > 0]
        self._lengths = [len(d) for d in self.datasets]
        self._cumsum = np.cumsum([0] + self._lengths)

    def __len__(self) -> int:
        return int(self._cumsum[-1]) if len(self._cumsum) > 0 else 0

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
        ds_idx = int(np.searchsorted(self._cumsum, idx, side="right") - 1)
        inner_idx = idx - int(self._cumsum[ds_idx])
        return ds_idx, inner_idx

    def meta(self, idx: int) -> tuple[int, str]:
        ds_idx, inner_idx = self._locate(idx)
        ds = self.datasets[ds_idx]
        if hasattr(ds, "meta"):
            return ds.meta(inner_idx)
        _, label, gen = ds[inner_idx]
        return label, gen

    def group(self, idx: int) -> str:
        ds_idx, inner_idx = self._locate(idx)
        ds = self.datasets[ds_idx]
        if hasattr(ds, "group"):
            return f"{ds_idx}_{ds.group(inner_idx)}"
        return f"{ds_idx}_{inner_idx}"

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        ds_idx, inner_idx = self._locate(idx)
        return self.datasets[ds_idx][inner_idx]


def verify_no_group_leakage(train_ds: Dataset, val_ds: Dataset) -> bool:
    """Verifies that no pair group (identity / source photo) leaks across splits."""
    def get_groups(ds: Dataset) -> set[str]:
        groups = set()
        for i in range(len(ds)):
            if hasattr(ds, "group"):
                groups.add(ds.group(i))
            elif hasattr(ds, "meta"):
                # fallback if group not present
                groups.add(str(ds.meta(i)))
        return groups

    train_g = get_groups(train_ds)
    val_g = get_groups(val_ds)
    overlap = train_g & val_g
    if overlap:
        raise AssertionError(
            f"Pair-group leakage detected: {len(overlap)} groups cross splits! Sample: {list(overlap)[:3]}"
        )
    return True


class DeepFakeFaceDataset(Dataset):
    """DeepFakeFace (OpenRL/DeepFakeFace).
    
    Contains wiki.zip (real, IMDB-WIKI) and fake zips: inpainting.zip, insight.zip, text2img.zip.
    Images are paired by matching relative paths inside each archive.
    """

    SUPPORTED_GENERATORS = ("inpainting", "insight", "text2img")

    def __init__(
        self,
        root: str | Path | None = None,
        generators: tuple[str, ...] = SUPPORTED_GENERATORS,
        include_reals: bool = True,
        split: str = "all",  # "all" | "train" | "val"
        val_frac: float = 0.2,
        format_align: bool = True,
        target_size: int = 384,
        quality_range: tuple[int, int] = (85, 95),
        seed: int = 0,
        max_samples: int | None = None,
        download: bool = False,
        cache_dir: str | Path | None = None,
    ):
        self.generators = tuple(generators)
        self.include_reals = include_reals
        self.split = split
        self.val_frac = val_frac
        self.format_align = format_align
        self.target_size = target_size
        self.quality_range = quality_range
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._zips: dict[str, zipfile.ZipFile] = {}
        self.zip_paths: dict[str, Path] = {}
        self.folder_paths: dict[str, Path] = {}
        self._is_zip_mode = True

        self._resolve_paths(root, download)
        self._index_items()
        if max_samples is not None and len(self.items) > max_samples:
            rng = np.random.default_rng(seed)
            keep_indices = sorted(rng.choice(len(self.items), size=max_samples, replace=False).tolist())
            self.items = [self.items[i] for i in keep_indices]

    def _resolve_paths(self, root: str | Path | None, download: bool) -> None:
        needed_types = list(self.generators)
        if self.include_reals:
            needed_types.append("wiki")

        if root is not None:
            r = Path(root)
            # Check if root contains zip files
            has_zips = any((r / f"{t}.zip").exists() for t in needed_types)
            if has_zips:
                self._is_zip_mode = True
                for t in needed_types:
                    zp = r / f"{t}.zip"
                    if zp.exists():
                        self.zip_paths[t] = zp
            else:
                # Check for extracted subfolders
                self._is_zip_mode = False
                for t in needed_types:
                    dp = r / t
                    if dp.exists() and dp.is_dir():
                        self.folder_paths[t] = dp
        elif download:
            from huggingface_hub import hf_hub_download
            self._is_zip_mode = True
            for t in needed_types:
                zp = hf_hub_download(
                    repo_id="OpenRL/DeepFakeFace",
                    filename=f"{t}.zip",
                    repo_type="dataset",
                    cache_dir=self.cache_dir,
                )
                self.zip_paths[t] = Path(zp)

    def _index_items(self) -> None:
        # Group items by relative path
        # Map: rel_path -> list of (category, full_inner_name or file_path)
        group_to_items: dict[str, list[tuple[str, str | Path]]] = {}

        if self._is_zip_mode:
            for cat, zp in self.zip_paths.items():
                with zipfile.ZipFile(zp, "r") as z:
                    for name in z.namelist():
                        if name.endswith("/"):
                            continue
                        ext = Path(name).suffix.lower()
                        if ext in IMAGE_EXTS:
                            # Normalize relative path: strip leading folder if present (e.g. wiki/61/... -> 61/...)
                            parts = Path(name).parts
                            rel_key = "/".join(parts[1:]) if len(parts) > 1 and parts[0] == cat else name
                            group_to_items.setdefault(rel_key, []).append((cat, name))
        else:
            for cat, dp in self.folder_paths.items():
                for p in sorted(dp.rglob("*")):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                        rel_key = str(p.relative_to(dp))
                        group_to_items.setdefault(rel_key, []).append((cat, p))

        # Split groups deterministically
        unique_groups = sorted(group_to_items.keys())
        rng = np.random.default_rng(self.seed)
        shuffled_groups = rng.permutation(unique_groups).tolist()
        n_val = int(round(len(shuffled_groups) * self.val_frac))
        val_groups = set(shuffled_groups[:n_val])
        train_groups = set(shuffled_groups[n_val:])

        if self.split == "train":
            allowed_groups = train_groups
        elif self.split == "val":
            allowed_groups = val_groups
        else:
            allowed_groups = set(unique_groups)

        # Build final flattened items list: (source_key, inner_target, label, generator, group_id)
        self.items: list[tuple[str, str | Path, int, str, str]] = []
        for g in sorted(allowed_groups):
            entries = group_to_items.get(g, [])
            for cat, path_or_name in entries:
                if cat == "wiki":
                    if self.include_reals:
                        self.items.append((cat, path_or_name, 0, "real", g))
                elif cat in self.generators:
                    self.items.append((cat, path_or_name, 1, cat, g))

    def _get_zip(self, cat: str) -> zipfile.ZipFile:
        if cat not in self._zips:
            self._zips[cat] = zipfile.ZipFile(self.zip_paths[cat], "r")
        return self._zips[cat]

    def __len__(self) -> int:
        return len(self.items)

    def meta(self, idx: int) -> tuple[int, str]:
        item = self.items[idx]
        return item[2], item[3]

    def group(self, idx: int) -> str:
        return self.items[idx][4]

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        cat, target, label, gen, group_id = self.items[idx]
        if self._is_zip_mode:
            z = self._get_zip(cat)
            with z.open(str(target)) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
        else:
            img = Image.open(target).convert("RGB")

        if self.format_align:
            # 1. Centre-crop to square based on short side
            w, h = img.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            img = img.crop((left, top, left + crop_size, top + crop_size))

            # 2. Resize to 384x384 (bicubic downscale)
            img = img.resize((self.target_size, self.target_size), Image.Resampling.BICUBIC)

            # 3. Deterministic JPEG re-encode seeded by group_id so real & fake in the same group get identical quality & DQT
            h_val = int(hashlib.md5(group_id.encode("utf-8")).hexdigest()[:8], 16)
            q = self.quality_range[0] + (h_val % (self.quality_range[1] - self.quality_range[0] + 1))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf)
            img.load()

        return img, label, gen


class SowaibaDataset(Dataset):
    """Sowaiba01/Deepfake paired face dataset.
    
    Structure: real/real_XXXXX.jpg and fake/fake_XXXXX.jpg paired via metadata.csv.
    Method: inswapper_128 (generator='inswapper_128', label=1; real label=0).
    """

    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "all",  # "all" | "train" | "val" | "test"
        val_frac: float = 0.2,
        seed: int = 0,
        max_samples: int | None = None,
        download: bool = False,
        cache_dir: str | Path | None = None,
    ):
        self.split = split
        self.val_frac = val_frac
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.root = Path(root) if root else None

        self._resolve_data(download)
        self._index_items()
        if max_samples is not None and len(self.items) > max_samples:
            rng = np.random.default_rng(seed)
            keep_indices = sorted(rng.choice(len(self.items), size=max_samples, replace=False).tolist())
            self.items = [self.items[i] for i in keep_indices]

    def _resolve_data(self, download: bool) -> None:
        if self.root is not None and (self.root / "metadata.csv").exists():
            self.meta_csv_path = self.root / "metadata.csv"
            self.data_dir = self.root
        elif download:
            from huggingface_hub import hf_hub_download, snapshot_download
            # Download metadata.csv and snapshot image folder
            self.meta_csv_path = Path(hf_hub_download(
                repo_id="Sowaiba01/Deepfake",
                filename="metadata.csv",
                repo_type="dataset",
                cache_dir=self.cache_dir,
            ))
            # Snapshot directory holds the real/ and fake/ directories
            snap = snapshot_download(
                repo_id="Sowaiba01/Deepfake",
                repo_type="dataset",
                allow_patterns=["real/*", "fake/*", "metadata.csv"],
                cache_dir=self.cache_dir,
            )
            self.data_dir = Path(snap)
            self.meta_csv_path = self.data_dir / "metadata.csv"
        else:
            self.meta_csv_path = None
            self.data_dir = None

    def _index_items(self) -> None:
        if not self.meta_csv_path or not self.meta_csv_path.exists():
            self.items = []
            return

        pairs: list[dict[str, str]] = []
        with open(self.meta_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append(row)

        # Unique pair group by real_file or id
        unique_groups = sorted(list({r.get("real_file", r.get("id")) for r in pairs}))
        rng = np.random.default_rng(self.seed)
        shuffled_groups = rng.permutation(unique_groups).tolist()
        n_val = int(round(len(shuffled_groups) * self.val_frac))
        val_groups = set(shuffled_groups[:n_val])
        train_groups = set(shuffled_groups[n_val:])

        if self.split == "train":
            allowed_groups = train_groups
        elif self.split in ("val", "test"):
            allowed_groups = val_groups
        else:
            allowed_groups = set(unique_groups)

        self.items: list[tuple[Path, int, str, str]] = []
        for r in pairs:
            grp = r.get("real_file", r.get("id"))
            if grp not in allowed_groups:
                continue
            real_p = self.data_dir / r["real_file"]
            fake_p = self.data_dir / r["fake_file"]
            method = r.get("method", "inswapper_128") or "inswapper_128"
            self.items.append((real_p, 0, "real", grp))
            self.items.append((fake_p, 1, method, grp))

    def __len__(self) -> int:
        return len(self.items)

    def meta(self, idx: int) -> tuple[int, str]:
        item = self.items[idx]
        return item[1], item[2]

    def group(self, idx: int) -> str:
        return self.items[idx][3]

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        path, label, gen, _ = self.items[idx]
        img = Image.open(path).convert("RGB")
        return img, label, gen
