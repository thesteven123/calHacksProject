"""External-dataset loaders backed by the `dfd-data` Modal volume layout.

- GenImageArrowDataset: nebula/GenImage-arrow leaf bundles under <root>/data/<split>/<DIR>/data-*.arrow.
- WildDeepfakeDataset: xingjunm/WildDeepfake uncompressed .tar.gz tars (eval-only, *_test dirs).
- SFHQT2IDataset: bitmind/SyntheticFacesHQ T2I zip, fakes only.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from dfd.data.align import align_format, quality_for
from dfd.data.wilddeepfake_index import build_wilddeepfake_index, write_wilddeepfake_index  # noqa: F401

# Canonical names match scripts/modal_logo.py GENERATORS + SD14 (absent from Tiny-GenImage only).
GENIMAGE_DIRS = {
    "ADM": "ADM",
    "BigGAN": "BigGAN",
    "GLIDE": "glide",
    "Midjourney": "Midjourney",
    "SD14": "stable_diffusion_v_1_4",
    "SD15": "stable_diffusion_v_1_5",
    "VQDM": "VQDM",
    "Wukong": "wukong",
}


def select_shards(n: int, k: int | None) -> list[int]:
    """Shard indices for a k-of-n subset. Rows are ordered reals-first/fakes-last within a leaf, so a
    contiguous prefix would be single-class; round(np.linspace) spreads across the ordering."""
    if k is None or k >= n:
        return list(range(n))
    return sorted(set(int(i) for i in np.linspace(0, n - 1, k).round()))


class GenImageArrowDataset(Dataset):
    """Memory-mapped GenImage-arrow leaf bundles, one canonical generator per leaf directory."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        generators: tuple[str, ...] = (),
        max_shards_per_generator: int | None = None,
        align_size: int = 256,
        balance: bool = True,
        max_samples: int | None = None,
        seed: int = 0,
    ):
        import datasets

        self.align_size = align_size
        self.stats: dict[str, dict] = {}
        self._dss: list = []
        self._rows: list[tuple[int, int]] = []  # (dataset ref index, row)
        self._meta: list[tuple[int, str]] = []
        self._groups: list[str] = []
        rng = np.random.default_rng(seed)
        seen_real_basenames: set[str] = set()

        for gen in (tuple(generators) or tuple(GENIMAGE_DIRS)):
            if gen not in GENIMAGE_DIRS:
                raise ValueError(f"Unknown GenImage generator {gen!r}; expected one of {sorted(GENIMAGE_DIRS)}")
            leaf = Path(root) / "data" / split / GENIMAGE_DIRS[gen]
            if not leaf.is_dir():
                raise FileNotFoundError(f"GenImage leaf directory missing: {leaf}")
            shards = sorted(leaf.glob("data-*.arrow"))
            if not shards:
                raise FileNotFoundError(f"No data-*.arrow shards under {leaf}")
            shards = [shards[i] for i in select_shards(len(shards), max_shards_per_generator)]
            reals, fakes, dupe_reals = [], [], 0
            for shard in shards:
                ds = datasets.Dataset.from_file(str(shard))
                di = len(self._dss)
                self._dss.append(ds)
                labels = ds.data.column("label").to_pylist()  # ClassLabel ints: 0=real, 1=fake
                paths = ds.data.column("image_path").to_pylist()
                for r, (y, p) in enumerate(zip(labels, paths)):
                    y = int(y)
                    basename = Path(str(p)).name
                    if y == 0:
                        if basename in seen_real_basenames:
                            dupe_reals += 1
                            continue
                        seen_real_basenames.add(basename)
                        reals.append((di, r, basename, y))
                    else:
                        fakes.append((di, r, basename, y))
            if balance and reals and fakes and len(reals) != len(fakes):
                if len(reals) > len(fakes):
                    keep = sorted(rng.choice(len(reals), size=len(fakes), replace=False).tolist())
                    reals = [reals[i] for i in keep]
                else:
                    keep = sorted(rng.choice(len(fakes), size=len(reals), replace=False).tolist())
                    fakes = [fakes[i] for i in keep]
            kept = reals + fakes
            self.stats[gen] = {"reals": len(reals), "fakes": len(fakes), "dupe_reals_dropped": dupe_reals,
                               "shards": len(shards)}
            for di, r, basename, y in kept:
                self._rows.append((di, r))
                if y == 0:
                    self._meta.append((0, "real"))
                    self._groups.append(basename)
                else:
                    self._meta.append((1, gen))
                    self._groups.append(str(len(self._groups)))

        if max_samples is not None and len(self._rows) > max_samples:
            keep = sorted(rng.choice(len(self._rows), size=max_samples, replace=False).tolist())
            self._rows = [self._rows[i] for i in keep]
            self._meta = [self._meta[i] for i in keep]
            self._groups = [self._groups[i] for i in keep]

    def __len__(self) -> int:
        return len(self._rows)

    def meta(self, idx: int) -> tuple[int, str]:
        return self._meta[idx]

    def group(self, idx: int) -> str:
        return self._groups[idx]

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        di, r = self._rows[idx]
        row = self._dss[di][r]
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = align_format(img, self.align_size, quality_for(str(row["image_path"])))
        label, gen = self._meta[idx]
        return img, label, gen


class WildDeepfakeDataset(Dataset):
    """Eval-only frames from WildDeepfake *_test tars, read directly at tar member offsets."""

    def __init__(self, root: str | Path, index_path: str | Path | None = None,
                 frames_per_clip: int | None = 8):
        self.root = Path(root)
        path = Path(index_path) if index_path else self.root / "index_test.json"
        entries = json.loads(path.read_text())
        by_clip: dict[tuple, list[dict]] = {}
        for e in entries:
            by_clip.setdefault((e["label"], e["video"], e["clip"]), []).append(e)
        self.items: list[dict] = []
        for key in sorted(by_clip):
            frames = sorted(by_clip[key], key=lambda e: e["frame"])
            if frames_per_clip and len(frames) > frames_per_clip:
                sel = np.unique(np.linspace(0, len(frames) - 1, frames_per_clip).round().astype(int))
                frames = [frames[i] for i in sel]
            self.items.extend(frames)
        self._handles: dict[str, object] = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_handles"] = {}
        return d

    def _handle(self, tar: str):
        if tar not in self._handles:
            self._handles[tar] = open(self.root / tar, "rb")
        return self._handles[tar]

    def __len__(self) -> int:
        return len(self.items)

    def meta(self, idx: int) -> tuple[int, str]:
        e = self.items[idx]
        return e["label"], ("real" if e["label"] == 0 else "wilddeepfake")

    def clip_id(self, idx: int) -> str:
        e = self.items[idx]
        return f"{'real' if e['label'] == 0 else 'fake'}/{e['video']}/{e['clip']}"

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        e = self.items[idx]
        f = self._handle(e["tar"])
        f.seek(e["offset"])
        img = Image.open(io.BytesIO(f.read(e["size"]))).convert("RGB")
        return img, *self.meta(idx)


_SFHQ_PREFIX = re.compile(r"^(DALLE3|FLUX1_dev|FLUX1_pro|FLUX1_schnell|SDXL)_image_\d+\.jpe?g$")
SFHQ_GENERATORS = {"DALLE3": "DALLE3", "FLUX1_dev": "FLUX1-dev", "FLUX1_pro": "FLUX1-pro",
                   "FLUX1_schnell": "FLUX1-schnell", "SDXL": "SDXL"}


class SFHQT2IDataset(Dataset):
    """bitmind/SyntheticFacesHQ T2I zip (fakes only); generator = filename prefix."""

    def __init__(self, zip_path: str | Path, generators: tuple[str, ...] | None = None,
                 max_per_generator: int | None = None, align_size: int = 256, seed: int = 0):
        self.zip_path = Path(zip_path)
        self.align_size = align_size
        rng = np.random.default_rng(seed)
        by_gen: dict[str, list[str]] = {}
        with zipfile.ZipFile(self.zip_path) as zf:
            for name in zf.namelist():
                m = _SFHQ_PREFIX.match(Path(name).name)
                if m:
                    by_gen.setdefault(SFHQ_GENERATORS[m.group(1)], []).append(name)
        if generators:
            by_gen = {g: v for g, v in by_gen.items() if g in generators}
        self.items: list[tuple[str, str]] = []
        self.stats: dict[str, int] = {}
        for gen in sorted(by_gen):
            names = sorted(by_gen[gen])
            if max_per_generator is not None and len(names) > max_per_generator:
                keep = sorted(rng.choice(len(names), size=max_per_generator, replace=False).tolist())
                names = [names[i] for i in keep]
            self.stats[gen] = len(names)
            self.items += [(n, gen) for n in names]
        self._zip: zipfile.ZipFile | None = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_zip"] = None
        return d

    def _z(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def __len__(self) -> int:
        return len(self.items)

    def meta(self, idx: int) -> tuple[int, str]:
        return 1, self.items[idx][1]

    def group(self, idx: int) -> str:
        return str(idx)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        name, gen = self.items[idx]
        with self._z().open(name) as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        return align_format(img, self.align_size, quality_for(name)), 1, gen
