"""Paired evaluation: every route scores the *same* degraded image.

PTE is applied once per image with a deterministic RNG keyed on (seed, split index). The core detector then
takes its native 224 centre crop and the VLM its 448 square crop, so score differences between routes come
from the models, not from different compression draws.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from PIL import Image

from dfd.config import Config
from dfd.data.pte import PlatformTranscodingEmulator

PAIRED_SEED_OFFSET = 1  # matches the validation seed used by scripts/train.py for selection splits


def paired_pte_images(cfg: Config, base, indices) -> Iterator[tuple[int, Image.Image, int, str, str]]:
    """Yield (idx, degraded_image, label, generator, severity) for split-local indices."""
    pte = PlatformTranscodingEmulator(cfg.data.pte)
    for i in indices:
        img, y, g = base[i]
        img, meta = pte(img, np.random.default_rng([cfg.train.seed + PAIRED_SEED_OFFSET, 0, int(i)]))
        yield int(i), img, int(y), g, meta.severity


def balanced_indices(base, per_class: int, seed: int) -> list[int]:
    """All indices if per_class <= 0, else a seeded balanced sample of `per_class` per label."""
    if per_class <= 0:
        return list(range(len(base)))
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for i in range(len(base)):
        by_label[base.meta(i)[0]].append(i)
    return sorted(np.concatenate([rng.choice(v, min(per_class, len(v)), replace=False)
                                  for v in by_label.values()]).astype(int).tolist())
