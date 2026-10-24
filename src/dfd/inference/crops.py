"""Multi-crop scoring: tile the native-resolution image so off-centre edits are not missed.

Crops are taken at native resolution (no resize, same as training) on a grid spanning the image,
plus the centre crop. Per-crop logits are pooled with top-k mean: a localized manipulation only
needs to light up a few tiles, while k > 1 damps single-tile false alarms.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

from dfd.config import PTEConfig
from dfd.data.pte import PlatformTranscodingEmulator, PTEMeta, resize_short_side
from dfd.data.transforms import to_tensor
from dfd.models.temporal import topk_pool


def crop_boxes(w: int, h: int, size: int, grid: int = 3) -> list[tuple[int, int, int, int]]:
    """Grid of up to grid x grid crops covering the image edge to edge, plus the centre crop."""

    def starts(extent: int) -> list[int]:
        n = min(grid, max(1, math.ceil(extent / size)))
        return sorted({int(round(v)) for v in np.linspace(0, extent - size, n)})

    boxes = [(x, y, x + size, y + size) for y in starts(h) for x in starts(w)]
    cx, cy = (w - size) // 2, (h - size) // 2
    centre = (cx, cy, cx + size, cy + size)
    if centre not in boxes:
        boxes.append(centre)
    return boxes


class MultiCropTransform:
    def __init__(self, pte: PTEConfig, size: int, grid: int = 3):
        self.pte, self.size, self.grid = PlatformTranscodingEmulator(pte), size, grid

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> tuple[torch.Tensor, PTEMeta]:
        img, meta = self.pte(img, rng)
        if min(img.size) < self.size:
            img = resize_short_side(img, self.size, Image.BICUBIC)
        crops = [to_tensor(img.crop(b)) for b in crop_boxes(*img.size, self.size, self.grid)]
        return torch.stack(crops), meta


def pool_crops(crop_logits: torch.Tensor, k_frac: float = 0.25) -> torch.Tensor:
    """[N] per-crop logits -> scalar image logit (top-k mean)."""
    return topk_pool(crop_logits.unsqueeze(0), None, k_frac)[0]


@torch.no_grad()
def predict_multicrop(model, base, pte: PTEConfig, device: torch.device, seed: int = 1234,
                      grid: int = 3, k_frac: float = 0.25) -> dict[str, list]:
    """Score a `(image, label, generator)` dataset with multi-crop pooling (deterministic per index)."""
    model.eval()
    transform = MultiCropTransform(pte, model.cfg.image_size, grid)
    out: dict[str, list] = {"logits": [], "labels": [], "generators": [], "severities": [], "n_crops": []}
    for idx in range(len(base)):
        img, label, gen = base[idx]
        crops, meta = transform(img, np.random.default_rng([seed, 0, idx]))
        logits = model(crops.to(device))["logit"].float().cpu()
        out["logits"].append(float(pool_crops(logits, k_frac)))
        out["labels"].append(label)
        out["generators"].append(gen)
        out["severities"].append(meta.severity)
        out["n_crops"].append(len(crops))
    return out
