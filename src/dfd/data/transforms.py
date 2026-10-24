"""Train/eval transforms: PTE -> min-size guard -> crop (not resize, to preserve pixel traces) -> tensor."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from dfd.config import PTEConfig
from dfd.data.pte import PlatformTranscodingEmulator, PTEMeta, resize_short_side


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)


class DetectorTransform:
    def __init__(self, pte: PTEConfig, image_size: int, train: bool):
        self.pte = PlatformTranscodingEmulator(pte)
        self.size = image_size
        self.train = train

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> tuple[torch.Tensor, PTEMeta]:
        img, meta = self.pte(img, rng)
        if min(img.size) < self.size:
            img = resize_short_side(img, self.size, Image.BICUBIC)
        w, h = img.size
        if self.train:
            x, y = int(rng.integers(0, w - self.size + 1)), int(rng.integers(0, h - self.size + 1))
        else:
            x, y = (w - self.size) // 2, (h - self.size) // 2
        img = img.crop((x, y, x + self.size, y + self.size))
        if self.train and rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(img), meta
