"""Deterministic format alignment: square crop, resize, seeded JPEG re-encode.

Applied identically to real and fake rows so resolution/format are not a label shortcut
(GenImage fakes are square PNGs, reals are varied-size ImageNet JPEGs).
"""

from __future__ import annotations

import hashlib
import io

from PIL import Image


def quality_for(key: str, lo: int = 75, hi: int = 95) -> int:
    """Deterministic JPEG quality in [lo, hi] seeded by the row key (e.g. image_path), label-independent."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return lo + h % (hi - lo + 1)


def align_format(img: Image.Image, size: int, quality: int) -> Image.Image:
    """Centre square crop -> resize to size x size -> JPEG 4:2:0 re-encode -> RGB."""
    img = img.convert("RGB")
    w, h = img.size
    crop = min(w, h)
    img = img.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=2)
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img.convert("RGB")
