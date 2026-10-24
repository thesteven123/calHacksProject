"""Platform Transcoding Emulation (PTE) for still images.

Simulates multi-hop social-media distribution chains (e.g. WhatsApp -> CapCut -> platform): device-side
beautification, resize buckets, down/up rescale, JPEG/WebP re-encoding with 4:2:0 or 4:4:4 chroma, blur,
noise and tone jitter. Applied with identical distributions to real and fake samples so compression can't
become a label shortcut (DDA / PLADA).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from dfd.config import PTEConfig
from dfd.data.beauty import apply_beauty_filter

_INTERP = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX)


@dataclass
class PTEMeta:
    hops: int = 0
    beauty: bool = False
    min_quality: int = 100
    ops: list[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if self.hops == 0:
            return "clean"
        if self.min_quality < 50 or self.hops >= 3:
            return "heavy"
        return "light"


def encode(img: Image.Image, fmt: str, quality: int, chroma_444: bool = False) -> Image.Image:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality, subsampling=0 if chroma_444 else 2)
    else:
        img.save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_short_side(img: Image.Image, short: int, interp: int) -> Image.Image:
    w, h = img.size
    scale = short / min(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), interp)


class PlatformTranscodingEmulator:
    def __init__(self, cfg: PTEConfig):
        self.cfg = cfg

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, PTEMeta]:
        cfg, meta = self.cfg, PTEMeta()
        img = img.convert("RGB")
        if not cfg.enabled:
            return img, meta
        if rng.random() < cfg.beauty_prob:
            img, meta.beauty = apply_beauty_filter(img, rng), True
            meta.ops.append("beauty")
        meta.hops = int(rng.integers(cfg.min_hops, cfg.max_hops + 1))
        for _ in range(meta.hops):
            img = self._hop(img, rng, meta)
        return img, meta

    def _hop(self, img: Image.Image, rng: np.random.Generator, meta: PTEMeta) -> Image.Image:
        cfg = self.cfg
        bucket = int(rng.choice(cfg.short_side_buckets))
        if min(img.size) > bucket:
            img = resize_short_side(img, bucket, _INTERP[rng.integers(len(_INTERP))])
            meta.ops.append(f"resize{bucket}")
        if rng.random() < cfg.rescale_prob:
            w, h = img.size
            f = rng.uniform(*cfg.rescale_range)
            small = img.resize((max(8, int(w * f)), max(8, int(h * f))), _INTERP[rng.integers(len(_INTERP))])
            img = small.resize((w, h), _INTERP[rng.integers(len(_INTERP))])
            meta.ops.append(f"rescale{f:.2f}")
        if rng.random() < cfg.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(*cfg.blur_sigma))))
            meta.ops.append("blur")
        if rng.random() < cfg.color_jitter_prob:
            img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.85, 1.15)))
            img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.9, 1.1)))
            img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.85, 1.2)))
            meta.ops.append("tone")
        if rng.random() < cfg.noise_prob:
            arr = np.asarray(img, dtype=np.float32)
            arr = arr + rng.normal(0, rng.uniform(*cfg.noise_std), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            meta.ops.append("noise")
        u = rng.random()
        if u < cfg.jpeg_prob + cfg.webp_prob:
            fmt = "JPEG" if u < cfg.jpeg_prob else "WEBP"
            q = int(rng.integers(cfg.quality_range[0], cfg.quality_range[1] + 1))
            img = encode(img, fmt, q, chroma_444=rng.random() < cfg.chroma_444_prob)
            meta.min_quality = min(meta.min_quality, q)
            meta.ops.append(f"{fmt.lower()}{q}")
        return img


SEVERITY_PRESETS: dict[str, PTEConfig] = {
    "clean": PTEConfig(enabled=False),
    "light": PTEConfig(
        min_hops=1, max_hops=1, quality_range=(75, 95), jpeg_prob=0.8, webp_prob=0.2,
        rescale_prob=0.0, blur_prob=0.0, noise_prob=0.0, color_jitter_prob=0.0, beauty_prob=0.0,
    ),
    "heavy": PTEConfig(
        min_hops=3, max_hops=3, quality_range=(30, 60), jpeg_prob=0.7, webp_prob=0.3,
        rescale_prob=0.5, blur_prob=0.3, noise_prob=0.2, beauty_prob=0.0,
    ),
    "beauty": PTEConfig(min_hops=1, max_hops=1, quality_range=(75, 95), beauty_prob=1.0),
}
