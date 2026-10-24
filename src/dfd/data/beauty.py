"""Mobile beautification-filter simulator (skin smoothing, optional sharpening and brightening)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def apply_beauty_filter(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    d = int(rng.choice([5, 7, 9]))
    sigma_color = float(rng.uniform(30, 90))
    sigma_space = float(rng.uniform(5, 15))
    smooth = cv2.bilateralFilter(arr, d, sigma_color, sigma_space)
    strength = float(rng.uniform(0.5, 1.0))
    out = cv2.addWeighted(smooth, strength, arr, 1 - strength, 0)
    if rng.random() < 0.5:
        blur = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    if rng.random() < 0.5:
        out = cv2.convertScaleAbs(out, alpha=float(rng.uniform(1.0, 1.1)), beta=float(rng.uniform(0, 12)))
    return Image.fromarray(out)
