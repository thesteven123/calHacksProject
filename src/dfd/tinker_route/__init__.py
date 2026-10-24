"""Stage 3 VLM route: Qwen-VL LoRA fine-tuned on Tinker as a real/fake classifier (requires `.[tinker]`)."""

PROMPT = (
    "Is this image an authentic, camera-captured photograph, or was it AI-generated or digitally "
    "manipulated? Answer with exactly one word: real or fake."
)
ANSWERS = ("real", "fake")
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_RENDERER = "qwen3_5_disable_thinking"
# A VLM sees the whole frame, so native size and aspect ratio leak into its token grid. On GenImage-style
# data every fake is square and ~98% of reals are not, which alone separates the classes (AUC ~0.99).
# Feeding a fixed-size square crop to BOTH classes removes that shortcut. 0 disables (native aspect).
DEFAULT_SQUARE_SIZE = 448


def square_crop(img, size: int):
    """Centre-crop to a square, then resize to `size` x `size` (bicubic). Same op for real and fake."""
    from PIL import Image

    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.BICUBIC)
