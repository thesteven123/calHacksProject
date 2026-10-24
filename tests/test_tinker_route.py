"""Tinker-route preprocessing tests (no Tinker SDK or network needed)."""

from PIL import Image

from dfd.tinker_route import DEFAULT_SQUARE_SIZE, square_crop


def test_square_crop_hides_native_size_and_aspect():
    """GenImage fakes are square, reals mostly 4:3 — after square_crop both must be indistinguishable in shape."""
    real = Image.new("RGB", (500, 375))
    fakes = [Image.new("RGB", (s, s)) for s in (128, 256, 512, 1024)]
    shapes = {square_crop(im, DEFAULT_SQUARE_SIZE).size for im in [real, *fakes]}
    assert shapes == {(DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE)}


def test_square_crop_takes_centre_window():
    img = Image.new("RGB", (300, 100), "black")
    img.paste((255, 255, 255), (100, 0, 200, 100))  # white centre square
    out = square_crop(img, 50)
    assert out.getpixel((25, 25)) == (255, 255, 255)
