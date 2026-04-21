"""
PIL.gpu.ImageEnhance — GPU-accelerated image enhancement.

Mirrors PIL.ImageEnhance, executing operations on the GPU.
"""

from __future__ import annotations

from .Image import Image


class _Enhance:
    """Base class for GPU enhancers."""

    image: Image
    degenerate: Image

    def enhance(self, factor: float) -> Image:
        return Image.blend(self.degenerate, self.image, factor)


class Color(_Enhance):
    """Adjust color saturation."""

    def __init__(self, image: Image):
        self.image = image
        self.degenerate = image.convert("L").convert(image.mode)


class Contrast(_Enhance):
    """Adjust image contrast."""

    def __init__(self, image: Image):
        self.image = image
        hist = image.convert("L").histogram()
        num_pixels = sum(hist)
        mean = int(
            sum(i * count for i, count in enumerate(hist)) / max(num_pixels, 1) + 0.5
        )
        self.degenerate = Image.new(image.mode, image.size, mean)


class Brightness(_Enhance):
    """Adjust image brightness."""

    def __init__(self, image: Image):
        self.image = image
        self.degenerate = Image.new(image.mode, image.size, 0)


class Sharpness(_Enhance):
    """Adjust image sharpness."""

    def __init__(self, image: Image):
        self.image = image
        self.degenerate = image.gaussian_blur(2.0)
