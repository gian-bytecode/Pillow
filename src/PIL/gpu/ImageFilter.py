"""
PIL.gpu.ImageFilter — GPU-accelerated image filters.

Mirrors PIL.ImageFilter, executing convolutions on the GPU.
"""

from __future__ import annotations

from PIL import ImageFilter as _CPUImageFilter

from .Image import Image

# Re-export kernel classes — they're just data containers, usable as-is
Kernel = _CPUImageFilter.Kernel
class RankFilter:
    """GPU-accelerated rank filter."""
    name = "Rank"

    def __init__(self, size: int, rank: int):
        self.size = size
        self.rank = rank

    def filter(self, image: Image) -> Image:
        return image.rank_filter(self.size, self.rank)


class MedianFilter(RankFilter):
    """GPU-accelerated median filter."""
    name = "Median"

    def __init__(self, size: int = 3):
        super().__init__(size, size * size // 2)


class MinFilter(RankFilter):
    """GPU-accelerated min filter."""
    name = "Min"

    def __init__(self, size: int = 3):
        super().__init__(size, 0)


class MaxFilter(RankFilter):
    """GPU-accelerated max filter."""
    name = "Max"

    def __init__(self, size: int = 3):
        super().__init__(size, size * size - 1)


class ModeFilter:
    """GPU-accelerated mode filter."""
    name = "Mode"

    def __init__(self, size: int = 3):
        self.size = size

    def filter(self, image: Image) -> Image:
        return image.mode_filter(self.size)

# Built-in filter instances (same kernel data as PIL)
BLUR = _CPUImageFilter.BLUR
CONTOUR = _CPUImageFilter.CONTOUR
DETAIL = _CPUImageFilter.DETAIL
EDGE_ENHANCE = _CPUImageFilter.EDGE_ENHANCE
EDGE_ENHANCE_MORE = _CPUImageFilter.EDGE_ENHANCE_MORE
EMBOSS = _CPUImageFilter.EMBOSS
FIND_EDGES = _CPUImageFilter.FIND_EDGES
SHARPEN = _CPUImageFilter.SHARPEN
SMOOTH = _CPUImageFilter.SMOOTH
SMOOTH_MORE = _CPUImageFilter.SMOOTH_MORE


class GaussianBlur:
    """GPU-accelerated Gaussian blur."""

    name = "GaussianBlur"

    def __init__(self, radius: float = 2):
        self.radius = radius

    def filter(self, image: Image) -> Image:
        return image.gaussian_blur(self.radius)


class BoxBlur:
    """GPU-accelerated box blur."""

    name = "BoxBlur"

    def __init__(self, radius: float = 2):
        self.radius = radius

    def filter(self, image: Image) -> Image:
        return image.box_blur(self.radius)


class UnsharpMask:
    """GPU-accelerated unsharp mask."""

    name = "UnsharpMask"

    def __init__(self, radius: float = 2, percent: int = 150, threshold: int = 3):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def filter(self, image: Image) -> Image:
        return image.unsharp_mask(self.radius, self.percent, self.threshold)
