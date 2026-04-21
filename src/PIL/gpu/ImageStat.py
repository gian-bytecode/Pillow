"""
PIL.gpu.ImageStat — GPU-accelerated image statistics.

Mirrors PIL.ImageStat, computing statistics from GPU histograms.
"""

from __future__ import annotations

import math
from typing import List

from .Image import Image


class Stat:
    """Compute statistics for a GPU image (downloaded via histogram)."""

    def __init__(self, image: Image):
        self.h = image.histogram()
        self._bands = image.bands

    @property
    def _band_histograms(self) -> List[List[int]]:
        bands = self._bands
        step = 256
        return [self.h[i * step : (i + 1) * step] for i in range(bands)]

    @property
    def count(self) -> List[int]:
        return [sum(bh) for bh in self._band_histograms]

    @property
    def sum(self) -> List[float]:
        return [
            float(sum(i * v for i, v in enumerate(bh)))
            for bh in self._band_histograms
        ]

    @property
    def sum2(self) -> List[float]:
        return [
            float(sum(i * i * v for i, v in enumerate(bh)))
            for bh in self._band_histograms
        ]

    @property
    def mean(self) -> List[float]:
        return [s / c if c else 0.0 for s, c in zip(self.sum, self.count)]

    @property
    def median(self) -> List[int]:
        result = []
        for bh in self._band_histograms:
            total = sum(bh)
            half = total // 2
            acc = 0
            for i, v in enumerate(bh):
                acc += v
                if acc > half:
                    result.append(i)
                    break
            else:
                result.append(0)
        return result

    @property
    def rms(self) -> List[float]:
        return [
            math.sqrt(s2 / c) if c else 0.0
            for s2, c in zip(self.sum2, self.count)
        ]

    @property
    def var(self) -> List[float]:
        return [
            (s2 / c) - (s / c) ** 2 if c else 0.0
            for s, s2, c in zip(self.sum, self.sum2, self.count)
        ]

    @property
    def stddev(self) -> List[float]:
        return [math.sqrt(v) for v in self.var]

    @property
    def extrema(self) -> List[tuple]:
        result = []
        for bh in self._band_histograms:
            lo = next((i for i, v in enumerate(bh) if v), 0)
            hi = next((255 - i for i, v in enumerate(reversed(bh)) if v), 255)
            result.append((lo, hi))
        return result
