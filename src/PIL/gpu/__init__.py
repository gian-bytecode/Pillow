"""
PIL.gpu — GPU-accelerated imaging, mirroring the PIL.* API.

Images persist in VRAM between operations.  Use ``Image.open()`` to upload,
and ``img.to_cpu()`` to download back when you need to save or inspect pixels.

Quick start::

    from PIL.gpu import Image

    img = Image.open("photo.jpg")          # load to GPU
    img = img.resize((800, 600))           # stays on GPU
    img = img.filter(ImageFilter.BLUR)     # still on GPU
    img.to_cpu().save("out.jpg")           # download + save
"""

from __future__ import annotations

try:
    from PIL import _imaging_gpu as _core  # type: ignore[attr-defined]
except ImportError:
    _core = None

__all__ = [
    "Image",
    "ImageFilter",
    "ImageChops",
    "ImageOps",
    "ImageEnhance",
    "ImageStat",
    "ImageDraw",
    "ImageMath",
    "ImageMorph",
    "ImageSequence",
    "ImagePalette",
    "ImageTransform",
]


def is_available() -> bool:
    """Return True if the GPU backend is loaded and a device is available."""
    if _core is None:
        return False
    return bool(_core.is_available())
