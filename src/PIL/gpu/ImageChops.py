"""
PIL.gpu.ImageChops — GPU-accelerated channel operations.

Mirrors PIL.ImageChops, executing pixel-level binary ops on the GPU.
"""

from __future__ import annotations

from PIL import _imaging_gpu as _core

from .Image import Image


def add(image1: Image, image2: Image,
        scale: float = 1.0, offset: int = 0) -> Image:
    return image1._chop(image2, _core.CHOP_ADD, scale, offset)


def subtract(image1: Image, image2: Image,
             scale: float = 1.0, offset: int = 0) -> Image:
    return image1._chop(image2, _core.CHOP_SUBTRACT, scale, offset)


def multiply(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_MULTIPLY)


def screen(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_SCREEN)


def overlay(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_OVERLAY)


def difference(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_DIFFERENCE)


def lighter(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_LIGHTER)


def darker(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_DARKER)


def add_modulo(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_ADD_MODULO)


def subtract_modulo(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_SUBTRACT_MODULO)


def soft_light(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_SOFT_LIGHT)


def hard_light(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_HARD_LIGHT)


def logical_and(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_AND)


def logical_or(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_OR)


def logical_xor(image1: Image, image2: Image) -> Image:
    return image1._chop(image2, _core.CHOP_XOR)


def invert(image: Image) -> Image:
    """Invert pixel values.  Uses chop with self (second operand is ignored)."""
    return image._chop(image, _core.CHOP_INVERT)


def constant(image: Image, value: int) -> Image:
    """Fill a channel with a constant value."""
    return Image.new(image.mode, image.size, value)


def duplicate(image: Image) -> Image:
    """Copy an image."""
    return image.copy()


def composite(image1: Image, image2: Image, mask: Image) -> Image:
    """Create composite using a mask."""
    # Where mask is 255, use image1; where 0, use image2.
    result = image2.copy()
    result.paste(image1, mask=mask)
    return result


def offset(image: Image, xoffset: int, yoffset: int = 0) -> Image:
    """Circular-shift the image."""
    return image.offset(xoffset, yoffset)
